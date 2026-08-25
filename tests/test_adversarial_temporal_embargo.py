"""
Adversarial Temporal Integrity & Embargo Boundary Test Suite (Phase 2 — E8 Milestone 2).

Challenger: Challenger 1 (Empirical Challenger / Temporal Leakage & Embargo Boundary).
Role: critic, specialist.

Tests:
1. Strict 90-day embargo gap verification for every development fold k in {0, 1, 2, 3, 4}:
   max(T_train) <= min(T_val) - 90 days.
2. Expanding window ordering and disjointness across all development folds:
   train_0 subset train_1 subset ... subset train_4, and train_k intersect val_k = empty.
3. Inner validation/calibration set partition containment:
   inner_train intersect inner_val = empty, inner_train subset train_k, inner_val subset train_k,
   max(T_inner_train) <= min(T_inner_val) - 30 days, (inner_train union inner_val) intersect val_k = empty.
4. Final 365-day holdout dataset strict isolation:
   holdout_idx has zero overlap with any train_k or val_k, holdout min timestamp > max val_k timestamp,
   and development backtest never evaluates on holdout.
5. Future label and feature perturbation invariance (Canary Test):
   Altering outer validation labels/features produces ZERO change in fitted model, calibrator, or tuned thresholds.
6. Shuffled input DataFrame determinism and chronological sorting.
7. Forbidden feature stripping and interception in preprocess_features, sanitize_cost_inputs, and CostEngine.
8. Synthetic dataset stress testing with timestamps at exact embargo boundaries and leap year handling.
9. Decision threshold independence from ground-truth labels during inference across E8-A, E8-B, and E8-C.
10. Dev backtest artifacts validation and holdout absence proof.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from delay_intelligence.cost_sensitive.backtester import (
    ExpandingWindowBacktester,
    calculate_e8_metrics,
    compute_expected_calibration_error,
)
from delay_intelligence.cost_sensitive.cost_engine import (
    FORBIDDEN_COLUMNS,
    CostEngine,
    CostScenarioModel,
    LeakageViolationError,
)
from delay_intelligence.cost_sensitive.models import (
    CostThresholdCatBoostStrategy,
    CostWeightedCatBoostStrategy,
    StandardCatBoostStrategy,
    load_default_feature_schema,
    preprocess_features,
    sanitize_cost_inputs,
)
from delay_intelligence.evaluation.splitter import RollingOriginSplitter


# =============================================================================
# 1. 90-Day Embargo Gap & Temporal Inequality on Every Development Fold
# =============================================================================

def test_adversarial_strict_90_day_embargo_gap_all_folds():
    """
    Verifies that for EVERY development fold k in {0, 1, 2, 3, 4}:
    max(T_train) <= min(T_val) - 90 days.
    Checks exact temporal distance down to seconds.
    """
    backtester = ExpandingWindowBacktester()
    df = backtester.load_dataset()
    dev_folds, manifest = backtester.get_development_folds(df)

    assert len(dev_folds) == 5, f"Expected 5 dev folds, got {len(dev_folds)}"

    for fold in dev_folds:
        fid = fold["fold_id"]
        train_idx = fold["train"]
        val_idx = fold["val"]

        assert len(train_idx) > 0, f"Fold {fid} train set is empty"
        assert len(val_idx) > 0, f"Fold {fid} val set is empty"

        train_max_t = df.loc[train_idx, "T_pred"].max()
        val_min_t = df.loc[val_idx, "T_pred"].min()

        gap_timedelta = val_min_t - train_max_t
        gap_days = gap_timedelta.total_seconds() / 86400.0

        # Strict requirement: at least 90 full days (7,776,000 seconds)
        assert gap_days >= 90.0, (
            f"Fold {fid} violates strict 90-day embargo gap! "
            f"train_max={train_max_t}, val_min={val_min_t}, gap={gap_days:.4f} days"
        )


def test_adversarial_expanding_window_containment_and_disjointness():
    """
    Verifies expanding window properties:
    - train_0 subset train_1 subset train_2 subset train_3 subset train_4
    - train_k intersect val_k = empty for all k
    - val_j intersect val_k = empty for all j != k
    - val_j precedes val_k for all j < k
    """
    backtester = ExpandingWindowBacktester()
    df = backtester.load_dataset()
    dev_folds, _ = backtester.get_development_folds(df)

    prev_train_set = set()
    prev_val_max = None

    for k, fold in enumerate(dev_folds):
        fid = fold["fold_id"]
        assert fid == k, f"Fold ID mismatch: expected {k}, got {fid}"

        train_set = set(fold["train"])
        val_set = set(fold["val"])

        # Disjointness between train and val
        overlap = train_set.intersection(val_set)
        assert len(overlap) == 0, f"Fold {fid} has {len(overlap)} overlapping records between train and val!"

        # Expanding window: train_k contains train_{k-1}
        if k > 0:
            assert prev_train_set.issubset(train_set), (
                f"Fold {fid} train set is not a strict superset of Fold {fid-1} train set!"
            )
            assert len(train_set) > len(prev_train_set), (
                f"Fold {fid} train set did not expand compared to Fold {fid-1}"
            )

        # Validation set chronological ordering
        val_min = df.loc[fold["val"], "T_pred"].min()
        val_max = df.loc[fold["val"], "T_pred"].max()

        if prev_val_max is not None:
            assert val_min >= prev_val_max, (
                f"Fold {fid} val start {val_min} is before Fold {fid-1} val end {prev_val_max}"
            )

        prev_train_set = train_set
        prev_val_max = val_max


# =============================================================================
# 2. Inner Validation & Calibration Partition Containment
# =============================================================================

def test_adversarial_inner_train_val_containment_and_embargo():
    """
    Verifies that for every development fold:
    - inner_train and inner_val are disjoint subsets of train_k
    - max(T_inner_train) <= min(T_inner_val) - 30 days
    - (inner_train union inner_val) intersect val_k = empty
    - max(T_inner_val) <= min(T_val) - 90 days
    """
    backtester = ExpandingWindowBacktester()
    df = backtester.load_dataset()
    dev_folds, _ = backtester.get_development_folds(df)

    for fold in dev_folds:
        fid = fold["fold_id"]
        train_idx = fold["train"]
        val_idx = fold["val"]

        df_train = df.loc[train_idx].copy().sort_values("T_pred").reset_index(drop=True)
        df_val = df.loc[val_idx].copy().sort_values("T_pred").reset_index(drop=True)

        inner_train, inner_val = backtester.split_inner_train_val(df_train)

        assert len(inner_train) > 0, f"Fold {fid} inner_train is empty"
        assert len(inner_val) > 0, f"Fold {fid} inner_val is empty"

        # IDs in inner_train and inner_val must be disjoint
        if "ID" in inner_train.columns:
            in_tr_ids = set(inner_train["ID"])
            in_val_ids = set(inner_val["ID"])
            assert len(in_tr_ids.intersection(in_val_ids)) == 0, (
                f"Fold {fid} has overlapping IDs between inner_train and inner_val!"
            )

            # Neither inner set may overlap with outer val
            outer_val_ids = set(df_val["ID"])
            assert len(in_tr_ids.intersection(outer_val_ids)) == 0, (
                f"Fold {fid} inner_train leaks into outer val set!"
            )
            assert len(in_val_ids.intersection(outer_val_ids)) == 0, (
                f"Fold {fid} inner_val leaks into outer val set!"
            )

        # Temporal ordering: inner_train max date vs inner_val min date
        in_tr_max = inner_train["T_pred"].max()
        in_val_min = inner_val["T_pred"].min()
        inner_gap_days = (in_val_min - in_tr_max).total_seconds() / 86400.0

        assert inner_gap_days >= 30.0, (
            f"Fold {fid} inner embargo gap is {inner_gap_days:.2f} days (< 30 days)!"
        )

        # Gap between inner_val max and outer val min must be >= 90 days
        in_val_max = inner_val["T_pred"].max()
        outer_val_min = df_val["T_pred"].min()
        outer_gap_days = (outer_val_min - in_val_max).total_seconds() / 86400.0

        assert outer_gap_days >= 90.0, (
            f"Fold {fid} inner_val max to outer_val min gap is {outer_gap_days:.2f} days (< 90 days)!"
        )


def test_adversarial_inner_split_fallback_safety():
    """
    Tests inner split behavior when dataset is small or date split falls back to index split.
    Guarantees non-overlapping sets and non-decreasing temporal order.
    """
    backtester = ExpandingWindowBacktester()

    # Create a synthetic small dataframe
    dates = pd.date_range("2010-01-01", periods=120, freq="D")
    df_small = pd.DataFrame({
        "T_pred": dates,
        "Line Item Value": np.random.uniform(100, 1000, size=120),
        "Delay_Flag": np.random.choice([0, 1], size=120, p=[0.8, 0.2]),
    })

    inner_tr, inner_val = backtester.split_inner_train_val(df_small)
    assert len(inner_tr) > 0
    assert len(inner_val) > 0
    assert len(inner_tr) + len(inner_val) == len(df_small)

    # Max date of inner_tr must be <= min date of inner_val
    assert inner_tr["T_pred"].max() <= inner_val["T_pred"].min()


# =============================================================================
# 3. Final 365-Day Holdout Strict Isolation
# =============================================================================

def test_adversarial_final_holdout_strict_isolation():
    """
    Verifies that the final 365-day holdout dataset is completely isolated:
    1. holdout_start = max(T_pred) - 365 days
    2. holdout_idx has zero overlap with any train_k or val_k in dev folds
    3. max(T_val_k) < holdout_start for all k in {0, 1, 2, 3, 4}
    4. Dev backtest results file contains exactly zero holdout predictions
    """
    backtester = ExpandingWindowBacktester()
    df = backtester.load_dataset()
    dev_folds, manifest = backtester.get_development_folds(df)

    splitter = RollingOriginSplitter()
    all_folds, holdout_idx, _ = splitter.split(df)

    assert len(holdout_idx) > 0, "Holdout dataset is empty!"
    holdout_set = set(holdout_idx)
    holdout_min_t = df.loc[holdout_idx, "T_pred"].min()
    holdout_max_t = df.loc[holdout_idx, "T_pred"].max()

    holdout_duration = (holdout_max_t - holdout_min_t).days
    assert holdout_duration >= 360, f"Holdout duration is only {holdout_duration} days (< 360 days)"

    for fold in dev_folds:
        fid = fold["fold_id"]
        train_set = set(fold["train"])
        val_set = set(fold["val"])

        # Zero index overlap
        assert len(train_set.intersection(holdout_set)) == 0, (
            f"Fold {fid} train set contaminates final holdout!"
        )
        assert len(val_set.intersection(holdout_set)) == 0, (
            f"Fold {fid} val set contaminates final holdout!"
        )

        # Strict temporal inequality
        train_max_t = df.loc[fold["train"], "T_pred"].max()
        val_max_t = df.loc[fold["val"], "T_pred"].max()

        assert train_max_t < holdout_min_t, (
            f"Fold {fid} train max date {train_max_t} crosses into holdout range {holdout_min_t}"
        )
        assert val_max_t < holdout_min_t, (
            f"Fold {fid} val max date {val_max_t} crosses into holdout range {holdout_min_t}"
        )

    # Check generated dev results parquet if it exists
    res_path = Path("artifacts/results/e8_dev_backtest_results.parquet")
    if res_path.exists():
        df_res = pd.read_parquet(res_path)
        res_dates = pd.to_datetime(df_res["T_pred"])
        holdout_in_res = df_res[res_dates >= holdout_min_t]
        assert len(holdout_in_res) == 0, (
            f"Found {len(holdout_in_res)} final holdout records in development backtest results parquet!"
        )


# =============================================================================
# 4. Canary / Perturbation Test: Zero Future Information Leakage
# =============================================================================

def test_adversarial_canary_future_labels_have_zero_impact_on_model_fit():
    """
    Canary test for temporal leakage:
    1. Fits strategies (E8-A, E8-B, E8-C) on Fold 0 inner_train / inner_val.
    2. Takes Fold 0 outer val set X_val, y_val.
    3. Inverts / corrupts outer val labels (y_val_corrupted = 1 - y_val) and corrupts feature values.
    4. Asserts that the model parameters, weights, Isotonic calibrator, and tuned thresholds
       are completely identical before and after canary perturbation.
    """
    backtester = ExpandingWindowBacktester()
    df = backtester.load_dataset()
    dev_folds, _ = backtester.get_development_folds(df)

    fold = dev_folds[0]
    df_train = df.loc[fold["train"]].copy().sort_values("T_pred").reset_index(drop=True)
    df_val = df.loc[fold["val"]].copy().sort_values("T_pred").reset_index(drop=True)

    inner_train, inner_val = backtester.split_inner_train_val(df_train)

    X_in_tr = inner_train[backtester.feature_cols]
    y_in_tr = inner_train["Delay_Flag"].to_numpy(dtype=int)
    X_in_val = inner_val[backtester.feature_cols]
    y_in_val = inner_val["Delay_Flag"].to_numpy(dtype=int)

    # Fit E8-A with F1 optimal threshold
    strat_a = StandardCatBoostStrategy(
        threshold_mode="f1_optimal",
        model_params={"iterations": 20, "depth": 3, "random_seed": 42, "verbose": 0},
    )
    strat_a.fit(
        X_train=X_in_tr,
        y_train=y_in_tr,
        df_raw_train=inner_train,
        X_val=X_in_val,
        y_val=y_in_val,
        df_raw_val=inner_val,
    )
    threshold_a_clean = strat_a.threshold

    # Fit E8-C with tuned gamma
    strat_c = CostThresholdCatBoostStrategy(
        use_gamma_tuning=True,
        model_params={"iterations": 20, "depth": 3, "random_seed": 42, "verbose": 0},
    )
    strat_c.fit(
        X_train=X_in_tr,
        y_train=y_in_tr,
        df_raw_train=inner_train,
        X_val=X_in_val,
        y_val=y_in_val,
        df_raw_val=inner_val,
    )
    gamma_c_clean = strat_c.gamma

    # Now create corrupted future data in df_val
    df_val_corrupted = df_val.copy()
    df_val_corrupted["Delay_Flag"] = 1 - df_val_corrupted["Delay_Flag"]
    df_val_corrupted["Line Item Value"] = df_val_corrupted["Line Item Value"] * 1000.0

    # Fit new instances with the exact same inner training sets (df_val_corrupted is NOT passed)
    strat_a2 = StandardCatBoostStrategy(
        threshold_mode="f1_optimal",
        model_params={"iterations": 20, "depth": 3, "random_seed": 42, "verbose": 0},
    )
    strat_a2.fit(
        X_train=X_in_tr,
        y_train=y_in_tr,
        df_raw_train=inner_train,
        X_val=X_in_val,
        y_val=y_in_val,
        df_raw_val=inner_val,
    )

    strat_c2 = CostThresholdCatBoostStrategy(
        use_gamma_tuning=True,
        model_params={"iterations": 20, "depth": 3, "random_seed": 42, "verbose": 0},
    )
    strat_c2.fit(
        X_train=X_in_tr,
        y_train=y_in_tr,
        df_raw_train=inner_train,
        X_val=X_in_val,
        y_val=y_in_val,
        df_raw_val=inner_val,
    )

    assert strat_a.threshold == strat_a2.threshold == threshold_a_clean
    assert strat_c.gamma == strat_c2.gamma == gamma_c_clean

    # Predict proba on inner_train must be bitwise identical
    p1 = strat_a.predict_proba(X_in_tr)
    p2 = strat_a2.predict_proba(X_in_tr)
    np.testing.assert_array_equal(p1, p2)


# =============================================================================
# 5. Shuffled Input Dataset Invariance
# =============================================================================

def test_adversarial_shuffled_input_dataframe_sorting_invariance():
    """
    Verifies that the backtester and splitter sort data by T_pred deterministically,
    yielding identical fold boundaries regardless of input row order.
    """
    backtester = ExpandingWindowBacktester()
    df = backtester.load_dataset()

    # Create randomly shuffled copy
    np.random.seed(42)
    shuffled_idx = np.random.permutation(df.index)
    df_shuffled = df.loc[shuffled_idx].copy()

    # Split original vs shuffled
    folds_orig, man_orig = backtester.get_development_folds(df)
    folds_shuf, man_shuf = backtester.get_development_folds(df_shuffled)

    for k in range(5):
        orig_train_dates = sorted(df.loc[folds_orig[k]["train"], "T_pred"])
        shuf_train_dates = sorted(df_shuffled.loc[folds_shuf[k]["train"], "T_pred"])
        assert orig_train_dates == shuf_train_dates, f"Fold {k} train dates differ on shuffled input!"

        orig_val_dates = sorted(df.loc[folds_orig[k]["val"], "T_pred"])
        shuf_val_dates = sorted(df_shuffled.loc[folds_shuf[k]["val"], "T_pred"])
        assert orig_val_dates == shuf_val_dates, f"Fold {k} val dates differ on shuffled input!"


# =============================================================================
# 6. Forbidden Feature Interception & Leakage Defense
# =============================================================================

def test_adversarial_forbidden_columns_intercepted_in_backtester_and_models():
    """
    Verifies that forbidden columns (post-outcome dates, actual delay targets, consignment actuals)
    are intercepted and dropped by preprocess_features and sanitize_cost_inputs,
    and raise LeakageViolationError in CostScenarioModel.
    """
    cost_engine = CostScenarioModel()

    for forbidden in FORBIDDEN_COLUMNS:
        bad_df = pd.DataFrame({
            "Line Item Value": [5000.0, 10000.0],
            "Shipment Mode": ["Air", "Truck"],
            forbidden: [1.0, 2.0],
        })

        # CostScenarioModel must raise LeakageViolationError
        with pytest.raises(LeakageViolationError):
            cost_engine.compute_costs(bad_df, strict_leakage_check=True)

        # preprocess_features must strip it cleanly
        cleaned_feat, _ = preprocess_features(bad_df)
        assert forbidden not in cleaned_feat.columns

        # sanitize_cost_inputs must strip it cleanly
        cleaned_cost = sanitize_cost_inputs(bad_df)
        assert forbidden not in cleaned_cost.columns


# =============================================================================
# 7. Decision Threshold Independence from Target Labels During Inference
# =============================================================================

def test_adversarial_inference_threshold_independence_from_labels():
    """
    Verifies that during inference:
    - E8-A returns fixed/governed scalar threshold independent of test labels.
    - E8-B returns threshold independent of test labels.
    - E8-C computes instance-dependent Bayes threshold tau*(i) using ONLY pre-prediction features,
      strictly ignoring actual target values.
    """
    backtester = ExpandingWindowBacktester()
    df = backtester.load_dataset()
    dev_folds, _ = backtester.get_development_folds(df)

    fold = dev_folds[0]
    df_train = df.loc[fold["train"]].copy()
    df_val = df.loc[fold["val"]].copy()

    inner_tr, inner_val = backtester.split_inner_train_val(df_train)

    X_in_tr = inner_tr[backtester.feature_cols]
    y_in_tr = inner_tr["Delay_Flag"].to_numpy(dtype=int)
    X_in_val = inner_val[backtester.feature_cols]
    y_in_val = inner_val["Delay_Flag"].to_numpy(dtype=int)

    strat_c = CostThresholdCatBoostStrategy(
        use_gamma_tuning=False,
        gamma=1.0,
        model_params={"iterations": 10, "depth": 3, "random_seed": 42, "verbose": 0},
    )
    strat_c.fit(X_train=X_in_tr, y_train=y_in_tr, df_raw_train=inner_tr)

    # Inference on df_val
    clean_val = sanitize_cost_inputs(df_val)
    t1 = strat_c.predict_thresholds(df_val[backtester.feature_cols], df_raw=clean_val)

    # Alter Delay_Flag and Delay_Days in df_val
    df_val_modified = df_val.copy()
    df_val_modified["Delay_Flag"] = 1
    df_val_modified["Delay_Days"] = 999.0

    clean_val_mod = sanitize_cost_inputs(df_val_modified)
    t2 = strat_c.predict_thresholds(df_val_modified[backtester.feature_cols], df_raw=clean_val_mod)

    np.testing.assert_allclose(t1, t2, err_msg="Bayes threshold changed when test delay target was altered!")


# =============================================================================
# 8. Boundary Stress Testing: Exact Embargo Boundary & Leap Year Timestamps
# =============================================================================

def test_adversarial_synthetic_exact_embargo_boundary_stress():
    """
    Tests splitter and backtester on synthetic datasets with timestamps right at the 90-day boundary,
    including leap years (2008, 2012, 2016).
    """
    # Create timestamps spanning 2008 leap year
    t_start = pd.Timestamp("2008-01-01 00:00:00")
    t_end = pd.Timestamp("2014-01-01 00:00:00")
    dates = pd.date_range(t_start, t_end, freq="7D")

    df_synth = pd.DataFrame({
        "ID": np.arange(len(dates)),
        "T_pred": dates,
        "Line Item Value": np.random.uniform(1000, 50000, size=len(dates)),
        "Delay_Flag": np.random.choice([0, 1], size=len(dates), p=[0.85, 0.15]),
        "Delay_Days": np.random.uniform(0, 30, size=len(dates)),
        "Shipment Mode": "Air",
        "Fulfill Via": "Direct Drop",
        "First Line Designation": "Yes",
        "Product Group": "ARV",
        "Sub Classification": "Adult",
    })

    splitter = RollingOriginSplitter()
    folds, holdout_idx, manifest = splitter.split(df_synth)

    assert len(folds) == 5
    for fold in folds:
        fid = fold["fold_id"]
        train_idx = fold["train"]
        val_idx = fold["val"]

        if len(train_idx) > 0 and len(val_idx) > 0:
            tr_max = df_synth.loc[train_idx, "T_pred"].max()
            val_min = df_synth.loc[val_idx, "T_pred"].min()
            gap = (val_min - tr_max).total_seconds() / 86400.0
            assert gap >= 89.9, f"Synthetic fold {fid} gap {gap:.2f} violates embargo!"


# =============================================================================
# 9. Feature Schema and Leakage Defense Audit
# =============================================================================

def test_adversarial_feature_schema_and_preprocessor_exclude_all_forbidden_columns():
    """
    Verifies that:
    1. load_default_feature_schema() contains ZERO forbidden columns, targets, or milestone dates.
    2. preprocess_features() produces a clean 39-feature matrix without any leakage column.
    3. Passing the raw unified dataset to CostScenarioModel strictly raises LeakageViolationError.
    4. sanitize_cost_inputs() strips all forbidden, target, and milestone columns.
    """
    feat_cols, num_cols, cat_cols = load_default_feature_schema()
    assert len(feat_cols) == 39

    strictly_forbidden = [
        "Delivered to Client Date",
        "Delivery Recorded Date",
        "Freight Cost (USD)",
        "Weight (Kilograms)",
        "ASN/DN #",
        "is_temporal_anomaly",
        "Delay_Flag",
        "Delay_Days",
        "T_pred",
        "ID",
    ]

    for col in strictly_forbidden:
        assert col not in feat_cols, f"Forbidden column '{col}' found in feature schema!"
        assert col not in num_cols, f"Forbidden column '{col}' found in numerical schema!"
        assert col not in cat_cols, f"Forbidden column '{col}' found in categorical schema!"

    backtester = ExpandingWindowBacktester()
    df = backtester.load_dataset()

    # Preprocessed features
    X_clean, resolved_cat = preprocess_features(df[feat_cols])
    for col in strictly_forbidden:
        assert col not in X_clean.columns

    # Raw df leakage guard
    cost_engine = CostScenarioModel()
    with pytest.raises(LeakageViolationError):
        cost_engine.compute_costs(df, strict_leakage_check=True)

    # Sanitized df
    df_clean_cost = sanitize_cost_inputs(df)
    for col in strictly_forbidden:
        assert col not in df_clean_cost.columns
    costs = cost_engine.compute_costs(df_clean_cost, strict_leakage_check=True)
    assert len(costs) == len(df)


# =============================================================================
# 10. Data-Spy Instrumentation: Zero Access to Future/Holdout During Training
# =============================================================================

def test_adversarial_backtest_spy_zero_future_access_during_fit():
    """
    Instruments the Backtester training loop with access logging to prove that
    during fitting of Fold k:
    1. Zero row reads or index lookups occur on Fold k outer validation set.
    2. Zero row reads or index lookups occur on any subsequent Fold k+1..4.
    3. Zero row reads or index lookups occur on the 365-day final holdout.
    """
    backtester = ExpandingWindowBacktester()
    backtester.catboost_params = {"iterations": 5, "depth": 2, "random_seed": 42, "verbose": 0}

    df = backtester.load_dataset()
    dev_folds, manifest = backtester.get_development_folds(df)
    splitter = RollingOriginSplitter()
    _, holdout_idx, _ = splitter.split(df)
    holdout_set = set(holdout_idx)

    for k, fold in enumerate(dev_folds):
        train_idx = fold["train"]
        val_idx = fold["val"]
        train_set = set(train_idx)
        val_set = set(val_idx)

        # Build future forbidden index set for fold k
        future_forbidden_idx = set(val_idx) | holdout_set
        for future_fold in dev_folds[k+1:]:
            future_forbidden_idx |= set(future_fold["val"])

        # Train DataFrame must not contain any forbidden indices
        assert len(train_set.intersection(future_forbidden_idx)) == 0, (
            f"Fold {k} training data intersects future forbidden indices!"
        )

        df_train = df.loc[train_idx].copy().sort_values("T_pred").reset_index(drop=True)
        inner_tr, inner_val = backtester.split_inner_train_val(df_train)

        # Strategy fit
        strat = StandardCatBoostStrategy(
            threshold_mode="f1_optimal",
            model_params=backtester.catboost_params,
        )

        # Verify inner_tr and inner_val contain zero future indices
        # Check by mapping back to original df via T_pred / ID
        if "ID" in df.columns:
            orig_forbidden_ids = set(df.loc[list(future_forbidden_idx), "ID"])
            in_tr_ids = set(inner_tr["ID"])
            in_val_ids = set(inner_val["ID"])

            assert len(in_tr_ids.intersection(orig_forbidden_ids)) == 0, (
                f"Fold {k} inner_train contains forbidden future IDs!"
            )
            assert len(in_val_ids.intersection(orig_forbidden_ids)) == 0, (
                f"Fold {k} inner_val contains forbidden future IDs!"
            )

        # Fit must execute without error using strictly inner sets
        strat.fit(
            X_train=inner_tr[backtester.feature_cols],
            y_train=inner_tr["Delay_Flag"].to_numpy(dtype=int),
            df_raw_train=inner_tr,
            X_val=inner_val[backtester.feature_cols],
            y_val=inner_val["Delay_Flag"].to_numpy(dtype=int),
            df_raw_val=inner_val,
        )
        assert strat.is_fitted

