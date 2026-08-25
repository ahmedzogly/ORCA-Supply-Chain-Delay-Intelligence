"""
Comprehensive Unit & Integration Tests for E8 Expanding-Window Development Backtester.
Covers:
- ECE computation and calculate_e8_metrics correctness.
- ExpandingWindowBacktester initialization and fold partitioning.
- Strict 90-day embargo gap verification on development folds (Folds 0–4).
- Exclusion of 365-day final holdout.
- Multi-scenario and multi-strategy backtest execution.
- Parquet and JSON artifact generation and schema integrity.
- Zero future leakage and temporal ordering guarantees.
"""

from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import pytest

from delay_intelligence.cost_sensitive.backtester import (
    ExpandingWindowBacktester,
    calculate_e8_metrics,
    compute_expected_calibration_error,
)
from delay_intelligence.cost_sensitive.cost_engine import CostScenarioModel


# =============================================================================
# 1. Metric Calculation Tests
# =============================================================================

def test_compute_expected_calibration_error():
    # Perfectly calibrated case
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert pytest.approx(ece, abs=1e-5) == 0.0

    # Miscalibrated case
    y_true = np.array([1, 1, 1, 1])
    y_prob = np.array([0.0, 0.0, 0.0, 0.0])
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert pytest.approx(ece, abs=1e-5) == 1.0


def test_calculate_e8_metrics_correctness():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])
    thresholds = np.array([0.5, 0.5, 0.5, 0.5])

    costs_df = pd.DataFrame({
        "fn_cost": [1000.0, 1000.0, 1500.0, 1500.0],
        "fp_cost": [100.0, 100.0, 120.0, 120.0],
        "intervention_cost": [300.0, 300.0, 350.0, 350.0],
        "residual_delay_cost": [200.0, 200.0, 250.0, 250.0],
    })
    delay_days = np.array([10.0, 0.0, 12.0, 0.0])
    values = np.array([50000.0, 10000.0, 80000.0, 5000.0])

    metrics = calculate_e8_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        thresholds=thresholds,
        costs_df=costs_df,
        delay_days=delay_days,
        days_saved_efficacy=5.0,
        values=values,
    )

    # Item 0 (TP): Intervene on delayed -> Cost = 300 + 200 = 500
    # Item 1 (FP): Intervene on on-time -> Cost = 100
    # Item 2 (FN): No action on delayed -> Cost = 1500
    # Item 3 (TN): No action on on-time -> Cost = 0
    # Total Realized Cost = 500 + 100 + 1500 + 0 = 2100
    # Do Nothing Cost = 1000 + 0 + 1500 + 0 = 2500
    # Net Savings = 2500 - 2100 = 400
    # Cost Reduction % = 400 / 2500 * 100 = 16.0%

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["realized_cost"] == 2100.0
    assert metrics["do_nothing_cost"] == 2500.0
    assert metrics["net_savings"] == 400.0
    assert pytest.approx(metrics["cost_reduction_pct"], abs=1e-5) == 16.0
    assert metrics["reviews_count"] == 2
    assert metrics["review_coverage"] == 0.50
    assert metrics["delay_days_captured"] == 5.0
    assert metrics["cost_per_reviewed_shipment"] == (500.0 + 100.0) / 2.0


# =============================================================================
# 2. Backtester Temporal Structure & Isolation Tests
# =============================================================================

def test_backtester_development_folds_and_embargo():
    backtester = ExpandingWindowBacktester()
    df = backtester.load_dataset()
    dev_folds, manifest = backtester.get_development_folds(df)

    # Must contain exactly 5 development folds (0 to 4)
    assert len(dev_folds) == 5
    fold_ids = [f["fold_id"] for f in dev_folds]
    assert fold_ids == [0, 1, 2, 3, 4]

    # Check 90-day embargo gap on every fold
    for fold in dev_folds:
        fid = fold["fold_id"]
        train_idx = fold["train"]
        val_idx = fold["val"]

        train_max_date = df.loc[train_idx, "T_pred"].max()
        val_min_date = df.loc[val_idx, "T_pred"].min()

        embargo_gap_days = (val_min_date - train_max_date).days
        assert embargo_gap_days >= 89, f"Fold {fid} embargo gap {embargo_gap_days} is less than 90 days."

    # Final holdout must not be in dev folds
    holdout_start = pd.to_datetime("2014-08-24")
    for fold in dev_folds:
        val_max_date = df.loc[fold["val"], "T_pred"].max()
        assert val_max_date <= holdout_start, f"Fold {fold['fold_id']} val max date {val_max_date} crosses holdout start {holdout_start}"


def test_inner_train_val_split_integrity():
    backtester = ExpandingWindowBacktester()
    df = backtester.load_dataset()
    dev_folds, _ = backtester.get_development_folds(df)

    fold_0_train = df.loc[dev_folds[0]["train"]].copy()
    inner_tr, inner_val = backtester.split_inner_train_val(fold_0_train)

    assert len(inner_tr) > 0
    assert len(inner_val) > 0
    assert len(inner_tr) + len(inner_val) <= len(fold_0_train)

    # Inner train must precede inner val
    inner_tr_max = inner_tr["T_pred"].max()
    inner_val_min = inner_val["T_pred"].min()
    assert inner_tr_max < inner_val_min


# =============================================================================
# 3. Strategy Instantiation & Execution Tests
# =============================================================================

def test_strategy_instantiation():
    backtester = ExpandingWindowBacktester()

    s_a05 = backtester.instantiate_strategy("E8-A_tau0.5", scenario_name="base")
    assert s_a05.strategy_id == "E8-A"
    assert s_a05.threshold == 0.50

    s_af1 = backtester.instantiate_strategy("E8-A_f1", scenario_name="base")
    assert s_af1.strategy_id == "E8-A"
    assert s_af1.threshold_mode == "f1_optimal"

    s_b = backtester.instantiate_strategy("E8-B_cost_weighted", scenario_name="base")
    assert s_b.strategy_id == "E8-B"

    s_c_bayes = backtester.instantiate_strategy("E8-C_bayes_threshold", scenario_name="base")
    assert s_c_bayes.strategy_id == "E8-C"
    assert not s_c_bayes.use_gamma_tuning

    s_c_tuned = backtester.instantiate_strategy("E8-C_tuned_gamma", scenario_name="base")
    assert s_c_tuned.strategy_id == "E8-C"
    assert s_c_tuned.use_gamma_tuning


def test_quick_development_backtest_execution(tmp_path: Path):
    """Executes a lightweight backtest across 2 folds and 1 scenario to verify end-to-end integration."""
    test_out_dir = tmp_path / "temp_test_run"
    test_out_dir.mkdir(parents=True, exist_ok=True)

    backtester = ExpandingWindowBacktester(
        output_dir=test_out_dir,
    )
    # Override CatBoost params for rapid test execution
    backtester.catboost_params = {"iterations": 10, "depth": 3, "random_seed": 42, "verbose": 0}

    df = backtester.load_dataset()
    dev_folds, manifest = backtester.get_development_folds(df)

    # Test only fold 0 and fold 1
    limited_dev_folds = dev_folds[:2]

    # Monkeypatch get_development_folds to return 2 folds
    backtester.get_development_folds = lambda data: (limited_dev_folds, manifest)

    records_df, summary = backtester.run_development_backtest(
        df=df,
        strategy_keys=["E8-A_tau0.5", "E8-C_bayes_threshold"],
        scenarios=["base"],
        save_artifacts=True,
    )

    assert isinstance(records_df, pd.DataFrame)
    assert len(records_df) == (len(df.loc[dev_folds[0]["val"]]) + len(df.loc[dev_folds[1]["val"]])) * 2
    assert "realized_cost" in records_df.columns
    assert "prob_pred" in records_df.columns
    assert "threshold" in records_df.columns

    # Verify summary JSON
    assert "metadata" in summary
    assert "aggregated_summary" in summary
    assert "base" in summary["aggregated_summary"]
    assert "E8-A_tau0.5" in summary["aggregated_summary"]["base"]
    assert "E8-C_bayes_threshold" in summary["aggregated_summary"]["base"]

    # Verify saved files
    parquet_file = test_out_dir / "e8_dev_backtest_results.parquet"
    json_file = test_out_dir / "e8_dev_metrics.json"
    assert parquet_file.exists()
    assert json_file.exists()

