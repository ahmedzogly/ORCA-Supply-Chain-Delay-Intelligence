"""
Comprehensive Unit & Integration Tests for E8 Cost-Sensitive Model Strategies.
Covers:
- BaseE8Strategy interface, preprocessing, schema loading, and ranking scores.
- StandardCatBoostStrategy (E8-A): Fixed threshold, F1-optimal threshold, probability calibration.
- CostWeightedCatBoostStrategy (E8-B): Instance sample weights, normalization, cost-optimal thresholding.
- CostThresholdCatBoostStrategy (E8-C): Bayes optimal thresholding, instance-level variation, gamma tuning.
- Edge cases: Single class, extreme values, missing categories, leakage enforcement.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from delay_intelligence.cost_sensitive.cost_engine import (
    CostEngine,
    CostScenario,
    CostScenarioModel,
    FORBIDDEN_COLUMNS,
    LeakageViolationError,
)
from delay_intelligence.cost_sensitive.models import (
    BaseE8Strategy,
    CostThresholdCatBoostStrategy,
    CostWeightedCatBoostStrategy,
    StandardCatBoostStrategy,
    load_default_feature_schema,
    preprocess_features,
    sanitize_cost_inputs,
)


@pytest.fixture
def sample_feature_df():
    """Generates synthetic supply-chain shipment features for testing."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "ID": np.arange(1, n + 1),
        "T_pred": pd.date_range("2012-01-01", periods=n, freq="D"),
        "Line Item Value": np.random.uniform(500.0, 500000.0, size=n),
        "Pack Price": np.random.uniform(5.0, 100.0, size=n),
        "Unit Price": np.random.uniform(0.1, 5.0, size=n),
        "Line Item Quantity": np.random.randint(100, 50000, size=n),
        "Line Item Insurance (USD)": np.random.uniform(5.0, 500.0, size=n),
        "Scheduled_Transit_Days": np.random.uniform(10.0, 120.0, size=n),
        "Forecast_Horizon_Days": np.random.uniform(30.0, 180.0, size=n),
        "PQ_to_PO_Days": np.random.uniform(0.0, 60.0, size=n),
        "T_pred_year": np.full(n, 2012),
        "T_pred_month": np.random.randint(1, 13, size=n),
        "T_pred_quarter": np.random.randint(1, 5, size=n),
        "T_pred_dayofweek": np.random.randint(0, 7, size=n),
        "vendor_hist_volume": np.random.randint(1, 200, size=n),
        "vendor_hist_delay_rate": np.random.uniform(0.0, 0.4, size=n),
        "vendor_hist_delay_median": np.random.uniform(0.0, 15.0, size=n),
        "country_hist_volume": np.random.randint(1, 300, size=n),
        "country_hist_delay_rate": np.random.uniform(0.0, 0.3, size=n),
        "country_hist_delay_median": np.random.uniform(0.0, 10.0, size=n),
        "site_hist_delay_rate": np.random.uniform(0.0, 0.3, size=n),
        "is_rdc_fulfillment": np.random.choice([0, 1], size=n),
        "is_pre_pq_process": np.random.choice([0, 1], size=n),
        "po_sent_is_date": np.random.choice([0, 1], size=n),
        "pq_first_sent_is_date": np.random.choice([0, 1], size=n),
        "weight_is_numeric": np.random.choice([0, 1], size=n),
        "freight_is_numeric": np.random.choice([0, 1], size=n),
        "Unit of Measure (Per Pack)": np.random.randint(10, 100, size=n),
        "Country": np.random.choice(["Nigeria", "Uganda", "Zambia", "Vietnam"], size=n),
        "Brand": np.random.choice(["Generic", "Aluvia", "Viread"], size=n),
        "Fulfill Via": np.random.choice(["From RDC", "Direct Drop"], size=n),
        "Molecule/Test Type": np.random.choice(["Efavirenz", "Tenofovir", "Nevirapine"], size=n),
        "Manufacturing Site": np.random.choice(["Site A", "Site B", "Site C"], size=n),
        "First Line Designation": np.random.choice(["Yes", "No"], size=n),
        "Dosage Form": np.random.choice(["Tablet", "Capsule", "Oral Solution"], size=n),
        "Dosage": np.random.choice(["300mg", "600mg", "200mg"], size=n),
        "Shipment Mode": np.random.choice(["Air", "Truck", "Ocean"], size=n),
        "Product Group": np.random.choice(["ARV", "HRDT"], size=n),
        "Sub Classification": np.random.choice(["Adult", "Pediatric", "HIV test"], size=n),
        "Vendor INCO Term": np.random.choice(["EXW", "CIP", "DDP"], size=n),
        "Vendor": np.random.choice(["Vendor 1", "Vendor 2", "Vendor 3"], size=n),
        "Delay_Days": np.random.choice([0.0, 0.0, 0.0, 5.0, 14.0, 25.0], size=n),
    })
    df["Delay_Flag"] = (df["Delay_Days"] > 0).astype(int)
    return df


@pytest.fixture
def cost_engine():
    return CostScenarioModel(scenario_name="base")


# =============================================================================
# 1. Feature Preprocessing and Schema Tests
# =============================================================================

def test_feature_preprocessing_and_schema(sample_feature_df):
    feat_cols, num_cols, cat_cols = load_default_feature_schema()
    assert len(feat_cols) == 39
    assert len(num_cols) == 26
    assert len(cat_cols) == 13

    # Add dirty NaNs and forbidden columns
    df_dirty = sample_feature_df.copy()
    df_dirty.loc[0, "Line Item Value"] = np.nan
    df_dirty.loc[1, "Country"] = None
    df_dirty["Delivered to Client Date"] = "2012-05-01"

    clean_df, resolved_cat = preprocess_features(
        df_dirty,
        cat_cols=cat_cols,
        num_cols=num_cols,
        feature_cols=feat_cols,
    )

    assert "Delivered to Client Date" not in clean_df.columns
    assert "Delay_Flag" not in clean_df.columns
    assert clean_df.loc[0, "Line Item Value"] == 0.0
    assert clean_df.loc[1, "Country"] == "missing"
    assert set(resolved_cat) == set(cat_cols)


def test_sanitize_cost_inputs():
    df = pd.DataFrame({
        "Line Item Value": [1000.0],
        "Delivered to Client Date": ["2014-01-01"],
        "Delay_Flag": [1],
        "Delay_Days": [5.0],
    })
    cleaned = sanitize_cost_inputs(df)
    assert "Delivered to Client Date" not in cleaned.columns
    assert "Delay_Flag" not in cleaned.columns
    assert "Delay_Days" not in cleaned.columns
    assert "Line Item Value" in cleaned.columns


# =============================================================================
# 2. Strategy E8-A: StandardCatBoostStrategy Tests
# =============================================================================

def test_standard_catboost_fixed_threshold(sample_feature_df, cost_engine):
    train_df = sample_feature_df.iloc[:80].copy()
    test_df = sample_feature_df.iloc[80:].copy()

    strategy = StandardCatBoostStrategy(
        threshold_mode="fixed",
        fixed_threshold=0.50,
        cost_engine=cost_engine,
        scenario_name="base",
        model_params={"iterations": 20, "depth": 4, "random_seed": 42},
        calibrate=False,
    )

    strategy.fit(
        X_train=train_df,
        y_train=train_df["Delay_Flag"],
    )

    assert strategy.is_fitted
    assert strategy.threshold == 0.50

    probs = strategy.predict_proba(test_df)
    assert len(probs) == len(test_df)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    preds = strategy.predict(test_df)
    assert len(preds) == len(test_df)
    assert np.array_equal(preds, (probs >= 0.50).astype(int))


def test_standard_catboost_f1_optimal_and_calibration(sample_feature_df, cost_engine):
    train_df = sample_feature_df.iloc[:60].copy()
    val_df = sample_feature_df.iloc[60:90].copy()
    test_df = sample_feature_df.iloc[90:].copy()

    strategy = StandardCatBoostStrategy(
        threshold_mode="f1_optimal",
        cost_engine=cost_engine,
        scenario_name="base",
        model_params={"iterations": 30, "depth": 4, "random_seed": 42},
        calibrate=True,
    )

    strategy.fit(
        X_train=train_df,
        y_train=train_df["Delay_Flag"],
        X_val=val_df,
        y_val=val_df["Delay_Flag"],
    )

    assert strategy.is_fitted
    assert strategy.calibrator is not None
    assert 0.05 <= strategy.threshold <= 0.95

    probs = strategy.predict_proba(test_df)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    preds = strategy.predict(test_df)
    assert np.array_equal(preds, (probs >= strategy.threshold).astype(int))


# =============================================================================
# 3. Strategy E8-B: CostWeightedCatBoostStrategy Tests
# =============================================================================

def test_cost_weighted_catboost_weights_and_fit(sample_feature_df, cost_engine):
    train_df = sample_feature_df.iloc[:80].copy()
    val_df = sample_feature_df.iloc[80:].copy()

    strategy = CostWeightedCatBoostStrategy(
        threshold_mode="cost_optimal",
        epsilon=10.0,
        normalize=True,
        cost_engine=cost_engine,
        scenario_name="base",
        model_params={"iterations": 25, "depth": 4, "random_seed": 42},
        calibrate=False,
    )

    # Test sample weights calculation
    weights = strategy.compute_sample_weights(
        X=train_df,
        y=train_df["Delay_Flag"],
        df_raw=train_df,
    )

    assert len(weights) == len(train_df)
    assert np.all(weights > 0)
    assert pytest.approx(np.mean(weights), rel=1e-5) == 1.0

    # Fit with inner validation
    strategy.fit(
        X_train=train_df,
        y_train=train_df["Delay_Flag"],
        df_raw_train=train_df,
        X_val=val_df,
        y_val=val_df["Delay_Flag"],
        df_raw_val=val_df,
    )

    assert strategy.is_fitted
    assert 0.05 <= strategy.threshold <= 0.95
    assert "sample_weight_mean" in strategy.metadata

    val_preds = strategy.predict(val_df, df_raw=val_df)
    assert len(val_preds) == len(val_df)
    assert set(np.unique(val_preds)).issubset({0, 1})


def test_cost_weighted_asymmetry_behavior(sample_feature_df, cost_engine):
    """High criticality, high commodity value shipments must yield higher positive sample weights."""
    low_val_row = sample_feature_df.iloc[[0]].copy()
    high_val_row = sample_feature_df.iloc[[0]].copy()

    low_val_row["Line Item Value"] = 100.0
    low_val_row["First Line Designation"] = "No"
    low_val_row["Product Group"] = "Other"

    high_val_row["Line Item Value"] = 1000000.0
    high_val_row["First Line Designation"] = "Yes"
    high_val_row["Product Group"] = "ARV"

    strategy = CostWeightedCatBoostStrategy(
        cost_engine=cost_engine,
        scenario_name="high",
        normalize=False,
    )

    w_low_pos = strategy.compute_sample_weights(low_val_row, [1], df_raw=low_val_row)[0]
    w_high_pos = strategy.compute_sample_weights(high_val_row, [1], df_raw=high_val_row)[0]

    assert w_high_pos > w_low_pos * 3.0, f"Expected high weight {w_high_pos} > 3x low weight {w_low_pos}"


# =============================================================================
# 4. Strategy E8-C: CostThresholdCatBoostStrategy Tests
# =============================================================================

def test_cost_threshold_catboost_bayes_thresholds(sample_feature_df, cost_engine):
    train_df = sample_feature_df.iloc[:70].copy()
    val_df = sample_feature_df.iloc[70:].copy()

    strategy = CostThresholdCatBoostStrategy(
        use_gamma_tuning=False,
        gamma=1.0,
        cost_engine=cost_engine,
        scenario_name="base",
        model_params={"iterations": 20, "depth": 4, "random_seed": 42},
        calibrate=True,
    )

    strategy.fit(
        X_train=train_df,
        y_train=train_df["Delay_Flag"],
        X_val=val_df,
        y_val=val_df["Delay_Flag"],
    )

    assert strategy.is_fitted
    thresholds = strategy.predict_thresholds(val_df, df_raw=val_df)

    assert len(thresholds) == len(val_df)
    assert np.all(thresholds >= 0.0) and np.all(thresholds <= 1.0)
    # Instance-dependent: thresholds should not all be identical
    assert np.std(thresholds) > 0.0, "Instance Bayes optimal thresholds must vary dynamically across shipments."

    preds = strategy.predict(val_df, df_raw=val_df)
    probs = strategy.predict_proba(val_df)
    assert np.array_equal(preds, (probs >= thresholds).astype(int))


def test_cost_threshold_catboost_gamma_tuning(sample_feature_df, cost_engine):
    train_df = sample_feature_df.iloc[:60].copy()
    val_df = sample_feature_df.iloc[60:90].copy()
    test_df = sample_feature_df.iloc[90:].copy()

    strategy = CostThresholdCatBoostStrategy(
        use_gamma_tuning=True,
        gamma_range=(0.50, 1.50, 0.25),
        cost_engine=cost_engine,
        scenario_name="base",
        model_params={"iterations": 25, "depth": 4, "random_seed": 42},
        calibrate=True,
    )

    strategy.fit(
        X_train=train_df,
        y_train=train_df["Delay_Flag"],
        df_raw_train=train_df,
        X_val=val_df,
        y_val=val_df["Delay_Flag"],
        df_raw_val=val_df,
    )

    assert strategy.is_fitted
    assert 0.50 <= strategy.gamma <= 1.50
    assert "val_best_gamma" in strategy.metadata

    test_preds = strategy.predict(test_df, df_raw=test_df)
    assert len(test_preds) == len(test_df)


# =============================================================================
# 5. Ranking Scores & Leakage Guard Tests
# =============================================================================

def test_strategy_ranking_scores(sample_feature_df, cost_engine):
    train_df = sample_feature_df.iloc[:60].copy()
    test_df = sample_feature_df.iloc[60:].copy()

    strategy = StandardCatBoostStrategy(
        cost_engine=cost_engine,
        scenario_name="base",
        model_params={"iterations": 15, "depth": 4, "random_seed": 42},
    )
    strategy.fit(train_df, train_df["Delay_Flag"])

    # Test all ranking policies
    score_cb = strategy.compute_ranking_scores(test_df, df_raw=test_df, policy="cost_benefit")
    score_risk = strategy.compute_ranking_scores(test_df, df_raw=test_df, policy="risk_only")
    score_val = strategy.compute_ranking_scores(test_df, df_raw=test_df, policy="value_only")
    score_loss = strategy.compute_ranking_scores(test_df, df_raw=test_df, policy="expected_loss")

    assert len(score_cb) == len(test_df)
    assert len(score_risk) == len(test_df)
    assert len(score_val) == len(test_df)
    assert len(score_loss) == len(test_df)


def test_cost_engine_strict_leakage_guard(sample_feature_df, cost_engine):
    """Directly passing forbidden target or post-outcome columns to compute_costs must raise LeakageViolationError."""
    leaky_df = sample_feature_df.copy()
    leaky_df["Delivered to Client Date"] = "2012-10-10"

    with pytest.raises(LeakageViolationError):
        cost_engine.compute_costs(leaky_df, strict_leakage_check=True)
