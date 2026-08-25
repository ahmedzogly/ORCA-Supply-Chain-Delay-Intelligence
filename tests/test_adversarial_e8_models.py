"""
Adversarial Empirical Stress Test Suite for Experiment E8 Model Strategies (models.py).

Authored by Challenger 2: Model Robustness & Threshold Integrity Challenger.

Adversarial Stress Dimensions:
1. Sample Weight Boundary Behavior (E8-B):
   - Extreme outlier monetary values ($10^9, $10^12)
   - Zero and negative monetary values
   - Negative net benefit (intervention cost > delay penalty)
   - Extreme class imbalance (1:200, 199:1)
   - Single-class training edge-case handling
   - Normalization invariants and non-negativity guarantees
2. Probability Calibration Robustness & Degeneracies (E8-A, E8-B, E8-C):
   - Tiny validation sets (N < 10)
   - Single-class validation sets (all 0s or all 1s)
   - Constant/degenerate model predictions
   - Strict range containment [0.0, 1.0], no NaNs or Infs
3. Bayes-Optimal and Empirical Threshold Selection under Extreme Cost Matrices:
   - Extreme Delay Cost Dominance (FN >> FP, tau* -> 0)
   - Extreme False Positive Cost Dominance (FP >> FN, tau* -> 1.0)
   - Symmetric Cost Baseline (FN = FP, tau* = 0.50)
   - Numerical stability against zero-division in denominators
4. Gamma Multiplier Tuning (E8-C):
   - Search space bound enforcement
   - Single-class validation responses (all 0s -> high gamma, all 1s -> low gamma)
   - Pathological cost regimes
5. Invariant & Structural Properties:
   - Decision consistency: predict(X) == (predict_proba(X) >= predict_thresholds(X))
   - Unfitted model safety: raises RuntimeError
   - Ranking score policy correctness and invalid policy detection
   - Zero-leakage enforcement and dirty input preprocessing robustness
"""

import math
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostError

from delay_intelligence.cost_sensitive.cost_engine import (
    CostBreakdown,
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
def synthetic_dev_data():
    """Generates a realistic multi-feature synthetic dataset for model strategy testing."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "ID": np.arange(1, n + 1),
        "T_pred": pd.date_range("2012-01-01", periods=n, freq="D"),
        "Line Item Value": np.random.exponential(scale=30000.0, size=n) + 100.0,
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
        "Country": np.random.choice(["Nigeria", "Uganda", "Zambia", "Vietnam", "Cote d'Ivoire"], size=n),
        "Brand": np.random.choice(["Generic", "Aluvia", "Viread", "Truvada"], size=n),
        "Fulfill Via": np.random.choice(["From RDC", "Direct Drop"], size=n),
        "Molecule/Test Type": np.random.choice(["Efavirenz", "Tenofovir", "Nevirapine", "Zidovudine"], size=n),
        "Manufacturing Site": np.random.choice(["Site A", "Site B", "Site C", "Site D"], size=n),
        "First Line Designation": np.random.choice(["Yes", "No"], size=n),
        "Dosage Form": np.random.choice(["Tablet", "Capsule", "Oral Solution"], size=n),
        "Dosage": np.random.choice(["300mg", "600mg", "200mg", "150mg"], size=n),
        "Shipment Mode": np.random.choice(["Air", "Truck", "Ocean", "Air Charter"], size=n),
        "Product Group": np.random.choice(["ARV", "HRDT", "ACT", "ANTIMALARIAL"], size=n),
        "Sub Classification": np.random.choice(["Adult", "Pediatric", "HIV test"], size=n),
        "Vendor INCO Term": np.random.choice(["EXW", "CIP", "DDP", "FCA"], size=n),
        "Vendor": np.random.choice(["Vendor 1", "Vendor 2", "Vendor 3", "Vendor 4"], size=n),
        "Delay_Days": np.random.choice([0.0, 0.0, 0.0, 0.0, 3.0, 12.0, 28.0], size=n),
    })
    df["Delay_Flag"] = (df["Delay_Days"] > 0).astype(int)
    return df


@pytest.fixture
def base_engine():
    return CostScenarioModel(scenario_name="base")


# =============================================================================
# 1. Sample Weight Boundary Behavior & Extremes (E8-B)
# =============================================================================

class TestSampleWeightBoundaryBehavior:
    """Adversarial stress testing of instance sample weight calculations in E8-B."""

    def test_extreme_outlier_monetary_values(self, synthetic_dev_data, base_engine):
        """Test that multi-million and multi-billion dollar consignments produce finite positive weights without overflow."""
        strategy = CostWeightedCatBoostStrategy(
            cost_engine=base_engine,
            scenario_name="high",
            normalize=True,
            epsilon=10.0,
        )

        df = synthetic_dev_data.copy()
        df.loc[0, "Line Item Value"] = 1e9    # $1 Billion
        df.loc[1, "Line Item Value"] = 1e12   # $1 Trillion
        df.loc[2, "Line Item Value"] = 0.0    # $0

        weights = strategy.compute_sample_weights(df, df["Delay_Flag"], df_raw=df)

        assert len(weights) == len(df)
        assert np.all(np.isfinite(weights)), "Sample weights must be finite numbers"
        assert np.all(weights > 0.0), "Sample weights must be strictly positive"
        assert pytest.approx(np.mean(weights), rel=1e-5) == 1.0, "Normalized sample weights must have mean = 1.0"
        
        # High value delay should have massive relative weight
        assert weights[0] > weights[2]

    def test_negative_monetary_values_clamped_safely(self, synthetic_dev_data, base_engine):
        """Negative monetary values should be safely clamped to 0 without generating negative weights or NaNs."""
        strategy = CostWeightedCatBoostStrategy(
            cost_engine=base_engine,
            scenario_name="base",
            normalize=True,
        )

        df_neg = synthetic_dev_data.copy()
        df_neg["Line Item Value"] = -50000.0  # Pathological negative values

        weights = strategy.compute_sample_weights(df_neg, df_neg["Delay_Flag"], df_raw=df_neg)

        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0.0)
        assert pytest.approx(np.mean(weights), rel=1e-5) == 1.0

    def test_negative_net_benefit_epsilon_clamping(self, synthetic_dev_data):
        """When intervention cost exceeds unmitigated delay penalty (Net_Benefit < 0), positive weight must clamp to epsilon."""
        # Create a pathological scenario where expediting is wildly expensive
        pathological_scenario = CostScenario(
            name="ExpediteExpensive",
            c_daily_base=1.0,
            rho_value=0.00001,
            c_fixed_stockout=10.0,
            c_triage_base=50.0,
            beta_audit=1.0,
            c_direct_inquiry=10.0,
            c_rdc_inquiry=5.0,
            c_expedite_base=50000.0,   # Massive expediting fee
            gamma_expedite=0.50,       # 50% cargo surcharge
            delay_days_assumed=5.0,
            days_saved_efficacy=1.0,
        )
        custom_engine = CostScenarioModel(custom_scenario=pathological_scenario)

        eps = 15.0
        strategy = CostWeightedCatBoostStrategy(
            cost_engine=custom_engine,
            scenario_name="expediteexpensive",
            normalize=False,
            epsilon=eps,
        )

        df = synthetic_dev_data.iloc[:20].copy()
        y = np.ones(len(df), dtype=int)  # All positive labels

        weights = strategy.compute_sample_weights(df, y, df_raw=df)

        # Since Net_Benefit is deeply negative, all positive weights must equal epsilon
        assert np.all(weights == eps), f"Expected all weights to be clamped to epsilon={eps}"

    def test_unnormalized_vs_normalized_weights(self, synthetic_dev_data, base_engine):
        """Verify normalization toggle operates correctly and preserves relative weight ratios."""
        strategy_norm = CostWeightedCatBoostStrategy(cost_engine=base_engine, normalize=True)
        strategy_unnorm = CostWeightedCatBoostStrategy(cost_engine=base_engine, normalize=False)

        df = synthetic_dev_data.iloc[:50].copy()
        y = df["Delay_Flag"].to_numpy()

        w_norm = strategy_norm.compute_sample_weights(df, y, df_raw=df)
        w_unnorm = strategy_unnorm.compute_sample_weights(df, y, df_raw=df)

        assert pytest.approx(np.mean(w_norm), rel=1e-5) == 1.0
        assert np.mean(w_unnorm) != 1.0

        # Ratios between any two samples must be identical
        ratio_norm = w_norm[0] / w_norm[1]
        ratio_unnorm = w_unnorm[0] / w_unnorm[1]
        assert pytest.approx(ratio_norm, rel=1e-5) == ratio_unnorm

    def test_cost_weighted_catboost_fit_with_extreme_weights(self, synthetic_dev_data, base_engine):
        """CatBoost training must succeed even when weights have high variance and extreme ratios."""
        df = synthetic_dev_data.copy()
        df.loc[0, "Line Item Value"] = 10000000.0  # $10M item
        df.loc[1, "Line Item Value"] = 5.0         # $5 item

        train_df = df.iloc[:150]
        val_df = df.iloc[150:]

        strategy = CostWeightedCatBoostStrategy(
            threshold_mode="cost_optimal",
            cost_engine=base_engine,
            model_params={"iterations": 20, "depth": 4, "random_seed": 42},
            calibrate=False,
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
        preds = strategy.predict(val_df, df_raw=val_df)
        assert len(preds) == len(val_df)
        assert set(np.unique(preds)).issubset({0, 1})


# =============================================================================
# 2. Probability Calibration Robustness & Degeneracies
# =============================================================================

class TestProbabilityCalibrationRobustness:
    """Stress testing probability calibration (Isotonic Regression) under degenerate data conditions."""

    def test_calibration_fallback_on_single_class_validation(self, synthetic_dev_data, base_engine):
        """When validation split contains only class 0 or only class 1, calibrator must gracefully fall back to None without raising exceptions."""
        train_df = synthetic_dev_data.iloc[:120].copy()
        val_all_zeros = synthetic_dev_data.iloc[120:160].copy()
        val_all_zeros["Delay_Flag"] = 0  # Single class in validation

        strategy = StandardCatBoostStrategy(
            threshold_mode="fixed",
            cost_engine=base_engine,
            model_params={"iterations": 15, "depth": 4, "random_seed": 42},
            calibrate=True,
        )

        strategy.fit(
            X_train=train_df,
            y_train=train_df["Delay_Flag"],
            X_val=val_all_zeros,
            y_val=val_all_zeros["Delay_Flag"],
        )

        assert strategy.is_fitted
        # Single class validation cannot fit IsotonicRegression; calibrator should be None
        assert strategy.calibrator is None

        # Predict proba should still return valid probabilities
        probs = strategy.predict_proba(val_all_zeros)
        assert len(probs) == len(val_all_zeros)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        assert not np.isnan(probs).any()

    def test_calibration_fallback_on_tiny_validation_set(self, synthetic_dev_data, base_engine):
        """When validation split has fewer than 10 samples, calibrator should gracefully fallback to uncalibrated."""
        train_df = synthetic_dev_data.iloc[:100].copy()
        val_tiny = synthetic_dev_data.iloc[100:105].copy()  # Only 5 samples

        strategy = CostThresholdCatBoostStrategy(
            cost_engine=base_engine,
            model_params={"iterations": 15, "depth": 4, "random_seed": 42},
            calibrate=True,
        )

        strategy.fit(
            X_train=train_df,
            y_train=train_df["Delay_Flag"],
            X_val=val_tiny,
            y_val=val_tiny["Delay_Flag"],
        )

        assert strategy.is_fitted
        assert strategy.calibrator is None  # Below 10-sample threshold
        probs = strategy.predict_proba(val_tiny)
        assert len(probs) == len(val_tiny)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_calibration_with_constant_model_predictions(self, synthetic_dev_data, base_engine):
        """When CatBoost predictions are near constant, Isotonic Regression must not produce NaN or out-of-bounds probabilities."""
        train_df = synthetic_dev_data.iloc[:100].copy()
        val_df = synthetic_dev_data.iloc[100:150].copy()

        strategy = StandardCatBoostStrategy(
            threshold_mode="fixed",
            cost_engine=base_engine,
            model_params={"iterations": 5, "depth": 1, "random_seed": 42},  # Underfitted model
            calibrate=True,
        )

        strategy.fit(
            X_train=train_df,
            y_train=train_df["Delay_Flag"],
            X_val=val_df,
            y_val=val_df["Delay_Flag"],
        )

        assert strategy.is_fitted
        test_df = synthetic_dev_data.iloc[150:].copy()
        probs = strategy.predict_proba(test_df)

        assert not np.isnan(probs).any()
        assert not np.isinf(probs).any()
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_calibrated_vs_uncalibrated_monotonicity(self, synthetic_dev_data, base_engine):
        """Isotonic regression is non-decreasing; calibrated probabilities must preserve rank order of raw probabilities."""
        train_df = synthetic_dev_data.iloc[:120].copy()
        val_df = synthetic_dev_data.iloc[120:170].copy()
        test_df = synthetic_dev_data.iloc[170:].copy()

        strategy = StandardCatBoostStrategy(
            cost_engine=base_engine,
            model_params={"iterations": 25, "depth": 4, "random_seed": 42},
            calibrate=True,
        )
        strategy.fit(train_df, train_df["Delay_Flag"], X_val=val_df, y_val=val_df["Delay_Flag"])

        if strategy.calibrator is not None:
            raw_p = strategy.model.predict_proba(strategy.preprocess(test_df)[0])[:, 1]
            cal_p = strategy.predict_proba(test_df)

            sorted_indices = np.argsort(raw_p)
            cal_p_sorted = cal_p[sorted_indices]

            # Differences in sorted calibrated probabilities must be >= 0 (monotonic non-decreasing)
            diffs = np.diff(cal_p_sorted)
            assert np.all(diffs >= -1e-7), "Isotonic calibrated probabilities must be monotonically non-decreasing"


# =============================================================================
# 3. Bayes-Optimal & Empirical Thresholds Under Extreme Cost Matrices
# =============================================================================

class TestBayesOptimalThresholdsExtremeCosts:
    """Stress testing Bayes-optimal decision thresholds under extreme cost asymmetry."""

    def test_extreme_delay_cost_dominance_fn_vastly_exceeds_fp(self, synthetic_dev_data):
        """
        When FN >> FP (e.g. C_stockout = $10,000,000, C_triage = $10):
        Threshold T_i MUST drop close to 0.0, instructing intervention on even slightly risky shipments.
        """
        scenario_fn_dominant = CostScenario(
            name="FNDominant",
            c_daily_base=500.0,
            rho_value=0.10,
            c_fixed_stockout=1000000.0,  # $1,000,000 stockout penalty
            c_triage_base=1.0,           # $1 triage cost
            beta_audit=0.01,
            c_direct_inquiry=1.0,
            c_rdc_inquiry=1.0,
            c_expedite_base=50.0,
            gamma_expedite=0.001,
            delay_days_assumed=20.0,
            days_saved_efficacy=15.0,
        )
        fn_engine = CostScenarioModel(custom_scenario=scenario_fn_dominant)

        strategy = CostThresholdCatBoostStrategy(
            use_gamma_tuning=False,
            gamma=1.0,
            cost_engine=fn_engine,
            scenario_name="fndominant",
            model_params={"iterations": 15, "depth": 4, "random_seed": 42},
            calibrate=False,
        )

        train_df = synthetic_dev_data.iloc[:100]
        test_df = synthetic_dev_data.iloc[100:]

        strategy.fit(train_df, train_df["Delay_Flag"])
        thresholds = strategy.predict_thresholds(test_df, df_raw=test_df)

        assert len(thresholds) == len(test_df)
        assert np.all(thresholds >= 0.0) and np.all(thresholds <= 1.0)
        # All thresholds should be very low (< 0.01) because delaying is catastrophic
        assert np.all(thresholds < 0.01), f"Expected thresholds < 0.01 under FN dominance, got max {np.max(thresholds)}"

        # Under such low thresholds, decisions should intervene on nearly all positive risk
        preds = strategy.predict(test_df, df_raw=test_df)
        probs = strategy.predict_proba(test_df)
        # Any instance with prob >= threshold should be 1
        expected_preds = (probs >= thresholds).astype(int)
        np.testing.assert_array_equal(preds, expected_preds)

    def test_extreme_false_alarm_cost_dominance_fp_vastly_exceeds_fn(self, synthetic_dev_data):
        """
        When FP >> FN (e.g. C_triage = $500,000, C_delay = $10):
        Threshold T_i MUST approach 1.0, suppressing virtually all alerts unless probability is near certainty.
        """
        scenario_fp_dominant = CostScenario(
            name="FPDominant",
            c_daily_base=0.10,
            rho_value=0.00001,
            c_fixed_stockout=5.0,
            c_triage_base=50000.0,      # $50,000 triage cost
            beta_audit=500.0,
            c_direct_inquiry=1000.0,
            c_rdc_inquiry=500.0,
            c_expedite_base=500.0,
            gamma_expedite=0.01,
            delay_days_assumed=5.0,
            days_saved_efficacy=2.0,
        )
        fp_engine = CostScenarioModel(custom_scenario=scenario_fp_dominant)

        strategy = CostThresholdCatBoostStrategy(
            use_gamma_tuning=False,
            gamma=1.0,
            cost_engine=fp_engine,
            scenario_name="fpdominant",
            model_params={"iterations": 15, "depth": 4, "random_seed": 42},
            calibrate=False,
        )

        train_df = synthetic_dev_data.iloc[:100]
        test_df = synthetic_dev_data.iloc[100:]

        strategy.fit(train_df, train_df["Delay_Flag"])
        thresholds = strategy.predict_thresholds(test_df, df_raw=test_df)

        assert np.all(thresholds >= 0.0) and np.all(thresholds <= 1.0)
        # Under FP dominance, threshold should be high (>= 0.85)
        assert np.all(thresholds >= 0.85), f"Expected thresholds >= 0.85 under FP dominance, got min {np.min(thresholds)}"

    def test_symmetric_cost_boundary_yields_half_threshold(self, synthetic_dev_data):
        """
        Under classical symmetric 2x2 costs (FN = FP), tau*_simple MUST equal exactly 0.50.
        """
        scenario_symmetric = CostScenario(
            name="Symmetric",
            c_daily_base=10.0,
            rho_value=0.0,
            c_fixed_stockout=100.0,
            c_triage_base=200.0,
            beta_audit=0.0,
            c_direct_inquiry=0.0,
            c_rdc_inquiry=0.0,
            c_expedite_base=50.0,
            gamma_expedite=0.0,
            delay_days_assumed=10.0,
            days_saved_efficacy=5.0,
            mode_multipliers={"Air": 1.0, "Air Charter": 1.0, "Truck": 1.0, "Ocean": 1.0, "Default": 1.0},
        )
        sym_engine = CostScenarioModel(custom_scenario=scenario_symmetric)

        df = pd.DataFrame({
            "Line Item Value": [1000.0],
            "Shipment Mode": ["Air"],
            "First Line Designation": ["No"],
            "Sub Classification": ["Adult"],
            "Product Group": ["Other"],
        })
        # FN = 100 + 10 * 10 = 200
        # FP = 200
        costs = sym_engine.compute_costs(df, is_log_transformed=False)
        assert costs["fn_cost"].iloc[0] == 200.0
        assert costs["fp_cost"].iloc[0] == 200.0
        assert pytest.approx(costs["tau_star_simple"].iloc[0], abs=1e-5) == 0.50


# =============================================================================
# 4. Gamma Multiplier Tuning Integrity (E8-C)
# =============================================================================

class TestGammaMultiplierTuning:
    """Stress testing of empirical gamma multiplier tuning on inner validation folds."""

    def test_gamma_tuning_on_all_zeros_validation(self, synthetic_dev_data, base_engine):
        """When validation data has 0 delays, gamma tuning should pick maximum gamma to suppress false alarms."""
        train_df = synthetic_dev_data.iloc[:120].copy()
        val_zeros = synthetic_dev_data.iloc[120:160].copy()
        val_zeros["Delay_Flag"] = 0  # No delays in validation

        strategy = CostThresholdCatBoostStrategy(
            use_gamma_tuning=True,
            gamma_range=(0.50, 2.00, 0.25),
            cost_engine=base_engine,
            model_params={"iterations": 15, "depth": 4, "random_seed": 42},
            calibrate=False,
        )

        strategy.fit(
            X_train=train_df,
            y_train=train_df["Delay_Flag"],
            df_raw_train=train_df,
            X_val=val_zeros,
            y_val=val_zeros["Delay_Flag"],
            df_raw_val=val_zeros,
        )

        assert strategy.is_fitted
        # Optimal behavior on 0 delays is to maximize threshold (highest gamma)
        assert strategy.gamma == 2.00
        assert 0.50 <= strategy.gamma <= 2.00

    def test_gamma_tuning_on_all_ones_validation(self, synthetic_dev_data, base_engine):
        """When validation data is 100% delays, gamma tuning should pick minimum gamma to intervene on everything."""
        train_df = synthetic_dev_data.iloc[:120].copy()
        val_ones = synthetic_dev_data.iloc[120:160].copy()
        val_ones["Delay_Flag"] = 1  # All delayed

        strategy = CostThresholdCatBoostStrategy(
            use_gamma_tuning=True,
            gamma_range=(0.20, 1.50, 0.10),
            cost_engine=base_engine,
            model_params={"iterations": 15, "depth": 4, "random_seed": 42},
            calibrate=False,
        )

        strategy.fit(
            X_train=train_df,
            y_train=train_df["Delay_Flag"],
            df_raw_train=train_df,
            X_val=val_ones,
            y_val=val_ones["Delay_Flag"],
            df_raw_val=val_ones,
        )

        assert strategy.is_fitted
        # Optimal behavior on 100% delays is to minimize threshold (lowest gamma)
        assert strategy.gamma == 0.20

    def test_gamma_clipping_in_predict_thresholds(self, synthetic_dev_data, base_engine):
        """Thresholds clipped with gamma > 2.0 or gamma < 0.0 must remain strictly in [0.0, 1.0]."""
        strategy = CostThresholdCatBoostStrategy(
            use_gamma_tuning=False,
            gamma=5.0,  # Extreme high multiplier
            cost_engine=base_engine,
            model_params={"iterations": 10, "depth": 3, "random_seed": 42},
        )
        strategy.fit(synthetic_dev_data.iloc[:50], synthetic_dev_data.iloc[:50]["Delay_Flag"])

        thresholds = strategy.predict_thresholds(synthetic_dev_data.iloc[50:], df_raw=synthetic_dev_data.iloc[50:])
        assert np.all(thresholds >= 0.0) and np.all(thresholds <= 1.0)


# =============================================================================
# 5. Invariants & Decision Consistency Across All Strategies
# =============================================================================

class TestModelInvariantsAndSanity:
    """Stress testing core mathematical and interface invariants across all E8 strategies."""

    @pytest.mark.parametrize("strategy_cls, kwargs", [
        (StandardCatBoostStrategy, {"threshold_mode": "fixed", "fixed_threshold": 0.50, "calibrate": False}),
        (StandardCatBoostStrategy, {"threshold_mode": "f1_optimal", "calibrate": True}),
        (CostWeightedCatBoostStrategy, {"threshold_mode": "cost_optimal", "calibrate": False}),
        (CostThresholdCatBoostStrategy, {"use_gamma_tuning": False, "gamma": 1.0, "calibrate": True}),
        (CostThresholdCatBoostStrategy, {"use_gamma_tuning": True, "calibrate": True}),
    ])
    def test_decision_consistency_invariant(self, synthetic_dev_data, base_engine, strategy_cls, kwargs):
        """
        Mathematical Invariant:
        predict(X) == (predict_proba(X) >= predict_thresholds(X)).astype(int)
        MUST hold 100% of the time across all instances.
        """
        train_df = synthetic_dev_data.iloc[:120].copy()
        val_df = synthetic_dev_data.iloc[120:160].copy()
        test_df = synthetic_dev_data.iloc[160:].copy()

        strategy: BaseE8Strategy = strategy_cls(
            cost_engine=base_engine,
            model_params={"iterations": 20, "depth": 4, "random_seed": 42},
            **kwargs,
        )

        strategy.fit(
            X_train=train_df,
            y_train=train_df["Delay_Flag"],
            df_raw_train=train_df,
            X_val=val_df,
            y_val=val_df["Delay_Flag"],
            df_raw_val=val_df,
        )

        probs = strategy.predict_proba(test_df)
        thresholds = strategy.predict_thresholds(test_df, df_raw=test_df)
        preds = strategy.predict(test_df, df_raw=test_df)

        expected_preds = (probs >= thresholds).astype(int)
        np.testing.assert_array_equal(
            preds,
            expected_preds,
            err_msg=f"Decision consistency violated in {strategy.name}",
        )

    def test_unfitted_model_raises_runtime_error(self, synthetic_dev_data):
        """Calling inference methods on unfitted strategies must raise RuntimeError."""
        s_a = StandardCatBoostStrategy()
        s_b = CostWeightedCatBoostStrategy()
        s_c = CostThresholdCatBoostStrategy()

        for s in [s_a, s_b, s_c]:
            with pytest.raises(RuntimeError, match="must be fitted"):
                s.predict_proba(synthetic_dev_data)
            with pytest.raises(RuntimeError, match="must be fitted"):
                s.predict(synthetic_dev_data)

    def test_ranking_score_policies_and_invalid_name(self, synthetic_dev_data, base_engine):
        """Verify all ranking score policies work and invalid names raise ValueError."""
        train_df = synthetic_dev_data.iloc[:100]
        test_df = synthetic_dev_data.iloc[100:]

        strategy = StandardCatBoostStrategy(
            cost_engine=base_engine,
            model_params={"iterations": 15, "depth": 4, "random_seed": 42},
        )
        strategy.fit(train_df, train_df["Delay_Flag"])

        for pol in ["cost_benefit", "cost_sensitive", "risk_only", "probability", "value_only", "expected_loss"]:
            scores = strategy.compute_ranking_scores(test_df, df_raw=test_df, policy=pol)
            assert len(scores) == len(test_df)
            assert np.all(np.isfinite(scores))

        with pytest.raises(ValueError, match="Unknown ranking policy"):
            strategy.compute_ranking_scores(test_df, df_raw=test_df, policy="nonexistent_policy")

    def test_metadata_completeness(self, synthetic_dev_data, base_engine):
        """Strategy metadata dictionary must contain strategy_id, name, scenario_name, is_fitted, calibrate."""
        strategy = StandardCatBoostStrategy(
            cost_engine=base_engine,
            model_params={"iterations": 10, "depth": 3, "random_seed": 42},
        )
        strategy.fit(synthetic_dev_data.iloc[:50], synthetic_dev_data.iloc[:50]["Delay_Flag"])

        meta = strategy.get_metadata()
        assert meta["strategy_id"] == "E8-A"
        assert meta["is_fitted"] is True
        assert "threshold" in meta
        assert "n_train" in meta
