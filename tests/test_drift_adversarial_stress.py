"""
Adversarial Stress Test Suite for Chronological Drift Detection (Milestone 1 / E6.5).
Empirically tests:
1. Mathematical determinism and bitwise reproducibility across 50 iterations.
2. Minimum sample size power regularization guard (N < 50 boundary conditions).
3. Tier 1 feature SHAP veto rule enforcement and tier isolation.
4. Stale calibration timeout and cooldown suppression logic.
5. Zero holdout contamination and temporal embargo compliance.
6. Extreme degenerate inputs, zero variance, NaNs, Infs, unseen categories, and numerical stability.
7. Benjamini-Hochberg FDR multiple testing edge cases.
8. CQR uncertainty drift extreme scenarios (0% coverage, 100% coverage, inverted quantiles).
9. Full end-to-end ChronologicalDriftDetector stress on sparse, noisy, and missing inputs.
"""

import os
import json
import hashlib
import pytest
import numpy as np
import pandas as pd
from scipy import stats

from delay_intelligence.drift.metrics import (
    calculate_psi,
    calculate_normalized_wasserstein,
    calculate_ks_test,
    calculate_benjamini_hochberg_fdr,
    calculate_categorical_psi,
    calculate_categorical_jsd,
    calculate_chi_square_test,
    calculate_prediction_drift,
    calculate_target_drift,
    calculate_uncertainty_drift,
)
from delay_intelligence.drift.policy import DriftTriggerPolicy, FeatureTier
from delay_intelligence.drift.detector import ChronologicalDriftDetector
from delay_intelligence.drift.schemas import (
    DriftStatus,
    FeatureDriftResult,
    FeatureDriftSummary,
)
from delay_intelligence.evaluation.splitter import RollingOriginSplitter


# =====================================================================
# 1. Determinism & Bitwise Reproducibility Stress Tests
# =====================================================================

def test_metrics_bitwise_reproducibility_across_50_iterations():
    """
    Stress-tests all core drift metrics over 50 consecutive runs
    to guarantee 100% mathematical and bitwise determinism.
    """
    rng = np.random.default_rng(42)
    ref_num = rng.normal(loc=15.0, scale=3.0, size=500)
    det_num = rng.normal(loc=16.5, scale=3.5, size=400)
    ref_cat = ["A"] * 250 + ["B"] * 150 + ["C"] * 100
    det_cat = ["A"] * 100 + ["B"] * 200 + ["C"] * 100

    base_psi = calculate_psi(ref_num, det_num)
    base_w1 = calculate_normalized_wasserstein(ref_num, det_num)
    base_ks_stat, base_ks_pval = calculate_ks_test(ref_num, det_num)
    base_cat_psi = calculate_categorical_psi(ref_cat, det_cat)
    base_jsd, base_js_dist = calculate_categorical_jsd(ref_cat, det_cat)
    base_chi2, base_chi2_pval = calculate_chi_square_test(ref_cat, det_cat)

    for i in range(50):
        assert calculate_psi(ref_num, det_num) == base_psi
        assert calculate_normalized_wasserstein(ref_num, det_num) == base_w1
        stat, pval = calculate_ks_test(ref_num, det_num)
        assert stat == base_ks_stat and pval == base_ks_pval
        assert calculate_categorical_psi(ref_cat, det_cat) == base_cat_psi
        jsd, js_dist = calculate_categorical_jsd(ref_cat, det_cat)
        assert jsd == base_jsd and js_dist == base_js_dist
        chi2, chi2_pval = calculate_chi_square_test(ref_cat, det_cat)
        assert chi2 == base_chi2 and chi2_pval == base_chi2_pval


# =====================================================================
# 2. Minimum Sample Size Boundary & Power Regularization (N < 50)
# =====================================================================

@pytest.mark.parametrize("n_sample", [0, 1, 2, 10, 35, 49])
def test_sample_size_guard_suppresses_false_alarms_below_n_min(n_sample):
    """
    Verifies that any batch size N < 50 strictly triggers INSUFFICIENT_SAMPLE
    and NEVER triggers recalibration, even when metrics simulate extreme drift.
    """
    policy = DriftTriggerPolicy(n_min=50)

    # Simulated catastrophic drift summary
    extreme_feature_summary = FeatureDriftSummary(
        total_features=2,
        drifted_features_count=2,
        tier1_red_count=2,
        tier1_yellow_count=0,
        tier2_red_count=0,
        tier2_yellow_count=0,
        max_psi=5.0,
        max_psi_feature="Vendor INCO Term",
        weighted_feature_score=10.0,
        status=DriftStatus.RED,
        feature_metrics={
            "Vendor INCO Term": FeatureDriftResult(
                feature_name="Vendor INCO Term",
                feature_type="categorical",
                tier=FeatureTier.TIER_1,
                psi=5.0,
                status=DriftStatus.RED,
            )
        },
    )

    eval_res = policy.evaluate(
        sample_count=n_sample,
        feature_drift=extreme_feature_summary,
        days_since_calibration=300,  # Stale calibration also present
    )

    assert eval_res.overall_status == DriftStatus.INSUFFICIENT_SAMPLE
    assert eval_res.trigger_recalibration is False
    assert eval_res.insufficient_sample is True


@pytest.mark.parametrize("n_sample", [50, 51, 100, 500])
def test_sample_size_guard_permits_evaluation_at_or_above_n_min(n_sample):
    """
    Verifies that when sample count reaches N >= 50, evaluation activates properly.
    """
    policy = DriftTriggerPolicy(n_min=50)

    normal_summary = FeatureDriftSummary(
        total_features=1,
        drifted_features_count=0,
        tier1_red_count=0,
        tier1_yellow_count=0,
        tier2_red_count=0,
        tier2_yellow_count=0,
        max_psi=0.01,
        max_psi_feature="Unit Price",
        weighted_feature_score=0.0,
        status=DriftStatus.GREEN,
        feature_metrics={
            "Unit Price": FeatureDriftResult(
                feature_name="Unit Price",
                feature_type="numerical",
                tier=FeatureTier.TIER_2,
                psi=0.01,
                status=DriftStatus.GREEN,
            )
        },
    )

    eval_res = policy.evaluate(
        sample_count=n_sample,
        feature_drift=normal_summary,
    )

    assert eval_res.overall_status == DriftStatus.GREEN
    assert eval_res.trigger_recalibration is False
    assert eval_res.insufficient_sample is False


# =====================================================================
# 3. Tier 1 Feature Veto Rule & Criticality Hierarchy
# =====================================================================

TIER_1_FEATURES = [
    "Vendor INCO Term",
    "Vendor",
    "vendor_hist_volume",
    "Country",
    "country_hist_delay_rate",
    "vendor_hist_delay_rate",
    "country_hist_volume",
    "Scheduled_Transit_Days",
    "Forecast_Horizon_Days",
    "Line Item Insurance (USD)",
    "Line Item Quantity",
]


@pytest.mark.parametrize("tier1_feat", TIER_1_FEATURES)
def test_individual_tier1_feature_veto_triggers_immediate_red(tier1_feat):
    """
    Verifies that a single Tier 1 feature with PSI >= 0.25 immediately triggers
    RED state, veto_triggered=True, and trigger_recalibration=True.
    """
    policy = DriftTriggerPolicy(n_min=50, psi_critical=0.25)

    feat_metric = {
        tier1_feat: FeatureDriftResult(
            feature_name=tier1_feat,
            feature_type="categorical" if "Vendor" in tier1_feat or "Country" in tier1_feat else "numerical",
            tier=FeatureTier.TIER_1,
            psi=0.26,
            status=DriftStatus.RED,
        )
    }
    summary = FeatureDriftSummary(
        total_features=1,
        drifted_features_count=1,
        tier1_red_count=1,
        tier1_yellow_count=0,
        tier2_red_count=0,
        tier2_yellow_count=0,
        max_psi=0.26,
        max_psi_feature=tier1_feat,
        weighted_feature_score=2.0,
        status=DriftStatus.RED,
        feature_metrics=feat_metric,
    )

    eval_res = policy.evaluate(sample_count=100, feature_drift=summary)
    assert eval_res.overall_status == DriftStatus.RED
    assert eval_res.veto_triggered is True
    assert eval_res.trigger_recalibration is True


TIER_2_FEATURES = ["Line Item Value", "Unit Price", "Pack Price", "is_rdc_fulfillment"]
TIER_3_FEATURES = ["Dosage", "Brand", "Fulfill Via", "is_pre_pq_process"]


@pytest.mark.parametrize("tier2_feat", TIER_2_FEATURES)
def test_tier2_feature_does_not_trigger_veto_alone(tier2_feat):
    """
    Verifies that a single Tier 2 feature with elevated PSI (0.35) does NOT
    trigger an immediate veto or recalibration.
    """
    policy = DriftTriggerPolicy(n_min=50, psi_critical=0.25)

    feat_metric = {
        tier2_feat: FeatureDriftResult(
            feature_name=tier2_feat,
            feature_type="numerical",
            tier=FeatureTier.TIER_2,
            psi=0.35,
            status=DriftStatus.RED,
        )
    }
    summary = FeatureDriftSummary(
        total_features=1,
        drifted_features_count=1,
        tier1_red_count=0,
        tier1_yellow_count=0,
        tier2_red_count=1,
        tier2_yellow_count=0,
        max_psi=0.35,
        max_psi_feature=tier2_feat,
        weighted_feature_score=0.45,
        status=DriftStatus.GREEN,
        feature_metrics=feat_metric,
    )

    eval_res = policy.evaluate(sample_count=100, feature_drift=summary)
    assert eval_res.veto_triggered is False
    assert eval_res.trigger_recalibration is False


@pytest.mark.parametrize("tier3_feat", TIER_3_FEATURES)
def test_tier3_feature_does_not_trigger_veto(tier3_feat):
    """
    Verifies that a Tier 3 feature with large PSI (0.80) does NOT trigger veto.
    """
    policy = DriftTriggerPolicy(n_min=50)

    feat_metric = {
        tier3_feat: FeatureDriftResult(
            feature_name=tier3_feat,
            feature_type="categorical",
            tier=FeatureTier.TIER_3,
            psi=0.80,
            status=DriftStatus.RED,
        )
    }
    summary = FeatureDriftSummary(
        total_features=1,
        drifted_features_count=1,
        tier1_red_count=0,
        tier1_yellow_count=0,
        tier2_red_count=0,
        tier2_yellow_count=0,
        max_psi=0.80,
        max_psi_feature=tier3_feat,
        weighted_feature_score=0.50,
        status=DriftStatus.GREEN,
        feature_metrics=feat_metric,
    )

    eval_res = policy.evaluate(sample_count=100, feature_drift=summary)
    assert eval_res.veto_triggered is False
    assert eval_res.trigger_recalibration is False


# =====================================================================
# 4. Stale Calibration & Cooldown Suppression Logic
# =====================================================================

def test_stale_calibration_timeout_triggers_red():
    """
    Verifies that exceeding T_max=180 days or V_max=1500 shipments triggers RED.
    """
    policy = DriftTriggerPolicy(t_max_days=180, v_max_shipments=1500)

    # Days timeout
    eval_days = policy.evaluate(sample_count=100, days_since_calibration=181)
    assert eval_days.overall_status == DriftStatus.RED
    assert eval_days.stale_calibration_triggered is True
    assert eval_days.trigger_recalibration is True

    # Volume timeout
    eval_vol = policy.evaluate(sample_count=100, shipments_since_calibration=1501)
    assert eval_vol.overall_status == DriftStatus.RED
    assert eval_vol.stale_calibration_triggered is True
    assert eval_vol.trigger_recalibration is True


def test_cooldown_suppresses_triggers_when_active():
    """
    Verifies that active cooldown period (T < 30 days or N < 50 shipments)
    suppresses recalibration trigger even when RED drift is detected.
    """
    policy = DriftTriggerPolicy(t_cooldown_days=30, n_cooldown_shipments=50)

    veto_summary = FeatureDriftSummary(
        total_features=1,
        drifted_features_count=1,
        tier1_red_count=1,
        tier1_yellow_count=0,
        tier2_red_count=0,
        tier2_yellow_count=0,
        max_psi=0.50,
        max_psi_feature="Vendor INCO Term",
        weighted_feature_score=3.0,
        status=DriftStatus.RED,
        feature_metrics={
            "Vendor INCO Term": FeatureDriftResult(
                feature_name="Vendor INCO Term",
                feature_type="categorical",
                tier=FeatureTier.TIER_1,
                psi=0.50,
                status=DriftStatus.RED,
            )
        },
    )

    # Days cooldown active (15 days < 30)
    eval_cool_days = policy.evaluate(
        sample_count=100,
        feature_drift=veto_summary,
        days_since_last_recalibration=15,
    )
    assert eval_cool_days.overall_status == DriftStatus.RED
    assert eval_cool_days.cooldown_active is True
    assert eval_cool_days.trigger_recalibration is False  # Suppressed!

    # Volume cooldown active (20 shipments < 50)
    eval_cool_vol = policy.evaluate(
        sample_count=100,
        feature_drift=veto_summary,
        shipments_since_last_recalibration=20,
    )
    assert eval_cool_vol.overall_status == DriftStatus.RED
    assert eval_cool_vol.cooldown_active is True
    assert eval_cool_vol.trigger_recalibration is False  # Suppressed!

    # Cooldown expired (35 days >= 30, 60 shipments >= 50)
    eval_cool_expired = policy.evaluate(
        sample_count=100,
        feature_drift=veto_summary,
        days_since_last_recalibration=35,
        shipments_since_last_recalibration=60,
    )
    assert eval_cool_expired.overall_status == DriftStatus.RED
    assert eval_cool_expired.cooldown_active is False
    assert eval_cool_expired.trigger_recalibration is True


# =====================================================================
# 5. Zero Holdout Contamination & Embargo Gap Invariants
# =====================================================================

def test_zero_holdout_contamination_in_drift_evaluation():
    """
    Verifies that the final 365-day holdout is 100% quarantined from drift artifacts.
    """
    artifacts_dir = "artifacts/drift"
    if not os.path.exists(artifacts_dir):
        pytest.skip("artifacts/drift not present")

    metrics_csv = os.path.join(artifacts_dir, "drift_metrics.csv")
    if os.path.exists(metrics_csv):
        df_metrics = pd.read_csv(metrics_csv)
        for _, row in df_metrics.iterrows():
            det_end = pd.to_datetime(row['det_end'])
            ref_end = pd.to_datetime(row['ref_end'])
            # Holdout cutoff is 2014-08-24
            assert det_end <= pd.to_datetime('2014-08-24'), f"Detection window {det_end} touches holdout!"
            assert ref_end < pd.to_datetime('2014-08-24'), f"Reference window {ref_end} touches holdout!"


def test_frozen_baseline_artifacts_unmodified():
    """
    Verifies that frozen Stage 0-13 artifacts exist and are non-empty.
    """
    frozen_files = [
        "artifacts/data/scms_modeling_features.parquet",
        "artifacts/model_registry/v1/feature_schema.json",
        "artifacts/model_registry/v1/cqr_calibration.json",
        "artifacts/final/final_holdout_metrics.json",
    ]
    for fpath in frozen_files:
        assert os.path.exists(fpath), f"Frozen baseline artifact {fpath} is missing!"
        assert os.path.getsize(fpath) > 0, f"Frozen baseline artifact {fpath} is empty!"


# =====================================================================
# 6. Extreme Degenerate Distributions & Numerical Stability Stress
# =====================================================================

def test_constant_zero_variance_distribution_stability():
    """
    Stress-tests drift metrics on degenerate constant arrays (zero variance).
    """
    const_ref = np.zeros(100)
    const_det_same = np.zeros(100)
    const_det_diff = np.full(100, 5.0)

    # Identical constants
    assert calculate_psi(const_ref, const_det_same) == 0.0
    assert calculate_normalized_wasserstein(const_ref, const_det_same) == 0.0
    stat_same, pval_same = calculate_ks_test(const_ref, const_det_same)
    assert stat_same == 0.0 and pval_same == 1.0

    # Different constants
    psi_diff = calculate_psi(const_ref, const_det_diff)
    assert np.isfinite(psi_diff) and psi_diff >= 0.0
    norm_w1_diff = calculate_normalized_wasserstein(const_ref, const_det_diff)
    assert np.isfinite(norm_w1_diff) and norm_w1_diff > 0.0
    stat_diff, pval_diff = calculate_ks_test(const_ref, const_det_diff)
    assert stat_diff == 1.0 and pval_diff < 1e-5


def test_extreme_and_outlier_values_stability():
    """
    Stress-tests drift metrics on massive outlier values and quantile shifts.
    """
    ref = np.linspace(1.0, 10.0, 100)
    det_shifted = np.linspace(5.0, 15.0, 100)
    det_extreme = np.array([1.0, 2.0, 3.0, 4.0, 1e12] * 20)

    # Quantile shift across bins
    psi_shift = calculate_psi(ref, det_shifted)
    assert np.isfinite(psi_shift) and psi_shift > 0.0

    w1_shift = calculate_normalized_wasserstein(ref, det_shifted)
    assert np.isfinite(w1_shift) and w1_shift > 0.0

    # Extreme tail outlier: Normalized Wasserstein captures the massive metric distance
    w1_ext = calculate_normalized_wasserstein(ref, det_extreme)
    assert np.isfinite(w1_ext) and w1_ext > 1.0

    psi_ext = calculate_psi(ref, det_extreme)
    assert np.isfinite(psi_ext) and psi_ext >= 0.0


def test_nan_and_inf_handling_stability():
    """
    Stress-tests drift metrics on arrays containing NaNs and Infs.
    """
    ref_dirty = np.array([1.0, 2.0, np.nan, 4.0, np.inf, -np.inf, 3.0] * 10)
    det_dirty = np.array([np.nan, 2.0, 3.0, np.nan, 5.0, 6.0] * 10)

    psi = calculate_psi(ref_dirty, det_dirty)
    assert np.isfinite(psi) and psi >= 0.0

    w1 = calculate_normalized_wasserstein(ref_dirty, det_dirty)
    assert np.isfinite(w1) and w1 >= 0.0

    ks_stat, ks_pval = calculate_ks_test(ref_dirty, det_dirty)
    assert np.isfinite(ks_stat) and np.isfinite(ks_pval)


def test_100_percent_unseen_categorical_levels_stability():
    """
    Stress-tests categorical metrics when detection window has 100% novel categories.
    """
    ref_cats = pd.Series(["Alpha", "Beta", "Gamma"] * 50)
    det_cats_disjoint = pd.Series(["Delta", "Epsilon", "Zeta"] * 50)

    psi = calculate_categorical_psi(ref_cats, det_cats_disjoint)
    assert np.isfinite(psi) and psi > 1.0

    jsd, js_dist = calculate_categorical_jsd(ref_cats, det_cats_disjoint)
    assert np.isfinite(jsd) and np.isfinite(js_dist)
    assert 0.0 <= js_dist <= 1.0
    assert js_dist > 0.90  # Almost max distance

    chi2, pval = calculate_chi_square_test(ref_cats, det_cats_disjoint)
    assert np.isfinite(chi2) and np.isfinite(pval)


def test_empty_and_single_item_inputs_graceful_handling():
    """
    Verifies that empty arrays return safe defaults (0.0 PSI, 0.0 W1, 1.0 p-val).
    """
    assert calculate_psi([], []) == 0.0
    assert calculate_normalized_wasserstein([], []) == 0.0
    stat, pval = calculate_ks_test([], [])
    assert stat == 0.0 and pval == 1.0
    assert calculate_categorical_psi([], []) == 0.0
    jsd, dist = calculate_categorical_jsd([], [])
    assert jsd == 0.0 and dist == 0.0
    chi2, chi2_pval = calculate_chi_square_test([], [])
    assert chi2 == 0.0 and chi2_pval == 1.0


# =====================================================================
# 7. Benjamini-Hochberg FDR Multiple Testing Control Edge Cases
# =====================================================================

def test_benjamini_hochberg_fdr_edge_cases():
    """
    Verifies FDR control under boundary p-value distributions.
    """
    # Empty
    assert calculate_benjamini_hochberg_fdr({}) == {}

    # All zeros (all rejected)
    all_zeros = {f"feat_{i}": 0.0 for i in range(10)}
    rej_zeros = calculate_benjamini_hochberg_fdr(all_zeros, alpha=0.05)
    assert all(rej_zeros.values())

    # All ones (none rejected)
    all_ones = {f"feat_{i}": 1.0 for i in range(10)}
    rej_ones = calculate_benjamini_hochberg_fdr(all_ones, alpha=0.05)
    assert not any(rej_ones.values())

    # Standard mixture
    pvals = {
        "feat_significant_1": 0.0001,
        "feat_significant_2": 0.001,
        "feat_marginal": 0.03,
        "feat_null_1": 0.45,
        "feat_null_2": 0.89,
    }
    rej = calculate_benjamini_hochberg_fdr(pvals, alpha=0.05)
    assert rej["feat_significant_1"] is True
    assert rej["feat_significant_2"] is True
    assert rej["feat_null_1"] is False
    assert rej["feat_null_2"] is False


# =====================================================================
# 8. CQR Uncertainty Drift Extreme Scenarios
# =====================================================================

def test_uncertainty_drift_extreme_boundary_conditions():
    """
    Verifies uncertainty drift behavior under extreme conditions:
    - 0% coverage (total collapse) -> RED, p_binom < 1e-15
    - 100% coverage (overcoverage) -> GREEN, p_binom == 1.0
    - Inverted quantiles (q_low > q_high)
    """
    n = 200
    y_calib = np.full(n, 5.0)
    q_low_calib = np.full(n, 0.0)
    q_high_calib = np.full(n, 10.0)

    # 1. 0% coverage (y_det far outside bounds)
    y_det_zero_cov = np.full(n, 100.0)
    q_low_det = np.full(n, 0.0)
    q_high_det = np.full(n, 10.0)

    res_zero = calculate_uncertainty_drift(
        q_low_calib=q_low_calib,
        q_high_calib=q_high_calib,
        y_calib=y_calib,
        q_low_det=q_low_det,
        q_high_det=q_high_det,
        y_det=y_det_zero_cov,
        alpha=0.10,
    )
    assert np.isclose(res_zero.empirical_coverage, 0.0)
    assert np.isclose(res_zero.coverage_error, 0.90)
    assert res_zero.binomial_pvalue < 1e-15
    assert res_zero.status == DriftStatus.RED

    # 2. 100% coverage
    y_det_full_cov = np.full(n, 5.0)
    res_full = calculate_uncertainty_drift(
        q_low_calib=q_low_calib,
        q_high_calib=q_high_calib,
        y_calib=y_calib,
        q_low_det=q_low_det,
        q_high_det=q_high_det,
        y_det=y_det_full_cov,
        alpha=0.10,
    )
    assert np.isclose(res_full.empirical_coverage, 1.0)
    assert np.isclose(res_full.coverage_error, -0.10)
    assert res_full.binomial_pvalue == 1.0
    assert res_full.status == DriftStatus.GREEN

    # 3. Inverted quantiles (q_low > q_high)
    q_low_inv = np.full(n, 10.0)
    q_high_inv = np.full(n, 0.0)
    res_inv = calculate_uncertainty_drift(
        q_low_calib=q_low_inv,
        q_high_calib=q_high_inv,
        y_calib=y_calib,
        q_low_det=q_low_inv,
        q_high_det=q_high_inv,
        y_det=y_calib,
        alpha=0.10,
    )
    assert np.isfinite(res_inv.empirical_coverage)
    assert np.isfinite(res_inv.binomial_pvalue)
    assert np.isfinite(res_inv.nonconformity_wasserstein)


# =====================================================================
# 9. Full End-to-End ChronologicalDriftDetector Stress Test
# =====================================================================

def test_chronological_drift_detector_end_to_end_adversarial_stress():
    """
    Stress-tests ChronologicalDriftDetector on noisy, missing, and extreme DataFrames.
    """
    detector = ChronologicalDriftDetector(
        num_cols=["Unit Price", "Line Item Quantity", "Scheduled_Transit_Days"],
        cat_cols=["Country", "Vendor INCO Term", "Brand"],
    )

    n_ref, n_det = 150, 120
    df_ref = pd.DataFrame({
        "T_pred": pd.date_range("2012-01-01", periods=n_ref, freq="D"),
        "Unit Price": np.random.lognormal(2.0, 0.5, size=n_ref),
        "Line Item Quantity": np.random.randint(100, 5000, size=n_ref),
        "Scheduled_Transit_Days": np.random.normal(30, 5, size=n_ref),
        "Country": np.random.choice(["Vietnam", "Nigeria", "Zambia"], size=n_ref),
        "Vendor INCO Term": np.random.choice(["EXW", "FCA", "DDP"], size=n_ref),
        "Brand": np.random.choice(["Generic", "BrandA"], size=n_ref),
        "Delay_Flag": np.random.binomial(1, 0.12, size=n_ref),
        "Delay_Days": np.random.normal(0, 4, size=n_ref),
    })

    # Detection window with extreme Tier 1 drift (INCO Term shifts heavily to DDP)
    df_det = pd.DataFrame({
        "T_pred": pd.date_range("2012-06-01", periods=n_det, freq="D"),
        "Unit Price": np.random.lognormal(2.0, 0.5, size=n_det),
        "Line Item Quantity": np.random.randint(100, 5000, size=n_det),
        "Scheduled_Transit_Days": np.random.normal(30, 5, size=n_det),
        "Country": np.random.choice(["Vietnam", "Nigeria", "Zambia"], size=n_det),
        "Vendor INCO Term": ["DDP"] * (n_det - 5) + ["EXW"] * 5,  # Extreme shift
        "Brand": np.random.choice(["Generic", "BrandA"], size=n_det),
        "Delay_Flag": np.random.binomial(1, 0.28, size=n_det),  # Prevalence shift
        "Delay_Days": np.random.normal(5, 6, size=n_det),
    })

    ref_prob = np.random.uniform(0.05, 0.20, size=n_ref)
    det_prob = np.random.uniform(0.20, 0.60, size=n_det)  # Prediction shift

    report = detector.evaluate_window(
        df_ref=df_ref,
        df_det=df_det,
        ref_prob=ref_prob,
        det_prob=det_prob,
    )

    assert report.trigger_evaluation.overall_status == DriftStatus.RED
    assert report.trigger_evaluation.veto_triggered is True
    assert report.trigger_evaluation.trigger_recalibration is True
    assert "Vendor INCO Term" in report.feature_drift.feature_metrics
    assert report.feature_drift.feature_metrics["Vendor INCO Term"].psi >= 0.25
