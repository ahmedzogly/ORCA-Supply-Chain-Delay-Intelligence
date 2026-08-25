"""
Test Suite: Chronological Drift Detection Determinism and Mathematical Rigor.
Verifies:
1. Mathematical metric axioms (PSI identity, non-negativity, Laplace smoothing).
2. Wasserstein-1 metric axioms (identity, symmetry, triangle inequality, monotonicity).
3. Categorical metrics (JSD, JS Distance, Chi-Square pooling, Categorical PSI).
4. Uncertainty CQR drift metrics (nonconformity shift, binomial undercoverage test).
5. 3-Tier trigger policy state transitions (Tier 1 veto, sample size guard, stale calibration, cooldown).
6. 100% bitwise determinism and reproducibility across independent runs.
"""

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


def test_psi_mathematical_properties_and_robustness():
    """
    Verifies metric properties of Laplace-regularized PSI:
    - Identity: PSI(P, P) == 0.0
    - Non-negativity: PSI(P, Q) >= 0.0
    - Laplace smoothing avoids ZeroDivisionError or Inf on unpopulated bins
    - Hand-calculated 2-bin distribution analytical verification
    """
    np.random.seed(42)
    p = np.random.normal(loc=10.0, scale=2.0, size=1000)
    
    # 1. Identity
    psi_ident = calculate_psi(p, p)
    assert psi_ident == 0.0, f"Expected PSI(P, P) == 0.0, got {psi_ident}"
    
    # 2. Non-negativity on shifted distribution
    q = np.random.normal(loc=12.0, scale=2.0, size=1000)
    psi_shifted = calculate_psi(p, q)
    assert psi_shifted > 0.0, f"Expected PSI > 0.0, got {psi_shifted}"
    
    # 3. Robustness against completely disjoint / empty bins
    disjoint_q = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    psi_disjoint = calculate_psi(p, disjoint_q)
    assert np.isfinite(psi_disjoint), f"PSI on disjoint bins returned non-finite: {psi_disjoint}"
    assert psi_disjoint > 0.5, "Disjoint distributions should yield large PSI"

    # 4. Analytical match for known 2-bin distribution
    # Bin 1: (-inf, 0], Bin 2: (0, +inf)
    # Ref: 50 in bin 1, 50 in bin 2 (p1=0.5, p2=0.5)
    # Det: 20 in bin 1, 80 in bin 2 (q1=0.2, q2=0.8)
    ref_exact = np.concatenate([np.full(50, -1.0), np.full(50, 1.0)])
    det_exact = np.concatenate([np.full(20, -1.0), np.full(80, 1.0)])
    psi_exact = calculate_psi(ref_exact, det_exact, num_bins=2, epsilon=1e-7)
    
    # Analytical: 0.3 * ln(4) = 0.4158883
    expected_psi = (0.2 - 0.5) * np.log(0.2 / 0.5) + (0.8 - 0.5) * np.log(0.8 / 0.5)
    assert np.isclose(psi_exact, expected_psi, atol=1e-3), f"Expected {expected_psi}, got {psi_exact}"


def test_wasserstein_metric_axioms():
    """
    Verifies metric space axioms for 1-Wasserstein Distance:
    - Identity of Indiscernibles: W_1(P, P) == 0.0
    - Symmetry: W_1(P, Q) == W_1(Q, P)
    - Triangle Inequality: W_1(P, R) <= W_1(P, Q) + W_1(Q, R)
    - Monotonicity under shift
    """
    np.random.seed(123)
    p = np.random.uniform(0, 10, size=500)
    q = np.random.uniform(2, 12, size=500)
    r = np.random.uniform(5, 15, size=500)
    
    # 1. Identity
    assert calculate_normalized_wasserstein(p, p) == 0.0
    
    # 2. Symmetry (Raw Wasserstein)
    w_pq = stats.wasserstein_distance(p, q)
    w_qp = stats.wasserstein_distance(q, p)
    assert np.isclose(w_pq, w_qp), f"Symmetry violated: W(P, Q)={w_pq}, W(Q, P)={w_qp}"
    
    # 3. Triangle Inequality: W(P, R) <= W(P, Q) + W(Q, R)
    w_pr = stats.wasserstein_distance(p, r)
    w_qr = stats.wasserstein_distance(q, r)
    assert w_pr <= w_pq + w_qr + 1e-7, f"Triangle inequality violated: {w_pr} > {w_pq} + {w_qr}"
    
    # 4. Monotonicity
    p_shift_1 = p + 1.0
    p_shift_5 = p + 5.0
    w_1 = stats.wasserstein_distance(p, p_shift_1)
    w_5 = stats.wasserstein_distance(p, p_shift_5)
    assert w_5 > w_1, f"Monotonicity violated: W(P, P+5)={w_5} <= W(P, P+1)={w_1}"


def test_categorical_metrics_axioms():
    """
    Verifies categorical drift metrics:
    - JSD and JS Distance identity and bounded range [0, 1]
    - Categorical PSI identity and non-negativity
    - Chi-Square test rare category pooling and p-values
    """
    ref_cats = pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 20)
    det_cats_same = pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 20)
    det_cats_shifted = pd.Series(["A"] * 10 + ["B"] * 10 + ["C"] * 80)
    
    # 1. Identity
    jsd_same, js_dist_same = calculate_categorical_jsd(ref_cats, det_cats_same)
    assert jsd_same == 0.0 and js_dist_same == 0.0
    assert calculate_categorical_psi(ref_cats, det_cats_same) == 0.0
    
    # 2. Range & Shift
    jsd_shift, js_dist_shift = calculate_categorical_jsd(ref_cats, det_cats_shifted)
    assert 0.0 <= js_dist_shift <= 1.0
    assert js_dist_shift > 0.20, f"Expected severe JS distance shift, got {js_dist_shift}"
    
    # 3. Chi-Square test
    chi2_stat_same, pval_same = calculate_chi_square_test(ref_cats, det_cats_same)
    assert pval_same > 0.90, f"Expected high p-value for identical, got {pval_same}"
    
    chi2_stat_shift, pval_shift = calculate_chi_square_test(ref_cats, det_cats_shifted)
    assert pval_shift < 0.001, f"Expected small p-value for shifted, got {pval_shift}"

    # 4. Rare category pooling handles unseen category without crash
    det_with_rare = pd.Series(["A"] * 48 + ["B"] * 30 + ["C"] * 20 + ["UNSEEN_NEW"] * 2)
    chi2_rare, pval_rare = calculate_chi_square_test(ref_cats, det_with_rare, min_freq=5)
    assert np.isfinite(chi2_rare) and np.isfinite(pval_rare)


def test_uncertainty_drift_metrics_axioms():
    """
    Verifies CQR uncertainty drift metrics:
    - Coverage calculation
    - Nonconformity Wasserstein distance
    - Exact one-sided binomial test undercoverage detection
    """
    np.random.seed(99)
    n = 200
    y_calib = np.random.normal(loc=5.0, scale=2.0, size=n)
    q_low_calib = y_calib - 2.5
    q_high_calib = y_calib + 2.5
    
    # Nominal case: test distribution matches calibration
    y_det_nominal = np.random.normal(loc=5.0, scale=2.0, size=n)
    q_low_det = y_det_nominal - 2.5
    q_high_det = y_det_nominal + 2.5
    
    res_nominal = calculate_uncertainty_drift(
        q_low_calib=q_low_calib,
        q_high_calib=q_high_calib,
        y_calib=y_calib,
        q_low_det=q_low_det,
        q_high_det=q_high_det,
        y_det=y_det_nominal,
        alpha=0.10,
    )
    assert res_nominal.empirical_coverage >= 0.85
    assert res_nominal.status == DriftStatus.GREEN
    assert res_nominal.binomial_pvalue > 0.05
    
    # Severe Undercoverage Case: ground truth shifted outside prediction interval
    y_det_shifted = y_det_nominal + 10.0
    res_shifted = calculate_uncertainty_drift(
        q_low_calib=q_low_calib,
        q_high_calib=q_high_calib,
        y_calib=y_calib,
        q_low_det=q_low_det,
        q_high_det=q_high_det,
        y_det=y_det_shifted,
        alpha=0.10,
    )
    assert res_shifted.empirical_coverage < 0.20
    assert res_shifted.coverage_error > 0.70
    assert res_shifted.binomial_pvalue < 1e-10
    assert res_shifted.status == DriftStatus.RED


def test_composite_trigger_state_transitions():
    """
    Verifies policy state transitions and decision rules:
    - Tier 1 Veto Rule
    - Tier 3 Non-Veto Rule
    - Multi-feature Yellow Escalation
    - Small Sample Size Guard (N < 50)
    - Stale Calibration Timeout
    - Recalibration Cooldown Period
    """
    policy = DriftTriggerPolicy(
        n_min=50,
        t_max_days=180,
        t_cooldown_days=30,
        k_persistence=2,
    )
    
    # 1. Tier 1 Feature Veto -> RED
    feat_res_veto = {
        "Vendor INCO Term": FeatureDriftResult(
            feature_name="Vendor INCO Term",
            feature_type="categorical",
            tier=FeatureTier.TIER_1,
            psi=0.28,
            status=DriftStatus.RED,
        ),
        "Unit Price": FeatureDriftResult(
            feature_name="Unit Price",
            feature_type="numerical",
            tier=FeatureTier.TIER_2,
            psi=0.02,
            status=DriftStatus.GREEN,
        ),
    }
    summary_veto = FeatureDriftSummary(
        total_features=2,
        drifted_features_count=1,
        tier1_red_count=1,
        tier1_yellow_count=0,
        tier2_red_count=0,
        tier2_yellow_count=0,
        max_psi=0.28,
        max_psi_feature="Vendor INCO Term",
        weighted_feature_score=2.5,
        status=DriftStatus.RED,
        feature_metrics=feat_res_veto,
    )
    eval_veto = policy.evaluate(sample_count=100, feature_drift=summary_veto)
    assert eval_veto.overall_status == DriftStatus.RED
    assert eval_veto.trigger_recalibration is True
    assert eval_veto.veto_triggered is True

    # 2. Tier 3 Non-Veto -> GREEN / YELLOW (no veto trigger)
    feat_res_t3 = {
        "Dosage": FeatureDriftResult(
            feature_name="Dosage",
            feature_type="categorical",
            tier=FeatureTier.TIER_3,
            psi=0.28,
            status=DriftStatus.RED,
        ),
    }
    summary_t3 = FeatureDriftSummary(
        total_features=1,
        drifted_features_count=1,
        tier1_red_count=0,
        tier1_yellow_count=0,
        tier2_red_count=0,
        tier2_yellow_count=0,
        max_psi=0.28,
        max_psi_feature="Dosage",
        weighted_feature_score=0.4,
        status=DriftStatus.GREEN,
        feature_metrics=feat_res_t3,
    )
    eval_t3 = policy.evaluate(sample_count=100, feature_drift=summary_t3)
    assert eval_t3.veto_triggered is False
    assert eval_t3.trigger_recalibration is False

    # 3. Small Sample Size Suppression (N = 35 < 50)
    eval_small = policy.evaluate(sample_count=35, feature_drift=summary_veto)
    assert eval_small.overall_status == DriftStatus.INSUFFICIENT_SAMPLE
    assert eval_small.trigger_recalibration is False
    assert eval_small.insufficient_sample is True

    # 4. Stale Calibration Timeout (days >= 180)
    eval_stale = policy.evaluate(sample_count=100, days_since_calibration=190)
    assert eval_stale.overall_status == DriftStatus.RED
    assert eval_stale.trigger_recalibration is True
    assert eval_stale.stale_calibration_triggered is True

    # 5. Cooldown Period Suppression (recalibrated 10 days ago < 30 days)
    eval_cooldown = policy.evaluate(
        sample_count=100,
        feature_drift=summary_veto,
        days_since_last_recalibration=10,
    )
    assert eval_cooldown.overall_status == DriftStatus.RED
    assert eval_cooldown.cooldown_active is True
    assert eval_cooldown.trigger_recalibration is False  # Suppressed by cooldown


def test_end_to_end_drift_determinism_across_runs():
    """
    Verifies that running drift detection twice on the same input data
    produces 100% bitwise identical results.
    """
    np.random.seed(42)
    df_ref = pd.DataFrame({
        "Unit Price": np.random.lognormal(mean=2.0, sigma=0.5, size=200),
        "Country": np.random.choice(["Vietnam", "Nigeria", "Cote d'Ivoire"], size=200),
        "Delay_Flag": np.random.binomial(1, 0.15, size=200),
        "Delay_Days": np.random.normal(loc=0.0, scale=5.0, size=200),
    })
    df_det = pd.DataFrame({
        "Unit Price": np.random.lognormal(mean=2.2, sigma=0.5, size=150),
        "Country": np.random.choice(["Vietnam", "Nigeria", "Cote d'Ivoire"], size=150),
        "Delay_Flag": np.random.binomial(1, 0.18, size=150),
        "Delay_Days": np.random.normal(loc=1.0, scale=5.0, size=150),
    })
    
    detector = ChronologicalDriftDetector(
        num_cols=["Unit Price"],
        cat_cols=["Country"],
    )
    
    prob_ref = np.random.uniform(0.05, 0.35, size=200)
    prob_det = np.random.uniform(0.05, 0.40, size=150)
    
    rep1 = detector.evaluate_window(df_ref, df_det, prob_ref, prob_det)
    rep2 = detector.evaluate_window(df_ref, df_det, prob_ref, prob_det)
    
    dict1 = rep1.to_dict()
    dict2 = rep2.to_dict()
    
    assert dict1 == dict2, "Non-deterministic behavior detected across consecutive runs!"
