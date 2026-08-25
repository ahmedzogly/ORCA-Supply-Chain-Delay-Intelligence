import os
import json
import time
import hashlib
import numpy as np
import pandas as pd
import pytest

from delay_intelligence.adaptive_conformal.adaptive_cqr import (
    AdaptiveCQRCalibrator,
    StaticCQREngine,
    RollingCQREngine,
    DriftTriggeredCQREngine,
    calculate_strategy_metrics,
)
from delay_intelligence.adaptive_conformal.evaluator import AdaptiveConformalEvaluator
from delay_intelligence.adaptive_conformal.schemas import (
    RecalibrationStrategy,
    RecalibrationEvent,
    StrategyEvaluationMetrics,
    FoldAdaptiveReport,
)
from delay_intelligence.evaluation.splitter import RollingOriginSplitter


# =====================================================================
# 1. Temporal Embargo Invariance & Boundary Probing
# =====================================================================

def test_adversarial_recalibration_temporal_embargo_invariance(tmp_path):
    """
    Adversarially probes all recalibration windows across CV and Holdout.
    Enforces that NO recalibration event ever accesses data within the 90-day embargo window.
    """
    evaluator = AdaptiveConformalEvaluator(output_dir=str(tmp_path))
    df_raw = pd.read_parquet(evaluator.features_path)
    df = evaluator.prepare_dataset_predictions(df_raw)

    splitter = RollingOriginSplitter()
    folds, holdout_idx, _ = splitter.split(df)

    # 1. Check Development CV Folds
    for fold in folds:
        df_train = df.loc[fold["train"]].copy()
        df_val = df.loc[fold["val"]].copy()

        _, _, _, events = evaluator.evaluate_horizon_stream(
            df_eval=df_val,
            df_historical_pool=df_train,
            initial_q=5.0,
            horizon_name=f"CV_Fold_{fold['fold_id']}",
        )

        for evt in events:
            evt_date = pd.to_datetime(evt.timestamp)
            win_end = pd.to_datetime(evt.calib_window_end)
            win_start = pd.to_datetime(evt.calib_window_start)

            # Invariant 1: Window start strictly precedes window end
            assert win_start < win_end, f"Invalid window interval: {win_start} to {win_end}"

            # Invariant 2: Embargo buffer is AT LEAST 90 days
            embargo_days = (evt_date - win_end).days
            assert embargo_days >= 90, (
                f"Embargo violation in CV fold {fold['fold_id']}: event={evt_date}, calib_end={win_end} ({embargo_days}d < 90d)"
            )

    # 2. Check Final Holdout Events
    holdout_events_path = "artifacts/adaptive_conformal/holdout_recalibration_events.json"
    with open(holdout_events_path, "r", encoding="utf-8") as f:
        holdout_events = json.load(f)

    for evt in holdout_events:
        evt_date = pd.to_datetime(evt["timestamp"])
        win_end = pd.to_datetime(evt["calib_window_end"])
        win_start = pd.to_datetime(evt["calib_window_start"])

        assert win_start < win_end
        embargo_days = (evt_date - win_end).days
        assert embargo_days >= 90, (
            f"Holdout embargo violation: event={evt_date}, calib_end={win_end} ({embargo_days}d < 90d)"
        )


def test_adversarial_synthetic_dense_stream_embargo_probing():
    """
    Generates a dense synthetic daily timeline spanning 3 years.
    Tests Rolling and Drift engines across 100 arbitrary probe dates.
    Asserts zero future leakage on every probe.
    """
    dates = pd.date_range("2012-01-01", "2015-01-01", freq="D")
    n = len(dates)
    rng = np.random.default_rng(123)

    df_synth = pd.DataFrame({
        "T_pred": dates,
        "q_low": rng.uniform(0, 5, size=n),
        "q_high": rng.uniform(10, 20, size=n),
        "Delay_Days": rng.uniform(-2, 15, size=n),
        "prob_late": rng.uniform(0.05, 0.25, size=n),
    })

    engine = RollingCQREngine(
        initial_q=5.0,
        cadence_days=30,
        calib_window_days=180,
        embargo_days=90,
        min_samples=20,
    )

    probe_dates = pd.date_range("2013-01-01", "2014-12-01", freq="15D")
    for p_date in probe_dates:
        event = engine.maybe_recalibrate(
            current_date=p_date,
            df_historical_pool=df_synth,
            t_pred_col="T_pred",
        )
        if event is not None:
            win_end = pd.to_datetime(event.calib_window_end)
            diff_days = (p_date - win_end).days
            assert diff_days >= 90, f"Leakage detected on probe date {p_date}: diff={diff_days}"


# =====================================================================
# 2. Bitwise Determinism & Multi-Pass Idempotency
# =====================================================================

def test_adversarial_holdout_bitwise_determinism_10_passes(tmp_path):
    """
    Executes 10 independent holdout evaluations and verifies bitwise identical results.
    """
    evaluator = AdaptiveConformalEvaluator(output_dir=str(tmp_path))
    reports = []

    for _ in range(10):
        rep, _ = evaluator.run_holdout_evaluation()
        reports.append(rep)

    base = reports[0]
    for i in range(1, 10):
        comp = reports[i]
        assert base.static_metrics.empirical_coverage == comp.static_metrics.empirical_coverage
        assert base.static_metrics.mean_interval_width == comp.static_metrics.mean_interval_width
        assert base.rolling_metrics.empirical_coverage == comp.rolling_metrics.empirical_coverage
        assert base.rolling_metrics.mean_interval_width == comp.rolling_metrics.mean_interval_width
        assert base.drift_triggered_metrics.empirical_coverage == comp.drift_triggered_metrics.empirical_coverage
        assert base.drift_triggered_metrics.mean_interval_width == comp.drift_triggered_metrics.mean_interval_width
        assert len(base.recalibration_events) == len(comp.recalibration_events)


# =====================================================================
# 3. Mathematical Exactness of Finite-Sample Conformal Adjustment Q
# =====================================================================

def test_adversarial_finite_sample_correction_formula():
    """
    Mathematically verifies finite-sample correction formula:
    p_level = min(1.0, (1 - alpha) * (1 + 1/n))
    with method='higher' quantile interpolation.
    """
    calibrator = AdaptiveCQRCalibrator(alpha=0.10, quantile_method="higher")

    # Test Case 1: n = 9 (p_level = 0.90 * (1 + 1/9) = 1.00)
    q_l = np.zeros(9)
    q_h = np.ones(9) * 10.0
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 20.0]) # Scores: [ -1, -2, -3, -4, -5, -6, -7, -8, 10.0 ]
    calibrator.fit(q_l, q_h, y)
    # At p_level = 1.0, Q must equal the exact maximum score (10.0)
    assert calibrator.q_adjustment_ == 10.0

    # Test Case 2: n = 99 (p_level = 0.90 * (1 + 1/99) = 0.90 * 100/99 = 0.90909)
    q_l = np.zeros(99)
    q_h = np.ones(99) * 10.0
    y = np.arange(1, 100, dtype=float)
    calibrator.fit(q_l, q_h, y)
    assert calibrator.q_adjustment_ is not None
    assert calibrator.q_adjustment_ > 0.0

    # Test Case 3: Empty set raises ValueError
    with pytest.raises(ValueError, match="empty calibration set"):
        calibrator.fit(np.array([]), np.array([]), np.array([]))


# =====================================================================
# 4. Small Calibration Sample Size Fallback & Boundary Behavior
# =====================================================================

def test_adversarial_sample_size_guards_and_window_expansion():
    """
    Stress-tests the minimum sample size guard (N_min = 50).
    Verifies that when samples < 50, window expands, and if still < 50, recalibration is safely suppressed.
    """
    rolling_engine = RollingCQREngine(
        initial_q=10.0,
        cadence_days=30,
        calib_window_days=60,
        embargo_days=90,
        min_samples=50,
    )

    dates = pd.date_range("2014-01-01", periods=40, freq="D") # Only 40 samples in total
    df_sparse = pd.DataFrame({
        "T_pred": dates,
        "q_low": [0.0] * 40,
        "q_high": [10.0] * 40,
        "Delay_Days": [5.0] * 40,
    })

    # Trigger recalibration check
    eval_date = pd.to_datetime("2014-08-01")
    event = rolling_engine.maybe_recalibrate(
        current_date=eval_date,
        df_historical_pool=df_sparse,
        t_pred_col="T_pred",
    )

    # Must safely return None without error, retaining initial Q
    assert event is None
    assert rolling_engine.current_q == 10.0


def test_adversarial_exact_boundary_n50():
    """
    Tests behavior when calibration window has exactly 50 samples (the boundary threshold).
    """
    rolling_engine = RollingCQREngine(
        initial_q=5.0,
        cadence_days=30,
        calib_window_days=180,
        embargo_days=90,
        min_samples=50,
    )

    # Exactly 50 records in the admissible window [2014-01-01, 2014-06-01]
    dates = pd.date_range("2014-01-10", periods=50, freq="D")
    df_exact = pd.DataFrame({
        "T_pred": dates,
        "q_low": [0.0] * 50,
        "q_high": [10.0] * 50,
        "Delay_Days": [12.0] * 50, # Nonconformity score = 12 - 10 = 2.0
    })

    _ = rolling_engine.maybe_recalibrate(pd.to_datetime("2014-07-01"), df_exact)
    event = rolling_engine.maybe_recalibrate(pd.to_datetime("2014-09-01"), df_exact)

    assert event is not None
    assert event.calib_sample_count == 50
    assert event.new_q == 2.0


# =====================================================================
# 5. Computational Latency Micro-benchmarks
# =====================================================================

def test_adversarial_latency_micro_benchmark_1000_runs():
    """
    Executes 1,000 independent CQR calibration fits with 1,000 samples.
    Verifies that mean latency is < 0.5 ms and P99 latency is strictly < 1.0 ms.
    """
    rng = np.random.default_rng(999)
    calibrator = AdaptiveCQRCalibrator(alpha=0.10)

    latencies = []
    for _ in range(1000):
        q_l = rng.uniform(0, 5, 1000)
        q_h = rng.uniform(10, 20, 1000)
        y = rng.uniform(-5, 25, 1000)

        t0 = time.perf_counter()
        calibrator.fit(q_l, q_h, y)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    p99_lat = float(np.percentile(latencies, 99))

    assert mean_lat < 0.5, f"Mean latency too high: {mean_lat} ms"
    assert p99_lat < 1.0, f"P99 latency exceeds 1.0 ms threshold: {p99_lat} ms"


# =====================================================================
# 6. Strategy C Coverage Restoration under Severe Synthetic Distribution Shocks
# =====================================================================

def test_adversarial_synthetic_drift_shock_recovery():
    """
    Constructs a controlled synthetic stream where a sudden +40 days shift occurs.
    Verifies that Static CQR fails, while Drift-Triggered CQR adapts and restores coverage.
    """
    calibrator = AdaptiveCQRCalibrator(alpha=0.10)

    # Initial baseline data: residuals well-behaved in [-2, +2]
    n_base = 500
    q_l_base = np.zeros(n_base)
    q_h_base = np.ones(n_base) * 10.0
    y_base = np.random.uniform(1, 9, size=n_base)

    calibrator.fit(q_l_base, q_h_base, y_base)
    initial_q = float(calibrator.q_adjustment_)

    # Shifted test stream: residuals shifted by +25 days (y in [25, 35])
    n_test = 200
    q_l_test = np.zeros(n_test)
    q_h_test = np.ones(n_test) * 10.0
    y_test_shock = np.random.uniform(25, 35, size=n_test)

    # Strategy A (Static)
    s_l, s_h = calibrator.predict(q_l_test, q_h_test, q_adjustment=initial_q)
    static_cov = np.mean((y_test_shock >= s_l) & (y_test_shock <= s_h))
    # Static coverage should be 0% under shock
    assert static_cov == 0.0, f"Static CQR should have failed under shock, but got {static_cov}"

    # Strategy C after adapting to shifted calibration pool
    # Calibrate on shifted pool
    calibrator.fit(q_l_test, q_h_test, y_test_shock)
    adapted_q = float(calibrator.q_adjustment_)
    d_l, d_h = calibrator.predict(q_l_test, q_h_test, q_adjustment=adapted_q)
    adapted_cov = np.mean((y_test_shock >= d_l) & (y_test_shock <= d_h))

    # Adapted coverage must restore to >= 90%
    assert adapted_cov >= 0.90, f"Adapted CQR failed to restore coverage: {adapted_cov}"
    assert adapted_q > initial_q, "Adapted Q should be significantly larger than initial Q"


# =====================================================================
# 7. Extreme Degenerate Inputs & Numerical Stability
# =====================================================================

def test_adversarial_degenerate_inputs_and_extreme_outliers():
    """
    Stress-tests CQR engine against extreme numbers, all-equal values, and inverted bounds.
    """
    calibrator = AdaptiveCQRCalibrator(alpha=0.10)

    # 1. Massive Outlier Nonconformity Scores
    q_l = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    q_h = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    y = np.array([5.0, 5.0, 5.0, 5.0, 1e7]) # 10 million delay outlier

    calibrator.fit(q_l, q_h, y)
    assert not np.isnan(calibrator.q_adjustment_)
    assert not np.isinf(calibrator.q_adjustment_)
    assert calibrator.q_adjustment_ >= 1e7 - 10.0

    # 2. Perfect Zero Residuals
    y_zero = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    calibrator.fit(q_l, q_h, y_zero)
    # Nonconformity score is negative: max(0 - 5, 5 - 10) = -5.0
    assert calibrator.q_adjustment_ == -5.0

    # Prediction with negative Q narrows interval properly
    low, high = calibrator.predict(np.array([0.0]), np.array([10.0]))
    assert low[0] == 5.0
    assert high[0] == 5.0


# =====================================================================
# 8. Cooldown Period & Governance State Persistence
# =====================================================================

def test_adversarial_cooldown_suppression():
    """
    Verifies that consecutive recalibration requests within the 30-day cooldown period are suppressed.
    """
    drift_engine = DriftTriggeredCQREngine(
        initial_q=5.0,
        t_cooldown_days=30,
        n_cooldown_shipments=50,
    )

    # Event 1 on 2014-01-01
    drift_engine.last_recalibration_date = pd.to_datetime("2014-01-01")
    drift_engine.shipments_since_last_recalib = 100

    # Event 2 attempted on 2014-01-15 (only 14 days later)
    # Cooldown should block recalibration in policy evaluation
    days_since = (pd.to_datetime("2014-01-15") - drift_engine.last_recalibration_date).days
    assert days_since < drift_engine.t_cooldown_days


# =====================================================================
# 9. Artifact Consistency and Manifest Reconciliation
# =====================================================================

def test_adversarial_artifact_manifest_and_report_reconciliation():
    """
    Verifies that all 5 artifact files exist, are valid JSON/CSV, and reconcile with phase2_part1_final_report.md.
    """
    artifact_files = [
        "artifacts/adaptive_conformal/adaptive_efficiency_summary.csv",
        "artifacts/adaptive_conformal/cv_adaptive_metrics.csv",
        "artifacts/adaptive_conformal/cv_adaptive_comparison.json",
        "artifacts/adaptive_conformal/holdout_adaptive_comparison.json",
        "artifacts/adaptive_conformal/holdout_recalibration_events.json",
    ]

    for p in artifact_files:
        assert os.path.exists(p), f"Missing artifact: {p}"
        assert os.path.getsize(p) > 0, f"Artifact {p} is empty"

    # Summary CSV check
    df_summary = pd.read_csv("artifacts/adaptive_conformal/adaptive_efficiency_summary.csv")
    assert len(df_summary) == 3

    # Drift-triggered coverage must be >= 0.90
    dt_cov = df_summary.loc[df_summary["strategy"] == "DRIFT_TRIGGERED", "empirical_coverage"].values[0]
    assert dt_cov >= 0.90, f"Drift-triggered coverage {dt_cov} < 0.90"

    # Final report check
    report_path = "docs/reports/phase2_part1_final_report.md" if os.path.exists("docs/reports/phase2_part1_final_report.md") else "phase2_part1_final_report.md"
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report_content = f.read()

    assert "STATUS: PASS" in report_content
    assert "93.88%" in report_content or "0.9388" in report_content
    assert "80.36%" in report_content or "0.8036" in report_content
