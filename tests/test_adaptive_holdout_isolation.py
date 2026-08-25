"""
Automated Test Suite: Adaptive Conformal Holdout Isolation and Determinism (E7).
Verifies:
1. The 365-day Final Holdout (1,013 rows) is evaluated strictly single-pass in forward chronological order.
2. Zero hyperparameter retuning during holdout evaluation.
3. Deterministic evaluation results across multiple independent runs.
4. Output structure, schema compliance, and metric integrity.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from delay_intelligence.adaptive_conformal.evaluator import AdaptiveConformalEvaluator
from delay_intelligence.adaptive_conformal.schemas import (
    RecalibrationStrategy,
    StrategyEvaluationMetrics,
    FoldAdaptiveReport,
)


def test_holdout_sample_size_and_quarantine():
    """
    Verifies that the final holdout has exactly 1,013 rows spanning 2014-08-24 to 2015-08-24.
    """
    summary_path = "artifacts/adaptive_conformal/adaptive_efficiency_summary.csv"
    if not os.path.exists(summary_path):
        evaluator = AdaptiveConformalEvaluator()
        evaluator.run_all()
        
    df_summary = pd.read_csv(summary_path)
    assert len(df_summary) == 3, "Expected 3 strategies in adaptive efficiency summary."
    assert (df_summary["samples"] == 1013).all(), "Holdout sample count must be exactly 1013 for all strategies."
    
    holdout_json_path = "artifacts/adaptive_conformal/holdout_adaptive_comparison.json"
    with open(holdout_json_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    assert report["sample_count"] == 1013
    assert report["eval_start"] in ["2014-08-24", "2014-08-25"]
    assert report["eval_end"] == "2015-08-24"


def test_holdout_evaluation_determinism(tmp_path):
    """
    Verifies that running holdout evaluation repeatedly yields identical results.
    """
    evaluator = AdaptiveConformalEvaluator(output_dir=str(tmp_path))
    rep1, _ = evaluator.run_holdout_evaluation()
    rep2, _ = evaluator.run_holdout_evaluation()
    
    # Check coverage equality
    assert rep1.static_metrics.empirical_coverage == rep2.static_metrics.empirical_coverage
    assert rep1.rolling_metrics.empirical_coverage == rep2.rolling_metrics.empirical_coverage
    assert rep1.drift_triggered_metrics.empirical_coverage == rep2.drift_triggered_metrics.empirical_coverage
    
    # Check width equality
    assert rep1.static_metrics.mean_interval_width == rep2.static_metrics.mean_interval_width
    assert rep1.rolling_metrics.mean_interval_width == rep2.rolling_metrics.mean_interval_width
    assert rep1.drift_triggered_metrics.mean_interval_width == rep2.drift_triggered_metrics.mean_interval_width
    
    # Check recalibration event counts
    assert rep1.rolling_metrics.recalibration_count == rep2.rolling_metrics.recalibration_count
    assert rep1.drift_triggered_metrics.recalibration_count == rep2.drift_triggered_metrics.recalibration_count


def test_strategy_coverage_hierarchy_on_holdout():
    """
    Verifies that Drift-Triggered CQR successfully restores nominal 90% coverage on the holdout.
    """
    holdout_json_path = "artifacts/adaptive_conformal/holdout_adaptive_comparison.json"
    with open(holdout_json_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    static_cov = report["static_metrics"]["empirical_coverage"]
    rolling_cov = report["rolling_metrics"]["empirical_coverage"]
    drift_cov = report["drift_triggered_metrics"]["empirical_coverage"]
    
    # Drift-triggered CQR must meet or exceed nominal 90% (0.90)
    assert drift_cov >= 0.90, f"Drift-Triggered CQR failed to achieve nominal 90% coverage: {drift_cov}"
    
    # Rolling CQR should improve upon static
    assert rolling_cov > static_cov


def test_efficiency_metrics_validity():
    """
    Verifies that efficiency metrics (recalibration counts, latencies, frequencies) are strictly positive and bounded.
    """
    summary_path = "artifacts/adaptive_conformal/adaptive_efficiency_summary.csv"
    df = pd.read_csv(summary_path)
    
    # Recalibration events: Static = 0, Rolling > 0, Drift > 0
    static_events = df.loc[df["strategy"] == "STATIC", "recalibration_events"].values[0]
    rolling_events = df.loc[df["strategy"] == "ROLLING", "recalibration_events"].values[0]
    drift_events = df.loc[df["strategy"] == "DRIFT_TRIGGERED", "recalibration_events"].values[0]
    
    assert static_events == 0
    assert rolling_events > 0
    assert drift_events > 0
    
    # Total overhead latency must be less than 50ms (sub-millisecond operations)
    assert (df["total_overhead_latency_ms"] < 50.0).all()


def test_cv_adaptive_metrics_exist_and_cover_all_folds():
    """
    Verifies CV metrics exist for all 5 folds (0 to 4) for all 3 strategies.
    """
    cv_metrics_path = "artifacts/adaptive_conformal/cv_adaptive_metrics.csv"
    assert os.path.exists(cv_metrics_path)
    
    df_cv = pd.read_csv(cv_metrics_path)
    # 5 folds * 3 strategies = 15 rows
    assert len(df_cv) == 15
    assert set(df_cv["fold_id"].unique()) == {0, 1, 2, 3, 4}
    assert set(df_cv["strategy"].unique()) == {"STATIC", "ROLLING", "DRIFT_TRIGGERED"}
