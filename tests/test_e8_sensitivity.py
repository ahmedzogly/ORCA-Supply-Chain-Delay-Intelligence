"""
Comprehensive Unit & Integration Tests for E8 Cost Sensitivity & Policy Robustness.

Covers:
- RobustnessClassification enum and classification thresholds (ROBUST, SENSITIVE, UNSUPPORTED).
- Dynamic creation of perturbed CostScenario models with strict parameter bounds.
- 1D parameter perturbation sweeps across all 8 core economic parameters (+/-20%, +/-50%).
- Multi-parameter joint stress tests (penalties, frictions, efficacy, asymmetry extremes).
- Mathematical win-rate calculation and robustness classification verification.
- End-to-end sensitivity analysis execution over dev backtest data and JSON artifact generation.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from delay_intelligence.cost_sensitive.cost_engine import CostScenario
from delay_intelligence.cost_sensitive.sensitivity import (
    CostSensitivityAnalyzer,
    PolicyRobustnessReport,
    RobustnessClassification,
    run_e8_dev_sensitivity_analysis,
)


# =============================================================================
# 1. Unit Tests: Scenario Perturbation & Classification Logic
# =============================================================================

def test_robustness_classification_enum():
    assert RobustnessClassification.ROBUST.value == "ROBUST"
    assert RobustnessClassification.SENSITIVE.value == "SENSITIVE"
    assert RobustnessClassification.UNSUPPORTED.value == "UNSUPPORTED"


def test_create_perturbed_scenario_bounds_and_values():
    analyzer = CostSensitivityAnalyzer(base_scenario_name="base")

    # 1. Test 50% increase in daily penalty
    sc_plus50 = analyzer.create_perturbed_scenario(
        perturbations={"c_daily_base": 1.50},
        base_scenario_name="base",
    )
    assert pytest.approx(sc_plus50.c_daily_base) == 150.0 * 1.50
    # Other parameters unchanged
    assert pytest.approx(sc_plus50.c_fixed_stockout) == 500.0

    # 2. Test bounds on days_saved_efficacy (cannot exceed delay_days_assumed)
    sc_eff = analyzer.create_perturbed_scenario(
        perturbations={"days_saved_efficacy": 3.0},  # 5 * 3 = 15 > delay_days_assumed (12)
        base_scenario_name="base",
    )
    assert sc_eff.days_saved_efficacy < sc_eff.delay_days_assumed


def test_classify_robustness_decision_boundaries():
    # Build synthetic perturbation comparison data
    # Case 1: 9 wins out of 10 -> Win Rate 90% -> ROBUST
    records_robust = []
    for i in range(10):
        records_robust.append({
            "perturbation_type": "one_at_a_time",
            "parameter_name": f"param_{i}",
            "multiplier": 1.0,
            "delta_pct": 0.0,
            "strategy": "CANDIDATE",
            "realized_cost": 100.0 if i < 9 else 300.0,
            "do_nothing_cost": 1000.0,
            "net_savings": 900.0 if i < 9 else 700.0,
        })
        records_robust.append({
            "perturbation_type": "one_at_a_time",
            "parameter_name": f"param_{i}",
            "multiplier": 1.0,
            "delta_pct": 0.0,
            "strategy": "BASELINE",
            "realized_cost": 200.0,
            "do_nothing_cost": 1000.0,
            "net_savings": 800.0,
        })

    report_robust = CostSensitivityAnalyzer.classify_robustness(
        results_df=pd.DataFrame(records_robust),
        candidate_strategy="CANDIDATE",
        reference_baseline="BASELINE",
    )
    assert report_robust.classification == RobustnessClassification.ROBUST
    assert report_robust.win_rate == 0.90
    assert report_robust.wins_count == 9

    # Case 2: 6 wins out of 10 -> Win Rate 60% -> SENSITIVE
    records_sensitive = []
    for i in range(10):
        records_sensitive.append({
            "perturbation_type": "one_at_a_time",
            "parameter_name": f"param_{i}",
            "multiplier": 1.0,
            "delta_pct": 0.0,
            "strategy": "CANDIDATE",
            "realized_cost": 100.0 if i < 6 else 300.0,
            "do_nothing_cost": 1000.0,
            "net_savings": 900.0 if i < 6 else 700.0,
        })
        records_sensitive.append({
            "perturbation_type": "one_at_a_time",
            "parameter_name": f"param_{i}",
            "multiplier": 1.0,
            "delta_pct": 0.0,
            "strategy": "BASELINE",
            "realized_cost": 200.0,
            "do_nothing_cost": 1000.0,
            "net_savings": 800.0,
        })

    report_sensitive = CostSensitivityAnalyzer.classify_robustness(
        results_df=pd.DataFrame(records_sensitive),
        candidate_strategy="CANDIDATE",
        reference_baseline="BASELINE",
    )
    assert report_sensitive.classification == RobustnessClassification.SENSITIVE
    assert report_sensitive.win_rate == 0.60

    # Case 3: 3 wins out of 10 -> Win Rate 30% -> UNSUPPORTED
    records_unsupported = []
    for i in range(10):
        records_unsupported.append({
            "perturbation_type": "one_at_a_time",
            "parameter_name": f"param_{i}",
            "multiplier": 1.0,
            "delta_pct": 0.0,
            "strategy": "CANDIDATE",
            "realized_cost": 100.0 if i < 3 else 300.0,
            "do_nothing_cost": 1000.0,
            "net_savings": 900.0 if i < 3 else 700.0,
        })
        records_unsupported.append({
            "perturbation_type": "one_at_a_time",
            "parameter_name": f"param_{i}",
            "multiplier": 1.0,
            "delta_pct": 0.0,
            "strategy": "BASELINE",
            "realized_cost": 200.0,
            "do_nothing_cost": 1000.0,
            "net_savings": 800.0,
        })

    report_unsupported = CostSensitivityAnalyzer.classify_robustness(
        results_df=pd.DataFrame(records_unsupported),
        candidate_strategy="CANDIDATE",
        reference_baseline="BASELINE",
    )
    assert report_unsupported.classification == RobustnessClassification.UNSUPPORTED
    assert report_unsupported.win_rate == 0.30


# =============================================================================
# 2. Integration Tests: Real Backtest Evaluation & Artifact Generation
# =============================================================================

def test_run_e8_dev_sensitivity_analysis_e2e(tmp_path):
    parquet_path = Path("artifacts/results/e8_dev_backtest_results.parquet")
    if not parquet_path.exists():
        pytest.skip("Development backtest results parquet file not found")

    out_json = tmp_path / "test_sensitivity_results.json"
    results = run_e8_dev_sensitivity_analysis(
        backtest_parquet_path=parquet_path,
        output_json_path=out_json,
        scenario_name="base",
    )

    assert out_json.exists()
    assert "metadata" in results
    assert "strategy_robustness_reports" in results
    assert "budget_robustness_reports" in results
    assert "parameter_sensitivity_curves" in results

    strat_reports = results["strategy_robustness_reports"]
    assert "E8-C_tuned_gamma" in strat_reports
    assert "E8-C_bayes_threshold" in strat_reports
    assert "E8-B_cost_weighted" in strat_reports

    # E8-C champion must maintain ROBUST classification vs Standard CatBoost E8-A_f1
    e8c_report = strat_reports["E8-C_tuned_gamma"]
    assert e8c_report["classification"] == "ROBUST"
    assert e8c_report["win_rate"] >= 0.85

    # Budget robustness for 10% budget: COST_SENSITIVE vs VALUE_ONLY must be ROBUST
    budget_reports = results["budget_robustness_reports"]
    assert "k_10pct" in budget_reports
    assert "BUDGET_VALUE_ONLY_k10" in budget_reports["k_10pct"]
    cs_vs_val = budget_reports["k_10pct"]["BUDGET_VALUE_ONLY_k10"]
    assert cs_vs_val["classification"] == "ROBUST"
