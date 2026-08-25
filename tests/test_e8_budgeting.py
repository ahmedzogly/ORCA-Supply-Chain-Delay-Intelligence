"""
Comprehensive Unit & Integration Tests for E8 Operational Review Budget Simulator.

Covers:
- OperationalPolicyType enum definitions and policy name handling.
- Mathematical correctness of priority scores across VALUE_ONLY, RISK_ONLY, STANDARD, and COST_SENSITIVE.
- Strict review capacity enforcement under K in {0.05, 0.10, 0.20}.
- Strictly-positive economic benefit gating behavior.
- Realized vs Expected cost metrics, net savings, and delay-days captured.
- Pairwise policy benchmark comparisons (Net Savings vs Value-Only, Risk-Only, Standard).
- End-to-end simulation from dev backtest results and JSON artifact validation.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from delay_intelligence.cost_sensitive.budgeting import (
    BudgetMetrics,
    OperationalBudgetSimulator,
    OperationalPolicyType,
    run_e8_dev_budget_simulation,
)
from delay_intelligence.cost_sensitive.cost_engine import CostScenarioModel


# =============================================================================
# 1. Unit Tests: Priority Scores and Decision Rules
# =============================================================================

def test_operational_policy_type_enum():
    assert OperationalPolicyType.VALUE_ONLY.value == "VALUE_ONLY"
    assert OperationalPolicyType.RISK_ONLY.value == "RISK_ONLY"
    assert OperationalPolicyType.STANDARD.value == "STANDARD"
    assert OperationalPolicyType.COST_SENSITIVE.value == "COST_SENSITIVE"


def test_compute_priority_scores_mathematical_correctness():
    probs = np.array([0.1, 0.4, 0.7, 0.9])
    values = np.array([100000.0, 5000.0, 50000.0, 10000.0])
    costs_df = pd.DataFrame({
        "fn_cost": [1000.0, 2000.0, 3000.0, 4000.0],
        "fp_cost": [50.0, 50.0, 100.0, 100.0],
        "intervention_cost": [200.0, 200.0, 300.0, 300.0],
        "residual_delay_cost": [100.0, 100.0, 200.0, 200.0],
        "net_benefit": [700.0, 1700.0, 2500.0, 3500.0],  # FN - (Interv + Resid)
    })

    # 1. VALUE_ONLY
    scores_val = OperationalBudgetSimulator.compute_priority_scores(
        OperationalPolicyType.VALUE_ONLY, probs, costs_df, values
    )
    np.testing.assert_array_equal(scores_val, values)

    # 2. RISK_ONLY
    scores_risk = OperationalBudgetSimulator.compute_priority_scores(
        OperationalPolicyType.RISK_ONLY, probs, costs_df, values
    )
    np.testing.assert_array_equal(scores_risk, probs)

    # 3. STANDARD (tau=0.5)
    scores_std = OperationalBudgetSimulator.compute_priority_scores(
        OperationalPolicyType.STANDARD, probs, costs_df, values, threshold_std=0.50
    )
    np.testing.assert_allclose(scores_std, probs - 0.50)

    # 4. COST_SENSITIVE: E[Delta Cost] = p * Net_Benefit - (1 - p) * FP_Cost
    expected_cs = probs * costs_df["net_benefit"].to_numpy() - (1.0 - probs) * costs_df["fp_cost"].to_numpy()
    # For item 0: 0.1 * 700 - 0.9 * 50 = 70 - 45 = 25
    # For item 1: 0.4 * 1700 - 0.6 * 50 = 680 - 30 = 650
    # For item 2: 0.7 * 2500 - 0.3 * 100 = 1750 - 30 = 1720
    # For item 3: 0.9 * 3500 - 0.1 * 100 = 3150 - 10 = 3140
    scores_cs = OperationalBudgetSimulator.compute_priority_scores(
        OperationalPolicyType.COST_SENSITIVE, probs, costs_df, values
    )
    np.testing.assert_allclose(scores_cs, expected_cs)
    assert pytest.approx(scores_cs[0]) == 25.0
    assert pytest.approx(scores_cs[1]) == 650.0
    assert pytest.approx(scores_cs[2]) == 1720.0
    assert pytest.approx(scores_cs[3]) == 3140.0


def test_compute_policy_decisions_budget_capacities():
    # 100 shipments cohort
    n = 100
    np.random.seed(42)
    probs = np.linspace(0.01, 0.99, n)
    values = np.random.uniform(1000, 500000, n)
    costs_df = pd.DataFrame({
        "fn_cost": np.full(n, 2000.0),
        "fp_cost": np.full(n, 50.0),
        "intervention_cost": np.full(n, 300.0),
        "residual_delay_cost": np.full(n, 200.0),
        "net_benefit": np.full(n, 1500.0),
    })

    for k in [0.05, 0.10, 0.20]:
        expected_m = int(np.floor(k * n))
        for pol in [OperationalPolicyType.VALUE_ONLY, OperationalPolicyType.RISK_ONLY, OperationalPolicyType.COST_SENSITIVE]:
            decisions, capacity = OperationalBudgetSimulator.compute_policy_decisions(
                policy=pol,
                probs=probs,
                costs_df=costs_df,
                values=values,
                budget_k=k,
            )
            assert capacity == expected_m
            assert np.sum(decisions) <= expected_m


def test_cost_sensitive_strictly_positive_benefit_gating():
    # 4 items where only 1 has positive expected net benefit
    probs = np.array([0.01, 0.02, 0.03, 0.80])
    values = np.array([1000.0, 2000.0, 3000.0, 50000.0])
    costs_df = pd.DataFrame({
        "fn_cost": [100.0, 100.0, 100.0, 5000.0],
        "fp_cost": [200.0, 200.0, 200.0, 100.0],
        "intervention_cost": [80.0, 80.0, 80.0, 500.0],
        "residual_delay_cost": [50.0, 50.0, 50.0, 300.0],
        "net_benefit": [-30.0, -30.0, -30.0, 4200.0],
    })

    # With K=0.75 (allow up to 3 items), but only item 3 has expected net benefit > 0
    decisions, cap = OperationalBudgetSimulator.compute_policy_decisions(
        policy=OperationalPolicyType.COST_SENSITIVE,
        probs=probs,
        costs_df=costs_df,
        values=values,
        budget_k=0.75,
        strictly_positive_benefit=True,
    )
    # Only item 3 should be flagged
    assert np.sum(decisions) == 1
    assert decisions[3] == 1
    assert decisions[0] == 0


# =============================================================================
# 2. Integration Tests: Cohort Evaluation & Metric Consistency
# =============================================================================

def test_evaluate_cohort_under_budget_metrics_consistency():
    simulator = OperationalBudgetSimulator(scenario_name="base")

    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])  # 3 positives in 10 items
    probs = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    values = np.array([10000.0, 50000.0, 2000.0, 3000.0, 80000.0, 1000.0, 500.0, 200.0, 100.0, 50.0])
    delay_days = np.array([10.0, 15.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    costs_df = pd.DataFrame({
        "fn_cost": [2000.0, 5000.0, 1000.0, 1000.0, 8000.0, 800.0, 800.0, 800.0, 800.0, 800.0],
        "fp_cost": [100.0, 200.0, 50.0, 50.0, 300.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        "intervention_cost": [300.0, 600.0, 100.0, 100.0, 1000.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "residual_delay_cost": [200.0, 400.0, 100.0, 100.0, 600.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "net_benefit": [1500.0, 4000.0, 800.0, 800.0, 6400.0, 600.0, 600.0, 600.0, 600.0, 600.0],
    })

    # Evaluate K=0.20 (Capacity = 2 items)
    m = simulator.evaluate_cohort_under_budget(
        y_true=y_true,
        probs=probs,
        costs_df=costs_df,
        values=values,
        delay_days=delay_days,
        budget_k=0.20,
        policy=OperationalPolicyType.COST_SENSITIVE,
        days_saved_efficacy=5.0,
    )

    assert m.budget_k == 0.20
    assert m.cohort_size == 10
    assert m.budget_capacity_count == 2
    assert m.reviewed_count == 2
    assert m.budget_utilization_pct == 100.0
    assert m.review_coverage_pct == 20.0
    assert m.positives_in_cohort == 3
    assert m.positives_captured >= 1
    assert m.realized_business_cost < m.do_nothing_cost
    assert m.net_savings_vs_do_nothing > 0.0
    assert m.cost_reduction_pct > 0.0
    assert m.delay_days_captured > 0.0
    assert m.commodity_value_delayed_captured_usd > 0.0


def test_simulate_all_policies_pairwise_savings():
    simulator = OperationalBudgetSimulator(scenario_name="base")

    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    probs = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    values = np.array([10000.0, 50000.0, 2000.0, 3000.0, 80000.0, 1000.0, 500.0, 200.0, 100.0, 50.0])

    costs_df = pd.DataFrame({
        "fn_cost": [2000.0, 5000.0, 1000.0, 1000.0, 8000.0, 800.0, 800.0, 800.0, 800.0, 800.0],
        "fp_cost": [100.0, 200.0, 50.0, 50.0, 300.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        "intervention_cost": [300.0, 600.0, 100.0, 100.0, 1000.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "residual_delay_cost": [200.0, 400.0, 100.0, 100.0, 600.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "net_benefit": [1500.0, 4000.0, 800.0, 800.0, 6400.0, 600.0, 600.0, 600.0, 600.0, 600.0],
    })

    res = simulator.simulate_all_policies_for_budget(
        y_true=y_true,
        probs=probs,
        costs_df=costs_df,
        values=values,
        budget_k=0.20,
    )

    assert "VALUE_ONLY" in res
    assert "RISK_ONLY" in res
    assert "STANDARD" in res
    assert "COST_SENSITIVE" in res

    cs_m = res["COST_SENSITIVE"]
    assert cs_m.net_savings_vs_value_only is not None
    assert cs_m.net_savings_vs_risk_only is not None
    assert cs_m.net_savings_vs_standard is not None


# =============================================================================
# 3. End-to-End Simulation over Real Backtest Parquet
# =============================================================================

def test_run_e8_dev_budget_simulation_e2e(tmp_path):
    parquet_path = Path("artifacts/results/e8_dev_backtest_results.parquet")
    if not parquet_path.exists():
        pytest.skip("Development backtest results parquet file not found")

    out_json = tmp_path / "test_budget_results.json"
    results = run_e8_dev_budget_simulation(
        backtest_parquet_path=parquet_path,
        output_json_path=out_json,
    )

    assert out_json.exists()
    assert "metadata" in results
    assert "aggregated_summary" in results
    assert "detailed_fold_records" in results

    agg = results["aggregated_summary"]
    assert "base" in agg
    assert "E8-C_tuned_gamma" in agg["base"]
    assert "k_05pct" in agg["base"]["E8-C_tuned_gamma"]
    assert "k_10pct" in agg["base"]["E8-C_tuned_gamma"]
    assert "k_20pct" in agg["base"]["E8-C_tuned_gamma"]

    # Verify COST_SENSITIVE policy under 10% budget achieves higher net savings than baseline policies
    k10_cs = agg["base"]["E8-C_tuned_gamma"]["k_10pct"]["COST_SENSITIVE"]
    k10_val = agg["base"]["E8-C_tuned_gamma"]["k_10pct"]["VALUE_ONLY"]
    assert k10_cs["total_net_savings"] > k10_val["total_net_savings"]
