"""
Unit tests for 3x3 Multi-Dimensional Sensitivity Grid Evaluator.
"""

from __future__ import annotations

import pandas as pd
import pytest

from delay_intelligence.counterfactual.sensitivity import (
    SENSITIVITY_GRID_CELLS,
    SensitivityGridEvaluator,
)
from delay_intelligence.counterfactual.state import ObservableShipmentState


@pytest.fixture
def sample_states() -> list[ObservableShipmentState]:
    return [
        ObservableShipmentState(
            shipment_id=f"S_{i}",
            pred_date=pd.Timestamp("2013-01-01"),
            line_item_value=val,
            clinical_criticality=crit,
            transport_mode_factor=1.0,
            fulfillment_channel="Direct Drop" if i % 2 == 0 else "From RDC",
            delay_prob=prob,
            expected_delay_days=12.0,
            uncertainty_width=10.0,
        )
        for i, (val, crit, prob) in enumerate([
            (5000.0, 1.0, 0.10),
            (50000.0, 1.3, 0.40),
            (250000.0, 1.6, 0.75),
            (800000.0, 2.0, 0.90),
        ])
    ]


@pytest.fixture
def base_cost_scenarios() -> dict:
    return {
        "base": {
            "c_daily_base": 150.0,
            "rho_value": 0.0010,
            "c_fixed_stockout": 500.0,
            "c_triage_base": 50.0,
            "beta_audit": 10.0,
            "c_direct_inquiry": 30.0,
            "c_rdc_inquiry": 10.0,
            "c_expedite_base": 500.0,
            "gamma_expedite": 0.0050,
            "c_mode_review_base": 200.0,
            "beta_mode": 20.0,
            "c_escalation_base": 150.0,
            "delay_days_assumed": 12.0,
            "days_saved_efficacy": 5.0,
        }
    }


def test_sensitivity_grid_dimensions(sample_states: list[ObservableShipmentState], base_cost_scenarios: dict):
    """Verifies that the sensitivity grid evaluates 9 cells across 7 policies (63 rows per scenario)."""
    evaluator = SensitivityGridEvaluator(cost_scenarios=base_cost_scenarios)
    df_grid = evaluator.evaluate_grid(sample_states, scenario_name="base")

    assert len(df_grid) == 9 * 7  # 9 cells * 7 policies (P0-P5 + Oracle)
    assert df_grid["grid_cell"].nunique() == 9
    assert set(df_grid["policy_id"].unique()) == {"P0", "P1", "P2", "P3", "P4", "P5", "Oracle"}


def test_sensitivity_monotonicity(sample_states: list[ObservableShipmentState], base_cost_scenarios: dict):
    """
    Verifies monotonicity across cost and efficacy scaling:
    - Higher efficacy with same cost multiplier yields higher or equal net benefit for Oracle.
    """
    evaluator = SensitivityGridEvaluator(cost_scenarios=base_cost_scenarios)
    df_grid = evaluator.evaluate_grid(sample_states, scenario_name="base")

    oracle_low_eff = df_grid[(df_grid["policy_id"] == "Oracle") & (df_grid["grid_cell"] == "Cost_Base__Eff_Low")]["net_benefit_vs_p0"].iloc[0]
    oracle_high_eff = df_grid[(df_grid["policy_id"] == "Oracle") & (df_grid["grid_cell"] == "Cost_Base__Eff_High")]["net_benefit_vs_p0"].iloc[0]

    assert oracle_high_eff >= oracle_low_eff - 1e-6
