"""
Unit tests for ReviewBudgetAllocator and operational capacity constraints.
"""

from __future__ import annotations

import pandas as pd
import pytest

from delay_intelligence.counterfactual.budget import ReviewBudgetAllocator
from delay_intelligence.counterfactual.state import ObservableShipmentState


@pytest.fixture
def sample_states() -> list[ObservableShipmentState]:
    # 20 synthetic shipments with varied risk and value profiles
    states = []
    for i in range(20):
        val = 1000.0 * (i + 1) * 10
        prob = min(0.95, 0.05 + 0.04 * i)
        crit = 1.0 + (0.3 if i % 2 == 0 else 0.0)
        st = ObservableShipmentState(
            shipment_id=f"SHIP_{i:03d}",
            pred_date=pd.Timestamp("2013-05-01"),
            line_item_value=val,
            clinical_criticality=crit,
            transport_mode_factor=1.0,
            fulfillment_channel="Direct Drop" if i % 2 == 0 else "From RDC",
            delay_prob=prob,
            expected_delay_days=15.0,
            uncertainty_width=10.0,
        )
        states.append(st)
    return states


@pytest.fixture
def base_cost_params() -> dict:
    return {
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


def test_budget_capacity_bounds(sample_states: list[ObservableShipmentState], base_cost_params: dict):
    """Verifies that allocated interventions never exceed floor(K * N)."""
    allocator = ReviewBudgetAllocator(cost_params=base_cost_params)

    for k in [0.05, 0.10, 0.20]:
        res = allocator.allocate_budget(sample_states, capacity_k=k)
        max_allowed = int(len(sample_states) * k)
        assert res["allocated_count"] <= max_allowed
        assert len(res["records"]) == len(sample_states)


def test_budget_conservation_and_positive_benefit(sample_states: list[ObservableShipmentState], base_cost_params: dict):
    """Verifies budget conservation and that only positive net benefit actions are allocated."""
    allocator = ReviewBudgetAllocator(cost_params=base_cost_params)
    res = allocator.allocate_budget(sample_states, capacity_k=0.20)

    # Check sum of records matches total
    sum_realized = sum(r["realized_cost"] for r in res["records"])
    sum_no_action = sum(r["no_action_cost"] for r in res["records"])

    assert res["total_realized_cost"] == pytest.approx(sum_realized)
    assert res["total_no_action_cost"] == pytest.approx(sum_no_action)
    assert res["total_net_benefit"] == pytest.approx(sum_no_action - sum_realized)

    # Check every intervened item had positive benefit
    for r in res["records"]:
        if r["is_intervened"]:
            assert r["net_benefit"] > 0.0
