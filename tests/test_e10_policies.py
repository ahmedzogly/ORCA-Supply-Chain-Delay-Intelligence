"""
Unit tests for Operational Policy Suite (P0..P5).
"""

from __future__ import annotations

import pandas as pd
import pytest

from delay_intelligence.counterfactual.policies import (
    PolicyP0_NoAction,
    PolicyP1_E8CostSensitive,
    PolicyP2_Expedite,
    PolicyP3_TransportModeReview,
    PolicyP4_SupplierEscalation,
    PolicyP5_HumanReview,
    get_policy,
    list_standard_policies,
)
from delay_intelligence.counterfactual.state import ObservableShipmentState


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


def test_policy_registry_coverage():
    """Verifies all 6 policies are registered and accessible."""
    policies = list_standard_policies()
    assert len(policies) == 6
    assert set(policies.keys()) == {"P0", "P1", "P2", "P3", "P4", "P5"}

    for p in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        pol = get_policy(p)
        assert pol.policy_id == p


def test_p0_no_action_always_returns_no_action(base_cost_params: dict):
    """P0 must strictly return NO_ACTION regardless of state risk."""
    state_high_risk = ObservableShipmentState(
        shipment_id="HIGH_RISK",
        pred_date=pd.Timestamp("2013-01-01"),
        line_item_value=1000000.0,
        clinical_criticality=2.0,
        transport_mode_factor=1.25,
        fulfillment_channel="Direct Drop",
        delay_prob=0.95,
        expected_delay_days=25.0,
        uncertainty_width=20.0,
    )
    pol = PolicyP0_NoAction()
    assert pol.select_action(state_high_risk, base_cost_params) == "NO_ACTION"


def test_p1_e8_cost_sensitive_threshold_logic(base_cost_params: dict):
    """Verifies P1 threshold computation with gamma*=1.20."""
    pol = PolicyP1_E8CostSensitive(gamma_multiplier=1.20)

    # State with low risk below threshold (delay_prob=0.05 < tau_star) -> NO_ACTION
    state_low_risk = ObservableShipmentState(
        shipment_id="LOW_RISK",
        pred_date=pd.Timestamp("2013-01-01"),
        line_item_value=100.0,
        clinical_criticality=1.0,
        transport_mode_factor=1.0,
        fulfillment_channel="From RDC",
        delay_prob=0.05,
        expected_delay_days=5.0,
        uncertainty_width=8.0,
    )
    tau_low = pol.compute_threshold(state_low_risk, base_cost_params)
    assert 0.05 < tau_low
    assert pol.select_action(state_low_risk, base_cost_params) == "NO_ACTION"

    # High value item with high delay risk -> threshold should trigger EXPEDITE
    state_high_val = ObservableShipmentState(
        shipment_id="HIGH_VAL",
        pred_date=pd.Timestamp("2013-01-01"),
        line_item_value=500000.0,
        clinical_criticality=1.5,
        transport_mode_factor=1.0,
        fulfillment_channel="Direct Drop",
        delay_prob=0.60,
        expected_delay_days=15.0,
        uncertainty_width=10.0,
    )
    tau = pol.compute_threshold(state_high_val, base_cost_params)
    assert 0.0 < tau < 0.60
    assert pol.select_action(state_high_val, base_cost_params) == "EXPEDITE"


def test_p5_human_review_triggers(base_cost_params: dict):
    """Verifies P5 triggers on conformal uncertainty W_i > 14.0 or IoT telemetry alarms."""
    pol = PolicyP5_HumanReview()

    # Normal state -> NO_ACTION
    state_normal = ObservableShipmentState(
        shipment_id="NORM",
        pred_date=pd.Timestamp("2013-01-01"),
        line_item_value=20000.0,
        clinical_criticality=1.0,
        transport_mode_factor=1.0,
        fulfillment_channel="Direct Drop",
        delay_prob=0.10,
        expected_delay_days=5.0,
        uncertainty_width=10.0,
    )
    assert pol.select_action(state_normal, base_cost_params) == "NO_ACTION"

    # Wide CQR interval -> HUMAN_REVIEW
    state_high_uncert = ObservableShipmentState(
        shipment_id="UNCERT",
        pred_date=pd.Timestamp("2013-01-01"),
        line_item_value=20000.0,
        clinical_criticality=1.0,
        transport_mode_factor=1.0,
        fulfillment_channel="Direct Drop",
        delay_prob=0.10,
        expected_delay_days=5.0,
        uncertainty_width=16.5,
    )
    assert pol.select_action(state_high_uncert, base_cost_params) == "HUMAN_REVIEW"

    # Temperature excursion -> HUMAN_REVIEW
    state_temp_alert = ObservableShipmentState(
        shipment_id="TEMP",
        pred_date=pd.Timestamp("2013-01-01"),
        line_item_value=20000.0,
        clinical_criticality=1.0,
        transport_mode_factor=1.0,
        fulfillment_channel="Direct Drop",
        delay_prob=0.10,
        expected_delay_days=5.0,
        uncertainty_width=10.0,
        iot_temperature_c=12.0,  # Excursion > 8.0 C
    )
    assert pol.select_action(state_temp_alert, base_cost_params) == "HUMAN_REVIEW"
