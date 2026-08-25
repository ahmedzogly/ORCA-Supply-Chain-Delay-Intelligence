"""
Unit tests for ObservableShipmentState and DeterministicTransitionEngine.
"""

from __future__ import annotations

import dataclasses
import pandas as pd
import pytest

from delay_intelligence.counterfactual.provenance import ProvenanceTag, ProvenanceValidationError
from delay_intelligence.counterfactual.state import (
    CounterfactualTransitionResult,
    ObservableShipmentState,
)
from delay_intelligence.counterfactual.transitions import (
    DeterministicTransitionEngine,
    apply_counterfactual_transition,
    normalize_action_name,
)


@pytest.fixture
def base_state() -> ObservableShipmentState:
    return ObservableShipmentState(
        shipment_id="TEST_001",
        pred_date=pd.Timestamp("2012-05-15"),
        line_item_value=50000.0,
        clinical_criticality=1.30,
        transport_mode_factor=1.00,
        fulfillment_channel="Direct Drop",
        delay_prob=0.35,
        expected_delay_days=12.0,
        uncertainty_width=10.0,
        iot_temperature_c=4.5,
        iot_route_deviation_km=10.0,
        provenance_tag=ProvenanceTag.SYNTHETIC_E9_STATE.value,
    )


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


def test_observable_state_immutability(base_state: ObservableShipmentState):
    """Verifies that ObservableShipmentState is strictly frozen/immutable."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        base_state.line_item_value = 99999.0  # type: ignore

    with pytest.raises(dataclasses.FrozenInstanceError):
        base_state.delay_prob = 0.99  # type: ignore


def test_observable_state_validation_bounds():
    """Verifies bounds checks and error handling on state construction."""
    with pytest.raises(ValueError, match="line_item_value"):
        ObservableShipmentState(
            shipment_id="ERR",
            pred_date=pd.Timestamp("2012-01-01"),
            line_item_value=-100.0,
            clinical_criticality=1.0,
            transport_mode_factor=1.0,
            fulfillment_channel="Direct Drop",
            delay_prob=0.5,
            expected_delay_days=10.0,
            uncertainty_width=10.0,
        )

    with pytest.raises(ValueError, match="delay_prob"):
        ObservableShipmentState(
            shipment_id="ERR",
            pred_date=pd.Timestamp("2012-01-01"),
            line_item_value=1000.0,
            clinical_criticality=1.0,
            transport_mode_factor=1.0,
            fulfillment_channel="Direct Drop",
            delay_prob=1.5,
            expected_delay_days=10.0,
            uncertainty_width=10.0,
        )

    with pytest.raises(ProvenanceValidationError):
        ObservableShipmentState(
            shipment_id="ERR",
            pred_date=pd.Timestamp("2012-01-01"),
            line_item_value=1000.0,
            clinical_criticality=1.0,
            transport_mode_factor=1.0,
            fulfillment_channel="Direct Drop",
            delay_prob=0.5,
            expected_delay_days=10.0,
            uncertainty_width=10.0,
            provenance_tag="INVALID_UNVERIFIED_TAG",
        )


def test_transition_no_action(base_state: ObservableShipmentState, base_cost_params: dict):
    """Verifies NO_ACTION transition leaves state intact with zero action cost."""
    engine = DeterministicTransitionEngine(cost_params=base_cost_params)
    res = engine.transition(base_state, "NO_ACTION")

    assert res.action == "NO_ACTION"
    assert res.action_cost == 0.0
    assert res.residual_delay_days == base_state.expected_delay_days
    assert res.residual_delay_prob == base_state.delay_prob
    assert res.expected_realized_cost > 0.0
    assert res.provenance_tag == ProvenanceTag.SIMULATED_COUNTERFACTUAL.value


def test_transition_expedite_frozen_effects(base_state: ObservableShipmentState, base_cost_params: dict):
    """Verifies EXPEDITE applies Delta D = -3.0 days and computes linear expedite fee."""
    engine = DeterministicTransitionEngine(cost_params=base_cost_params)
    res = engine.transition(base_state, "EXPEDITE")

    assert res.action == "EXPEDITE"
    assert res.residual_delay_days == base_state.expected_delay_days - 3.0
    assert res.residual_delay_prob == base_state.delay_prob

    # Expected expedite cost: c_expedite_base (500) + gamma (0.005) * 50000 = 500 + 250 = 750
    assert res.action_cost == pytest.approx(750.0)
    assert res.expected_realized_cost == pytest.approx(res.action_cost + res.residual_delay_cost + res.residual_risk_cost)


def test_transition_transport_mode_review_effects(base_state: ObservableShipmentState, base_cost_params: dict):
    """Verifies TRANSPORT_MODE_REVIEW applies Delta D = -2.0 days and log-scaled review fee."""
    engine = DeterministicTransitionEngine(cost_params=base_cost_params)
    res = engine.transition(base_state, "TRANSPORT_MODE_REVIEW")

    assert res.action == "TRANSPORT_MODE_REVIEW"
    assert res.residual_delay_days == base_state.expected_delay_days - 2.0
    assert res.residual_delay_prob == base_state.delay_prob
    assert res.action_cost > 200.0


def test_transition_supplier_escalation_effects(base_state: ObservableShipmentState, base_cost_params: dict):
    """Verifies SUPPLIER_ESCALATION applies Delta R = -15% relative delay risk."""
    engine = DeterministicTransitionEngine(cost_params=base_cost_params)
    res = engine.transition(base_state, "SUPPLIER_ESCALATION")

    assert res.action == "SUPPLIER_ESCALATION"
    assert res.residual_delay_days == base_state.expected_delay_days
    assert res.residual_delay_prob == pytest.approx(base_state.delay_prob * 0.85)
    # Direct drop inquiry: 150 + 30 = 180
    assert res.action_cost == pytest.approx(180.0)


def test_transition_human_review_effects(base_state: ObservableShipmentState, base_cost_params: dict):
    """Verifies HUMAN_REVIEW applies Delta W = -50% uncertainty reduction."""
    engine = DeterministicTransitionEngine(cost_params=base_cost_params)
    res = engine.transition(base_state, "HUMAN_REVIEW")

    assert res.action == "HUMAN_REVIEW"
    assert res.residual_uncertainty_width == pytest.approx(base_state.uncertainty_width * 0.50)
    assert res.residual_delay_days == base_state.expected_delay_days


def test_action_normalization():
    """Verifies robust normalization of policy prefixes and action codes."""
    assert normalize_action_name("P0") == "NO_ACTION"
    assert normalize_action_name("P1_E8_COST_SENSITIVE") == "E8_COST_SENSITIVE"
    assert normalize_action_name("P2_EXPEDITE") == "EXPEDITE"
    assert normalize_action_name("P3") == "TRANSPORT_MODE_REVIEW"
    assert normalize_action_name("P4_SUPPLIER_ESCALATION") == "SUPPLIER_ESCALATION"
    assert normalize_action_name("P5_HUMAN_REVIEW") == "HUMAN_REVIEW"
