"""
Adversarial QA Challenger Test Suite for Experiment E10.
Focus Areas:
1. Attack Action Effect Alteration (Immutability against docs/e9_simulation_assumptions.json).
2. Attack Oracle Isolation & AST Integrity (Zero Oracle leakage in online/policy code).
3. Attack Budget Constraints & Capacity Strictness (M <= floor(K*N) under all regimes).
4. Attack Extreme Values & Numerical Stability (Zero value, huge value, p=0/1, W=100, empty queues, zero NaN/Inf).
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest

from delay_intelligence.counterfactual.budget import ReviewBudgetAllocator
from delay_intelligence.counterfactual.oracle import (
    ORACLE_ACTION_SPACE,
    OfflineOraclePolicy,
)
from delay_intelligence.counterfactual.policies import (
    BasePolicy,
    PolicyP0_NoAction,
    PolicyP1_E8CostSensitive,
    PolicyP2_Expedite,
    PolicyP3_TransportModeReview,
    PolicyP4_SupplierEscalation,
    PolicyP5_HumanReview,
    list_standard_policies,
)
from delay_intelligence.counterfactual.provenance import (
    NON_CAUSAL_DISCLAIMER,
    ProvenanceTag,
    ProvenanceValidationError,
    attach_provenance_metadata,
    validate_provenance_tag,
)
from delay_intelligence.counterfactual.state import (
    CounterfactualTransitionResult,
    ObservableShipmentState,
)
from delay_intelligence.counterfactual.transitions import (
    DeterministicTransitionEngine,
    normalize_action_name,
)


@pytest.fixture
def base_cost_params() -> Dict[str, Any]:
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
        "delta_first_line": 0.30,
        "delta_pediatric": 0.20,
        "delta_arv": 0.15,
        "mode_multipliers": {
            "Air": 1.00,
            "Air Charter": 0.90,
            "Truck": 1.10,
            "Ocean": 1.25,
            "Default": 1.00,
        },
    }


# =============================================================================
# 1. ATTACK ACTION EFFECT ALTERATION & IMMUTABILITY
# =============================================================================

def test_action_effects_exact_match_e9_simulation_assumptions(base_cost_params):
    """
    Verifies that action effects match docs/e9_simulation_assumptions.json bit-for-bit:
    - EXPEDITE: -3.0 days
    - TRANSPORT_MODE_REVIEW: -2.0 days
    - SUPPLIER_ESCALATION: -0.15 (-15% risk)
    """
    assumptions_path = Path("docs/e9_simulation_assumptions.json")
    assert assumptions_path.exists(), "docs/e9_simulation_assumptions.json is missing!"

    with open(assumptions_path, "r", encoding="utf-8") as f:
        e9_assumptions = json.load(f)

    assumptions_map = {item["Action"]: item["Frozen_Value"] for item in e9_assumptions}

    engine = DeterministicTransitionEngine(cost_params=base_cost_params)

    test_state = ObservableShipmentState(
        shipment_id="TEST_STATE_IMMUTABILITY",
        pred_date=pd.Timestamp("2013-05-10"),
        line_item_value=50000.0,
        clinical_criticality=1.35,
        transport_mode_factor=1.0,
        fulfillment_channel="Direct Drop",
        delay_prob=0.60,
        expected_delay_days=10.0,
        uncertainty_width=8.0,
    )

    # 1. Test EXPEDITE
    res_exp = engine.transition(test_state, "EXPEDITE")
    expected_exp_reduction = float(assumptions_map["EXPEDITE_SIMULATED"])
    assert res_exp.residual_delay_days == pytest.approx(test_state.expected_delay_days - expected_exp_reduction)

    # 2. Test TRANSPORT_MODE_REVIEW
    res_mode = engine.transition(test_state, "TRANSPORT_MODE_REVIEW")
    expected_mode_adjustment = float(assumptions_map["TRANSPORT_MODE_REVIEW"])  # -2.0
    assert res_mode.residual_delay_days == pytest.approx(test_state.expected_delay_days + expected_mode_adjustment)

    # 3. Test SUPPLIER_ESCALATION
    res_esc = engine.transition(test_state, "SUPPLIER_ESCALATION")
    expected_risk_pct = float(assumptions_map["SUPPLIER_ESCALATION_SIMULATED"])  # 0.15
    assert res_esc.residual_delay_prob == pytest.approx(test_state.delay_prob * (1.0 - expected_risk_pct))

    # 4. Test HUMAN_REVIEW (E10 extension: -50% uncertainty width)
    res_hum = engine.transition(test_state, "HUMAN_REVIEW")
    assert res_hum.residual_uncertainty_width == pytest.approx(test_state.uncertainty_width * 0.50)

    # 5. Test NO_ACTION (0.0 deltas)
    res_no = engine.transition(test_state, "NO_ACTION")
    assert res_no.residual_delay_days == pytest.approx(test_state.expected_delay_days)
    assert res_no.residual_delay_prob == pytest.approx(test_state.delay_prob)
    assert res_no.residual_uncertainty_width == pytest.approx(test_state.uncertainty_width)
    assert res_no.action_cost == 0.0


def test_transition_engine_determinism_under_repetition(base_cost_params):
    """
    Verifies that transition results are 100% deterministic and invariant across repeated runs.
    """
    engine = DeterministicTransitionEngine(cost_params=base_cost_params)
    state = ObservableShipmentState(
        shipment_id="DETERMINISM_TEST",
        pred_date=pd.Timestamp("2012-08-15"),
        line_item_value=75000.0,
        clinical_criticality=1.45,
        transport_mode_factor=1.10,
        fulfillment_channel="Direct Drop",
        delay_prob=0.48,
        expected_delay_days=14.0,
        uncertainty_width=11.5,
    )

    baseline_res = engine.transition(state, "EXPEDITE")
    for _ in range(500):
        rep_res = engine.transition(state, "EXPEDITE")
        assert rep_res.action_cost == baseline_res.action_cost
        assert rep_res.residual_delay_days == baseline_res.residual_delay_days
        assert rep_res.residual_delay_prob == baseline_res.residual_delay_prob
        assert rep_res.expected_realized_cost == baseline_res.expected_realized_cost


# =============================================================================
# 2. ATTACK ORACLE ISOLATION & AST INTEGRITY
# =============================================================================

def test_ast_deep_oracle_isolation():
    """
    Exhaustive AST inspection across all source files in src/delay_intelligence/
    to verify that OfflineOraclePolicy and oracle.py are NEVER imported or referenced
    by operational policies, decision engines, API endpoints, or model training.
    """
    src_root = Path("src/delay_intelligence")
    assert src_root.exists(), "src/delay_intelligence directory not found"

    # Modules that MUST NEVER contain references to oracle
    prohibited_subtrees = [
        src_root / "decision",
        src_root / "api",
        src_root / "models",
        src_root / "cost_sensitive",
        src_root / "adaptive_conformal",
        src_root / "drift",
        src_root / "causal",
        src_root / "features",
        src_root / "validation",
        src_root / "data",
    ]

    prohibited_files = [
        src_root / "counterfactual" / "policies.py",
        src_root / "counterfactual" / "transitions.py",
        src_root / "counterfactual" / "state.py",
        src_root / "counterfactual" / "budget.py",
    ]

    all_scanned_files: List[Path] = []

    for subtree in prohibited_subtrees:
        if subtree.exists():
            for py_file in subtree.rglob("*.py"):
                all_scanned_files.append(py_file)

    for py_file in prohibited_files:
        if py_file.exists():
            all_scanned_files.append(py_file)

    assert len(all_scanned_files) > 0, "No files found for AST oracle isolation scan!"

    for fpath in all_scanned_files:
        with open(fpath, "r", encoding="utf-8-sig") as f:
            code = f.read()

        tree = ast.parse(code, filename=str(fpath))

        for node in ast.walk(tree):
            # Check import statements
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "oracle" not in alias.name.lower(), (
                        f"CRITICAL ISOLATION VIOLATION: '{alias.name}' imported in {fpath}"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "oracle" not in mod.lower(), (
                    f"CRITICAL ISOLATION VIOLATION: 'from {mod}' imported in {fpath}"
                )
                for alias in node.names:
                    assert "offlineoraclepolicy" not in alias.name.lower(), (
                        f"CRITICAL ISOLATION VIOLATION: '{alias.name}' imported in {fpath}"
                    )
            # Check name references
            elif isinstance(node, ast.Name):
                assert node.id != "OfflineOraclePolicy", (
                    f"CRITICAL ISOLATION VIOLATION: Direct reference to OfflineOraclePolicy in {fpath}"
                )


def test_policies_independence_from_oracle():
    """
    Verifies that policies do not access any hidden Oracle attributes or oracle singletons.
    """
    policies = list_standard_policies()
    dummy_cost_params = {
        "c_daily_base": 150.0,
        "rho_value": 0.001,
        "c_fixed_stockout": 500.0,
        "c_triage_base": 50.0,
        "beta_audit": 10.0,
        "c_direct_inquiry": 30.0,
        "c_rdc_inquiry": 10.0,
        "c_expedite_base": 500.0,
        "gamma_expedite": 0.005,
        "c_mode_review_base": 200.0,
        "beta_mode": 20.0,
        "c_escalation_base": 150.0,
        "delay_days_assumed": 12.0,
        "days_saved_efficacy": 5.0,
    }

    state = ObservableShipmentState(
        shipment_id="ORACLE_INDEPENDENCE_TEST",
        pred_date=pd.Timestamp("2013-01-01"),
        line_item_value=10000.0,
        clinical_criticality=1.0,
        transport_mode_factor=1.0,
        fulfillment_channel="Direct Drop",
        delay_prob=0.3,
        expected_delay_days=10.0,
        uncertainty_width=5.0,
    )

    for pol_id, pol in policies.items():
        action = pol.select_action(state, dummy_cost_params)
        assert isinstance(action, str)
        assert action in ["NO_ACTION", "EXPEDITE", "TRANSPORT_MODE_REVIEW", "SUPPLIER_ESCALATION", "HUMAN_REVIEW"]
        # Verify policy object does not hold oracle instances
        for attr_name, attr_val in pol.__dict__.items():
            assert "oracle" not in attr_name.lower()
            assert not isinstance(attr_val, OfflineOraclePolicy)


# =============================================================================
# 3. ATTACK BUDGET CONSTRAINTS & ALLOCATION RIGIDITY
# =============================================================================

def test_budget_allocator_strict_capacity_bounds(base_cost_params):
    """
    Adversarial test: Tests that allocated_count <= floor(K * N) under ALL population sizes N,
    capacities K, and high-incentive environments where EVERY shipment has massive positive net benefit.
    """
    allocator = ReviewBudgetAllocator(cost_params=base_cost_params)

    capacities = [0.01, 0.05, 0.10, 0.15, 0.20, 0.33, 0.50, 0.99, 1.00]
    sample_sizes = [0, 1, 2, 3, 5, 10, 27, 100, 500, 1000]

    for n in sample_sizes:
        # Create states with large positive net benefit for expedite
        states = [
            ObservableShipmentState(
                shipment_id=f"HIGH_VAL_{i}",
                pred_date=pd.Timestamp("2012-01-01") + pd.Timedelta(days=i % 300),
                line_item_value=500000.0 + i * 1000.0,
                clinical_criticality=1.5,
                transport_mode_factor=1.0,
                fulfillment_channel="Direct Drop",
                delay_prob=0.95,
                expected_delay_days=20.0,
                uncertainty_width=10.0,
            )
            for i in range(n)
        ]

        for k in capacities:
            result = allocator.allocate_budget(states, capacity_k=k)
            max_allowed = int(math.floor(k * n))

            assert result["allocated_count"] <= max_allowed, (
                f"BUDGET OVERFLOW: N={n}, K={k}, allowed={max_allowed}, got={result['allocated_count']}"
            )
            assert result["total_shipments"] == n
            assert result["capacity_limit_count"] == max_allowed

            # Check that unallocated shipments receive strictly NO_ACTION
            for rec in result["records"]:
                if rec["rank"] > max_allowed:
                    assert rec["action_assigned"] == "NO_ACTION", (
                        f"Rank {rec['rank']} > max_allowed {max_allowed} was given action {rec['action_assigned']}"
                    )
                    assert rec["is_intervened"] is False


def test_budget_allocator_empty_and_single_shipment_queues(base_cost_params):
    """
    Edge case test: Empty queue (N=0) and single shipment queue (N=1) under various capacities.
    """
    allocator = ReviewBudgetAllocator(cost_params=base_cost_params)

    # Empty queue
    res_empty = allocator.allocate_budget([], capacity_k=0.10)
    assert res_empty["total_shipments"] == 0
    assert res_empty["allocated_count"] == 0
    assert res_empty["total_realized_cost"] == 0.0
    assert res_empty["total_net_benefit"] == 0.0
    assert len(res_empty["records"]) == 0

    # Single shipment with K=0.05 -> floor(0.05 * 1) = 0 -> 0 allocated
    single_state = [
        ObservableShipmentState(
            shipment_id="SINGLE_1",
            pred_date=pd.Timestamp("2013-01-01"),
            line_item_value=100000.0,
            clinical_criticality=1.2,
            transport_mode_factor=1.0,
            fulfillment_channel="Direct Drop",
            delay_prob=0.8,
            expected_delay_days=15.0,
            uncertainty_width=10.0,
        )
    ]
    res_single_5pct = allocator.allocate_budget(single_state, capacity_k=0.05)
    assert res_single_5pct["capacity_limit_count"] == 0
    assert res_single_5pct["allocated_count"] == 0
    assert res_single_5pct["records"][0]["action_assigned"] == "NO_ACTION"

    # Single shipment with K=1.0 -> floor(1.0 * 1) = 1 -> 1 allocated
    res_single_100pct = allocator.allocate_budget(single_state, capacity_k=1.0)
    assert res_single_100pct["capacity_limit_count"] == 1
    assert res_single_100pct["allocated_count"] == 1
    assert res_single_100pct["records"][0]["action_assigned"] != "NO_ACTION"


def test_budget_allocator_no_negative_benefit_allocation(base_cost_params):
    """
    Verifies that shipments with negative net benefit are NEVER allocated, even if budget capacity is 100%.
    """
    allocator = ReviewBudgetAllocator(cost_params=base_cost_params)

    # Low value, low delay prob shipments where all interventions cost more than delay savings
    states_worthless = [
        ObservableShipmentState(
            shipment_id=f"WORTHLESS_{i}",
            pred_date=pd.Timestamp("2013-01-01"),
            line_item_value=5.0,  # $5 shipment
            clinical_criticality=1.0,
            transport_mode_factor=1.0,
            fulfillment_channel="From RDC",
            delay_prob=0.01,  # 1% delay prob
            expected_delay_days=1.0,
            uncertainty_width=2.0,
        )
        for i in range(20)
    ]

    res = allocator.allocate_budget(states_worthless, capacity_k=1.0)
    assert res["allocated_count"] == 0, "Allocated interventions on shipments with negative net benefit!"
    assert res["total_net_benefit"] == 0.0


# =============================================================================
# 4. ATTACK EXTREME VALUES & NUMERICAL STABILITY (FUZZING)
# =============================================================================

def test_extreme_zero_line_item_value(base_cost_params):
    """
    Tests zero line-item value (V_i = $0.00).
    Verifies log(1 + 0) = 0 calculations, threshold calculations, and zero division protection.
    """
    state_zero_val = ObservableShipmentState(
        shipment_id="ZERO_VAL",
        pred_date=pd.Timestamp("2012-05-01"),
        line_item_value=0.0,
        clinical_criticality=1.0,
        transport_mode_factor=1.0,
        fulfillment_channel="Direct Drop",
        delay_prob=0.50,
        expected_delay_days=10.0,
        uncertainty_width=5.0,
    )

    engine = DeterministicTransitionEngine(cost_params=base_cost_params)
    oracle = OfflineOraclePolicy(cost_params=base_cost_params)
    policies = list_standard_policies()

    for act in ORACLE_ACTION_SPACE:
        res = engine.transition(state_zero_val, act)
        assert not math.isnan(res.expected_realized_cost)
        assert not math.isinf(res.expected_realized_cost)
        assert res.expected_realized_cost >= 0.0

    for pol_id, pol in policies.items():
        act = pol.select_action(state_zero_val, base_cost_params)
        assert isinstance(act, str)

    opt_act, opt_cost, _ = oracle.evaluate_optimal_action(state_zero_val)
    assert not math.isnan(opt_cost)
    assert not math.isinf(opt_cost)


def test_extreme_massive_line_item_value(base_cost_params):
    """
    Tests massive line-item value (V_i = $1,000,000,000.00 / 1B USD).
    Verifies numerical stability without overflow or NaN.
    """
    state_massive_val = ObservableShipmentState(
        shipment_id="MASSIVE_VAL",
        pred_date=pd.Timestamp("2012-05-01"),
        line_item_value=1e9,
        clinical_criticality=2.0,
        transport_mode_factor=1.25,
        fulfillment_channel="Direct Drop",
        delay_prob=0.99,
        expected_delay_days=30.0,
        uncertainty_width=25.0,
    )

    engine = DeterministicTransitionEngine(cost_params=base_cost_params)
    oracle = OfflineOraclePolicy(cost_params=base_cost_params)
    policies = list_standard_policies()

    for act in ORACLE_ACTION_SPACE:
        res = engine.transition(state_massive_val, act)
        assert not math.isnan(res.expected_realized_cost)
        assert not math.isinf(res.expected_realized_cost)
        assert res.expected_realized_cost > 0.0

    p1 = policies["P1"]
    assert isinstance(p1, PolicyP1_E8CostSensitive)
    tau = p1.compute_threshold(state_massive_val, base_cost_params)
    assert 0.0 <= tau <= 1.0


def test_boundary_probabilities_zero_and_one(base_cost_params):
    """
    Tests exact boundary probabilities p_i = 0.0 and p_i = 1.0.
    """
    engine = DeterministicTransitionEngine(cost_params=base_cost_params)
    policies = list_standard_policies()

    # Case 1: p_i = 0.0 (zero risk)
    state_p0 = ObservableShipmentState(
        shipment_id="P_ZERO",
        pred_date=pd.Timestamp("2011-01-01"),
        line_item_value=50000.0,
        clinical_criticality=1.0,
        transport_mode_factor=1.0,
        fulfillment_channel="Direct Drop",
        delay_prob=0.0,
        expected_delay_days=10.0,
        uncertainty_width=5.0,
    )

    res_p0 = engine.transition(state_p0, "NO_ACTION")
    assert res_p0.residual_delay_cost == 0.0
    assert res_p0.residual_risk_cost == 0.0
    assert res_p0.expected_realized_cost == 0.0
    assert policies["P1"].select_action(state_p0, base_cost_params) == "NO_ACTION"
    assert policies["P2"].select_action(state_p0, base_cost_params) == "NO_ACTION"

    # Case 2: p_i = 1.0 (certain risk)
    state_p1 = ObservableShipmentState(
        shipment_id="P_ONE",
        pred_date=pd.Timestamp("2011-01-01"),
        line_item_value=50000.0,
        clinical_criticality=1.5,
        transport_mode_factor=1.0,
        fulfillment_channel="Direct Drop",
        delay_prob=1.0,
        expected_delay_days=15.0,
        uncertainty_width=5.0,
    )

    res_p1_esc = engine.transition(state_p1, "SUPPLIER_ESCALATION")
    assert res_p1_esc.residual_delay_prob == pytest.approx(0.85)  # 1.0 * (1 - 0.15)


def test_extreme_uncertainty_widths(base_cost_params):
    """
    Tests extreme uncertainty widths W_i = 100.0d and minimal W_i = 0.1d.
    """
    engine = DeterministicTransitionEngine(cost_params=base_cost_params)
    p5 = PolicyP5_HumanReview()

    # W_i = 100.0 days
    state_w_high = ObservableShipmentState(
        shipment_id="W_HIGH",
        pred_date=pd.Timestamp("2013-01-01"),
        line_item_value=20000.0,
        clinical_criticality=1.0,
        transport_mode_factor=1.0,
        fulfillment_channel="Direct Drop",
        delay_prob=0.20,
        expected_delay_days=5.0,
        uncertainty_width=100.0,
    )
    assert p5.select_action(state_w_high, base_cost_params) == "HUMAN_REVIEW"
    res_w_high = engine.transition(state_w_high, "HUMAN_REVIEW")
    assert res_w_high.residual_uncertainty_width == pytest.approx(50.0)

    # W_i = 0.1 days (floor check)
    state_w_low = ObservableShipmentState(
        shipment_id="W_LOW",
        pred_date=pd.Timestamp("2013-01-01"),
        line_item_value=20000.0,
        clinical_criticality=1.0,
        transport_mode_factor=1.0,
        fulfillment_channel="Direct Drop",
        delay_prob=0.20,
        expected_delay_days=5.0,
        uncertainty_width=0.1,
    )
    res_w_low = engine.transition(state_w_low, "HUMAN_REVIEW")
    assert res_w_low.residual_uncertainty_width >= 0.1


def test_telemetry_extreme_anomalies(base_cost_params):
    """
    Tests IoT telemetry extremes:
    - Temperature out of range (<2C or >8C) triggers P5 HUMAN_REVIEW.
    - Route deviation > 50km triggers P5 HUMAN_REVIEW.
    """
    p5 = PolicyP5_HumanReview()

    # Temperature freezing anomaly (-15C)
    st_freeze = ObservableShipmentState(
        shipment_id="TEL_FREEZE",
        pred_date=pd.Timestamp("2013-01-01"),
        line_item_value=10000.0,
        clinical_criticality=1.0,
        transport_mode_factor=1.0,
        fulfillment_channel="Direct Drop",
        delay_prob=0.10,
        expected_delay_days=2.0,
        uncertainty_width=5.0,
        iot_temperature_c=-15.0,
    )
    assert p5.select_action(st_freeze, base_cost_params) == "HUMAN_REVIEW"

    # Temperature heat anomaly (+45C)
    st_heat = ObservableShipmentState(
        shipment_id="TEL_HEAT",
        pred_date=pd.Timestamp("2013-01-01"),
        line_item_value=10000.0,
        clinical_criticality=1.0,
        transport_mode_factor=1.0,
        fulfillment_channel="Direct Drop",
        delay_prob=0.10,
        expected_delay_days=2.0,
        uncertainty_width=5.0,
        iot_temperature_c=45.0,
    )
    assert p5.select_action(st_heat, base_cost_params) == "HUMAN_REVIEW"

    # Route deviation anomaly (250 km)
    st_route = ObservableShipmentState(
        shipment_id="TEL_ROUTE",
        pred_date=pd.Timestamp("2013-01-01"),
        line_item_value=10000.0,
        clinical_criticality=1.0,
        transport_mode_factor=1.0,
        fulfillment_channel="Direct Drop",
        delay_prob=0.10,
        expected_delay_days=2.0,
        uncertainty_width=5.0,
        iot_route_deviation_km=250.0,
    )
    assert p5.select_action(st_route, base_cost_params) == "HUMAN_REVIEW"


def test_adversarial_fuzzing_1000_randomized_states(base_cost_params):
    """
    Stress test with 1,000 randomized state vectors spanning all extreme parameters.
    Verifies that no NaN, Inf, or negative cost occurs in any transition or policy evaluation.
    """
    rng = np.random.default_rng(seed=42)
    engine = DeterministicTransitionEngine(cost_params=base_cost_params)
    oracle = OfflineOraclePolicy(cost_params=base_cost_params)
    policies = list_standard_policies()

    for i in range(1000):
        val = float(rng.choice([0.0, 10.0, 1000.0, 50000.0, 1e6, 1e8]))
        crit = float(rng.uniform(0.5, 3.0))
        mode_f = float(rng.choice([0.9, 1.0, 1.1, 1.25]))
        channel = str(rng.choice(["Direct Drop", "From RDC"]))
        p = float(rng.uniform(0.0, 1.0))
        d_exp = float(rng.choice([0.0, 1.0, 5.0, 15.0, 45.0, 120.0]))
        w = float(rng.uniform(0.1, 100.0))
        temp = float(rng.uniform(-20.0, 40.0)) if rng.random() > 0.5 else None
        dev_km = float(rng.uniform(0.0, 200.0)) if rng.random() > 0.5 else None

        st = ObservableShipmentState(
            shipment_id=f"FUZZ_{i}",
            pred_date=pd.Timestamp("2012-01-01"),
            line_item_value=val,
            clinical_criticality=crit,
            transport_mode_factor=mode_f,
            fulfillment_channel=channel,
            delay_prob=p,
            expected_delay_days=d_exp,
            uncertainty_width=w,
            iot_temperature_c=temp,
            iot_route_deviation_km=dev_km,
        )

        # Evaluate transitions across all candidate actions
        for act in ORACLE_ACTION_SPACE:
            res = engine.transition(st, act)
            assert not math.isnan(res.expected_realized_cost), f"NaN cost for {act} on fuzz {i}"
            assert not math.isinf(res.expected_realized_cost), f"Inf cost for {act} on fuzz {i}"
            assert res.expected_realized_cost >= 0.0, f"Negative cost for {act} on fuzz {i}"
            assert 0.0 <= res.residual_delay_prob <= 1.0, f"Invalid residual prob for {act} on fuzz {i}"
            assert res.residual_delay_days >= 0.0, f"Negative residual delay for {act} on fuzz {i}"

        # Evaluate all policies
        for pol_id, pol in policies.items():
            chosen_act = pol.select_action(st, base_cost_params)
            assert chosen_act in ORACLE_ACTION_SPACE

        # Evaluate Oracle
        best_act, opt_cost, _ = oracle.evaluate_optimal_action(st)
        assert not math.isnan(opt_cost)
        assert not math.isinf(opt_cost)
        assert opt_cost >= 0.0
        assert best_act in ORACLE_ACTION_SPACE
