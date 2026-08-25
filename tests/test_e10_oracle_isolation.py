"""
Unit tests for Offline Oracle Benchmark and Strict Architectural Isolation.
"""

from __future__ import annotations

import ast
from pathlib import Path
import pandas as pd
import pytest

from delay_intelligence.counterfactual.oracle import OfflineOraclePolicy
from delay_intelligence.counterfactual.policies import list_standard_policies
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


def test_oracle_cost_minimization(base_cost_params: dict):
    """Verifies that the Oracle selects the minimum cost action among all candidates."""
    oracle = OfflineOraclePolicy(cost_params=base_cost_params)

    state = ObservableShipmentState(
        shipment_id="TEST_ORACLE",
        pred_date=pd.Timestamp("2013-06-01"),
        line_item_value=120000.0,
        clinical_criticality=1.40,
        transport_mode_factor=1.10,
        fulfillment_channel="Direct Drop",
        delay_prob=0.75,
        expected_delay_days=18.0,
        uncertainty_width=12.0,
    )

    all_actions = oracle.evaluate_all_actions(state)
    best_act, min_cost, _ = oracle.evaluate_optimal_action(state)

    for act, res in all_actions.items():
        assert min_cost <= res.expected_realized_cost + 1e-6
    assert best_act in all_actions


def test_policy_regret_non_negative(base_cost_params: dict):
    """Verifies that Policy Regret is always >= 0.0 for every policy."""
    oracle = OfflineOraclePolicy(cost_params=base_cost_params)
    policies = list_standard_policies()

    test_states = [
        ObservableShipmentState(
            shipment_id=f"S_{i}",
            pred_date=pd.Timestamp("2012-01-01"),
            line_item_value=val,
            clinical_criticality=crit,
            transport_mode_factor=1.0,
            fulfillment_channel="Direct Drop" if i % 2 == 0 else "From RDC",
            delay_prob=prob,
            expected_delay_days=12.0,
            uncertainty_width=10.0,
        )
        for i, (val, crit, prob) in enumerate([
            (500.0, 1.0, 0.05),
            (25000.0, 1.2, 0.35),
            (200000.0, 1.6, 0.85),
            (800000.0, 2.0, 0.90),
        ])
    ]

    for st in test_states:
        for pol_id, pol in policies.items():
            act = pol.select_action(st, base_cost_params)
            regret, oracle_cost, oracle_act = oracle.compute_policy_regret(act, st)
            assert regret >= 0.0, f"Regret negative for {pol_id} on {st.shipment_id}"


def test_ast_architectural_isolation():
    """
    Scans policy and model source files to strictly verify that OfflineOraclePolicy
    or oracle.py is NEVER imported or referenced during online execution or model training.
    """
    src_dir = Path("src/delay_intelligence")
    prohibited_files = [
        src_dir / "counterfactual" / "policies.py",
        src_dir / "counterfactual" / "transitions.py",
        src_dir / "counterfactual" / "state.py",
        src_dir / "cost_sensitive" / "models.py",
        src_dir / "cost_sensitive" / "cost_engine.py",
        src_dir / "models" / "train.py",
    ]

    for fpath in prohibited_files:
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(fpath))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "oracle" not in alias.name.lower(), f"Isolation violation: {alias.name} in {fpath}"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "oracle" not in module.lower(), f"Isolation violation: from {module} in {fpath}"
                for alias in node.names:
                    assert "oracle" not in alias.name.lower(), f"Isolation violation: import {alias.name} in {fpath}"
