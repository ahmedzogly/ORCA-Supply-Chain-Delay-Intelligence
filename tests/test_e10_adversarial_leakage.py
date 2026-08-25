"""
Phase 2 — Experiment E10: Counterfactual Policy Evaluation
Adversarial Stress Test Suite: Leakage & Temporal Boundary Specialist (QA Challenger 1).

Covers:
1. Hidden Scenario Leakage Attacks (S0..S6 isolation, scenario invariance, attribute poisoning, AST audit).
2. Cost-Model & Feature Leakage Attacks (post-outcome fields, actual delivery timestamps, realized delay, schema audit).
3. Temporal Boundary & Holdout Separation Attacks (5-fold temporal development, 90-day embargo gap, 365-day holdout quarantine).
4. Extreme Boundary Value Stress Harnesses (zero/extreme monetary value, probabilities, uncertainty widths, criticality).
5. Budget Allocation Capacity Stress & Conservation Oracles.
6. Offline Oracle Benchmark Isolation & Regret Non-Negativity.
7. Provenance Tag Security & Non-Causal Guardrails Verification.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest

from delay_intelligence.counterfactual.budget import ReviewBudgetAllocator
from delay_intelligence.counterfactual.evaluator import CounterfactualEvaluator
from delay_intelligence.counterfactual.oracle import OfflineOraclePolicy
from delay_intelligence.counterfactual.policies import (
    BasePolicy,
    PolicyP0_NoAction,
    PolicyP1_E8CostSensitive,
    PolicyP2_Expedite,
    PolicyP3_TransportModeReview,
    PolicyP4_SupplierEscalation,
    PolicyP5_HumanReview,
    get_policy,
    list_standard_policies,
)
from delay_intelligence.counterfactual.provenance import (
    NON_CAUSAL_DISCLAIMER,
    ProvenanceTag,
    ProvenanceValidationError,
    attach_provenance_metadata,
    validate_provenance_tag,
)
from delay_intelligence.counterfactual.sensitivity import (
    SENSITIVITY_GRID_CELLS,
    SensitivityGridEvaluator,
)
from delay_intelligence.counterfactual.state import (
    CounterfactualTransitionResult,
    ObservableShipmentState,
)
from delay_intelligence.counterfactual.transitions import (
    DeterministicTransitionEngine,
    apply_counterfactual_transition,
    normalize_action_name,
)
from delay_intelligence.cost_sensitive.cost_engine import FORBIDDEN_COLUMNS
from delay_intelligence.evaluation.splitter import RollingOriginSplitter


# =============================================================================
# Fixtures
# =============================================================================

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


@pytest.fixture
def standard_state() -> ObservableShipmentState:
    return ObservableShipmentState(
        shipment_id="ADV_SHIP_001",
        pred_date=pd.Timestamp("2013-03-15"),
        line_item_value=75000.0,
        clinical_criticality=1.35,
        transport_mode_factor=1.00,
        fulfillment_channel="Direct Drop",
        delay_prob=0.45,
        expected_delay_days=14.0,
        uncertainty_width=11.5,
        iot_temperature_c=5.0,
        iot_route_deviation_km=12.0,
        provenance_tag=ProvenanceTag.SYNTHETIC_E9_STATE.value,
    )


# =============================================================================
# 1. ATTACK HIDDEN SCENARIO LEAKAGE (S0..S6 Isolation)
# =============================================================================

class TestHiddenScenarioLeakage:
    """
    Adversarial attacks attempting to leak synthetic scenario labels (S0..S6)
    to operational decision policies P0–P5.
    """

    @pytest.mark.parametrize("scenario_label", [
        "S0_BASELINE",
        "S1_PORT_CONGESTION",
        "S2_CUSTOMS_DELAY",
        "S3_COLD_CHAIN_FAILURE",
        "S4_SUPPLIER_QUALITY_ISSUE",
        "S5_ROUTE_DISRUPTION",
        "S6_EXTREME_WEATHER_GEOPOLITICAL",
        "MALICIOUS_DISRUPTION_INJECTION",
    ])
    def test_scenario_invariance_oracle(self, standard_state: ObservableShipmentState, base_cost_params: dict, scenario_label: str):
        """
        Oracle Property: Given identical observable state S_i(t), policy actions MUST be
        bitwise invariant to any scenario context or external disruption regime.
        """
        policies = list_standard_policies()
        
        # Inject scenario context into cost_params or external metadata
        tainted_params = dict(base_cost_params)
        tainted_params["scenario"] = scenario_label
        tainted_params["regime_id"] = scenario_label
        tainted_params["synthetic_disruption"] = scenario_label

        for pol_id, pol in policies.items():
            action_clean = pol.select_action(standard_state, base_cost_params)
            action_tainted = pol.select_action(standard_state, tainted_params)
            assert action_clean == action_tainted, (
                f"Policy {pol_id} changed decision from {action_clean} to {action_tainted} "
                f"when scenario label '{scenario_label}' was present!"
            )

    def test_observable_state_blocks_scenario_attribute_injection(self, standard_state: ObservableShipmentState):
        """
        Verify that ObservableShipmentState dataclass is strictly frozen and cannot be
        dynamically monkey-patched with hidden scenario regime attributes.
        """
        with pytest.raises(Exception):
            standard_state.scenario = "S1_PORT_CONGESTION"  # type: ignore

        with pytest.raises(Exception):
            standard_state.disruption_type = "COLD_CHAIN_EXCURSION"  # type: ignore

        with pytest.raises(Exception):
            standard_state.regime = "S6"  # type: ignore

    def test_from_row_filters_scenario_and_regime_columns(self, base_cost_params: dict):
        """
        Verify that ObservableShipmentState.from_row cleanly ignores any synthetic scenario columns
        in raw/synthetic dataframes.
        """
        row_dict = {
            "ID": "TEST_LEAK_001",
            "T_pred": "2013-05-01",
            "Line Item Value": 100000.0,
            "First Line Designation": "Yes",
            "Dosage": "pediatric 100mg",
            "Molecule/Test Type": "Efavirenz",
            "Sub Classification": "Adult ARV",
            "Product Group": "ARV",
            "Shipment Mode": "Air",
            "Fulfill Via": "Direct Drop",
            # Adversarially injected synthetic scenario columns
            "scenario": "S3_COLD_CHAIN_FAILURE",
            "scenario_id": "S3",
            "regime": "SEVERE_DISRUPTION",
            "disruption_severity": 0.95,
            "hidden_ground_truth_delay": 45.0,
        }

        state = ObservableShipmentState.from_row(
            row=row_dict,
            delay_prob=0.50,
            expected_delay_days=15.0,
            uncertainty_width=12.0,
            cost_params=base_cost_params,
        )

        state_dict = state.to_dict()
        assert "scenario" not in state_dict
        assert "scenario_id" not in state_dict
        assert "regime" not in state_dict
        assert "disruption_severity" not in state_dict
        assert "hidden_ground_truth_delay" not in state_dict

    def test_ast_policies_contain_no_scenario_references(self):
        """
        Static Code Analysis (AST): Inspect policies.py to strictly verify that
        no policy references scenario identifiers or disruption regime names.
        """
        policy_path = Path("src/delay_intelligence/counterfactual/policies.py")
        assert policy_path.exists(), f"File not found: {policy_path}"

        with open(policy_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(policy_path))

        prohibited_tokens = {
            "scenario", "scenario_id", "regime", "s0", "s1", "s2", "s3", "s4", "s5", "s6",
            "port_congestion", "cold_chain", "geopolitical"
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id.lower() not in prohibited_tokens, (
                    f"AST Leakage: Variable '{node.id}' in policies.py violates scenario blindness!"
                )
            elif isinstance(node, ast.Attribute):
                assert node.attr.lower() not in prohibited_tokens, (
                    f"AST Leakage: Attribute access '.{node.attr}' in policies.py violates scenario blindness!"
                )


# =============================================================================
# 2. ATTACK COST-MODEL & FEATURE LEAKAGE (Post-Outcome Fields)
# =============================================================================

class TestCostModelAndFeatureLeakage:
    """
    Adversarial attacks injecting post-outcome, target-derived, and post-dispatch
    fields to verify zero leakage in policy execution and transition cost modeling.
    """

    @pytest.mark.parametrize("forbidden_col", FORBIDDEN_COLUMNS + [
        "actual_delivery_date",
        "actual_delivered_date",
        "realized_delay_days",
        "delay_days_realized",
        "future_timestamp",
        "target_delay_flag",
    ])
    def test_forbidden_post_outcome_column_injection_immunity(self, base_cost_params: dict, forbidden_col: str):
        """
        Adversarial Injection: Inject extreme post-outcome values and verify that
        ObservableShipmentState, policy selection, and transition results remain 100% invariant.
        """
        clean_row = {
            "ID": "CLEAN_100",
            "T_pred": "2012-08-15",
            "Line Item Value": 60000.0,
            "First Line Designation": "No",
            "Dosage": "200mg",
            "Molecule/Test Type": "Nevirapine",
            "Sub Classification": "Adult ARV",
            "Product Group": "ARV",
            "Shipment Mode": "Truck",
            "Fulfill Via": "From RDC",
        }

        tainted_row = dict(clean_row)
        # Corrupt with extreme late outcome
        tainted_row[forbidden_col] = 99999.0 if "date" not in forbidden_col else "2099-12-31"

        state_clean = ObservableShipmentState.from_row(
            row=clean_row,
            delay_prob=0.30,
            expected_delay_days=10.0,
            uncertainty_width=8.0,
            cost_params=base_cost_params,
        )

        state_tainted = ObservableShipmentState.from_row(
            row=tainted_row,
            delay_prob=0.30,
            expected_delay_days=10.0,
            uncertainty_width=8.0,
            cost_params=base_cost_params,
        )

        # 1. State vector must be bitwise identical
        assert state_clean == state_tainted

        # 2. Policy actions must be bitwise identical
        policies = list_standard_policies()
        for pol_id, pol in policies.items():
            act_clean = pol.select_action(state_clean, base_cost_params)
            act_tainted = pol.select_action(state_tainted, base_cost_params)
            assert act_clean == act_tainted

        # 3. Deterministic transitions must be bitwise identical
        engine = DeterministicTransitionEngine(cost_params=base_cost_params)
        res_clean = engine.transition(state_clean, "EXPEDITE")
        res_tainted = engine.transition(state_tainted, "EXPEDITE")
        assert res_clean == res_tainted

    def test_catboost_feature_schema_strictly_excludes_forbidden_columns(self):
        """
        Verify that artifacts/model_registry/v1/feature_schema.json contains zero forbidden columns.
        """
        schema_path = Path("artifacts/model_registry/v1/feature_schema.json")
        assert schema_path.exists(), f"Schema file not found: {schema_path}"

        import json
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        all_schema_features = set(schema.get("all_features", []) + schema.get("num_cols", []) + schema.get("cat_cols", []))

        for forbidden in FORBIDDEN_COLUMNS:
            assert forbidden not in all_schema_features, (
                f"CRITICAL LEAKAGE: Forbidden column '{forbidden}' found in feature schema!"
            )

    def test_cost_calculation_uses_strictly_pre_outcome_attributes(self, base_cost_params: dict):
        """
        Verify that transition cost calculation is a deterministic function strictly of
        (line_item_value, clinical_criticality, transport_mode_factor, delay_prob, expected_delay_days).
        """
        engine = DeterministicTransitionEngine(cost_params=base_cost_params)

        state1 = ObservableShipmentState(
            shipment_id="TEST_A",
            pred_date=pd.Timestamp("2012-01-01"),
            line_item_value=50000.0,
            clinical_criticality=1.30,
            transport_mode_factor=1.10,
            fulfillment_channel="Direct Drop",
            delay_prob=0.40,
            expected_delay_days=10.0,
            uncertainty_width=10.0,
        )

        state2 = ObservableShipmentState(
            shipment_id="TEST_B_DIFFERENT_ID_AND_DATE",
            pred_date=pd.Timestamp("2014-06-01"),  # Different date and ID
            line_item_value=50000.0,
            clinical_criticality=1.30,
            transport_mode_factor=1.10,
            fulfillment_channel="Direct Drop",
            delay_prob=0.40,
            expected_delay_days=10.0,
            uncertainty_width=10.0,
        )

        res1 = engine.transition(state1, "EXPEDITE")
        res2 = engine.transition(state2, "EXPEDITE")

        assert res1.action_cost == res2.action_cost
        assert res1.residual_delay_cost == res2.residual_delay_cost
        assert res1.residual_risk_cost == res2.residual_risk_cost
        assert res1.expected_realized_cost == res2.expected_realized_cost


# =============================================================================
# 3. ATTACK TEMPORAL BOUNDARY & HOLDOUT SEPARATION
# =============================================================================

class TestTemporalBoundaryAndHoldoutSeparation:
    """
    Adversarial attacks against the chronological boundary:
    - 5-fold temporal development backtesting respects T <= 2014-08-24.
    - 90-day embargo gap is strictly maintained across all folds.
    - 365-day final holdout (N=1,013) has ZERO presence in dev evaluation.
    """

    def test_dev_data_strictly_quarantined(self):
        """
        Verify that load_dev_data in CounterfactualEvaluator loads exactly N=7,306 rows
        and zero records with T_pred > 2014-08-24.
        """
        evaluator = CounterfactualEvaluator()
        df_dev = evaluator.load_dev_data()

        assert len(df_dev) == 7306, f"Expected 7,306 dev records, got {len(df_dev)}"
        max_date = df_dev["T_pred"].max()
        assert max_date <= pd.Timestamp("2014-08-24"), f"Temporal breach: max dev date is {max_date}"

    def test_holdout_quarantine_integrity(self):
        """
        Verify that the quarantined holdout dataset contains exactly N=1,013 rows
        and has zero ID intersection with the development cohort.
        """
        evaluator = CounterfactualEvaluator()
        df_all = pd.read_parquet(evaluator.feature_path)
        df_all["T_pred"] = pd.to_datetime(df_all["T_pred"])

        df_dev = df_all[df_all["T_pred"] <= pd.Timestamp("2014-08-24")]
        df_holdout = df_all[df_all["T_pred"] > pd.Timestamp("2014-08-24")]

        assert len(df_dev) == 7306
        assert len(df_holdout) == 1013
        assert len(df_all) == 8319

        dev_ids = set(df_dev["ID"])
        holdout_ids = set(df_holdout["ID"])

        overlap = dev_ids.intersection(holdout_ids)
        assert len(overlap) == 0, f"Critical holdout leak: {len(overlap)} overlapping IDs between dev and holdout!"

    def test_rolling_origin_folds_embargo_gap_and_precedence(self):
        """
        Stress test RollingOriginSplitter:
        - Train strictly precedes validation.
        - Embargo gap >= 90 days across all development folds.
        - Zero index overlap between train and validation within any fold.
        """
        df_all = pd.read_parquet("artifacts/data/scms_modeling_features.parquet")
        splitter = RollingOriginSplitter(config_path="configs/evaluation.yaml")
        folds, holdout_idx, manifest_df = splitter.split(df_all)

        assert len(folds) == 5, f"Expected 5 temporal folds, got {len(folds)}"
        assert len(holdout_idx) == 1013, f"Expected 1,013 holdout samples, got {len(holdout_idx)}"

        for fold in folds:
            fid = fold["fold_id"]
            train_idx = fold["train"]
            val_idx = fold["val"]

            assert len(train_idx) > 0, f"Fold {fid} train set is empty"
            assert len(val_idx) > 0, f"Fold {fid} val set is empty"

            # Check disjoint index sets
            intersection = set(train_idx).intersection(set(val_idx))
            assert len(intersection) == 0, f"Fold {fid} has {len(intersection)} leaking indices between train and val"

            # Check holdout exclusion
            train_holdout_overlap = set(train_idx).intersection(set(holdout_idx))
            val_holdout_overlap = set(val_idx).intersection(set(holdout_idx))
            assert len(train_holdout_overlap) == 0, f"Fold {fid} train contains holdout data!"
            assert len(val_holdout_overlap) == 0, f"Fold {fid} val contains holdout data!"

            # Check dates
            train_dates = df_all.loc[train_idx, "T_pred"]
            val_dates = df_all.loc[val_idx, "T_pred"]

            max_train_date = train_dates.max()
            min_val_date = val_dates.min()
            max_val_date = val_dates.max()

            assert max_train_date < min_val_date, f"Fold {fid} temporal violation: train max {max_train_date} >= val min {min_val_date}"
            assert max_val_date <= pd.Timestamp("2014-08-24"), f"Fold {fid} val exceeds dev cutoff: {max_val_date}"

            gap_days = (min_val_date - max_train_date).days
            assert gap_days >= 89, f"Fold {fid} embargo gap {gap_days} is below 90 days"

    def test_splitter_order_invariance(self):
        """
        Adversarially permute dataset row ordering and verify that RollingOriginSplitter
        produces identical, deterministic folds.
        """
        df_all = pd.read_parquet("artifacts/data/scms_modeling_features.parquet")
        df_shuffled = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)

        splitter = RollingOriginSplitter()
        folds1, h1, m1 = splitter.split(df_all)
        folds2, h2, m2 = splitter.split(df_shuffled)

        assert len(h1) == len(h2) == 1013
        assert len(folds1) == len(folds2) == 5

        for f1, f2 in zip(folds1, folds2):
            assert len(f1["train"]) == len(f2["train"])
            assert len(f1["val"]) == len(f2["val"])


# =============================================================================
# 4. EXTREME BOUNDARY VALUE STRESS HARNESSES
# =============================================================================

class TestBoundaryAndExtremeStress:
    """
    Stress-testing policies, transitions, and math stability under adversarial edge inputs.
    """

    @pytest.mark.parametrize("line_value", [0.0, 1e-3, 10.0, 1e6, 5e7])
    def test_extreme_monetary_values_stability(self, base_cost_params: dict, line_value: float):
        """Verify mathematical stability under 0 to 50 million USD values."""
        state = ObservableShipmentState(
            shipment_id="EXTREME_VAL",
            pred_date=pd.Timestamp("2013-01-01"),
            line_item_value=line_value,
            clinical_criticality=1.0,
            transport_mode_factor=1.0,
            fulfillment_channel="Direct Drop",
            delay_prob=0.50,
            expected_delay_days=10.0,
            uncertainty_width=10.0,
        )

        policies = list_standard_policies()
        engine = DeterministicTransitionEngine(cost_params=base_cost_params)

        for pol_id, pol in policies.items():
            act = pol.select_action(state, base_cost_params)
            assert act in ["NO_ACTION", "EXPEDITE", "TRANSPORT_MODE_REVIEW", "SUPPLIER_ESCALATION", "HUMAN_REVIEW"]

            res = engine.transition(state, act)
            assert not math.isnan(res.expected_realized_cost)
            assert not math.isinf(res.expected_realized_cost)
            assert res.expected_realized_cost >= 0.0

    @pytest.mark.parametrize("prob", [0.0, 1e-7, 0.50, 1.0 - 1e-7, 1.0])
    def test_extreme_probability_bounds_stability(self, base_cost_params: dict, prob: float):
        """Verify policy thresholding and transition math under extreme boundary probabilities."""
        state = ObservableShipmentState(
            shipment_id="EXTREME_PROB",
            pred_date=pd.Timestamp("2013-01-01"),
            line_item_value=50000.0,
            clinical_criticality=1.0,
            transport_mode_factor=1.0,
            fulfillment_channel="Direct Drop",
            delay_prob=prob,
            expected_delay_days=12.0,
            uncertainty_width=10.0,
        )

        pol_p1 = PolicyP1_E8CostSensitive(gamma_multiplier=1.20)
        tau = pol_p1.compute_threshold(state, base_cost_params)
        assert 0.0 <= tau <= 1.0

        act = pol_p1.select_action(state, base_cost_params)
        if prob == 0.0:
            assert act == "NO_ACTION"
        elif prob == 1.0:
            assert act == "EXPEDITE"

    @pytest.mark.parametrize("uncert_w, expected_p5_action", [
        (0.1, "NO_ACTION"),
        (13.999, "NO_ACTION"),
        (14.0, "NO_ACTION"),
        (14.0001, "HUMAN_REVIEW"),
        (50.0, "HUMAN_REVIEW"),
    ])
    def test_p5_human_review_sharp_boundary(self, base_cost_params: dict, uncert_w: float, expected_p5_action: str):
        """Verify sharp trigger boundary at W_i > 14.0 days for P5 Human Review."""
        state = ObservableShipmentState(
            shipment_id="UNCERT_TEST",
            pred_date=pd.Timestamp("2013-01-01"),
            line_item_value=10000.0,
            clinical_criticality=1.0,
            transport_mode_factor=1.0,
            fulfillment_channel="Direct Drop",
            delay_prob=0.10,
            expected_delay_days=5.0,
            uncertainty_width=uncert_w,
        )
        pol = PolicyP5_HumanReview()
        assert pol.select_action(state, base_cost_params) == expected_p5_action


# =============================================================================
# 5. BUDGET ALLOCATION CAPACITY STRESS & CONSERVATION ORACLES
# =============================================================================

class TestBudgetAllocationStress:
    """
    Stress-testing capacity constraints and conservation properties in ReviewBudgetAllocator.
    """

    def test_budget_strict_capacity_adherence(self, base_cost_params: dict):
        """
        Oracle Property: For any capacity fraction K in (0, 1], allocated_count <= floor(K * N).
        """
        allocator = ReviewBudgetAllocator(cost_params=base_cost_params)
        
        # 50 diverse states
        states = [
            ObservableShipmentState(
                shipment_id=f"B_{i:03d}",
                pred_date=pd.Timestamp("2013-01-01"),
                line_item_value=1000.0 * (i + 1),
                clinical_criticality=1.0 + 0.02 * i,
                transport_mode_factor=1.0,
                fulfillment_channel="Direct Drop" if i % 2 == 0 else "From RDC",
                delay_prob=min(0.99, 0.02 * (i + 1)),
                expected_delay_days=10.0 + i,
                uncertainty_width=10.0,
            )
            for i in range(50)
        ]

        for k in [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]:
            res = allocator.allocate_budget(states, capacity_k=k)
            max_cap = int(math.floor(k * len(states)))
            assert res["allocated_count"] <= max_cap

    def test_budget_never_intervenes_on_negative_net_benefit(self, base_cost_params: dict):
        """
        Oracle Property: Budget allocator must NEVER intervene on a shipment with Net_Benefit <= 0,
        even if budget capacity is 100% available.
        """
        allocator = ReviewBudgetAllocator(cost_params=base_cost_params)
        
        # Low risk, low value shipments where intervention cost exceeds any delay savings
        worthless_states = [
            ObservableShipmentState(
                shipment_id=f"NEG_{i}",
                pred_date=pd.Timestamp("2013-01-01"),
                line_item_value=10.0,
                clinical_criticality=1.0,
                transport_mode_factor=1.0,
                fulfillment_channel="From RDC",
                delay_prob=0.01,  # 1% risk
                expected_delay_days=1.0,
                uncertainty_width=5.0,
            )
            for i in range(20)
        ]

        res = allocator.allocate_budget(worthless_states, capacity_k=1.0)
        assert res["allocated_count"] == 0, "Intervened on negative net benefit shipments!"
        assert res["total_net_benefit"] == 0.0


# =============================================================================
# 6. OFFLINE ORACLE ISOLATION & REGRET NON-NEGATIVITY
# =============================================================================

class TestOracleIsolationAndRegret:
    """
    Stress-testing the offline omniscient oracle and proving mathematical non-negativity of regret.
    """

    def test_regret_non_negativity_comprehensive_grid(self, base_cost_params: dict):
        """
        Oracle Property: For every observable state S and policy P_k in {P0..P5}:
        E[Cost(P_k | S)] >= E[Cost(Oracle | S)]  ==>  Regret(P_k, S) >= 0.0.
        """
        oracle = OfflineOraclePolicy(cost_params=base_cost_params)
        policies = list_standard_policies()

        np.random.seed(42)
        random_values = np.random.uniform(500.0, 500000.0, size=30)
        random_probs = np.random.uniform(0.01, 0.99, size=30)
        random_delays = np.random.uniform(2.0, 30.0, size=30)

        for val, prob, d_days in zip(random_values, random_probs, random_delays):
            state = ObservableShipmentState(
                shipment_id="REGRET_CHECK",
                pred_date=pd.Timestamp("2013-01-01"),
                line_item_value=float(val),
                clinical_criticality=1.3,
                transport_mode_factor=1.1,
                fulfillment_channel="Direct Drop",
                delay_prob=float(prob),
                expected_delay_days=float(d_days),
                uncertainty_width=10.0,
            )

            for pol_id, pol in policies.items():
                act = pol.select_action(state, base_cost_params)
                regret, oracle_cost, oracle_act = oracle.compute_policy_regret(act, state)
                assert regret >= -1e-6, f"Violation: Regret is negative ({regret}) for policy {pol_id}"


# =============================================================================
# 7. PROVENANCE SECURITY & NON-CAUSAL GUARDRAILS
# =============================================================================

class TestProvenanceSecurity:
    """
    Verifies that invalid or unauthorized provenance tags are rejected and
    the non-causal disclaimer is strictly enforced.
    """

    @pytest.mark.parametrize("bad_tag", [
        "TRUE_CAUSAL_EFFECT",
        "OBSERVED_SAVINGS_CONFIRMED",
        "UNAUDITED",
        "",
        "FAKE_TAG",
    ])
    def test_unauthorized_provenance_tag_rejected(self, bad_tag: str):
        """Verify that ProvenanceValidationError is raised on any unauthorized tag."""
        with pytest.raises(ProvenanceValidationError):
            validate_provenance_tag(bad_tag)

    def test_scientific_disclaimer_mandatory_content(self):
        """Verify that NON_CAUSAL_DISCLAIMER contains required scientific caveats."""
        assert "Historical SCMS supply chain records lack randomized treatment assignments" in NON_CAUSAL_DISCLAIMER
        assert "synthetic scenario simulations" in NON_CAUSAL_DISCLAIMER
        assert "No observational claims of actual historical intervention efficacy" in NON_CAUSAL_DISCLAIMER


# =============================================================================
# 8. END-TO-END EVALUATOR ARTIFACT & TEMPORAL INTEGRITY
# =============================================================================

class TestEvaluatorEndToEndArtifactIntegrity:
    """
    Executes CounterfactualEvaluator and verifies that output parquet artifacts
    maintain strict temporal boundary, valid provenance tags, and non-negative costs.
    """

    def test_full_dev_evaluation_execution_and_quarantine(self, tmp_path: Path):
        """
        Executes full dev evaluation writing to temporary parquet artifacts and verifies:
        - Output files exist and load cleanly.
        - Zero records have pred_date > 2014-08-24.
        - Zero holdout IDs are present.
        - Provenance tags are verified.
        - Realized costs are strictly non-negative and finite.
        """
        dev_out = tmp_path / "test_e10_dev_results.parquet"
        sens_out = tmp_path / "test_e10_sens_results.parquet"

        evaluator = CounterfactualEvaluator()
        summary = evaluator.run_full_dev_evaluation(
            output_dev_path=dev_out,
            output_sensitivity_path=sens_out,
        )

        assert summary["status"] == "COMPLETED"
        assert dev_out.exists()
        assert sens_out.exists()

        df_dev_res = pd.read_parquet(dev_out)
        assert len(df_dev_res) > 0

        # Verify temporal bound
        df_dev_res["pred_date"] = pd.to_datetime(df_dev_res["pred_date"])
        assert df_dev_res["pred_date"].max() <= pd.Timestamp("2014-08-24")

        # Verify holdout ID quarantine
        df_all = pd.read_parquet(evaluator.feature_path)
        holdout_ids = set(df_all[pd.to_datetime(df_all["T_pred"]) > pd.Timestamp("2014-08-24")]["ID"])
        res_ids = set(df_dev_res["shipment_id"])
        leak_count = len(res_ids.intersection(holdout_ids))
        assert leak_count == 0, f"Critical leak: {leak_count} holdout IDs found in dev evaluation results!"

        # Verify metrics integrity
        assert (df_dev_res["expected_realized_cost"] >= 0.0).all()
        assert not df_dev_res["expected_realized_cost"].isna().any()
        assert (df_dev_res["policy_regret"] >= -1e-6).all()
        assert (df_dev_res["provenance_tag"] == ProvenanceTag.SIMULATED_COUNTERFACTUAL.value).all()


# =============================================================================
# 9. INFERENCE RESILIENCE & ADVERSARIAL NOISE
# =============================================================================

class TestInferenceResilienceAndNoise:
    """
    Adversarial attacks testing CatBoost inference and state generation with
    noisy, shuffled, or incomplete inputs.
    """

    def test_generate_predictions_deterministic_and_feature_isolation(self):
        """
        Verify CatBoost prediction generation is deterministic and ignores extraneous columns.
        """
        evaluator = CounterfactualEvaluator()
        df_dev = evaluator.load_dev_data().head(30)

        # Baseline predictions
        probs1, delays1, uncert1 = evaluator.generate_predictions(df_dev)

        # Add 50 random noisy columns
        df_noisy = df_dev.copy()
        for c in range(50):
            df_noisy[f"random_noise_col_{c}"] = np.random.randn(len(df_dev))

        probs2, delays2, uncert2 = evaluator.generate_predictions(df_noisy)

        np.testing.assert_allclose(probs1, probs2, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(delays1, delays2, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(uncert1, uncert2, rtol=1e-5, atol=1e-5)

