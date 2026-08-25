"""
Independent QA Adversarial Challenge & Stress Test Suite for Phase 2 — Experiment E8 Milestone 5
(Final 365-Day Holdout Verification).

Adversarial Verification Vectors:
1. Strict Temporal Holdout Isolation & Data Split Counts (7,306 dev vs 1,013 holdout).
2. Frozen Policy Compliance, Immutability & Zero Retuning Audit.
3. Holdout Parquet & JSON Artifact Schema, Types, and Null Audits (3,039 rows, 47 columns, 0 nulls).
4. Instance-Level Mathematical Identities & Economic Loss Matrix Consistency.
5. Operational Review Priority Monotonicity & Budget Capacity Enforcement.
6. Cross-Artifact Metric Reconciliation (Parquet vs JSON vs Reports).
7. Decision Rationality & Multi-Scenario Economic Superiority Invariants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from delay_intelligence.cost_sensitive.cost_engine import (
    CostScenarioModel,
    FORBIDDEN_COLUMNS,
)
from delay_intelligence.cost_sensitive.holdout_evaluator import (
    FinalHoldoutEvaluator,
    run_e8_holdout_evaluation,
)
from delay_intelligence.cost_sensitive.policy_freeze import (
    FrozenCostPolicy,
    MAX_ALLOWED_DEV_DATE_UTC,
    verify_temporal_holdout_isolation,
)


@pytest.fixture(scope="module")
def frozen_policy_path() -> Path:
    return Path("artifacts/results/e8_frozen_policy.json")


@pytest.fixture(scope="module")
def holdout_results_parquet_path() -> Path:
    return Path("artifacts/results/e8_final_holdout_results.parquet")


@pytest.fixture(scope="module")
def holdout_metrics_json_path() -> Path:
    return Path("artifacts/results/e8_final_holdout_metrics.json")


@pytest.fixture(scope="module")
def features_path() -> Path:
    return Path("artifacts/data/scms_modeling_features.parquet")


@pytest.fixture(scope="module")
def holdout_evaluator(
    frozen_policy_path: Path, features_path: Path
) -> FinalHoldoutEvaluator:
    return FinalHoldoutEvaluator(
        frozen_policy_path=frozen_policy_path,
        features_path=features_path,
    )


@pytest.fixture(scope="module")
def holdout_parquet_df(holdout_results_parquet_path: Path) -> pd.DataFrame:
    assert holdout_results_parquet_path.exists(), "Holdout parquet results artifact missing"
    return pd.read_parquet(holdout_results_parquet_path)


@pytest.fixture(scope="module")
def holdout_metrics_dict(holdout_metrics_json_path: Path) -> Dict[str, Any]:
    assert holdout_metrics_json_path.exists(), "Holdout metrics JSON artifact missing"
    with open(holdout_metrics_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestAdversarialHoldoutIsolation:
    """Vector 1: Adversarial verification of temporal holdout isolation and data boundaries."""

    def test_exact_holdout_partition_counts_and_dates(
        self, holdout_evaluator: FinalHoldoutEvaluator
    ):
        """Verify strict row count (7,306 dev, 1,013 holdout, 8,319 total) and date bounds."""
        df_dev, df_holdout = holdout_evaluator.load_and_split_data()

        assert len(df_dev) == 7306, f"Expected 7,306 dev rows, got {len(df_dev)}"
        assert len(df_holdout) == 1013, f"Expected 1,013 holdout rows, got {len(df_holdout)}"
        assert len(df_dev) + len(df_holdout) == 8319

        max_dev_date = pd.to_datetime(df_dev["T_pred"]).max()
        min_holdout_date = pd.to_datetime(df_holdout["T_pred"]).min()

        assert max_dev_date <= pd.to_datetime("2014-08-24 23:59:59")
        assert min_holdout_date > pd.to_datetime("2014-08-24 23:59:59")

    def test_dev_isolation_verification_raises_no_error(
        self, holdout_evaluator: FinalHoldoutEvaluator
    ):
        """Verify development split passes strict temporal isolation audit without error."""
        df_dev, _ = holdout_evaluator.load_and_split_data()
        verify_temporal_holdout_isolation(
            df_dev, date_col="T_pred", max_allowed_date=MAX_ALLOWED_DEV_DATE_UTC
        )

    def test_dev_inner_train_val_embargo_integrity(
        self, holdout_evaluator: FinalHoldoutEvaluator
    ):
        """Verify development inner-train and inner-val are separated by at least 30-day embargo."""
        df_dev, _ = holdout_evaluator.load_and_split_data()
        inner_tr, inner_val = holdout_evaluator.split_dev_inner_train_val(df_dev, inner_gap_days=30)

        max_tr_date = pd.to_datetime(inner_tr["T_pred"]).max()
        min_val_date = pd.to_datetime(inner_val["T_pred"]).min()

        gap_days = (min_val_date - max_tr_date).days
        assert gap_days >= 30, f"Embargo gap violated: got {gap_days} days"
        assert len(inner_tr) == 4928
        assert len(inner_val) == 2283
        assert len(inner_tr) + len(inner_val) + 95 == 7306

class TestAdversarialFrozenPolicyIntegrity:
    """Vector 2: Frozen policy parameters, zero retuning, and feature contract audit."""

    def test_frozen_policy_hyperparameters(
        self, holdout_evaluator: FinalHoldoutEvaluator
    ):
        """Verify champion hyperparameters match frozen specification."""
        champ = holdout_evaluator.frozen_policy.champion
        assert champ.strategy_id == "E8-C"
        assert champ.strategy_name == "E8-C_tuned_gamma"
        assert champ.gamma_tuned_multiplier == 1.20
        assert champ.governed_standard_threshold == 0.50
        assert champ.governed_f1_threshold == 0.170
        assert champ.model_hyperparameters.get("iterations") == 300
        assert champ.model_hyperparameters.get("learning_rate") == 0.05
        assert champ.model_hyperparameters.get("depth") == 6

    def test_feature_contract_contains_no_forbidden_columns(
        self, holdout_evaluator: FinalHoldoutEvaluator
    ):
        """Verify feature contract strictly forbids all post-outcome columns."""
        contract = holdout_evaluator.frozen_policy.feature_contract
        all_features = set(contract.all_features)
        forbidden = set(contract.forbidden_columns)

        assert len(forbidden) == 8
        assert all_features.isdisjoint(forbidden)
        for col in FORBIDDEN_COLUMNS:
            assert col in forbidden


class TestAdversarialParquetAndJsonSchema:
    """Vector 3: Parquet and JSON schema, datatype, null-free, and completeness audit."""

    def test_holdout_parquet_shape_and_no_nulls(
        self, holdout_parquet_df: pd.DataFrame
    ):
        """Verify holdout parquet has 3,039 rows, 47 columns, and zero nulls."""
        assert holdout_parquet_df.shape == (3039, 47), f"Unexpected parquet shape: {holdout_parquet_df.shape}"
        assert holdout_parquet_df.isna().sum().sum() == 0, "Found NaNs in holdout parquet"

        for sc in ["low", "base", "high"]:
            assert (holdout_parquet_df["scenario"] == sc).sum() == 1013

    def test_holdout_parquet_datatypes_and_ranges(
        self, holdout_parquet_df: pd.DataFrame
    ):
        """Verify probabilities, thresholds, costs, decisions, and scores are within valid physical domains."""
        # Probabilities in [0.0, 1.0]
        prob_cols = [c for c in holdout_parquet_df.columns if c.endswith("_prob")]
        for c in prob_cols:
            vals = holdout_parquet_df[c].to_numpy(dtype=float)
            assert np.all((vals >= 0.0) & (vals <= 1.0)), f"Probability out of bounds in {c}"

        # Thresholds in [0.0, 1.0]
        thresh_cols = [c for c in holdout_parquet_df.columns if c.endswith("_threshold") or "tau_star" in c]
        for c in thresh_cols:
            vals = holdout_parquet_df[c].to_numpy(dtype=float)
            assert np.all((vals >= 0.0) & (vals <= 1.0)), f"Threshold out of bounds in {c}"

        # Binary decisions in {0, 1}
        dec_cols = [c for c in holdout_parquet_df.columns if "decision" in c or c == "y_true"]
        for c in dec_cols:
            vals = holdout_parquet_df[c].to_numpy(dtype=int)
            assert np.all(np.isin(vals, [0, 1])), f"Non-binary decision in {c}"

        # Costs >= 0.0
        cost_cols = ["fn_cost", "fp_cost", "intervention_cost", "residual_delay_cost", "do_nothing_cost"]
        for c in cost_cols:
            vals = holdout_parquet_df[c].to_numpy(dtype=float)
            assert np.all(vals >= 0.0), f"Negative cost in {c}"

    def test_holdout_json_structure_and_completeness(
        self, holdout_metrics_dict: Dict[str, Any]
    ):
        """Verify JSON metrics summary contains all required metadata, unconstrained, and budget evaluations."""
        assert "metadata" in holdout_metrics_dict
        assert "unconstrained_evaluation" in holdout_metrics_dict
        assert "operational_budget_evaluation" in holdout_metrics_dict
        assert "champion_performance_summary" in holdout_metrics_dict

        meta = holdout_metrics_dict["metadata"]
        assert meta["holdout_sample_count"] == 1013
        assert meta["holdout_positives_count"] == 61
        assert meta["champion_strategy"] == "E8-C_tuned_gamma"
        assert meta["champion_gamma"] == 1.20
        assert meta["evaluation_status"] == "STRICT_SINGLE_PASS_COMPLETE"

        for sc in ["low", "base", "high"]:
            assert sc in holdout_metrics_dict["unconstrained_evaluation"]
            assert sc in holdout_metrics_dict["operational_budget_evaluation"]
            assert sc in holdout_metrics_dict["champion_performance_summary"]


class TestAdversarialMathematicalIdentities:
    """Vector 4: Mathematical identities, economic loss equations, and decision consistency."""

    def test_bayes_and_champion_threshold_identities(
        self, holdout_parquet_df: pd.DataFrame
    ):
        """Verify Bayes and Champion threshold formulas hold for every single row."""
        for sc in ["low", "base", "high"]:
            sub = holdout_parquet_df[holdout_parquet_df["scenario"] == sc]
            denom = np.maximum(1e-9, sub["net_benefit"] + sub["fp_cost"])
            expected_tau_bayes = np.clip(sub["fp_cost"] / denom, 0.0, 1.0)
            diff_bayes = np.abs(sub["tau_star_bayes"] - expected_tau_bayes)
            assert np.all(diff_bayes < 1e-5), f"Bayes threshold formula violated in {sc}"

            expected_tau_champ = np.clip(1.20 * sub["tau_star_bayes"], 0.0, 1.0)
            diff_champ = np.abs(sub["tau_star_champion"] - expected_tau_champ)
            assert np.all(diff_champ < 1e-5), f"Champion threshold formula violated in {sc}"

    def test_strategy_decision_rule_and_cost_accounting(
        self, holdout_parquet_df: pd.DataFrame
    ):
        """Verify decision == (prob >= threshold) and realized cost matches economic loss matrix."""
        strategies = [
            "E8-A_tau0.5",
            "E8-A_f1",
            "E8-B_cost_weighted",
            "E8-C_bayes_threshold",
            "E8-C_tuned_gamma",
        ]
        for sc in ["low", "base", "high"]:
            sub = holdout_parquet_df[holdout_parquet_df["scenario"] == sc]
            y = sub["y_true"].to_numpy(dtype=int)

            for strat in strategies:
                p = sub[f"{strat}_prob"].to_numpy(dtype=float)
                t = sub[f"{strat}_threshold"].to_numpy(dtype=float)
                d = sub[f"{strat}_decision"].to_numpy(dtype=int)

                expected_dec = (p >= t).astype(int)
                assert np.array_equal(d, expected_dec), f"Decision mismatch for {strat} in {sc}"

                interv = sub["intervention_cost"].to_numpy(dtype=float)
                resid = sub["residual_delay_cost"].to_numpy(dtype=float)
                fp = sub["fp_cost"].to_numpy(dtype=float)
                fn = sub["fn_cost"].to_numpy(dtype=float)

                expected_cost = np.where(
                    d == 1,
                    np.where(y == 1, interv + resid, fp),
                    np.where(y == 1, fn, 0.0),
                )
                actual_cost = sub[f"{strat}_realized_cost"].to_numpy(dtype=float)
                diff = np.abs(actual_cost - expected_cost)
                assert np.all(diff < 1e-5), f"Realized cost formula violated for {strat} in {sc}"

    def test_do_nothing_cost_identity(
        self, holdout_parquet_df: pd.DataFrame
    ):
        """Verify do_nothing_cost == (fn_cost if y_true == 1 else 0.0) for every row."""
        for sc in ["low", "base", "high"]:
            sub = holdout_parquet_df[holdout_parquet_df["scenario"] == sc]
            expected = np.where(sub["y_true"] == 1, sub["fn_cost"], 0.0)
            actual = sub["do_nothing_cost"].to_numpy(dtype=float)
            diff = np.abs(actual - expected)
            assert np.all(diff < 1e-5), f"Do-Nothing cost identity violated in {sc}"


class TestAdversarialPriorityMonotonicityAndBudgeting:
    """Vector 5: Operational review priority monotonicity, capacity limits, and hierarchical nesting."""

    def test_budget_capacity_counts(
        self, holdout_parquet_df: pd.DataFrame
    ):
        """Verify exact capacity floor counts (50, 101, 202) are enforced."""
        for sc in ["low", "base", "high"]:
            sub = holdout_parquet_df[holdout_parquet_df["scenario"] == sc]
            assert int(sub["budget_decision_cs_k05"].sum()) == 50
            assert int(sub["budget_decision_cs_k10"].sum()) == 101
            assert int(sub["budget_decision_cs_k20"].sum()) == 202
            assert int(sub["budget_decision_val_k10"].sum()) == 101
            assert int(sub["budget_decision_risk_k10"].sum()) == 101

            # Standard is capped at those with prob >= 0.50 (5 items)
            assert int(sub["budget_decision_std_k10"].sum()) == 5

    def test_priority_ranking_strict_monotonicity(
        self, holdout_parquet_df: pd.DataFrame
    ):
        """Verify selected items have higher priority scores than unselected items."""
        for sc in ["low", "base", "high"]:
            sub = holdout_parquet_df[holdout_parquet_df["scenario"] == sc]

            for k_val, col in [(0.05, "budget_decision_cs_k05"), (0.10, "budget_decision_cs_k10"), (0.20, "budget_decision_cs_k20")]:
                selected = sub[sub[col] == 1]
                unselected = sub[sub[col] == 0]
                min_sel = selected["priority_score_cost_sensitive"].min()
                max_unsel = unselected["priority_score_cost_sensitive"].max()
                assert min_sel >= max_unsel - 1e-6, (
                    f"Priority monotonicity violated in {sc} for CS K={k_val}: min_sel ({min_sel}) < max_unsel ({max_unsel})"
                )

    def test_budget_hierarchical_nesting(
        self, holdout_parquet_df: pd.DataFrame
    ):
        """Verify items selected at K=5% are a strict subset of K=10%, which is a subset of K=20%."""
        for sc in ["low", "base", "high"]:
            sub = holdout_parquet_df[holdout_parquet_df["scenario"] == sc]
            idx_k05 = set(sub[sub["budget_decision_cs_k05"] == 1]["ID"])
            idx_k10 = set(sub[sub["budget_decision_cs_k10"] == 1]["ID"])
            idx_k20 = set(sub[sub["budget_decision_cs_k20"] == 1]["ID"])

            assert idx_k05.issubset(idx_k10), f"Hierarchy violated: K=5% not subset of K=10% in {sc}"
            assert idx_k10.issubset(idx_k20), f"Hierarchy violated: K=10% not subset of K=20% in {sc}"


class TestAdversarialCrossMetricReconciliation:
    """Vector 6: Reconcile Parquet aggregates with JSON metrics summary."""

    def test_unconstrained_metrics_reconciliation(
        self, holdout_parquet_df: pd.DataFrame, holdout_metrics_dict: Dict[str, Any]
    ):
        """Verify unconstrained realized cost, net savings, reviews, and recall match JSON exactly."""
        for sc in ["low", "base", "high"]:
            sub = holdout_parquet_df[holdout_parquet_df["scenario"] == sc]
            json_uncon = holdout_metrics_dict["unconstrained_evaluation"][sc]

            # Do-Nothing
            none_cost = float(sub[sub["y_true"] == 1]["fn_cost"].sum())
            assert abs(none_cost - json_uncon["Do-Nothing"]["realized_cost"]) < 1e-2

            # Always-Intervene
            always_cost = float(np.where(sub["y_true"] == 1, sub["intervention_cost"] + sub["residual_delay_cost"], sub["fp_cost"]).sum())
            assert abs(always_cost - json_uncon["Always-Intervene"]["realized_cost"]) < 1e-2

            for strat in ["E8-A_tau0.5", "E8-A_f1", "E8-B_cost_weighted", "E8-C_bayes_threshold", "E8-C_tuned_gamma"]:
                cost_calc = float(sub[f"{strat}_realized_cost"].sum())
                cost_json = json_uncon[strat]["realized_cost"]
                assert abs(cost_calc - cost_json) < 1e-2, f"Cost mismatch for {strat} in {sc}"

                savings_calc = none_cost - cost_calc
                savings_json = json_uncon[strat]["net_savings"]
                assert abs(savings_calc - savings_json) < 1e-2, f"Savings mismatch for {strat} in {sc}"

                rev_calc = int(sub[f"{strat}_decision"].sum())
                rev_json = json_uncon[strat]["reviews_count"]
                assert rev_calc == rev_json, f"Reviews mismatch for {strat} in {sc}"

                rec_calc = float(recall_score(sub["y_true"], sub[f"{strat}_decision"]))
                rec_json = json_uncon[strat]["recall"]
                assert abs(rec_calc - rec_json) < 1e-3, f"Recall mismatch for {strat} in {sc}"

    def test_budget_metrics_savings_math_consistency(
        self, holdout_metrics_dict: Dict[str, Any]
    ):
        """Verify net savings formulas vs Value-Only, Risk-Only, and Standard in JSON."""
        budget_data = holdout_metrics_dict["operational_budget_evaluation"]
        for sc in ["low", "base", "high"]:
            for k_key in ["K_05pct", "K_10pct", "K_20pct"]:
                cost_val = budget_data[sc][k_key]["VALUE_ONLY"]["realized_business_cost"]
                cost_risk = budget_data[sc][k_key]["RISK_ONLY"]["realized_business_cost"]
                cost_std = budget_data[sc][k_key]["STANDARD"]["realized_business_cost"]
                cost_cs = budget_data[sc][k_key]["COST_SENSITIVE"]["realized_business_cost"]

                sav_val = budget_data[sc][k_key]["COST_SENSITIVE"]["net_savings_vs_value_only"]
                sav_risk = budget_data[sc][k_key]["COST_SENSITIVE"]["net_savings_vs_risk_only"]
                sav_std = budget_data[sc][k_key]["COST_SENSITIVE"]["net_savings_vs_standard"]

                assert abs(sav_val - (cost_val - cost_cs)) < 1e-2
                assert abs(sav_risk - (cost_risk - cost_cs)) < 1e-2
                assert abs(sav_std - (cost_std - cost_cs)) < 1e-2


class TestAdversarialEconomicSuperiorityAndDecisionRationality:
    """Vector 7: Economic superiority, positive net savings, and rationality across all scenarios."""

    def test_champion_net_savings_strictly_positive(
        self, holdout_metrics_dict: Dict[str, Any]
    ):
        """Verify champion achieves positive net savings vs Do-Nothing and vs Standard in all scenarios."""
        summary = holdout_metrics_dict["champion_performance_summary"]
        for sc in ["low", "base", "high"]:
            assert summary[sc]["net_savings_vs_do_nothing"] > 0.0
            assert summary[sc]["net_savings_vs_standard_tau05"] > 0.0
            assert summary[sc]["delay_capture_rate"] >= 0.55

    def test_cost_sensitive_budgeting_dominates_baselines_at_k10(
        self, holdout_metrics_dict: Dict[str, Any]
    ):
        """Verify COST_SENSITIVE beats VALUE_ONLY, RISK_ONLY, and STANDARD at K=10%."""
        budget_data = holdout_metrics_dict["operational_budget_evaluation"]
        for sc in ["low", "base", "high"]:
            k10 = budget_data[sc]["K_10pct"]
            cost_cs = k10["COST_SENSITIVE"]["realized_business_cost"]
            cost_val = k10["VALUE_ONLY"]["realized_business_cost"]
            cost_risk = k10["RISK_ONLY"]["realized_business_cost"]
            cost_std = k10["STANDARD"]["realized_business_cost"]

            assert cost_cs < cost_val, f"COST_SENSITIVE failed to beat VALUE_ONLY in {sc} at K=10%"
            assert cost_cs < cost_risk, f"COST_SENSITIVE failed to beat RISK_ONLY in {sc} at K=10%"
            assert cost_cs < cost_std, f"COST_SENSITIVE failed to beat STANDARD in {sc} at K=10%"
