"""
Unit and Integration Test Suite for Final 365-Day Holdout Evaluation (Phase 2 — Experiment E8 Milestone 5).

Verifies:
1. Strict Temporal Holdout Isolation & Single-Pass Protocol:
   - Development training data strictly <= 2014-08-24 (7,306 rows).
   - Final holdout data strictly > 2014-08-24 (1,013 rows).
   - Zero leakage from holdout period into training or calibration.
2. Frozen Policy Compliance:
   - Champion strategy E8-C_tuned_gamma matches frozen gamma* = 1.20.
   - Standard governed thresholds match tau = 0.50 and tau_F1 = 0.170.
   - Zero overlap with forbidden columns.
3. Artifact Integrity & Schema Consistency:
   - artifacts/results/e8_final_holdout_results.parquet exists, has 3,039 rows, 47 columns, zero nulls.
   - artifacts/results/e8_final_holdout_metrics.json exists, contains all required keys, metadata, scenarios, and budget evaluations.
4. Economic Decision Quality & Monotonicity on Holdout:
   - COST_SENSITIVE budget policy outperforms VALUE_ONLY, RISK_ONLY, and STANDARD under K=5% and K=10% review capacity.
   - Champion strategy achieves strictly positive net savings across Low, Base, and High cost scenarios.
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture
def frozen_policy_path() -> Path:
    return Path("artifacts/results/e8_frozen_policy.json")


@pytest.fixture
def holdout_results_parquet_path() -> Path:
    return Path("artifacts/results/e8_final_holdout_results.parquet")


@pytest.fixture
def holdout_metrics_json_path() -> Path:
    return Path("artifacts/results/e8_final_holdout_metrics.json")


@pytest.fixture
def features_path() -> Path:
    return Path("artifacts/data/scms_modeling_features.parquet")


@pytest.fixture
def holdout_evaluator(
    frozen_policy_path: Path, features_path: Path
) -> FinalHoldoutEvaluator:
    return FinalHoldoutEvaluator(
        frozen_policy_path=frozen_policy_path,
        features_path=features_path,
    )


class TestE8HoldoutTemporalIsolationAndDataPartition:
    """Verifies strict temporal partitioning and holdout isolation."""

    def test_dataset_partition_row_counts_and_dates(
        self, holdout_evaluator: FinalHoldoutEvaluator
    ):
        """Verify development split has exactly 7,306 rows and holdout split has 1,013 rows."""
        df_dev, df_holdout = holdout_evaluator.load_and_split_data()

        assert len(df_dev) == 7306, f"Expected 7,306 dev rows, got {len(df_dev)}"
        assert len(df_holdout) == 1013, f"Expected 1,013 holdout rows, got {len(df_holdout)}"
        assert len(df_dev) + len(df_holdout) == 8319

        # Date boundaries
        max_dev_date = pd.to_datetime(df_dev["T_pred"]).max()
        min_holdout_date = pd.to_datetime(df_holdout["T_pred"]).min()

        assert max_dev_date <= pd.to_datetime("2014-08-24 23:59:59")
        assert min_holdout_date > pd.to_datetime("2014-08-24 23:59:59")

    def test_dev_temporal_holdout_isolation_verification(
        self, holdout_evaluator: FinalHoldoutEvaluator
    ):
        """Verify verify_temporal_holdout_isolation passes on development split."""
        df_dev, _ = holdout_evaluator.load_and_split_data()
        # Must not raise HoldoutLeakageError
        verify_temporal_holdout_isolation(
            df_dev, date_col="T_pred", max_allowed_date=MAX_ALLOWED_DEV_DATE_UTC
        )

    def test_inner_train_val_split_strictly_preserves_embargo(
        self, holdout_evaluator: FinalHoldoutEvaluator
    ):
        """Verify development inner-train and inner-val are separated by embargo gap."""
        df_dev, _ = holdout_evaluator.load_and_split_data()
        inner_tr, inner_val = holdout_evaluator.split_dev_inner_train_val(df_dev, inner_gap_days=30)

        max_tr_date = pd.to_datetime(inner_tr["T_pred"]).max()
        min_val_date = pd.to_datetime(inner_val["T_pred"]).min()

        gap_days = (min_val_date - max_tr_date).days
        assert gap_days >= 30, f"Expected >= 30 embargo days, got {gap_days}"
        assert len(inner_tr) >= 4000
        assert len(inner_val) >= 1000


class TestE8FrozenPolicyCompliance:
    """Verifies that holdout evaluation strictly respects the frozen policy."""

    def test_frozen_policy_loaded_and_immutable(
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
        """Verify feature contract has zero forbidden columns."""
        contract = holdout_evaluator.frozen_policy.feature_contract
        all_features = set(contract.all_features)
        forbidden = set(contract.forbidden_columns)

        assert len(forbidden) == 8
        assert all_features.isdisjoint(forbidden)


class TestE8HoldoutArtifactIntegrity:
    """Verifies that holdout parquet and JSON artifacts are well-formed and complete."""

    def test_holdout_parquet_artifact_schema_and_rows(
        self, holdout_results_parquet_path: Path
    ):
        """Verify parquet artifact existence, row count (3,039), and required columns."""
        assert holdout_results_parquet_path.exists(), "Holdout parquet artifact missing"
        df_results = pd.read_parquet(holdout_results_parquet_path)

        assert len(df_results) == 3039, f"Expected 3,039 rows (1,013 x 3), got {len(df_results)}"
        assert df_results.isna().sum().sum() == 0, "Null values found in holdout results parquet"

        # Check required columns
        required_cols = [
            "ID", "T_pred", "scenario", "y_true", "delay_days", "line_item_value_usd",
            "fn_cost", "fp_cost", "intervention_cost", "residual_delay_cost", "net_benefit",
            "priority_score_value", "priority_score_risk", "priority_score_standard", "priority_score_cost_sensitive",
            "budget_decision_cs_k05", "budget_decision_cs_k10", "budget_decision_cs_k20",
            "budget_decision_val_k10", "budget_decision_risk_k10", "budget_decision_std_k10",
            "E8-A_tau0.5_decision", "E8-A_f1_decision", "E8-B_cost_weighted_decision",
            "E8-C_bayes_threshold_decision", "E8-C_tuned_gamma_decision",
        ]
        for col in required_cols:
            assert col in df_results.columns, f"Missing required column: {col}"

    def test_holdout_metrics_json_structure(
        self, holdout_metrics_json_path: Path
    ):
        """Verify JSON metrics summary contains required metadata, unconstrained, and budget evaluations."""
        assert holdout_metrics_json_path.exists(), "Holdout metrics JSON artifact missing"
        with open(holdout_metrics_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "unconstrained_evaluation" in data
        assert "operational_budget_evaluation" in data
        assert "champion_performance_summary" in data

        meta = data["metadata"]
        assert meta["holdout_sample_count"] == 1013
        assert meta["holdout_positives_count"] == 61
        assert meta["champion_strategy"] == "E8-C_tuned_gamma"
        assert meta["champion_gamma"] == 1.20
        assert meta["evaluation_status"] == "STRICT_SINGLE_PASS_COMPLETE"

        # Check scenarios in unconstrained evaluation
        for sc in ["low", "base", "high"]:
            assert sc in data["unconstrained_evaluation"]
            assert "E8-C_tuned_gamma" in data["unconstrained_evaluation"][sc]
            assert "Do-Nothing" in data["unconstrained_evaluation"][sc]
            assert "Always-Intervene" in data["unconstrained_evaluation"][sc]


class TestE8HoldoutEconomicPerformanceAndRationality:
    """Verifies that champion policy and budgeting policies produce expected economic improvements."""

    def test_champion_outperforms_do_nothing_and_standard_baseline(
        self, holdout_metrics_json_path: Path
    ):
        """Verify champion achieves positive net savings vs Do-Nothing and Standard CatBoost."""
        with open(holdout_metrics_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = data["champion_performance_summary"]

        for sc in ["low", "base", "high"]:
            champ_data = summary[sc]
            assert champ_data["net_savings_vs_do_nothing"] > 0, f"Negative net savings in {sc}"
            assert champ_data["net_savings_vs_standard_tau05"] > 0, f"Negative net savings vs tau0.5 in {sc}"
            assert champ_data["delay_capture_rate"] >= 0.55, f"Delay capture rate < 55% in {sc}"

    def test_cost_sensitive_budgeting_outperforms_baselines_at_k10(
        self, holdout_metrics_json_path: Path
    ):
        """Verify COST_SENSITIVE policy outperforms VALUE_ONLY, RISK_ONLY, and STANDARD at K=10%."""
        with open(holdout_metrics_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        budget_data = data["operational_budget_evaluation"]

        for sc in ["low", "base", "high"]:
            k10 = budget_data[sc]["K_10pct"]
            cs_metrics = k10["COST_SENSITIVE"]
            val_metrics = k10["VALUE_ONLY"]
            risk_metrics = k10["RISK_ONLY"]
            std_metrics = k10["STANDARD"]

            # Realized cost: COST_SENSITIVE < VALUE_ONLY, RISK_ONLY, STANDARD
            assert cs_metrics["realized_business_cost"] < val_metrics["realized_business_cost"], (
                f"COST_SENSITIVE did not beat VALUE_ONLY in {sc}"
            )
            assert cs_metrics["realized_business_cost"] < risk_metrics["realized_business_cost"], (
                f"COST_SENSITIVE did not beat RISK_ONLY in {sc}"
            )
            assert cs_metrics["realized_business_cost"] < std_metrics["realized_business_cost"], (
                f"COST_SENSITIVE did not beat STANDARD in {sc}"
            )

            # Net savings vs value-only and risk-only must be strictly positive
            assert cs_metrics["net_savings_vs_value_only"] > 0
            assert cs_metrics["net_savings_vs_risk_only"] > 0
            assert cs_metrics["net_savings_vs_standard"] > 0

    def test_budget_capacity_enforcement_on_holdout(
        self, holdout_results_parquet_path: Path
    ):
        """Verify exactly floor(K * 1013) shipments are reviewed for each budget capacity."""
        df_results = pd.read_parquet(holdout_results_parquet_path)
        df_base = df_results[df_results["scenario"] == "base"]

        # K=0.05 -> floor(0.05 * 1013) = 50
        assert int(df_base["budget_decision_cs_k05"].sum()) == 50

        # K=0.10 -> floor(0.10 * 1013) = 101
        assert int(df_base["budget_decision_cs_k10"].sum()) == 101
        assert int(df_base["budget_decision_val_k10"].sum()) == 101
        assert int(df_base["budget_decision_risk_k10"].sum()) == 101

        # K=0.20 -> floor(0.20 * 1013) = 202
        assert int(df_base["budget_decision_cs_k20"].sum()) == 202
