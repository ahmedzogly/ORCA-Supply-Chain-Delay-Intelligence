"""
Unit and integration tests for Milestone 5: Single-Pass Final Holdout Evaluation & Invariance Verification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

from delay_intelligence.counterfactual.evaluator import CounterfactualEvaluator
from delay_intelligence.counterfactual.provenance import ProvenanceTag


@pytest.fixture(scope="module")
def evaluator() -> CounterfactualEvaluator:
    return CounterfactualEvaluator()


@pytest.fixture(scope="module")
def holdout_df() -> pd.DataFrame:
    path = Path("artifacts/phase2/e10/e10_holdout_evaluation_results.parquet")
    if not path.exists():
        evaluator = CounterfactualEvaluator()
        evaluator.run_holdout_evaluation()
    return pd.read_parquet(path)


def test_holdout_cohort_loading_and_quarantine(evaluator: CounterfactualEvaluator):
    """
    Verifies that the final holdout cohort contains exactly 1,013 rows
    with T_pred strictly > 2014-08-24.
    """
    df_holdout = evaluator.load_holdout_data()
    assert len(df_holdout) == 1013
    assert df_holdout["T_pred"].min() > pd.Timestamp("2014-08-24")
    assert df_holdout["T_pred"].max() <= pd.Timestamp("2015-09-30")


def test_holdout_evaluation_parquet_structure(holdout_df: pd.DataFrame):
    """
    Verifies holdout evaluation results parquet structure, row counts, and schema.
    Total expected rows: 1,013 shipments * 7 policies * 3 scenarios = 21,273 rows.
    """
    assert len(holdout_df) == 1013 * 7 * 3  # 21,273

    expected_cols = [
        "fold_id",
        "scenario",
        "policy_id",
        "policy_name",
        "shipment_id",
        "pred_date",
        "action_selected",
        "action_cost",
        "residual_delay_days",
        "residual_delay_prob",
        "residual_delay_cost",
        "residual_risk_cost",
        "expected_realized_cost",
        "no_action_cost",
        "net_benefit",
        "oracle_cost",
        "oracle_action",
        "policy_regret",
        "hysteresis_stable",
        "provenance_tag",
    ]
    for col in expected_cols:
        assert col in holdout_df.columns

    assert set(holdout_df["scenario"].unique()) == {"low", "base", "high"}
    assert set(holdout_df["policy_id"].unique()) == {"P0", "P1", "P2", "P3", "P4", "P5", "Oracle"}
    assert (holdout_df["provenance_tag"] == ProvenanceTag.SIMULATED_COUNTERFACTUAL.value).all()


def test_holdout_oracle_properties(holdout_df: pd.DataFrame):
    """
    Verifies that Oracle regret is identically zero and all operational policies
    have non-negative regret across all holdout records.
    """
    oracle_records = holdout_df[holdout_df["policy_id"] == "Oracle"]
    assert (oracle_records["policy_regret"] == 0.0).all()

    # All policy regrets must be non-negative
    assert (holdout_df["policy_regret"] >= 0.0).all()

    # Oracle expected realized cost must be <= P0 expected realized cost
    assert (oracle_records["expected_realized_cost"] <= oracle_records["no_action_cost"] + 1e-6).all()


def test_holdout_budget_allocation_constraints(evaluator: CounterfactualEvaluator):
    """
    Verifies that review budget allocations strictly obey operational capacity bounds:
    - 5% capacity: allocated <= 50 shipments (floor(0.05 * 1013))
    - 10% capacity: allocated <= 101 shipments (floor(0.10 * 1013))
    - 20% capacity: allocated <= 202 shipments (floor(0.20 * 1013))
    """
    summary = evaluator.run_holdout_evaluation()
    budget_dict = summary["budget_summary"]

    for sc_name in ["low", "base", "high"]:
        k5 = budget_dict[sc_name]["k_5pct"]
        k10 = budget_dict[sc_name]["k_10pct"]
        k20 = budget_dict[sc_name]["k_20pct"]

        assert k5["allocated_count"] <= 50
        assert k10["allocated_count"] <= 101
        assert k20["allocated_count"] <= 202

        assert k5["total_net_benefit_usd"] >= 0.0
        assert k10["total_net_benefit_usd"] >= k5["total_net_benefit_usd"] - 1e-6
        assert k20["total_net_benefit_usd"] >= k10["total_net_benefit_usd"] - 1e-6


def test_post_holdout_invariance_manifest_matches_pre_freeze():
    """
    Verifies that the post-holdout manifest exists and matches the pre-freeze manifest
    with 100% bitwise invariance across all 36 baseline artifacts.
    """
    pre_path = Path("artifacts/phase2/e10/e10_pre_freeze_manifest.json")
    post_path = Path("artifacts/phase2/e10/e10_post_holdout_manifest.json")

    assert pre_path.exists(), "Pre-freeze manifest missing!"
    assert post_path.exists(), "Post-holdout manifest missing!"

    with open(pre_path, "r", encoding="utf-8") as f:
        pre_data = json.load(f)

    with open(post_path, "r", encoding="utf-8") as f:
        post_data = json.load(f)

    pre_artifacts = pre_data.get("artifacts", {})
    post_artifacts = post_data.get("artifacts", {})

    assert len(pre_artifacts) == 36, f"Expected 36 pre-freeze artifacts, found {len(pre_artifacts)}"
    assert len(post_artifacts) == 36, f"Expected 36 post-holdout artifacts, found {len(post_artifacts)}"

    for path, pre_entry in pre_artifacts.items():
        assert path in post_artifacts, f"Artifact {path} missing in post-holdout manifest!"
        post_entry = post_artifacts[path]
        assert pre_entry["sha256"] == post_entry["sha256"], f"SHA-256 mismatch for {path}!"
        assert pre_entry["size_bytes"] == post_entry["size_bytes"], f"Size mismatch for {path}!"
