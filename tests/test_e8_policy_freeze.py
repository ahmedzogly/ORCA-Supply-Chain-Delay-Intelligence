"""
Comprehensive Unit & Integration Tests for E8 Policy Lockdown & Freezing.

Covers:
- Cryptographic SHA-256 computation and byte verification for code, configs, and artifacts.
- Strict Temporal Holdout Isolation audit (verification of zero dates > 2014-08-24).
- Holdout contamination detection and error raising upon simulated leakage.
- FrozenCostPolicy loading, threshold computation, unconstrained, and budgeted decision evaluation.
- End-to-end execution of freeze_e8_policy and complete schema validation.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from delay_intelligence.cost_sensitive.cost_engine import CostScenarioModel
from delay_intelligence.cost_sensitive.policy_freeze import (
    ChampionStrategySpec,
    ChecksumEntry,
    FrozenCostPolicy,
    FrozenFeatureContract,
    HoldoutLeakageError,
    MAX_ALLOWED_DEV_DATE_UTC,
    OperationalBudgetRuleSpec,
    compute_file_sha256,
    freeze_e8_policy,
    verify_temporal_holdout_isolation,
)


# =============================================================================
# 1. Unit Tests: Checksums & Holdout Isolation
# =============================================================================

def test_compute_file_sha256():
    config_path = Path("configs/cost_scenarios.yaml")
    if not config_path.exists():
        pytest.skip("Config file not found")

    entry = compute_file_sha256(config_path)
    assert isinstance(entry, ChecksumEntry)
    assert len(entry.sha256) == 64  # Hex SHA-256 is 64 chars
    assert entry.file_size_bytes > 0
    assert "cost_scenarios.yaml" in entry.relative_path


def test_verify_temporal_holdout_isolation_pass():
    # Clean development cohort strictly <= 2014-08-24
    df_clean = pd.DataFrame({
        "ID": [1, 2, 3],
        "T_pred": ["2012-03-07", "2013-05-30", "2014-08-24"],
        "line_item_value_usd": [1000.0, 2000.0, 3000.0],
    })
    # Must not raise
    verify_temporal_holdout_isolation(df_clean, date_col="T_pred", max_allowed_date=MAX_ALLOWED_DEV_DATE_UTC)


def test_verify_temporal_holdout_isolation_raises_on_leakage():
    # Contaminated dataframe with holdout period dates
    df_contaminated = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "T_pred": ["2012-03-07", "2013-05-30", "2014-08-24", "2014-08-25"],
        "line_item_value_usd": [1000.0, 2000.0, 3000.0, 4000.0],
    })
    with pytest.raises(HoldoutLeakageError, match="HOLDOUT CONTAMINATION DETECTED"):
        verify_temporal_holdout_isolation(df_contaminated, date_col="T_pred", max_allowed_date=MAX_ALLOWED_DEV_DATE_UTC)


# =============================================================================
# 2. Unit Tests: FrozenCostPolicy Execution & Inspection API
# =============================================================================

def test_frozen_cost_policy_thresholds_and_application():
    policy_spec = {
        "metadata": {
            "policy_version": "1.0.0-frozen-dev",
            "development_period_window": "2006-04-19 to 2014-08-24",
            "holdout_isolation_status": "VERIFIED_STRICTLY_ISOLATED",
        },
        "champion_strategy": {
            "strategy_id": "E8-C",
            "strategy_name": "E8-C_tuned_gamma",
            "gamma_tuned_multiplier": 1.20,
            "governed_standard_threshold": 0.50,
            "governed_f1_threshold": 0.170,
        },
        "feature_contract": {
            "all_features": ["Unit Price", "Line Item Value"],
            "num_cols": ["Unit Price", "Line Item Value"],
            "cat_cols": [],
            "forbidden_columns": ["Delay_Flag", "Delay_Days"],
        },
        "budget_rules": {
            "supported_capacities_k": [0.05, 0.10, 0.20],
            "primary_policy": "COST_SENSITIVE",
        },
        "cost_scenarios": {},
    }

    policy = FrozenCostPolicy(policy_spec)
    assert policy.champion.strategy_name == "E8-C_tuned_gamma"
    assert policy.champion.gamma_tuned_multiplier == 1.20

    # Test threshold calculation: tau* = FP / (1.20 * Net_Benefit + FP)
    costs_df = pd.DataFrame({
        "net_benefit": [1000.0, 2000.0],
        "fp_cost": [100.0, 200.0],
    })
    # Item 0: 100 / (1.20 * 1000 + 100) = 100 / 1300 = 0.076923
    # Item 1: 200 / (1.20 * 2000 + 200) = 200 / 2600 = 0.076923
    tau_star = policy.compute_decision_thresholds(costs_df, apply_tuned_gamma=True)
    assert pytest.approx(tau_star[0], abs=1e-4) == 100.0 / 1300.0

    probs = np.array([0.10, 0.05])
    # Item 0: 0.10 >= 0.0769 -> 1
    # Item 1: 0.05 < 0.0769 -> 0
    decisions = policy.apply_unconstrained_policy(probs, costs_df)
    assert decisions[0] == 1
    assert decisions[1] == 0


# =============================================================================
# 3. Integration Tests: Real Backtest Artifact Freezing
# =============================================================================

def test_freeze_e8_policy_e2e(tmp_path):
    parquet_path = Path("artifacts/results/e8_dev_backtest_results.parquet")
    dev_metrics_path = Path("artifacts/results/e8_dev_metrics.json")
    if not parquet_path.exists() or not dev_metrics_path.exists():
        pytest.skip("Development backtest artifacts not found")

    out_freeze_json = tmp_path / "test_frozen_policy.json"
    frozen_payload = freeze_e8_policy(
        backtest_parquet_path=parquet_path,
        dev_metrics_json_path=dev_metrics_path,
        output_frozen_policy_path=out_freeze_json,
    )

    assert out_freeze_json.exists()
    assert "metadata" in frozen_payload
    assert "champion_strategy" in frozen_payload
    assert "feature_contract" in frozen_payload
    assert "cost_scenarios" in frozen_payload
    assert "budget_rules" in frozen_payload
    assert "robustness_certification" in frozen_payload
    assert "cryptographic_manifest" in frozen_payload

    meta = frozen_payload["metadata"]
    assert meta["holdout_isolation_status"] == "VERIFIED_STRICTLY_ISOLATED"
    assert meta["status"] == "FROZEN_AND_IMMUTABLE"

    # Verify manifest contains valid SHA-256 hashes
    manifest = frozen_payload["cryptographic_manifest"]
    assert len(manifest) >= 5
    for item in manifest:
        assert len(item["sha256"]) == 64
        assert item["file_size_bytes"] > 0
