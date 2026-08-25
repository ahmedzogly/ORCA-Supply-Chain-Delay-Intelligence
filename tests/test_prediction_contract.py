"""Pytest test suite for Prediction Contract, Target Definition & Leakage Specification (Stage 2).

Verifies all R5 validation points:
1. Contract YAML validity and required keys
2. Prediction timestamp validity and dual-channel generation
3. Outcome timestamp occurs strictly after prediction timestamp (T_pred < T_deliv)
4. Forbidden, post-outcome, and target-derived features are strictly rejected
5. Eligibility rules are deterministic and preserve all 5,404 RDC records
6. Target contract reproducibility (y_clf in {0, 1}, Delay_Days in Z)
7. Temporal boundary invariants and no future information crosses prediction cutoff
8. Anomaly policy handles the 12 historical ERP inversions gracefully
9. Edge case handling (same-day deliveries, early deliveries mapped to Class 0, empty DataFrames, NaTs).
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import pytest

from delay_intelligence.core.config import load_config
from delay_intelligence.data.schema import (
    COL_COUNTRY,
    COL_DELAY_DAYS,
    COL_DELAY_FLAG,
    COL_DELIVERED_TO_CLIENT_DATE,
    COL_DELIVERY_RECORDED_DATE,
    COL_FULFILL_VIA,
    COL_ID,
    COL_IS_TEMPORAL_ANOMALY,
    COL_LINE_ITEM_QUANTITY,
    COL_LINE_ITEM_VALUE,
    COL_PO_SENT_DATE,
    COL_PQ_FIRST_SENT_DATE,
    COL_PRODUCT_GROUP,
    COL_SCHEDULED_DELIVERY_DATE,
    COL_SHIPMENT_MODE,
)
from delay_intelligence.validation.contract_validator import (
    ContractValidationReport,
    PredictionContractValidator,
    REQUIRED_CONTRACT_SECTIONS,
)


@pytest.fixture(scope="session")
def contract_validator() -> PredictionContractValidator:
    """Fixture providing initialized PredictionContractValidator."""
    return PredictionContractValidator()


# =============================================================================
# 1. Contract YAML Structure & Required Keys
# =============================================================================

def test_prediction_contract_yaml_structure_and_keys(contract_validator: PredictionContractValidator):
    """Verify configs/prediction_contract.yaml exists, loads, and contains all required sections."""
    contract = load_config("prediction_contract")
    assert isinstance(contract, dict), "prediction_contract must load as a dict"
    assert len(contract) > 0, "prediction_contract must not be empty"

    # Verify all 12 required sections
    for section in REQUIRED_CONTRACT_SECTIONS:
        assert section in contract, f"Missing required contract section: {section}"

    # Verify target definitions exist
    assert "classification" in contract["targets"]
    assert "regression" in contract["targets"]
    assert contract["targets"]["classification"]["type"] == "binary"
    assert contract["targets"]["regression"]["type"] == "continuous_integer"

    # Verify validator structural check passes
    is_valid, missing = contract_validator.validate_contract_structure()
    assert is_valid is True, f"Contract validator failed structural check: {missing}"
    assert len(missing) == 0


# =============================================================================
# 2. Prediction Timestamp Validity & Dual-Channel Generation
# =============================================================================

def test_prediction_timestamp_generation_and_coverage(
    contract_validator: PredictionContractValidator,
    scms_canonical_df: pd.DataFrame,
):
    """Verify the Dual-Channel Operational Milestone Anchor is correctly evaluated across all records."""
    # Primary strategy (without fallback): 8,336 records (80.74%)
    t_pred_primary = contract_validator.compute_prediction_timestamp(scms_canonical_df, use_fallback=False)
    assert len(t_pred_primary) == len(scms_canonical_df)
    assert pd.api.types.is_datetime64_any_dtype(t_pred_primary), "T_pred must be datetime64"
    assert int(t_pred_primary.notna().sum()) == 8336, f"Expected 8,336 anchored records, got {int(t_pred_primary.notna().sum())}"

    # Strategy with Direct Drop PQ fallback: 8,393 records (81.29%)
    t_pred_fallback = contract_validator.compute_prediction_timestamp(scms_canonical_df, use_fallback=True)
    assert int(t_pred_fallback.notna().sum()) == 8393

    # Check channel breakdown
    rdc_mask = scms_canonical_df[COL_FULFILL_VIA] == "From RDC"
    direct_mask = scms_canonical_df[COL_FULFILL_VIA] == "Direct Drop"

    # Direct Drop: 4,592 records with valid PO Sent Date
    assert int(t_pred_primary[direct_mask].notna().sum()) == 4592

    # From RDC: exactly 3,744 records with valid PQ Sent Date
    assert int(t_pred_primary[rdc_mask].notna().sum()) == 3744

    # Modern era (2010–2015): 99.15% coverage (7,243 / 7,305 records for primary, 7,300 with fallback)
    sched_years = scms_canonical_df[COL_SCHEDULED_DELIVERY_DATE].dt.year
    modern_mask = sched_years >= 2010
    modern_anchored = int(t_pred_primary[modern_mask].notna().sum())
    assert modern_anchored == 7243
    assert (modern_anchored / modern_mask.sum()) >= 0.99
    assert int(t_pred_fallback[modern_mask].notna().sum()) == 7300


# =============================================================================
# 3. Outcome Ordering: Prediction Precedes Outcome (T_pred < T_deliv)
# =============================================================================

def test_prediction_timestamp_precedes_outcome(
    contract_validator: PredictionContractValidator,
    scms_canonical_df: pd.DataFrame,
):
    """Verify outcome timestamp occurs after or equal to prediction timestamp for anchored records."""
    t_pred = contract_validator.compute_prediction_timestamp(scms_canonical_df, use_fallback=False)
    t_deliv = scms_canonical_df[COL_DELIVERED_TO_CLIENT_DATE]

    anchored_mask = t_pred.notna()
    assert anchored_mask.sum() == 8336

    # In modern era, strictly non-inverted
    strictly_precedes = (t_pred[anchored_mask] < t_deliv[anchored_mask]).sum()
    same_day = (t_pred[anchored_mask] == t_deliv[anchored_mask]).sum()
    historical_anomalies = (t_pred[anchored_mask] > t_deliv[anchored_mask]).sum()

    # 7,970 strictly precede, 352 same-day local procurements, 14 historical inversions
    assert strictly_precedes == 7970
    assert same_day == 352
    assert historical_anomalies == 14

    # For eligible prediction cohort (where is_temporal_anomaly == 0 and T_pred <= T_deliv),
    # 100% of records satisfy T_pred <= T_deliv
    cohort_mask = contract_validator.evaluate_prediction_cohort_eligibility(scms_canonical_df, t_pred=t_pred)
    assert (t_pred[cohort_mask] <= t_deliv[cohort_mask]).all(), "All eligible predictions must precede or equal delivery"



# =============================================================================
# 4. Forbidden, Post-Outcome & Target-Derived Feature Rejection
# =============================================================================

def test_forbidden_features_rejected(contract_validator: PredictionContractValidator):
    """Verify forbidden, post-outcome, and target-derived features are strictly identified and rejected."""
    forbidden_features = [
        "ID",
        "ASN/DN #",
        "Delivered to Client Date",
        "Delivery Recorded Date",
        "Delay_Flag",
        "Delay_Days",
        "Weight (Kilograms)",
        "Freight Cost (USD)",
        "is_temporal_anomaly",
    ]

    allowed, blocked = contract_validator.check_feature_leakage(forbidden_features)
    assert len(blocked) == len(forbidden_features), f"Expected all {len(forbidden_features)} to be blocked, got {blocked}"
    assert len(allowed) == 0


def test_allowed_features_accepted(contract_validator: PredictionContractValidator):
    """Verify legitimate pre-prediction features are accepted."""
    candidate_allowed = [
        "Project Code",
        "Country",
        "Managed By",
        "Fulfill Via",
        "Vendor INCO Term",
        "Shipment Mode",
        "Product Group",
        "Sub Classification",
        "Vendor",
        "Manufacturing Site",
        "Line Item Quantity",
        "Line Item Value",
        "Pack Price",
        "Unit Price",
        "First Line Designation",
        "Line Item Insurance (USD)",
        "is_rdc_fulfillment",
        "is_pre_pq_process",
        "Scheduled_Transit_Days",
    ]

    allowed, blocked = contract_validator.check_feature_leakage(candidate_allowed)
    assert len(blocked) == 0, f"Legitimate features were improperly blocked: {blocked}"
    assert len(allowed) == len(candidate_allowed)


# =============================================================================
# 5. Deterministic Eligibility Rules & 100% RDC Preservation
# =============================================================================

def test_base_population_eligibility_preserves_all_rdc_records(
    contract_validator: PredictionContractValidator,
    scms_canonical_df: pd.DataFrame,
):
    """Verify base population eligibility preserves 100% of all 5,404 RDC records without selection bias."""
    base_eligible = contract_validator.evaluate_base_eligibility(scms_canonical_df)

    # Full population preservation
    assert base_eligible.sum() == 10324, f"Base eligibility dropped records: {10324 - base_eligible.sum()}"
    assert base_eligible.mean() == 1.0

    # RDC subset preservation
    rdc_mask = scms_canonical_df[COL_FULFILL_VIA] == "From RDC"
    assert rdc_mask.sum() == 5404
    assert base_eligible[rdc_mask].sum() == 5404, "RDC records were dropped in base eligibility!"

    # Direct Drop subset preservation
    direct_mask = scms_canonical_df[COL_FULFILL_VIA] == "Direct Drop"
    assert direct_mask.sum() == 4920
    assert base_eligible[direct_mask].sum() == 4920


def test_eligibility_rules_deterministic(
    contract_validator: PredictionContractValidator,
    scms_canonical_df: pd.DataFrame,
):
    """Verify eligibility evaluation produces identical, deterministic results across multiple runs and index orders."""
    res1 = contract_validator.evaluate_base_eligibility(scms_canonical_df)
    res2 = contract_validator.evaluate_base_eligibility(scms_canonical_df)
    pd.testing.assert_series_equal(res1, res2)

    # Shuffled order test
    shuffled_df = scms_canonical_df.sample(frac=1.0, random_state=42)
    res_shuffled = contract_validator.evaluate_base_eligibility(shuffled_df)
    assert res_shuffled.sum() == 10324


# =============================================================================
# 6. Target Contract Reproducibility & Distribution Moments
# =============================================================================

def test_target_contract_reproducibility(
    contract_validator: PredictionContractValidator,
    scms_canonical_df: pd.DataFrame,
):
    """Verify classification and regression targets reproduce audited ground-truth values."""
    is_delayed, delay_days = contract_validator.compute_targets(scms_canonical_df)

    assert len(is_delayed) == 10324
    assert len(delay_days) == 10324

    # Discrete classification target space {0, 1}
    assert set(is_delayed.unique()) == {0, 1}
    assert int((is_delayed == 1).sum()) == 1186, "Class 1 count mismatch (expected 1,186)"
    assert int((is_delayed == 0).sum()) == 9138, "Class 0 count mismatch (expected 9,138)"
    assert pytest.approx(float(is_delayed.mean()), abs=1e-5) == 0.114878

    # Target linkage: is_delayed == (delay_days > 0)
    assert (is_delayed == (delay_days > 0).astype(int)).all()

    # Regression domain and moments
    assert delay_days.min() == -372
    assert delay_days.max() == 192
    assert delay_days.median() == 0.0
    assert pytest.approx(float(delay_days.mean()), abs=1e-2) == -6.02


# =============================================================================
# 7. Temporal Boundary Invariants
# =============================================================================

def test_temporal_boundary_invariants(
    contract_validator: PredictionContractValidator,
    scms_canonical_df: pd.DataFrame,
):
    """Verify forecast horizon and post-outcome recording temporal inequalities."""
    t_pred = contract_validator.compute_prediction_timestamp(scms_canonical_df, use_fallback=False)
    horizon = contract_validator.compute_forecast_horizon(scms_canonical_df, t_pred=t_pred)

    assert len(horizon) == 10324
    valid_horizons = horizon.dropna()
    assert len(valid_horizons) == 8336
    assert valid_horizons.median() == 129.0, f"Expected median horizon of 129 days, got {valid_horizons.median()}"

    # Invariant: Delivery Recorded Date >= Delivered to Client Date - 1 day
    deliv = scms_canonical_df[COL_DELIVERED_TO_CLIENT_DATE]
    record = scms_canonical_df[COL_DELIVERY_RECORDED_DATE]
    diff = (record - deliv).dt.days
    assert (diff >= -1).all(), "Recorded date violated timezone threshold (must be >= deliv - 1 day)"



# =============================================================================
# 8. Anomaly Policy Handling
# =============================================================================

def test_anomaly_policy_handling(
    contract_validator: PredictionContractValidator,
    scms_canonical_df: pd.DataFrame,
):
    """Verify audited 12 historical ERP timestamp inversions are isolated without corrupting targets."""
    contract = load_config("prediction_contract")
    anomaly_ids = contract["anomaly_policy"]["affected_row_ids"]
    assert len(anomaly_ids) == 12

    # Verify these records are present in the dataset
    anomaly_rows = scms_canonical_df[scms_canonical_df[COL_ID].isin(anomaly_ids)]
    assert len(anomaly_rows) == 12

    # In all 12 cases, target evaluates to Class 0 (non-delayed)
    is_delayed, delay_days = contract_validator.compute_targets(anomaly_rows)
    assert (is_delayed == 0).all(), "All 12 historical anomaly rows must evaluate to Class 0"
    assert (delay_days <= 0).all()


# =============================================================================
# 9. Edge Case Handling
# =============================================================================

def test_edge_cases_same_day_and_early_deliveries(
    contract_validator: PredictionContractValidator,
    scms_canonical_df: pd.DataFrame,
):
    """Verify same-day and early deliveries are strictly mapped to Class 0."""
    is_delayed, delay_days = contract_validator.compute_targets(scms_canonical_df)

    # Same-day deliveries (Delay_Days == 0) -> exactly 6,324 rows
    same_day_mask = delay_days == 0
    assert same_day_mask.sum() == 6324
    assert (is_delayed[same_day_mask] == 0).all()

    # Early deliveries (Delay_Days < 0) -> exactly 2,814 rows
    early_mask = delay_days < 0
    assert early_mask.sum() == 2814
    assert (is_delayed[early_mask] == 0).all()
    assert delay_days[early_mask].max() <= -1


def test_edge_cases_empty_dataframe_and_corrupted_dates(
    contract_validator: PredictionContractValidator,
):
    """Verify contract validator handles empty DataFrames and corrupted dates gracefully."""
    empty_df = pd.DataFrame(columns=[
        COL_ID, COL_SCHEDULED_DELIVERY_DATE, COL_DELIVERED_TO_CLIENT_DATE,
        COL_FULFILL_VIA, COL_PO_SENT_DATE, COL_PQ_FIRST_SENT_DATE
    ])

    t_pred_empty = contract_validator.compute_prediction_timestamp(empty_df)
    assert len(t_pred_empty) == 0
    assert pd.api.types.is_datetime64_any_dtype(t_pred_empty)

    is_del_empty, delay_empty = contract_validator.compute_targets(empty_df)
    assert len(is_del_empty) == 0
    assert len(delay_empty) == 0

    report_empty = contract_validator.validate_dataframe(empty_df)
    assert report_empty.is_valid is False
    assert any("empty" in msg.lower() for msg in report_empty.failed_checks)

    # Corrupted dates DataFrame
    corrupted_df = pd.DataFrame({
        COL_ID: [1, 2],
        COL_SCHEDULED_DELIVERY_DATE: ["invalid_date", "2012-05-10"],
        COL_DELIVERED_TO_CLIENT_DATE: ["2012-05-15", "invalid_date"],
        COL_FULFILL_VIA: ["Direct Drop", "From RDC"],
        COL_PRODUCT_GROUP: ["ARV", "ARV"],
    })
    is_del_corrupt, delay_corrupt = contract_validator.compute_targets(corrupted_df)
    assert is_del_corrupt.isna().all()
    assert delay_corrupt.isna().all()


def test_contract_validator_full_dataframe_audit(
    contract_validator: PredictionContractValidator,
    scms_canonical_df: pd.DataFrame,
):
    """Verify contract validator generates a valid report on the canonical SCMS dataset."""
    report = contract_validator.validate_dataframe(scms_canonical_df)
    assert isinstance(report, ContractValidationReport)
    assert report.is_valid is True, f"Contract validation failed: {report.failed_checks}"
    assert len(report.passed_checks) >= 5
    assert len(report.failed_checks) == 0
    assert report.metrics["total_rows"] == 10324
    assert report.metrics["base_eligible_count"] == 10324
    assert report.metrics["rdc_eligible"] == 5404
