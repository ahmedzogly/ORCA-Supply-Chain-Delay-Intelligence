"""Tests for SCMS schema validation, column presence, datatypes, and null bounds (Requirement R3)."""

import pandas as pd
import pytest

from delay_intelligence.data.schema import (
    COL_DELIVERED_TO_CLIENT_DATE,
    COL_DELIVERY_RECORDED_DATE,
    COL_DOSAGE,
    COL_FREIGHT_COST_USD,
    COL_ID,
    COL_LINE_ITEM_INSURANCE_USD,
    COL_LINE_ITEM_QUANTITY,
    COL_LINE_ITEM_VALUE,
    COL_PACK_PRICE,
    COL_PO_SENT_DATE,
    COL_PQ_FIRST_SENT_DATE,
    COL_SCHEDULED_DELIVERY_DATE,
    COL_SHIPMENT_MODE,
    COL_UNIT_OF_MEASURE,
    COL_UNIT_PRICE,
    COL_WEIGHT_KG,
    SCMS_ALLOWED_DOMAINS,
    SCMS_CRITICAL_COLUMNS,
    SCMS_RAW_COLUMNS,
)
from delay_intelligence.validation.scms_validator import SCMSValidator, ValidationReport


def test_scms_exact_column_names_and_count(scms_canonical_df: pd.DataFrame):
    """Verify all 33 original raw columns are present in the canonical DataFrame."""
    for col in SCMS_RAW_COLUMNS:
        assert col in scms_canonical_df.columns, f"Missing required column: {col}"


def test_scms_canonical_data_types(scms_canonical_df: pd.DataFrame):
    """Verify canonical column data types conform to architectural contracts."""
    assert pd.api.types.is_integer_dtype(scms_canonical_df[COL_ID]), "ID must be integer"
    assert pd.api.types.is_integer_dtype(scms_canonical_df[COL_LINE_ITEM_QUANTITY]), "Quantity must be integer"
    assert pd.api.types.is_integer_dtype(scms_canonical_df[COL_UNIT_OF_MEASURE]), "UoM must be integer"

    assert pd.api.types.is_float_dtype(scms_canonical_df[COL_LINE_ITEM_VALUE]), "Value must be float"
    assert pd.api.types.is_float_dtype(scms_canonical_df[COL_PACK_PRICE]), "Pack Price must be float"
    assert pd.api.types.is_float_dtype(scms_canonical_df[COL_UNIT_PRICE]), "Unit Price must be float"
    assert pd.api.types.is_float_dtype(scms_canonical_df[COL_LINE_ITEM_INSURANCE_USD]), "Insurance must be float"
    assert pd.api.types.is_float_dtype(scms_canonical_df[COL_WEIGHT_KG]), "Weight must be float"
    assert pd.api.types.is_float_dtype(scms_canonical_df[COL_FREIGHT_COST_USD]), "Freight must be float"

    for date_col in [
        COL_SCHEDULED_DELIVERY_DATE,
        COL_DELIVERED_TO_CLIENT_DATE,
        COL_DELIVERY_RECORDED_DATE,
        COL_PO_SENT_DATE,
        COL_PQ_FIRST_SENT_DATE,
    ]:
        assert pd.api.types.is_datetime64_any_dtype(scms_canonical_df[date_col]), f"{date_col} must be datetime64"


@pytest.mark.parametrize("critical_col", SCMS_CRITICAL_COLUMNS)
def test_scms_critical_columns_zero_nulls(
    scms_canonical_df: pd.DataFrame, critical_col: str
):
    """Verify mandatory operational columns contain zero null values."""
    assert critical_col in scms_canonical_df.columns, f"Critical column {critical_col} not in DataFrame"
    null_count = scms_canonical_df[critical_col].isna().sum()
    assert null_count == 0, f"Critical column '{critical_col}' contains {null_count} nulls (expected 0)"


def test_scms_null_tolerances_within_bounds(scms_canonical_df: pd.DataFrame):
    """Verify non-critical columns with structural missingness stay within audited bounds."""
    # Shipment Mode: exactly 360 missing (3.49% <= 5%)
    mode_nulls = scms_canonical_df[COL_SHIPMENT_MODE].isna().sum()
    assert mode_nulls == 360, f"Shipment Mode null count mismatch: {mode_nulls} != 360"
    assert mode_nulls / len(scms_canonical_df) <= 0.05

    # Dosage: exactly 1,736 missing (16.82% <= 40%)
    dosage_nulls = scms_canonical_df[COL_DOSAGE].isna().sum()
    assert dosage_nulls == 1736, f"Dosage null count mismatch: {dosage_nulls} != 1736"
    assert dosage_nulls / len(scms_canonical_df) <= 0.40

    # Line Item Insurance (USD): exactly 287 missing (2.78% <= 5%)
    ins_nulls = scms_canonical_df[COL_LINE_ITEM_INSURANCE_USD].isna().sum()
    assert ins_nulls == 287, f"Insurance null count mismatch: {ins_nulls} != 287"
    assert ins_nulls / len(scms_canonical_df) <= 0.05

    # Weight (Kilograms): exactly 3,952 non-numeric NaNs (38.28% <= 40%)
    weight_nulls = scms_canonical_df[COL_WEIGHT_KG].isna().sum()
    assert weight_nulls == 3952, f"Weight null count mismatch: {weight_nulls} != 3952"

    # Freight Cost (USD): exactly 4,126 non-numeric NaNs (39.97% <= 40%)
    freight_nulls = scms_canonical_df[COL_FREIGHT_COST_USD].isna().sum()
    assert freight_nulls == 4126, f"Freight null count mismatch: {freight_nulls} != 4126"


def test_scms_numeric_range_and_positivity(scms_canonical_df: pd.DataFrame):
    """Verify numeric values satisfy non-negativity and strictly positive quantity constraints."""
    assert (scms_canonical_df[COL_LINE_ITEM_QUANTITY] >= 1).all(), "Quantities must be >= 1"
    assert (scms_canonical_df[COL_UNIT_OF_MEASURE] >= 1).all(), "Unit of Measure must be >= 1"
    assert (scms_canonical_df[COL_LINE_ITEM_VALUE] >= 0.0).all(), "Line Item Value must be >= 0.0"
    assert (scms_canonical_df[COL_PACK_PRICE] >= 0.0).all(), "Pack Price must be >= 0.0"
    assert (scms_canonical_df[COL_UNIT_PRICE] >= 0.0).all(), "Unit Price must be >= 0.0"

    valid_ins = scms_canonical_df[COL_LINE_ITEM_INSURANCE_USD].dropna()
    assert (valid_ins >= 0.0).all(), "Insurance must be >= 0.0"

    valid_wt = scms_canonical_df[COL_WEIGHT_KG].dropna()
    assert (valid_wt >= 0.0).all(), "Weight must be >= 0.0"

    valid_fr = scms_canonical_df[COL_FREIGHT_COST_USD].dropna()
    assert (valid_fr >= 0.0).all(), "Freight must be >= 0.0"


def test_scms_categorical_domain_validity(scms_canonical_df: pd.DataFrame):
    """Verify categorical columns contain only allowed discrete levels."""
    for col, allowed_set in SCMS_ALLOWED_DOMAINS.items():
        if col in scms_canonical_df.columns:
            actual_set = set(scms_canonical_df[col].dropna().unique())
            invalid = actual_set - allowed_set
            assert not invalid, f"Column '{col}' has invalid categorical values: {invalid}"


def test_scms_validator_passes_on_valid_data(
    scms_canonical_df: pd.DataFrame, scms_enriched_df: pd.DataFrame
):
    """Verify SCMSValidator passes on standardized and enriched SCMS DataFrames."""
    validator = SCMSValidator()
    report_std = validator.validate(scms_canonical_df)
    assert isinstance(report_std, ValidationReport)
    assert report_std.is_valid is True, f"Validation failed: {report_std.failed_checks}"
    assert len(report_std.passed_checks) > 0
    assert len(report_std.failed_checks) == 0

    report_enriched = validator.validate(scms_enriched_df)
    assert report_enriched.is_valid is True
    assert len(report_enriched.failed_checks) == 0


def test_scms_validator_fails_on_corrupted_data(scms_canonical_df: pd.DataFrame):
    """Verify SCMSValidator correctly catches and fails on corrupted data."""
    validator = SCMSValidator()

    # Corruption 1: Duplicate primary keys
    corrupted_df = scms_canonical_df.copy()
    corrupted_df.loc[1, COL_ID] = corrupted_df.loc[0, COL_ID]
    report_dup = validator.validate(corrupted_df)
    assert report_dup.is_valid is False
    assert any("duplicate" in msg.lower() for msg in report_dup.failed_checks)

    # Corruption 2: Negative prices
    corrupted_price = scms_canonical_df.copy()
    corrupted_price.loc[0, COL_UNIT_PRICE] = -50.0
    report_price = validator.validate(corrupted_price)
    assert report_price.is_valid is False
    assert any("unit price" in msg.lower() for msg in report_price.failed_checks)
