"""Adversarial stress-testing suite for SCMS ingestion adapter and validator.

Tests boundary conditions, corrupted/malformed inputs, schema permutations,
mixed-type fields, extreme values, concurrency, and immutability.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest

from delay_intelligence.data.adapters.scms import SCMSAdapter
from delay_intelligence.data.schema import (
    COL_DELAY_DAYS,
    COL_DELAY_FLAG,
    COL_FIRST_LINE_DESIGNATION,
    COL_FREIGHT_COST_USD,
    COL_FREIGHT_IS_NUMERIC,
    COL_FULFILL_VIA,
    COL_ID,
    COL_LINE_ITEM_QUANTITY,
    COL_LINE_ITEM_VALUE,
    COL_MANAGED_BY,
    COL_PO_SENT_DATE,
    COL_PO_SENT_IS_DATE,
    COL_PQ_FIRST_SENT_DATE,
    COL_PQ_FIRST_SENT_IS_DATE,
    COL_PRODUCT_GROUP,
    COL_SCHEDULED_DELIVERY_DATE,
    COL_SHIPMENT_MODE,
    COL_UNIT_OF_MEASURE,
    COL_WEIGHT_IS_NUMERIC,
    COL_WEIGHT_KG,
    SCMS_EXPECTED_ROW_COUNT,
    SCMS_RAW_COLUMNS,
)
from delay_intelligence.validation.scms_validator import SCMSValidator, ValidationReport


# =============================================================================
# Helper Fixtures & Builders
# =============================================================================

@pytest.fixture
def minimal_valid_scms_row() -> Dict[str, str]:
    """Provide a minimal single valid row matching raw SCMS CSV string format."""
    return {
        "ID": "1",
        "Project Code": "100-PK-T01",
        "PQ #": "Pre-PQ Process",
        "PO / SO #": "SCMS-1",
        "ASN/DN #": "ASN-1",
        "Country": "Cote d'Ivoire",
        "Managed By": "PMO - US",
        "Fulfill Via": "Direct Drop",
        "Vendor INCO Term": "EXW",
        "Shipment Mode": "Air",
        "PQ First Sent to Client Date": "Pre-PQ Process",
        "PO Sent to Vendor Date": "Date Needed Delay",
        "Scheduled Delivery Date": "02-Jun-06",
        "Delivered to Client Date": "02-Jun-06",
        "Delivery Recorded Date": "02-Jun-06",
        "Product Group": "ARV",
        "Sub Classification": "Adult",
        "Vendor": "RANBAXY Laboratories Ltd.",
        "Item Description": "Efavirenz 600mg, 30 Tabs",
        "Molecule/Test Type": "Efavirenz",
        "Brand": "Generic",
        "Dosage": "600mg",
        "Dosage Form": "Tablet",
        "Unit of Measure (Per Pack)": "30",
        "Line Item Quantity": "1000",
        "Line Item Value": "10000.0",
        "Pack Price": "10.0",
        "Unit Price": "0.33",
        "Manufacturing Site": "Ranbaxy Paonta",
        "First Line Designation": "Yes",
        "Weight (Kilograms)": "100",
        "Freight Cost (USD)": "1500.50",
        "Line Item Insurance (USD)": "15.0",
    }


def make_raw_df_from_rows(rows: List[Dict[str, str]]) -> pd.DataFrame:
    """Construct DataFrame with all 33 SCMS raw columns from partial row dicts."""
    df = pd.DataFrame(rows)
    for col in SCMS_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = "N/A"
    return df[SCMS_RAW_COLUMNS]


# =============================================================================
# 1. Malformed / Corrupted Dates and Non-Standard Date Formats
# =============================================================================

class TestAdversarialDateHandling:
    """Stress-test date parsing against malformed, corrupted, and non-standard formats."""

    @pytest.mark.parametrize(
        "bad_date",
        [
            "2020-99-99",
            "INVALID_DATE",
            "UNKNOWN",
            "00-00-00",
            "32-Dec-11",
            "2012/13/40",
            "30-Feb-11",
            "",
            "   ",
            "None",
            "N/A",
            "2011-05-12",  # ISO format vs expected %d-%b-%y
            "05/12/2011",  # US format vs expected %d-%b-%y
            "12-31-2011",
        ],
    )
    def test_milestone_dates_malformed_string_coercion(
        self, scms_adapter: SCMSAdapter, minimal_valid_scms_row: Dict[str, str], bad_date: str
    ):
        """Verify adapter safely coerces malformed milestone dates to NaT without crashing."""
        row = minimal_valid_scms_row.copy()
        row["Scheduled Delivery Date"] = bad_date
        df_raw = make_raw_df_from_rows([row])

        df_std = scms_adapter.standardize_schema(df_raw)
        assert pd.isna(df_std.loc[0, COL_SCHEDULED_DELIVERY_DATE]), (
            f"Expected NaT for malformed date '{bad_date}', got {df_std.loc[0, COL_SCHEDULED_DELIVERY_DATE]}"
        )

    def test_milestone_dates_nat_validator_rejection(
        self, scms_adapter: SCMSAdapter, minimal_valid_scms_row: Dict[str, str]
    ):
        """Verify validator flags failure when milestone date contains NaT."""
        row = minimal_valid_scms_row.copy()
        row["Scheduled Delivery Date"] = "CORRUPTED_DATE"
        df_raw = make_raw_df_from_rows([row])

        df_std = scms_adapter.standardize_schema(df_raw)
        validator = SCMSValidator()
        report = validator.validate(df_std)

        assert report.is_valid is False
        assert any("nat" in msg.lower() for msg in report.failed_checks)

    @pytest.mark.parametrize(
        "procurement_sentinel",
        [
            "Date Needed Delay",
            "Pre-PQ Process",
            "From RDC",
            "Invoiced Separately",
            "N/A",
            "Unknown",
            "See Note 42",
            "",
        ],
    )
    def test_procurement_dates_sentinels_coercion(
        self, scms_adapter: SCMSAdapter, minimal_valid_scms_row: Dict[str, str], procurement_sentinel: str
    ):
        """Verify procurement dates with diverse text sentinels coerce to NaT and flag indicator properly."""
        row = minimal_valid_scms_row.copy()
        row["PO Sent to Vendor Date"] = procurement_sentinel
        row["PQ First Sent to Client Date"] = procurement_sentinel
        df_raw = make_raw_df_from_rows([row])

        df_std = scms_adapter.standardize_schema(df_raw)
        assert pd.isna(df_std.loc[0, COL_PO_SENT_DATE])
        assert df_std.loc[0, COL_PO_SENT_IS_DATE] == 0
        assert pd.isna(df_std.loc[0, COL_PQ_FIRST_SENT_DATE])
        assert df_std.loc[0, COL_PQ_FIRST_SENT_IS_DATE] == 0

    def test_temporal_features_safe_handling_when_dates_are_nat(
        self, scms_adapter: SCMSAdapter, minimal_valid_scms_row: Dict[str, str]
    ):
        """Verify extract_temporal_features safely handles NaT dates without raising IntCastingNaNError."""
        row = minimal_valid_scms_row.copy()
        row["Delivered to Client Date"] = "CORRUPTED_DELIVERY_DATE"
        df_raw = make_raw_df_from_rows([row])
        df_std = scms_adapter.standardize_schema(df_raw)

        df_enriched = scms_adapter.extract_temporal_features(df_std)
        assert pd.isna(df_enriched.loc[0, COL_DELAY_DAYS])
        assert pd.isna(df_enriched.loc[0, COL_DELAY_FLAG])


# =============================================================================
# 2. Missing Columns and Unexpected Column Permutations
# =============================================================================

class TestAdversarialSchemaPermutations:
    """Stress-test adapter and validator against missing columns, extra columns, and column reordering."""

    @pytest.mark.parametrize(
        "missing_col",
        [
            "Project Code",
            "Country",
            "Scheduled Delivery Date",
            "Delivered to Client Date",
            "Delivery Recorded Date",
            "Line Item Quantity",
            "Line Item Value",
        ],
    )
    def test_validator_fails_gracefully_on_missing_critical_columns(
        self, scms_canonical_df: pd.DataFrame, missing_col: str
    ):
        """Verify validator flags failure without crashing when critical columns (except ID) are missing."""
        df_corrupted = scms_canonical_df.drop(columns=[missing_col])
        validator = SCMSValidator()
        report = validator.validate(df_corrupted)

        assert report.is_valid is False
        assert any(missing_col in msg or "Missing required" in msg for msg in report.failed_checks)

    def test_validator_fails_gracefully_on_missing_id_column(self, scms_canonical_df: pd.DataFrame):
        """Verify SCMSValidator returns is_valid=False without crashing with KeyError when ID column is missing."""
        df_corrupted = scms_canonical_df.drop(columns=[COL_ID])
        validator = SCMSValidator()
        report = validator.validate(df_corrupted)

        assert report.is_valid is False
        assert any("ID" in msg or "Primary key" in msg for msg in report.failed_checks)

    def test_adapter_and_validator_with_extra_columns(
        self, scms_adapter: SCMSAdapter, minimal_valid_scms_row: Dict[str, str]
    ):
        """Verify adapter and validator handle datasets with extra unexpected columns."""
        row = minimal_valid_scms_row.copy()
        row["__unexpected_debug_col__"] = "EXTRA_VALUE"
        row["malicious_payload"] = "<script>alert(1)</script>"
        df_raw = pd.DataFrame([row])
        for col in SCMS_RAW_COLUMNS:
            if col not in df_raw.columns:
                df_raw[col] = "Valid"

        df_std = scms_adapter.standardize_schema(df_raw)
        assert "__unexpected_debug_col__" in df_std.columns
        assert df_std.loc[0, "__unexpected_debug_col__"] == "EXTRA_VALUE"

        validator = SCMSValidator()
        report = validator.validate(df_std)
        assert report.is_valid is True

    def test_adapter_and_validator_with_reversed_column_order(
        self, scms_adapter: SCMSAdapter, minimal_valid_scms_row: Dict[str, str]
    ):
        """Verify adapter and validator are agnostic to column ordering."""
        df_raw = make_raw_df_from_rows([minimal_valid_scms_row])
        reversed_cols = list(reversed(df_raw.columns))
        df_reversed = df_raw[reversed_cols]

        df_std = scms_adapter.standardize_schema(df_reversed)
        df_enriched = scms_adapter.extract_temporal_features(df_std)

        validator = SCMSValidator()
        report = validator.validate(df_enriched)
        assert report.is_valid is True


# =============================================================================
# 3. Mixed Text / Numeric Strings in Numeric Fields
# =============================================================================

class TestAdversarialMixedNumericFields:
    """Stress-test mixed strings, currencies, sentinels, and anomalies in numeric columns."""

    @pytest.mark.parametrize(
        "raw_weight, expected_numeric_flag, is_nan_value",
        [
            ("100", 1, False),
            ("1234.56", 1, False),
            ("0", 1, False),
            ("Weight Captured Separately", 0, True),
            ("See Line Item 5", 0, True),
            ("Freight Included in Price", 0, True),
            ("N/A", 0, True),
            ("TBD", 0, True),
            ("approx 500kg", 0, True),
            ("$1,234.50", 0, True),
            ("", 0, True),
            ("   ", 0, True),
            ("-50.5", 1, False),
        ],
    )
    def test_weight_mixed_string_coercion_and_indicator(
        self,
        scms_adapter: SCMSAdapter,
        minimal_valid_scms_row: Dict[str, str],
        raw_weight: str,
        expected_numeric_flag: int,
        is_nan_value: bool,
    ):
        """Verify weight field correctly sets indicator flag and coerces text to NaN."""
        row = minimal_valid_scms_row.copy()
        row["Weight (Kilograms)"] = raw_weight
        df_raw = make_raw_df_from_rows([row])

        df_std = scms_adapter.standardize_schema(df_raw)
        assert df_std.loc[0, COL_WEIGHT_IS_NUMERIC] == expected_numeric_flag
        if is_nan_value:
            assert pd.isna(df_std.loc[0, COL_WEIGHT_KG])
        else:
            assert not pd.isna(df_std.loc[0, COL_WEIGHT_KG])
            assert np.isclose(df_std.loc[0, COL_WEIGHT_KG], float(raw_weight))

    @pytest.mark.parametrize(
        "raw_freight, expected_numeric_flag, is_nan_value",
        [
            ("1500.50", 1, False),
            ("0.0", 1, False),
            ("Invoiced Separately", 0, True),
            ("Freight Included in Price", 0, True),
            ("See Note 12", 0, True),
            ("FREE", 0, True),
            ("N/A", 0, True),
            ("$500.00", 0, True),
            ("EUR 400", 0, True),
            ("", 0, True),
        ],
    )
    def test_freight_cost_mixed_string_coercion_and_indicator(
        self,
        scms_adapter: SCMSAdapter,
        minimal_valid_scms_row: Dict[str, str],
        raw_freight: str,
        expected_numeric_flag: int,
        is_nan_value: bool,
    ):
        """Verify freight cost field correctly sets indicator flag and coerces text to NaN."""
        row = minimal_valid_scms_row.copy()
        row["Freight Cost (USD)"] = raw_freight
        df_raw = make_raw_df_from_rows([row])

        df_std = scms_adapter.standardize_schema(df_raw)
        assert df_std.loc[0, COL_FREIGHT_IS_NUMERIC] == expected_numeric_flag
        if is_nan_value:
            assert pd.isna(df_std.loc[0, COL_FREIGHT_COST_USD])
        else:
            assert not pd.isna(df_std.loc[0, COL_FREIGHT_COST_USD])
            assert np.isclose(df_std.loc[0, COL_FREIGHT_COST_USD], float(raw_freight))

    def test_strictly_integer_columns_invalid_values_coercion(
        self, scms_adapter: SCMSAdapter, minimal_valid_scms_row: Dict[str, str]
    ):
        """Verify non-numeric text in quantity and UoM columns gracefully coerce to default integers."""
        row = minimal_valid_scms_row.copy()
        row["Line Item Quantity"] = "Ten Thousand"
        row["Unit of Measure (Per Pack)"] = "Pack of 30"
        df_raw = make_raw_df_from_rows([row])

        df_std = scms_adapter.standardize_schema(df_raw)
        assert df_std.loc[0, COL_LINE_ITEM_QUANTITY] == 0
        assert df_std.loc[0, COL_UNIT_OF_MEASURE] == 1


# =============================================================================
# 4. Boundary Values, Zero Cases, and Extreme Scenarios
# =============================================================================

class TestAdversarialBoundaries:
    """Stress-test boundary values: empty DataFrame, single-row, extreme ranges, negative lead times."""

    def test_validator_on_empty_dataframe(self):
        """Verify validator handles a completely empty DataFrame (0 rows) gracefully."""
        empty_df = pd.DataFrame()
        validator = SCMSValidator()
        report = validator.validate(empty_df)

        assert isinstance(report, ValidationReport)
        assert report.is_valid is False
        assert any("empty" in msg.lower() for msg in report.failed_checks)

    def test_validator_on_empty_schema_dataframe(self):
        """Verify validator handles an empty DataFrame with all 33 column headers but 0 rows."""
        empty_df = pd.DataFrame(columns=SCMS_RAW_COLUMNS)
        validator = SCMSValidator()
        report = validator.validate(empty_df)

        assert isinstance(report, ValidationReport)
        assert report.is_valid is False
        assert any("empty" in msg.lower() for msg in report.failed_checks)

    def test_validator_handles_empty_dataframe_with_delay_days_column(self):
        """Verify SCMSValidator handles empty DataFrame containing Delay_Days without crashing."""
        empty_enriched_df = pd.DataFrame(columns=SCMS_RAW_COLUMNS + [COL_DELAY_DAYS, COL_DELAY_FLAG])
        validator = SCMSValidator()
        report = validator.validate(empty_enriched_df)

        assert isinstance(report, ValidationReport)
        assert report.is_valid is False
        assert any("empty" in msg.lower() for msg in report.failed_checks)

    def test_single_row_dataframe_full_lifecycle(
        self, scms_adapter: SCMSAdapter, minimal_valid_scms_row: Dict[str, str]
    ):
        """Verify end-to-end ingestion and validation pipeline works for N=1 row dataset."""
        df_raw = make_raw_df_from_rows([minimal_valid_scms_row])
        df_std = scms_adapter.standardize_schema(df_raw)
        df_enriched = scms_adapter.extract_temporal_features(df_std)

        validator = SCMSValidator()
        report = validator.validate(df_enriched)

        assert report.is_valid is True
        assert report.metrics["row_count"] == 1
        assert len(report.failed_checks) == 0

    def test_extreme_numeric_values_handling(
        self, scms_adapter: SCMSAdapter, minimal_valid_scms_row: Dict[str, str]
    ):
        """Verify extreme numbers (e.g. 10^12) do not cause overflow crashes."""
        row = minimal_valid_scms_row.copy()
        row["Line Item Quantity"] = "999999999999"
        row["Line Item Value"] = "999999999999999.9"
        row["Pack Price"] = "1000000000.0"
        df_raw = make_raw_df_from_rows([row])

        df_std = scms_adapter.standardize_schema(df_raw)
        assert df_std.loc[0, COL_LINE_ITEM_QUANTITY] == 999999999999
        assert df_std.loc[0, COL_LINE_ITEM_VALUE] > 1e14

        validator = SCMSValidator()
        report = validator.validate(df_std)
        assert report.is_valid is True

    def test_negative_delay_and_extreme_delay_days(
        self, scms_adapter: SCMSAdapter, minimal_valid_scms_row: Dict[str, str]
    ):
        """Verify extreme positive and negative delays compute correctly without crashes."""
        row_early = minimal_valid_scms_row.copy()
        row_early["ID"] = "1"
        row_early["Scheduled Delivery Date"] = "01-Jan-11"
        row_early["Delivered to Client Date"] = "01-Jan-10"

        row_late = minimal_valid_scms_row.copy()
        row_late["ID"] = "2"
        row_late["Scheduled Delivery Date"] = "01-Jan-10"
        row_late["Delivered to Client Date"] = "01-Jan-12"

        df_raw = make_raw_df_from_rows([row_early, row_late])
        df_std = scms_adapter.standardize_schema(df_raw)
        df_enriched = scms_adapter.extract_temporal_features(df_std)

        assert df_enriched.loc[0, COL_DELAY_DAYS] == -365
        assert df_enriched.loc[0, COL_DELAY_FLAG] == 0

        assert df_enriched.loc[1, COL_DELAY_DAYS] == 730
        assert df_enriched.loc[1, COL_DELAY_FLAG] == 1

    @pytest.mark.parametrize(
        "cat_col, invalid_value",
        [
            (COL_SHIPMENT_MODE, "Teleportation"),
            (COL_SHIPMENT_MODE, "Spacecraft"),
            (COL_FULFILL_VIA, "Drone Direct"),
            (COL_PRODUCT_GROUP, "UNKNOWN_VACCINE"),
            (COL_FIRST_LINE_DESIGNATION, "Maybe"),
            (COL_MANAGED_BY, "Alien Field Office"),
        ],
    )
    def test_validator_rejects_unseen_categorical_levels(
        self,
        scms_adapter: SCMSAdapter,
        minimal_valid_scms_row: Dict[str, str],
        cat_col: str,
        invalid_value: str,
    ):
        """Verify validator flags any categorical values outside the strict allowed domains."""
        row = minimal_valid_scms_row.copy()
        row[cat_col] = invalid_value
        df_raw = make_raw_df_from_rows([row])

        df_std = scms_adapter.standardize_schema(df_raw)
        validator = SCMSValidator()
        report = validator.validate(df_std)

        assert report.is_valid is False
        assert any(cat_col in msg or "invalid values" in msg for msg in report.failed_checks)


# =============================================================================
# 5. Ingestion Immutability, Concurrency, and Repeatability
# =============================================================================

class TestAdversarialConcurrencyAndImmutability:
    """Stress-test multi-threaded concurrency, raw file immutability, and state isolation."""

    def test_concurrent_multi_threaded_ingestion(self, scms_raw_path: Path, scms_expected_sha256: str):
        """Verify 10 concurrent threads can safely instantiate adapters and ingest data without collision."""
        def worker_task(thread_id: int) -> Dict[str, Any]:
            adapter = SCMSAdapter(data_path=scms_raw_path)
            raw = adapter.load_raw()
            std = adapter.standardize_schema(raw)
            enriched = adapter.extract_temporal_features(std)
            validator = SCMSValidator()
            report = validator.validate(enriched)
            return {
                "thread_id": thread_id,
                "row_count": len(enriched),
                "is_valid": report.is_valid,
                "sha256": adapter.raw_sha256,
            }

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_task, i) for i in range(10)]
            results = [f.result() for f in futures]

        assert len(results) == 10
        for res in results:
            assert res["row_count"] == SCMS_EXPECTED_ROW_COUNT
            assert res["is_valid"] is True
            assert res["sha256"] == scms_expected_sha256

    def test_in_place_mutation_isolation(self, scms_adapter: SCMSAdapter):
        """Verify mutating an ingested DataFrame does not pollute subsequent load operations."""
        df_raw_1 = scms_adapter.load_raw()
        original_shape = df_raw_1.shape
        original_val = df_raw_1.iloc[0, 0]

        # Mutate df_raw_1 in place
        df_raw_1.iloc[0, 0] = "MUTATED"
        df_raw_1.drop(columns=[COL_ID], inplace=True)

        # Load fresh copy
        df_raw_2 = scms_adapter.load_raw()
        assert df_raw_2.shape == original_shape
        assert df_raw_2.iloc[0, 0] == original_val
        assert COL_ID in df_raw_2.columns

    def test_raw_csv_checksum_stability_across_50_runs(
        self, scms_raw_path: Path, scms_expected_sha256: str, scms_adapter: SCMSAdapter
    ):
        """Verify repeatedly running the adapter 50 times produces identical hashes and zero file changes."""
        for _ in range(50):
            _ = scms_adapter.load_raw()

        hasher = hashlib.sha256()
        with open(scms_raw_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        final_hash = hasher.hexdigest()

        assert final_hash == scms_expected_sha256


# =============================================================================
# 6. High-Throughput & Scale Stress Testing
# =============================================================================

class TestAdversarialScaleStress:
    """Stress-test ingestion adapter and validator under synthetic high volume (50,000 rows)."""

    def test_high_volume_synthetic_dataset_throughput(
        self, scms_adapter: SCMSAdapter, minimal_valid_scms_row: Dict[str, str]
    ):
        """Verify adapter and validator scale linearly and handle 50,000 rows within 5 seconds."""
        n_rows = 50_000
        base_df = pd.DataFrame([minimal_valid_scms_row])
        # Replicate to 50,000 rows with unique IDs
        repeated_df = pd.concat([base_df] * n_rows, ignore_index=True)
        repeated_df[COL_ID] = [str(i + 1) for i in range(n_rows)]

        t0 = time.perf_counter()
        df_std = scms_adapter.standardize_schema(repeated_df)
        df_enriched = scms_adapter.extract_temporal_features(df_std)
        t_adapter = time.perf_counter() - t0

        t1 = time.perf_counter()
        validator = SCMSValidator()
        report = validator.validate(df_enriched)
        t_validator = time.perf_counter() - t1

        assert len(df_enriched) == n_rows
        assert report.is_valid is True
        assert t_adapter + t_validator < 5.0, f"Processing {n_rows} rows took {t_adapter + t_validator:.2f}s (> 5.0s limit)"
