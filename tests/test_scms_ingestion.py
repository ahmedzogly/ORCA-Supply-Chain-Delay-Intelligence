"""Tests for SCMS data ingestion adapter, row-count reconciliation, and Parquet caching."""

import hashlib
from pathlib import Path
import pandas as pd
import pytest

from delay_intelligence.data.adapters.base import BaseIngestionAdapter
from delay_intelligence.data.adapters.scms import SCMSAdapter
from delay_intelligence.data.loader import DataLoader, ingest_scms_pipeline
from delay_intelligence.data.schema import (
    COL_ID,
    SCMS_EXPECTED_BYTE_SIZE,
    SCMS_EXPECTED_COL_COUNT,
    SCMS_EXPECTED_ROW_COUNT,
)


def test_scms_raw_file_existence_and_size(scms_raw_path: Path):
    """Verify raw SCMS dataset file exists, is a regular file, and matches expected byte size."""
    assert scms_raw_path.exists(), f"Raw SCMS dataset file not found at: {scms_raw_path}"
    assert scms_raw_path.is_file(), f"Path is not a regular file: {scms_raw_path}"
    file_size = scms_raw_path.stat().st_size
    assert file_size >= 3_700_000, f"File size ({file_size}) is unexpectedly small"
    assert file_size == SCMS_EXPECTED_BYTE_SIZE, f"Exact file size mismatch: {file_size} != {SCMS_EXPECTED_BYTE_SIZE}"


def test_scms_raw_hash_integrity_before_and_after_ingestion(
    scms_raw_path: Path, scms_expected_sha256: str, scms_adapter: SCMSAdapter
):
    """Verify raw CSV file SHA-256 hash is invariant before and after adapter execution."""
    # Compute SHA-256 before ingestion
    hasher_before = hashlib.sha256()
    with open(scms_raw_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher_before.update(chunk)
    hash_before = hasher_before.hexdigest()

    assert hash_before == scms_expected_sha256, f"Initial SHA-256 hash mismatch: {hash_before}"

    # Run adapter ingestion operations
    raw_df = scms_adapter.load_raw()
    std_df = scms_adapter.standardize_schema(raw_df)
    _ = scms_adapter.extract_temporal_features(std_df)

    # Compute SHA-256 after ingestion
    hasher_after = hashlib.sha256()
    with open(scms_raw_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher_after.update(chunk)
    hash_after = hasher_after.hexdigest()

    assert hash_after == hash_before == scms_expected_sha256, (
        f"Raw file was modified during ingestion! Hash changed: {hash_before} -> {hash_after}"
    )


def test_scms_adapter_contract_implementation(scms_adapter: SCMSAdapter):
    """Verify SCMSAdapter implements BaseIngestionAdapter ABC and required methods."""
    assert isinstance(scms_adapter, BaseIngestionAdapter), "SCMSAdapter must inherit from BaseIngestionAdapter"
    assert hasattr(scms_adapter, "load_raw"), "SCMSAdapter missing load_raw method"
    assert hasattr(scms_adapter, "standardize_schema"), "SCMSAdapter missing standardize_schema method"
    assert hasattr(scms_adapter, "extract_temporal_features"), "SCMSAdapter missing extract_temporal_features method"
    assert hasattr(scms_adapter, "get_dataset_metadata"), "SCMSAdapter missing get_dataset_metadata method"
    assert callable(scms_adapter.load_raw)
    assert callable(scms_adapter.standardize_schema)
    assert callable(scms_adapter.extract_temporal_features)
    assert callable(scms_adapter.get_dataset_metadata)


def test_scms_adapter_invalid_path_handling():
    """Verify initializing SCMSAdapter with a non-existent path raises FileNotFoundError upon load_raw."""
    invalid_adapter = SCMSAdapter(data_path="C:/non/existent/path/to/scms.csv")
    with pytest.raises(FileNotFoundError):
        invalid_adapter.load_raw()


def test_scms_load_raw_dimensions(scms_adapter: SCMSAdapter):
    """Verify load_raw loads exactly 10,324 rows and 33 columns."""
    df_raw = scms_adapter.load_raw()
    assert isinstance(df_raw, pd.DataFrame), "Raw data must be a pandas DataFrame"
    assert df_raw.shape == (SCMS_EXPECTED_ROW_COUNT, SCMS_EXPECTED_COL_COUNT), (
        f"Expected shape ({SCMS_EXPECTED_ROW_COUNT}, {SCMS_EXPECTED_COL_COUNT}), got {df_raw.shape}"
    )


def test_scms_row_count_reconciliation_zero_loss(
    scms_raw_df: pd.DataFrame, scms_canonical_df: pd.DataFrame
):
    """Verify zero data loss: canonical DataFrame contains 100% of raw rows (10,324)."""
    assert len(scms_canonical_df) == len(scms_raw_df) == SCMS_EXPECTED_ROW_COUNT, (
        f"Row loss detected! Canonical: {len(scms_canonical_df)}, Raw: {len(scms_raw_df)}"
    )
    retention_rate = len(scms_canonical_df) / len(scms_raw_df)
    assert retention_rate == 1.0, f"Retention rate must be 100%, got {retention_rate:.4%}"


def test_scms_primary_key_uniqueness(scms_canonical_df: pd.DataFrame):
    """Verify primary key 'ID' is 100% unique and non-null."""
    assert COL_ID in scms_canonical_df.columns, "Primary key 'ID' missing from canonical schema"
    assert scms_canonical_df[COL_ID].isna().sum() == 0, "Primary key 'ID' contains null values"
    assert scms_canonical_df[COL_ID].is_unique, "Primary key 'ID' contains duplicate values"
    assert scms_canonical_df[COL_ID].nunique() == SCMS_EXPECTED_ROW_COUNT, (
        f"Distinct ID count mismatch: {scms_canonical_df[COL_ID].nunique()} != {SCMS_EXPECTED_ROW_COUNT}"
    )


def test_scms_metadata_summary_contract(scms_adapter: SCMSAdapter):
    """Verify get_dataset_metadata returns complete structured dictionary."""
    meta = scms_adapter.get_dataset_metadata()
    assert isinstance(meta, dict), "Metadata must be a dictionary"
    assert meta["name"] == "scms"
    assert meta["row_count"] == SCMS_EXPECTED_ROW_COUNT
    assert meta["column_count"] == SCMS_EXPECTED_COL_COUNT
    assert meta["primary_key"] == COL_ID
    assert meta["is_primary_key_unique"] is True
    assert meta["duplicate_rows"] == 0
    assert "date_milestone_coverage" in meta
    assert "target_distribution" in meta
    assert meta["target_distribution"]["delayed_count"] == 1186
    assert meta["target_distribution"]["on_time_count"] == 9138


def test_scms_dataloader_and_parquet_serialization(clean_tmp_path: Path):
    """Verify DataLoader executes and saves Bronze Parquet artifact with bit-level integrity."""
    loader = DataLoader()
    parquet_path = clean_tmp_path / "bronze_scms.parquet"
    df, report = loader.load_scms(save_bronze=True, bronze_path=parquet_path)

    assert len(df) == SCMS_EXPECTED_ROW_COUNT
    assert report.is_valid is True
    assert parquet_path.exists(), f"Parquet file was not written to {parquet_path}"
    assert parquet_path.stat().st_size > 0

    # Read back Parquet and verify shape and types
    df_reloaded = pd.read_parquet(parquet_path, engine="pyarrow")
    assert df_reloaded.shape[0] == SCMS_EXPECTED_ROW_COUNT
    assert COL_ID in df_reloaded.columns
    assert df_reloaded[COL_ID].dtype == "int64"


def test_scms_ingest_pipeline_end_to_end(clean_tmp_path: Path):
    """Verify ingest_scms_pipeline convenience function runs without errors."""
    parquet_path = clean_tmp_path / "pipeline_scms.parquet"
    df, report = ingest_scms_pipeline(bronze_output_path=parquet_path, save_parquet=True)
    assert len(df) == SCMS_EXPECTED_ROW_COUNT
    assert report.is_valid is True
    assert parquet_path.exists()
