"""Pytest global configuration, path discovery, and common test fixtures."""

from pathlib import Path
import shutil
import sys
import tempfile
from typing import Dict, Generator

import pandas as pd
import pytest

# Ensure delay_intelligence package in src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

WORKSPACE_ROOT = REPO_ROOT.parent
BUNDLED_DATA_ROOT = REPO_ROOT / "data"

SCMS_EXPECTED_SHA256 = "918b992dd3e8d4b64d2a727b2c4ea607603d0c58f19484e73f7b78528c6a8673"
SCMS_EXPECTED_BYTE_SIZE = 3785904
SCMS_EXPECTED_ROW_COUNT = 10324
SCMS_EXPECTED_COL_COUNT = 33


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    """Return the absolute path to the configs/ directory."""
    return REPO_ROOT / "configs"


@pytest.fixture(scope="session")
def workspace_root() -> Path:
    """Return the absolute path to the parent workspace directory."""
    return WORKSPACE_ROOT


@pytest.fixture(scope="session")
def raw_data_dirs(workspace_root: Path) -> Dict[str, Path]:
    """Return dictionary of raw read-only data source directory paths."""
    return {
        "scms": REPO_ROOT / "data" / "raw",
        "olist": REPO_ROOT / "data" / "external" / "olist",
        "dataco": REPO_ROOT / "data" / "external" / "dataco",
    }


@pytest.fixture(scope="session")
def scms_raw_path(raw_data_dirs: Dict[str, Path]) -> Path:
    """Return absolute path to raw SCMS CSV."""
    return raw_data_dirs["scms"] / "SCMS_Delivery_History_Dataset.csv"


@pytest.fixture(scope="session")
def scms_expected_sha256() -> str:
    """Return authoritative SHA-256 hash for raw SCMS dataset."""
    return SCMS_EXPECTED_SHA256


@pytest.fixture(scope="session")
def scms_raw_df(scms_raw_path: Path) -> pd.DataFrame:
    """Load raw SCMS dataframe with pandas default parsing (read-only)."""
    return pd.read_csv(scms_raw_path, encoding="utf-8-sig", dtype=str)


@pytest.fixture(scope="session")
def scms_adapter(scms_raw_path: Path):
    """Instantiate SCMS Ingestion Adapter."""
    from delay_intelligence.data.adapters.scms import SCMSAdapter

    return SCMSAdapter(data_path=scms_raw_path)


@pytest.fixture(scope="session")
def scms_canonical_df(scms_adapter) -> pd.DataFrame:
    """Return standardized canonical SCMS DataFrame from adapter."""
    raw = scms_adapter.load_raw()
    return scms_adapter.standardize_schema(raw)


@pytest.fixture(scope="session")
def scms_enriched_df(scms_adapter, scms_canonical_df) -> pd.DataFrame:
    """Return DataFrame enriched with temporal features and targets."""
    return scms_adapter.extract_temporal_features(scms_canonical_df)


@pytest.fixture
def clean_tmp_path() -> Generator[Path, None, None]:
    """Provide a reliable, isolated temporary directory without relying on system temp scanning."""
    temp_dir = tempfile.mkdtemp(prefix="test_di_")
    path_obj = Path(temp_dir).resolve()
    try:
        yield path_obj
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
