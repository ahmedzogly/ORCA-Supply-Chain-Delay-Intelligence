"""Data-source integrity checks for the self-contained demo archive."""
from pathlib import Path
from typing import Dict
import pytest

SCMS_FILE = "SCMS_Delivery_History_Dataset.csv"


def test_bundled_scms_directory_exists(raw_data_dirs: Dict[str, Path]):
    path = raw_data_dirs["scms"]
    assert path.is_dir()
    assert (path / SCMS_FILE).is_file()


def test_bundled_scms_inventory_and_volume(raw_data_dirs: Dict[str, Path]):
    path = raw_data_dirs["scms"] / SCMS_FILE
    assert path.stat().st_size >= 3_700_000


def test_bundled_scms_read_only_stream(raw_data_dirs: Dict[str, Path]):
    path = raw_data_dirs["scms"] / SCMS_FILE
    with path.open("rb") as f:
        assert len(f.read(100)) == 100


@pytest.mark.parametrize("key", ["olist", "dataco"])
def test_external_sources_are_optional_not_demo_evidence(raw_data_dirs: Dict[str, Path], key: str):
    """External files may be mounted for future validation but are not required by the demo."""
    path = raw_data_dirs[key]
    if not path.exists():
        pytest.skip(f"{key} source intentionally not bundled; external validation is NOT VALIDATED")
    assert path.is_dir()
