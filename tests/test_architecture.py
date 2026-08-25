"""Tests for architecture invariants, module imports, and interface contracts."""

import importlib
from pathlib import Path
import pytest

from delay_intelligence.core.exceptions import (
    CausalIdentificationError,
    ConformalCalibrationError,
    ConfigurationError,
    DataImmutabilityError,
    DataValidationError,
    DelayIntelligenceError,
    LeakageViolationError,
    ModelTrainingError,
    PrescriptiveOptimizationError,
)
from delay_intelligence.core.logging import get_logger, setup_logging
from delay_intelligence.data.adapters.base import BaseIngestionAdapter


REQUIRED_DOC_FILES = [
    "ARCHITECTURE.md",
    "README.md",
    "pyproject.toml",
    "docs/technology_decision_record.md",
    "docs/repository_structure.md",
    "docs/pipeline_specification.md",
    "docs/data_dictionary.md",
    "docs/scms_data_audit.md",
    "docs/scms_schema.md",
]

REQUIRED_MODULES = [
    "delay_intelligence",
    "delay_intelligence.core",
    "delay_intelligence.core.config",
    "delay_intelligence.core.logging",
    "delay_intelligence.core.exceptions",
    "delay_intelligence.data",
    "delay_intelligence.data.adapters",
    "delay_intelligence.data.adapters.base",
    "delay_intelligence.validation",
    "delay_intelligence.features",
    "delay_intelligence.evaluation",
    "delay_intelligence.models",
    "delay_intelligence.uncertainty",
    "delay_intelligence.causal",
    "delay_intelligence.decision",
    "delay_intelligence.api",
    "delay_intelligence.dashboard",
]

REQUIRED_ARTIFACT_DIRS = [
    "artifacts/data",
    "artifacts/models",
    "artifacts/uncertainty",
    "artifacts/causal",
    "artifacts/metrics",
    "artifacts/reports",
    "models",
]


@pytest.mark.parametrize("doc_file", REQUIRED_DOC_FILES)
def test_required_documentation_files_exist(doc_file: str, repo_root: Path):
    """Verify all required architectural documentation files exist and are non-empty."""
    target = repo_root / doc_file
    assert target.is_file(), f"Required file missing: {doc_file}"
    assert target.stat().st_size > 50, f"File appears unexpectedly empty: {doc_file}"


@pytest.mark.parametrize("mod_name", REQUIRED_MODULES)
def test_all_modules_importable(mod_name: str):
    """Verify every module in the package can be successfully imported."""
    mod = importlib.import_module(mod_name)
    assert mod is not None


@pytest.mark.parametrize("art_dir", REQUIRED_ARTIFACT_DIRS)
def test_artifact_directories_exist(art_dir: str, repo_root: Path):
    """Verify all artifact staging directories exist."""
    target = repo_root / art_dir
    assert target.is_dir(), f"Artifact directory missing: {art_dir}"


def test_base_ingestion_adapter_contract():
    """Verify BaseIngestionAdapter is an abstract base class enforcing required interface methods."""
    # Cannot instantiate abstract class directly
    with pytest.raises(TypeError):
        BaseIngestionAdapter("dummy/path")  # type: ignore

    # Concrete implementation subclassing all required methods
    class DummyAdapter(BaseIngestionAdapter):
        def load_raw(self):
            return {"dummy": "data"}

        def standardize_schema(self, df):
            return df

        def extract_temporal_features(self, df):
            return df

        def get_dataset_metadata(self):
            return {"name": "dummy"}

    adapter = DummyAdapter("dummy/path", config={"opt": 1})
    assert adapter.load_raw() == {"dummy": "data"}
    assert adapter.standardize_schema({"test": 1}) == {"test": 1}
    assert adapter.extract_temporal_features({"test": 1}) == {"test": 1}
    assert adapter.get_dataset_metadata()["name"] == "dummy"
    assert adapter.config["opt"] == 1


def test_custom_exception_hierarchy():
    """Verify custom exception hierarchy inherits from DelayIntelligenceError."""
    assert issubclass(ConfigurationError, DelayIntelligenceError)
    assert issubclass(DataValidationError, DelayIntelligenceError)
    assert issubclass(LeakageViolationError, DelayIntelligenceError)
    assert issubclass(DataImmutabilityError, DelayIntelligenceError)
    assert issubclass(ModelTrainingError, DelayIntelligenceError)
    assert issubclass(ConformalCalibrationError, DelayIntelligenceError)
    assert issubclass(CausalIdentificationError, DelayIntelligenceError)
    assert issubclass(PrescriptiveOptimizationError, DelayIntelligenceError)


def test_logging_setup(clean_tmp_path: Path):
    """Verify logging setup and get_logger functionality including file logging."""
    log_file = clean_tmp_path / "test.log"
    logger = setup_logging(
        level="DEBUG",
        log_to_file=True,
        log_file=str(log_file),
        format_str="%(name)s - %(levelname)s - %(message)s",
        date_format="%H:%M:%S",
    )
    assert logger.name == "delay_intelligence"
    child1 = get_logger("data.scms")
    assert child1.name == "delay_intelligence.data.scms"
    child2 = get_logger("delay_intelligence.models")
    assert child2.name == "delay_intelligence.models"

    child1.info("Test log message for child logger")
    assert log_file.exists()
