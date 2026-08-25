"""Tests for runtime environment, dependency tiering, and packaging compliance."""

import platform
import sys
from pathlib import Path

import yaml


def test_python_version_compatibility():
    """Verify active Python runtime satisfies the >=3.10 requirement."""
    major, minor = sys.version_info.major, sys.version_info.minor
    assert (
        major == 3 and minor >= 10
    ), f"Python version must be >=3.10, found: {sys.version}"


def test_platform_is_supported():
    """Verify operating system environment is recognized and supported."""
    current_os = platform.system()
    assert current_os in ["Windows", "Linux", "Darwin"], f"Unsupported OS platform: {current_os}"


def test_baseline_dependencies_importable():
    """Verify all Stage 0 baseline core dependencies are importable."""
    import setuptools

    assert setuptools.__version__ is not None
    assert yaml.__version__ is not None


def test_pyproject_toml_structure(repo_root: Path):
    """Verify pyproject.toml exists and declares Stage 0 baseline vs deferred extras."""
    pyproject_path = repo_root / "pyproject.toml"
    assert pyproject_path.is_file(), "pyproject.toml not found in repository root"

    content = pyproject_path.read_text(encoding="utf-8")
    assert "[build-system]" in content
    assert "build-backend = \"setuptools.build_meta\"" in content
    assert "name = \"delay-intelligence\"" in content
    assert "requires-python = \">=3.10\"" in content

    # Stage 0 baseline dependencies check (must be lean)
    assert "dependencies = [" in content
    assert "pyyaml" in content

    # Deferred optional-dependencies groups check
    assert "[project.optional-dependencies]" in content
    assert "dev =" in content
    assert "data =" in content
    assert "ml =" in content
    assert "dl =" in content
    assert "uncertainty =" in content
    assert "causal =" in content
    assert "decision =" in content
    assert "api =" in content
    assert "dashboard =" in content
    assert "all =" in content


def test_proportional_architecture_no_cloud_hardcoding(repo_root: Path):
    """Verify baseline packaging does not mandate cloud frameworks (Airflow, BigQuery, Spark)."""
    pyproject_path = repo_root / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")

    # Ensure baseline dependencies do not force heavy cloud orchestrators or distributed engines
    baseline_section = content.split("[project.optional-dependencies]")[0]
    forbidden_baseline_strings = [
        "apache-airflow",
        "google-cloud-bigquery",
        "pyspark",
        "kubernetes",
        "celery",
    ]
    for forbidden in forbidden_baseline_strings:
        assert (
            forbidden not in baseline_section
        ), f"Baseline dependencies should not mandate {forbidden}"
