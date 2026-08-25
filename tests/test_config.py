"""Tests for configuration loading, schema consistency, and path resolution."""

from pathlib import Path
import pytest

from delay_intelligence.core.config import find_config_dir, get_data_paths, load_config
from delay_intelligence.core.exceptions import ConfigurationError


REQUIRED_CONFIGS = [
    "base",
    "data",
    "validation",
    "features",
    "models",
    "uncertainty",
    "causal",
    "decision",
    "serving",
]


@pytest.mark.parametrize("config_name", REQUIRED_CONFIGS)
def test_load_all_required_configs(config_name: str, configs_dir: Path):
    """Verify that every required YAML configuration file can be successfully loaded."""
    cfg = load_config(config_name, config_dir=configs_dir)
    assert isinstance(cfg, dict), f"Config {config_name} did not load as a dictionary"
    assert len(cfg) > 0, f"Config {config_name} is empty"


def test_base_config_keys(configs_dir: Path):
    """Verify required keys in base.yaml."""
    cfg = load_config("base", config_dir=configs_dir)
    assert "project" in cfg
    assert "paths" in cfg
    assert "logging" in cfg
    assert "compute" in cfg
    assert cfg["project"]["name"] == "delay-intelligence"
    assert cfg["project"]["seed"] == 42


def test_data_config_keys_and_datasets(configs_dir: Path):
    """Verify required keys in data.yaml and presence of datasets."""
    cfg = load_config("data", config_dir=configs_dir)
    assert "datasets" in cfg
    datasets = cfg["datasets"]
    assert "scms" in datasets
    assert "olist" in datasets
    assert "dataco" in datasets
    assert datasets["scms"]["is_read_only"] is True
    assert datasets["olist"]["is_read_only"] is True
    assert datasets["dataco"]["is_read_only"] is True


def test_get_data_paths_resolution(configs_dir: Path, repo_root: Path):
    """Verify get_data_paths resolves all 3 raw data paths to existing filesystem targets."""
    data_paths = get_data_paths(config_dir=configs_dir, base_dir=repo_root)

    assert "scms" in data_paths
    assert "olist" in data_paths
    assert "dataco" in data_paths

    scms_path = data_paths["scms"]
    olist_path = data_paths["olist"]
    dataco_path = data_paths["dataco"]

    assert scms_path.exists(), f"Bundled SCMS path does not exist: {scms_path}"
    # External datasets are intentionally optional in the demo archive; their
    # configured locations need not exist until a real validation study is run.
    assert "data/external/olist" in olist_path.as_posix()
    assert "data/external/dataco" in dataco_path.as_posix()


def test_get_data_paths_with_default_base_dir(configs_dir: Path):
    """Verify get_data_paths resolves correctly when base_dir is omitted (uses repo root)."""
    data_paths = get_data_paths(config_dir=configs_dir)
    assert "scms" in data_paths
    assert "olist" in data_paths
    assert "dataco" in data_paths


def test_load_nonexistent_config_raises_error(configs_dir: Path):
    """Verify requesting a non-existent configuration raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        load_config("non_existent_config_xyz", config_dir=configs_dir)


def test_find_config_dir_invalid_path():
    """Verify passing an invalid custom path to find_config_dir raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        find_config_dir("C:/non/existent/path/for/testing/12345")


def test_find_config_dir_default():
    """Verify default discovery of configs directory."""
    cfg_dir = find_config_dir()
    assert cfg_dir.is_dir()
    assert (cfg_dir / "base.yaml").is_file()


def test_load_empty_and_non_dict_yaml(clean_tmp_path: Path):
    """Verify handling of empty or non-dict YAML files."""
    empty_file = clean_tmp_path / "empty.yaml"
    empty_file.write_text("", encoding="utf-8")
    assert load_config("empty", config_dir=clean_tmp_path) == {}

    list_file = clean_tmp_path / "list.yaml"
    list_file.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config("list", config_dir=clean_tmp_path)

    bad_syntax = clean_tmp_path / "bad.yaml"
    bad_syntax.write_text("key: [unclosed list", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config("bad", config_dir=clean_tmp_path)


def test_get_data_paths_missing_datasets_section(clean_tmp_path: Path):
    """Verify error when data.yaml is missing datasets section."""
    dummy_data = clean_tmp_path / "data.yaml"
    dummy_data.write_text("storage:\n  format: parquet\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="No 'datasets' section"):
        get_data_paths(config_dir=clean_tmp_path)
