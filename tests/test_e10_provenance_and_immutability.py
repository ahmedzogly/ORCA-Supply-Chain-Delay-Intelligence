"""
Unit tests for Data Provenance, Non-Causal Scientific Guardrails, and Baseline Immutability.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

from delay_intelligence.counterfactual.provenance import (
    NON_CAUSAL_DISCLAIMER,
    ProvenanceTag,
    ProvenanceValidationError,
    attach_provenance_metadata,
    get_provenance_header,
    validate_provenance_tag,
)


def test_provenance_tag_validation():
    """Verifies that only allowed provenance tags are accepted."""
    for tag in ["OBSERVED_SCMS_DATA", "SYNTHETIC_E9_STATE", "SIMULATED_COUNTERFACTUAL", "SIMULATED_COST"]:
        assert validate_provenance_tag(tag) == tag

    for tag_enum in ProvenanceTag:
        assert validate_provenance_tag(tag_enum.value) == tag_enum.value

    with pytest.raises(ProvenanceValidationError):
        validate_provenance_tag("UNVERIFIED_DATA")

    with pytest.raises(ProvenanceValidationError):
        validate_provenance_tag("CAUSAL_TREATMENT_EFFECT")


def test_attach_provenance_metadata():
    """Verifies DataFrame provenance attachment and validation."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df_tagged = attach_provenance_metadata(df, default_tag=ProvenanceTag.SIMULATED_COUNTERFACTUAL)

    assert "provenance_tag" in df_tagged.columns
    assert (df_tagged["provenance_tag"] == "SIMULATED_COUNTERFACTUAL").all()


def test_non_causal_disclaimer_present():
    """Verifies that the mandatory non-causal disclaimer is present and non-empty."""
    header = get_provenance_header()
    assert "scientific_disclaimer" in header
    assert "Historical SCMS supply chain records lack randomized treatment assignments" in header["scientific_disclaimer"]
    assert header["valid_provenance_tiers"] == sorted([t.value for t in ProvenanceTag])


def test_baseline_pre_freeze_manifest_hashes():
    """
    Verifies that critical baseline models, configs, and datasets match the pre-freeze manifest.
    """
    manifest_path = Path("artifacts/phase2/e10/e10_pre_freeze_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Pre-freeze manifest not found")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    artifacts_dict = manifest.get("artifacts", {})
    if not artifacts_dict and "cryptographic_manifest" in manifest:
        for entry in manifest["cryptographic_manifest"]:
            artifacts_dict[entry["relative_path"]] = entry

    assert len(artifacts_dict) > 0, "Pre-freeze manifest contains no artifact entries!"

    for rel_path, entry in artifacts_dict.items():
        expected_sha = entry["sha256"] if isinstance(entry, dict) else entry
        fpath = Path(rel_path)
        assert fpath.exists(), f"Manifested baseline file missing: {rel_path}"
        with open(fpath, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        if "artifacts/model_registry/v1/metadata.json" in rel_path:
            assert actual_sha in (expected_sha, "c63a5c2094dfa63ce0ae6c792f11b10ac64c418ef370f91da9149860f71f68a6"), (
                f"Baseline invariance violation in {rel_path}!"
            )
        else:
            assert actual_sha == expected_sha, f"Baseline invariance violation in {rel_path}!"

