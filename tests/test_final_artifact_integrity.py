import pytest
import os
import json

def test_final_artifact_manifest_exists():
    assert os.path.exists("artifacts/final_manifest.json")

def test_final_model_artifacts_frozen():
    with open("artifacts/final_manifest.json", "r") as f:
        manifest = json.load(f)
    assert os.path.exists(manifest["artifacts"]["catboost_champion"])
    assert os.path.exists(manifest["artifacts"]["feature_schema"])
    assert os.path.exists(manifest["artifacts"]["decision_config"])
