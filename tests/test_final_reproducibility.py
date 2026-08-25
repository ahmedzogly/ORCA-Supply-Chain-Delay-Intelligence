import pytest
import os

def test_final_metrics_written():
    assert os.path.exists("artifacts/final/final_holdout_metrics.json")
    
def test_final_metrics_are_deterministic():
    import json
    with open("artifacts/final/final_holdout_metrics.json", "r") as f:
        m = json.load(f)
    # Based on our frozen evaluation script
    assert m["holdout_size"] == 1013
    assert m["classification"]["status"] == "STABLE"
