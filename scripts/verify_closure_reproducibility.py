"""Reproducibility checks for the patched v2 demo serving path."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient
from delay_intelligence.api.main import app
from delay_intelligence.data.adapters.scms import SCMSAdapter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "SCMS_Delivery_History_Dataset.csv"
REGISTRY = ROOT / "artifacts" / "model_registry" / "v2"
EXPECTED_SHA256 = "918b992dd3e8d4b64d2a727b2c4ea607603d0c58f19484e73f7b78528c6a8673"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_reproducibility_checks():
    assert RAW.exists(), f"Missing bundled SCMS source: {RAW}"
    assert sha256(RAW) == EXPECTED_SHA256
    raw = SCMSAdapter(data_path=RAW).load_raw()
    assert len(raw) == 10324

    required = [
        "catboost_classifier.cbm",
        "lightgbm_q05.txt",
        "lightgbm_q50.txt",
        "lightgbm_q95.txt",
        "probability_calibration.json",
        "cqr_calibration.json",
        "feature_schema.json",
        "serving_validation.json",
        "metadata.json",
    ]
    missing = [name for name in required if not (REGISTRY / name).exists()]
    assert not missing, f"Missing v2 registry artifacts: {missing}"

    validation = json.loads((REGISTRY / "serving_validation.json").read_text(encoding="utf-8"))
    assert validation["data_sha256"] == EXPECTED_SHA256
    assert validation["splits"]["holdout"]["rows"] == 1013

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    payload = {"features": {"Line Item Value": 1500, "Shipment Mode": "Air"}}
    for endpoint in ("/predict", "/explain", "/recommend"):
        r = client.post(endpoint, json=payload)
        assert r.status_code == 200, (endpoint, r.text)

    pred = client.post("/predict", json=payload).json()
    assert pred["evidence_label"] == "MODEL OUTPUT"
    rec = client.post("/recommend", json=payload).json()
    assert rec["evidence_label"] == "SIMULATED SCENARIO"

    print("PASS: raw hash, row count, v2 registry, validation metadata, and API smoke checks.")


if __name__ == "__main__":
    run_reproducibility_checks()
