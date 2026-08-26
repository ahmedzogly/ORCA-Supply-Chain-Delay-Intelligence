"""Local dashboard client for the research demo.

Provides resilient inference execution (via FastAPI TestClient or direct in-process serving),
guaranteeing zero runtime failure even in isolated cloud environments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from delay_intelligence.api.main import (
    app,
    explain as direct_explain,
    get_decision_engine,
    get_model_loader,
    predict as direct_predict,
    recommend as direct_recommend,
)
from delay_intelligence.api.schemas import PredictRequest

# ── Resilient TestClient Initialisation ───────────────────────────────────────
client = None
try:
    from fastapi.testclient import TestClient
    client = TestClient(app)
except Exception:
    client = None

# ── Dynamic Demo Dataset Path Resolution ─────────────────────────────────────
POSSIBLE_DEMO_PATHS = [
    Path(__file__).resolve().parents[3] / "artifacts" / "demo" / "demo_shipments.csv",
    Path("artifacts/demo/demo_shipments.csv").resolve(),
    Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "demo" / "demo_shipments.csv",
]

DEMO_DATA = next((p for p in POSSIBLE_DEMO_PATHS if p.exists()), POSSIBLE_DEMO_PATHS[0])

OUTCOME_OR_NONFEATURE = {
    "ID",
    "T_pred",
    "Delay_Days",
    "Delay_Flag",
    "Delivered to Client Date",
    "Delivery Recorded Date",
    "is_temporal_anomaly",
}


def load_data(limit: int = 100) -> pd.DataFrame:
    """Load the frozen real-data demo sample generated from the untouched holdout."""
    demo_file = next((p for p in POSSIBLE_DEMO_PATHS if p.exists()), None)
    if demo_file is None or not demo_file.exists():
        raise FileNotFoundError(
            f"Demo sample not found in expected locations: {POSSIBLE_DEMO_PATHS}. Run scripts/build_serving_registry.py first."
        )
    return pd.read_csv(demo_file).head(limit)


def row_to_features(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    clean: dict[str, Any] = {}
    for key, value in data.items():
        if key in OUTCOME_OR_NONFEATURE:
            continue
        if pd.isna(value):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        clean[str(key)] = value
    return clean


def _post(path: str, features_dict: dict[str, Any]) -> dict[str, Any]:
    """Posts to TestClient if available, otherwise delegates directly to in-process handlers."""
    if client is not None:
        try:
            res = client.post(path, json={"features": features_dict})
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass

    # Direct in-process fallback
    req = PredictRequest(features=features_dict)
    loader = get_model_loader()

    if path == "/predict":
        pred = direct_predict(req, loader)
        return pred.model_dump() if hasattr(pred, "model_dump") else pred.dict()
    elif path == "/explain":
        expl = direct_explain(req, loader)
        return expl.model_dump() if hasattr(expl, "model_dump") else expl.dict()
    elif path == "/recommend":
        engine = get_decision_engine()
        rec = direct_recommend(req, loader, engine)
        return rec.model_dump() if hasattr(rec, "model_dump") else rec.dict()
    else:
        raise ValueError(f"Unknown path: {path}")


def api_predict(features_dict: dict[str, Any]) -> dict[str, Any]:
    return _post("/predict", features_dict)


def api_explain(features_dict: dict[str, Any]) -> dict[str, Any]:
    return _post("/explain", features_dict)


def api_recommend(features_dict: dict[str, Any]) -> dict[str, Any]:
    return _post("/recommend", features_dict)


def find_default_demo_shipment() -> str:
    """Find the highest-risk shipment ID for the default demo selection."""
    df = load_data()
    best_id = str(df["ID"].iloc[0]) if "ID" in df.columns else "0"
    best_prob = -1.0
    for _, row in df.iterrows():
        features = row_to_features(row)
        pred = api_predict(features)
        p = float(pred["probability_late"])
        if p > best_prob:
            best_prob = p
            best_id = str(row.get("ID", _))
    return best_id
