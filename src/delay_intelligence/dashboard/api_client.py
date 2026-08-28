"""Local dashboard inference client for the ORCA platform.

Directly evaluates ModelLoader, DecisionEngine, and FeatureBuilder in-process,
completely independent of external HTTP servers or web frameworks (FastAPI/Uvicorn),
guaranteeing 100% reliability on Streamlit Cloud.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from delay_intelligence.decision.engine import DecisionEngine
from delay_intelligence.serving.feature_builder import build_features
from delay_intelligence.serving.model_loader import ModelLoader

# ── Dynamic Path Resolution ──────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
DECISION_CONFIG = REPO_ROOT / "configs" / "decision.yaml"
if not DECISION_CONFIG.exists():
    DECISION_CONFIG = Path("configs/decision.yaml").resolve()

CAUSAL_STABILITY = REPO_ROOT / "artifacts" / "causal" / "causal_edge_stability.csv"
if not CAUSAL_STABILITY.exists():
    CAUSAL_STABILITY = Path("artifacts/causal/causal_edge_stability.csv").resolve()

POSSIBLE_DEMO_PATHS = [
    REPO_ROOT / "artifacts" / "demo" / "demo_shipments.csv",
    Path("artifacts/demo/demo_shipments.csv").resolve(),
    Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "demo" / "demo_shipments.csv",
]

OUTCOME_OR_NONFEATURE = {
    "ID",
    "T_pred",
    "Delay_Days",
    "Delay_Flag",
    "Delivered to Client Date",
    "Delivery Recorded Date",
    "is_temporal_anomaly",
}

_decision_engine_instance: DecisionEngine | None = None


def get_engine() -> DecisionEngine:
    global _decision_engine_instance
    if _decision_engine_instance is None:
        cfg = str(DECISION_CONFIG) if DECISION_CONFIG.exists() else None
        _decision_engine_instance = DecisionEngine(config_path=cfg)
    return _decision_engine_instance


def _risk_tier(p_late: float) -> str:
    if p_late <= 0.30:
        return "LOW_RISK"
    if p_late <= 0.60:
        return "WATCH"
    if p_late <= 0.85:
        return "HIGH_RISK"
    return "CRITICAL"


def _exploratory_causal_hypotheses(top_features: List[str]) -> List[str]:
    if not CAUSAL_STABILITY.exists():
        return []
    try:
        edges = pd.read_csv(CAUSAL_STABILITY)
    except Exception:
        return []

    source_col = next((c for c in ["source", "from", "Source", "From"] if c in edges.columns), None)
    target_col = next((c for c in ["target", "to", "Target", "To"] if c in edges.columns), None)
    stable_col = next((c for c in ["stable", "is_stable", "Stable", "stability_class"] if c in edges.columns), None)
    if not source_col or not target_col:
        return []

    if stable_col:
        stable = edges[stable_col].astype(str).str.lower().isin(["true", "1", "yes", "stable"])
        edges = edges.loc[stable]

    top = set(top_features)
    hypotheses = []
    for _, row in edges.iterrows():
        src, dst = str(row[source_col]), str(row[target_col])
        if src in top and "Delay" in dst:
            hypotheses.append(f"{src} -> {dst}")
    return hypotheses[:5]


def load_data(limit: int = 100) -> pd.DataFrame:
    """Load the frozen real-data demo sample generated from the untouched holdout."""
    demo_file = next((p for p in POSSIBLE_DEMO_PATHS if p.exists()), None)
    if demo_file is None or not demo_file.exists():
        raise FileNotFoundError(
            f"Demo sample not found in expected locations: {POSSIBLE_DEMO_PATHS}."
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


def api_predict(features_dict: dict[str, Any]) -> dict[str, Any]:
    """In-process model scoring returning standardized predict contract dictionary."""
    loader = ModelLoader.get_instance()
    df = build_features(features_dict, loader.feature_schema)
    prob = float(loader.calibrated_probability(df))
    severity = loader.severity_if_delayed(df)

    return {
        "probability_late": prob,
        "classification_decision": bool(prob >= loader.decision_threshold),
        "decision_threshold": float(loader.decision_threshold),
        "risk_tier": _risk_tier(prob),
        "severity_p50": float(severity["p50"]),
        "severity_interval_90": [float(x) for x in severity["interval_90"]],
        "severity_definition": severity["definition"],
        "model_version": loader.metadata["model_version"],
        "prediction_contract_version": loader.metadata["prediction_contract_version"],
    }


def api_explain(features_dict: dict[str, Any]) -> dict[str, Any]:
    """In-process local SHAP explanation."""
    loader = ModelLoader.get_instance()
    df = build_features(features_dict, loader.feature_schema)
    prob = float(loader.calibrated_probability(df))
    shap_rows = loader.shap_explanation(df, top_k=5)
    top_features = [x["feature"] for x in shap_rows]
    hypotheses = _exploratory_causal_hypotheses(top_features)

    return {
        "probability_late": prob,
        "top_predictive_drivers": top_features,
        "shap_contributions": shap_rows,
        "causal_candidates": hypotheses,
        "causal_stability": "exploratory_hypothesis_only",
    }


def api_recommend(features_dict: dict[str, Any]) -> dict[str, Any]:
    """In-process prescriptive decision recommendation."""
    pred = api_predict(features_dict)
    expl = api_explain(features_dict)
    engine = get_engine()

    value = features_dict.get("Line Item Value", 0.0)
    value = 0.0 if value is None else float(value)

    decision = engine.evaluate_sensitivity(
        shipment_id="dash_req",
        p_late=pred["probability_late"],
        severity_p50=pred["severity_p50"],
        severity_interval_90=pred["severity_interval_90"],
        line_item_value=value,
        fulfillment_channel=features_dict.get("Fulfill Via", "Unknown"),
        shap_drivers=expl["top_predictive_drivers"],
        causal_candidates=expl["causal_candidates"],
    )

    return {
        "recommendation": decision["recommended_action"],
        "decision_reason": decision["decision_reason"],
        "expected_impact_type": decision["expected_impact"]["type"],
        "robustness": decision["robustness_class"],
        "human_approval_required": decision["human_approval_required"],
        "evidence_label": "SIMULATED SCENARIO",
        "impact_disclaimer": "Scenario estimate based on configurable assumptions; not realized financial savings.",
    }



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
