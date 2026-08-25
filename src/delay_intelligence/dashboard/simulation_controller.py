"""Simulation Controller & Digital Twin Injection Module.

Handles in-memory feature mutations and dynamic batch re-scoring across the active
cohort for real-time scenario simulation (S0..S6) and interactive sliders.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from delay_intelligence.dashboard.api_client import load_data, row_to_features
from delay_intelligence.decision.engine import DecisionEngine
from delay_intelligence.serving.feature_builder import build_features
from delay_intelligence.serving.model_loader import ModelLoader

# ── Preset Scenarios Definition ──────────────────────────────────────────────
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "S0": {
        "name": "S0: Normal Baseline",
        "badge": "BASELINE",
        "description": "Historical pre-dispatch conditions with standard operations and no active disruptions.",
        "icon": ":material/check_circle:",
    },
    "S1": {
        "name": "S1: Cold-Chain Temperature Excursion",
        "badge": "TELEMETRY ANOMALY",
        "description": "Temperature spike detected (>8.5°C). Increases supplier delay risk by +15% and flags cold-chain compliance.",
        "icon": ":material/thermostat:",
    },
    "S2": {
        "name": "S2: Port Congestion / Route Shift",
        "badge": "GEOGRAPHIC BOTTLENECK",
        "description": "Major transshipment hub congestion and route rerouting. Extends scheduled transit by +30% (+25km route drift).",
        "icon": ":material/alt_route:",
    },
    "S3": {
        "name": "S3: Customs / Border Slowdown",
        "badge": "PROCESS DELAY",
        "description": "Regulatory inspection backlog at destination port. Adds +5 days directly to lead time and transit horizon.",
        "icon": ":material/hourglass_top:",
    },
    "S4": {
        "name": "S4: Carrier Capacity Shock & ETA Surge",
        "badge": "CARRIER DISRUPTION",
        "description": "Severe freight capacity shortage and carrier schedule collapse. Adds +10 days lead time and +25% vendor risk.",
        "icon": ":material/warning:",
    },
    "S5": {
        "name": "S5: Compound Multi-Signal Disruption",
        "badge": "SYSTEMIC CRISIS",
        "description": "Compound crisis combining cold-chain excursions, transshipment bottlenecks, and carrier capacity failure.",
        "icon": ":material/crisis_alert:",
    },
    "S6": {
        "name": "S6: Post-Intervention Recovery",
        "badge": "MITIGATED STATE",
        "description": "Post-mitigation recovery state: Express clearance priority, -3 days transit reduction, and -20% vendor delay risk.",
        "icon": ":material/health_and_safety:",
    },
}

_GLOBAL_DECISION_ENGINE: DecisionEngine | None = None


def get_decision_engine() -> DecisionEngine:
    global _GLOBAL_DECISION_ENGINE
    if _GLOBAL_DECISION_ENGINE is None:
        _GLOBAL_DECISION_ENGINE = DecisionEngine()
    return _GLOBAL_DECISION_ENGINE


def _risk_tier(p_late: float) -> str:
    """Classifies late probability into standard risk tiers."""
    if p_late <= 0.30:
        return "LOW_RISK"
    elif p_late <= 0.60:
        return "WATCH"
    elif p_late <= 0.85:
        return "HIGH_RISK"
    return "CRITICAL"


def apply_scenario_perturbation(
    df: pd.DataFrame,
    scenario_key: str = "S0",
    custom_params: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Mutate features in memory for the active cohort based on scenarios (S0..S6) and custom sliders."""
    mutated = df.copy()
    params = custom_params or {}

    # Ensure baseline numerical columns exist
    if "vendor_hist_delay_rate" not in mutated.columns:
        mutated["vendor_hist_delay_rate"] = 0.15
    if "Forecast_Horizon_Days" not in mutated.columns:
        mutated["Forecast_Horizon_Days"] = 45.0
    if "Scheduled_Transit_Days" not in mutated.columns:
        mutated["Scheduled_Transit_Days"] = 45.0
    if "country_hist_delay_rate" not in mutated.columns:
        mutated["country_hist_delay_rate"] = 0.10
    if "Line Item Value" not in mutated.columns:
        mutated["Line Item Value"] = 10000.0

    # ── Scenario-Specific Injections ─────────────────────────────────────────
    if scenario_key == "S1":
        # Cold-chain excursion: vendor risk rate +0.15
        mutated["vendor_hist_delay_rate"] = np.clip(
            mutated["vendor_hist_delay_rate"] + 0.15, 0.0, 1.0
        )
        mutated["iot_temperature_c"] = 9.5

    elif scenario_key == "S2":
        # Port congestion / route shift: +30% transit days
        mutated["Scheduled_Transit_Days"] = np.maximum(
            1.0, mutated["Scheduled_Transit_Days"] * 1.30
        )
        mutated["Forecast_Horizon_Days"] = np.maximum(
            1.0, mutated["Forecast_Horizon_Days"] * 1.30
        )
        mutated["iot_route_deviation_km"] = 25.0

    elif scenario_key == "S3":
        # Customs slowdown: +5 days lead time
        mutated["Scheduled_Transit_Days"] = np.maximum(
            1.0, mutated["Scheduled_Transit_Days"] + 5.0
        )
        mutated["Forecast_Horizon_Days"] = np.maximum(
            1.0, mutated["Forecast_Horizon_Days"] + 5.0
        )

    elif scenario_key == "S4":
        # Carrier shock: +10 days lead time and +25% vendor delay rate
        mutated["Scheduled_Transit_Days"] = np.maximum(
            1.0, mutated["Scheduled_Transit_Days"] + 10.0
        )
        mutated["Forecast_Horizon_Days"] = np.maximum(
            1.0, mutated["Forecast_Horizon_Days"] + 10.0
        )
        mutated["vendor_hist_delay_rate"] = np.clip(
            mutated["vendor_hist_delay_rate"] * 1.25 + 0.05, 0.0, 1.0
        )

    elif scenario_key == "S5":
        # Compound disruption (S1 + S2 + S4)
        mutated["Scheduled_Transit_Days"] = np.maximum(
            1.0, mutated["Scheduled_Transit_Days"] * 1.35 + 12.0
        )
        mutated["Forecast_Horizon_Days"] = np.maximum(
            1.0, mutated["Forecast_Horizon_Days"] * 1.35 + 12.0
        )
        mutated["vendor_hist_delay_rate"] = np.clip(
            mutated["vendor_hist_delay_rate"] * 1.40 + 0.15, 0.0, 1.0
        )
        mutated["country_hist_delay_rate"] = np.clip(
            mutated["country_hist_delay_rate"] * 1.30 + 0.10, 0.0, 1.0
        )
        mutated["iot_temperature_c"] = 10.2
        mutated["iot_route_deviation_km"] = 45.0

    elif scenario_key == "S6":
        # Post-intervention recovery: -3 days transit reduction, -20% vendor delay risk
        mutated["Scheduled_Transit_Days"] = np.maximum(
            1.0, mutated["Scheduled_Transit_Days"] - 3.0
        )
        mutated["Forecast_Horizon_Days"] = np.maximum(
            1.0, mutated["Forecast_Horizon_Days"] - 3.0
        )
        mutated["vendor_hist_delay_rate"] = np.clip(
            mutated["vendor_hist_delay_rate"] * 0.80, 0.0, 1.0
        )
        mutated["iot_temperature_c"] = 4.0
        mutated["iot_route_deviation_km"] = 0.0

    # ── Custom Interactive Slider Adjustments ────────────────────────────────
    lead_time_lag = float(params.get("lead_time_lag", 0.0))
    if lead_time_lag != 0.0:
        mutated["Forecast_Horizon_Days"] = np.maximum(
            1.0, mutated["Forecast_Horizon_Days"] + lead_time_lag
        )
        mutated["Scheduled_Transit_Days"] = np.maximum(
            1.0, mutated["Scheduled_Transit_Days"] + lead_time_lag
        )

    vendor_risk_mult = float(params.get("vendor_risk_multiplier", 1.0))
    if vendor_risk_mult != 1.0:
        mutated["vendor_hist_delay_rate"] = np.clip(
            mutated["vendor_hist_delay_rate"] * vendor_risk_mult, 0.0, 1.0
        )
        mutated["country_hist_delay_rate"] = np.clip(
            mutated["country_hist_delay_rate"] * vendor_risk_mult, 0.0, 1.0
        )

    value_pct = float(params.get("line_item_value_pct", 0.0))
    if value_pct != 0.0:
        mutated["Line Item Value"] = np.maximum(
            0.0, mutated["Line Item Value"] * (1.0 + value_pct / 100.0)
        )

    criticality_boost = bool(params.get("criticality_boost", False))
    if criticality_boost:
        mutated["First Line Designation"] = "Yes"

    return mutated


def rescore_cohort(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Execute fast in-process batch re-scoring across the active cohort.

    Returns:
        tuple (mutated_df, scored_results_list)
    """
    loader = ModelLoader.get_instance()
    engine = get_decision_engine()

    # 1. Vectorized feature construction
    schema = loader.feature_schema
    num_cols = schema["num_cols"]
    cat_cols = schema["cat_cols"]
    all_features = schema["all_features"]

    feat_df = df.copy()
    for c in num_cols:
        if c not in feat_df.columns:
            feat_df[c] = 0.0
        else:
            feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce").fillna(0.0).astype(float)

    for c in cat_cols:
        if c not in feat_df.columns:
            feat_df[c] = "missing"
        else:
            feat_df[c] = (
                feat_df[c]
                .fillna("missing")
                .astype(str)
                .replace({"nan": "missing", "<NA>": "missing", "None": "missing"})
            )

    model_input_df = feat_df[all_features]

    # 2. Batch CatBoost predictions & Isotonic Calibration
    raw_probs = loader.classifier.predict_proba(model_input_df)[:, 1]
    x_cal = np.asarray(loader.probability_calibration["x_thresholds"], dtype=float)
    y_cal = np.asarray(loader.probability_calibration["y_thresholds"], dtype=float)
    calibrated_probs = np.interp(raw_probs, x_cal, y_cal)

    # 3. Batch LightGBM Severity Quantiles & CQR Adjustment
    lgb_input = loader._as_lightgbm_frame(model_input_df)
    q05_preds = loader.q05.predict(lgb_input)
    q50_preds = loader.q50.predict(lgb_input)
    q95_preds = loader.q95.predict(lgb_input)
    cqr_adj = float(loader.cqr_params["q_adjustment"])

    low_intervals = np.maximum(0.0, q05_preds - cqr_adj)
    med_severities = np.maximum(0.0, q50_preds)
    high_intervals = np.maximum(low_intervals, q95_preds + cqr_adj)

    decision_threshold = float(loader.decision_threshold)
    results: List[Dict[str, Any]] = []

    for i, (_, row) in enumerate(df.iterrows()):
        prob = float(calibrated_probs[i])
        p50 = float(med_severities[i])
        lo = float(low_intervals[i])
        hi = float(high_intervals[i])
        decision = bool(prob >= decision_threshold)
        tier = _risk_tier(prob)
        shipment_id = str(row.get("ID", f"ID_{i}"))
        val = float(row.get("Line Item Value", 0.0))

        dec = engine.evaluate(
            shipment_id=shipment_id,
            p_late=prob,
            severity_p50=p50,
            severity_interval_90=[lo, hi],
            line_item_value=val,
            fulfillment_channel=str(row.get("Fulfill Via", "Direct Drop")),
            shap_drivers=[],
            causal_candidates=[],
        )

        results.append({
            "Shipment ID": shipment_id,
            "probability_late": prob,
            "risk_tier": tier,
            "classification_decision": decision,
            "severity_p50": p50,
            "severity_lo": lo,
            "severity_hi": hi,
            "recommendation": dec["recommended_action"],
            "human_approval_required": dec["human_approval_required"],
            "robustness": "ROBUST" if prob < 0.6 else "SENSITIVE",
            "Fulfill Via": str(row.get("Fulfill Via", "Direct Drop")),
            "Shipment Mode": str(row.get("Shipment Mode", "Air")),
            "Country": str(row.get("Country", "Unknown")),
            "Line Item Value": val,
            "Forecast_Horizon_Days": float(row.get("Forecast_Horizon_Days", 45.0)),
            "Scheduled_Transit_Days": float(row.get("Scheduled_Transit_Days", 45.0)),
        })

    return df, results


def init_simulation_state(force_reset: bool = False) -> None:
    """Initialize or reset the simulation state in Streamlit session_state."""
    if force_reset or "baseline_df" not in st.session_state:
        raw_df = load_data()
        st.session_state.baseline_df = raw_df.copy()
        st.session_state.active_scenario = "S0"
        st.session_state.custom_params = {
            "lead_time_lag": 0.0,
            "vendor_risk_multiplier": 1.0,
            "line_item_value_pct": 0.0,
            "criticality_boost": False,
        }
        df_scored, results = rescore_cohort(raw_df)
        st.session_state.portfolio_df = df_scored
        st.session_state.portfolio_results = results

        # Set default selected shipment if not set
        if st.session_state.get("selected_shipment_id") is None and results:
            res_df = pd.DataFrame(results)
            st.session_state.selected_shipment_id = res_df.loc[
                res_df["probability_late"].idxmax(), "Shipment ID"
            ]


def trigger_simulation_update(scenario_key: str, custom_params: Dict[str, Any]) -> None:
    """Applies new simulation parameters, re-scores the cohort, and updates session state."""
    init_simulation_state()
    baseline = st.session_state.baseline_df
    mutated_df = apply_scenario_perturbation(baseline, scenario_key, custom_params)
    df_scored, results = rescore_cohort(mutated_df)

    st.session_state.active_scenario = scenario_key
    st.session_state.custom_params = custom_params
    st.session_state.portfolio_df = df_scored
    st.session_state.portfolio_results = results
