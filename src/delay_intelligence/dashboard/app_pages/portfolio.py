"""Portfolio Intelligence — aggregate model-output views and operational storytelling."""
import numpy as np
import pandas as pd
import streamlit as st

from delay_intelligence.dashboard.api_client import (
    api_explain,
    load_data,
    row_to_features,
)
from delay_intelligence.dashboard.simulation_controller import (
    SCENARIOS,
    init_simulation_state,
)
from delay_intelligence.dashboard.ui import (
    disclaimer_box,
    evidence_badges,
    format_days,
    format_pct,
    kpi_row,
    section_header,
    simulation_banner,
)

# ── Ensure Simulation State is Loaded ────────────────────────────────────────
init_simulation_state()

df = st.session_state.get("portfolio_df")
results = st.session_state.get("portfolio_results")

if not results:
    st.warning("No results available.")
    st.stop()

records = []
for r in results:
    records.append({
        "Shipment ID": r["Shipment ID"],
        "Risk Tier": r["risk_tier"],
        "Late Probability": r["probability_late"],
        "Severity P50 if Late": r["severity_p50"],
        "Severity Lo": r["severity_lo"],
        "Severity Hi": r["severity_hi"],
        "Fulfillment Channel": r.get("Fulfill Via", "Direct Drop"),
        "Shipment Mode": r.get("Shipment Mode", "Air"),
    })

res = pd.DataFrame(records)

# ── Portfolio overview ───────────────────────────────────────────────────────
active_scen_key = st.session_state.get("active_scenario", "S0")
scen_info = SCENARIOS.get(active_scen_key)

if active_scen_key != "S0":
    evidence_badges("REAL DATA", "MODEL OUTPUT", "SIMULATED SCENARIO")
    simulation_banner(scen_info)
else:
    evidence_badges("REAL DATA", "MODEL OUTPUT")

mean_risk = float(res["Late Probability"].mean())
median_severity = float(res["Severity P50 if Late"].median())
high_risk_count = int((res["Risk Tier"].isin(["HIGH_RISK", "CRITICAL"])).sum())
highest_id = str(res.loc[res["Late Probability"].idxmax(), "Shipment ID"])

section_header("Portfolio overview", "MODEL OUTPUT")

kpi_row([
    {"label": "Mean late risk", "value": format_pct(mean_risk)},
    {"label": "Median conditional delay", "value": format_days(median_severity)},
    {"label": "High-risk count", "value": str(high_risk_count)},
    {"label": "Highest-risk shipment in the active portfolio", "value": str(highest_id)},
])

disclaimer_box(f"Based on {len(res)} monitored shipments under active simulated conditions.")

st.divider()

# ── Risk by fulfillment channel ──────────────────────────────────────────────
section_header("Risk by fulfillment channel", "MODEL OUTPUT")

channel_stats = res.groupby("Fulfillment Channel").agg(
    Count=("Late Probability", "count"),
    Mean_Risk=("Late Probability", "mean"),
).reset_index()
channel_stats["Mean Late Risk"] = channel_stats["Mean_Risk"].apply(lambda x: format_pct(x))
st.dataframe(
    channel_stats[["Fulfillment Channel", "Count", "Mean Late Risk"]],
    hide_index=True,
)

channel_chart = res.groupby(["Fulfillment Channel", "Risk Tier"]).size().unstack(fill_value=0)
st.bar_chart(channel_chart)

st.divider()

# ── Risk by shipment mode ────────────────────────────────────────────────────
section_header("Risk by shipment mode", "MODEL OUTPUT")

mode_stats = res.groupby("Shipment Mode").agg(
    Count=("Late Probability", "count"),
    Mean_Risk=("Late Probability", "mean"),
).reset_index()
mode_stats["Mean Late Risk"] = mode_stats["Mean_Risk"].apply(lambda x: format_pct(x))
st.dataframe(
    mode_stats[["Shipment Mode", "Count", "Mean Late Risk"]],
    hide_index=True,
)

mode_chart = res.groupby("Shipment Mode")["Late Probability"].mean()
st.bar_chart(mode_chart, y_label="Mean Late Probability")

st.divider()

# ── Delay severity distribution ──────────────────────────────────────────────
section_header("Conditional delay severity distribution", "MODEL OUTPUT")

severity_vals = res["Severity P50 if Late"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Median", format_days(float(severity_vals.median())))
col2.metric("P75", format_days(float(severity_vals.quantile(0.75))))
col3.metric("P90", format_days(float(severity_vals.quantile(0.90))))
col4.metric("Maximum", format_days(float(severity_vals.max())))

severity_chart = pd.DataFrame({"Conditional Delay P50 (days)": severity_vals})
st.bar_chart(severity_chart, x_label="Conditional delay P50 (days)", y_label="Count")

with st.expander("Technical summary statistics"):
    st.dataframe(
        severity_vals.describe().to_frame("days"),
    )

st.divider()

# ── Top risk drivers across portfolio ────────────────────────────────────────
section_header("Top risk drivers across portfolio", "MODEL OUTPUT")
st.caption("Aggregated |SHAP| across the top-10 highest-risk shipments in active cohort.")


def _compute_top_shap(df_cohort: pd.DataFrame, top_shipments: list[str]) -> pd.DataFrame:
    shap_accum: dict[str, float] = {}
    if df_cohort is None or df_cohort.empty:
        return pd.DataFrame()

    matched_rows = df_cohort[df_cohort["ID"].astype(str).isin(top_shipments)]
    if matched_rows.empty:
        matched_rows = df_cohort.head(10)

    for _, row in matched_rows.iterrows():
        try:
            f = row_to_features(row)
            expl = api_explain(f)
            for contrib in expl.get("shap_contributions", []):
                feat = contrib["feature"]
                shap_accum[feat] = shap_accum.get(feat, 0.0) + abs(contrib["shap_value"])
        except Exception:
            continue

    if not shap_accum:
        return pd.DataFrame()

    shap_df = pd.DataFrame([
        {"Feature": k, "Mean |SHAP|": v / len(matched_rows)}
        for k, v in sorted(shap_accum.items(), key=lambda x: x[1], reverse=True)[:10]
    ])
    return shap_df


top_10_ids = res.nlargest(10, "Late Probability")["Shipment ID"].tolist()
shap_df = _compute_top_shap(df, top_10_ids)

if not shap_df.empty:
    chart_shap = shap_df.set_index("Feature")
    st.bar_chart(chart_shap, horizontal=True)
else:
    st.info("Portfolio SHAP aggregation unavailable.")

disclaimer_box(
    "Predicted portfolio states are model outputs from the active simulated cohort. "
    "This page does not present them as observed future outcomes."
)
