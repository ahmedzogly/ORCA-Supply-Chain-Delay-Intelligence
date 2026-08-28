"""Executive Control Tower — portfolio risk overview and prioritization."""
import pandas as pd
import streamlit as st

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
    risk_badge,
    section_header,
    simulation_banner,
)

# ── Ensure Simulation State is Loaded ────────────────────────────────────────
init_simulation_state()

raw_df = st.session_state.get("portfolio_df")
results = st.session_state.get("portfolio_results")

if not results:
    st.error("No model outputs returned from simulation cohort.")
    st.stop()

res_df = pd.DataFrame(results)

# ── Compute KPIs ─────────────────────────────────────────────────────────────
total = len(results)
mean_risk = float(res_df["probability_late"].mean())
flagged = int(res_df["classification_decision"].sum())
highest_risk = float(res_df["probability_late"].max())
highest_risk_id = str(res_df.loc[res_df["probability_late"].idxmax(), "Shipment ID"])

# Set default demo shipment to highest-risk if not set
if st.session_state.get("selected_shipment_id") is None:
    st.session_state.selected_shipment_id = highest_risk_id

# ── Page Header & Evidence Badges ────────────────────────────────────────────
active_scen_key = st.session_state.get("active_scenario", "S0")
scen_info = SCENARIOS.get(active_scen_key)

if active_scen_key != "S0":
    evidence_badges("REAL DATA", "MODEL OUTPUT", "SIMULATED SCENARIO")
    simulation_banner(scen_info)
else:
    evidence_badges("REAL DATA", "MODEL OUTPUT")

kpi_row([
    {"label": "Shipments monitored", "value": str(total), "help": "Active cohort from SCMS holdout"},
    {"label": "Above decision threshold", "value": str(flagged), "help": "Threshold = 0.23"},
    {"label": "Mean calibrated late risk", "value": format_pct(mean_risk), "help": "Portfolio average"},
    {"label": "Highest portfolio risk", "value": format_pct(highest_risk), "help": f"Shipment {highest_risk_id}"},
])

st.divider()

# ── Risk distribution ────────────────────────────────────────────────────────
section_header("Risk overview", "MODEL OUTPUT")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### Calibrated late-risk distribution")
    chart_data = pd.DataFrame({"Late Probability": res_df["probability_late"]})
    st.bar_chart(chart_data, x_label="Late probability", y_label="Count")

with col2:
    st.markdown("##### Risk tier distribution")
    tier_order = ["LOW_RISK", "WATCH", "HIGH_RISK", "CRITICAL"]
    tier_counts = res_df["risk_tier"].value_counts()
    tier_df = pd.DataFrame({
        "Tier": tier_order,
        "Count": [int(tier_counts.get(t, 0)) for t in tier_order],
    }).set_index("Tier")
    st.bar_chart(tier_df, y_label="Count")

st.divider()

# ── Priority queue ───────────────────────────────────────────────────────────
section_header("Priority shipments", "MODEL OUTPUT")

top_n = min(10, len(res_df))
priority = res_df.nlargest(top_n, "probability_late").copy()

display_df = pd.DataFrame({
    "Shipment ID": priority["Shipment ID"],
    "Late Probability": priority["probability_late"].apply(lambda x: format_pct(x)),
    "Risk Tier": priority["risk_tier"].str.replace("_", " "),
    "Delay P50 if Late": priority["severity_p50"].apply(lambda x: format_days(x)),
    "90% CQR Interval": priority.apply(
        lambda r: f"{r['severity_lo']:.1f}–{r['severity_hi']:.1f} days", axis=1
    ),
    "Recommendation": priority["recommendation"].str.replace("_", " "),
    "Review Required": priority["human_approval_required"].map({True: "Yes", False: "No"}),
})

st.dataframe(
    display_df,
    hide_index=True,
    column_config={
        "Late Probability": st.column_config.TextColumn(width="small"),
        "Risk Tier": st.column_config.TextColumn(width="small"),
    },
)

st.divider()

# ── Investigate CTA ──────────────────────────────────────────────────────────
st.markdown(f"**Highest-risk shipment in the active portfolio:** {highest_risk_id} ({format_pct(highest_risk)})")

if st.button(
    f":material/search: Investigate shipment {highest_risk_id}",
    type="primary",
):
    st.session_state.selected_shipment_id = highest_risk_id
    target_page = str(Path(__file__).resolve().parent / "explorer.py")
    st.switch_page(target_page)


disclaimer_box(
    "No realized ROI is claimed. Recommendations use modeled cost/efficacy "
    "assumptions and are decision-support scenarios only."
)
