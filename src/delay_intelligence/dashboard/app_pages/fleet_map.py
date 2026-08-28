"""Live Fleet & Geospatial Digital Twin — 3D interactive global fleet tracking."""
from pathlib import Path
import pandas as pd
import streamlit as st


from delay_intelligence.dashboard.simulation_controller import (
    SCENARIOS,
    init_simulation_state,
    trigger_simulation_update,
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
from delay_intelligence.geospatial.deck_layers import build_fleet_deck_map
from delay_intelligence.geospatial.geo_registry import prepare_fleet_geo_records

# ── Ensure Simulation State is Loaded ────────────────────────────────────────
init_simulation_state()

df = st.session_state.get("portfolio_df")
results = st.session_state.get("portfolio_results")

if not results or df is None:
    st.error("No fleet records available.")
    st.stop()

# ── Page Header & Context ────────────────────────────────────────────────────
active_scen_key = st.session_state.get("active_scenario", "S0")
scen_info = SCENARIOS.get(active_scen_key)

st.title("🌐 Live Fleet & Geospatial Digital Twin")
st.caption(
    "3D Great-Circle Transit Tracking, Real-Time Geo-Interpolation, and Risk-Reactive Layering."
)

if active_scen_key != "S0":
    evidence_badges("REAL DATA", "MODEL OUTPUT", "SIMULATED SCENARIO")
    simulation_banner(scen_info)
else:
    evidence_badges("REAL DATA", "MODEL OUTPUT")

# ── Compute Executive Fleet KPIs ─────────────────────────────────────────────
res_df = pd.DataFrame(results)
total_fleet = len(results)
high_risk_count = int(res_df["risk_tier"].isin(["HIGH_RISK", "CRITICAL"]).sum())
mean_risk = float(res_df["probability_late"].mean())
mean_p50 = float(res_df["severity_p50"].mean())

kpi_row([
    {"label": "Active Fleet Monitored", "value": str(total_fleet), "help": "Active in-transit cohort"},
    {"label": "High-Risk Shipments", "value": str(high_risk_count), "help": "Calibrated late probability > 60%"},
    {"label": "Mean Late Risk", "value": format_pct(mean_risk), "help": "Portfolio average risk"},
    {"label": "Mean Expected Delay if Late", "value": format_days(mean_p50), "help": "LightGBM conditional P50"},
])

st.divider()

# ── Interactive Simulation & Control Bar ─────────────────────────────────────
st.markdown("##### 🎛️ Digital Twin Simulation & View Controls")

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.5, 2.0])

with ctrl_col1:
    # Quick Disruption Regime Injection
    scenario_keys = list(SCENARIOS.keys())
    scenario_labels = [f"{k}: {SCENARIOS[k]['name'].split(': ')[1]}" for k in scenario_keys]
    cur_idx = scenario_keys.index(active_scen_key) if active_scen_key in scenario_keys else 0

    selected_scen_label = st.selectbox(
        "Inject Scenario Shock",
        scenario_labels,
        index=cur_idx,
        key="fleet_scenario_quick_select",
        help="Perturb active fleet risk in real time.",
    )
    new_scen_key = scenario_keys[scenario_labels.index(selected_scen_label)]
    if new_scen_key != active_scen_key:
        trigger_simulation_update(new_scen_key, st.session_state.get("custom_params", {}))
        st.rerun()

with ctrl_col2:
    # Mode filter
    available_modes = sorted(list(set(res_df["Shipment Mode"].dropna())))
    selected_modes = st.multiselect(
        "Filter by Mode",
        options=available_modes,
        default=available_modes,
        key="fleet_mode_filter",
    )

with ctrl_col3:
    # Risk Tier filter
    all_tiers = ["LOW_RISK", "WATCH", "HIGH_RISK", "CRITICAL"]
    selected_tiers = st.multiselect(
        "Filter by Risk Tier",
        options=all_tiers,
        default=all_tiers,
        key="fleet_tier_filter",
    )

# Transit Progress Animation Slider
transit_progress_pct = st.slider(
    "⏱️ Transit Timeline Simulation (Elapsed Flight / Voyage Progress)",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    format="%d%%",
    help="Interpolates fleet markers along great-circle arcs from Departure (0%) to Arrival (100%).",
)
transit_progress = float(transit_progress_pct) / 100.0

# ── Prepare Geo Records & Apply Filters ──────────────────────────────────────
all_fleet_records, hub_nodes = prepare_fleet_geo_records(
    df, results, transit_progress=transit_progress
)

filtered_fleet = [
    r for r in all_fleet_records
    if r["shipment_mode"] in selected_modes and r["risk_tier"] in selected_tiers
]

# ── 3D PyDeck Map Rendering ──────────────────────────────────────────────────
section_header("3D Global Fleet Tracking & Telemetry Map", "MODEL OUTPUT")

if not filtered_fleet:
    st.warning("No shipments match the selected filters.")
else:
    # Center map on Africa / Indian Ocean region
    deck_map = build_fleet_deck_map(
        filtered_fleet, hub_nodes, initial_view=(4.0, 32.0, 2)
    )
    st.pydeck_chart(deck_map, use_container_width=True)

# ── Map Legend & Telemetry Guide ─────────────────────────────────────────────
leg_c1, leg_c2, leg_c3, leg_c4 = st.columns(4)
leg_c1.markdown("🟢 **Low Risk (≤30%)**")
leg_c2.markdown("🟡 **Watch (30%–60%)**")
leg_c3.markdown("🟠 **High Risk (60%–85%)**")
leg_c4.markdown("🔴 **Critical (>85%)**")

st.divider()

# ── Quick Action Drawer & Target Inspection ──────────────────────────────────
section_header("Shipment Quick Action Drawer", "MODEL OUTPUT")

filtered_ids = [r["shipment_id"] for r in filtered_fleet]
if not filtered_ids:
    filtered_ids = [str(r["Shipment ID"]) for r in results]

default_shipment = str(st.session_state.get("selected_shipment_id", filtered_ids[0]))
if default_shipment not in filtered_ids:
    default_shipment = filtered_ids[0]

inspect_id = st.selectbox(
    "Select Shipment from Map",
    filtered_ids,
    index=filtered_ids.index(default_shipment),
    key="fleet_inspect_select",
)
st.session_state.selected_shipment_id = inspect_id

target_record = next((r for r in all_fleet_records if r["shipment_id"] == inspect_id), None)

if target_record:
    card_c1, card_c2, card_c3, card_c4 = st.columns([1.5, 1.5, 1.5, 1.5])
    with card_c1:
        st.markdown(f"**Route:** {target_record['origin_name']} ➔ {target_record['destination_name']}")
        st.markdown(f"**Mode:** {target_record['shipment_mode']}")
    with card_c2:
        st.markdown(f"**Calibrated Late Risk:** {format_pct(target_record['probability_late'])}")
        risk_badge(target_record["risk_tier"])
    with card_c3:
        st.markdown(f"**Delay (P50):** {format_days(target_record['severity_p50'])}")
        st.markdown(f"**90% CQR:** {target_record['severity_lo']:.1f}–{target_record['severity_hi']:.1f} days")
    with card_c4:
        st.markdown(f"**Action:** `{target_record['recommendation']}`")
        if st.button("🔍 Investigate in Explorer", type="primary", use_container_width=True):
            st.session_state.selected_shipment_id = inspect_id
            target_page = str(Path(__file__).resolve().parent / "explorer.py")
            st.switch_page(target_page)


disclaimer_box(
    "Great-circle arcs and interpolated coordinates are spatial representations "
    "parameterized by origin manufacturing hubs and destination country centroids. "
    "Telemetry offsets represent scenario perturbations."
)
