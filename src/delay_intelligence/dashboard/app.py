"""Delay Intelligence — Streamlit entry point.

Uses st.navigation / st.Page for professional multi-page navigation.
Houses the Global Simulation Controller and shared session state for dynamic re-scoring.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from delay_intelligence.dashboard.simulation_controller import (
    SCENARIOS,
    init_simulation_state,
    trigger_simulation_update,
)

st.set_page_config(
    page_title="Delay Intelligence",
    page_icon=":material/package_2:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize Simulation Session State ──────────────────────────────────────
init_simulation_state()

# ── Sidebar: Digital Twin & Simulation Controller ────────────────────────────
with st.sidebar:
    st.markdown("### :material/dynamic_form: Simulation Controller")
    st.caption("Inject digital-twin scenarios & custom operational shocks.")

    # 1. Preset Scenarios
    scenario_keys = list(SCENARIOS.keys())
    scenario_labels = [f"{k}: {SCENARIOS[k]['name'].split(': ')[1]}" for k in scenario_keys]
    cur_scen = st.session_state.get("active_scenario", "S0")
    cur_idx = scenario_keys.index(cur_scen) if cur_scen in scenario_keys else 0

    selected_label = st.selectbox(
        "Scenario Preset",
        scenario_labels,
        index=cur_idx,
        key="sidebar_scenario_preset",
        help="Select a synthetic operational disruption scenario to perturb the active cohort.",
    )
    selected_scenario_key = scenario_keys[scenario_labels.index(selected_label)]

    # Scenario details card
    scen_info = SCENARIOS[selected_scenario_key]
    st.info(f"{scen_info['icon']} **{scen_info['badge']}**\n\n{scen_info['description']}")

    # 2. Advanced Custom Perturbation Sliders
    custom_params = st.session_state.get("custom_params", {})
    with st.expander("🛠️ Advanced Perturbations", expanded=False):
        lead_time_lag = st.slider(
            "Added Lead Time (Days)",
            min_value=-10.0,
            max_value=30.0,
            value=float(custom_params.get("lead_time_lag", 0.0)),
            step=1.0,
            help="Simulates port dwell or transit delays.",
        )
        vendor_risk_mult = st.slider(
            "Vendor Risk Multiplier",
            min_value=0.5,
            max_value=2.5,
            value=float(custom_params.get("vendor_risk_multiplier", 1.0)),
            step=0.1,
            help="Scales vendor historical late-delivery probability.",
        )
        value_pct = st.slider(
            "Line Item Value Shift (%)",
            min_value=-50.0,
            max_value=100.0,
            value=float(custom_params.get("line_item_value_pct", 0.0)),
            step=5.0,
            help="Simulates high-value cargo cost fluctuations.",
        )
        crit_boost = st.checkbox(
            "WHO First-Line Priority Boost",
            value=bool(custom_params.get("criticality_boost", False)),
            help="Forces First Line designation on all shipments.",
        )

    # 3. Apply / Reset Action Buttons
    col1, col2 = st.columns(2)
    with col1:
        apply_clicked = st.button("Apply Shock", type="primary", use_container_width=True)
    with col2:
        reset_clicked = st.button("Reset Base", use_container_width=True)

    new_params = {
        "lead_time_lag": lead_time_lag,
        "vendor_risk_multiplier": vendor_risk_mult,
        "line_item_value_pct": value_pct,
        "criticality_boost": crit_boost,
    }

    # Detect if changes occurred or button pressed
    has_param_changed = (
        selected_scenario_key != st.session_state.get("active_scenario")
        or new_params != st.session_state.get("custom_params")
    )

    if apply_clicked or (has_param_changed and not reset_clicked):
        trigger_simulation_update(selected_scenario_key, new_params)
        st.rerun()

    if reset_clicked:
        init_simulation_state(force_reset=True)
        st.rerun()

    if st.session_state.get("active_scenario") != "S0" or any(
        v != 0.0 and v != 1.0 for k, v in st.session_state.get("custom_params", {}).items() if k != "criticality_boost"
    ):
        st.warning("⚠️ **Active Simulation Injected**")

    st.divider()

# ── Navigation ───────────────────────────────────────────────────────────────
page = st.navigation(
    [
        st.Page("app_pages/landing.py", title="Delay Intelligence", icon=":material/package_2:", default=True),
        st.Page("app_pages/executive.py", title="Executive Control Tower", icon=":material/monitoring:"),
        st.Page("app_pages/fleet_map.py", title="Live Fleet & Digital Twin", icon=":material/public:"),
        st.Page("app_pages/explorer.py", title="Shipment Risk Explorer", icon=":material/search:"),
        st.Page("app_pages/action_center.py", title="Decision & Action Center", icon=":material/gavel:"),
        st.Page("app_pages/portfolio.py", title="Portfolio Intelligence", icon=":material/analytics:"),
        st.Page("app_pages/evidence.py", title="Model Evidence", icon=":material/science:"),
    ],
    position="sidebar",
)

page.run()
