"""PyDeck Layer Builders and 3D Visual Styling for Geospatial Digital Twin.

Configures dynamic 3D ArcLayers, ScatterplotLayers, and interactive hover tooltips
color-coded by calibrated risk tiers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
import pydeck as pdk

# ── Calibrated Risk Tier RGBA Color Palette ──────────────────────────────────
RISK_COLORS: Dict[str, List[int]] = {
    "LOW_RISK": [0, 230, 118, 200],    # Neon Green
    "WATCH": [255, 214, 0, 200],       # Amber Yellow
    "HIGH_RISK": [255, 145, 0, 220],   # Warning Orange
    "CRITICAL": [255, 23, 68, 255],    # Crimson Red
}

HUB_COLOR_ORIGIN = [66, 165, 245, 180]      # Electric Blue
HUB_COLOR_DESTINATION = [171, 71, 188, 180] # Vibrant Purple


def get_risk_rgba(tier: str) -> List[int]:
    """Returns RGBA color corresponding to risk tier."""
    return RISK_COLORS.get(tier, [0, 230, 118, 200])


def enrich_fleet_records_for_deck(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enriches fleet records with visual styling attributes (colors, radiuses, formatted strings)."""
    enriched: List[Dict[str, Any]] = []

    for r in records:
        rec = dict(r)
        tier = rec.get("risk_tier", "LOW_RISK")
        p_late = float(rec.get("probability_late", 0.05))
        val = float(rec.get("line_item_value", 10000.0))
        p50 = float(rec.get("severity_p50", 12.0))
        lo = float(rec.get("severity_lo", 0.0))
        hi = float(rec.get("severity_hi", 30.0))

        color = get_risk_rgba(tier)
        rec["marker_color"] = color
        rec["arc_color"] = color

        # Scale radius smoothly based on cargo value and risk
        base_radius = 45000 + math.log1p(val) * 6000
        if tier in ("HIGH_RISK", "CRITICAL"):
            base_radius *= 1.3
        rec["marker_radius"] = base_radius

        # Arc line width proportional to value
        rec["arc_width"] = max(1.5, min(8.0, 1.0 + math.log1p(val) * 0.4))

        # Format tooltip text fields
        rec["formatted_prob"] = f"{p_late:.1%}"
        rec["formatted_p50"] = f"{p50:.1f} days"
        rec["formatted_interval"] = f"[{lo:.1f} – {hi:.1f}] days"
        rec["formatted_val"] = f"${val:,.0f}"

        enriched.append(rec)

    return enriched


def build_fleet_deck_map(
    fleet_records: List[Dict[str, Any]],
    hub_nodes: List[Dict[str, Any]],
    initial_view: Tuple[float, float, int] = (5.0, 25.0, 2),
) -> pdk.Deck:
    """Builds a complete PyDeck 3D visualization deck.

    Args:
        fleet_records: Formatted fleet data list.
        hub_nodes: Origin and Destination landmark nodes.
        initial_view: (center_lat, center_lon, zoom_level).

    Returns:
        pdk.Deck configured with 3D ArcLayer and ScatterplotLayers.
    """
    enriched_fleet = enrich_fleet_records_for_deck(fleet_records)

    # 1. 3D Great-Circle Transit Arcs Layer
    arc_layer = pdk.Layer(
        "ArcLayer",
        data=enriched_fleet,
        id="transit-arcs-layer",
        get_source_position=["origin_lon", "origin_lat"],
        get_target_position=["dest_lon", "dest_lat"],
        get_source_color="arc_color",
        get_target_color="arc_color",
        get_width="arc_width",
        get_height=0.45,
        tilt=15,
        pickable=True,
        auto_highlight=True,
    )

    # 2. Live Fleet Position Markers Layer
    fleet_layer = pdk.Layer(
        "ScatterplotLayer",
        data=enriched_fleet,
        id="fleet-markers-layer",
        get_position=["current_lon", "current_lat"],
        get_fill_color="marker_color",
        get_line_color=[255, 255, 255, 220],
        line_width_min_pixels=1.5,
        get_radius="marker_radius",
        radius_min_pixels=6,
        radius_max_pixels=18,
        pickable=True,
        auto_highlight=True,
    )

    # 3. Fixed Logistics Hub Landmarks Layer
    for h in hub_nodes:
        h["color"] = HUB_COLOR_ORIGIN if h.get("type") == "Origin" else HUB_COLOR_DESTINATION
        h["radius"] = 35000 + min(50000, h.get("count", 1) * 3000)

    hub_layer = pdk.Layer(
        "ScatterplotLayer",
        data=hub_nodes,
        id="logistics-hubs-layer",
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_line_color=[255, 255, 255, 180],
        line_width_min_pixels=1.0,
        get_radius="radius",
        radius_min_pixels=5,
        radius_max_pixels=14,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=initial_view[0],
        longitude=initial_view[1],
        zoom=initial_view[2],
        pitch=40,
        bearing=0,
    )

    tooltip_html = {
        "html": """
        <div style="background-color: #0e1117; color: #ffffff; padding: 10px; border-radius: 8px; font-family: sans-serif; font-size: 12px; border: 1px solid #30363d; box-shadow: 0 4px 12px rgba(0,0,0,0.5);">
            <div style="font-weight: bold; font-size: 14px; margin-bottom: 4px; color: #58a6ff;">📦 Shipment {shipment_id}</div>
            <div style="margin-bottom: 6px; color: #8b949e;">{origin_name} ➔ {destination_name} ({shipment_mode})</div>
            <hr style="border-color: #30363d; margin: 4px 0;" />
            <div><b>Calibrated Late Risk:</b> <span style="color: #ff7b72;">{formatted_prob}</span> [{risk_tier}]</div>
            <div><b>Expected Delay (P50):</b> {formatted_p50}</div>
            <div><b>90% CQR Interval:</b> {formatted_interval}</div>
            <div><b>Cargo Value:</b> {formatted_val} ({product_group})</div>
            <div style="margin-top: 4px; font-size: 11px; color: #e3b341;"><b>IoT Telemetry:</b> Temp={iot_temperature_c}, Deviation={iot_route_deviation_km}km</div>
            <div style="margin-top: 2px; color: #3fb950;"><b>Recommendation:</b> {recommendation}</div>
        </div>
        """,
        "style": {
            "backgroundColor": "transparent",
            "zIndex": "1000",
        },
    }

    deck = pdk.Deck(
        layers=[arc_layer, hub_layer, fleet_layer],
        initial_view_state=view_state,
        map_style="dark",
        tooltip=tooltip_html,
    )

    return deck
