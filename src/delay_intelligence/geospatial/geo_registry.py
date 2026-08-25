"""Geo-Coordinate Registry and Transit Position Interpolation Engine.

Provides coordinate mappings for SCMS countries, manufacturing sites, and suppliers,
and computes real-time spherical/geodesic interpolated fleet positions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
import numpy as np
import pandas as pd

# ── Primary Destination Country Coordinates (Lat, Lon) ──────────────────────
COUNTRY_COORDINATES: Dict[str, Tuple[float, float]] = {
    "south africa": (-30.5595, 22.9375),
    "nigeria": (9.0820, 8.6753),
    "kenya": (-0.0236, 37.9062),
    "uganda": (1.3733, 32.2903),
    "tanzania": (-6.3690, 34.8888),
    "united republic of tanzania": (-6.3690, 34.8888),
    "mozambique": (-18.6657, 35.5296),
    "zambia": (-13.1339, 27.8493),
    "zimbabwe": (-19.0154, 29.1549),
    "haiti": (18.9712, -72.2852),
    "rwanda": (-1.9403, 29.8739),
    "vietnam": (14.0583, 108.2772),
    "guyana": (4.8604, -58.9302),
    "côte d'ivoire": (7.5400, -5.5471),
    "cote d'ivoire": (7.5400, -5.5471),
    "ivory coast": (7.5400, -5.5471),
    "ethiopia": (9.1450, 40.4897),
    "congo, drc": (-4.0383, 21.7587),
    "drc": (-4.0383, 21.7587),
    "democratic republic of the congo": (-4.0383, 21.7587),
    "ghana": (7.9465, -1.0232),
    "malawi": (-13.2543, 34.3015),
    "namibia": (-22.9576, 18.4904),
    "botswana": (-22.3285, 24.6849),
    "south sudan": (6.8770, 31.3070),
    "pakistan": (30.3753, 69.3451),
    "dominican republic": (18.7357, -70.1627),
    "guatemala": (15.7835, -90.2308),
    "burundi": (-3.3731, 29.9189),
    "cameroon": (7.3697, 12.3547),
    "swaziland": (-26.5225, 31.4659),
    "eswatini": (-26.5225, 31.4659),
    "lesotho": (-29.6099, 28.2336),
    "benin": (9.3077, 2.3158),
    "liberia": (6.4281, -9.4295),
}

# ── Primary Manufacturing Sites & Origin Hub Coordinates ─────────────────────
MANUFACTURING_COORDINATES: Dict[str, Tuple[float, float]] = {
    # India Hubs
    "aurobindo": (17.3850, 78.4867),  # Hyderabad, India
    "mylan": (19.9975, 73.7898),      # Nashik, India
    "cipla": (18.5204, 73.8567),      # Pune / Kurkumbh, India
    "hetero": (17.3850, 78.4867),     # Hyderabad, India
    "strides": (12.9716, 77.5946),    # Bangalore, India
    "ranbaxy": (30.7333, 76.7794),    # Mohali / Paonta Sahib, India
    "matrix": (19.9975, 73.7898),     # Nashik, India
    "india": (20.5937, 78.9629),      # India centroid
    
    # European Hubs
    "janssen": (41.4676, 12.9037),    # Latina, Italy
    "latina": (41.4676, 12.9037),     # Latina, Italy
    "glaxosmithkline": (51.5074, -0.1278), # London / Ware, UK
    "gsk": (51.5074, -0.1278),
    "roche": (47.5596, 7.5886),       # Basel, Switzerland
    "novartis": (47.5596, 7.5886),    # Basel, Switzerland
    "sanofi": (48.8566, 2.3522),      # Paris / Lyon, France
    "belgium": (50.8503, 4.3517),     # Brussels Hub
    "netherlands": (52.3676, 4.9041), # Amsterdam / Schiphol Hub
    "germany": (50.1109, 8.6821),     # Frankfurt Hub
    
    # North American Hubs
    "abbvie": (42.3042, -87.8920),    # North Chicago, USA
    "abbott": (42.3042, -87.8920),
    "gilead": (37.5585, -122.2711),   # Foster City, California, USA
    "bristol-myers": (40.3573, -74.6672), # New Jersey, USA
    "bms": (40.3573, -74.6672),
    "usa": (39.8283, -98.5795),       # USA centroid
    
    # African Regional Distribution Centers (RDC)
    "aspen": (-33.9608, 25.6022),     # Port Elizabeth, South Africa
    "pharmacy direct": (-25.7479, 28.2293), # Pretoria, South Africa
    "south africa": (-26.2041, 28.0473), # Johannesburg RDC
    "kenya rdc": (-1.2921, 36.8219),  # Nairobi RDC
}

DEFAULT_ORIGIN: Tuple[float, float] = (17.3850, 78.4867)     # Hyderabad, India
DEFAULT_DESTINATION: Tuple[float, float] = (-26.2041, 28.0473) # Johannesburg, South Africa


def get_country_coordinates(country_name: str | None) -> Tuple[float, float]:
    """Resolves destination country to (latitude, longitude)."""
    if not country_name:
        return DEFAULT_DESTINATION
    key = str(country_name).strip().lower()
    if key in COUNTRY_COORDINATES:
        return COUNTRY_COORDINATES[key]
    for c_key, coords in COUNTRY_COORDINATES.items():
        if c_key in key or key in c_key:
            return coords
    return DEFAULT_DESTINATION


def get_manufacturing_coordinates(
    site_name: str | None,
    vendor_name: str | None = None,
) -> Tuple[float, float]:
    """Resolves manufacturing site or vendor name to (latitude, longitude)."""
    text = f"{site_name or ''} {vendor_name or ''}".lower()
    for key, coords in MANUFACTURING_COORDINATES.items():
        if key in text:
            return coords
    return DEFAULT_ORIGIN


def interpolate_transit_position(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    progress: float,
    deviation_km: float = 0.0,
) -> Tuple[float, float]:
    """Computes spherical / geodesic interpolated fleet coordinates.

    Args:
        origin: (lat, lon) of departure.
        destination: (lat, lon) of arrival.
        progress: Alpha fraction in [0.0, 1.0].
        deviation_km: Perturbation offset in kilometers (e.g. S2 route shift).

    Returns:
        (interpolated_lat, interpolated_lon)
    """
    alpha = float(np.clip(progress, 0.0, 1.0))
    lat1, lon1 = math.radians(origin[0]), math.radians(origin[1])
    lat2, lon2 = math.radians(destination[0]), math.radians(destination[1])

    # Spherical Great-Circle Interpolation (Slerp)
    d_lon = lon2 - lon1
    cos_c = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(d_lon)
    cos_c = max(-1.0, min(1.0, cos_c))
    c = math.acos(cos_c)

    if c < 1e-6:
        # Origin and destination are identical
        cur_lat = origin[0]
        cur_lon = origin[1]
    else:
        sin_c = math.sin(c)
        a = math.sin((1.0 - alpha) * c) / sin_c
        b = math.sin(alpha * c) / sin_c

        x = a * math.cos(lat1) * math.cos(lon1) + b * math.cos(lat2) * math.cos(lon2)
        y = a * math.cos(lat1) * math.sin(lon1) + b * math.cos(lat2) * math.sin(lon2)
        z = a * math.sin(lat1) + b * math.sin(lat2)

        cur_lat = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
        cur_lon = math.degrees(math.atan2(y, x))

    # Apply Disruption Drift Perturbation (orthogonally or waypoint drift)
    if deviation_km > 0.0:
        # 1 degree latitude ~ 111 km
        lat_offset = (deviation_km / 111.0) * math.sin(alpha * math.pi)
        lon_offset = (deviation_km / 111.0) * math.cos(alpha * math.pi) * 1.5
        cur_lat += lat_offset
        cur_lon += lon_offset

    return (float(cur_lat), float(cur_lon))


def prepare_fleet_geo_records(
    df: pd.DataFrame,
    results: List[Dict[str, Any]],
    transit_progress: float = 0.50,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generates structured fleet tracking and hub node records for PyDeck map layers.

    Returns:
        (fleet_records, hub_nodes)
    """
    fleet_records: List[Dict[str, Any]] = []
    hub_dict: Dict[str, Dict[str, Any]] = {}

    for i, r in enumerate(results):
        shipment_id = str(r.get("Shipment ID", f"ID_{i}"))
        row = df[df["ID"].astype(str) == shipment_id].iloc[0] if "ID" in df.columns else df.iloc[i]

        dest_country = str(row.get("Country", r.get("Country", "South Africa")))
        mfg_site = str(row.get("Manufacturing Site", ""))
        vendor = str(row.get("Vendor", ""))
        mode = str(r.get("Shipment Mode", row.get("Shipment Mode", "Air")))
        prod_group = str(row.get("Product Group", "ARV"))
        line_val = float(r.get("Line Item Value", row.get("Line Item Value", 10000.0)))
        dev_km = float(row.get("iot_route_deviation_km", 0.0))
        temp_c = row.get("iot_temperature_c", None)

        orig_lat, orig_lon = get_manufacturing_coordinates(mfg_site, vendor)
        dest_lat, dest_lon = get_country_coordinates(dest_country)

        # Compute dynamic interpolated position
        cur_lat, cur_lon = interpolate_transit_position(
            (orig_lat, orig_lon),
            (dest_lat, dest_lon),
            progress=transit_progress,
            deviation_km=dev_km,
        )

        p_late = float(r.get("probability_late", 0.05))
        tier = str(r.get("risk_tier", "LOW_RISK"))
        p50 = float(r.get("severity_p50", 12.0))
        lo = float(r.get("severity_lo", 0.0))
        hi = float(r.get("severity_hi", 30.0))
        rec = str(r.get("recommendation", "NO_ACTION"))

        fleet_records.append({
            "shipment_id": shipment_id,
            "origin_name": mfg_site if mfg_site else "Origin Hub",
            "destination_name": dest_country,
            "origin_lon": orig_lon,
            "origin_lat": orig_lat,
            "dest_lon": dest_lon,
            "dest_lat": dest_lat,
            "current_lon": cur_lon,
            "current_lat": cur_lat,
            "shipment_mode": mode,
            "product_group": prod_group,
            "line_item_value": line_val,
            "probability_late": p_late,
            "risk_tier": tier,
            "severity_p50": p50,
            "severity_lo": lo,
            "severity_hi": hi,
            "recommendation": rec,
            "iot_temperature_c": temp_c if temp_c is not None else "Normal",
            "iot_route_deviation_km": dev_km,
        })

        # Track distinct Origin and Destination Hub landmarks
        orig_key = f"{orig_lon:.2f}_{orig_lat:.2f}"
        if orig_key not in hub_dict:
            hub_dict[orig_key] = {
                "name": mfg_site if mfg_site else "Manufacturing Hub",
                "type": "Origin",
                "lon": orig_lon,
                "lat": orig_lat,
                "count": 0,
            }
        hub_dict[orig_key]["count"] += 1

        dest_key = f"{dest_lon:.2f}_{dest_lat:.2f}"
        if dest_key not in hub_dict:
            hub_dict[dest_key] = {
                "name": dest_country,
                "type": "Destination",
                "lon": dest_lon,
                "lat": dest_lat,
                "count": 0,
            }
        hub_dict[dest_key]["count"] += 1

    return fleet_records, list(hub_dict.values())
