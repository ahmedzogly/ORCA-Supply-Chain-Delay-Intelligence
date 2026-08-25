"""Unit tests for Geospatial Digital Twin and Fleet Tracking Map."""
import pydeck as pdk
import pytest

from delay_intelligence.dashboard.api_client import load_data
from delay_intelligence.dashboard.simulation_controller import rescore_cohort
from delay_intelligence.geospatial.deck_layers import (
    RISK_COLORS,
    build_fleet_deck_map,
    enrich_fleet_records_for_deck,
    get_risk_rgba,
)
from delay_intelligence.geospatial.geo_registry import (
    get_country_coordinates,
    get_manufacturing_coordinates,
    interpolate_transit_position,
    prepare_fleet_geo_records,
)


def test_country_coordinates_resolution():
    sa = get_country_coordinates("South Africa")
    assert isinstance(sa, tuple)
    assert len(sa) == 2
    assert sa[0] < 0  # Southern hemisphere

    kenya = get_country_coordinates("Kenya")
    assert kenya[0] != 0.0 or kenya[1] != 0.0

    # Test unknown fallback
    fallback = get_country_coordinates("Atlantis")
    assert isinstance(fallback, tuple)
    assert len(fallback) == 2


def test_manufacturing_coordinates_resolution():
    auro = get_manufacturing_coordinates("Aurobindo Unit III, India")
    assert isinstance(auro, tuple)
    assert auro[1] > 70  # Indian longitude

    janssen = get_manufacturing_coordinates("Janssen-Cilag, Latina, IT")
    assert isinstance(janssen, tuple)
    assert janssen[0] > 40  # Italy latitude

    fallback = get_manufacturing_coordinates("Unknown Supplier Site")
    assert isinstance(fallback, tuple)


def test_interpolate_transit_position_boundaries():
    orig = (17.3850, 78.4867)   # Hyderabad
    dest = (-26.2041, 28.0473)  # Johannesburg

    # Alpha = 0.0 (Departure)
    pos_0 = interpolate_transit_position(orig, dest, progress=0.0)
    assert abs(pos_0[0] - orig[0]) < 1e-3
    assert abs(pos_0[1] - orig[1]) < 1e-3

    # Alpha = 1.0 (Arrival)
    pos_1 = interpolate_transit_position(orig, dest, progress=1.0)
    assert abs(pos_1[0] - dest[0]) < 1e-3
    assert abs(pos_1[1] - dest[1]) < 1e-3

    # Alpha = 0.5 (Midpoint)
    pos_mid = interpolate_transit_position(orig, dest, progress=0.5)
    assert min(orig[0], dest[0]) <= pos_mid[0] <= max(orig[0], dest[0])


def test_interpolate_with_deviation_perturbation():
    orig = (17.3850, 78.4867)
    dest = (-26.2041, 28.0473)

    pos_clean = interpolate_transit_position(orig, dest, progress=0.5, deviation_km=0.0)
    pos_deviated = interpolate_transit_position(orig, dest, progress=0.5, deviation_km=50.0)

    assert pos_clean != pos_deviated
    assert abs(pos_deviated[0] - pos_clean[0]) > 0.0


def test_prepare_fleet_geo_records_and_deck_construction():
    df = load_data(limit=10)
    df_scored, results = rescore_cohort(df)

    fleet_records, hub_nodes = prepare_fleet_geo_records(df_scored, results, transit_progress=0.45)
    assert len(fleet_records) == len(results)
    assert len(hub_nodes) > 0

    first = fleet_records[0]
    required_keys = [
        "shipment_id", "origin_lon", "origin_lat", "dest_lon", "dest_lat",
        "current_lon", "current_lat", "probability_late", "risk_tier",
        "severity_p50", "severity_lo", "severity_hi", "recommendation",
    ]
    for k in required_keys:
        assert k in first

    # Enrich and build deck map
    deck = build_fleet_deck_map(fleet_records, hub_nodes)
    assert isinstance(deck, pdk.Deck)
    assert len(deck.layers) == 3


def test_risk_color_palette():
    for tier in ["LOW_RISK", "WATCH", "HIGH_RISK", "CRITICAL"]:
        color = get_risk_rgba(tier)
        assert len(color) == 4
        assert all(0 <= c <= 255 for c in color)
