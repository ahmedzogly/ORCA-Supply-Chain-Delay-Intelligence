"""Tests for dynamic simulation controller and digital twin scenario injection."""
import pandas as pd
import pytest

from delay_intelligence.dashboard.api_client import load_data
from delay_intelligence.dashboard.simulation_controller import (
    SCENARIOS,
    apply_scenario_perturbation,
    rescore_cohort,
)


def test_simulation_scenarios_defined():
    expected = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
    for k in expected:
        assert k in SCENARIOS
        assert "name" in SCENARIOS[k]
        assert "badge" in SCENARIOS[k]
        assert "description" in SCENARIOS[k]


def test_scenario_perturbation_mutations():
    df = load_data(limit=10)

    # S0 Baseline
    s0 = apply_scenario_perturbation(df, "S0")
    assert s0["Forecast_Horizon_Days"].equals(df["Forecast_Horizon_Days"])

    # S1 Cold-chain excursion
    s1 = apply_scenario_perturbation(df, "S1")
    assert s1["iot_temperature_c"].iloc[0] == 9.5
    assert s1["vendor_hist_delay_rate"].iloc[0] >= df["vendor_hist_delay_rate"].iloc[0]

    # S2 Port congestion
    s2 = apply_scenario_perturbation(df, "S2")
    assert s2["Scheduled_Transit_Days"].iloc[0] > df["Scheduled_Transit_Days"].iloc[0]
    assert s2["iot_route_deviation_km"].iloc[0] == 25.0

    # S3 Customs slowdown
    s3 = apply_scenario_perturbation(df, "S3")
    assert s3["Forecast_Horizon_Days"].iloc[0] == df["Forecast_Horizon_Days"].iloc[0] + 5.0

    # S4 Carrier shock
    s4 = apply_scenario_perturbation(df, "S4")
    assert s4["Forecast_Horizon_Days"].iloc[0] == df["Forecast_Horizon_Days"].iloc[0] + 10.0

    # S5 Compound crisis
    s5 = apply_scenario_perturbation(df, "S5")
    assert s5["iot_temperature_c"].iloc[0] == 10.2
    assert s5["iot_route_deviation_km"].iloc[0] == 45.0

    # S6 Post-intervention recovery
    s6 = apply_scenario_perturbation(df, "S6")
    assert s6["vendor_hist_delay_rate"].iloc[0] <= df["vendor_hist_delay_rate"].iloc[0]


def test_custom_sliders_perturbation():
    df = load_data(limit=10)
    custom = {
        "lead_time_lag": 15.0,
        "vendor_risk_multiplier": 1.5,
        "line_item_value_pct": 20.0,
        "criticality_boost": True,
    }
    mutated = apply_scenario_perturbation(df, "S0", custom_params=custom)

    assert mutated["Forecast_Horizon_Days"].iloc[0] == df["Forecast_Horizon_Days"].iloc[0] + 15.0
    assert mutated["First Line Designation"].iloc[0] == "Yes"
    assert mutated["Line Item Value"].iloc[0] == df["Line Item Value"].iloc[0] * 1.2


def test_rescore_cohort_output_structure():
    df = load_data(limit=10)
    df_out, results = rescore_cohort(df)

    assert len(results) == len(df)
    for r in results:
        assert "Shipment ID" in r
        assert "probability_late" in r
        assert 0.0 <= r["probability_late"] <= 1.0
        assert "risk_tier" in r
        assert r["risk_tier"] in ["LOW_RISK", "WATCH", "HIGH_RISK", "CRITICAL"]
        assert "severity_p50" in r
        assert r["severity_p50"] >= 0.0
        assert "severity_lo" in r
        assert "severity_hi" in r
        assert r["severity_lo"] <= r["severity_hi"]
        assert "recommendation" in r
        assert "human_approval_required" in r
