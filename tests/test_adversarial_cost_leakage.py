"""
Adversarial Test Suite for Cost Scenario Engine (Phase 2 — Experiment E8).
Tests:
1. Systematic forbidden column rejection (DataFrame, Series, Dict, combinations).
2. Strict leakage validation behavior across all entry points.
3. Log vs. un-logged monetary value handling and double-logging prevention.
4. Input boundary, edge cases, NaN/Inf, negative, empty data, and extreme value resilience.
5. Invariant preservation (non-negativity, monotonicity, probability bounds [0, 1]).
6. Scenario hierarchy consistency (Low < Base < High).
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from delay_intelligence.cost_sensitive.cost_engine import (
    CostScenario,
    CostScenarioModel,
    CostEngine,
    LeakageViolationError,
    CostBreakdown,
    FORBIDDEN_COLUMNS,
)


@pytest.fixture
def config_path():
    return Path("configs/cost_scenarios.yaml")


@pytest.fixture
def engine(config_path):
    return CostScenarioModel(config_path=config_path, scenario_name="base")


# =============================================================================
# 1. FORBIDDEN COLUMN LEAKAGE ADVERSARIAL MATRIX
# =============================================================================

@pytest.mark.parametrize("forbidden_col", FORBIDDEN_COLUMNS)
def test_adversarial_single_forbidden_column_dataframe(engine, forbidden_col):
    """Verify that EVERY forbidden column triggers LeakageViolationError in DataFrame."""
    df_leak = pd.DataFrame({
        "Line Item Value": [50000.0, 120000.0],
        "Shipment Mode": ["Air", "Truck"],
        forbidden_col: [100.0, 200.0],
    })
    with pytest.raises(LeakageViolationError, match="Forbidden / target-leakage column"):
        engine.compute_costs(df_leak, strict_leakage_check=True)

    with pytest.raises(LeakageViolationError):
        engine.validate_features(df_leak, strict_leakage_check=True)


@pytest.mark.parametrize("forbidden_col", FORBIDDEN_COLUMNS)
def test_adversarial_single_forbidden_column_series(engine, forbidden_col):
    """Verify that EVERY forbidden column triggers LeakageViolationError in Series."""
    series_leak = pd.Series({
        "Line Item Value": 50000.0,
        "Shipment Mode": "Air",
        forbidden_col: "2012-05-01",
    })
    with pytest.raises(LeakageViolationError, match="Forbidden / target-leakage column"):
        engine.compute_costs(series_leak, strict_leakage_check=True)

    with pytest.raises(LeakageViolationError):
        engine.validate_features(series_leak, strict_leakage_check=True)


@pytest.mark.parametrize("forbidden_col", FORBIDDEN_COLUMNS)
def test_adversarial_single_forbidden_column_dict(engine, forbidden_col):
    """Verify that EVERY forbidden column triggers LeakageViolationError in Dict."""
    dict_leak = {
        "Line Item Value": 50000.0,
        "Shipment Mode": "Air",
        forbidden_col: 1,
    }
    with pytest.raises(LeakageViolationError, match="Forbidden / target-leakage column"):
        engine.compute_costs(dict_leak, strict_leakage_check=True)

    with pytest.raises(LeakageViolationError):
        engine.validate_features(dict_leak, strict_leakage_check=True)


def test_adversarial_all_forbidden_columns_simultaneous(engine):
    """Verify that passing all 8 forbidden columns simultaneously raises LeakageViolationError listing all of them."""
    full_leak = {col: 1.0 for col in FORBIDDEN_COLUMNS}
    full_leak["Line Item Value"] = 100000.0
    full_leak["Shipment Mode"] = "Ocean"

    df_full_leak = pd.DataFrame([full_leak])

    with pytest.raises(LeakageViolationError) as exc_info:
        engine.compute_costs(df_full_leak, strict_leakage_check=True)

    err_msg = str(exc_info.value)
    for col in FORBIDDEN_COLUMNS:
        assert col in err_msg, f"Expected {col} to be reported in violation error message"


def test_adversarial_sample_weights_and_benefit_ranking_enforce_leakage(engine):
    """Verify sample weights and ranking functions strictly reject forbidden columns."""
    df_leak = pd.DataFrame({
        "Line Item Value": [10000.0, 20000.0],
        "Delay_Days": [5, -2],  # Forbidden
    })
    y_true = np.array([1, 0])
    p_hat = np.array([0.9, 0.1])

    with pytest.raises(LeakageViolationError):
        engine.compute_sample_weights(df_leak, y_true)

    with pytest.raises(LeakageViolationError):
        engine.compute_expected_net_benefit_ranking(df_leak, p_hat, strict_leakage_check=True)


# =============================================================================
# 2. LOG VS UN-LOGGED VALUE HANDLING & DOUBLE-LOGGING DEFENSE
# =============================================================================

def test_adversarial_monetary_extraction_identity(engine):
    """
    Stress-test extract_monetary_values across a wide range of values
    to ensure expm1(log1p(v)) == v and no accidental double-logging.
    """
    test_values = [0.0, 0.5, 1.0, 10.0, 100.0, 1000.0, 50000.0, 1e6, 1e8]

    for val in test_values:
        # Raw value with is_log_transformed=False
        arr_raw = engine.extract_monetary_values([val], is_log_transformed=False)
        assert math.isclose(arr_raw[0], val, rel_tol=1e-7, abs_tol=1e-7)

        # Logged value with is_log_transformed=True
        log_val = math.log1p(val)
        arr_unlog = engine.extract_monetary_values([log_val], is_log_transformed=True)
        assert math.isclose(arr_unlog[0], val, rel_tol=1e-6, abs_tol=1e-6)


def test_adversarial_explicit_unlogged_flag_prevents_false_autodetect(engine):
    """
    If a shipment has a very small raw dollar value (e.g. $10.0), passing
    is_log_transformed=False MUST preserve $10.0 and NOT falsely treat it as log(1+V).
    """
    small_dollar = 10.0
    df_small = pd.DataFrame({"Line Item Value": [small_dollar]})

    res_explicit = engine.compute_costs(df_small, is_log_transformed=False, return_dataframe=False)
    # With V = 10.0:
    # daily holding penalty = 150.0 + 0.0010 * 10 = 150.01
    # FN = 1.0 * 1.0 * (500.0 + 150.01 * 12.0) = 500 + 1800.12 = 2300.12
    expected_fn = 500.0 + (150.0 + 0.0010 * 10.0) * 12.0
    assert math.isclose(res_explicit.fn_cost[0], expected_fn, rel_tol=1e-5)


def test_adversarial_negative_and_nan_monetary_values(engine):
    """Ensure negative and NaN monetary values are safely coerced to 0.0 without crash or NaN."""
    df_dirty = pd.DataFrame({
        "Line Item Value": [-500.0, np.nan, -0.0001, np.inf],
        "Shipment Mode": ["Air", "Truck", "Ocean", "Air Charter"],
    })
    vals = engine.extract_monetary_values(df_dirty, is_log_transformed=False)
    assert vals[0] == 0.0  # -500 -> 0.0
    assert vals[1] == 0.0  # NaN -> 0.0
    assert vals[2] == 0.0  # -0.0001 -> 0.0
    assert np.isfinite(vals[3])  # inf converted to finite


# =============================================================================
# 3. INPUT BOUNDARIES, MISSING COLUMNS, AND ROBUSTNESS
# =============================================================================

def test_adversarial_empty_dataframe(engine):
    """Empty dataframe input must return empty DataFrame with expected columns, no crash."""
    df_empty = pd.DataFrame(columns=["Line Item Value", "Shipment Mode"])
    res = engine.compute_costs(df_empty, is_log_transformed=False)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert "fn_cost" in res.columns
    assert "fp_cost" in res.columns
    assert "tau_star" in res.columns


def test_adversarial_missing_all_optional_columns(engine):
    """Dataframe with minimal columns (even without Line Item Value) falls back safely."""
    df_minimal = pd.DataFrame({"some_other_col": [1, 2, 3]})
    res = engine.compute_costs(df_minimal, is_log_transformed=False)
    assert len(res) == 3
    assert not res.isna().any().any()
    assert (res["fn_cost"] > 0).all()
    assert (res["fp_cost"] > 0).all()


def test_adversarial_categorical_edge_cases(engine):
    """Test weird/unexpected string values in categorical columns."""
    df_weird = pd.DataFrame({
        "Line Item Value": [10000.0, 20000.0, 30000.0, 40000.0],
        "Shipment Mode": ["UNKNOWN_MODE", "   Air  ", "", None],
        "First Line Designation": ["MAYBE", "yes", "0", None],
        "Sub Classification": ["unknown", "Pediatric-special", "adult", None],
        "Product Group": ["other", "ARV", "arv", None],
        "Fulfill Via": ["third-party", "From RDC", "Direct Drop", None],
    })
    res = engine.compute_costs(df_weird, is_log_transformed=False)
    assert len(res) == 4
    assert not res.isna().any().any()
    assert (res["fn_cost"] > 0).all()
    assert (res["fp_cost"] > 0).all()


# =============================================================================
# 4. MATHEMATICAL INVARIANTS & MONOTONICITY
# =============================================================================

def test_adversarial_cost_monotonicity_with_value(engine):
    """Increasing shipment value V must monotonically increase FN_Cost, FP_Cost, and Intervention_Cost."""
    values = [100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
    df_mono = pd.DataFrame({"Line Item Value": values})
    res = engine.compute_costs(df_mono, is_log_transformed=False)

    fn_costs = res["fn_cost"].to_numpy()
    fp_costs = res["fp_cost"].to_numpy()
    interv_costs = res["intervention_cost"].to_numpy()

    for i in range(len(values) - 1):
        assert fn_costs[i] < fn_costs[i+1], f"FN_Cost not monotonic: {fn_costs[i]} >= {fn_costs[i+1]}"
        assert fp_costs[i] < fp_costs[i+1], f"FP_Cost not monotonic: {fp_costs[i]} >= {fp_costs[i+1]}"
        assert interv_costs[i] < interv_costs[i+1], f"Intervention_Cost not monotonic: {interv_costs[i]} >= {interv_costs[i+1]}"


def test_adversarial_criticality_boost_invariants(engine):
    """Clinical criticality factors must strictly increase FN_Cost."""
    base_record = {"Line Item Value": 50000.0, "Shipment Mode": "Air"}
    fl_record = {**base_record, "First Line Designation": "Yes"}
    ped_record = {**base_record, "Sub Classification": "Pediatric"}
    arv_record = {**base_record, "Product Group": "ARV"}
    all_record = {**base_record, "First Line Designation": "Yes", "Sub Classification": "Pediatric", "Product Group": "ARV"}

    c_base = engine.compute_costs(base_record, is_log_transformed=False, return_dataframe=False)
    c_fl = engine.compute_costs(fl_record, is_log_transformed=False, return_dataframe=False)
    c_ped = engine.compute_costs(ped_record, is_log_transformed=False, return_dataframe=False)
    c_arv = engine.compute_costs(arv_record, is_log_transformed=False, return_dataframe=False)
    c_all = engine.compute_costs(all_record, is_log_transformed=False, return_dataframe=False)

    assert c_fl["fn_cost"] > c_base["fn_cost"]
    assert c_ped["fn_cost"] > c_base["fn_cost"]
    assert c_arv["fn_cost"] > c_base["fn_cost"]
    assert c_all["fn_cost"] > max(c_fl["fn_cost"], c_ped["fn_cost"], c_arv["fn_cost"])


def test_adversarial_scenario_hierarchy_invariants(engine):
    """For any given shipment, costs must strictly follow: Low < Base < High."""
    record = {
        "Line Item Value": 75000.0,
        "Shipment Mode": "Truck",
        "First Line Designation": "Yes",
        "Sub Classification": "Pediatric",
        "Product Group": "ARV",
    }
    c_low = engine.compute_costs(record, scenario_name="low", is_log_transformed=False, return_dataframe=False)
    c_base = engine.compute_costs(record, scenario_name="base", is_log_transformed=False, return_dataframe=False)
    c_high = engine.compute_costs(record, scenario_name="high", is_log_transformed=False, return_dataframe=False)

    assert c_low["fn_cost"] < c_base["fn_cost"] < c_high["fn_cost"]
    assert c_low["fp_cost"] < c_base["fp_cost"] < c_high["fp_cost"]
    assert c_low["intervention_cost"] < c_base["intervention_cost"] < c_high["intervention_cost"]


def test_adversarial_tau_star_bounds_and_validity(engine):
    """Bayes-optimal decision thresholds must strictly lie in (0, 1]."""
    values = np.logspace(1, 7, 50)
    df_eval = pd.DataFrame({
        "Line Item Value": values,
        "Shipment Mode": np.random.choice(["Air", "Truck", "Ocean"], size=50),
    })

    for sc in ["low", "base", "high"]:
        res = engine.compute_costs(df_eval, scenario_name=sc, is_log_transformed=False)
        assert (res["tau_star"] > 0.0).all()
        assert (res["tau_star"] <= 1.0).all()
        assert (res["tau_star_simple"] > 0.0).all()
        assert (res["tau_star_simple"] <= 1.0).all()


# =============================================================================
# 5. EXCEPTION HIERARCHY & BOUNDARY CORNER CASES
# =============================================================================

def test_leakage_violation_is_subclass_of_value_error():
    """Verify that LeakageViolationError strictly inherits from ValueError."""
    assert issubclass(LeakageViolationError, ValueError)
    err = LeakageViolationError("Test leakage error")
    assert isinstance(err, ValueError)


def test_adversarial_negative_net_benefit_clamps_tau_star_to_one(engine):
    """
    For extremely low-value shipments where intervention cost exceeds potential delay penalty,
    Net_Benefit <= 0, and tau* must clamp to 1.0 (indicating intervention is economically unjustified).
    """
    # Tiny value shipment ($1.00)
    tiny_item = {
        "Line Item Value": 1.0,
        "Shipment Mode": "Air",
        "Fulfill Via": "Direct Drop",
    }
    # Expediting cost is $500, whereas total delay penalty is ~$2300, wait:
    # Let's check with very small delay_days custom or very high expedite cost
    custom_sc = CostScenario(
        name="HighExpedite",
        c_daily_base=10.0,
        rho_value=0.0001,
        c_fixed_stockout=50.0,
        c_triage_base=20.0,
        beta_audit=5.0,
        c_direct_inquiry=10.0,
        c_rdc_inquiry=5.0,
        c_expedite_base=5000.0,  # Huge expediting fee
        gamma_expedite=0.01,
        delay_days_assumed=5.0,
        days_saved_efficacy=2.0,
    )
    custom_engine = CostScenarioModel(custom_scenario=custom_sc)
    res = custom_engine.compute_costs(tiny_item, is_log_transformed=False, return_dataframe=False)

    assert res["net_benefit"] < 0, f"Expected negative net benefit, got {res['net_benefit']}"
    assert res["tau_star"] == 1.0, f"Expected tau_star to be clamped to 1.0, got {res['tau_star']}"


def test_adversarial_sample_weights_mismatched_length(engine):
    """Mismatched df and y_true must raise ValueError."""
    df = pd.DataFrame({"Line Item Value": [1000.0, 2000.0]})
    y_bad = np.array([1, 0, 1])  # 3 elements vs 2 rows
    with pytest.raises(ValueError, match="Length mismatch"):
        engine.compute_sample_weights(df, y_bad)


def test_adversarial_compute_expected_cost_unsupported_type():
    """Passing invalid cost structure to compute_expected_cost must raise TypeError."""
    y = np.array([1, 0])
    d = np.array([1, 0])
    with pytest.raises(TypeError, match="Unsupported costs type"):
        CostScenarioModel.compute_expected_cost(y, d, "invalid_costs_object")


def test_adversarial_large_batch_stress(engine):
    """Stress test with N=50,000 synthetic shipments for numerical stability and zero NaNs."""
    np.random.seed(999)
    n = 50000
    df_large = pd.DataFrame({
        "Line Item Value": np.exp(np.random.uniform(0, 16, size=n)) - 1.0,
        "Shipment Mode": np.random.choice(["Air", "Air Charter", "Truck", "Ocean", "Unknown"], size=n),
        "Fulfill Via": np.random.choice(["From RDC", "Direct Drop", "Other"], size=n),
        "First Line Designation": np.random.choice(["Yes", "No", "1", "0"], size=n),
        "Sub Classification": np.random.choice(["Pediatric", "Adult", "HIV", "Other"], size=n),
        "Product Group": np.random.choice(["ARV", "HRDT", "ACT", "ANTM", "OTHER"], size=n),
    })

    res = engine.compute_costs(df_large, is_log_transformed=False)
    assert len(res) == n
    assert not res.isna().any().any()
    assert not np.isinf(res.to_numpy()).any()
    assert (res["fn_cost"] > 0).all()
    assert (res["fp_cost"] > 0).all()
    assert (res["intervention_cost"] > 0).all()
    assert (res["tau_star"] >= 0.0).all()
    assert (res["tau_star"] <= 1.0).all()
