"""
Exhaustive Unit Tests for Phase 2 — Experiment E8 Cost Scenario Engine.

Tests:
1. Scenario loading, validation, and switching (Low, Base, High).
2. Strict leakage prevention (raising LeakageViolationError on forbidden columns).
3. Exact mathematical correctness against analytic formulas.
4. Log vs un-logged monetary value handling and auto-detection.
5. Batch vectorization vs single-instance consistency.
6. Positive cost asymmetry (FN / FP > 1) across all real SCMS modeling rows.
7. Sample weight generation for E8-B cost-sensitive learning.
8. Realized cost and net savings computation under decision policies.
9. Expected net benefit ranking for review budget allocation.
"""

import math
import time
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


@pytest.fixture
def modeling_df():
    path = Path("artifacts/data/scms_modeling_features.parquet")
    if not path.exists():
        pytest.skip(f"Modeling features file not found at {path}")
    return pd.read_parquet(path)


@pytest.fixture
def clean_features(modeling_df):
    """Features with all forbidden/post-outcome columns strictly removed."""
    forbidden_set = set(FORBIDDEN_COLUMNS)
    cols = [c for c in modeling_df.columns if c not in forbidden_set]
    return modeling_df[cols].copy()


# =============================================================================
# 1. Configuration Loading & Scenario Management Tests
# =============================================================================

def test_config_loading_and_available_scenarios(engine):
    """Test that all three required scenarios (low, base, high) are loaded and valid."""
    scenarios = engine.list_scenarios()
    assert "low" in scenarios
    assert "base" in scenarios
    assert "high" in scenarios
    assert len(scenarios) >= 3

    for sc_name in ["low", "base", "high"]:
        sc = engine.get_scenario(sc_name)
        assert isinstance(sc, CostScenario)
        assert sc.c_daily_base > 0
        assert sc.c_fixed_stockout >= 0
        assert sc.c_triage_base > 0
        assert sc.c_expedite_base > 0
        assert sc.delay_days_assumed > 0
        assert sc.days_saved_efficacy > 0
        assert sc.days_saved_efficacy <= sc.delay_days_assumed


def test_scenario_switching(engine):
    """Test active scenario switching and retrieval."""
    assert engine.active_scenario_name == "base"
    engine.set_scenario("low")
    assert engine.active_scenario_name == "low"
    assert engine.get_scenario().name == "Low Cost Scenario"

    engine.set_scenario("high")
    assert engine.active_scenario_name == "high"
    assert engine.get_scenario().name == "High Cost Scenario"


def test_invalid_scenario_raises_key_error(engine):
    """Test that requesting an unknown scenario raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        engine.get_scenario("non_existent_scenario")

    with pytest.raises(KeyError, match="not found"):
        engine.set_scenario("non_existent_scenario")


def test_cost_scenario_pydantic_validation():
    """Test Pydantic validator checks on impossible parameters."""
    with pytest.raises(ValueError):
        # days_saved_efficacy cannot exceed delay_days_assumed
        CostScenario(
            name="Invalid",
            c_daily_base=100.0,
            rho_value=0.001,
            c_fixed_stockout=500.0,
            c_triage_base=50.0,
            beta_audit=10.0,
            c_direct_inquiry=30.0,
            c_rdc_inquiry=10.0,
            c_expedite_base=500.0,
            gamma_expedite=0.005,
            delay_days_assumed=10.0,
            days_saved_efficacy=15.0,  # Invalid: > 10.0
        )


def test_cost_engine_alias_equivalence(config_path):
    """Test that CostEngine alias is identical to CostScenarioModel."""
    assert CostEngine is CostScenarioModel
    engine_alias = CostEngine(config_path=config_path)
    assert isinstance(engine_alias, CostScenarioModel)


# =============================================================================
# 2. Strict Leakage Prevention & Security Guard Tests
# =============================================================================

@pytest.mark.parametrize("forbidden_col", FORBIDDEN_COLUMNS)
def test_leakage_guard_rejects_forbidden_columns_in_dataframe(engine, forbidden_col):
    """Test that presence of any forbidden column in a DataFrame raises LeakageViolationError."""
    df_with_leak = pd.DataFrame({
        "Line Item Value": [10000.0, 20000.0],
        "Shipment Mode": ["Air", "Truck"],
        forbidden_col: [1, 0],
    })
    with pytest.raises(LeakageViolationError, match="Forbidden / target-leakage column"):
        engine.compute_costs(df_with_leak, strict_leakage_check=True)


def test_leakage_guard_rejects_forbidden_keys_in_dict(engine):
    """Test that dictionary input with forbidden keys raises LeakageViolationError."""
    sample_dict = {
        "Line Item Value": 50000.0,
        "Shipment Mode": "Ocean",
        "Delay_Flag": 1,
    }
    with pytest.raises(LeakageViolationError, match="Forbidden / target-leakage column"):
        engine.compute_costs(sample_dict, strict_leakage_check=True)


def test_leakage_guard_can_be_disabled_explicitly(engine):
    """Test that strict_leakage_check=False allows computation if explicitly requested."""
    df_with_leak = pd.DataFrame({
        "Line Item Value": [10000.0],
        "Shipment Mode": ["Air"],
        "Delay_Flag": [1],
    })
    # Should not raise when strict_leakage_check is False
    res = engine.compute_costs(df_with_leak, strict_leakage_check=False)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 1


# =============================================================================
# 3. Exact Mathematical Correctness Tests
# =============================================================================

def test_exact_mathematical_calculation_base_scenario(engine):
    """
    Verify exact analytical calculation of all cost components on a known reference case.
    Reference Case:
      V = $100,000
      First Line = 'Yes' (delta = 0.30)
      Pediatric = 'Pediatric' (delta = 0.20)
      ARV = 'ARV' (delta = 0.15)
      kappa = 1.0 + 0.30 + 0.20 + 0.15 = 1.65
      Shipment Mode = 'Ocean' (lambda_mode = 1.25)
      Fulfill Via = 'Direct Drop' (c_inquiry = 30.0)
    """
    sample = {
        "Line Item Value": 100000.0,
        "First Line Designation": "Yes",
        "Sub Classification": "Pediatric",
        "Product Group": "ARV",
        "Shipment Mode": "Ocean",
        "Fulfill Via": "Direct Drop",
    }

    res = engine.compute_costs(sample, scenario_name="base", is_log_transformed=False, return_dataframe=False)

    # Expected values
    # Daily penalty = 150.0 + 0.0010 * 100000 = 250.0
    # FN = 1.65 * 1.25 * [500.0 + 250.0 * 12.0] = 2.0625 * 3500.0 = 7218.75
    expected_fn = 7218.75
    assert math.isclose(res["fn_cost"], expected_fn, rel_tol=1e-5)

    # FP = 50.0 + 10.0 * ln(100001) + 30.0 = 80.0 + 10.0 * 11.51293546 = 195.12935
    expected_fp = 50.0 + 10.0 * math.log1p(100000.0) + 30.0
    assert math.isclose(res["fp_cost"], expected_fp, rel_tol=1e-5)

    # Intervention = 500.0 + 0.0050 * 100000 = 1000.0
    expected_interv = 1000.0
    assert math.isclose(res["intervention_cost"], expected_interv, rel_tol=1e-5)

    # Residual Delay = 1.65 * 1.25 * 250.0 * (12.0 - 5.0) = 2.0625 * 1750.0 = 3609.375
    expected_resid = 3609.375
    assert math.isclose(res["residual_delay_cost"], expected_resid, rel_tol=1e-5)

    # Net Benefit = 7218.75 - (1000.0 + 3609.375) = 2609.375
    expected_net_ben = 2609.375
    assert math.isclose(res["net_benefit"], expected_net_ben, rel_tol=1e-5)

    # tau* = FP / (Net_Benefit + FP)
    expected_tau_star = expected_fp / (expected_net_ben + expected_fp)
    assert math.isclose(res["tau_star"], expected_tau_star, rel_tol=1e-5)

    # tau*_simple = FP / (FN + FP)
    expected_tau_simple = expected_fp / (expected_fn + expected_fp)
    assert math.isclose(res["tau_star_simple"], expected_tau_simple, rel_tol=1e-5)

    # Asymmetry ratio = FN / FP
    expected_asym = expected_fn / expected_fp
    assert math.isclose(res["asymmetry_ratio"], expected_asym, rel_tol=1e-5)


def test_exact_mathematical_calculation_low_and_high_scenarios(engine):
    """Verify calculations for Low and High scenarios."""
    sample = {
        "Line Item Value": 50000.0,
        "First Line Designation": "No",
        "Sub Classification": "Adult",
        "Product Group": "HRDT",
        "Shipment Mode": "Air",
        "Fulfill Via": "From RDC",
    }

    # Low Scenario
    res_low = engine.compute_costs(sample, scenario_name="low", is_log_transformed=False, return_dataframe=False)
    # kappa = 1.0, lambda_mode = 1.0, c_inquiry = 5.0
    # Daily penalty = 50.0 + 0.0005 * 50000 = 75.0
    # FN = 1.0 * 1.0 * [200.0 + 75.0 * 10.0] = 950.0
    assert math.isclose(res_low["fn_cost"], 950.0, rel_tol=1e-5)
    # FP = 25.0 + 5.0 * ln(50001) + 5.0 = 30.0 + 5.0 * 10.819798 = 84.09899
    expected_fp_low = 30.0 + 5.0 * math.log1p(50000.0)
    assert math.isclose(res_low["fp_cost"], expected_fp_low, rel_tol=1e-5)

    # High Scenario
    res_high = engine.compute_costs(sample, scenario_name="high", is_log_transformed=False, return_dataframe=False)
    # kappa = 1.0, lambda_mode = 1.0, c_inquiry = 20.0
    # Daily penalty = 350.0 + 0.0020 * 50000 = 450.0
    # FN = 1.0 * 1.0 * [1500.0 + 450.0 * 15.0] = 8250.0
    assert math.isclose(res_high["fn_cost"], 8250.0, rel_tol=1e-5)
    # FP = 100.0 + 20.0 * ln(50001) + 20.0 = 120.0 + 20.0 * 10.819798 = 336.39596
    expected_fp_high = 120.0 + 20.0 * math.log1p(50000.0)
    assert math.isclose(res_high["fp_cost"], expected_fp_high, rel_tol=1e-5)


# =============================================================================
# 4. Log vs Un-logged Value Handling Tests
# =============================================================================

def test_log_and_unlogged_value_equivalence(engine):
    """Test that log-transformed value ln(1 + V) matches raw V after transformation."""
    raw_dollars = 75000.0
    log_val = math.log1p(raw_dollars)

    df_raw = pd.DataFrame({
        "Line Item Value": [raw_dollars],
        "Shipment Mode": ["Truck"],
        "Fulfill Via": ["Direct Drop"],
    })
    df_log = pd.DataFrame({
        "Line Item Value": [log_val],
        "Shipment Mode": ["Truck"],
        "Fulfill Via": ["Direct Drop"],
    })

    res_raw = engine.compute_costs(df_raw, is_log_transformed=False)
    res_log = engine.compute_costs(df_log, is_log_transformed=True)
    res_auto = engine.compute_costs(df_log, is_log_transformed=None)

    pd.testing.assert_frame_equal(res_raw, res_log)
    pd.testing.assert_frame_equal(res_raw, res_auto)


def test_zero_and_negative_value_handling(engine):
    """Test that zero and negative values are cleanly handled without NaN/Inf."""
    df_zero = pd.DataFrame({
        "Line Item Value": [0.0, -100.0, np.nan],
        "Shipment Mode": ["Air", "Air", "Air"],
    })
    res = engine.compute_costs(df_zero, is_log_transformed=False)
    assert not res.isna().any().any()
    assert (res["fn_cost"] > 0).all()
    assert (res["fp_cost"] > 0).all()
    assert (res["asymmetry_ratio"] > 1.0).all()


# =============================================================================
# 5. Vectorization vs Single-Instance Consistency Tests
# =============================================================================

def test_vectorization_vs_single_record_consistency(engine):
    """Test that batch DataFrame calculation matches record-by-record single instance calculation."""
    np.random.seed(42)
    n = 100
    df_batch = pd.DataFrame({
        "Line Item Value": np.random.uniform(100.0, 500000.0, size=n),
        "Shipment Mode": np.random.choice(["Air", "Air Charter", "Truck", "Ocean"], size=n),
        "Fulfill Via": np.random.choice(["From RDC", "Direct Drop"], size=n),
        "First Line Designation": np.random.choice(["Yes", "No"], size=n),
        "Sub Classification": np.random.choice(["Pediatric", "Adult"], size=n),
        "Product Group": np.random.choice(["ARV", "HRDT", "ACT"], size=n),
    })

    batch_res = engine.compute_costs(df_batch, is_log_transformed=False)

    for i in range(n):
        single_row = df_batch.iloc[i].to_dict()
        single_res = engine.compute_costs(single_row, is_log_transformed=False, return_dataframe=False)

        assert math.isclose(batch_res.loc[i, "fn_cost"], single_res["fn_cost"], rel_tol=1e-6)
        assert math.isclose(batch_res.loc[i, "fp_cost"], single_res["fp_cost"], rel_tol=1e-6)
        assert math.isclose(batch_res.loc[i, "intervention_cost"], single_res["intervention_cost"], rel_tol=1e-6)
        assert math.isclose(batch_res.loc[i, "residual_delay_cost"], single_res["residual_delay_cost"], rel_tol=1e-6)
        assert math.isclose(batch_res.loc[i, "net_benefit"], single_res["net_benefit"], rel_tol=1e-6)
        assert math.isclose(batch_res.loc[i, "tau_star"], single_res["tau_star"], rel_tol=1e-6)
        assert math.isclose(batch_res.loc[i, "asymmetry_ratio"], single_res["asymmetry_ratio"], rel_tol=1e-6)


# =============================================================================
# 6. Real Dataset Asymmetry & Scenario Distribution Tests
# =============================================================================

@pytest.mark.parametrize("sc_name", ["low", "base", "high"])
def test_positive_asymmetry_across_real_scms_modeling_data(engine, clean_features, sc_name):
    """
    Verify that FN_Cost > FP_Cost (asymmetry ratio > 1.0) holds for 100% of rows
    in the real SCMS modeling cohort across all 3 scenarios.
    """
    assert len(clean_features) == 8319
    # In scms_modeling_features.parquet, Line Item Value is log1p transformed
    costs_df = engine.compute_costs(clean_features, scenario_name=sc_name, is_log_transformed=True)

    assert len(costs_df) == 8319
    assert (costs_df["asymmetry_ratio"] > 1.0).all(), "FN/FP ratio must exceed 1.0 for all records"
    assert (costs_df["fn_cost"] > 0).all()
    assert (costs_df["fp_cost"] > 0).all()
    assert (costs_df["intervention_cost"] > 0).all()
    assert (costs_df["residual_delay_cost"] >= 0).all()
    assert (costs_df["tau_star"] > 0).all()
    assert (costs_df["tau_star"] <= 1.0).all()

    # Verify median asymmetry matches expected domain bounds
    med_ratio = costs_df["asymmetry_ratio"].median()
    if sc_name == "low":
        assert 14.0 <= med_ratio <= 20.0
    elif sc_name == "base":
        assert 22.0 <= med_ratio <= 30.0
    elif sc_name == "high":
        assert 32.0 <= med_ratio <= 45.0


# =============================================================================
# 7. Sample Weights Computation (E8-B) Tests
# =============================================================================

def test_sample_weights_computation(engine, clean_features):
    """Test sample weight generation for cost-weighted training."""
    n = 200
    sub_df = clean_features.iloc[:n].copy()
    y_dummy = np.random.choice([0, 1], size=n, p=[0.85, 0.15])

    weights = engine.compute_sample_weights(sub_df, y_dummy, scenario_name="base", normalize=True, is_log_transformed=True)

    assert len(weights) == n
    assert (weights > 0).all()
    assert math.isclose(np.mean(weights), 1.0, rel_tol=1e-5)

    # Check that delayed instances (y=1) have systematically higher weights than on-time (y=0)
    assert np.mean(weights[y_dummy == 1]) > np.mean(weights[y_dummy == 0])


# =============================================================================
# 8. Expected Cost & Savings Computation Tests
# =============================================================================

def test_expected_cost_and_net_savings(engine):
    """Test expected cost and net savings calculations under various decision rules."""
    n = 100
    y_true = np.array([1]*20 + [0]*80)
    costs_df = pd.DataFrame({
        "fn_cost": [3000.0]*100,
        "fp_cost": [150.0]*100,
        "intervention_cost": [500.0]*100,
        "residual_delay_cost": [1000.0]*100,
    })

    # Policy 0: Do Nothing (d = 0)
    d_none = np.zeros(n, dtype=int)
    cost_none = engine.compute_expected_cost(y_true, d_none, costs_df)
    # 20 delayed * 3000 = 60,000
    assert cost_none == 60000.0
    savings_none = engine.compute_expected_net_savings(y_true, d_none, costs_df)
    assert savings_none == 0.0

    # Policy 1: Perfect Oracle (d = y)
    d_oracle = y_true.copy()
    cost_oracle = engine.compute_expected_cost(y_true, d_oracle, costs_df)
    # 20 delayed * (500 + 1000) + 80 * 0 = 30,000
    assert cost_oracle == 30000.0
    savings_oracle = engine.compute_expected_net_savings(y_true, d_oracle, costs_df)
    assert savings_oracle == 30000.0

    # Policy 2: Intervene on Everything (d = 1)
    d_all = np.ones(n, dtype=int)
    cost_all = engine.compute_expected_cost(y_true, d_all, costs_df)
    # 20 * 1500 + 80 * 150 = 30000 + 12000 = 42,000
    assert cost_all == 42000.0
    assert engine.compute_expected_net_savings(y_true, d_all, costs_df) == 18000.0


# =============================================================================
# 9. Expected Net Benefit Ranking Tests
# =============================================================================

def test_expected_net_benefit_ranking(engine):
    """Test ranking calculation for operational review budgets."""
    df_sample = pd.DataFrame({
        "Line Item Value": [100000.0, 100000.0, 1000.0],
        "Shipment Mode": ["Air", "Air", "Air"],
        "Fulfill Via": ["Direct Drop", "Direct Drop", "Direct Drop"],
    })
    p_hat = np.array([0.80, 0.20, 0.05])

    expected_benefits = engine.compute_expected_net_benefit_ranking(
        df_sample, p_hat, scenario_name="base", is_log_transformed=False
    )
    assert len(expected_benefits) == 3
    # High-risk shipment (80% risk, $100k) yields highest net benefit (> 0)
    assert expected_benefits[0] > 0
    assert expected_benefits[0] > expected_benefits[1]
    # Low-risk shipment (20% risk, $100k) yields higher net benefit than negligible-risk small shipment (5%, $1k)
    assert expected_benefits[1] > expected_benefits[2]


# =============================================================================
# 10. Vectorization Performance Benchmark
# =============================================================================

def test_vectorization_performance_benchmark(engine, clean_features):
    """Benchmark cost calculation runtime across all 8,319 records."""
    t0 = time.perf_counter()
    res = engine.compute_costs(clean_features, scenario_name="base", is_log_transformed=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(res) == 8319
    # Expect vectorized computation to take < 100 ms on typical CPU
    assert elapsed_ms < 250.0, f"Vectorized calculation took {elapsed_ms:.2f} ms (expected < 250 ms)"
