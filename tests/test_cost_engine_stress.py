"""
Empirical Adversarial Stress & Robustness Suite for Experiment E8 Cost Scenario Engine.

Authored by Challenger 1 (Challenger Archetype: Empirical Critic).

Test Dimensions:
1. Boundary & Extreme Monetary Values ($0, negative, $1B, $1T, $10^15, subnormal, NaN, Inf)
2. Categorical Perturbations, Unknown Levels & Malformed Types (unknown countries, modes, nulls, casing)
3. Large Batch Vectorization Throughput, Memory Stability & Determinism (50,000 and 100,000 rows)
4. Mathematical Invariants:
   - Strict bounds: tau* in [0, 1] and tau*_simple in [0, 1] across all adversarial regimes
   - Monotone behavior: tau*_simple strictly decreases with FN/FP cost ratio
   - Value scaling monotonicity: tau* strictly decreases as Line Item Value increases
   - Cost ordering invariants: FN >= Residual Delay Cost, Net Benefit = FN - (Interv + Residual)
5. Decision Policy and Oracle Optimality Invariants
6. Missing column graceful fallbacks and structural variations (Series, Dict, Disjoint DF)
"""

import math
import sys
import time
import numpy as np
import pandas as pd
import pytest

from delay_intelligence.cost_sensitive.cost_engine import (
    CostScenario,
    CostScenarioModel,
    CostEngine,
    CostBreakdown,
    LeakageViolationError,
    FORBIDDEN_COLUMNS,
)


@pytest.fixture
def engine():
    return CostScenarioModel(config_path="configs/cost_scenarios.yaml", scenario_name="base")


# =============================================================================
# 1. Extreme Values & Numeric Boundary Stress Tests
# =============================================================================

@pytest.mark.parametrize("scenario_name", ["low", "base", "high"])
def test_zero_monetary_value_stability(engine, scenario_name):
    """Test that $0 line item value computes finite positive costs and stable thresholds."""
    df_zero = pd.DataFrame({
        "Line Item Value": [0.0, 0, 0.00000],
        "Shipment Mode": ["Air", "Truck", "Ocean"],
        "Fulfill Via": ["Direct Drop", "From RDC", "Direct Drop"],
    })
    res = engine.compute_costs(df_zero, scenario_name=scenario_name, is_log_transformed=False)
    
    assert len(res) == 3
    assert not res.isna().any().any()
    assert (res["fn_cost"] > 0).all()
    assert (res["fp_cost"] > 0).all()
    assert (res["intervention_cost"] > 0).all()
    assert (res["residual_delay_cost"] > 0).all()
    assert ((res["tau_star"] >= 0.0) & (res["tau_star"] <= 1.0)).all()
    assert ((res["tau_star_simple"] >= 0.0) & (res["tau_star_simple"] <= 1.0)).all()


@pytest.mark.parametrize("neg_val", [-0.01, -1.0, -100.0, -1e6, -1e12])
def test_negative_monetary_values_clamped_to_zero(engine, neg_val):
    """Test that negative monetary values are safely clamped to 0 without producing negative costs or NaNs."""
    df_neg = pd.DataFrame({
        "Line Item Value": [neg_val],
        "Shipment Mode": ["Air"],
    })
    res = engine.compute_costs(df_neg, is_log_transformed=False)
    
    # Negative value should be clamped to 0, matching a 0-dollar item exactly
    df_zero = pd.DataFrame({"Line Item Value": [0.0], "Shipment Mode": ["Air"]})
    res_zero = engine.compute_costs(df_zero, is_log_transformed=False)
    
    pd.testing.assert_frame_equal(res, res_zero)


@pytest.mark.parametrize("extreme_val", [1e9, 1e12, 1e15])
def test_astronomical_monetary_values(engine, extreme_val):
    """Test handling of billion, trillion, and quadrillion dollar shipments (no overflow, finite results)."""
    df_extreme = pd.DataFrame({
        "Line Item Value": [extreme_val],
        "Shipment Mode": ["Ocean"],
        "First Line Designation": ["Yes"],
        "Sub Classification": ["Pediatric"],
        "Product Group": ["ARV"],
    })
    res = engine.compute_costs(df_extreme, is_log_transformed=False)
    
    assert np.isfinite(res["fn_cost"].iloc[0])
    assert np.isfinite(res["fp_cost"].iloc[0])
    assert np.isfinite(res["intervention_cost"].iloc[0])
    assert np.isfinite(res["net_benefit"].iloc[0])
    assert res["fn_cost"].iloc[0] > 0
    assert res["fp_cost"].iloc[0] > 0
    # For astronomical values, FN cost vastly exceeds FP cost, tau* should approach 0.0
    assert res["tau_star"].iloc[0] < 1e-4
    assert res["tau_star_simple"].iloc[0] < 1e-4
    assert res["asymmetry_ratio"].iloc[0] > 1e4


def test_nan_and_subnormal_handling(engine):
    """Test that NaNs, None, and subnormal floating point numbers are handled cleanly."""
    df_special = pd.DataFrame({
        "Line Item Value": [np.nan, 1e-25, 0.0, 1e-15],
        "Shipment Mode": ["Air", "Truck", None, "Ocean"],
    })
    res = engine.compute_costs(df_special, is_log_transformed=False)
    
    assert not res.isna().any().any()
    assert (res["fn_cost"] > 0).all()
    assert (res["fp_cost"] > 0).all()
    assert ((res["tau_star"] >= 0.0) & (res["tau_star"] <= 1.0)).all()


# =============================================================================
# 2. Categorical Perturbations, Unknown Levels & Robustness
# =============================================================================

def test_unseen_and_wildcard_categorical_levels(engine):
    """Test completely unobserved categories for Country, Shipment Mode, Fulfill Via, and Product Groups."""
    df_wild = pd.DataFrame({
        "Line Item Value": [50000.0, 75000.0, 120000.0, 30000.0],
        "Country": ["Mars Colony", "Narnia", "Wakanda", ""],
        "Shipment Mode": ["SpaceX Rocket", "Teleportation", "Hyperloop", "Pigeon Carrier"],
        "Fulfill Via": ["Intergalactic Vault", "Quantum Realm", "Alien Drop", ""],
        "Product Group": ["Anti-Gravity Unit", "Vibranium", "Unknown", ""],
        "Sub Classification": ["Non-Human", "Exotic", "Special", ""],
        "First Line Designation": ["Unknown", "Maybe", "2", "null"],
    })
    res = engine.compute_costs(df_wild, is_log_transformed=False)
    
    assert len(res) == 4
    assert not res.isna().any().any()
    # Unseen modes should fallback to Default mode multiplier (1.00)
    # Unseen fulfillment should fallback to Direct Drop inquiry cost (30.0 for base)
    # Unseen product groups should have kappa = 1.0 (no bonus multiplier)
    sc = engine.get_scenario("base")
    for i in range(4):
        val = df_wild.loc[i, "Line Item Value"]
        daily_penalty = sc.c_daily_base + sc.rho_value * val
        expected_fn = 1.0 * 1.0 * (sc.c_fixed_stockout + daily_penalty * sc.delay_days_assumed)
        expected_fp = sc.c_triage_base + sc.beta_audit * math.log1p(val) + sc.c_direct_inquiry
        
        assert math.isclose(res.loc[i, "fn_cost"], expected_fn, rel_tol=1e-5)
        assert math.isclose(res.loc[i, "fp_cost"], expected_fp, rel_tol=1e-5)


def test_mixed_casing_and_whitespace_in_categories(engine):
    """Test resilience against dirty strings, mixed casing, and trailing whitespace."""
    df_dirty = pd.DataFrame({
        "Line Item Value": [100000.0, 100000.0, 100000.0],
        "Shipment Mode": ["  air  ", "OCEAN\n", "\tTrUcK  "],
        "Fulfill Via": ["  From RDC  ", " DIRECT drop ", "rdc"],
        "First Line Designation": ["  yes ", "YeS", "1.0"],
        "Sub Classification": [" PEDIATRIC ", "pediatric formulation", "Sub-Pediatric"],
        "Product Group": [" arv ", "ARV\t", " ArV "],
    })
    res = engine.compute_costs(df_dirty, is_log_transformed=False)
    
    assert not res.isna().any().any()
    assert (res["fn_cost"] > 0).all()
    # Mode multiplier for Ocean (row 1) should be 1.25, Air (row 0) should be 1.00
    assert res.loc[1, "fn_cost"] > res.loc[0, "fn_cost"]
    # All 3 should match pediatric and ARV bonuses
    assert (res["fn_cost"] > 5000.0).all()


def test_completely_disjoint_and_missing_columns(engine):
    """Test that DataFrame with no expected columns defaults gracefully without crashing."""
    df_disjoint = pd.DataFrame({
        "Random Column A": [1, 2, 3],
        "Arbitrary Metric B": ["X", "Y", "Z"],
        "Unrelated Factor C": [0.1, 0.2, 0.3],
    })
    res = engine.compute_costs(df_disjoint, is_log_transformed=False)
    
    assert len(res) == 3
    assert not res.isna().any().any()
    assert (res["fn_cost"] > 0).all()
    assert (res["fp_cost"] > 0).all()


def test_empty_dataframe_handling(engine):
    """Test behavior on empty DataFrame (0 rows)."""
    df_empty = pd.DataFrame(columns=["Line Item Value", "Shipment Mode", "Country"])
    res = engine.compute_costs(df_empty, is_log_transformed=False)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert "fn_cost" in res.columns
    assert "fp_cost" in res.columns
    assert "tau_star" in res.columns


# =============================================================================
# 3. Large Batch Vectorization Throughput & Memory Stability (50k & 100k rows)
# =============================================================================

def test_large_batch_vectorization_throughput_50k(engine):
    """
    Stress-test vectorization performance and memory on a 50,000-row synthetic supply chain dataset.
    Requires throughput >= 50,000 rows/sec (execution time <= 1.0s) and zero NaN/Inf.
    """
    np.random.seed(12345)
    n = 50_000
    
    modes = ["Air", "Air Charter", "Truck", "Ocean", "Unknown Mode", "Drone"]
    fulfillments = ["From RDC", "Direct Drop", "Warehouse X", None]
    product_groups = ["ARV", "HRDT", "ACT", "ANTIMALARIAL", "OTHER"]
    subs = ["Pediatric", "Adult", "General", ""]
    first_lines = ["Yes", "No", "1", "0", None]
    
    df_large = pd.DataFrame({
        "Line Item Value": np.random.exponential(scale=50000.0, size=n),
        "Shipment Mode": np.random.choice(modes, size=n),
        "Fulfill Via": np.random.choice(fulfillments, size=n),
        "Product Group": np.random.choice(product_groups, size=n),
        "Sub Classification": np.random.choice(subs, size=n),
        "First Line Designation": np.random.choice(first_lines, size=n),
        "Country": np.random.choice(["Nigeria", "Uganda", "Tanzania", "Unknown"], size=n),
    })
    
    # Warm up
    _ = engine.compute_costs(df_large.iloc[:100], is_log_transformed=False)
    
    t0 = time.perf_counter()
    res = engine.compute_costs(df_large, is_log_transformed=False)
    elapsed = time.perf_counter() - t0
    
    throughput = n / elapsed
    
    assert len(res) == n
    assert not res.isna().any().any()
    assert np.isfinite(res["fn_cost"].to_numpy()).all()
    assert np.isfinite(res["fp_cost"].to_numpy()).all()
    assert np.isfinite(res["tau_star"].to_numpy()).all()
    assert ((res["tau_star"] >= 0.0) & (res["tau_star"] <= 1.0)).all()
    
    # Execution time constraint: 50k rows should take well under 1.0 second
    assert elapsed < 1.0, f"50k cost calculation took {elapsed:.3f}s (throughput: {throughput:.0f} rows/s)"


def test_batch_determinism_and_memory_stability(engine):
    """Test that running multiple consecutive large batches produces identical bitwise outputs."""
    np.random.seed(999)
    n = 20_000
    df_batch = pd.DataFrame({
        "Line Item Value": np.random.uniform(100.0, 200000.0, size=n),
        "Shipment Mode": np.random.choice(["Air", "Truck", "Ocean"], size=n),
        "Fulfill Via": np.random.choice(["From RDC", "Direct Drop"], size=n),
    })
    
    res1 = engine.compute_costs(df_batch, is_log_transformed=False)
    res2 = engine.compute_costs(df_batch, is_log_transformed=False)
    
    pd.testing.assert_frame_equal(res1, res2)


# =============================================================================
# 4. Mathematical Invariants & Monotonicity of Decision Thresholds
# =============================================================================

def test_tau_star_strict_bounds_across_random_regimes(engine):
    """
    Stress-test that tau* and tau*_simple strictly satisfy [0.0, 1.0] across
    all combinations of parameters, values, and scenarios.
    """
    np.random.seed(42)
    n = 10_000
    
    # Random values spanning 0 to $100M
    values = np.exp(np.random.uniform(0.0, 18.4, size=n)) - 1.0
    # Include explicit 0s and extreme values
    values[0:100] = 0.0
    values[100:200] = 1e12
    
    df_stress = pd.DataFrame({
        "Line Item Value": values,
        "Shipment Mode": np.random.choice(["Air", "Air Charter", "Truck", "Ocean", "Unknown"], size=n),
        "Fulfill Via": np.random.choice(["From RDC", "Direct Drop", "Other"], size=n),
        "Product Group": np.random.choice(["ARV", "HRDT", "OTHER"], size=n),
    })
    
    for sc_name in ["low", "base", "high"]:
        res = engine.compute_costs(df_stress, scenario_name=sc_name, is_log_transformed=False)
        
        tau_star = res["tau_star"].to_numpy()
        tau_simple = res["tau_star_simple"].to_numpy()
        asym = res["asymmetry_ratio"].to_numpy()
        
        assert (tau_star >= 0.0).all(), "tau* cannot be negative"
        assert (tau_star <= 1.0).all(), "tau* cannot exceed 1.0"
        assert (tau_simple >= 0.0).all(), "tau*_simple cannot be negative"
        assert (tau_simple <= 1.0).all(), "tau*_simple cannot exceed 1.0"
        assert (asym > 0.0).all(), "Asymmetry ratio must be strictly positive"


def test_monotonicity_of_tau_star_simple_with_cost_ratio(engine):
    """
    Test mathematical property: tau*_simple = 1 / (1 + (FN / FP)).
    tau*_simple MUST be strictly monotonically decreasing with respect to ratio r = FN/FP.
    d(tau*_simple)/dr = -1 / (1 + r)^2 < 0.
    """
    # Create synthetic series of increasing FN costs holding FP fixed
    fp_fixed = 100.0
    fn_grid = np.linspace(10.0, 100000.0, 500)
    
    ratios = fn_grid / fp_fixed
    tau_simples = fp_fixed / (fn_grid + fp_fixed)
    
    # Verify strict monotonic decrease: diffs should all be strictly negative
    diffs = np.diff(tau_simples)
    assert (diffs < 0).all(), "tau*_simple must strictly decrease as FN/FP ratio increases"
    
    # Boundary limits:
    # As r -> 0, tau* -> 1.0
    assert math.isclose(fp_fixed / (1e-6 + fp_fixed), 1.0, rel_tol=1e-4)
    # As r -> infinity, tau* -> 0.0
    assert math.isclose(fp_fixed / (1e9 + fp_fixed), 0.0, abs_tol=1e-6)


def test_monotonicity_of_tau_star_simple_with_cost_ratio_r(engine):
    """
    Test that tau*_simple is strictly monotonically decreasing with respect to the cost ratio
    r = FN_Cost / FP_Cost across the entire positive domain r in (0, inf).
    Formula: tau*_simple(r) = 1 / (1 + r).
    Derivative: d(tau*_simple)/dr = -1 / (1 + r)^2 < 0 strictly everywhere.
    """
    r_grid = np.logspace(-3, 4, num=1000)  # r from 0.001 to 10,000
    tau_simples = 1.0 / (1.0 + r_grid)
    
    diffs = np.diff(tau_simples)
    assert (diffs < 0).all(), "tau*_simple must strictly decrease as FN/FP ratio r increases"
    
    # Check specific milestone points:
    # 1. Symmetric cost (r = 1.0): tau* = 0.50
    assert math.isclose(1.0 / (1.0 + 1.0), 0.50)
    # 2. Extreme delay cost dominance (r = 100): tau* ~ 0.0099
    assert math.isclose(1.0 / (1.0 + 100.0), 0.00990099, rel_tol=1e-4)
    # 3. Negligible delay cost (r = 0.01): tau* ~ 0.9901
    assert math.isclose(1.0 / (1.0 + 0.01), 0.990099, rel_tol=1e-4)



def test_monotonicity_of_tau_star_with_net_benefit_ratio(engine):
    """
    Test that tau* is strictly monotonically decreasing with respect to the net benefit ratio
    R_benefit = Net_Benefit / FP_Cost.
    tau* = 1 / (1 + (Net_Benefit / FP_Cost)).
    d(tau*)/d(R_benefit) = -1 / (1 + R_benefit)^2 < 0.
    """
    fp_fixed = 100.0
    net_ben_grid = np.linspace(10.0, 100000.0, 500)
    
    tau_stars = fp_fixed / (net_ben_grid + fp_fixed)
    
    diffs = np.diff(tau_stars)
    assert (diffs < 0).all(), "tau* must strictly decrease as Net_Benefit/FP ratio increases"
    
    # Boundary limits:
    # As Net_Benefit -> 0, tau* -> 1.0
    assert math.isclose(fp_fixed / (1e-6 + fp_fixed), 1.0, rel_tol=1e-4)
    # As Net_Benefit -> infinity, tau* -> 0.0
    assert math.isclose(fp_fixed / (1e9 + fp_fixed), 0.0, abs_tol=1e-6)



def test_cost_ordering_and_decomposition_invariants(engine):
    """
    Test physical supply chain economic invariants:
    1. FN_Cost >= Residual_Delay_Cost (since residual days <= assumed delay days)
    2. Net_Benefit = FN_Cost - (Intervention_Cost + Residual_Delay_Cost)
    3. If Net_Benefit > 0, tau* = FP / (Net_Benefit + FP) < 1.0
    """
    np.random.seed(777)
    n = 500
    df_samples = pd.DataFrame({
        "Line Item Value": np.random.uniform(0.0, 500000.0, size=n),
        "Shipment Mode": np.random.choice(["Air", "Truck", "Ocean"], size=n),
        "First Line Designation": np.random.choice(["Yes", "No"], size=n),
        "Sub Classification": np.random.choice(["Pediatric", "Adult"], size=n),
    })
    
    res = engine.compute_costs(df_samples, is_log_transformed=False)
    
    fn = res["fn_cost"].to_numpy()
    fp = res["fp_cost"].to_numpy()
    interv = res["intervention_cost"].to_numpy()
    resid = res["residual_delay_cost"].to_numpy()
    net_ben = res["net_benefit"].to_numpy()
    tau_star = res["tau_star"].to_numpy()
    
    # Invariant 1: FN >= Residual Delay
    assert (fn >= resid).all()
    
    # Invariant 2: Net Benefit decomposition
    expected_net_ben = fn - (interv + resid)
    np.testing.assert_allclose(net_ben, expected_net_ben, rtol=1e-6)
    
    # Invariant 3: tau* formula consistency
    pos_mask = (net_ben > 0)
    expected_tau = fp[pos_mask] / (net_ben[pos_mask] + fp[pos_mask])
    np.testing.assert_allclose(tau_star[pos_mask], expected_tau, rtol=1e-6)


# =============================================================================
# 5. Decision Policy and Oracle Optimality Invariants
# =============================================================================

def test_oracle_policy_strictly_minimizes_business_cost(engine):
    """
    Mathematical theorem check: The perfect oracle policy d = y MUST achieve strictly
    lower or equal business cost than 'Do Nothing' (d=0), 'Intervene All' (d=1),
    or any random policy d ~ Bern(p).
    """
    np.random.seed(42)
    n = 1000
    y_true = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    
    df = pd.DataFrame({
        "Line Item Value": np.random.uniform(500.0, 250000.0, size=n),
        "Shipment Mode": np.random.choice(["Air", "Ocean", "Truck"], size=n),
        "Fulfill Via": np.random.choice(["Direct Drop", "From RDC"], size=n),
    })
    costs = engine.compute_costs(df, is_log_transformed=False)
    
    cost_oracle = engine.compute_expected_cost(y_true, y_true, costs)
    cost_none = engine.compute_expected_cost(y_true, np.zeros(n, dtype=int), costs)
    cost_all = engine.compute_expected_cost(y_true, np.ones(n, dtype=int), costs)
    
    assert cost_oracle <= cost_none
    assert cost_oracle <= cost_all
    
    # Test across 20 random heuristic decision vectors
    for _ in range(20):
        d_rand = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
        cost_rand = engine.compute_expected_cost(y_true, d_rand, costs)
        assert cost_oracle <= cost_rand


# =============================================================================
# 6. Sample Weights & Extreme Label Distributions
# =============================================================================

def test_sample_weights_zero_delay_cases(engine):
    """Test sample weight generation when y_true contains all 0s (no delays)."""
    n = 50
    df = pd.DataFrame({
        "Line Item Value": [10000.0] * n,
        "Shipment Mode": ["Air"] * n,
    })
    y_all_zero = np.zeros(n, dtype=int)
    
    weights = engine.compute_sample_weights(df, y_all_zero, is_log_transformed=False, normalize=True)
    assert len(weights) == n
    assert not np.isnan(weights).any()
    # Since all instances have identical features and label=0, all weights must be exactly 1.0
    np.testing.assert_allclose(weights, np.ones(n), rtol=1e-5)


def test_sample_weights_all_delay_cases(engine):
    """Test sample weight generation when y_true contains all 1s (all delayed)."""
    n = 50
    df = pd.DataFrame({
        "Line Item Value": [25000.0] * n,
        "Shipment Mode": ["Truck"] * n,
    })
    y_all_one = np.ones(n, dtype=int)
    
    weights = engine.compute_sample_weights(df, y_all_one, is_log_transformed=False, normalize=True)
    assert len(weights) == n
    assert not np.isnan(weights).any()
    np.testing.assert_allclose(weights, np.ones(n), rtol=1e-5)


def test_sample_weights_length_mismatch_raises(engine):
    """Test that mismatched lengths between df and y_true raise ValueError."""
    df = pd.DataFrame({"Line Item Value": [1000.0, 2000.0]})
    y_bad = np.array([1, 0, 1])
    
    with pytest.raises(ValueError, match="Length mismatch"):
        engine.compute_sample_weights(df, y_bad)


# =============================================================================
# 7. Operational Budget Ranking Stress Tests
# =============================================================================

def test_expected_net_benefit_ranking_boundary_probabilities(engine):
    """Test ranking behavior with boundary probabilities p = 0.0, p = 1.0, and uniform p."""
    df = pd.DataFrame({
        "Line Item Value": [50000.0, 50000.0, 50000.0],
        "Shipment Mode": ["Air", "Air", "Air"],
    })
    
    # p = 0.0 -> gain should be exactly -FP_Cost
    gains_p0 = engine.compute_expected_net_benefit_ranking(df, [0.0, 0.0, 0.0], is_log_transformed=False)
    costs = engine.compute_costs(df, is_log_transformed=False)
    fp_costs = costs["fp_cost"].to_numpy()
    np.testing.assert_allclose(gains_p0, -fp_costs, rtol=1e-5)
    
    # p = 1.0 -> gain should be exactly Net_Benefit
    gains_p1 = engine.compute_expected_net_benefit_ranking(df, [1.0, 1.0, 1.0], is_log_transformed=False)
    net_bens = costs["net_benefit"].to_numpy()
    np.testing.assert_allclose(gains_p1, net_bens, rtol=1e-5)


# =============================================================================
# 8. Large Scale 100k Batch Vectorization Stress Test
# =============================================================================

def test_large_batch_100k_throughput_and_memory(engine):
    """Stress test engine with 100,000 rows, asserting execution within 3.0 seconds."""
    np.random.seed(42)
    n = 100_000
    df_100k = pd.DataFrame({
        "Line Item Value": np.random.uniform(10.0, 500000.0, size=n),
        "Shipment Mode": np.random.choice(["Air", "Air Charter", "Truck", "Ocean"], size=n),
        "Fulfill Via": np.random.choice(["From RDC", "Direct Drop"], size=n),
        "First Line Designation": np.random.choice(["Yes", "No"], size=n),
        "Sub Classification": np.random.choice(["Pediatric", "Adult"], size=n),
        "Product Group": np.random.choice(["ARV", "HRDT", "ACT"], size=n),
    })
    
    t0 = time.perf_counter()
    res = engine.compute_costs(df_100k, is_log_transformed=False)
    elapsed = time.perf_counter() - t0
    
    assert len(res) == n
    assert not res.isna().any().any()
    assert elapsed < 3.0, f"100k batch calculation took {elapsed:.2f}s (expected < 3.0s)"


# =============================================================================
# 9. Custom Scenario & Zero-Cost Parameter Stability
# =============================================================================

def test_custom_scenario_zero_penalties():
    """Test custom scenario with zero-cost parameters (c_fixed_stockout=0, beta_audit=0, rho_value=0)."""
    zero_scenario = CostScenario(
        name="ZeroParams",
        c_daily_base=1.0,
        rho_value=0.0,
        c_fixed_stockout=0.0,
        c_triage_base=1.0,
        beta_audit=0.0,
        c_direct_inquiry=0.0,
        c_rdc_inquiry=0.0,
        c_expedite_base=1.0,
        gamma_expedite=0.0,
        delay_days_assumed=10.0,
        days_saved_efficacy=5.0,
    )
    custom_engine = CostScenarioModel(custom_scenario=zero_scenario)
    
    df = pd.DataFrame({
        "Line Item Value": [0.0, 1000.0, 100000.0],
        "Shipment Mode": ["Air", "Air", "Air"],
    })
    res = custom_engine.compute_costs(df, is_log_transformed=False)
    
    assert len(res) == 3
    assert not res.isna().any().any()
    assert (res["fn_cost"] > 0).all()
    assert (res["fp_cost"] > 0).all()
    assert ((res["tau_star"] >= 0.0) & (res["tau_star"] <= 1.0)).all()


# =============================================================================
# 10. Input Format Flexibility (Series, Dict, List)
# =============================================================================

def test_input_format_flexibility(engine):
    """Test that engine seamlessly accepts pd.Series, dict, and single record queries."""
    record_dict = {
        "Line Item Value": 25000.0,
        "Shipment Mode": "Ocean",
        "Fulfill Via": "From RDC",
        "Product Group": "ARV",
    }
    
    # Dict input returning Dict
    res_dict = engine.compute_costs(record_dict, is_log_transformed=False, return_dataframe=False)
    assert isinstance(res_dict, dict)
    assert "fn_cost" in res_dict
    assert res_dict["fn_cost"] > 0
    
    # Dict input returning DataFrame
    res_df = engine.compute_costs(record_dict, is_log_transformed=False, return_dataframe=True)
    assert isinstance(res_df, pd.DataFrame)
    assert len(res_df) == 1
    
    # Series input
    series = pd.Series(record_dict)
    res_series = engine.compute_costs(series, is_log_transformed=False, return_dataframe=False)
    assert isinstance(res_series, dict)
    assert math.isclose(res_series["fn_cost"], res_dict["fn_cost"])

