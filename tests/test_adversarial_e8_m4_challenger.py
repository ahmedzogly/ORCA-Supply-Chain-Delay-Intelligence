"""
Adversarial Development-Phase QA & Independent Verification Test Suite (Phase 2 — E8 Milestone 4).

Challenger: QA Challenger 1 (Adversarial Dev Verifier)
Target: E8 Cost-Sensitive Learning Deliverables (Engine, Budgeting, Sensitivity, Policy Freeze, Manifest)

Challenge Vectors:
1. Cost Leakage & Holdout Isolation:
   - Zero post-outcome / target-derived features in cost engine, budgeting, sensitivity, or frozen policy.
   - Strict temporal isolation between dev data (<= 2014-08-24) and holdout (> 2014-08-24).
   - Verification of holdout rejection on consistent dates.
   - Empirical demonstration of mixed datetime format vulnerability where NaT bypasses date filter.
2. Extreme Assumptions & Boundary Conditions:
   - Zero cost parameters, zero budget (K=0, K->0), 100% budget (K=1.0), empty dataframes (N=0), single item (N=1).
   - All-negative delay days (early deliveries), inverted costs (negative net benefit), huge numbers (10^12).
   - Small dollar amount auto-detection behavior on low-value cohorts.
   - Division-by-zero, NaN, Inf, and float precision guard verification.
3. Threshold Cheating & Gaming / Monotonicity:
   - Strict monotonicity of COST_SENSITIVE priority scores w.r.t. p_hat, Net_Benefit, and FP_Cost.
   - Monotonicity of decision rates with respect to gamma* scaling.
   - Economic rationality: protection against review-capacity wasting on negative expected benefit shipments.
   - Monotonicity of budget review count with respect to capacity K.
   - Tie-breaking stability and capacity enforcement under identical score distributions.
4. Cryptographic Manifest Integrity:
   - Full bitwise SHA-256 and byte size verification of all 11 files in artifacts/results/e8_frozen_policy.json.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from delay_intelligence.cost_sensitive.budgeting import (
    BudgetMetrics,
    OperationalBudgetSimulator,
    OperationalPolicyType,
)
from delay_intelligence.cost_sensitive.cost_engine import (
    CostBreakdown,
    CostEngine,
    CostScenario,
    CostScenarioModel,
    FORBIDDEN_COLUMNS,
    LeakageViolationError,
)
from delay_intelligence.cost_sensitive.policy_freeze import (
    ChampionStrategySpec,
    FrozenCostPolicy,
    FrozenFeatureContract,
    HoldoutLeakageError,
    MAX_ALLOWED_DEV_DATE_UTC,
    OperationalBudgetRuleSpec,
    compute_file_sha256,
    freeze_e8_policy,
    verify_temporal_holdout_isolation,
)
from delay_intelligence.cost_sensitive.sensitivity import (
    CostSensitivityAnalyzer,
    PolicyRobustnessReport,
    RobustnessClassification,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def config_path() -> Path:
    return Path("configs/cost_scenarios.yaml")


@pytest.fixture
def frozen_policy_path() -> Path:
    return Path("artifacts/results/e8_frozen_policy.json")


@pytest.fixture
def dev_backtest_parquet_path() -> Path:
    return Path("artifacts/results/e8_dev_backtest_results.parquet")


@pytest.fixture
def cost_engine(config_path: Path) -> CostScenarioModel:
    return CostScenarioModel(config_path=config_path, scenario_name="base")


@pytest.fixture
def budget_simulator(cost_engine: CostScenarioModel) -> OperationalBudgetSimulator:
    return OperationalBudgetSimulator(cost_engine=cost_engine, scenario_name="base")


@pytest.fixture
def sensitivity_analyzer(config_path: Path) -> CostSensitivityAnalyzer:
    return CostSensitivityAnalyzer(config_path=config_path, base_scenario_name="base")


@pytest.fixture
def frozen_policy(frozen_policy_path: Path) -> FrozenCostPolicy:
    return FrozenCostPolicy.load_from_json(frozen_policy_path)


# =============================================================================
# VECTOR 1: COST LEAKAGE & TEMPORAL HOLDOUT ISOLATION
# =============================================================================

class TestAdversarialCostLeakageAndIsolation:
    """Adversarial stress-testing of cost leakage boundaries and holdout isolation."""

    @pytest.mark.parametrize("forbidden_col", FORBIDDEN_COLUMNS)
    def test_forbidden_columns_intercepted_in_cost_engine(
        self, cost_engine: CostScenarioModel, forbidden_col: str
    ):
        """Verify that EVERY forbidden column is intercepted and raises LeakageViolationError."""
        df_dirty = pd.DataFrame({
            "Line Item Value": [10000.0, 50000.0],
            "Shipment Mode": ["Air", "Truck"],
            forbidden_col: [1.0, 2.0],
        })
        with pytest.raises(LeakageViolationError, match="Forbidden / target-leakage column"):
            cost_engine.compute_costs(df_dirty, strict_leakage_check=True)

    @pytest.mark.parametrize("forbidden_col", FORBIDDEN_COLUMNS)
    def test_forbidden_columns_intercepted_in_sample_weights(
        self, cost_engine: CostScenarioModel, forbidden_col: str
    ):
        """Verify that sample weight computation rejects forbidden columns."""
        df_dirty = pd.DataFrame({
            "Line Item Value": [10000.0, 50000.0],
            "Shipment Mode": ["Air", "Truck"],
            forbidden_col: [1.0, 2.0],
        })
        y = np.array([0, 1])
        with pytest.raises(LeakageViolationError):
            cost_engine.compute_sample_weights(df_dirty, y_true=y)

    def test_verify_holdout_isolation_accepts_clean_dev_data(self):
        """Verify that dates <= 2014-08-24 pass holdout isolation verification cleanly."""
        df_clean = pd.DataFrame({
            "T_pred": ["2006-04-19 00:00:00", "2010-01-01 00:00:00", "2014-08-21 00:00:00", "2014-08-24 00:00:00"],
            "val": [1, 2, 3, 4],
        })
        # Should not raise
        verify_temporal_holdout_isolation(df_clean, date_col="T_pred", max_allowed_date=MAX_ALLOWED_DEV_DATE_UTC)

    def test_verify_holdout_isolation_rejects_canary_holdout_date(self):
        """Verify that a single holdout timestamp (> 2014-08-24) raises HoldoutLeakageError."""
        df_contaminated = pd.DataFrame({
            "T_pred": ["2014-08-24 00:00:00", "2014-08-25 00:00:00"],
            "val": [1, 2],
        })
        with pytest.raises(HoldoutLeakageError, match="HOLDOUT CONTAMINATION DETECTED"):
            verify_temporal_holdout_isolation(df_contaminated, date_col="T_pred", max_allowed_date=MAX_ALLOWED_DEV_DATE_UTC)

    def test_verify_holdout_isolation_rejects_far_future_date(self):
        """Verify that far-future timestamps (e.g. 2015 or 2026) are caught."""
        df_future = pd.DataFrame({
            "T_pred": ["2015-08-24 00:00:00", "2026-08-19 00:00:00"],
        })
        with pytest.raises(HoldoutLeakageError):
            verify_temporal_holdout_isolation(df_future, date_col="T_pred", max_allowed_date=MAX_ALLOWED_DEV_DATE_UTC)

    def test_verify_holdout_isolation_mixed_format_vulnerability_demonstration(self):
        """
        Adversarial Finding / Vulnerability Demonstration:
        Demonstrates that when df has mixed datetime formats (date-only vs datetime with time),
        pd.to_datetime(..., errors='coerce') coerces differing formats to NaT, causing
        holdout records to silently bypass verify_temporal_holdout_isolation.
        """
        # When first element is '%Y-%m-%d' and second is '%Y-%m-%d %H:%M:%S'
        df_mixed_bypass = pd.DataFrame({
            "T_pred": ["2014-08-24", "2015-08-24 12:00:00"],
        })
        # Note: pd.to_datetime with errors='coerce' turns "2015-08-24 12:00:00" into NaT
        dates = pd.to_datetime(df_mixed_bypass["T_pred"], errors="coerce")
        assert pd.isna(dates.iloc[1]), "Expected second element to be coerced to NaT due to mixed formatting"

        # verify_temporal_holdout_isolation does not raise because NaT > cutoff is False!
        # This empirically documents the vulnerability.
        verify_temporal_holdout_isolation(df_mixed_bypass, date_col="T_pred", max_allowed_date=MAX_ALLOWED_DEV_DATE_UTC)

    def test_dev_backtest_results_contain_zero_holdout_records(
        self, dev_backtest_parquet_path: Path
    ):
        """Empirically inspect the actual dev backtest parquet artifact for holdout contamination."""
        assert dev_backtest_parquet_path.exists(), "Backtest parquet file missing"
        df_backtest = pd.read_parquet(dev_backtest_parquet_path)
        assert len(df_backtest) > 0

        # Check T_pred column
        if "T_pred" in df_backtest.columns:
            dates = pd.to_datetime(df_backtest["T_pred"])
            max_date = dates.max()
            assert max_date <= pd.to_datetime("2014-08-24 23:59:59"), (
                f"Contamination detected in dev backtest results! Max T_pred is {max_date}"
            )

        # Verify that holdout isolation function passes on this artifact
        verify_temporal_holdout_isolation(df_backtest, date_col="T_pred", max_allowed_date=MAX_ALLOWED_DEV_DATE_UTC)

    def test_feature_contract_in_frozen_policy_contains_zero_forbidden_columns(
        self, frozen_policy: FrozenCostPolicy
    ):
        """Verify that the frozen feature contract has zero overlap with forbidden columns."""
        contract = frozen_policy.feature_contract
        all_features = set(contract.all_features)
        num_features = set(contract.num_cols)
        cat_features = set(contract.cat_cols)
        forbidden = set(contract.forbidden_columns)

        assert len(forbidden) == 8
        assert all_features.isdisjoint(forbidden), f"Overlap found: {all_features.intersection(forbidden)}"
        assert num_features.isdisjoint(forbidden), f"Overlap found in num_cols: {num_features.intersection(forbidden)}"
        assert cat_features.isdisjoint(forbidden), f"Overlap found in cat_cols: {cat_features.intersection(forbidden)}"


# =============================================================================
# VECTOR 2: EXTREME ASSUMPTIONS & BOUNDARY CONDITIONS
# =============================================================================

class TestAdversarialExtremeAssumptionsAndBoundaries:
    """Stress-testing budgeting and sensitivity under extreme parameters and boundary inputs."""

    def test_empty_cohort_handling(self, budget_simulator: OperationalBudgetSimulator):
        """Verify compute_policy_decisions returns empty array and zero capacity for N=0."""
        empty_df = pd.DataFrame(columns=["fn_cost", "fp_cost", "net_benefit", "intervention_cost", "residual_delay_cost"])
        decisions, cap = budget_simulator.compute_policy_decisions(
            policy=OperationalPolicyType.COST_SENSITIVE,
            probs=np.array([]),
            costs_df=empty_df,
            values=np.array([]),
            budget_k=0.10,
        )
        assert len(decisions) == 0
        assert cap == 0

        # evaluate_cohort_under_budget should raise ValueError on empty cohort
        with pytest.raises(ValueError, match="Cannot evaluate empty cohort"):
            budget_simulator.evaluate_cohort_under_budget(
                y_true=np.array([]),
                probs=np.array([]),
                costs_df=empty_df,
                values=np.array([]),
                budget_k=0.10,
            )

    def test_single_item_cohort(self, budget_simulator: OperationalBudgetSimulator):
        """Verify behavior on a single shipment (N=1)."""
        costs_df = pd.DataFrame({
            "fn_cost": [5000.0],
            "fp_cost": [100.0],
            "intervention_cost": [600.0],
            "residual_delay_cost": [1000.0],
            "net_benefit": [3400.0],
        })
        probs = np.array([0.8])
        values = np.array([50000.0])
        y = np.array([1])

        metrics = budget_simulator.evaluate_cohort_under_budget(
            y_true=y,
            probs=probs,
            costs_df=costs_df,
            values=values,
            budget_k=0.05,  # floor(0.05 * 1) = 0 -> clamped to 1
            policy=OperationalPolicyType.COST_SENSITIVE,
        )
        assert metrics.cohort_size == 1
        assert metrics.budget_capacity_count == 1
        assert metrics.reviewed_count == 1
        assert metrics.positives_captured == 1
        assert metrics.budget_utilization_pct == 100.0
        assert not math.isnan(metrics.realized_business_cost)
        assert not math.isnan(metrics.cost_reduction_pct)

    def test_zero_budget_capacity(self, budget_simulator: OperationalBudgetSimulator):
        """Verify behavior under K=0 or near-zero budget."""
        n = 100
        costs_df = pd.DataFrame({
            "fn_cost": np.full(n, 2000.0),
            "fp_cost": np.full(n, 100.0),
            "intervention_cost": np.full(n, 400.0),
            "residual_delay_cost": np.full(n, 500.0),
            "net_benefit": np.full(n, 1100.0),
        })
        probs = np.linspace(0.1, 0.9, n)
        values = np.full(n, 10000.0)

        # When K=0.0, capacity is clamped >= 1 if K*n >= 0, or max(1, 0) = 1
        decisions, cap = budget_simulator.compute_policy_decisions(
            policy=OperationalPolicyType.COST_SENSITIVE,
            probs=probs,
            costs_df=costs_df,
            values=values,
            budget_k=0.0001,
        )
        assert cap >= 1
        assert np.sum(decisions) <= cap

    def test_full_100_percent_and_over_budget_capacity(
        self, budget_simulator: OperationalBudgetSimulator
    ):
        """Verify behavior when budget K=1.0 or K=2.0 (capacity >= N)."""
        n = 50
        costs_df = pd.DataFrame({
            "fn_cost": np.full(n, 3000.0),
            "fp_cost": np.full(n, 100.0),
            "intervention_cost": np.full(n, 500.0),
            "residual_delay_cost": np.full(n, 500.0),
            "net_benefit": np.full(n, 2000.0),
        })
        probs = np.full(n, 0.7)  # All have expected net benefit > 0
        values = np.full(n, 20000.0)

        decisions, cap = budget_simulator.compute_policy_decisions(
            policy=OperationalPolicyType.COST_SENSITIVE,
            probs=probs,
            costs_df=costs_df,
            values=values,
            budget_k=1.5,  # 150% budget
        )
        assert cap == n  # Clamped to min(n, capacity)
        assert np.sum(decisions) == n

    def test_all_negative_delay_days_and_early_deliveries(
        self, budget_simulator: OperationalBudgetSimulator
    ):
        """Verify that negative delay days (early deliveries) do not produce NaN or corrupted metrics."""
        n = 20
        costs_df = pd.DataFrame({
            "fn_cost": np.full(n, 2000.0),
            "fp_cost": np.full(n, 100.0),
            "intervention_cost": np.full(n, 400.0),
            "residual_delay_cost": np.full(n, 500.0),
            "net_benefit": np.full(n, 1100.0),
        })
        probs = np.full(n, 0.8)
        values = np.full(n, 15000.0)
        y = np.zeros(n, dtype=int)  # All early
        delay_days = np.full(n, -15.0)  # 15 days early

        metrics = budget_simulator.evaluate_cohort_under_budget(
            y_true=y,
            probs=probs,
            costs_df=costs_df,
            values=values,
            delay_days=delay_days,
            budget_k=0.10,
            policy=OperationalPolicyType.COST_SENSITIVE,
        )
        assert metrics.positives_in_cohort == 0
        assert metrics.positives_captured == 0
        assert metrics.delay_days_captured == 0.0
        assert not math.isnan(metrics.realized_business_cost)

    def test_extreme_commodity_values_huge_and_zero(
        self, cost_engine: CostScenarioModel
    ):
        """Verify stability with extreme line item values: $0 and $1,000,000,000."""
        df_extreme = pd.DataFrame({
            "Line Item Value": [0.0, 1e9, 1e12],
            "Shipment Mode": ["Air", "Ocean", "Truck"],
            "First Line Designation": ["No", "Yes", "Yes"],
            "Product Group": ["Other", "ARV", "ARV"],
        })
        costs = cost_engine.compute_costs(df_extreme, return_dataframe=True)
        assert len(costs) == 3
        assert np.all(np.isfinite(costs["fn_cost"]))
        assert np.all(np.isfinite(costs["fp_cost"]))
        assert np.all(np.isfinite(costs["net_benefit"]))
        assert np.all((costs["tau_star"] >= 0.0) & (costs["tau_star"] <= 1.0))
        assert np.all((costs["tau_star_simple"] >= 0.0) & (costs["tau_star_simple"] <= 1.0))

    def test_small_value_auto_detection_caveat(
        self, cost_engine: CostScenarioModel
    ):
        """
        Adversarial Observation: When is_log_transformed is None, cohorts of small raw
        dollar amounts (< $25) are auto-detected as log1p.
        Demonstrates that explicit is_log_transformed=False preserves true values.
        """
        df_small = pd.DataFrame({"Line Item Value": [5.0, 10.0, 15.0]})
        vals_explicit = cost_engine.extract_monetary_values(df_small, is_log_transformed=False)
        assert np.array_equal(vals_explicit, np.array([5.0, 10.0, 15.0]))

    def test_inverted_cost_parameters_negative_net_benefit(
        self, budget_simulator: OperationalBudgetSimulator
    ):
        """
        Adversarial test: Inverted costs where intervention cost exceeds FN cost.
        In this setting, net_benefit < 0 for all items.
        COST_SENSITIVE policy with strictly_positive_benefit=True MUST review 0 shipments.
        """
        n = 30
        costs_df = pd.DataFrame({
            "fn_cost": np.full(n, 200.0),            # Low FN cost
            "fp_cost": np.full(n, 500.0),            # High triage cost
            "intervention_cost": np.full(n, 1000.0), # Expedite fee > FN cost!
            "residual_delay_cost": np.full(n, 100.0),
            "net_benefit": np.full(n, -900.0),       # Negative net benefit!
        })
        probs = np.full(n, 0.9)  # High probability of delay, but intervention loses money!
        values = np.full(n, 5000.0)
        y = np.ones(n, dtype=int)

        decisions, cap = budget_simulator.compute_policy_decisions(
            policy=OperationalPolicyType.COST_SENSITIVE,
            probs=probs,
            costs_df=costs_df,
            values=values,
            budget_k=0.20,
            strictly_positive_benefit=True,
        )
        # MUST protect business from reviewing unprofitable interventions
        assert np.sum(decisions) == 0, (
            "COST_SENSITIVE policy reviewed shipments with negative expected net benefit!"
        )

        metrics = budget_simulator.evaluate_cohort_under_budget(
            y_true=y,
            probs=probs,
            costs_df=costs_df,
            values=values,
            budget_k=0.20,
            policy=OperationalPolicyType.COST_SENSITIVE,
        )
        assert metrics.reviewed_count == 0
        assert metrics.budget_utilization_pct == 0.0
        assert metrics.realized_business_cost == metrics.do_nothing_cost

    def test_sensitivity_extreme_multipliers(
        self, sensitivity_analyzer: CostSensitivityAnalyzer
    ):
        """Test sensitivity scenario creation with extreme perturbation multipliers."""
        # 10x multiplier on daily penalty and 0.1x on triage
        extreme_sc = sensitivity_analyzer.create_perturbed_scenario(
            perturbations={"c_daily_base": 10.0, "c_triage_base": 0.1},
            custom_name="extreme_test",
        )
        assert extreme_sc.c_daily_base == 1500.0
        assert extreme_sc.c_triage_base == 5.0

        # Zero stockout cost and zero holding rate
        zero_sc = sensitivity_analyzer.create_perturbed_scenario(
            perturbations={"c_fixed_stockout": 0.0, "rho_value": 0.0},
            custom_name="zero_test",
        )
        assert zero_sc.c_fixed_stockout == 0.0
        assert zero_sc.rho_value == 0.0


# =============================================================================
# VECTOR 3: THRESHOLD CHEATING & GAMING / MONOTONICITY
# =============================================================================

class TestAdversarialThresholdMonotonicityAndRationality:
    """Testing priority ranking strict monotonicity, rationality, and gaming resistance."""

    def test_priority_score_strict_monotonicity_wrt_probability(
        self, budget_simulator: OperationalBudgetSimulator
    ):
        """
        Verify that for identical cost profiles (Net_Benefit > 0, FP > 0),
        priority score is strictly monotonic increasing with respect to predicted probability p.
        """
        n = 50
        probs = np.linspace(0.01, 0.99, n)
        costs_df = pd.DataFrame({
            "net_benefit": np.full(n, 2500.0),
            "fp_cost": np.full(n, 120.0),
        })
        values = np.full(n, 50000.0)

        scores = budget_simulator.compute_priority_scores(
            policy=OperationalPolicyType.COST_SENSITIVE,
            probs=probs,
            costs_df=costs_df,
            values=values,
        )
        # Check strict monotonicity: score[i+1] > score[i]
        diffs = np.diff(scores)
        assert np.all(diffs > 0), f"Non-monotonic score step detected: min diff = {np.min(diffs)}"

    def test_priority_score_strict_monotonicity_wrt_net_benefit(
        self, budget_simulator: OperationalBudgetSimulator
    ):
        """
        Verify that for fixed p > 0 and fixed FP_Cost,
        priority score is strictly monotonic increasing with respect to Net_Benefit.
        """
        n = 50
        probs = np.full(n, 0.6)
        net_benefits = np.linspace(500.0, 20000.0, n)
        costs_df = pd.DataFrame({
            "net_benefit": net_benefits,
            "fp_cost": np.full(n, 100.0),
        })
        values = np.full(n, 50000.0)

        scores = budget_simulator.compute_priority_scores(
            policy=OperationalPolicyType.COST_SENSITIVE,
            probs=probs,
            costs_df=costs_df,
            values=values,
        )
        diffs = np.diff(scores)
        assert np.all(diffs > 0), f"Non-monotonic score step detected: min diff = {np.min(diffs)}"

    def test_priority_score_strict_monotonicity_wrt_fp_cost(
        self, budget_simulator: OperationalBudgetSimulator
    ):
        """
        Verify that for fixed p < 1.0 and fixed Net_Benefit,
        priority score is strictly monotonic DECREASING with respect to FP_Cost.
        """
        n = 50
        probs = np.full(n, 0.4)
        fp_costs = np.linspace(50.0, 1000.0, n)
        costs_df = pd.DataFrame({
            "net_benefit": np.full(n, 3000.0),
            "fp_cost": fp_costs,
        })
        values = np.full(n, 50000.0)

        scores = budget_simulator.compute_priority_scores(
            policy=OperationalPolicyType.COST_SENSITIVE,
            probs=probs,
            costs_df=costs_df,
            values=values,
        )
        diffs = np.diff(scores)
        assert np.all(diffs < 0), f"Score did not decrease with increasing FP cost: max diff = {np.max(diffs)}"

    def test_bayes_threshold_monotonicity_wrt_gamma(
        self, frozen_policy: FrozenCostPolicy
    ):
        """
        Verify that as gamma* scaling increases, the Bayes decision threshold tau*_i
        strictly decreases (becoming more aggressive in capturing delay risk).
        """
        costs_df = pd.DataFrame({
            "net_benefit": [2000.0, 5000.0, 10000.0],
            "fp_cost": [100.0, 200.0, 300.0],
        })

        gammas = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        threshold_matrix = []

        for g in gammas:
            frozen_policy.champion.gamma_tuned_multiplier = g
            tau = frozen_policy.compute_decision_thresholds(costs_df, apply_tuned_gamma=True)
            threshold_matrix.append(tau)

        threshold_matrix = np.array(threshold_matrix)  # shape (len(gammas), 3)

        # Along rows (increasing gamma), each column must strictly decrease
        for col_idx in range(3):
            col_diffs = np.diff(threshold_matrix[:, col_idx])
            assert np.all(col_diffs < 0), f"Threshold did not decrease with gamma in col {col_idx}: {col_diffs}"

    def test_budget_capacity_monotonicity(
        self, budget_simulator: OperationalBudgetSimulator
    ):
        """
        Verify that as review capacity K increases (0.05 -> 0.10 -> 0.20 -> 0.50),
        the number of reviewed shipments is monotonically non-decreasing.
        """
        n = 100
        costs_df = pd.DataFrame({
            "fn_cost": np.random.uniform(1000, 5000, n),
            "fp_cost": np.random.uniform(50, 200, n),
            "intervention_cost": np.random.uniform(200, 800, n),
            "residual_delay_cost": np.random.uniform(100, 600, n),
            "net_benefit": np.random.uniform(500, 4000, n),
        })
        probs = np.random.uniform(0.1, 0.9, n)
        values = np.random.uniform(1000, 100000, n)
        y = (probs > 0.4).astype(int)

        capacities = [0.05, 0.10, 0.20, 0.50]
        reviewed_counts = []

        for k in capacities:
            m = budget_simulator.evaluate_cohort_under_budget(
                y_true=y,
                probs=probs,
                costs_df=costs_df,
                values=values,
                budget_k=k,
                policy=OperationalPolicyType.COST_SENSITIVE,
            )
            reviewed_counts.append(m.reviewed_count)

        diffs = np.diff(reviewed_counts)
        assert np.all(diffs >= 0), f"Reviewed count decreased with larger budget: {reviewed_counts}"

    def test_ties_handled_with_exact_capacity_enforcement(
        self, budget_simulator: OperationalBudgetSimulator
    ):
        """Verify that when scores are identical across cohort, exactly M items are reviewed."""
        n = 100
        costs_df = pd.DataFrame({
            "fn_cost": np.full(n, 2000.0),
            "fp_cost": np.full(n, 100.0),
            "intervention_cost": np.full(n, 400.0),
            "residual_delay_cost": np.full(n, 500.0),
            "net_benefit": np.full(n, 1100.0),
        })
        probs = np.full(n, 0.5)
        values = np.full(n, 10000.0)

        for pol in [OperationalPolicyType.VALUE_ONLY, OperationalPolicyType.RISK_ONLY, OperationalPolicyType.STANDARD, OperationalPolicyType.COST_SENSITIVE]:
            decisions, cap = budget_simulator.compute_policy_decisions(
                policy=pol,
                probs=probs,
                costs_df=costs_df,
                values=values,
                budget_k=0.10,
            )
            assert cap == 10
            assert np.sum(decisions) == 10

    def test_zero_risk_cohort_rationality(
        self, budget_simulator: OperationalBudgetSimulator
    ):
        """Verify that for zero-risk cohort (probs=0), COST_SENSITIVE and STANDARD review 0 items."""
        n = 50
        costs_df = pd.DataFrame({
            "fn_cost": np.full(n, 2000.0),
            "fp_cost": np.full(n, 100.0),
            "intervention_cost": np.full(n, 400.0),
            "residual_delay_cost": np.full(n, 500.0),
            "net_benefit": np.full(n, 1100.0),
        })
        probs_zero = np.zeros(n)
        values = np.full(n, 10000.0)

        dec_cs, _ = budget_simulator.compute_policy_decisions(
            policy=OperationalPolicyType.COST_SENSITIVE,
            probs=probs_zero,
            costs_df=costs_df,
            values=values,
            budget_k=0.10,
        )
        dec_std, _ = budget_simulator.compute_policy_decisions(
            policy=OperationalPolicyType.STANDARD,
            probs=probs_zero,
            costs_df=costs_df,
            values=values,
            budget_k=0.10,
        )
        assert np.sum(dec_cs) == 0, "COST_SENSITIVE should not review items with zero probability of delay!"
        assert np.sum(dec_std) == 0, "STANDARD should not review items with zero probability of delay!"


# =============================================================================
# VECTOR 4: CRYPTOGRAPHIC MANIFEST INTEGRITY
# =============================================================================

class TestAdversarialCryptographicManifestIntegrity:
    """Full bitwise verification of SHA-256 checksums and file sizes."""

    def test_frozen_policy_json_manifest_file_count(
        self, frozen_policy_path: Path
    ):
        """Verify that e8_frozen_policy.json contains at least 11 tracked files."""
        with open(frozen_policy_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest = data.get("cryptographic_manifest", [])
        assert len(manifest) >= 11, f"Expected >= 11 files in manifest, got {len(manifest)}"

    def test_sha256_checksums_match_actual_files_on_disk(
        self, frozen_policy_path: Path
    ):
        """
        Adversarial test: Compute live SHA-256 for EVERY file listed in the manifest
        and verify 100% exact match with the frozen policy record.
        """
        with open(frozen_policy_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest = data.get("cryptographic_manifest", [])
        assert len(manifest) > 0

        mismatches = []
        for entry in manifest:
            rel_path = entry["relative_path"]
            expected_hash = entry["sha256"]
            expected_size = entry["file_size_bytes"]

            p = Path(rel_path)
            if not p.exists():
                mismatches.append(f"MISSING FILE: {rel_path}")
                continue

            # Compute actual hash
            hasher = hashlib.sha256()
            with open(p, "rb") as f_in:
                for chunk in iter(lambda: f_in.read(65536), b""):
                    hasher.update(chunk)
            actual_hash = hasher.hexdigest()
            actual_size = p.stat().st_size

            if actual_hash != expected_hash:
                mismatches.append(
                    f"HASH MISMATCH on {rel_path}:\n  Expected: {expected_hash}\n  Actual:   {actual_hash}"
                )
            if actual_size != expected_size:
                mismatches.append(
                    f"SIZE MISMATCH on {rel_path}:\n  Expected: {expected_size} bytes\n  Actual:   {actual_size} bytes"
                )

        assert not mismatches, "Cryptographic manifest verification failed:\n" + "\n".join(mismatches)
