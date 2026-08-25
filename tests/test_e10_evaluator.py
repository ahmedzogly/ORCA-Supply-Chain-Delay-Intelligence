"""
Unit and integration tests for CounterfactualEvaluator and dev temporal backtesting.
"""

from __future__ import annotations

import pandas as pd
import pytest

from delay_intelligence.counterfactual.evaluator import CounterfactualEvaluator


@pytest.fixture
def evaluator() -> CounterfactualEvaluator:
    return CounterfactualEvaluator()


def test_dev_cohort_loading_and_quarantine(evaluator: CounterfactualEvaluator):
    """
    Verifies that the development cohort contains exactly the dev records (N=7,306)
    and that no records from the final 365-day holdout (T_pred > 2014-08-24) are loaded.
    """
    df_dev = evaluator.load_dev_data()
    assert len(df_dev) == 7306
    assert df_dev["T_pred"].max() <= pd.Timestamp("2014-08-24")

    # Double check total SCMS dataset dimensions
    df_all = pd.read_parquet(evaluator.feature_path)
    holdout_count = (pd.to_datetime(df_all["T_pred"]) > pd.Timestamp("2014-08-24")).sum()
    assert holdout_count == 1013
    assert len(df_dev) + holdout_count == len(df_all) == 8319


def test_predictions_generation(evaluator: CounterfactualEvaluator):
    """Verifies CatBoost prediction generation on dev sample."""
    df_dev = evaluator.load_dev_data().head(50)
    probs, exp_delays, uncert_widths = evaluator.generate_predictions(df_dev)

    assert len(probs) == 50
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert (exp_delays >= 0.0).all()
    assert (uncert_widths >= 0.1).all()


def test_cohort_evaluation_structure(evaluator: CounterfactualEvaluator):
    """Verifies cohort evaluation outputs required columns and provenance tags."""
    df_dev = evaluator.load_dev_data().head(20)
    states = evaluator.build_observable_states(df_dev, scenario_name="base")
    res_df = evaluator.evaluate_cohort(states, scenario_name="base", fold_id=0)

    expected_cols = [
        "fold_id",
        "scenario",
        "policy_id",
        "policy_name",
        "shipment_id",
        "pred_date",
        "action_selected",
        "action_cost",
        "residual_delay_days",
        "residual_delay_prob",
        "residual_delay_cost",
        "residual_risk_cost",
        "expected_realized_cost",
        "no_action_cost",
        "net_benefit",
        "oracle_cost",
        "oracle_action",
        "policy_regret",
        "hysteresis_stable",
        "provenance_tag",
    ]
    for col in expected_cols:
        assert col in res_df.columns

    assert set(res_df["policy_id"].unique()) == {"P0", "P1", "P2", "P3", "P4", "P5", "Oracle"}
    assert (res_df["policy_regret"] >= 0.0).all()
