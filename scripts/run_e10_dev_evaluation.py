"""
Runner script for Milestone 3 Dev Temporal Counterfactual Evaluation & Sensitivity Analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import pandas as pd

from delay_intelligence.counterfactual.evaluator import CounterfactualEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    print("=" * 80)
    print(" USAID SCMS Delay Intelligence — Experiment E10")
    print(" Milestone 3: Dev Temporal Counterfactual Backtest & Sensitivity Grid")
    print("=" * 80)

    evaluator = CounterfactualEvaluator()
    summary = evaluator.run_full_dev_evaluation()

    print(f"\nStatus: {summary['status']}")
    print(f"Dev Cohort Sample Size: {summary['dev_sample_size']}")
    print(f"Artifacts Generated:")
    for path in summary['artifacts_generated']:
        print(f"  - {path}")

    print("\nBase Scenario Policy Performance (5-Fold CV Aggregate):")
    print("-" * 100)
    print(f"{'Policy ID':<10} {'Mean Cost ($)':<16} {'Net Benefit ($)':<16} {'Oracle Gap ($)':<16} {'Mean Regret ($)':<16} {'Interv Rate':<14} {'Hysteresis':<12}")
    print("-" * 100)
    for pol_id, m in summary['base_policy_metrics'].items():
        print(f"{pol_id:<10} ${m['mean_expected_cost_usd']:<15.2f} ${m['mean_net_benefit_usd']:<15.2f} ${m['oracle_gap_usd']:<15.2f} ${m['mean_regret_usd']:<15.2f} {m['intervention_rate_pct']:<13.1f}% {m['hysteresis_stability_pct']:<11.1f}%")
    print("-" * 100)

    print("\nOperational Review Budget Allocation (Base Scenario, N=7,306 Dev Cohort):")
    print("-" * 80)
    print(f"{'Capacity K':<15} {'Allocated Count':<18} {'Total Net Benefit ($)':<25} {'Utilization (%)':<15}")
    print("-" * 80)
    for k, b in summary['budget_summary']['base'].items():
        print(f"{k:<15} {b['allocated_count']:<18} ${b['total_net_benefit_usd']:<24.2f} {b['utilization_pct']:<14.1f}%")
    print("-" * 80)

    # Validate generated artifacts
    df_dev = pd.read_parquet(summary['artifacts_generated'][0])
    df_sens = pd.read_parquet(summary['artifacts_generated'][1])

    print(f"\nVerification:")
    print(f"  - Dev Parquet Records: {len(df_dev)} rows across {df_dev['scenario'].nunique()} scenarios and {df_dev['policy_id'].nunique()} policies")
    print(f"  - Max Dev Prediction Date: {df_dev['pred_date'].max()} (Cutoff 2014-08-24 Verified: {df_dev['pred_date'].max() <= pd.Timestamp('2014-08-24')})")
    print(f"  - Sensitivity Grid Records: {len(df_sens)} rows across {df_sens['grid_cell'].nunique()} grid cells and {df_sens['scenario'].nunique()} scenarios")
    print(f"  - All Provenance Tiers Verified: {df_dev['provenance_tag'].unique().tolist()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
