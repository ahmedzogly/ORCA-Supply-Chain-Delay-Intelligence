"""
Runner script for Milestone 5 Single-Pass Final Holdout Counterfactual Policy Evaluation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import pandas as pd

from delay_intelligence.counterfactual.evaluator import CounterfactualEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    print("=" * 100)
    print(" USAID SCMS Delay Intelligence — Experiment E10")
    print(" Milestone 5: Single-Pass Final Holdout Counterfactual Policy Evaluation")
    print("=" * 100)

    evaluator = CounterfactualEvaluator()
    summary = evaluator.run_holdout_evaluation()

    print(f"\nStatus: {summary['status']}")
    print(f"Holdout Cohort Sample Size: {summary['holdout_sample_size']}")
    print(f"Holdout Date Range: {summary['min_pred_date']} to {summary['max_pred_date']}")
    print(f"Artifacts Generated:")
    for path in summary['artifacts_generated']:
        print(f"  - {path}")

    for sc_name in ["low", "base", "high"]:
        print(f"\n{sc_name.upper()} Scenario Policy Performance (N=1,013 Final Holdout):")
        print("-" * 115)
        print(f"{'Policy ID':<10} {'Policy Name':<28} {'Mean Cost ($)':<16} {'Net Benefit ($)':<16} {'Oracle Gap ($)':<16} {'Mean Regret ($)':<16} {'Interv %':<10}")
        print("-" * 115)
        for pol_id, m in summary['policy_metrics_by_scenario'][sc_name].items():
            print(f"{pol_id:<10} {m['policy_name']:<28} ${m['mean_expected_cost_usd']:<15.2f} ${m['mean_net_benefit_usd']:<15.2f} ${m['oracle_gap_usd']:<15.2f} ${m['mean_regret_usd']:<15.2f} {m['intervention_rate_pct']:<9.1f}%")
        print("-" * 115)

    print("\nOperational Review Budget Allocation (N=1,013 Final Holdout):")
    print("-" * 100)
    print(f"{'Scenario':<10} {'Capacity K':<12} {'Allocated':<12} {'Limit':<10} {'Total Benefit ($)':<22} {'Utilization (%)':<15}")
    print("-" * 100)
    for sc_name, sc_dict in summary['budget_summary'].items():
        for k_str, b in sc_dict.items():
            print(f"{sc_name:<10} {k_str:<12} {b['allocated_count']:<12} {b['capacity_limit_count']:<10} ${b['total_net_benefit_usd']:<21.2f} {b['utilization_pct']:<14.1f}%")
    print("-" * 100)

    print("\nPolicy Switching Rate Across Cost Scenarios (Holdout N=1,013):")
    print("-" * 80)
    print(f"{'Policy ID':<12} {'Low -> Base (%)':<20} {'Base -> High (%)':<20} {'Low -> High (%)':<20}")
    print("-" * 80)
    for pol_id, sw in summary['switching_analysis'].items():
        print(f"{pol_id:<12} {sw['switching_rate_low_to_base_pct']:<19.1f}% {sw['switching_rate_base_to_high_pct']:<19.1f}% {sw['switching_rate_low_to_high_pct']:<19.1f}%")
    print("-" * 80)

    # Save structured metrics summary JSON
    metrics_path = Path("artifacts/phase2/e10/e10_holdout_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved holdout metrics summary JSON to: {metrics_path}")

    # Validate generated artifact
    df_holdout = pd.read_parquet(summary['artifacts_generated'][0])
    print(f"\nVerification:")
    print(f"  - Holdout Parquet Total Records: {len(df_holdout)} rows (Expected 1013 * 7 policies * 3 scenarios = {1013*7*3})")
    print(f"  - Min Prediction Date: {df_holdout['pred_date'].min()} (Post-cutoff > 2014-08-24 Verified: {df_holdout['pred_date'].min() > pd.Timestamp('2014-08-24')})")
    print(f"  - Max Prediction Date: {df_holdout['pred_date'].max()}")
    print(f"  - Scenarios: {df_holdout['scenario'].unique().tolist()}")
    print(f"  - Policies: {df_holdout['policy_id'].unique().tolist()}")
    print(f"  - Provenance Tags: {df_holdout['provenance_tag'].unique().tolist()}")
    print("=" * 100)


if __name__ == "__main__":
    main()
