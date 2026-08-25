"""
Script to re-compute SHA-256 hashes of all 36 frozen baseline artifacts post-holdout evaluation
and verify 100% bitwise invariance against artifacts/phase2/e10/e10_pre_freeze_manifest.json.

Outputs artifacts/phase2/e10/e10_post_holdout_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Complete list of 36 frozen baseline artifacts across Stages 0-13, E6.5, E7, E8, E9
BASELINE_FILES = [
    # Model Registry
    "artifacts/model_registry/v1/catboost_champion.cbm",
    "artifacts/model_registry/v1/cqr_calibration.json",
    "artifacts/model_registry/v1/feature_schema.json",
    "artifacts/model_registry/v1/metadata.json",
    "artifacts/model_registry/v1/decision.yaml",
    "artifacts/model_registry/v1/explainability.yaml",
    "artifacts/model_registry/v1/causal.yaml",
    
    # Datasets
    "artifacts/data/bronze_scms.parquet",
    "artifacts/data/scms_modeling_features.parquet",
    
    # Baseline Evaluation
    "artifacts/final/final_holdout_metrics.json",
    "artifacts/evaluation/fold_manifest.csv",
    "artifacts/evaluation/stage5_metrics.csv",
    "artifacts/evaluation/stage6_uncertainty_metrics.csv",
    
    # E6.5 Drift
    "artifacts/drift/cv_drift_summary.json",
    "artifacts/drift/drift_triggers.json",
    "artifacts/drift/drift_metrics.csv",
    "artifacts/drift/feature_drift_summary.csv",
    
    # E7 Adaptive Conformal
    "artifacts/adaptive_conformal/cv_adaptive_comparison.json",
    "artifacts/adaptive_conformal/holdout_adaptive_comparison.json",
    "artifacts/adaptive_conformal/adaptive_efficiency_summary.csv",
    "artifacts/adaptive_conformal/holdout_recalibration_events.json",
    
    # E8 Cost-Sensitive
    "artifacts/results/e8_frozen_policy.json",
    "artifacts/results/e8_final_holdout_results.parquet",
    "artifacts/results/e8_final_holdout_metrics.json",
    "artifacts/results/e8_dev_backtest_results.parquet",
    "artifacts/results/e8_dev_metrics.json",
    "artifacts/results/e8_dev_budget_results.json",
    "artifacts/results/e8_dev_sensitivity_results.json",
    
    # E9 Digital Twin & Stress Testing
    "artifacts/phase2/e9/e9_immutability_manifest.json",
    "artifacts/phase2/e9/e9_scenario_results.csv",
    "artifacts/phase2/e9/e9_multi_shipment_stress.csv",
    
    # Frozen Configs & Specifications
    "configs/prediction_contract.yaml",
    "configs/cost_scenarios.yaml",
    "configs/e8_experiments.yaml",
    "docs/e9_simulation_assumptions.json",
    "docs/e9_feature_contract.json",
]


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("=" * 90)
    print(" USAID SCMS Delay Intelligence — Experiment E10")
    print(" Milestone 5: Post-Holdout Baseline Artifact SHA-256 Invariance Audit")
    print("=" * 90)

    pre_freeze_manifest_path = PROJECT_ROOT / "artifacts/phase2/e10/e10_pre_freeze_manifest.json"
    if not pre_freeze_manifest_path.exists():
        raise FileNotFoundError(f"Pre-freeze manifest not found at: {pre_freeze_manifest_path}")

    with open(pre_freeze_manifest_path, "r", encoding="utf-8") as f:
        pre_freeze_data = json.load(f)

    pre_freeze_artifacts = pre_freeze_data.get("artifacts", {})
    post_manifest_entries = {}
    mismatches = []
    missing_files = []

    print(f"\nAuditing {len(BASELINE_FILES)} frozen baseline artifacts...")
    print("-" * 90)
    print(f"{'Artifact Path':<60} {'SHA-256 (Prefix)':<16} {'Status':<10}")
    print("-" * 90)

    for rel_path in BASELINE_FILES:
        norm_path = rel_path.replace("\\", "/")
        full_path = PROJECT_ROOT / norm_path

        if not full_path.exists():
            print(f"MISSING: {norm_path}")
            missing_files.append(norm_path)
            continue

        file_sha256 = compute_sha256(full_path)
        file_size = full_path.stat().st_size

        post_manifest_entries[norm_path] = {
            "sha256": file_sha256,
            "size_bytes": file_size,
        }

        # Compare with pre-freeze hash
        expected_sha = pre_freeze_artifacts.get(norm_path, {}).get("sha256")
        if expected_sha is None:
            status = "NEW/UNTRACKED"
            mismatches.append((norm_path, "NOT_IN_PRE_FREEZE", file_sha256))
        elif expected_sha != file_sha256:
            status = "MISMATCH"
            mismatches.append((norm_path, expected_sha, file_sha256))
        else:
            status = "MATCH (100%)"

        print(f"{norm_path:<60} {file_sha256[:12]}...    {status}")

    print("-" * 90)

    if missing_files:
        raise FileNotFoundError(f"Missing required baseline files ({len(missing_files)}): {missing_files}")

    if mismatches:
        print("\nFATAL ERROR: Baseline invariance audit FAILED with hash mismatches:")
        for path, exp, act in mismatches:
            print(f"  - {path}: expected {exp}, got {act}")
        sys.exit(1)

    print(f"\nALL {len(post_manifest_entries)} BASELINE ARTIFACTS CONFIRMED 100% BITWISE INVARIANT!")

    # Structure post-holdout manifest
    post_holdout_manifest = {
        "metadata": {
            "manifest_name": "E10 Post-Holdout Baseline Invariance Manifest",
            "phase": "Phase 2 — Experiment E10 (Counterfactual Policy Evaluation)",
            "milestone": "Milestone 5 (Single-Pass Final Holdout Evaluation & Invariance Verification)",
            "created_at_utc": "2026-08-22T16:38:00Z",
            "status": "VERIFIED_100_PERCENT_BITWISE_INVARIANT",
            "total_artifacts_verified": len(post_manifest_entries),
            "pre_freeze_manifest_ref": "artifacts/phase2/e10/e10_pre_freeze_manifest.json",
            "verification_result": "PASS",
            "verification_note": "All 36 baseline artifacts from Stages 0-13, E6.5, E7, E8, and E9 verified 100% bitwise invariant post single-pass holdout evaluation."
        },
        "artifacts": post_manifest_entries
    }

    out_dir = PROJECT_ROOT / "artifacts/phase2/e10"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "e10_post_holdout_manifest.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(post_holdout_manifest, f, indent=2)

    print(f"Post-holdout manifest successfully written to: {out_file}")
    print("=" * 90)


if __name__ == "__main__":
    main()
