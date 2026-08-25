"""Build a truthful, reproducible serving registry from the SCMS source data.

This script deliberately keeps the final 365-day holdout out of model fitting,
probability calibration, threshold selection, and conformal calibration.
It replaces the original demo packaging proxy/mocks with real artifacts:
- CatBoost delay classifier
- Isotonic probability calibration curve
- Calibrated decision threshold
- LightGBM q05/q50/q95 positive-delay quantile models
- 90% CQR nonconformity adjustment
- Real-data demo shipment sample
- Holdout serving validation report

Usage:
    python scripts/build_serving_registry.py
    python scripts/build_serving_registry.py --raw-path path/to/SCMS_Delivery_History_Dataset.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from delay_intelligence.data.adapters.scms import SCMSAdapter
from delay_intelligence.features.builder import TemporalFeatureBuilder
from delay_intelligence.validation.contract_validator import PredictionContractValidator

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "artifacts" / "model_registry" / "v2"
EXPECTED_RAW_SHA256 = "918b992dd3e8d4b64d2a727b2c4ea607603d0c58f19484e73f7b78528c6a8673"
EXCLUDE_COLS = {
    "ID",
    "T_pred",
    "Delivered to Client Date",
    "Delivery Recorded Date",
    "Delay_Days",
    "Delay_Flag",
    "is_temporal_anomaly",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_features(raw_path: Path) -> pd.DataFrame:
    adapter = SCMSAdapter(data_path=raw_path)
    raw = adapter.load_raw()
    standardized = adapter.standardize_schema(raw)
    enriched = adapter.extract_temporal_features(standardized)

    validator = PredictionContractValidator()
    eligible = validator.evaluate_base_eligibility(enriched)
    cohort = enriched.loc[eligible].copy()
    cohort["T_pred"] = validator.compute_prediction_timestamp(cohort, use_fallback=False)

    delivered = pd.to_datetime(cohort["Delivered to Client Date"], errors="coerce")
    strict = (
        cohort["T_pred"].notna()
        & (cohort["T_pred"] <= delivered)
        & (cohort["is_temporal_anomaly"] == 0)
    )
    cohort = cohort.loc[strict].copy()
    return TemporalFeatureBuilder().build_features(cohort)


def clean_feature_frame(df: pd.DataFrame, feature_cols: list[str], num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    out = df[feature_cols].copy()
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)
    for c in cat_cols:
        out[c] = (
            out[c]
            .fillna("missing")
            .astype(str)
            .replace({"nan": "missing", "<NA>": "missing", "None": "missing"})
        )
    return out


def best_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.02, 0.81, 0.01):
        pred = (probs >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(t), float(score)
    return best_t, best_f1


def cast_lgb_categories(df: pd.DataFrame, cat_cols: list[str], category_levels: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    for c in cat_cols:
        out[c] = pd.Categorical(out[c].astype(str), categories=category_levels[c])
    return out


def main(raw_path: Path) -> None:
    raw_path = raw_path.resolve()
    actual_hash = sha256_file(raw_path)
    if actual_hash != EXPECTED_RAW_SHA256:
        raise RuntimeError(
            f"Raw SCMS SHA-256 mismatch. Expected {EXPECTED_RAW_SHA256}, got {actual_hash}."
        )

    REGISTRY.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "artifacts" / "demo").mkdir(parents=True, exist_ok=True)

    df = prepare_features(raw_path).sort_values("T_pred").copy()

    # Preserve the historical canonical feature order where possible.
    old_schema_path = REPO_ROOT / "artifacts" / "model_registry" / "v1" / "feature_schema.json"
    old_schema = json.loads(old_schema_path.read_text(encoding="utf-8"))
    feature_cols = [c for c in old_schema["all_features"] if c in df.columns]
    missing = [c for c in df.columns if c not in EXCLUDE_COLS and c not in feature_cols]
    feature_cols.extend(sorted(missing))

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    max_t = pd.to_datetime(df["T_pred"]).max()
    holdout_start = max_t - pd.Timedelta(days=365)
    calib_start = holdout_start - pd.Timedelta(days=180)
    train_end = calib_start - pd.Timedelta(days=90)

    train = df[df["T_pred"] < train_end].copy()
    calibration = df[(df["T_pred"] >= calib_start) & (df["T_pred"] < holdout_start)].copy()
    holdout = df[df["T_pred"] >= holdout_start].copy()

    if min(len(train), len(calibration), len(holdout)) == 0:
        raise RuntimeError("Temporal train/calibration/holdout split produced an empty partition.")

    X_train = clean_feature_frame(train, feature_cols, num_cols, cat_cols)
    X_cal = clean_feature_frame(calibration, feature_cols, num_cols, cat_cols)
    X_hold = clean_feature_frame(holdout, feature_cols, num_cols, cat_cols)

    y_train = train["Delay_Flag"].astype(int).to_numpy()
    y_cal = calibration["Delay_Flag"].astype(int).to_numpy()
    y_hold = holdout["Delay_Flag"].astype(int).to_numpy()

    # 1) Real CatBoost classifier + real isotonic calibrator.
    classifier = cb.CatBoostClassifier(
        random_seed=42,
        auto_class_weights="Balanced",
        iterations=100,
        cat_features=cat_cols,
        verbose=0,
    )
    classifier.fit(X_train, y_train)
    classifier.save_model(str(REGISTRY / "catboost_classifier.cbm"))

    cal_raw = classifier.predict_proba(X_cal)[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(cal_raw, y_cal)
    cal_prob = isotonic.predict(cal_raw)
    threshold, threshold_f1 = best_f1_threshold(y_cal, cal_prob)

    calibration_curve = {
        "method": "isotonic",
        "x_thresholds": [float(x) for x in isotonic.X_thresholds_],
        "y_thresholds": [float(y) for y in isotonic.y_thresholds_],
        "decision_threshold": threshold,
        "calibration_window": {
            "start": str(calibration["T_pred"].min().date()),
            "end": str(calibration["T_pred"].max().date()),
            "rows": int(len(calibration)),
            "f1_at_selected_threshold": threshold_f1,
        },
    }
    (REGISTRY / "probability_calibration.json").write_text(
        json.dumps(calibration_curve, indent=2), encoding="utf-8"
    )

    # 2) Real LightGBM positive-delay quantile models + real 90% CQR adjustment.
    category_levels = {c: sorted(X_train[c].astype(str).unique().tolist()) for c in cat_cols}
    X_train_lgb = cast_lgb_categories(X_train, cat_cols, category_levels)
    X_cal_lgb = cast_lgb_categories(X_cal, cat_cols, category_levels)
    X_hold_lgb = cast_lgb_categories(X_hold, cat_cols, category_levels)

    # Severity is defined conditionally: how many days late, given that a shipment is late.
    # This avoids the zero-mass from on-time shipments overwhelming the quantile model.
    train_delayed = train["Delay_Flag"].astype(int).to_numpy() == 1
    cal_delayed = calibration["Delay_Flag"].astype(int).to_numpy() == 1
    hold_delayed = holdout["Delay_Flag"].astype(int).to_numpy() == 1
    y_train_sev = train.loc[train_delayed, "Delay_Days"].astype(float).to_numpy()
    y_cal_sev = calibration.loc[cal_delayed, "Delay_Days"].astype(float).to_numpy()
    y_hold_sev = holdout.loc[hold_delayed, "Delay_Days"].astype(float).to_numpy()

    X_train_lgb_delayed = X_train_lgb.loc[train_delayed].copy()
    X_cal_lgb_delayed = X_cal_lgb.loc[cal_delayed].copy()
    X_hold_lgb_delayed = X_hold_lgb.loc[hold_delayed].copy()

    quantiles = {"q05": 0.05, "q50": 0.50, "q95": 0.95}
    q_models: dict[str, lgb.LGBMRegressor] = {}
    for name, alpha in quantiles.items():
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            random_state=42,
            n_estimators=50,
            verbose=-1,
        )
        model.fit(X_train_lgb_delayed, y_train_sev, categorical_feature=cat_cols)
        model.booster_.save_model(str(REGISTRY / f"lightgbm_{name}.txt"))
        q_models[name] = model

    q05_cal = q_models["q05"].predict(X_cal_lgb_delayed)
    q95_cal = q_models["q95"].predict(X_cal_lgb_delayed)
    scores = np.maximum(q05_cal - y_cal_sev, y_cal_sev - q95_cal)
    n_cal = len(scores)
    conformal_level = min(1.0, 0.90 * (1.0 + 1.0 / n_cal))
    q_adjustment = float(np.quantile(scores, conformal_level, method="higher"))

    cqr = {
        "method": "split_cqr",
        "target": "Delay_Days_Conditional_On_Delay",
        "nominal_coverage": 0.90,
        "lower_quantile": 0.05,
        "median_quantile": 0.50,
        "upper_quantile": 0.95,
        "finite_sample_quantile_level": conformal_level,
        "q_adjustment": q_adjustment,
        "calibration_rows": int(n_cal),
        "train_delayed_rows": int(train_delayed.sum()),
        "calibration_window": {
            "start": str(calibration["T_pred"].min().date()),
            "end": str(calibration["T_pred"].max().date()),
        },
    }
    (REGISTRY / "cqr_calibration.json").write_text(json.dumps(cqr, indent=2), encoding="utf-8")

    feature_schema = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "all_features": feature_cols,
        "category_levels": category_levels,
    }
    (REGISTRY / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")

    # 3) Holdout-only serving validation. This is not used for fitting/tuning.
    hold_raw = classifier.predict_proba(X_hold)[:, 1]
    hold_prob = np.interp(
        hold_raw,
        np.asarray(calibration_curve["x_thresholds"]),
        np.asarray(calibration_curve["y_thresholds"]),
    )
    hold_pred = (hold_prob >= threshold).astype(int)

    q05 = q_models["q05"].predict(X_hold_lgb_delayed)
    q50 = q_models["q50"].predict(X_hold_lgb_delayed)
    q95 = q_models["q95"].predict(X_hold_lgb_delayed)
    low = np.maximum(0.0, q05 - q_adjustment)
    high = np.maximum(low, q95 + q_adjustment)
    p50 = np.maximum(0.0, q50)
    covered = (y_hold_sev >= low) & (y_hold_sev <= high)

    validation = {
        "evidence_label": "MODEL OUTPUT",
        "evaluation_role": "untouched temporal holdout; never used for fitting or threshold/CQR calibration",
        "data_sha256": actual_hash,
        "splits": {
            "train": {"rows": int(len(train)), "end_exclusive": str(train_end.date())},
            "embargo_days": 90,
            "calibration": {
                "rows": int(len(calibration)),
                "start": str(calib_start.date()),
                "end_exclusive": str(holdout_start.date()),
            },
            "holdout": {
                "rows": int(len(holdout)),
                "start": str(holdout_start.date()),
                "end": str(max_t.date()),
            },
        },
        "classification": {
            "pr_auc": float(average_precision_score(y_hold, hold_prob)),
            "roc_auc": float(roc_auc_score(y_hold, hold_prob)),
            "f1": float(f1_score(y_hold, hold_pred, zero_division=0)),
            "precision": float(precision_score(y_hold, hold_pred, zero_division=0)),
            "recall": float(recall_score(y_hold, hold_pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_hold, hold_pred)),
            "brier_score": float(brier_score_loss(y_hold, hold_prob)),
            "decision_threshold": threshold,
        },
        "severity_cqr": {
            "target": "delay days conditional on an actually delayed shipment",
            "holdout_delayed_rows": int(hold_delayed.sum()),
            "nominal_coverage": 0.90,
            "empirical_coverage_delayed_only": float(covered.mean()),
            "mean_interval_width_delayed_only": float(np.mean(high - low)),
            "median_interval_width_delayed_only": float(np.median(high - low)),
            "q_adjustment": q_adjustment,
            "median_prediction_mean": float(np.mean(p50)),
        },
    }
    (REGISTRY / "serving_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    # 4) Real historical feature rows for the dashboard demo (outcomes retained only for offline display/audit).
    demo_cols = ["ID", "T_pred", "Delay_Flag", "Delay_Days"] + feature_cols
    demo = holdout.sort_values("T_pred").tail(100)[demo_cols].copy()
    demo.to_csv(REPO_ROOT / "artifacts" / "demo" / "demo_shipments.csv", index=False)

    # 5) Registry metadata: no fake git commit. Export hash is added later by closure script.
    metadata = {
        "model_version": "v2.0.0-demo",
        "registry_role": "research-validated decision intelligence prototype serving registry",
        "classifier": "CatBoostClassifier(iterations=100, auto_class_weights=Balanced)",
        "probability_calibration": "IsotonicRegression on temporally later calibration window",
        "severity": "LightGBM quantile regressors q05/q50/q95 for delay days conditional on late shipment",
        "uncertainty": "90% split Conformalized Quantile Regression (CQR)",
        "explainability": "CatBoost native SHAP values at inference time",
        "prediction_contract_version": "v1.0",
        "training_data_version": "SCMS exact-hash source -> point-in-time features",
        "raw_data_sha256": actual_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": None,
        "git_commit_note": "Source archive contained no .git history; fake commit removed. Use source_tree_sha256 from closure_manifest.json.",
        "evidence_labels": ["REAL DATA", "MODEL OUTPUT", "SIMULATED SCENARIO"],
    }
    (REGISTRY / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "SCMS_Delivery_History_Dataset.csv",
    )
    args = parser.parse_args()
    main(args.raw_path)
