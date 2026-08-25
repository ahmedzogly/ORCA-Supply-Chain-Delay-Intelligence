"""Comprehensive Empirical Verification Suite for Stage 1 Data Integrity & Invariants.

Executed by Challenger 2 (Data Integrity & Invariant Challenger).
"""

import ast
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from delay_intelligence.core.config import get_data_paths
from delay_intelligence.data.adapters.scms import SCMSAdapter
from delay_intelligence.data.loader import ingest_scms_pipeline
from delay_intelligence.data.schema import (
    COL_DELAY_DAYS,
    COL_DELAY_FLAG,
    COL_DELIVERED_TO_CLIENT_DATE,
    COL_DELIVERY_RECORDED_DATE,
    COL_DOSAGE,
    COL_FREIGHT_COST_USD,
    COL_FULFILL_VIA,
    COL_ID,
    COL_IS_PRE_PQ_PROCESS,
    COL_IS_RDC_FULFILLMENT,
    COL_IS_TEMPORAL_ANOMALY,
    COL_LINE_ITEM_INSURANCE_USD,
    COL_LINE_ITEM_QUANTITY,
    COL_LINE_ITEM_VALUE,
    COL_PACK_PRICE,
    COL_PO_SENT_DATE,
    COL_PO_SENT_IS_DATE,
    COL_PQ_FIRST_SENT_DATE,
    COL_PQ_FIRST_SENT_IS_DATE,
    COL_PQ_NUMBER,
    COL_PROJECT_CODE,
    COL_SCHEDULED_DELIVERY_DATE,
    COL_SCHEDULED_TRANSIT_DAYS,
    COL_SHIPMENT_MODE,
    COL_UNIT_OF_MEASURE,
    COL_UNIT_PRICE,
    COL_WEIGHT_KG,
    SCMS_EXPECTED_SHA256,
)
from delay_intelligence.validation.scms_validator import SCMSValidator


def run_empirical_verification():
    raw_path = Path(r"C:\Users\Admin\Desktop\try1\scms\SCMS_Delivery_History_Dataset.csv")
    repo_root = Path(r"C:\Users\Admin\Desktop\try1\delay_intelligence_system")
    parquet_path = repo_root / "artifacts" / "data" / "bronze_scms.parquet"

    results = {}

    # =========================================================================
    # Check 1 & 2: Raw Hash Immutability & Row Counts Reconciliation
    # =========================================================================
    # Hash before
    h_before = hashlib.sha256()
    with open(raw_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h_before.update(chunk)
    sha256_before = h_before.hexdigest()
    raw_size_bytes = raw_path.stat().st_size

    # Raw CSV counting via csv module
    with open(raw_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        data_rows = list(reader)
        raw_csv_row_count = len(data_rows)
        raw_csv_col_count = len(header)

    # Load via adapter
    adapter = SCMSAdapter(data_path=raw_path)
    df_raw = adapter.load_raw()
    df_std = adapter.standardize_schema(df_raw)
    df_final = adapter.extract_temporal_features(df_std)

    # Ingest pipeline & write Parquet
    df_pipeline, val_report = ingest_scms_pipeline(
        bronze_output_path=parquet_path, save_parquet=True, validate=True
    )
    df_parquet = pd.read_parquet(parquet_path)

    # Hash after
    h_after = hashlib.sha256()
    with open(raw_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h_after.update(chunk)
    sha256_after = h_after.hexdigest()

    results["reconciliation"] = {
        "raw_csv_rows": raw_csv_row_count,
        "raw_csv_cols": raw_csv_col_count,
        "df_raw_rows": len(df_raw),
        "df_raw_cols": len(df_raw.columns),
        "df_std_rows": len(df_std),
        "df_std_cols": len(df_std.columns),
        "df_final_rows": len(df_final),
        "df_final_cols": len(df_final.columns),
        "df_parquet_rows": len(df_parquet),
        "df_parquet_cols": len(df_parquet.columns),
        "exact_10324_reconciled": (
            raw_csv_row_count == len(df_raw) == len(df_std) == len(df_final) == len(df_parquet) == 10324
        ),
        "sha256_before": sha256_before,
        "sha256_after": sha256_after,
        "expected_sha256": SCMS_EXPECTED_SHA256,
        "sha256_matches_expected": sha256_before == SCMS_EXPECTED_SHA256,
        "sha256_immutable": sha256_before == sha256_after,
        "raw_byte_size": raw_size_bytes,
        "parquet_byte_size": parquet_path.stat().st_size,
    }

    # =========================================================================
    # Check 3: Selection Bias Audit (RDC Shipments Retention)
    # =========================================================================
    raw_fulfill_via_counts = df_raw["Fulfill Via"].value_counts().to_dict()
    std_fulfill_via_counts = df_std[COL_FULFILL_VIA].value_counts().to_dict()
    final_fulfill_via_counts = df_final[COL_FULFILL_VIA].value_counts().to_dict()
    parquet_fulfill_via_counts = df_parquet[COL_FULFILL_VIA].value_counts().to_dict()

    rdc_count_raw = int((df_raw["Fulfill Via"] == "From RDC").sum())
    rdc_count_std = int((df_std[COL_FULFILL_VIA] == "From RDC").sum())
    rdc_flag_count = int((df_std[COL_IS_RDC_FULFILLMENT] == 1).sum())
    direct_drop_count = int((df_std[COL_FULFILL_VIA] == "Direct Drop").sum())

    # Missing PO Sent Date breakdown by fulfillment channel
    po_missing_total = int(df_final[COL_PO_SENT_DATE].isna().sum())
    po_missing_rdc = int((df_final[COL_IS_RDC_FULFILLMENT] == 1 & df_final[COL_PO_SENT_DATE].isna()).sum())
    po_missing_direct_drop = int((df_final[COL_IS_RDC_FULFILLMENT] == 0 & df_final[COL_PO_SENT_DATE].isna()).sum())

    # Empirical delay rate by channel
    rdc_mask = df_final[COL_IS_RDC_FULFILLMENT] == 1
    dd_mask = df_final[COL_IS_RDC_FULFILLMENT] == 0

    rdc_delayed_count = int((df_final.loc[rdc_mask, COL_DELAY_FLAG] == 1).sum())
    rdc_total = int(rdc_mask.sum())
    rdc_delay_rate = float(rdc_delayed_count / rdc_total) if rdc_total > 0 else 0.0

    dd_delayed_count = int((df_final.loc[dd_mask, COL_DELAY_FLAG] == 1).sum())
    dd_total = int(dd_mask.sum())
    dd_delay_rate = float(dd_delayed_count / dd_total) if dd_total > 0 else 0.0

    overall_delayed_count = int((df_final[COL_DELAY_FLAG] == 1).sum())
    overall_delay_rate = float(overall_delayed_count / len(df_final))

    results["selection_bias"] = {
        "rdc_count_raw": rdc_count_raw,
        "rdc_count_std": rdc_count_std,
        "rdc_flag_count": rdc_flag_count,
        "rdc_100_percent_retained": rdc_count_raw == 5404 and rdc_count_std == 5404 and rdc_flag_count == 5404,
        "direct_drop_count": direct_drop_count,
        "total_rows": len(df_final),
        "po_missing_total": po_missing_total,
        "po_missing_rdc": po_missing_rdc,
        "po_missing_direct_drop": po_missing_direct_drop,
        "rdc_delayed_count": rdc_delayed_count,
        "rdc_delay_rate": rdc_delay_rate,
        "dd_delayed_count": dd_delayed_count,
        "dd_delay_rate": dd_delay_rate,
        "overall_delayed_count": overall_delayed_count,
        "overall_delay_rate": overall_delay_rate,
        "bias_ratio_rdc_vs_dd": float(rdc_delay_rate / dd_delay_rate) if dd_delay_rate > 0 else 0.0,
    }

    # =========================================================================
    # Check 4: Mathematical Consistency of Delay_Days
    # =========================================================================
    # Independent mathematical computation
    expected_delay_series = (
        df_final[COL_DELIVERED_TO_CLIENT_DATE] - df_final[COL_SCHEDULED_DELIVERY_DATE]
    ).dt.days

    delay_days_diff = (df_final[COL_DELAY_DAYS] - expected_delay_series).abs()
    delay_days_mismatches = int((delay_days_diff != 0).sum())

    # Delay flag consistency
    expected_delay_flag = (df_final[COL_DELAY_DAYS] > 0).astype(int)
    flag_mismatches = int((df_final[COL_DELAY_FLAG] != expected_delay_flag).sum())

    # Delay distributions
    early_count = int((df_final[COL_DELAY_DAYS] < 0).sum())
    ontime_count = int((df_final[COL_DELAY_DAYS] == 0).sum())
    late_count = int((df_final[COL_DELAY_DAYS] > 0).sum())

    results["mathematical_consistency"] = {
        "delay_days_mismatches": delay_days_mismatches,
        "flag_mismatches": flag_mismatches,
        "all_10324_consistent": delay_days_mismatches == 0 and flag_mismatches == 0,
        "early_deliveries_count": early_count,
        "early_percentage": float(early_count / len(df_final)),
        "exact_ontime_count": ontime_count,
        "ontime_percentage": float(ontime_count / len(df_final)),
        "delayed_count": late_count,
        "delayed_percentage": float(late_count / len(df_final)),
        "min_delay_days": int(df_final[COL_DELAY_DAYS].min()),
        "max_delay_days": int(df_final[COL_DELAY_DAYS].max()),
        "median_delay_days_all": float(df_final[COL_DELAY_DAYS].median()),
        "median_delay_days_late_only": float(df_final.loc[df_final[COL_DELAY_DAYS] > 0, COL_DELAY_DAYS].median()),
    }

    # =========================================================================
    # Check 5: Financial Arithmetic Consistency
    # =========================================================================
    # Columns: Unit Price, Pack Price, Unit of Measure (Per Pack), Line Item Quantity, Line Item Value
    uom = df_final[COL_UNIT_OF_MEASURE]
    unit_price = df_final[COL_UNIT_PRICE]
    pack_price = df_final[COL_PACK_PRICE]
    quantity = df_final[COL_LINE_ITEM_QUANTITY]
    item_value = df_final[COL_LINE_ITEM_VALUE]

    # Analysis 1: Unit Price vs Pack Price / UoM
    calculated_unit_price = pack_price / uom
    price_abs_diff = (unit_price - calculated_unit_price).abs()
    # Check within 0.01 tolerance (rounding on 2 decimal places)
    price_close_mask_001 = price_abs_diff <= 0.01
    price_close_mask_005 = price_abs_diff <= 0.05
    exact_price_matches = int((unit_price == calculated_unit_price).sum())
    within_001_matches = int(price_close_mask_001.sum())
    within_005_matches = int(price_close_mask_005.sum())
    discrepancy_count = int((~price_close_mask_001).sum())

    # Discrepancy details
    discrepancies = []
    if discrepancy_count > 0:
        disc_df = df_final[~price_close_mask_001][
            [COL_ID, COL_PRODUCT_GROUP, COL_UNIT_OF_MEASURE, COL_PACK_PRICE, COL_UNIT_PRICE, COL_LINE_ITEM_QUANTITY, COL_LINE_ITEM_VALUE]
        ].copy()
        disc_df["Calculated_Unit_Price"] = calculated_unit_price[~price_close_mask_001]
        disc_df["Abs_Diff"] = price_abs_diff[~price_close_mask_001]
        discrepancies = disc_df.head(10).to_dict(orient="records")

    # Analysis 2: Line Item Value vs Line Item Quantity * Pack Price (or Quantity * Unit Price)
    # In SCMS, does Line Item Quantity mean number of packs or number of units?
    val_via_pack = (quantity * pack_price)
    val_diff_pack = (item_value - val_via_pack).abs()
    val_close_pack_01 = (val_diff_pack <= 1.0)  # within $1 due to rounding

    val_via_unit = (quantity * unit_price)
    val_diff_unit = (item_value - val_via_unit).abs()
    val_close_unit_01 = (val_diff_unit <= 1.0)

    # Let's check how many match pack vs unit
    pack_match_count = int(val_close_pack_01.sum())
    unit_match_count = int(val_close_unit_01.sum())

    results["financial_consistency"] = {
        "unit_price_exact_matches": exact_price_matches,
        "unit_price_within_0.01_matches": within_001_matches,
        "unit_price_within_0.01_ratio": float(within_001_matches / len(df_final)),
        "unit_price_discrepancy_count_gt_0.01": discrepancy_count,
        "sample_discrepancies": discrepancies,
        "value_matches_quantity_times_pack_price_count": pack_match_count,
        "value_matches_quantity_times_pack_price_ratio": float(pack_match_count / len(df_final)),
        "value_matches_quantity_times_unit_price_count": unit_match_count,
        "value_matches_quantity_times_unit_price_ratio": float(unit_match_count / len(df_final)),
        "zero_or_negative_uom_count": int((uom < 1).sum()),
        "negative_unit_price_count": int((unit_price < 0).sum()),
        "negative_pack_price_count": int((pack_price < 0).sum()),
        "negative_quantity_count": int((quantity < 0).sum()),
        "negative_value_count": int((item_value < 0).sum()),
    }

    # =========================================================================
    # Check 6: Codebase Leakage & Downstream Modeling Code Audit
    # =========================================================================
    src_dir = repo_root / "src"
    all_py_files = list(src_dir.rglob("*.py"))

    forbidden_tokens = [
        "sklearn", "lightgbm", "xgboost", "catboost", "torch", "scipy.optimize",
        "statsmodels", "dowhy", "causal_learn", "causal-learn", "nonconformist",
        "mapie", "fastapi", "streamlit", "fit(", ".fit_transform(", "train_test_split",
        "KFold", "cross_val_score", "SMOTE", "RandomForest", "LGBMClassifier",
        "CatBoostClassifier", "LogisticRegression"
    ]

    leakage_findings = []
    file_summaries = {}

    for py_file in all_py_files:
        rel_path = str(py_file.relative_to(repo_root)).replace("\\", "/")
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse AST to check valid python syntax and imports
        tree = ast.parse(content, filename=str(py_file))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        forbidden_hits = []
        for token in forbidden_tokens:
            if token in content:
                # Check if it's in a comment / docstring or active code
                forbidden_hits.append(token)

        file_summaries[rel_path] = {
            "lines": len(content.splitlines()),
            "imports": imports,
            "forbidden_hits": forbidden_hits,
        }

        if forbidden_hits:
            leakage_findings.append({
                "file": rel_path,
                "hits": forbidden_hits,
            })

    results["leakage_audit"] = {
        "scanned_py_files_count": len(all_py_files),
        "files_scanned": list(file_summaries.keys()),
        "leakage_findings": leakage_findings,
        "is_src_clean_of_downstream_leakage": len(leakage_findings) == 0,
    }

    # =========================================================================
    # Overall Validation Summary
    # =========================================================================
    results["validation_report_status"] = val_report.to_dict()

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    run_empirical_verification()
