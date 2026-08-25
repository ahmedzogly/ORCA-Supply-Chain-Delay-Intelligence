"""Tests for SCMS timestamp parsing, sentinel coercions, anomalies, and selection bias guard (Requirement R3)."""

import pandas as pd

from delay_intelligence.data.schema import (
    COL_DELAY_FLAG,
    COL_DELIVERED_TO_CLIENT_DATE,
    COL_DELIVERY_RECORDED_DATE,
    COL_FREIGHT_COST_USD,
    COL_FREIGHT_IS_NUMERIC,
    COL_FULFILL_VIA,
    COL_ID,
    COL_IS_RDC_FULFILLMENT,
    COL_PO_SENT_DATE,
    COL_PQ_FIRST_SENT_DATE,
    COL_SCHEDULED_DELIVERY_DATE,
    COL_WEIGHT_IS_NUMERIC,
    COL_WEIGHT_KG,
    SCMS_EXPECTED_ROW_COUNT,
)


def test_scms_no_full_row_duplicates(scms_raw_df: pd.DataFrame):
    """Verify raw dataset contains zero full-row duplicate records."""
    assert scms_raw_df.duplicated().sum() == 0, "Raw SCMS dataset contains duplicate rows"


def test_scms_core_milestone_dates_100_percent_parsed(scms_canonical_df: pd.DataFrame):
    """Verify core milestone dates are 100% parsed with zero NaT and fall in [2006, 2016]."""
    for col in [
        COL_SCHEDULED_DELIVERY_DATE,
        COL_DELIVERED_TO_CLIENT_DATE,
        COL_DELIVERY_RECORDED_DATE,
    ]:
        nat_count = scms_canonical_df[col].isna().sum()
        assert nat_count == 0, f"Milestone date '{col}' contains {nat_count} NaT values"
        assert scms_canonical_df[col].min() >= pd.Timestamp("2006-01-01")
        assert scms_canonical_df[col].max() <= pd.Timestamp("2016-01-01")


def test_scms_po_sent_sentinels_handling(scms_canonical_df: pd.DataFrame):
    """Verify PO Sent to Vendor Date sentinels coerced to NaT with exact expected counts."""
    valid_po_count = int(scms_canonical_df[COL_PO_SENT_DATE].notna().sum())
    nat_po_count = int(scms_canonical_df[COL_PO_SENT_DATE].isna().sum())

    assert valid_po_count == 4592, f"Expected 4,592 valid PO dates, got {valid_po_count}"
    assert nat_po_count == 5732, f"Expected 5,732 NaT PO dates, got {nat_po_count}"

    # Verify all 5,404 'From RDC' rows have NaT for PO Sent Date
    rdc_mask = scms_canonical_df[COL_FULFILL_VIA] == "From RDC"
    assert rdc_mask.sum() == 5404
    assert scms_canonical_df.loc[rdc_mask, COL_PO_SENT_DATE].isna().all(), (
        "All 'From RDC' shipments must have NaT for PO Sent to Vendor Date"
    )
    assert (scms_canonical_df.loc[rdc_mask, COL_IS_RDC_FULFILLMENT] == 1).all()


def test_scms_pq_sent_sentinels_handling(scms_canonical_df: pd.DataFrame):
    """Verify PQ First Sent to Client Date sentinels coerced to NaT with exact expected counts."""
    valid_pq_count = int(scms_canonical_df[COL_PQ_FIRST_SENT_DATE].notna().sum())
    nat_pq_count = int(scms_canonical_df[COL_PQ_FIRST_SENT_DATE].isna().sum())

    assert valid_pq_count == 7643, f"Expected 7,643 valid PQ dates, got {valid_pq_count}"
    assert nat_pq_count == 2681, f"Expected 2,681 NaT PQ dates, got {nat_pq_count}"


def test_scms_temporal_anomaly_detection(scms_canonical_df: pd.DataFrame):
    """Verify pipeline detects and isolates exact historical temporal inversions."""
    # 1. Delivered < PO Sent (Negative lead time): exactly 5 records
    po_gt_deliv = scms_canonical_df[
        scms_canonical_df[COL_PO_SENT_DATE] > scms_canonical_df[COL_DELIVERED_TO_CLIENT_DATE]
    ]
    assert len(po_gt_deliv) == 5, f"Expected 5 PO > Delivered anomalies, got {len(po_gt_deliv)}"
    assert set(po_gt_deliv[COL_ID].values) == {4190, 4432, 13148, 25539, 52710}

    # 2. Scheduled < PO Sent: exactly 4 records
    po_gt_sched = scms_canonical_df[
        scms_canonical_df[COL_PO_SENT_DATE] > scms_canonical_df[COL_SCHEDULED_DELIVERY_DATE]
    ]
    assert len(po_gt_sched) == 4, f"Expected 4 PO > Scheduled anomalies, got {len(po_gt_sched)}"
    assert set(po_gt_sched[COL_ID].values) == {4432, 13148, 25539, 52710}

    # 3. Recorded < Delivered: exactly 3 records
    rec_lt_deliv = scms_canonical_df[
        scms_canonical_df[COL_DELIVERY_RECORDED_DATE] < scms_canonical_df[COL_DELIVERED_TO_CLIENT_DATE]
    ]
    assert len(rec_lt_deliv) == 3, f"Expected 3 Recorded < Delivered anomalies, got {len(rec_lt_deliv)}"
    assert set(rec_lt_deliv[COL_ID].values) == {29140, 57447, 72832}


def test_scms_delay_distribution_baseline(scms_canonical_df: pd.DataFrame, scms_enriched_df: pd.DataFrame):
    """Verify delay distribution matches audited historical baseline."""
    delay_days = (
        scms_canonical_df[COL_DELIVERED_TO_CLIENT_DATE]
        - scms_canonical_df[COL_SCHEDULED_DELIVERY_DATE]
    ).dt.days

    delayed = (delay_days > 0).sum()
    ontime = (delay_days == 0).sum()
    early = (delay_days < 0).sum()

    assert delayed == 1186, f"Expected 1,186 delayed records, got {delayed}"
    assert ontime == 6324, f"Expected 6,324 on-time records, got {ontime}"
    assert early == 2814, f"Expected 2,814 early records, got {early}"
    assert delayed + ontime + early == SCMS_EXPECTED_ROW_COUNT

    # Enriched DataFrame target consistency
    assert (scms_enriched_df[COL_DELAY_FLAG] == 1).sum() == 1186
    assert (scms_enriched_df[COL_DELAY_FLAG] == 0).sum() == 9138


def test_scms_record_loss_selection_bias_guard(scms_canonical_df: pd.DataFrame):
    """Verify pipeline guards against selection bias: all 10,324 rows retained without dropping."""
    assert len(scms_canonical_df) == SCMS_EXPECTED_ROW_COUNT

    # Validate the risk: naive dropna on PO date would discard 55.52% of dataset
    naive_dropped = scms_canonical_df.dropna(subset=[COL_PO_SENT_DATE])
    assert len(naive_dropped) == 4592
    dropped_ratio = (len(scms_canonical_df) - len(naive_dropped)) / len(scms_canonical_df)
    assert abs(dropped_ratio - 0.5552) < 0.001
    # Naive drop would eliminate 100% of RDC warehouse shipments
    assert (naive_dropped[COL_FULFILL_VIA] == "From RDC").sum() == 0


def test_scms_weight_freight_mixed_value_coercion(scms_canonical_df: pd.DataFrame):
    """Verify mixed numeric text fields parsed to float with indicator flags."""
    numeric_weights = int(scms_canonical_df[COL_WEIGHT_KG].notna().sum())
    numeric_freights = int(scms_canonical_df[COL_FREIGHT_COST_USD].notna().sum())

    assert numeric_weights == 6372, f"Expected 6,372 numeric weights, got {numeric_weights}"
    assert numeric_freights == 6198, f"Expected 6,198 numeric freight values, got {numeric_freights}"

    assert (scms_canonical_df[COL_WEIGHT_IS_NUMERIC] == 1).sum() == 6372
    assert (scms_canonical_df[COL_FREIGHT_IS_NUMERIC] == 1).sum() == 6198
