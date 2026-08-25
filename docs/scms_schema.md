# SCMS Canonical Schema & Data Validation Specification

**System**: Supply Chain Delay Intelligence System  
**Milestone**: Stage 1 — SCMS Data Ingestion, Schema Audit & Validation  
**Document Version**: 1.0.0  
**Author**: Stage 1 Implementation & QA Specialist  
**Status**: APPROVED  

---

## 1. Architectural Principles

The canonical schema layer defines the contract between raw external logistics records and the downstream Medallion pipeline (Bronze $\to$ Silver $\to$ Gold). It adheres to four foundational architectural principles:

1. **Bit-Level Immutability**: Raw CSV files in `scms/` are strictly read-only. Ingestion processes streams without in-place modification.
2. **Zero-Loss Ingestion**: Every raw record ($N = 10,324$) is preserved during ingestion. No rows are dropped due to missing procurement milestones or non-numeric logistics annotations.
3. **Explicit Typing & Sentinel Coercion**: Sentinels (e.g. `'N/A - From RDC'`, `'Pre-PQ Process'`) are cleanly converted to typed nulls (`pd.NaT`, `np.nan`), while semantic meaning is preserved via boolean indicator flags (`is_rdc_fulfillment`, `is_pre_pq_process`).
4. **Leakage-Safe Interface**: Milestone timestamps are partitioned into pre-dispatch features and post-event target anchors (`Delivered to Client Date`, `Delivery Recorded Date`) to prevent temporal data leakage in downstream modeling.

---

## 2. Canonical Column Definitions & Data Dictionary

| # | Canonical Name | Raw CSV Header | Pandas Dtype | Arrow Parquet Dtype | Nullable | Domain / Valid Range | Description |
|---|---|---|---|---|---|---|---|
| 1 | `ID` | `ID` | `int64` | `int64` | No | $[1, 86824]$ | Unique line item primary key (100% unique). |
| 2 | `Project Code` | `Project Code` | `string` | `utf8` | No | 142 distinct codes | Public health project identifier (e.g. `100-CI-T01`). |
| 3 | `PQ #` | `PQ #` | `string` | `utf8` | No | 1,237 values | Price quotation identifier or `'Pre-PQ Process'`. |
| 4 | `PO / SO #` | `PO / SO #` | `string` | `utf8` | No | 6,233 values | Purchase order / Sales order tracking number. |
| 5 | `ASN/DN #` | `ASN/DN #` | `string` | `utf8` | No | 7,030 values | Advanced shipping notice / Delivery note identifier. |
| 6 | `Country` | `Country` | `string` | `utf8` | No | 43 country names | Destination country name. |
| 7 | `Managed By` | `Managed By` | `string` | `utf8` | No | 4 managing offices | Project management field office. |
| 8 | `Fulfill Via` | `Fulfill Via` | `string` | `utf8` | No | `{'From RDC', 'Direct Drop'}` | Logistics fulfillment channel. |
| 9 | `Vendor INCO Term` | `Vendor INCO Term` | `string` | `utf8` | No | `{'EXW', 'CIP', 'DDP', 'FCA', 'DDU', 'CIF', 'N/A - From RDC'}` | International Commercial Terms. |
| 10 | `Shipment Mode` | `Shipment Mode` | `string` | `utf8` | Yes | `{'Air', 'Truck', 'Ocean', 'Air Charter'}` | Freight transport modality (360 nulls). |
| 11 | `PQ First Sent to Client Date` | `PQ First Sent to Client Date` | `datetime64[ns]` | `timestamp[ns]` | Yes | 2006 to 2015 | Date quotation was submitted to recipient. |
| 12 | `PO Sent to Vendor Date` | `PO Sent to Vendor Date` | `datetime64[ns]` | `timestamp[ns]` | Yes | 2006 to 2015 | Date purchase order was issued to manufacturer. |
| 13 | `Scheduled Delivery Date` | `Scheduled Delivery Date` | `datetime64[ns]` | `timestamp[ns]` | No | 2006-05-02 to 2015-12-31 | Contractual delivery target date. |
| 14 | `Delivered to Client Date` | `Delivered to Client Date` | `datetime64[ns]` | `timestamp[ns]` | No | 2006-05-02 to 2015-09-14 | Actual delivery date (**Target Source**). |
| 15 | `Delivery Recorded Date` | `Delivery Recorded Date` | `datetime64[ns]` | `timestamp[ns]` | No | 2006-05-02 to 2015-09-14 | ERP logging date (**Post-Event**). |
| 16 | `Product Group` | `Product Group` | `string` | `utf8` | No | `{'ARV', 'HRDT', 'ANTM', 'ACT', 'MRDT'}` | High-level commodity category. |
| 17 | `Sub Classification` | `Sub Classification` | `string` | `utf8` | No | 6 sub-classes | Product sub-type (`Adult`, `Pediatric`, `HIV test`). |
| 18 | `Vendor` | `Vendor` | `string` | `utf8` | No | 73 vendors | Supplier or procurement agency name. |
| 19 | `Item Description` | `Item Description` | `string` | `utf8` | No | 184 descriptions | Full pharmaceutical formulation text. |
| 20 | `Molecule/Test Type` | `Molecule/Test Type` | `string` | `utf8` | No | 86 molecules | Active generic molecule or diagnostic assay. |
| 21 | `Brand` | `Brand` | `string` | `utf8` | No | 48 brand names | Commercial brand name or `'Generic'`. |
| 22 | `Dosage` | `Dosage` | `string` | `utf8` | Yes | 54 dosage specs | Strength specification (1,736 nulls for test kits). |
| 23 | `Dosage Form` | `Dosage Form` | `string` | `utf8` | No | 17 dosage forms | Physical administration form. |
| 24 | `Unit of Measure (Per Pack)` | `Unit of Measure (Per Pack)` | `int64` | `int64` | No | $[1, 1000]$ | Pack size / unit count per container. |
| 25 | `Line Item Quantity` | `Line Item Quantity` | `int64` | `int64` | No | $[1, 6199992]$ | Total pack units ordered. |
| 26 | `Line Item Value` | `Line Item Value` | `float64` | `double` | No | $[\$0.00, \$29,501,980.00]$ | Total commercial commodity value in USD. |
| 27 | `Pack Price` | `Pack Price` | `float64` | `double` | No | $[\$0.00, \$7250.00]$ | Contract purchase price per pack in USD. |
| 28 | `Unit Price` | `Unit Price` | `float64` | `double` | No | $[\$0.00, \$1250.00]$ | Price per single individual dosage unit in USD. |
| 29 | `Manufacturing Site` | `Manufacturing Site` | `string` | `utf8` | No | 88 manufacturing sites | Facility where pharmaceutical batch was produced. |
| 30 | `First Line Designation` | `First Line Designation` | `string` | `utf8` | No | `{'Yes', 'No'}` | WHO first-line clinical regimen designation. |
| 31 | `Weight (Kilograms)` | `Weight (Kilograms)` | `float64` | `double` | Yes | $[0.0, 896064.0]$ | Physical consignment gross weight in kg. |
| 32 | `Freight Cost (USD)` | `Freight Cost (USD)` | `float64` | `double` | Yes | $[\$0.00, \$2115127.30]$ | Total transport logistics cost in USD. |
| 33 | `Line Item Insurance (USD)` | `Line Item Insurance (USD)` | `float64` | `double` | Yes | $[\$0.00, \$50893.17]$ | Cargo insurance fee in USD (287 nulls). |

---

## 3. Derived Quality & Temporal Indicator Features

During ingestion, the canonical adapter augments the schema with standardized indicator flags to preserve structural information:

| Column Name | Type | Description |
|---|---|---|
| `Delay_Flag` | `int64` | **Preliminary Audit Variable** (Not final modeling target). Binary delay flag for audit: $1$ if $\text{Delivered} > \text{Scheduled}$, else $0$. |
| `Delay_Days` | `int64` | **Preliminary Audit Variable** (Not final modeling target). Continuous delay duration for audit: $(\text{Delivered} - \text{Scheduled})_{\text{days}}$. |
| `Scheduled_Transit_Days` | `Float64` | Scheduled lead time: $(\text{Scheduled} - \text{PO Sent})_{\text{days}}$. |
| `is_rdc_fulfillment` | `int64` | $1$ if fulfilled from Regional Distribution Center warehouse, $0$ if Direct Drop. |
| `is_pre_pq_process` | `int64` | $1$ if line item was expedited before formal price quotation, $0$ otherwise. |
| `weight_is_numeric` | `int64` | $1$ if direct numeric weight was available, $0$ if text note/consolidated reference. |
| `freight_is_numeric` | `int64` | $1$ if direct numeric freight was available, $0$ if text note/consolidated reference. |
| `is_temporal_anomaly` | `int64` | $1$ if record exhibits audited historical ERP timestamp inversion, $0$ otherwise. |

---

## 4. Sentinel Value Handling Standard

| Column | Raw Sentinel String | Canonical Coercion | Accompanying Indicator Flag |
|---|---|---|---|
| `PO Sent to Vendor Date` | `'N/A - From RDC'` | `pd.NaT` | `is_rdc_fulfillment = 1` |
| `PO Sent to Vendor Date` | `'Date Not Captured'` | `pd.NaT` | `po_sent_is_date = 0` |
| `PQ First Sent to Client Date` | `'Pre-PQ Process'` | `pd.NaT` | `is_pre_pq_process = 1` |
| `PQ First Sent to Client Date` | `'Date Not Captured'` | `pd.NaT` | `pq_first_sent_is_date = 0` |
| `Weight (Kilograms)` | `'Weight Captured Separately'` | `np.nan` | `weight_is_numeric = 0` |
| `Weight (Kilograms)` | `'See ASN-... (ID#:...)'` | `np.nan` | `weight_is_numeric = 0` |
| `Freight Cost (USD)` | `'Freight Included in Commodity Cost'` | `np.nan` | `freight_is_numeric = 0` |
| `Freight Cost (USD)` | `'Invoiced Separately'` | `np.nan` | `freight_is_numeric = 0` |
| `Freight Cost (USD)` | `'See ASN-... (ID#:...)'` | `np.nan` | `freight_is_numeric = 0` |

---

## 5. Automated Data Quality Gates & Constraints

The `SCMSValidator` validates every ingested batch against strict quality rules:

1. **Row-Count Gate**: Length must reconcile exactly with baseline ($10,324$ records).
2. **Primary Key Gate**: `ID` must be $100\%$ non-null and $100\%$ unique.
3. **Critical Zero-Null Gate**: Critical columns (`ID`, `Project Code`, `Country`, `Scheduled Delivery Date`, `Delivered to Client Date`, `Delivery Recorded Date`, `Line Item Quantity`, `Line Item Value`) must contain $0$ null values.
4. **Tolerance Bounds Gate**:
   - `Shipment Mode`: Null rate $\le 5.0\%$ (Actual: $3.49\%$).
   - `Dosage`: Null rate $\le 40.0\%$ (Actual: $16.82\%$).
   - `Line Item Insurance (USD)`: Null rate $\le 5.0\%$ (Actual: $2.78\%$).
   - `Weight (Kilograms)`: Non-numeric rate $\le 40.0\%$ (Actual: $38.28\%$).
   - `Freight Cost (USD)`: Non-numeric rate $\le 40.0\%$ (Actual: $39.97\%$).
5. **Positivity Gate**: Quantity $\ge 1$, Prices and Values $\ge 0.0$.
6. **Categorical Domain Gate**: Discrete columns must strictly belong to audited domain sets.
7. **Temporal Bound Gate**: Delivery milestone dates must be non-null and fall within `2006-01-01` to `2016-01-01`.
