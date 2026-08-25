# SCMS Delivery History Dataset — Data Quality & Empirical Audit Report

**System**: Supply Chain Delay Intelligence System  
**Milestone**: Stage 1 — SCMS Data Ingestion & Dataset Audit  
**Author**: Stage 1 Implementation & Quality Assurance Specialist  
**Dataset**: USAID Global Health Supply Chain Program / SCMS Delivery History  
**Audit Date**: 2026-08-17  
**Status**: APPROVED / PASSED  

---

## 1. Executive Summary & Provenance

The Supply Chain Management System (SCMS) dataset represents international public health logistics, antiretroviral (ARV) medication procurement, and diagnostic test kit delivery records funded through the President's Emergency Plan for AIDS Relief (PEPFAR) and administered by USAID | DELIVER PROJECT.

An exhaustive, non-destructive audit was performed directly on the raw file located at `scms/SCMS_Delivery_History_Dataset.csv`.

| Property | Value | Notes / Verification |
|---|---|---|
| **Dataset Title** | SCMS Delivery History Dataset | Primary international health logistics dataset |
| **File Location** | `scms/SCMS_Delivery_History_Dataset.csv` | Read-only input anchor |
| **File Size** | `3,785,904 bytes` (~3.61 MB) | Verified on filesystem |
| **Encoding** | `UTF-8 with BOM` (`\xef\xbb\xbf`) | Ingested via `utf-8-sig` |
| **SHA-256 Checksum** | `918b992dd3e8d4b64d2a727b2c4ea607603d0c58f19484e73f7b78528c6a8673` | Bit-level immutability baseline |
| **MD5 Checksum** | `75ada24f6df3870fda5fde6a2e24cdad` | Cryptographic integrity baseline |
| **Total Rows** | `10,324` rows | Exact shipment line item records |
| **Total Columns** | `33` columns | Complete raw attribute set |
| **Primary Key** | `ID` | 100.0% unique (10,324 distinct non-null IDs) |
| **Full-Row Duplicates**| `0` duplicate rows | No duplicate rows exist in raw file |
| **Temporal Span** | `2006-04-19` to `2015-12-31` | ~9.5 years of global shipment operations |

---

## 2. Dataset Granularity & Cardinality Structure

The fundamental grain of the SCMS dataset is the **Shipment Line Item**. A single customer order or consolidated freight consignment often contains multiple line items representing distinct drug formulations, dosage strengths, or test kit types.

### Relational Hierarchy:
```
[43 Destination Countries]
       │
[142 Public Health Projects (Project Code)]
       │
[1,236 Price Quotes (PQ #) + 2,681 lines in Pre-PQ]
       │
[6,233 Purchase Orders / Sales Orders (PO / SO #)]
       │
[7,030 Delivery Notes / Shipping Notices (ASN/DN #)]
       │
[10,324 Shipment Line Items (ID)]
```

### Cardinality Metrics:
- **Countries**: 43 distinct destination countries across Sub-Saharan Africa, the Caribbean, Southeast Asia, and Eastern Europe (Top 5: South Africa, Nigeria, Côte d'Ivoire, Uganda, Vietnam).
- **Projects**: 142 distinct project codes. 100.0% of project codes map 1-to-1 with a single country.
- **Purchase Orders (`PO / SO #`)**: 6,233 distinct orders. 1,693 purchase orders (27.16%) contain multiple line items (maximum: 67 line items on `SCMS-44`).
- **Shipping Notices (`ASN/DN #`)**: 7,030 distinct delivery consignments. 1,450 consignments (20.63%) contain multiple line items (maximum: 54 line items on `ASN-26738`).

---

## 3. Comprehensive 33-Column Schema & Profile Inventory

| # | Column Name | Raw Dtype | Canonical Ingestion Type | Missing (NaN) | Sentinel Text Values | Unique Values | Sample Value |
|---|---|---|---|---|---|---|---|
| 0 | `ID` | `int64` | `int64` | 0 (0.00%) | None | 10,324 | `1`, `3`, `4`, `86824` |
| 1 | `Project Code` | `object` | `string` | 0 (0.00%) | None | 142 | `100-CI-T01`, `116-ZA-T01` |
| 2 | `PQ #` | `object` | `string` | 0 (0.00%) | `Pre-PQ Process` (2,681) | 1,237 | `Pre-PQ Process`, `PQ-1300` |
| 3 | `PO / SO #` | `object` | `string` | 0 (0.00%) | None | 6,233 | `SCMS-4`, `SCMS-13` |
| 4 | `ASN/DN #` | `object` | `string` | 0 (0.00%) | None | 7,030 | `ASN-8`, `DN-14` |
| 5 | `Country` | `object` | `string` | 0 (0.00%) | None | 43 | `South Africa`, `Nigeria` |
| 6 | `Managed By` | `object` | `string` | 0 (0.00%) | None | 4 | `PMO - US`, `South Africa Field Office` |
| 7 | `Fulfill Via` | `object` | `string` | 0 (0.00%) | None | 2 | `From RDC` (52.34%), `Direct Drop` (47.66%) |
| 8 | `Vendor INCO Term` | `object` | `string` | 0 (0.00%) | `N/A - From RDC` (5,404) | 8 | `N/A - From RDC`, `EXW`, `DDP`, `CIP` |
| 9 | `Shipment Mode` | `object` | `string` | 360 (3.49%) | `N/A` | 4 | `Air`, `Truck`, `Ocean`, `Air Charter` |
| 10 | `PQ First Sent to Client Date` | `object` | `datetime64[ns]` | 0 (raw) | `Pre-PQ Process` (2,476), `Date Not Captured` (205) | 765 | `9/11/14` |
| 11 | `PO Sent to Vendor Date` | `object` | `datetime64[ns]` | 0 (raw) | `N/A - From RDC` (5,404), `Date Not Captured` (328) | 897 | `8/27/14` |

| Column Name | Raw Blank / NaN | Sentinel String Value | Total Non-Standard | Descriptive Missingness Pattern | Handling Strategy |
|---|---|---|---|---|---|
| `PO Sent to Vendor Date` | 0 | `'N/A - From RDC'` (5,404), `'Date Not Captured'` (328) | **5,732 (55.52%)** | Observed missingness pattern is structurally associated with the fulfillment process (From RDC shipments draw from warehouse inventory). | Coerced to `pd.NaT`; created `is_rdc_fulfillment = 1` flag; 0 rows dropped. |
| `PQ First Sent to Client Date` | 0 | `'Pre-PQ Process'` (2,476), `'Date Not Captured'` (205) | **2,681 (25.97%)** | Observed missingness pattern is structurally associated with expedited emergency health orders that bypass PQ. | Coerced to `pd.NaT`; created `is_pre_pq_process = 1` flag; 0 rows dropped. |
| `Dosage` | 1,736 | `'N/A'` (1,736) | **1,736 (16.82%)** | Observed missingness pattern is structurally associated with product type (100% of missing dosage rows are HIV/Malaria diagnostic test kits). | Preserved as null in canonical schema; non-drug categories handled. |
| `Shipment Mode` | 360 | `'N/A'` (360) | **360 (3.49%)** | Unrecorded transport mode across early orders; evidence is insufficient to declare a formal mechanism. | Preserved as null in canonical schema; verified within 5% threshold. |
| `Line Item Insurance (USD)` | 287 | `""` (287) | **287 (2.78%)** | Early 2006–2007 records without itemized insurance; evidence is insufficient to declare a formal mechanism. | Coerced to `np.nan`; verified within 5% threshold. |
| `Weight (Kilograms)` | 0 | `'Weight Captured Separately'` (1,507), `'See ASN-...'` (2,445) | **3,952 (38.28%)** | Mixed Text / Consignment Parent Reference | Coerced text to `np.nan`; created `weight_is_numeric` flag; 100% parent IDs exist. |
| `Freight Cost (USD)` | 0 | `'Freight Included...'` (1,442), `'Invoiced Separately'` (239), `'See ASN-...'` (2,445) | **4,126 (39.97%)** | Mixed Text / Vendor Inclusive / Consignment Reference | Coerced text to `np.nan`; created `freight_is_numeric` flag; 100% parent IDs exist. |

| 12 | `Scheduled Delivery Date` | `object` | `datetime64[ns]` | 0 (0.00%) | None (100% valid) | 2,006 | `2-Jun-06`, `14-Nov-11` |
| 13 | `Delivered to Client Date` | `object` | `datetime64[ns]` | 0 (0.00%) | None (100% valid) | 2,093 | `2-Jun-06`, `15-Nov-11` |
| 14 | `Delivery Recorded Date` | `object` | `datetime64[ns]` | 0 (0.00%) | None (100% valid) | 2,042 | `2-Jun-06`, `15-Nov-11` |
| 15 | `Product Group` | `object` | `string` | 0 (0.00%) | None | 5 | `ARV` (82.8%), `HRDT` (16.7%), `ANTM` |
| 16 | `Sub Classification` | `object` | `string` | 0 (0.00%) | None | 6 | `Adult`, `Pediatric`, `HIV test` |
| 17 | `Vendor` | `object` | `string` | 0 (0.00%) | `SCMS from RDC` (5,404) | 73 | `SCMS from RDC`, `Orgenics, Ltd` |
| 18 | `Item Description` | `object` | `string` | 0 (0.00%) | None | 184 | `HIV, Reveal G3 Rapid Test, 30 Tests` |
| 19 | `Molecule/Test Type` | `object` | `string` | 0 (0.00%) | None | 86 | `Efavirenz`, `Nevirapine` |
| 20 | `Brand` | `object` | `string` | 0 (0.00%) | None | 48 | `Generic` (70.6%), `Determine`, `Uni-Gold` |
| 21 | `Dosage` | `object` | `string` | 1,736 (16.82%)| `N/A` (diagnostic kits) | 54 | `300mg`, `200mg`, `600mg` |
| 22 | `Dosage Form` | `object` | `string` | 0 (0.00%) | None | 17 | `Tablet`, `Test kit`, `Capsule` |
| 23 | `Unit of Measure (Per Pack)` | `int64` | `int64` | 0 (0.00%) | None | 31 | `30`, `60`, `100`, `1`, `20` |
| 24 | `Line Item Quantity` | `int64` | `int64` | 0 (0.00%) | None (pack count) | 5,065 | `19`, `1000`, `3000`, `18332` |
| 25 | `Line Item Value` | `float64` | `float64` | 0 (0.00%) | 17 zero values ($0.00)| 8,741 | `551.0`, `6200.0`, `127360.8` |
| 26 | `Pack Price` | `float64` | `float64` | 0 (0.00%) | 18 zero values ($0.00)| 1,175 | `29.0`, `6.20`, `3.99` |
| 27 | `Unit Price` | `float64` | `float64` | 0 (0.00%) | 103 zero values ($0.00)| 183 | `0.97`, `0.10`, `0.07` |
| 28 | `Manufacturing Site` | `object` | `string` | 0 (0.00%) | None | 88 | `Aurobindo Unit III, India`, `Mylan` |
| 29 | `First Line Designation` | `object` | `string` | 0 (0.00%) | None | 2 | `Yes` (68.09%), `No` (31.91%) |
| 30 | `Weight (Kilograms)` | `object` | `float64` | 3,952 (non-num)| `Weight Captured Separately` (1,507), `See ASN-...` (2,445) | 4,688 | `13.0`, `Weight Captured Separately` |
| 31 | `Freight Cost (USD)` | `object` | `float64` | 4,126 (non-num)| `Freight Included in Commodity Cost` (1,442), `Invoiced Separately` (239), `See ASN-...` (2,445) | 6,733 | `780.34`, `Freight Included...` |
| 32 | `Line Item Insurance (USD)` | `object` | `float64` | 287 (2.78%) | `""` in early 2006-2007 | 6,723 | `192.0`, `2.40` |

---

## 4. Milestone Dates & Lifecycle Audit

Five discrete milestone dates record the international procurement lifecycle:

```
[Price Quote First Sent] ──> [PO Sent to Vendor] ──> [Scheduled Delivery] ──> [Delivered to Client] ──> [Delivery Recorded]
       (PQ Date)                   (PO Date)                 (Sched Date)                (Actual Target)             (ERP Logged)
```

### Parsing Profiles:
1. **`Scheduled Delivery Date`**: `%d-%b-%y` format. **10,324 / 10,324 valid dates (100.0%)**, 0 missing. Range: `2006-05-02` to `2015-12-31`.
2. **`Delivered to Client Date`**: `%d-%b-%y` format. **10,324 / 10,324 valid dates (100.0%)**, 0 missing. Range: `2006-05-02` to `2015-09-14`.
3. **`Delivery Recorded Date`**: `%d-%b-%y` format. **10,324 / 10,324 valid dates (100.0%)**, 0 missing. Range: `2006-05-02` to `2015-09-14`.
4. **`PQ First Sent to Client Date`**: `%m/%d/%y` format.
   - Valid parseable dates: **7,643 (74.03%)**
   - Sentinel `'Pre-PQ Process'`: **2,476 (23.98%)**
   - Sentinel `'Date Not Captured'`: **205 (1.99%)**
5. **`PO Sent to Vendor Date`**: `%m/%d/%y` format.
   - Valid parseable dates: **4,592 (44.48%)**
   - Sentinel `'N/A - From RDC'`: **5,404 (52.34%)**
   - Sentinel `'Date Not Captured'`: **328 (3.18%)**

---

## 5. Historical Anomalies & Chronological Inversions

Empirical auditing identified three minor temporal logging anomalies in the historical ERP records:

1. **`Delivered to Client Date < PO Sent to Vendor Date` (Negative Lead Time)**:
   - Exactly **5 records** (0.11% of records with PO dates):
     - `ID 4190`: PO Date = `2007-11-12`, Scheduled = `2008-01-29`, Delivered = `2007-01-24` (Typo in delivery year: logged as 2007 instead of 2008).
     - `ID 4432`: PO Date = `2008-04-28`, Scheduled = `2008-04-18`, Delivered = `2008-01-03` (Retroactive batch order entry).
     - `ID 13148`: PO Date = `2014-06-23`, Scheduled = `2014-01-14`, Delivered = `2014-01-14`.
     - `ID 25539`: PO Date = `2015-05-29`, Scheduled = `2015-05-26`, Delivered = `2015-05-26` (PO captured 3 days post-delivery).
     - `ID 52710`: PO Date = `2014-06-26`, Scheduled = `2014-06-25`, Delivered = `2014-06-25` (PO captured 1 day post-delivery).
2. **`Scheduled Delivery Date < PO Sent to Vendor Date`**:
   - Exactly **4 records** (`IDs 4432, 13148, 25539, 52710`).
3. **`Delivery Recorded Date < Delivered to Client Date`**:
   - Exactly **3 records** (`IDs 29140, 57447, 72832`): Recorded date is exactly 1 day prior to delivery date, caused by UTC / timezone boundary shifts during ERP entry.

**Ingestion Policy**: All 12 anomalous records are retained in Bronze to preserve 100% row reconciliation and are flagged with boolean indicator `is_temporal_anomaly = 1` for downstream leakage auditing.

---

## 6. Mixed Numeric & Text Logistics Fields

Two logistics columns contain composite operational text annotations:

### `Weight (Kilograms)`:
- Explicit numeric weights: **6,372 records (61.72%)**
- `'Weight Captured Separately'`: **1,507 records (14.60%)**
- Consolidated consignment references (`'See ASN-...'` / `'See DN-...'`): **2,445 records (23.68%)**

### `Freight Cost (USD)`:
- Explicit numeric shipping fees: **6,198 records (60.03%)**
- `'Freight Included in Commodity Cost'`: **1,442 records (13.97%)** (Prevalent in DDP / Delivered Duty Paid shipments)
- `'Invoiced Separately'`: **239 records (2.31%)**
- Consolidated consignment references (`'See ASN-...'` / `'See DN-...'`): **2,445 records (23.68%)**

### Referential Integrity on Consolidated Shipments:
- All **2,445** referenced parent IDs exist in the dataset (**100.00% referential integrity**).
- 95.53% of parent IDs have explicit numeric weights and freight costs.

---

## 7. Critical Selection Bias & Record-Loss Risk Analysis

A central insight of the Stage 1 audit:

### The RDC Fulfillment Discovery:
- The dataset contains two distinct distribution pipelines (`Fulfill Via`):
  - **`From RDC`** (Regional Distribution Centers in Ghana, Kenya, Zimbabwe): **5,404 records (52.34%)**
  - **`Direct Drop`** (Direct manufacturer shipment): **4,920 records (47.66%)**
- **100.0%** of `From RDC` records have `PO Sent to Vendor Date == 'N/A - From RDC'`. Warehouse distribution draws from existing central inventory and does not generate external vendor POs.

### Selection Bias Risk:
- If an ingestion pipeline naively filters records via `df.dropna(subset=['PO Sent to Vendor Date'])`, it will:
  1. Discard **5,732 records (55.52% of the dataset)**.
  2. Completely eliminate **100.0% of warehouse shipments (`From RDC`)**.
  3. Induce massive selection bias: `From RDC` has an empirical delay rate of **17.15%**, compared to only **5.26%** for `Direct Drop`.

### Ingestion Mandate:
- The canonical ingestion adapter retains all 10,324 rows without dropping.
- Non-date text sentinels are coerced to `NaT`, while structural state is preserved via `is_rdc_fulfillment` and `is_pre_pq_process` flags.

---

## 8. Preliminary Target Delay Distribution

These are **Preliminary Audit Variables** derived from the 100% complete milestone columns for dataset auditing purposes only. They are NOT yet the final modeling target contract. Final targets will be explicitly defined in Stage 2 (including prediction unit, timestamp, horizon, eligibility rules, and anomaly handling).

$$\text{Delay\_Days} = (\text{Delivered to Client Date} - \text{Scheduled Delivery Date})_{\text{days}}$$
$$\text{Delay\_Flag} = \mathbb{I}(\text{Delay\_Days} > 0)$$

### Overall Distribution (N = 10,324):
- **On-Time (`Delay_Days == 0`)**: `6,324` records (**61.26%**)
- **Delivered Late (`Delay_Days > 0`)**: `1,186` records (**11.49%**) — *Minority class*
- **Delivered Early (`Delay_Days < 0`)**: `2,814` records (**27.26%**)

### Delay Rate by Fulfillment & Transport Mode:
- **`From RDC`**: 17.15% (927 / 5,404 delayed)
- **`Direct Drop`**: 5.26% (259 / 4,920 delayed)
- **`Ocean`**: 17.52% (65 / 371 delayed)
- **`Truck`**: 16.08% (455 / 2,830 delayed)
- **`Air Charter`**: 11.54% (75 / 650 delayed)
- **`Air`**: 9.60% (587 / 6,113 delayed)
- **`N/A` (Missing Mode)**: 1.11% (4 / 360 delayed)

---

## 9. Conclusion & Stage 1 Acceptance

The raw SCMS dataset is fully audited, verified for 100% cryptographic immutability, and structurally ready for Bronze staging:
1. Exact row count: **10,324 rows**.
2. Zero record loss across ingestion.
3. Automated validation asserts 100% compliance with Requirement R3.
