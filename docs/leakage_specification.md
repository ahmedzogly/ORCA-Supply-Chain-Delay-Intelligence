# Supply Chain Delay Intelligence System
# Feature-Level Leakage Specification & Temporal Audit

**System**: Supply Chain Delay Intelligence System  
**Milestone**: Stage 2 — Prediction Contract, Target Definition & Leakage Specification  
**Document**: Leakage Specification (`docs/leakage_specification.md`)  
**Status**: AUTHORITATIVE SPECIFICATION & AUDIT RECORD  
**Dataset Reference**: USAID / SCMS Delivery History Dataset (10,324 records, 33 raw attributes)  

---

## 1. Executive Summary

In predictive supply chain machine learning, **temporal data leakage (lookahead bias)** and **post-outcome contamination** represent the single greatest threat to model reliability. When a model inadvertently accesses information generated during or after transit (e.g., actual carrier freight surcharges, actual gross scale weights, delivery confirmation timestamps), offline evaluation metrics become deceptively inflated while live production performance collapses.

To guarantee zero leakage:
1. **Exhaustive 44-Feature Audit**: Every single one of the **33 raw SCMS attributes** and **11 Stage 1 derived indicators** is audited and assigned a strict classification: `Allowed`, `Forbidden`, `Post-outcome`, `Target-derived`, or `Ambiguous`.
2. **Lifecycle Discretization**: Features are mapped to their true operational generation events ($M_0$ Program Inception $\to$ $M_1$ Price Quotation $\to$ $M_2$ Order Commitment $\to$ $M_3$ Consignment $\to$ $M_4$ Delivery $\to$ $M_5$ ERP Log).
3. **Problematic Field Hardening**: Deep-dive operational audits resolve boundary risks for `Delivery Recorded Date`, `Delivered to Client Date`, `Freight Cost (USD)`, `Weight (Kilograms)`, `Line Item Insurance (USD)`, `Shipment Mode`, and `Vendor INCO Term`.
4. **Mathematical Boundary Invariants**: Testable mathematical assertions protect training, inference, and temporal cross-validation.

---

## 2. Complete Feature-Level Leakage Audit Table

### 2.1 Raw SCMS Attributes (33 Columns)

| # | Attribute Name | Operational Source Event | Available at $T_{\text{pred}}$ ($M_2$)? | Leakage Classification | Explicit Technical & Business Rationale |
|---|---|---|---|---|---|
| 1 | `ID` | $M_0$ / System Ingestion | Yes (Database Key) | **Forbidden** (as ML feature) / **Allowed** (as Entity Key) | System surrogate primary key (integer $[1, 86824]$). Has no real-world physical meaning; feeding it as a predictor introduces artificial memorization and row-order leakage. |
| 2 | `Project Code` | $M_0$ (Project Inception) | Yes (100% complete) | **Allowed** | Public health project identifier (142 codes, e.g. `100-CI-T01`). Fully known at order requisition; encapsulates long-term country health program parameters. |
| 3 | `PQ #` | $M_1$ (Price Quotation) | Yes (100% complete) | **Allowed** (Categorical / Metadata) | Price quotation tracking number or sentinel `'Pre-PQ Process'` (23.98%). Fully established prior to order commitment. |
| 4 | `PO / SO #` | $M_2$ (Order Commitment) | Yes (100% complete) | **Allowed** (Entity / Group Key) | Purchase order tracking identifier (6,233 distinct orders). Useful for clustered multi-item cross-validation splits; forbidden as raw high-cardinality nominal predictor. |
| 5 | `ASN/DN #` | $M_3$ (Consignment Dispatch) | **No** (Generated post-PO) | **Forbidden** (at $M_2$ baseline) | Advanced Shipping Notice / Delivery Note tracking number (7,030 values). Generated upon physical batch dispatch and carrier tendering, weeks/months after order commitment. |
| 6 | `Country` | $M_0$ (Project Destination) | Yes (100% complete) | **Allowed** | Destination sovereign country (43 countries). Key determinant of customs complexity, port congestion, and overland transit routes. Known at order inception. |
| 7 | `Managed By` | $M_0$ (Field Office) | Yes (100% complete) | **Allowed** | Managing PMO / field office entity (4 offices: PMO-US, South Africa, Haiti, Ethiopia). Known at order inception. |
| 8 | `Fulfill Via` | $M_1$ / $M_2$ (Sourcing Decision) | Yes (100% complete) | **Allowed** | Supply chain routing topology (`From RDC` 52.34%, `Direct Drop` 47.66%). Decided during procurement planning before PO placement. |
| 9 | `Vendor INCO Term` | $M_2$ (Contractual Terms) | Yes (100% complete) | **Allowed** | Commercial terms (EXW, DDP, FCA, CIP, DDU, DAP, CIF, N/A - From RDC). Dictates freight responsibility and transfer of risk. Finalized in PO contract. |
| 10 | `Shipment Mode` | $M_2$ (Planned Logistics Routing) | Yes (96.51% valid, 3.49% null) | **Allowed** (Baseline Planned Mode) | Transport modality (Air, Truck, Ocean, Air Charter). Represents planned routing modality established during procurement requisition. |
| 11 | `PQ First Sent to Client Date` | $M_1$ (Quotation Submission) | Yes (74.03% valid dates) | **Allowed** (Historical Milestone) | Timestamp when quotation was submitted. Strictly precedes PO and delivery; used to derive quotation-to-order preparation lead time. |
| 12 | `PO Sent to Vendor Date` | $M_2$ (Order Placement) | Yes (44.48% valid, 52.34% RDC) | **Allowed** (Prediction Anchor / Feature) | Timestamp purchase order was sent to supplier. Serves as the primary Direct Drop prediction anchor ($T_{\text{pred}}$). |
| 13 | `Scheduled Delivery Date` | $M_2$ (Contractual Target) | Yes (100.00% complete) | **Allowed** (Horizon Benchmark) | Contractually committed delivery deadline agreed with recipient health program. Used to calculate planned lead time ($\text{Scheduled} - T_{\text{pred}}$). |
| 14 | `Delivered to Client Date` | $M_4$ (Physical Client Delivery) | **No** (Target Milestone) | **Target-derived** / **Forbidden** | Physical arrival timestamp at client warehouse. Constitutes the ground truth outcome ($T_{\text{outcome}}$) for computing delay targets. Inclusion in feature space is 100% target leakage. |
| 15 | `Delivery Recorded Date` | $M_5$ (Post-Delivery ERP Log) | **No** (Post-Outcome Administrative) | **Post-outcome** / **Forbidden** | Administrative ERP entry timestamp logged after delivery confirmation. Occurs at or after physical delivery ($T_{\text{Record}} \ge T_{\text{Deliv}}$). Inclusion in feature space is 100% post-outcome leakage. |
| 16 | `Product Group` | $M_1$ (Commodity Specification) | Yes (100% complete) | **Allowed** | High-level commodity category (ARV 82.8%, HRDT 16.7%, ANTM, ACT, MRDT). Known at requisition. |
| 17 | `Sub Classification` | $M_1$ (Commodity Specification) | Yes (100% complete) | **Allowed** | Product sub-type (Adult, Pediatric, HIV test). Known at requisition. |
| 18 | `Vendor` | $M_2$ (Supplier Contracting) | Yes (100% complete) | **Allowed** | Supplier or procurement agency contracted (73 distinct vendors). Known upon PO issuance. |
| 19 | `Item Description` | $M_1$ (Formulation Details) | Yes (100% complete) | **Allowed** (Text / Embedding) | Detailed clinical pharmaceutical specification (184 descriptions). Known at requisition. |
| 20 | `Molecule/Test Type` | $M_1$ (Active Chemical Molecule) | Yes (100% complete) | **Allowed** | Active pharmaceutical ingredient or test assay type (86 molecules). Known at requisition. |
| 21 | `Brand` | $M_1$ (Brand / Generic Type) | Yes (100% complete) | **Allowed** | Commercial brand name or generic indicator (48 brands, Generic = 70.6%). Known at requisition. |
| 22 | `Dosage` | $M_1$ (Dosage Strength) | Yes (83.18% valid, 16.82% N/A) | **Allowed** | Pharmaceutical strength (e.g. `300mg`). Structural nulls correspond 100% to diagnostic test kits. Known at requisition. |
| 23 | `Dosage Form` | $M_1$ (Physical Presentation) | Yes (100% complete) | **Allowed** | Physical form (Tablet, Capsule, Test kit, Oral solution). Known at requisition. |
| 24 | `Unit of Measure (Per Pack)` | $M_1$ (Packaging Spec) | Yes (100% complete) | **Allowed** | Number of units per container (e.g. 30, 60 pills). Known at requisition. |
| 25 | `Line Item Quantity` | $M_1$ / $M_2$ (Order Volume) | Yes (100% complete) | **Allowed** | Total pack units ordered. Primary metric of procurement volume and manufacturing batch scale. |
| 26 | `Line Item Value` | $M_1$ / $M_2$ (Financial Value) | Yes (100% complete) | **Allowed** | Total line item commodity value in USD. Finalized upon PO placement. |
| 27 | `Pack Price` | $M_1$ / $M_2$ (Contract Unit Price) | Yes (100% complete) | **Allowed** | Contract purchase price per pack in USD. Finalized upon PO placement. |
| 28 | `Unit Price` | $M_1$ / $M_2$ (Unit Dosage Price) | Yes (100% complete) | **Allowed** | Price per individual pill/assay in USD. Finalized upon PO placement. |
| 29 | `Manufacturing Site` | $M_2$ (Factory Allocation) | Yes (100% complete) | **Allowed** | Specific pharmaceutical production facility (88 sites, e.g. Aurobindo, Mylan, Cipla). Known at PO placement. |
| 30 | `First Line Designation` | $M_1$ (Clinical Protocol) | Yes (100% complete) | **Allowed** | WHO first-line clinical regimen designation (Yes 68.09%, No 31.91%). Static clinical priority. |
| 31 | `Weight (Kilograms)` | $M_3$ (Consignment Weigh-in) | **No / Ambiguous** (Actual Weighed) | **Ambiguous** / **Forbidden** (as raw actual at $M_2$) | Raw field mixes post-dispatch physical scale measurements (61.72%), consolidated ASN references (23.68%), and unmeasured cargo (14.60%). Actual weighed gross weight is measured at warehouse packing ($M_3$). Using raw actual weight at PO placement introduces dispatch-stage leakage. Pre-order theoretical catalog weight must be used instead. |
| 32 | `Freight Cost (USD)` | $M_3$ / $M_5$ (Carrier Invoicing) | **No / Ambiguous** (Final Invoiced) | **Ambiguous** / **Forbidden** (as raw actual at $M_2$) | Raw field captures final carrier freight invoices (60.03%), DDP inclusive costs (13.97%), and ASN consolidations (23.68%). Final freight invoices incorporate carrier demurrage, detention, and expedited air surcharges incurred *during* transport. Using raw actual freight cost introduces post-dispatch and delay-consequential leakage. |
| 33 | `Line Item Insurance (USD)` | $M_2$ (Contractual Premium) | Yes (97.22% valid, 2.78% legacy null) | **Allowed** | Contractual cargo insurance calculated as a fixed percentage (~0.10%–0.22%) of `Line Item Value`. Established upon PO valuation. (Note: Highly collinear with `Line Item Value`). |

---

### 2.2 Stage 1 Derived and Quality Indicator Attributes (11 Columns)

| # | Indicator Attribute Name | Operational Source Event | Available at $T_{\text{pred}}$? | Leakage Classification | Explicit Technical & Business Rationale |
|---|---|---|---|---|---|
| 34 | `Delay_Flag` | $M_4$ (Delivery Evaluation) | **No** (Ground Truth Target) | **Target-derived** / **Forbidden** | Binary classification target: $\mathbb{I}(\text{Delivered} > \text{Scheduled})$. Strictly forbidden in feature space. |
| 35 | `Delay_Days` | $M_4$ (Delivery Evaluation) | **No** (Ground Truth Target) | **Target-derived** / **Forbidden** | Continuous regression target: $(\text{Delivered} - \text{Scheduled})_{\text{days}}$. Strictly forbidden in feature space. |
| 36 | `Scheduled_Transit_Days` / `PO_to_Scheduled_Days` | $M_2$ (Contractual Lead Time) | Yes (Direct Drop: $\text{Sched} - \text{PO}$) | **Allowed** | Planned lead time horizon $(\text{Scheduled Delivery Date} - T_{\text{pred}})_{\text{days}}$. Core pre-dispatch baseline feature. |
| 37 | `is_rdc_fulfillment` | $M_2$ (Sourcing Topology) | Yes (100% complete) | **Allowed** | Binary indicator: $1$ if `Fulfill Via == 'From RDC'`, $0$ if `Direct Drop`. Fundamental structural routing feature. |
| 38 | `is_pre_pq_process` | $M_1$ (Requisition Process) | Yes (100% complete) | **Allowed** | Binary indicator: $1$ if emergency order bypassing formal quote requisition, $0$ otherwise. |
| 39 | `po_sent_is_date` | $M_2$ (Milestone Metadata) | Yes (100% complete) | **Allowed** | Binary indicator: $1$ if `PO Sent to Vendor Date` is valid parseable datetime, $0$ otherwise. |
| 40 | `pq_first_sent_is_date` | $M_1$ (Milestone Metadata) | Yes (100% complete) | **Allowed** | Binary indicator: $1$ if `PQ First Sent Date` is valid parseable datetime, $0$ otherwise. |
| 41 | `weight_is_numeric` | $M_3$ / Ingestion Metadata | Yes (Data Quality Flag) | **Allowed** (as Data Quality Indicator) | Binary indicator: $1$ if explicit numeric weight was captured, $0$ if consolidated ASN or uncaptured note. |
| 42 | `freight_is_numeric` | $M_3$ / Ingestion Metadata | Yes (Data Quality Flag) | **Allowed** (as Data Quality Indicator) | Binary indicator: $1$ if explicit numeric freight cost was captured, $0$ if DDP included, invoiced separately, or consolidated ASN. |
| 43 | `is_temporal_anomaly` | Data Quality Audit / Ingestion | Derived from $M_2, M_4, M_5$ | **Forbidden** (as ML feature) / **Cohort Filter** | Binary flag identifying 12 historical ERP timestamp inversions. Because it is calculated using $T_{\text{Deliv}}$ and $T_{\text{Record}}$, it cannot be used as an input feature; used strictly as an eligibility exclusion filter. |
| 44 | `PQ_to_PO_Days` | $M_2$ (Order Lead Time) | Yes (Direct Drop: $\text{PO} - \text{PQ}$) | **Allowed** | Pre-order procurement lag. Available at $T_{\text{pred}}$. |

---

## 3. Deep-Dive Operational Analysis on High-Risk Fields

### 3.1 `Delivery Recorded Date` (Post-Outcome ERP Administrative Log)
- **Empirical Summary**: 100% non-null (10,324 rows). Exactly 81.49% logged same day as delivery; 18.48% logged 1 to 546 days after delivery.
- **Leakage Reality**: `Delivery Recorded Date` represents an administrative data entry milestone when destination country health clinics transmit stamped delivery receipts back to USAID headquarters.
- **Classification**: **POST-OUTCOME / 100% FORBIDDEN**.
- **Constraint**: Must never enter feature matrix under any transform (e.g. lead time to recording).

### 3.2 `Delivered to Client Date` (Target Outcome Timestamp)
- **Empirical Summary**: 100% non-null (10,324 rows).
- **Leakage Reality**: Physical arrival timestamp at client destination.
- **Classification**: **TARGET-DERIVED / 100% FORBIDDEN IN FEATURE SPACE**.
- **Constraint**: Exclusively reserved as $T_{\text{outcome}}$ for calculating $y_{\text{clf}}$ and $y_{\text{reg}}$.

### 3.3 `Freight Cost (USD)` & `Weight (Kilograms)` (Consignment Scale Measurements vs Invoices)
- **Empirical Summary**:
  - `Weight (Kilograms)`: 61.72% numeric, 23.68% `'See ASN...'`, 14.60% `'Weight Captured Separately'`.
  - `Freight Cost (USD)`: 60.03% numeric, 23.68% `'See ASN...'`, 13.97% `'Freight Included in Commodity Cost'`, 2.31% `'Invoiced Separately'`.
- **Leakage Reality**:
  1. Gross weight is weighed on freight hub scales at physical dispatch ($M_3$), weeks after PO placement.
  2. Final freight invoices reflect carrier surcharges, detention/demurrage fees, or emergency mode conversions resulting from delays occurring *during* transit.
- **Classification**: **AMBIGUOUS / FORBIDDEN AS RAW ACTUALS AT $M_2$**.
- **Permissible Usage**: Data quality flags (`weight_is_numeric`, `freight_is_numeric`) and catalog-derived theoretical estimates ($\text{Unit\_Weight} \times \text{Quantity}$) are Allowed.

### 3.4 `Line Item Insurance (USD)` (Contractual Insurance Premium)
- **Empirical Summary**: 97.22% numeric, 2.78% legacy blanks. Fixed ratio of ~0.154% against `Line Item Value`.
- **Classification**: **ALLOWED** (Pre-transit financial invariant, though collinear with Line Item Value).

### 3.5 `Shipment Mode` (Planned Modality vs Operational Mode Shifting)
- **Empirical Summary**: 59.21% Air, 27.41% Truck, 6.30% Air Charter, 3.59% Ocean, 3.49% Missing.
- **Classification**: **ALLOWED as Planned Mode**.
- **Constraint**: Captures planned routing assigned at PO commitment. In-transit emergency mode modifications must not be backfilled into features.

### 3.6 `Vendor INCO Term` (Commercial Terms)
- **Empirical Summary**: 100% complete (5,404 `'N/A - From RDC'`, 2,778 `EXW`, 1,443 `DDP`, 397 `FCA`, 275 `CIP`, 27 other).
- **Classification**: **ALLOWED** (Finalized in vendor contract prior to PO release).

---

## 4. Strict Mathematical Temporal Boundary Invariants

1. **Precedence Invariant**: $\forall i \in \mathcal{D}_{\text{eligible}}, \quad T_{\text{pred}}(i) < T_{\text{Deliv}}(i)$.
2. **Feature Generation Boundary**: $\forall j, \quad \tau(X_{i,j}) \le T_{\text{pred}}(i)$.
3. **Target Orthogonality**: $\frac{\partial X_j}{\partial T_{\text{Deliv}}} = 0, \quad \frac{\partial X_j}{\partial T_{\text{Record}}} = 0$.
4. **Anomaly Cohort Filter**: Historical records with $T_{\text{Deliv}} < T_{\text{PO}}$ or $T_{\text{Sched}} < T_{\text{PO}}$ are filtered from training sets via `is_temporal_anomaly == 0`.
5. **Rolling-Origin Temporal Isolation**: In temporal splits, test instances must satisfy $T_{\text{pred}} \ge T_{\text{cutoff}}$ while all training instances satisfy $T_{\text{Deliv}} < T_{\text{cutoff}}$.
