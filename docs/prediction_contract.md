# Supply Chain Delay Intelligence System
# Formal Prediction Contract Specification

**System**: Supply Chain Delay Intelligence System  
**Milestone**: Stage 2 — Prediction Contract, Target Definition & Leakage Specification  
**Document**: Human-Readable Formal Prediction Contract (`docs/prediction_contract.md`)  
**Version**: `1.0.0`  
**Dataset Reference**: USAID / SCMS Delivery History Dataset ($N = 10,324$)  
**Status**: FORMALIZED, AUTHORITATIVE & APPROVED  

---

## Overview & Contract Authority

This document defines the formal, human-readable **Prediction Contract** for the Supply Chain Delay Intelligence Project. All downstream stages (Stage 3 Feature Engineering, Stage 4 Temporal Evaluation, Stage 5 Predictive Modeling & Uncertainty, Stage 6 Causal Discovery & Decision Engine, Stage 7 API & Dashboard) must strictly adhere to the mathematical formulas, temporal boundaries, feature classifications, and eligibility rules established herein.

---

## 1. Section: `prediction_unit`

- **Entity Level**: Shipment Line Item.
- **Granularity**: One single row in the canonical SCMS delivery database representing a specific commodity batch ordered, packed, and shipped to a recipient country health program.
- **Entity Primary Key**: `ID` (Integer, $[1, 86824]$, 100% unique, 0 nulls across 10,324 records).
- **Secondary Grouping Keys**:
  - `PO / SO #` (Purchase Order / Sales Order identifier, 6,233 distinct orders) — used for clustered group cross-validation to prevent intra-order correlation leakage.
  - `Project Code` (Public health program identifier, 142 distinct projects).
  - `Country` (Destination sovereign nation, 43 distinct countries).

---

## 2. Section: `prediction_timestamp`

The prediction timestamp ($T_{\text{pred}}$) defines the exact operational milestone at which delay intelligence inference occurs. 

### Mathematical Definition: Dual-Channel Operational Milestone Anchor
For any shipment line item $i \in \{1, \dots, N\}$:

$$T_{\text{pred}}(i) = \begin{cases}
\text{PO Sent to Vendor Date}_i, & \text{if } \text{Fulfill Via}_i = \text{'Direct Drop'} \land \text{PO Sent}_i \neq \text{NaT} \\
\text{PQ First Sent to Client Date}_i, & \text{if } \text{Fulfill Via}_i = \text{'From RDC'} \land \text{PQ Sent}_i \neq \text{NaT} \\
\text{PQ First Sent to Client Date}_i, & \text{if } \text{Fulfill Via}_i = \text{'Direct Drop'} \land \text{PO Sent}_i = \text{NaT} \land \text{PQ Sent}_i \neq \text{NaT} \\
\text{NaT}, & \text{otherwise (Legacy un-anchored records)}
\end{cases}$$

### Operational Interpretation:
- **Direct Drop Channel (47.66%)**: $T_{\text{pred}}$ is the moment the commercial Purchase Order is transmitted to the external manufacturer. Manufacturing lead time begins; shipment route planning is initiated.
- **From RDC Channel (52.34%)**: $T_{\text{pred}}$ is the moment the client Price Quotation / Stock Requisition is finalized. Regional warehouse stock allocation, picking, packing, and cross-border trucking schedules are committed.

### Anchor Coverage & Lead Time Distribution:
- **Total Anchored Records**: **8,336 records (80.74%)** across the 2006–2015 historical dataset.
- **Modern Era (2010–2015) Anchored Coverage**: **7,217 records (98.80%)**.
- **Median Operational Lead Time to Scheduled Date**: **129.0 days** (Mean: 140.3 days, IQR: $[73.0, 192.0]$ days).

---

## 3. Section: `outcome_timestamp`

- **Authoritative Outcome Field**: `Delivered to Client Date` ($100.00\%$ complete, $0$ nulls, $0$ unparseable strings).
- **Physical Definition**: The exact calendar date on which the consignment physically arrives and is formally accepted at the destination client warehouse / Central Medical Store (CMS).
- **Administrative Settlement Date ($T_{\text{Record}}$)**: `Delivery Recorded Date` ($100.00\%$ complete, strictly post-outcome, logged $+2.90$ days after physical arrival on average). Strictly forbidden in the feature space.

---

## 4. Section: `forecast_horizon`

- **Mathematical Formula**:
  $$\Delta t_{\text{horizon}}(i) = \left( T_{\text{Sched}}(i) - T_{\text{pred}}(i) \right)_{\text{days}}$$
  Where $T_{\text{Sched}}$ is the contractual `Scheduled Delivery Date`.
- **Measurement Unit**: Calendar Days ($\mathbb{Z}$).
- **Minimum Operational Lead Time**: $\Delta t_{\text{horizon}} \ge 0$ days for standard planned shipments. (Same-day emergency releases where $\Delta t_{\text{horizon}} = 0$ are retained with explicit flag).
- **Forecast Window Classifications**:
  - *Short Horizon* ($0 \le \Delta t \le 30$ days): Emergency replenishments, regional truck transfers.
  - *Standard Horizon* ($31 \le \Delta t \le 120$ days): Standard international air cargo, routine RDC orders.
  - *Long Horizon* ($\Delta t > 120$ days): Ocean freight consignments, annual bulk pharmaceutical procurements.

---

## 5. Section: `classification_target`

### Formal Definition: Binary Delay Indicator (`is_delayed` / `Delay_Flag`)
Let $T_{\text{Sched}}^{(i)}$ denote `Scheduled Delivery Date` and $T_{\text{Deliv}}^{(i)}$ denote `Delivered to Client Date`:

$$y_{\text{clf}}^{(i)} = \mathbb{I}\left( T_{\text{Deliv}}^{(i)} > T_{\text{Sched}}^{(i)} \right) = \begin{cases}
1 & \text{if } T_{\text{Deliv}}^{(i)} > T_{\text{Sched}}^{(i)} \quad (\text{Delayed / Late, Delay\_Days} > 0) \\
0 & \text{if } T_{\text{Deliv}}^{(i)} \le T_{\text{Sched}}^{(i)} \quad (\text{Non-Delayed: On-Time or Early})
\end{cases}$$

### Empirical Class Distribution ($N = 10,324$):
- **Class 1 (Delayed)**: **1,186 records (11.4878%)**
- **Class 0 (Non-Delayed)**: **9,138 records (88.5122%)**
  - *On-Time Subgroup* ($T_{\text{Deliv}} = T_{\text{Sched}}$): 6,324 records (61.2553%)
  - *Early Delivery Subgroup* ($T_{\text{Deliv}} < T_{\text{Sched}}$): 2,814 records (27.2569%)
- **Imbalance Ratio**: $1 : 7.71$ (Minority prevalence $11.49\%$).

### 5-Pillar Justification for Mapping Early Delivery to Class 0:
1. **Contractual & SLA Mandate**: Public health delivery contracts require delivery *on or before* the scheduled need date. Delivering early satisfies contractual obligations.
2. **Asymmetric Clinical Loss Function ($C_{\text{late}} \gg C_{\text{early}}$)**: Late delivery causes life-threatening clinical stockouts of ARVs and test kits. Early delivery provides buffer stock without clinical harm.
3. **Operational Warning System Alignment**: A positive alert ($\hat{y}=1$) initiates costly freight expediting, route changes, and supplier escalations. Early shipments do not require late-delivery mitigation.
4. **Depot Buffering Capability**: Recipient country central medical stores and RDCs operate buffer warehousing specifically intended to absorb early inventory.
5. **Separation of Concerns**: Magnitude of earliness is fully preserved in the continuous regression target (`Delay_Days`), keeping the binary classifier focused purely on SLA breach risk.

---

## 6. Section: `regression_target`

### Formal Definition: Continuous Delay Magnitude (`Delay_Days`)
The continuous regression target $y_{\text{reg}}^{(i)} \in \mathbb{Z}$ is the signed integer difference in calendar days:

$$y_{\text{reg}}^{(i)} = \text{Delay\_Days}^{(i)} = \left( T_{\text{Deliv}}^{(i)} - T_{\text{Sched}}^{(i)} \right)_{\text{days}}$$

### Empirical Moments & Distribution Summary ($N = 10,324$):
- **Mean**: $-6.02$ days
- **Standard Deviation**: $27.23$ days
- **Median ($Q_2$)**: $0.00$ days
- **Interquartile Range (IQR)**: $3.00$ days ($[-3.0, 0.0]$)
- **Minimum**: $-372.00$ days
- **Maximum**: $+192.00$ days
- **Delayed Subset ($N = 1,186$)**: Mean $= +21.37$ days, Median $= +12.00$ days, $P_{95} = +70.00$ days, Max $= +192.00$ days.
- **Early Subset ($N = 2,814$)**: Mean $= -31.11$ days, Median $= -18.00$ days, Min $= -372.00$ days.

### Downstream Hurdle Decomposition (Supported Architectural Variant):
- **Unconditional Regressor**: $\hat{y} \in \mathbb{R}$ trained directly on all eligible records.
- **Two-Stage Hurdle Architecture**:
  - Stage 1: $P(y_{\text{clf}} = 1 \mid \mathbf{x})$
  - Stage 2: $\mathbb{E}[y_{\text{reg}} \mid y_{\text{clf}} = 1, \mathbf{x}]$
  - Expected Delay: $\hat{y}_{\text{hurdle}} = P(y_{\text{clf}} = 1 \mid \mathbf{x}) \cdot \hat{\mu}_{\text{severity}}(\mathbf{x})$.

---

## 7. Section: `eligibility_rules`

A shipment line item $i$ is eligible for delay intelligence modeling if and only if all following deterministic boolean criteria are satisfied:

$$\text{Eligible}(i) \iff \mathcal{C}_1(i) \land \mathcal{C}_2(i) \land \mathcal{C}_3(i) \land \mathcal{C}_4(i) \land \mathcal{C}_5(i)$$

| Criterion | Rule Expression | Invariant Check | Pass Rate ($N=10,324$) |
|---|---|---|---|
| $\mathcal{C}_1$: Unique Identifier | `ID is not null and ID > 0` | Valid primary key | 10,324 / 10,324 (100.0%) |
| $\mathcal{C}_2$: Scheduled Milestone | `Scheduled Delivery Date is valid date` | $T_{\text{Sched}} \in [2006, 2016]$ | 10,324 / 10,324 (100.0%) |
| $\mathcal{C}_3$: Delivered Milestone | `Delivered to Client Date is valid date` | $T_{\text{Deliv}} \in [2006, 2016]$ | 10,324 / 10,324 (100.0%) |
| $\mathcal{C}_4$: Known Channel | `Fulfill Via in {'From RDC', 'Direct Drop'}` | Sourcing channel recognized | 10,324 / 10,324 (100.0%) |
| $\mathcal{C}_5$: Valid Product Group | `Product Group in {'ARV', 'HRDT', 'ANTM', 'ACT', 'MRDT'}` | Commodity domain valid | 10,324 / 10,324 (100.0%) |

### 100% Preservation of RDC Records:
- **Total RDC Records**: Exactly **5,404 records (52.34%)**.
- **Eligibility**: **100% Eligible** in base dataset. All 5,404 records are fully preserved to eliminate selection bias.

---

## 8. Section: `allowed_features`

Features finalized at or before $T_{\text{pred}}$ (Milestones $M_0, M_1, M_2$):

1. **Project & Geographic Identifiers**:
   - `Project Code` (Categorical, 142 categories)
   - `Country` (Categorical, 43 countries)
   - `Managed By` (Categorical, 4 field offices)
2. **Sourcing & Commercial Terms**:
   - `Fulfill Via` (Categorical: `From RDC`, `Direct Drop`)
   - `Vendor INCO Term` (Categorical: `EXW`, `DDP`, `FCA`, `CIP`, `DDU`, `DAP`, `CIF`, `N/A - From RDC`)
   - `Vendor` (Categorical, 73 vendors)
   - `Manufacturing Site` (Categorical, 88 manufacturing sites)
   - `Shipment Mode` (Categorical: `Air`, `Truck`, `Ocean`, `Air Charter` — planned mode at PO)
3. **Pharmaceutical & Clinical Specifications**:
   - `Product Group` (Categorical: `ARV`, `HRDT`, `ANTM`, `ACT`, `MRDT`)
   - `Sub Classification` (Categorical, e.g. Adult, Pediatric)
   - `Item Description` (Text / Categorical, 184 descriptions)
   - `Molecule/Test Type` (Categorical, 86 molecules)
   - `Brand` (Categorical, 48 brands / generic)
   - `Dosage` (Categorical / Numerical dosage strength)
   - `Dosage Form` (Categorical, e.g. Tablet, Capsule, Test kit)
   - `First Line Designation` (Binary: `Yes`, `No`)
   - `Unit of Measure (Per Pack)` (Integer pack size)
4. **Procurement Quantities & Financial Valuations**:
   - `Line Item Quantity` (Integer, pack volume)
   - `Line Item Value` (Float, total USD commodity value)
   - `Pack Price` (Float, USD price per pack)
   - `Unit Price` (Float, USD price per unit)
   - `Line Item Insurance (USD)` (Float, contractual cargo insurance premium)
5. **Derived Pre-Prediction Engineering & Quality Features**:
   - `is_rdc_fulfillment` (Binary indicator: $1$ if `From RDC`, $0$ if `Direct Drop`)
   - `is_pre_pq_process` (Binary indicator: $1$ if emergency pre-PQ order, $0$ otherwise)
   - `po_sent_is_date` (Binary indicator: $1$ if PO date is parseable datetime)
   - `pq_first_sent_is_date` (Binary indicator: $1$ if PQ date is parseable datetime)
   - `weight_is_numeric` (Binary data quality indicator)
   - `freight_is_numeric` (Binary data quality indicator)
   - `Scheduled_Transit_Days` / `PO_to_Scheduled_Days` (Planned lead time horizon: $T_{\text{Sched}} - T_{\text{pred}}$)
   - `PQ_to_PO_Days` (Pre-order quotation lag: $T_{\text{PO}} - T_{\text{PQ}}$)

---

## 9. Section: `forbidden_features`

The following attributes are strictly rejected from the predictive feature space:

| Attribute Name | Operational Lifecycle Origin | Rejection Classification | Rejection Rationale |
|---|---|---|---|
| `ID` | System Database Ingestion | Database Key | Primary surrogate key; causes memorization and row-order leakage. |
| `Delivered to Client Date` | $M_4$ Physical Delivery | **Target-derived** / Target Event | The ground-truth outcome timestamp ($T_{\text{outcome}}$). $100\%$ target leakage. |
| `Delivery Recorded Date` | $M_5$ Post-Delivery ERP Log | **Post-outcome** | Administrative ERP commit timestamp logged post-delivery. $100\%$ lookahead leakage. |
| `Delay_Flag` | $M_4$ Delivery Evaluation | **Target Variable** | Binary classification target. |
| `Delay_Days` | $M_4$ Delivery Evaluation | **Target Variable** | Continuous regression target. |
| `ASN/DN #` | $M_3$ Warehouse Packing | Dispatch Consignment | Advanced Shipping Notice / Delivery Note tracking number generated upon physical consignment. |
| `Weight (Kilograms)` (Raw) | $M_3$ Warehouse Scale Weigh-in | Ambiguous Actual | Raw gross scale weight measured at warehouse tender. Contains mixed ASN references and post-dispatch measurements. |
| `Freight Cost (USD)` (Raw) | $M_3 / M_5$ Carrier Invoicing | Ambiguous Actual | Final freight invoices reflecting in-transit demurrage, detention, and expedited air surcharges. |
| `is_temporal_anomaly` | Data Quality Audit / Ingestion | Target-derived Filter | Filter flag derived using $T_{\text{Deliv}}$ and $T_{\text{Record}}$. Used strictly for population filtering. |

---

## 10. Section: `temporal_constraints`

To guarantee mathematical consistency and eliminate lookahead bias across training, inference, and temporal validation:

1. **Prediction-to-Outcome Precedence**:
   $$\forall i \in \mathcal{D}_{\text{eligible}}, \quad T_{\text{pred}}(i) < T_{\text{Deliv}}(i)$$
2. **Strict Feature Availability Cutoff**:
   $$\forall X_k(i), \quad \tau\left(X_k(i)\right) \le T_{\text{pred}}(i)$$
   Where $\tau(X_k)$ is the generation timestamp of feature $X_k$.
3. **Forecast Horizon Non-Negativity**:
   $$\Delta t_{\text{horizon}}(i) = T_{\text{Sched}}(i) - T_{\text{pred}}(i) \ge 0$$
4. **Target Isolation Invariant**:
   $$\frac{\partial X_k(i)}{\partial T_{\text{Deliv}}(i)} = 0, \quad \frac{\partial X_k(i)}{\partial T_{\text{Record}}(i)} = 0, \quad \forall k$$
5. **Rolling-Origin Temporal Split Invariant**:
   For any temporal evaluation cutoff $T_{\text{cutoff}}$:
   $$\mathcal{D}_{\text{train}}(T_{\text{cutoff}}) = \left\{ i \mid T_{\text{Deliv}}(i) < T_{\text{cutoff}} \right\}$$
   $$\mathcal{D}_{\text{eval}}(T_{\text{cutoff}}) = \left\{ i \mid T_{\text{pred}}(i) \ge T_{\text{cutoff}} \quad \land \quad T_{\text{Deliv}}(i) < T_{\text{cutoff}} + H \right\}$$

---

## 11. Section: `anomaly_policy`

The SCMS historical dataset contains 12 audited historical ERP timestamp anomalies (5 negative lead times, 4 scheduled < PO sent, 3 recorded < delivered):

1. **Target Retention**:
   - In all 12 cases, the target outcome ($T_{\text{Deliv}} - T_{\text{Sched}}$) is physically valid and evaluates to Class 0 ($y_{\text{clf}} = 0$, non-delayed).
   - All 12 records are **fully preserved in Bronze and Silver datasets**.
2. **Feature-Layer Guard**:
   - The indicator flag `is_temporal_anomaly = 1` isolates these records.
   - For lead-time feature derivations ($T_{\text{Sched}} - T_{\text{PO}}$), negative values are clamped to `NaN` or 0 to avoid training distortions.
3. **Modern Era Cleanliness (2010–2015)**:
   - 0 temporal anomalies exist in the primary modern evaluation era (2011–2015).

---

## Sign-Off & Verification

This Prediction Contract is machine-enforced via `configs/prediction_contract.yaml`, programmatically validated via `src/delay_intelligence/validation/contract_validator.py`, and verified by automated unit tests in `tests/test_prediction_contract.py`.
