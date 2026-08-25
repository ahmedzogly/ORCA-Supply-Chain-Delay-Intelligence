# Comprehensive Inventory of Technical, Causal & Operational Limitations

**System**: Supply Chain Delay Intelligence Platform  
**Document**: Final Limitations, Boundary Conditions & Risk Analysis  
**Audience**: Enterprise Architects, Logistics Directors, ML Engineers, Forensic Auditors  
**Status**: **AUTHORITATIVE / GOVERNANCE SEALED**  

---

## 1. Executive Statement of Limitations

High-integrity machine learning systems require clear, uncompromising documentation of their structural limitations, boundary conditions, and failure modes. The **Supply Chain Delay Intelligence System** was developed and validated under rigorous academic and enterprise standards. However, like all empirical data-driven systems, its predictive, uncertainty, economic, and prescriptive outputs are subject to specific data, modeling, and operational constraints.

This document establishes a complete, honest inventory of all known limitations across five core dimensions:
1. **Data Ingestion & Observational Scope Boundaries**
2. **Causal Inference & Counterfactual Policy Boundaries**
3. **Economic Cost Modeling & ROI Assumptions**
4. **Digital Twin & IoT Telemetry Boundaries**
5. **Operational Serving, Compute & Capacity Boundaries**

---

## 2. Data Ingestion & Observational Scope Boundaries

### 2.1 Single-Source Public Health Bias (SCMS Dataset Scope)
- **Domain Specialization**: The primary modeling dataset (USAID / SCMS, 10,324 shipment line items, 2006–2015) reflects donor-funded global public health logistics (HIV/AIDS antiretrovirals, malaria diagnostics, medical consumables) delivered predominantly to developing nations across Sub-Saharan Africa, the Caribbean, and Southeast Asia.
- **Limited Generalizability**: Corridors, customs clearance procedures, and freight consolidation mechanisms in donor-funded health logistics differ substantially from commercial e-commerce, automotive, or industrial manufacturing supply chains. While cross-sector adapters for DataCo and Olist were constructed (Stage 10), baseline models must be re-benchmarked before deployment in non-pharma domains.

### 2.2 Unobserved Supplier & Tier-2 Manufacturing Internals
- **ERP Boundary**: The dataset captures milestones only from purchase requisition to final destination delivery. Unobserved factors include:
  - Raw material shortages at upstream active pharmaceutical ingredient (API) manufacturers.
  - Port congestion bottlenecks prior to carrier dispatch.
  - Sub-tier customs documentation delays.
- **Consequence**: Delays arising from sudden black-swan geopolitical strikes or upstream factory shutdowns cannot be forecasted from historical ERP milestone timestamps alone.

### 2.3 Historical ERP Timestamp Granularity & Missingness
- **Daily Aggregation**: Historical dates are recorded at calendar-day granularity without sub-daily hour/minute timestamps.
- **RDC Channel Structural Absence**: In the 5,404 shipments fulfilled `From RDC`, `PO Sent to Vendor Date` is structurally absent (`'N/A - From RDC'`). Although successfully resolved by anchoring predictions to the scheduling event ($T_{\text{pred}}$), granular pre-dispatch vendor staging milestones remain unobserved for RDC inventory.

---

## 3. Causal Inference & Counterfactual Policy Boundaries

### 3.1 Total Absence of Randomized Controlled Trial (RCT) Logs
- **Observational Historical Data**: Historical SCMS records represent purely observational operational history. There were no randomized A/B trials where identical shipments were assigned to expedited vs standard freight under controlled conditions.
- **Confounding Risk**: In historical data, expensive expediting actions were selectively applied to shipments already perceived to be high-risk or high-value. Naive observational regressions will confuse the intervention with the risk factor (confounding by indication).

### 3.2 Non-Causal Nature of Counterfactual Transitions
- **Simulation Assumptions**: All action effects evaluated in Experiments E9 and E10 ($\Delta D = -3.0\text{ days}$ for expediting, $\Delta D = -2.0\text{ days}$ for mode review, $\Delta R = -15\%$ for vendor escalation) represent **frozen simulation assumptions parameterized by domain experts**, NOT empirically estimated causal treatment effects.
- **Prohibited Claim**: The system CANNOT and DOES NOT claim that executing expediting in the physical world will causally guarantee a 3-day transit reduction across all future carriers.

### 3.3 Exploratory Causal Graphs (PC Algorithm)
- **Constraint-Based Hypotheses**: The causal DAGs produced by the PC Algorithm (Stage 7) reflect conditional independence relationships under specific statistical assumptions (Faithfulness, Causal Sufficiency). They serve exclusively as *exploratory hypotheses* for supply chain managers, not proven physical causal mechanisms.

---

## 4. Economic Cost Modeling & ROI Assumptions

### 4.1 Parameterized Cost Functions vs Accounting Realities
- **Cost Scenario Framework**: Economic evaluations (Stage 8, E8, E10) rely on explicit cost scenario models (Low, Base, High) with parameterized constants:
  - Base Daily Holding Cost: $c_{\text{daily}} = \$150/\text{day}$
  - Fixed Stockout Penalty: $c_{\text{stockout}} = \$500$
  - Expedited Carrier Surcharge: $C_{\text{expedite}} = \$500 + 0.5\% V_i$
  - Supplier Escalation Inquiry Fee: $C_{\text{inquiry}} = \$30$
- **Simulated vs Realized Savings**: The reported cost savings (e.g., $+\$31,489.44$ under $K=10\%$ budget in E8; $+\$2,194.78$ under $K=5\%$ budget in E10) are **simulated expected cost reductions under the defined cost parameters**, NOT audited accounting dollars. Real-world financial impact depends on actual carrier freight rate contracts, contractual SLA penalty clauses, and clinical storage costs.

### 4.2 Prohibited Financial Guarantees
- Enterprise leadership must NOT interpret model-based expected savings as guaranteed financial dividends.

---

## 5. Digital Twin & IoT Telemetry Boundaries

### 5.1 Synthetic Nature of IoT Streams (E9)
- **Monitoring-Only Telemetry**: Historical SCMS data contains no active IoT sensor streams (GPS tracking, continuous cold-chain temperature logs). All telemetry variables in Experiment E9 (`temperature_c`, `route_deviation_km`, `current_ETA`) were synthetically generated to stress-test the system architecture.
- **Monitoring Scope**: In production, continuous IoT telemetry acts as a **monitoring-only signal** that triggers drift detection and human review queues; it does NOT directly mutate the frozen Stage 5 CatBoost delay probability model.

---

## 6. Operational Serving, Compute & Capacity Boundaries

### 6.1 Control-Tower Review Capacity Saturation (Queue Pressure)
- **Capacity Bottlenecks**: As demonstrated in Experiment E9, unthrottled alert generation during systemic network disruptions increases human review demand by $+416\%$ ($\text{QueuePressure} = 5.16$).
- **Operational Requirement**: The system must ALWAYS be deployed with the `ReviewBudgetAllocator` ($K \le 10\%$). Operating the system with unconstrained thresholding will overwhelm logistics triage personnel and induce operator fatigue.

### 6.2 Prohibited Autonomous Physical Execution
- **Mandatory Human-in-the-Loop**: The platform is an advisory intelligence system. It is strictly forbidden to connect the REST API output directly to automatic purchase order modification, carrier cancellation, or warehouse re-routing workflows without human logistics officer approval.

---

## 7. Summary Risk Matrix & Governance Guidance

| Dimension | Known Risk / Limitation | Mandatory Operational Guardrail |
| :--- | :--- | :--- |
| **Data Scope** | Single-source donor health logistics bias | Mandatory baseline re-calibration prior to non-health commercial deployment |
| **Causality** | Observational data lacks RCT intervention logs | Prohibit causal efficacy claims; tag all policy outcomes as `SIMULATED_COUNTERFACTUAL` |
| **Economics** | Cost models rely on parameterized assumptions | Prohibit deterministic ROI guarantees; report across Low, Base, High scenarios |
| **IoT Telemetry** | Historical SCMS lacks real-time sensor streams | Categorize IoT telemetry strictly as `SYNTHETIC_E9_STATE` monitoring signals |
| **Operations** | Alert fatigue & queue collapse under shocks | Enforce `ReviewBudgetAllocator` ($K \le 10\%$) and mandatory human-in-the-loop sign-off |

---

## 8. 4-Tier Data Provenance Enforcement

All platform artifacts must preserve immutable provenance tagging:
1. `OBSERVED_SCMS_DATA`: Historical ground-truth records.
2. `SYNTHETIC_E9_STATE`: Observable operational dynamic variables.
3. `SIMULATED_COUNTERFACTUAL`: Hypothetical post-action trajectories.
4. `SIMULATED_COST`: Synthetic business economic costs computed from explicit parameter models.
