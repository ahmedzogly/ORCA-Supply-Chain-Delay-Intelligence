> **RESEARCH PROTOTYPE ARCHITECTURE / PRODUCTION ROADMAP — not a production certification. See `docs/FINAL_RESULTS_SOURCE_OF_TRUTH.md`.**

# Master Architecture Specification — Supply Chain Delay Intelligence Platform

**Document Version**: 2.0.0 (Phase 2 Final Closure)  
**System**: Supply Chain Delay Intelligence System  
**Coverage**: Foundational Pipeline (Stages 0–13) & Phase 2 Research Extensions (E6.5, E7, E8, E9, E10)  
**Status**: **RESEARCH PROTOTYPE ARCHITECTURE / PRODUCTION ROADMAP**  

---

## 1. Architectural Philosophy & Design Principles

The **Supply Chain Delay Intelligence System** is engineered under four foundational software and research engineering principles:

1. **Python-First & Local-First Reproducibility**:  
   The entire analytical stack—from raw data parsing, schema validation, and temporal feature engineering to gradient boosting, conformal quantile regression, causal discovery, discrete-event digital twin simulation, and API/UI serving—is implemented exclusively in native Python (`Python 3.14.5`). The system operates deterministically on local workstations without requiring external cloud databases, microservices, or distributed orchestrators.

2. **Strict Point-in-Time Temporal Causality ($t_{\text{pred}} \le t_{\text{event}}$)**:  
   Every feature, transformation, and model decision is mathematically constrained to consume information available strictly at or before the designated prediction anchor timestamp ($T_{\text{pred}}$). Future milestone dates, post-dispatch operational changes, and post-outcome delivery logs are hermetically sealed behind automated leakage gates.

3. **Distribution-Free Uncertainty with Adaptive Recalibration**:  
   Rather than relying on uncalibrated heuristic probabilities or narrow Gaussian assumptions, the system provides distribution-free finite-sample prediction intervals via Conformal Quantile Regression (CQR), reinforced by chronological drift-triggered recalibration under temporal domain shifts.

4. **Instance-Dependent Economic Optimization with Human-in-the-Loop Triage**:  
   Standard classification losses (Logloss, symmetric accuracy) are replaced with instance-dependent asymmetric cost functions reflecting real-world freight surcharges, holding costs, and clinical stockout risks. All high-stakes operational interventions are throttled through capacity-constrained review queues with mandatory human oversight.

---

## 2. Global Component Map (19 Subsystems)

The platform comprises 19 modular subsystems organized into 5 architectural layers:

```
===================================================================================================================
                                      COMPLETE 19-MODULE SYSTEM TOPOLOGY
===================================================================================================================

 [ LAYER 1: DATA INGESTION, VALIDATION & TEMPORAL HORIZON GATING ]
   ├── Module 1: Ingestion Adapters (`src/delay_intelligence/data/adapters/`)
   │     ├── SCMS Adapter (`scms.py`) -> Canonical shipment line item normalization
   │     ├── DataCo Adapter (`dataco.py`) -> Supply chain multi-facility adapter
   │     └── Olist Adapter (`olist.py`) -> E-commerce marketplace logistics adapter
   ├── Module 2: Schema Contracts & Type Integrity (`src/delay_intelligence/validation/schemas.py`)
   │     ├── Pandera DataFrame Schemas & Pydantic Validation Models
   │     └── Data Cleaning, Sentinel Parsing & Missingness Resolution
   └── Module 3: Leakage Audit & Prediction Contract (`src/delay_intelligence/validation/leakage.py`)
         ├── Point-in-Time Boundary Gate (`prediction_contract.yaml`)
         └── 4-Tier Feature Availability Classifier (Allowed, Forbidden, Post-Outcome, Target-Derived)

 [ LAYER 2: FEATURE ENGINEERING & ROLLING-ORIGIN EVALUATION HARNESS ]
   ├── Module 4: Temporal Feature Engineering Pipeline (`src/delay_intelligence/features/`)
   │     ├── 39 Point-in-Time Features: Cyclical Dates, Historical Lag Volumes, Vendor/Country Ratios
   │     └── Scaler, One-Hot Encoder, and Imputation Transformers (`pipeline.py`)
   └── Module 5: Purged Rolling-Origin Evaluator (`src/delay_intelligence/evaluation/`)
         ├── 5 Expanding-Window Chronological Folds with 90-Day Embargo Buffer
         └── Temporal Split Validator & Metrics Computer (`metrics.py`)

 [ LAYER 3: CORE PREDICTIVE, UNCERTAINTY & EXPLAINABILITY ENGINE ]
   ├── Module 6: Production Champion Classifier (`src/delay_intelligence/models/catboost_classifier.py`)
   │     └── Calibrated CatBoost Classifier ($\tau^* = 0.16$, PR-AUC: 0.2869, F1: 0.3889)
   ├── Module 7: Multi-Quantile Base Regressor (`src/delay_intelligence/models/lightgbm_regressor.py`)
   │     └── LightGBM Pinball Loss Estimator ($q_{0.05}, q_{0.50}, q_{0.95}$)
   ├── Module 8: Static Conformal Quantile Regression (`src/delay_intelligence/uncertainty/conformal.py`)
   │     └── Split CQR with Finite-Sample Correction ($89.3\%$ coverage in dev)
   ├── Module 9: TreeSHAP Feature Attribution (`src/delay_intelligence/explainability/shap_explainer.py`)
   │     └── Local and Global TreeSHAP Attribution with Top-Feature Ranking
   └── Module 10: Causal Graph Discovery (`src/delay_intelligence/causal/discovery.py`)
         └── Constraint-Based PC Algorithm DAG Generation & Exploratory Hypothesis Formulator

 [ LAYER 4: PHASE 2 ADVANCED RESEARCH & ADAPTIVE GOVERNANCE EXTENSIONS ]
   ├── Module 11: 4D Chronological Drift Engine (`src/delay_intelligence/drift/`) [E6.5]
   │     ├── Feature Drift: PSI ($\epsilon = 10^{-4}$), 1-Wasserstein ($\widetilde{\mathcal{W}}_1$), KS-FDR, JSD, Chi2
   │     ├── Prediction & Target Drift: $\text{PSI}(\hat{p})$, $\widetilde{\mathcal{W}}_1(\hat{y})$, Two-Proportion $z$-test
   │     └── Uncertainty Drift: $\mathcal{W}_1(S_{\text{calib}}, S_{\text{det}})$, Coverage Deficit $\text{CovErr}_{90\%}$, Binomial Test
   ├── Module 12: Adaptive Conformal Recalibration Engine (`src/delay_intelligence/adaptive_conformal/`) [E7]
   │     ├── Strategy A (Static), Strategy B (Rolling), Strategy C (Drift-Triggered)
   │     └── Embargoed Window Manager & Finite-Sample Quantile Re-evaluator
   ├── Module 13: Instance-Dependent Cost-Sensitive Engine (`src/delay_intelligence/cost_sensitive/`) [E8]
   │     ├── Parameterized Cost Models: Low, Base, High Scenarios
   │     └── Bayes Optimal Thresholding ($\tau_i^* = \frac{C_{\text{action}}}{\gamma^* \cdot C_{\text{delay}}}$) & Budget Simulator
   ├── Module 14: Digital Twin Simulation & Stress Engine (`src/delay_intelligence/digital_twin/`) [E9]
   │     ├── Synthetic IoT Generator, Discrete-Event Inference Loop, Disruptions S0..S6
   │     └── Control-Tower Queue Pressure Surge Analyzer ($\text{QueuePressure} = 5.16$)
   └── Module 15: Counterfactual Policy Engine (`src/delay_intelligence/counterfactual/`) [E10]
         ├── Operational Policy Suite ($P_0$ to $P_5$) & Deterministic Transition Engine
         ├── Isolated Offline Oracle Benchmark (`oracle.py` AST Isolated)
         └── Control-Tower Review Budget Allocator ($K \in \{5\%, 10\%, 20\%\}$)

 [ LAYER 5: OPERATIONAL SERVING & USER SURFACES ]
   ├── Module 16: REST API Microservice (`src/delay_intelligence/api/main.py`)
   │     ├── Low-Latency Endpoints: `/predict`, `/uncertainty`, `/explain`, `/recommend`, `/drift`, `/policy`
   │     └── Pydantic Request/Response Schema Validation & Health Monitoring
   ├── Module 17: Interactive Control Tower UI (`src/delay_intelligence/dashboard/app.py`)
   │     ├── Multi-Page Streamlit App: Executive, Triage, Drift, Digital Twin, Model Governance
   │     └── Real-Time Dynamic What-If Policy Simulation
   ├── Module 18: Cryptographic Manifest & Invariance Gate (`artifacts/final_project_manifest.json`)
   │     └── 36/36 Baseline Artifact SHA-256 Bitwise Verification
   └── Module 19: Full Automated Test Suite (`tests/`)
         └── Legacy 659-test suite (historical pass record; current export status in closure manifest)
===================================================================================================================
```

---

## 3. Data Flow & Interface Contracts

### 3.1 4-Tier Data Provenance Model

To prevent the conflation of historical facts, synthetic stress variables, simulated actions, and economic models, every data structure carries a strict 4-tier provenance classification:

```
+----------------------------------------------------------------------------------------------------+
| TIER 1: OBSERVED_SCMS_DATA                                                                         |
| Raw ERP milestone dates, line item quantities, pack costs, shipment modes, vendor IDs, country.   |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
| TIER 2: SYNTHETIC_E9_STATE                                                                         |
| Observable operational dynamic state vectors S_i(t) and post-dispatch telemetry indicators.       |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
| TIER 3: SIMULATED_COUNTERFACTUAL                                                                   |
| Deterministic post-action states (D_tilde, p_tilde, W_tilde) computed under frozen assumptions.   |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
| TIER 4: SIMULATED_COST                                                                             |
| Modeled economic business costs computed under parameterized scenario matrices (Low, Base, High). |
+----------------------------------------------------------------------------------------------------+
```

### 3.2 Feature Availability & Point-in-Time Boundary Contract

The formal prediction contract is anchored at the order scheduling event ($T_{\text{pred}} = \text{Scheduled Delivery Date} - \text{Estimated Lead Time}$):

| Feature Category | Count | Example Fields | Point-in-Time Availability | Status |
| :--- | :---: | :--- | :--- | :---: |
| **Order & Line Item** | 8 | `Unit Price`, `Pack Price`, `Line Item Value`, `Weight (Kilograms)` | Present at Order Commitment | **ALLOWED** |
| **Logistics Corridor** | 9 | `Country`, `Managed By`, `Fulfill Via`, `Vendor INCO Term`, `Shipment Mode` | Finalized at Routing Selection | **ALLOWED** |
| **Historical Aggregates**| 12 | `vendor_hist_volume`, `country_hist_delay_rate`, `route_avg_lead_time` | Computed from $t \le T_{\text{pred}}$ | **ALLOWED** |
| **Temporal Cycles** | 10 | `scheduled_month_sin`, `scheduled_month_cos`, `day_of_week` | Deterministic Calendar Transform | **ALLOWED** |
| **Post-Outcome Fields**| 6 | `Delivered to Client Date`, `Delivery Recorded Date`, `Actual Lead Time` | Generated at/after Delivery | **FORBIDDEN (Leakage)** |
| **Target Variables** | 2 | `Delay_Flag` ($\mathbb{I}(\text{Delay} > 0)$), `Delay_Days` | Ground Truth Outcomes | **FORBIDDEN (Target)** |

---

## 4. Algorithmic Formulations

### 4.1 Split Conformal Quantile Regression (CQR) with Finite-Sample Adjustment
Given a calibration set $\mathcal{D}_{\text{calib}} = \{(X_i, Y_i)\}_{i=1}^n$ and pinball-loss quantile regressors $\hat{q}_{\alpha/2}(X), \hat{q}_{1 - \alpha/2}(X)$:

1. **Nonconformity Score Computation**:
   $$S_i = \max\left( \hat{q}_{\alpha/2}(X_i) - Y_i, \; Y_i - \hat{q}_{1 - \alpha/2}(X_i) \right)$$

2. **Finite-Sample Quantile Level**:
   $$p_{\text{level}} = \min\left(1.0, \; (1 - \alpha) \cdot \left(1 + \frac{1}{n}\right)\right)$$

3. **Conformal Cutoff**:
   $$Q = \text{Quantile}\left(\{S_i\}_{i=1}^n, \; p_{\text{level}}, \; \text{method='higher'}\right)$$

4. **Prediction Interval Construction**:
   $$\mathcal{C}(X) = \left[ \hat{q}_{\alpha/2}(X) - Q, \; \hat{q}_{1 - \alpha/2}(X) + Q \right]$$

### 4.2 4-Dimensional Chronological Drift Detection Engine
Drift is evaluated chronologically across sliding reference and detection windows:

1. **Population Stability Index (PSI)**:
   $$\text{PSI} = \sum_{b=1}^B \left( P_{\text{det}}(b) - P_{\text{ref}}(b) \right) \cdot \ln\left( \frac{P_{\text{det}}(b) + \epsilon}{P_{\text{ref}}(b) + \epsilon} \right)$$

2. **Normalized 1-Wasserstein Distance**:
   $$\widetilde{\mathcal{W}}_1(P_{\text{ref}}, P_{\text{det}}) = \frac{1}{\sigma_{\text{ref}}} \int_{-\infty}^{\infty} |F_{\text{ref}}(t) - F_{\text{det}}(t)| \, dt$$

3. **Tier-1 Feature Veto Rule**:  
   If $\text{PSI}(X_j) \ge 0.25$ for any $X_j \in \text{Tier-1 Features}$ (`Vendor INCO Term`, `Vendor`, `Country`, `Transit Days`, `vendor_hist_volume`, etc.), `DriftTriggerPolicy` immediately emits a `RED_TRIGGER` activating CQR recalibration.

### 4.3 Instance-Dependent Bayes-Optimal Thresholding
For shipment $i$ with value $V_i$, transport multiplier $\lambda_{\text{mode}}$, and criticality $\kappa_i$:

1. **Cost Matrix Parameters**:
   - $C_{\text{action}}(i) = c_{\text{exp\_base}} + \gamma_{\text{exp}} \cdot V_i$
   - $C_{\text{delay\_loss}}(i) = c_{\text{daily\_base}} \cdot \lambda_{\text{mode}} \cdot \hat{D}_i + \rho_{\text{value}} \cdot V_i \cdot \hat{D}_i$
   - $C_{\text{stockout}}(i) = c_{\text{fixed\_stockout}} \cdot \kappa_i$

2. **Bayes-Optimal Decision Threshold**:
   $$\tau_i^* = \frac{C_{\text{action}}(i)}{\gamma^* \cdot \left( C_{\text{delay\_loss}}(i) + C_{\text{stockout}}(i) \right)}$$
   An intervention (`EXPEDITE`) is triggered if and only if $\hat{p}_i \ge \tau_i^*$.

### 4.4 Control-Tower Review Budget Prioritization
Under a maximum operational capacity of $M = \lfloor K \cdot N \rfloor$ shipments:

1. **Expected Net Benefit Score**:
   $$\text{Score}_i = \max_{a \in \mathcal{A}} \left( \mathbb{E}[\text{Cost}(\text{NO\_ACTION} \mid S_i)] - \mathbb{E}[\text{Cost}(a \mid S_i)] \right)$$

2. **Knapsack Allocation**:
   Rank all active shipments by $\text{Score}_i$ in descending order. Allocate operational actions to the top $M$ shipments satisfying $\text{Score}_i > 0$. If fewer than $M$ shipments have positive yield, the remaining review capacity remains unallocated to prevent capital waste.

---

## 5. Security, Isolation & Production Governance

1. **Offline Oracle Isolation (AST Verified)**:  
   The `OfflineOraclePolicy` is architecturally isolated from the online serving stack. Abstract Syntax Tree (AST) scanning guarantees zero runtime imports or references of `oracle.py` in production decision modules.
2. **Cryptographic Baseline Freezing**:  
   All 36 foundational model checkpoints, calibration tables, and feature contracts are sealed with immutable SHA-256 hashes in `artifacts/final_project_manifest.json`.
3. **Mandatory Human-in-the-Loop Triage**:  
   All automated recommendations generated by the REST API and Control Tower require explicit operational sign-off from logistics managers before mutating external enterprise ERP systems.
