> **LEGACY RESEARCH LEDGER — presentation claims are superseded by `docs/FINAL_RESULTS_SOURCE_OF_TRUTH.md`. Historical metrics are retained for provenance; do not mix them with v2 serving validation.**

# Master Results Ledger: Consolidated Empirical Benchmarks Across All 19 Stages

**Project**: Supply Chain Delay Intelligence Platform  
**Document**: Authoritative Frozen Results & Verification Matrix  
**Dataset**: USAID / SCMS Delivery History (10,324 shipment line items, 2006–2015)  
**Evaluation Scope**: Stages 0–13 & Phase 2 Experiments E6.5, E7, E8, E9, E10  
**Status**: **FROZEN AUDIT RECORD / 100% BITWISE VERIFIED**  

---

## 1. Stage-by-Stage Milestone Execution & Deliverable Registry

| Stage / Experiment | Stage Title & Objective | Status | Primary Formal Report | Key Source & Metric Artifacts |
| :--- | :--- | :---: | :--- | :--- |
| **Stage 0** | Repository, Architecture & Environment Setup | **PASS** | `stage_0_report.md` | `ARCHITECTURE.md`, `pyproject.toml`, `docs/technology_decision_record.md` |
| **Stage 1** | SCMS Data Ingestion, Adapters & Data Audit | **PASS** | `stage_1_report.md` | `src/delay_intelligence/data/adapters/scms.py`, `docs/scms_data_audit.md` |
| **Stage 2** | Prediction Contract & Target Definition | **PASS** | `stage_2_report.md` | `configs/prediction_contract.yaml`, `docs/prediction_contract.md` |
| **Stage 3** | Feature Leakage Audit & Point-in-Time Boundary Gate | **PASS** | `stage_3_report.md` | `docs/leakage_specification.md`, `docs/feature_availability_matrix.md` |
| **Stage 4** | Temporal Feature Engineering & Preprocessing | **PASS** | `stage_4_report.md` | `src/delay_intelligence/features/`, `artifacts/data/scms_modeling_features.parquet` |
| **Stage 5** | Production Champion Modeling & Threshold Governance | **PASS** | `stage_5_report.md` | `artifacts/model_registry/v1/catboost_champion.cbm`, `docs/production_model_report.md` |
| **Stage 6** | Probabilistic Severity & Conformal Quantile Regression | **PASS** | `stage_6_report.md` | `src/delay_intelligence/uncertainty/conformal.py`, `artifacts/model_registry/v1/cqr_calibration.json` |
| **Stage 7** | Explainability (SHAP) & Causal Discovery (PC Algorithm) | **PASS** | `stage_7_report.md` | `src/delay_intelligence/explainability/`, `src/delay_intelligence/causal/` |
| **Stage 8** | Prescriptive Decision Engine & Cost Matrix | **PASS** | `stage_8_report.md` | `src/delay_intelligence/decision/`, `artifacts/decision/prescriptive_decisions.csv` |
| **Stage 9** | Serving Infrastructure (FastAPI & Streamlit UI) | **PASS** | `stage_9_report.md` | `src/delay_intelligence/api/main.py`, `src/delay_intelligence/dashboard/app.py` |
| **Stage 10** | Cross-Sector Adaptation (DataCo & Olist Ingestion) | **PASS** | `stage_10_report.md` | `src/delay_intelligence/data/adapters/dataco.py`, `olist.py` |
| **Stage 11** | End-to-End Integration & Multi-Domain Stress | **PASS** | `stage_11_report.md` | `tests/test_end_to_end.py`, `docs/final_validation_report.md` |
| **Stage 12** | Final Scientific Validation & First Holdout Access | **PASS** | `stage_12_report.md` | `artifacts/final/final_holdout_metrics.json`, `docs/academic_claims_audit.md` |
| **Stage 13** | Packaging, Manifest Freezing & Baseline Closure | **PASS** | `FINAL_REPORT.md` | `README.md`, `artifacts/final_manifest.json` |
| **E6.5** | Chronological Multi-Dimensional Drift Detection | **PASS** | `stage_2_e6_5_report.md` | `src/delay_intelligence/drift/`, `configs/drift.yaml` |
| **E7** | Adaptive Conformal Recalibration Protocol | **PASS** | `phase2_part1_final_report.md` | `src/delay_intelligence/adaptive_conformal/`, `configs/adaptive_conformal.yaml` |
| **E8** | Instance-Dependent Cost-Sensitive Learning | **PASS** | `stage_e8_report.md` | `src/delay_intelligence/cost_sensitive/`, `configs/cost_scenarios.yaml` |
| **E9** | Digital Twin & Real-Time IoT Scenario Stress Testing | **PASS** | `stage_e9_report.md` | `src/delay_intelligence/digital_twin/`, `docs/e9_simulation_assumptions.json` |
| **E10** | Counterfactual Policy Evaluation & Optimization | **PASS** | `stage_e10_report.md` | `src/delay_intelligence/counterfactual/`, `configs/e10_counterfactual.yaml` |

---

## 2. Comprehensive Predictive Modeling Benchmarks (Stage 5 Development CV)

Evaluated across 5 chronological purged expanding-window folds ($N=7,306$ development cohort):

| Model Family | Hyperparameters / Features | PR-AUC | ROC-AUC | F1-Score | Optimal Threshold ($\tau^*$) | Brier Score | Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | L2 Penalty ($C=1.0$), Standard Scaling | 0.2458 | 0.6512 | 0.2728 | 0.1800 | 0.1420 | 3 |
| **LightGBM Classifier** | `n_estimators=100`, `max_depth=6`, `lr=0.05` | 0.2593 | 0.6784 | 0.0902 | 0.5000 | 0.1395 | 2 |
| **CatBoost Champion** | `iterations=500`, `depth=6`, `l2_leaf_reg=3.0` | **0.2869** | **0.7104** | **0.3889** | **0.1600** | **0.1370** | **1 (CHAMPION)** |

---

## 3. Stage 12 Baseline Final Holdout Benchmarks ($N=1,013$, Single Pass)

Evaluated on the 365-day final holdout ($T_{\text{pred}} > \text{2014-08-24}$, 61 delays):
- **Classification Performance**:
  - PR-AUC: **0.1810** (Domain shift drop from $0.2869$ in CV)
  - ROC-AUC: **0.6951**
  - F1-Score: **0.0606** (Precision = $0.4000$, Recall = $0.0328$ at standard threshold)
  - Balanced Accuracy: **0.5148**
  - Brier Score: **0.0542**
- **Static CQR Uncertainty Collapse**:
  - Empirical Coverage: **22.95%** at nominal $90\%$ (Coverage Error = $+0.6705$)
  - Mean Interval Width: **4.19 days**
  - *Core Scientific Finding*: Static conformal inference fails completely under macroeconomic temporal shift.

---

## 4. Phase 2 — E7 Adaptive Conformal Recalibration Benchmark (Nominal 90%)

Evaluated on the 365-day final holdout under a 90-day label maturity embargo buffer:

| Strategy | Empirical Coverage ($\text{Cov}_{90\%}$) | Coverage Error ($\text{CovErr}$) | Lower Bound Violations | Upper Bound Violations | Mean Interval Width | Recalibrations ($K$) | Annual Compute Overhead | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy A: Static CQR** | 80.36% | $+0.0964$ | 11.25% | 8.39% | 3.20 days | 0 | 0.000 ms | **FAIL (Undercovered)** |
| **Strategy B: Rolling CQR** | 86.48% | $+0.0352$ | 7.11% | 6.42% | 33.23 days | 3 | 0.330 ms | **FAIL (Marginal)** |
| **Strategy C: Drift-Triggered CQR** | **93.88%** | **$-0.0388$** | **2.67%** | **3.46%** | **49.93 days** | **4** | **0.538 ms** | **Observed empirical coverage; wide intervals** |

---

## 5. Phase 2 — E8 Instance-Dependent Cost-Sensitive Holdout Benchmark

Evaluated on the 365-day holdout ($N=1,013$, 61 delays):

### 5.1 Unconstrained Model Strategy Matrix (Base Scenario)

| Strategy / Baseline | Realized Business Cost ($) | Net Savings vs Do-Nothing ($) | Cost Reduction (%) | Reviews Count | Delay Capture Rate | Delay-Days Captured |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Do-Nothing (Baseline)` | \$411,378.96 | \$0.00 | 0.00% | 0 | 0.000 | 0.0 |
| `Always-Intervene` | \$460,326.50 | -\$48,947.54 | -11.90% | 1,013 | 1.000 | 260.0 |
| `E8-A_tau0.5 (Standard CatBoost)` | \$410,363.02 | \$1,015.94 | 0.25% | 5 | 0.016 | 5.0 |
| `E8-A_f1 (Tuned F1 Threshold)` | \$372,967.16 | \$38,411.80 | 9.34% | 282 | 0.672 | 167.0 |
| `E8-B_cost_weighted (Cost Loss)` | \$398,649.01 | \$12,729.95 | 3.09% | 519 | 0.754 | 192.0 |
| `E8-C_bayes_threshold (Bayes tau)`| \$410,985.95 | \$393.01 | 0.10% | 600 | 0.754 | 192.0 |
| **`E8-C_tuned_gamma` (Champion $\gamma^*=1.2$)**| **\$389,237.70** | **+\$22,141.26** | **5.38%** | **453** | **0.754** | **192.0** |

### 5.2 Operational Review Budget Benchmark on Holdout ($N=1,013$)

| Capacity ($K$) | Policy | Realized Cost ($) | Net Savings vs Do-Nothing ($) | Cost Reduction (%) | Delayed Value Captured (%) | Review Count |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$K = 5\%$** (50 items) | `VALUE_ONLY` | \$396,843.06 | \$14,535.90 | 3.53% | 49.7% | 50 |
| | `RISK_ONLY` | \$399,364.86 | \$12,014.10 | 2.92% | 21.9% | 50 |
| | `STANDARD` | \$410,363.02 | \$1,015.94 | 0.25% | 2.0% | 5 |
| | **`COST_SENSITIVE`** | **\$385,260.02** | **+\$26,118.94** | **6.35%** | **64.9%** | **50** |
| **$K = 10\%$** (101 items) | `VALUE_ONLY` | \$391,546.16 | \$19,832.81 | 4.82% | 75.1% | 101 |
| | `RISK_ONLY` | \$393,959.05 | \$17,419.92 | 4.23% | 37.2% | 101 |
| | `STANDARD` | \$410,363.02 | \$1,015.94 | 0.25% | 2.0% | 5 |
| | **`COST_SENSITIVE`** | **\$379,889.52** | **+\$31,489.44** | **7.65%** | **76.2%** | **101** |
| **$K = 20\%$** (202 items) | `VALUE_ONLY` | \$390,027.80 | \$21,351.16 | 5.19% | 94.6% | 202 |
| | `RISK_ONLY` | \$368,193.28 | \$43,185.68 | 10.50% | 86.2% | 202 |
| | `STANDARD` | \$410,363.02 | \$1,015.94 | 0.25% | 2.0% | 5 |
| | **`COST_SENSITIVE`** | **\$368,323.79** | **+\$43,055.17** | **10.47%** | **91.2%** | **202** |

---

## 6. Phase 2 — E9 Digital Twin Stress-Testing Benchmark

| Scenario | Severity | Injected Events | Detected Events | Detection Rate | False Alarms | Recalibrations | Interval Width | Human Review Rate | Simulated Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0 Normal** | 0 | 3,039 | 0 | 0.000 | 0 | 0 | 49.2 days | 4.1% | \$112,000 |
| **S1 Temp Excursion** | 1 | 303 | 258 | 0.854 | 30 | 0 | 54.1 days | 11.6% | \$176,675 |
| **S2 Route Deviation** | 2 | 303 | 275 | 0.908 | 30 | 1 | 59.0 days | 19.1% | \$240,700 |
| **S3 Transit Slowdown** | 2 | 303 | 275 | 0.908 | 30 | 1 | 56.6 days | 16.1% | \$240,960 |
| **S4 ETA Shock** | 3 | 303 | 299 | 0.987 | 30 | 2 | 73.8 days | 41.6% | \$302,125 |
| **S5 Multi-Signal** | 4 | 303 | 301 | 0.995 | 30 | 3 | 88.6 days | 56.6% | \$362,900 |
| **S6 Disrupt Recovery** | 4 | 303 | 301 | 0.995 | 30 | 3 | 68.9 days | 34.1% | \$366,800 |

- **Queue Pressure Surge**: Under $20\%$ network disruption, review load surges from $124$ to $641$ shipments (**Queue Pressure = 5.16**, a $+416\%$ increase).

---

## 7. Phase 2 — E10 Counterfactual Policy Benchmark (Base Scenario)

| Policy ID | Policy Name | Realized Expected Cost ($) | Net Economic Benefit ($) | Total Oracle Gap ($) | Mean Regret ($) | Intervention Rate (%) | Policy Stability (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$P_0$** | `NO_ACTION` | \$161,126.33 | \$0.00 | \$2,194.78 | \$2.17 | 0.0% | 100.0% |
| **$P_1$** | `E8_COST_SENSITIVE` (Unconstrained) | \$262,965.50 | **-\$101,839.18** | \$104,033.96 | \$102.70 | 23.3% | 41.6% |
| **$P_2$** | `EXPEDITE` (Value $\ge \$100\text{k}$) | \$161,126.33 | \$0.00 | \$2,194.78 | \$2.17 | 0.0% | 100.0% |
| **$P_3$** | `TRANSPORT_MODE_REVIEW` | \$161,102.74 | +\$23.59 | \$2,171.19 | \$2.14 | 0.2% | 100.0% |
| **$P_4$** | `SUPPLIER_ESCALATION` | **\$160,656.36** | **+\$469.96** | **\$1,724.82** | **\$1.70** | **0.8%** | **100.0%** |
| **$P_5$** | `HUMAN_REVIEW` | \$161,581.93 | -\$455.60 | \$2,650.38 | \$2.62 | 0.6% | 100.0% |
| **Oracle** | `Offline_Oracle_Benchmark` | **\$158,931.55** | **+\$2,194.78** | **\$0.00** | **\$0.00** | **2.8%** | **100.0%** |
| **Budget** | **`ReviewBudgetAllocator` ($K=5\%$)** | **\$158,931.55** | **+\$2,194.78** | **\$0.00** | **\$0.00** | **2.8% (28 items)**| **100.0%** |

---

## 8. Cryptographic Baseline Verification & Test Suite Summary

- **Baseline Artifact SHA-256 Invariance**: 36 of 36 files matched expected cryptographic hashes with zero discrepancies.
- **Representative Test Suite**: The original environment recorded **659/659** tests passing. This is historical evidence only; the patched export records its current verification status in `artifacts/closure_manifest.json`.
- **Physical Execution Latency**: 105.57 seconds across complete regression test suite.
