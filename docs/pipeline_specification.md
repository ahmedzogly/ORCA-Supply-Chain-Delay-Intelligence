# 10-Stage Pipeline Specification & Quality Gates

**Project**: Supply Chain Delay Intelligence System  
**Document Version**: 1.0.0  

This specification details the formal input contracts, output contracts, data artifacts, statistical algorithms, and quality gate assertions for all 10 stages of the research pipeline.

---

## Pipeline Overview Diagram

```
[Raw Data] -> (Stage 1: Ingest) -> [Bronze Parquet]
           -> (Stage 2: Validate) -> [Silver Parquet]
           -> (Stage 3: Leakage Gate) -> [Pre-Dispatch Silver]
           -> (Stage 4: Feature Eng) -> [Gold Features Matrix]
           -> (Stage 5: Temporal CV) -> [CV Split Folds]
           -> (Stage 6: Multi-Task ML) -> [Trained Models]
           -> (Stage 7: Conformal UQ) -> [Conformal Predictors]
           -> (Stage 8: Causal DML) -> [Causal Effects & DAGs]
           -> (Stage 9: Prescriptive) -> [Cost-Optimal Policies]
           -> (Stage 10: Serving) -> [FastAPI + Streamlit]
```

---

## Detailed Stage Specifications

### Stage 1: Data Ingestion & Source Adapters
- **Module**: `delay_intelligence.data`
- **Input**: Read-only raw CSV files from `scms/`, `olist/`, and `dataco/`.
- **Output**: Bronze Parquet datasets stored in `artifacts/data/bronze_{dataset}.parquet`.
- **Key Operations**:
  - `BaseIngestionAdapter.load_raw()`: Reads raw files using strict read-only file streams.
  - Character encoding handling (`utf-8` for SCMS/Olist, `latin1` for DataCo).
  - Parsing composite string fields (e.g. SCMS freight notes) without data loss.
- **Quality Gate**:
  - Raw source file modification timestamps and hashes must remain identical to baseline.
  - Bronze Parquet row count matches raw CSV record count exactly.

---

### Stage 2: Schema Validation & Data Quality Gate
- **Module**: `delay_intelligence.validation.schemas` & `integrity`
- **Input**: Bronze Parquet tables.
- **Output**: Silver Parquet tables stored in `artifacts/data/silver_{dataset}.parquet`, validation report in `artifacts/reports/validation_report.json`.
- **Key Operations**:
  - Type validation via Pandera schema contracts.
  - Critical field nullness checks ($< 5\%$ null threshold on primary keys and dates).
  - Range assertions (e.g., unit price $> 0$, line item quantity $\ge 1$).
  - Quarantine corrupted rows to `artifacts/quarantine/`.
- **Quality Gate**:
  - Zero schema violations on non-quarantined records.
  - Quarantine rate must not exceed 2.0% of total Bronze volume.

---

### Stage 3: Temporal Horizon & Leakage Audit Gate
- **Module**: `delay_intelligence.validation.leakage`
- **Input**: Silver Parquet tables.
- **Output**: Audited Silver dataset, `artifacts/reports/leakage_audit_report.md`.
- **Key Operations**:
  - Identification of prediction anchor timestamp ($t_{\text{pred}} = t_{\text{PO Sent to Vendor}}$ or order placement).
  - Automated scanning and removal of downstream lifecycle columns ($t > t_{\text{pred}}$), such as actual delivery date, recorded date, customer reviews, or actual transit duration.
- **Quality Gate**:
  - Strict mathematical invariant: $\max(\text{timestamp}(X)) \le t_{\text{pred}}$.
  - Absolute correlation of any single feature with delay target $< 0.95$ (leakage guard).

---

### Stage 4: Feature Engineering & Preprocessing
- **Module**: `delay_intelligence.features`
- **Input**: Audited Silver dataset.
- **Output**: Gold feature matrix `artifacts/data/gold_features.parquet`, `artifacts/data/feature_metadata.json`.
- **Key Operations**:
  - Cyclical temporal feature extraction (sine/cosine transformations of month, day-of-week, day-of-year).
  - Scheduled transit duration calculation ($\text{Scheduled Delivery Date} - \text{PO Sent Date}$).
  - Out-of-fold target encoding for high-cardinality categoricals (Country, Vendor, Manufacturing Site).
  - One-hot encoding for low-cardinality categoricals (INCO Term, Shipment Mode).
  - Robust scaling and log transformation for heavy-tailed numeric values.
- **Quality Gate**:
  - All stateful transformers must fit exclusively on training splits (zero test fold leakage).
  - Gold feature matrix contains zero unhandled NaN or Inf values.

---

### Stage 5: Temporal Rolling-Origin Cross-Validation
- **Module**: `delay_intelligence.evaluation.splitters`
- **Input**: Gold feature matrix.
- **Output**: Cross-validation fold indices `artifacts/data/cv_splits.json`.
- **Key Operations**:
  - `PurgedRollingOriginSplitter` execution.
  - Expanding training window (e.g., 24 months) and forward evaluation window (e.g., 6 months).
  - Embargo period (e.g., 14 days) to prevent overlapping in-transit shipments across split boundaries.
- **Quality Gate**:
  - Strict temporal ordering: $\max(\text{timestamp}(\text{Train}_k)) < \min(\text{timestamp}(\text{Val}_k))$.
  - Random k-fold splitting is strictly forbidden.

---

### Stage 6: Multi-Task Predictive Modeling Engine
- **Module**: `delay_intelligence.models`
- **Input**: Gold feature matrix + CV split indices.
- **Output**: Serialized models `artifacts/models/*.joblib`, `*.pt`, metrics summary `artifacts/metrics/model_benchmark.json`.
- **Key Operations**:
  - Multi-task training:
    1. Binary Classification ($P(\text{Delay}=1)$): Logistic Regression, LightGBM, CatBoost, XGBoost, PyTorch Tabular Neural Net.
    2. Continuous Magnitude Regression ($E[\text{Delay\_Days}]$): Ridge Regression, LightGBM Regressor, CatBoost Regressor.
  - Comprehensive metric computation: ROC-AUC, PR-AUC, F1, Brier, ECE, MAE, RMSE, Pinball Loss.
- **Quality Gate**:
  - Champion ML model must beat naive baselines (Dummy/Logistic/Ridge) by $\ge 15\%$ on ROC-AUC and MAE.
  - Expected Calibration Error (ECE) $< 0.10$.

---

### Stage 7: Conformal Uncertainty Quantification
- **Module**: `delay_intelligence.uncertainty`
- **Input**: Trained regression/classification models + calibration fold data.
- **Output**: Conformal calibrator `artifacts/uncertainty/conformal_calibrator.joblib`, coverage evaluation `artifacts/metrics/conformal_coverage.json`.
- **Key Operations**:
  - Split Conformal Prediction for regression intervals $[\hat{y}_{\text{lower}}, \hat{y}_{\text{upper}}]$ at $\alpha \in \{0.05, 0.10, 0.20\}$.
  - Conformalized Quantile Regression (CQR) for heteroscedastic, dynamic-width intervals.
  - Least Ambiguous Set-Valued Classifiers (LAC/APS) for classification prediction sets.
- **Quality Gate**:
  - Empirical test coverage must fall within $[1 - \alpha - \epsilon, 1 - \alpha + \epsilon]$ (where $\epsilon \le 0.03$).

---

### Stage 8: Causal Inference & Attribution
- **Module**: `delay_intelligence.causal`
- **Input**: Gold features, Silver data, SCM DAG specification (`configs/causal.yaml`).
- **Output**: Causal effect estimates `artifacts/causal/treatment_effects.json`, TreeSHAP plots `artifacts/reports/shap_summary.png`.
- **Key Operations**:
  - Structural Causal Model (DAG) formulation using NetworkX.
  - Average Treatment Effect (ATE) and Conditional Average Treatment Effect (CATE) estimation via Double Machine Learning (DML).
  - Refutation battery: Random common cause addition, placebo treatment refutation, subset validation.
  - TreeSHAP feature importance and interaction attribution.
- **Quality Gate**:
  - Placebo treatment refutation must produce estimated effect statistically indistinguishable from zero ($p > 0.05$).

---

### Stage 9: Prescriptive Decision Engine
- **Module**: `delay_intelligence.decision`
- **Input**: Model predictions, conformal intervals, CATE estimates, cost parameters (`configs/decision.yaml`).
- **Output**: Policy recommendations, cost-utility curves `artifacts/decision/policy_evaluation.json`.
- **Key Operations**:
  - Asymmetric cost-loss matrix optimization ($p^* = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}}$).
  - Decision threshold optimization minimizing total enterprise supply chain risk.
  - Prescriptive action selection (Standard Monitoring vs Buffer Allocation vs Expedited Air Freight).
- **Quality Gate**:
  - Prescriptive policy must achieve lower total expected operational cost compared to standard reactive policy.

---

### Stage 10: Operationalization & Delivery (API & Dashboard)
- **Module**: `delay_intelligence.api` & `delay_intelligence.dashboard`
- **Input**: Serialized model artifacts, conformal calibrators, and decision engines.
- **Output**: Running FastAPI service (`http://127.0.0.1:8000`) and Streamlit dashboard (`http://127.0.0.1:8501`).
- **Key Operations**:
  - FastAPI endpoints (`/predict`, `/prescribe`, `/health`) with Pydantic request/response validation.
  - Streamlit multi-page interface with interactive shipment inspection and what-if causal simulation.
- **Quality Gate**:
  - API endpoint response latency $< 50$ ms for single-order scoring.
  - 100% test pass on API integration test suite.
