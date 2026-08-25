# Engineering Team Handoff & Maintainer Manual

**Project**: Supply Chain Delay Intelligence Platform  
**Document**: Engineering Handoff, Maintenance Protocols & Codebase Tour  
**Target Audience**: Software Engineers, ML Engineers, MLOps Maintainers, Data Stewards  
**Status**: **HANDOFF APPROVED / REPOSITORY SEALED**  

---

## 1. Welcome to the Delay Intelligence Codebase

Welcome to the **Supply Chain Delay Intelligence System**. This document provides an engineering handoff for developers and maintainers inheriting this repository following the formal closure of Phase 2.

The repository contains a fully tested, hermetically sealed, local-first Python codebase spanning 19 stages and research extensions. It includes **659 automated tests**, **36 cryptographically frozen baseline artifacts**, and a modular package layout under `src/delay_intelligence/`.

---

## 2. Codebase Tour & Module Directory Guide

The codebase is organized into modular Python packages with strict single-responsibility boundaries:

```
delay_intelligence_system/
├── pyproject.toml                   # Packaging specification with granular optional extras
├── configs/                         # Centralized YAML configurations
│   ├── base.yaml                    # Global paths, random seeds, and logging levels
│   ├── data.yaml                    # Data paths, raw dataset pointers (read-only)
│   ├── prediction_contract.yaml     # Machine-readable prediction contract & horizon gates
│   ├── features.yaml                # Feature engineering parameters and lag windows
│   ├── models.yaml                  # Model architectures and hyperparameters
│   ├── uncertainty.yaml             # Conformal prediction alpha levels and split settings
│   ├── drift.yaml                   # E6.5 drift feature tiers, thresholds, and policies
│   ├── adaptive_conformal.yaml      # E7 adaptive window lengths, embargo, and trigger rules
│   ├── cost_scenarios.yaml          # E8 cost scenario models (Low, Base, High)
│   ├── e8_experiments.yaml          # E8 strategy definitions and review budget settings
│   ├── e10_counterfactual.yaml      # E10 policy definitions (P0..P5) and action fees
│   └── serving.yaml                 # FastAPI host/port and Streamlit dashboard settings
├── src/delay_intelligence/          # Core application namespace
│   ├── core/                        # Cross-cutting utilities, logging, config loader
│   ├── data/                        # Ingestion adapters
│   │   └── adapters/                # scms.py, dataco.py, olist.py, base.py
│   ├── validation/                  # Schema integrity & leakage gates
│   │   ├── schemas.py               # Pandera DataFrame schemas and Pydantic models
│   │   └── leakage.py               # Feature availability classifier & temporal boundary gates
│   ├── features/                    # Temporal & logistics feature transformations
│   │   ├── temporal.py              # Cyclical calendar transforms, lead time calculators
│   │   ├── lags.py                  # Historical vendor/country rolling statistics
│   │   └── pipeline.py              # Sklearn-compatible feature pipeline
│   ├── evaluation/                  # Purged rolling-origin evaluation harness
│   │   ├── splitters.py             # Expanding-window temporal splitters with embargo
│   │   └── metrics.py               # PR-AUC, ROC-AUC, Brier score, and Expected Cost calculators
│   ├── models/                      # Predictive model architectures
│   │   ├── catboost_classifier.py   # Stage 5 Champion classifier with isotonic calibration
│   │   ├── lightgbm_regressor.py    # Pinball-loss multi-quantile continuous regressor
│   │   └── logistic_baseline.py     # L2-regularized linear baseline
│   ├── uncertainty/                 # Conformal prediction engine
│   │   └── conformal.py             # Split Conformal Quantile Regression (CQR)
│   ├── explainability/              # Attribution & interpretability
│   │   └── shap_explainer.py        # TreeSHAP explainer with feature stability ranking
│   ├── causal/                      # Causal graph discovery
│   │   └── discovery.py             # Constraint-based PC Algorithm graph generator
│   ├── decision/                    # Prescriptive decisioning & cost matrices
│   │   ├── cost_matrix.py           # Asymmetric loss functions
│   │   └── engine.py                # Prescriptive triage engine
│   ├── drift/                       # Phase 2 — E6.5 Drift Detection Engine
│   │   ├── detector.py              # 4D Chronological Drift Detector (PSI, W_1, KS-FDR)
│   │   ├── metrics.py               # Mathematical formulation of divergence metrics
│   │   ├── policy.py                # 3-tier composite decision policy with Tier-1 SHAP veto
│   │   └── runner.py                # Drift historical evaluation harness
│   ├── adaptive_conformal/          # Phase 2 — E7 Adaptive Conformal Recalibration
│   │   ├── adaptive_cqr.py          # Adaptive CQR calibration manager
│   │   ├── evaluator.py             # CV and single-pass holdout orchestrator
│   │   └── schemas.py               # Pydantic models for adaptive events and intervals
│   ├── cost_sensitive/              # Phase 2 — E8 Instance-Dependent Cost-Sensitive
│   │   ├── cost_engine.py           # CostScenarioModel and instance-dependent loss
│   │   ├── models.py                # Standard, Cost-Weighted, and Bayes-Optimal strategies
│   │   ├── budgeting.py             # Review budget allocator (5%, 10%, 20% capacity)
│   │   └── sensitivity.py           # Multi-point cost sensitivity analyzer
│   ├── digital_twin/                # Phase 2 — E9 Digital Twin & Stress Testing
│   │   ├── generator.py             # Synthetic IoT telemetry generator
│   │   ├── simulator.py             # Discrete-event simulation loop and disruption injector
│   │   └── queue.py                 # Queue pressure surge analyzer
│   ├── counterfactual/              # Phase 2 — E10 Counterfactual Policy Evaluation
│   │   ├── policies.py              # Operational policy implementations (P0 through P5)
│   │   ├── transitions.py           # Deterministic state transition engine
│   │   ├── oracle.py                # Isolated offline Oracle benchmark (AST verified)
│   │   ├── budget.py                # Review budget prioritization engine
│   │   └── evaluator.py             # Counterfactual policy evaluator and regret calculator
│   ├── api/                         # FastAPI REST application
│   │   └── main.py                  # API endpoints (/predict, /uncertainty, /explain, /recommend)
│   └── dashboard/                   # Streamlit Control Tower application
│       ├── app.py                   # Multi-page dashboard entrypoint
│       └── pages/                   # Executive, Triage, Drift, Digital Twin, Technical
├── tests/                           # 659 Automated unit and integration tests
├── artifacts/                       # Serialized models, manifests, evaluation outputs
└── docs/                            # 12 comprehensive architectural and scientific documents
```

---

## 3. Environment & Dependency Management Protocols

### 3.1 Python Runtime Specifications
- **Target Version**: Python `3.11+` (developed and validated on `Python 3.14.5`).
- **Package Management**: Managed via standard `pyproject.toml`. Compatible with both `uv` and standard `pip`.

### 3.2 Activating the Environment
```powershell
# Windows PowerShell
cd delay_intelligence_system
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install all dependencies and extras in editable mode
pip install -e ".[all]"
```

### 3.3 Adding New Dependencies
Do not use global `pip install` without updating `pyproject.toml`. Always declare new dependencies in the appropriate section of `pyproject.toml` (e.g., under `dependencies` or `[project.optional-dependencies]`) and reinstall with `pip install -e .`.

---

## 4. Testing Protocols & Verification Guidelines

### 4.1 Running the Full Test Suite
The repository includes 659 tests across 73 test modules.

```powershell
# Windows PowerShell command with local scratch directory
pytest tests/ --basetemp=scratch/pytest_temp -v
```

> **Windows File Lock Guidance**: On Windows operating systems, standard `AppData\Local\Temp` directories can experience file handle locking during rapid SQLite or Parquet read/write cycles. Always pass `--basetemp=scratch/pytest_temp` when running full test suite sweeps.

### 4.2 Running Specific Test Subsystems
```powershell
# 1. Run Data Ingestion & Leakage Gate Tests
pytest tests/test_scms_adapter.py tests/test_prediction_contract.py tests/test_leakage_gates.py

# 2. Run Modeling & Conformal Tests
pytest tests/test_models.py tests/test_conformal.py tests/test_adaptive_conformal.py

# 3. Run Drift Detection Tests
pytest tests/test_drift_determinism.py tests/test_drift_temporal_safety.py

# 4. Run Cost-Sensitive & Budget Tests
pytest tests/test_e8_cost_engine.py tests/test_e8_budgeting.py tests/test_e8_sensitivity.py

# 5. Run Counterfactual & Oracle Isolation Tests
pytest tests/test_e10_evaluator.py tests/test_adversarial_e10_oracle_isolation.py

# 6. Run API & Dashboard UI Tests
pytest tests/test_api_*.py tests/test_dashboard_*.py
```

---

## 5. Model Registry Governance & Checkpoint Promotion

### 5.1 Model Registry Structure
Production model checkpoints reside in `artifacts/model_registry/v1/`:
- `catboost_champion.cbm`: Serialized Stage 5 CatBoost model.
- `cqr_calibration.json`: Baseline conformal nonconformity quantiles.
- `feature_schema.json`: Expected input schema and feature data types.
- `metadata.json`: Model version, training date, commit hash, and validation metrics.

### 5.2 Promotion Rules for New Model Versions:
To promote a candidate model to production (`v2`):
1. **Temporal Cross-Validation Gate**: The candidate model must demonstrate a statistically significant improvement in $\text{PR-AUC}$ on purged rolling-origin CV ($p < 0.01$).
2. **Economic Cost Gate**: The candidate model must achieve a lower Expected Realized Cost on backtesting across Low, Base, and High cost scenarios.
3. **Leakage Compliance**: Zero use of post-outcome or target-derived features ($t_{\text{pred}} \le t_{\text{event}}$ verified).
4. **Offline Holdout Isolation**: The model must NEVER be tuned against the final holdout period.
5. **Cryptographic Manifest Update**: Generate and commit updated SHA-256 hashes in `artifacts/`.

---

## 6. Extending Adapters for New Datasets

To onboard a new supply chain dataset (e.g., a regional ERP feed):
1. Create a new adapter in `src/delay_intelligence/data/adapters/<new_dataset>.py` inheriting from `BaseIngestionAdapter`.
2. Implement the mandatory interface methods:
   - `load_raw_data() -> pd.DataFrame`
   - `validate_schema(df: pd.DataFrame) -> pd.DataFrame`
   - `standardize_columns(df: pd.DataFrame) -> pd.DataFrame`
   - `extract_milestones(df: pd.DataFrame) -> pd.DataFrame`
3. Add corresponding Pandera DataFrame validation contracts in `src/delay_intelligence/validation/schemas.py`.
4. Create dedicated ingestion and leakage tests in `tests/test_<new_dataset>_adapter.py`.

---

## 7. Mandatory Governance & 4-Tier Data Provenance

All maintainers must preserve the 4-tier data provenance model across all future extensions:
1. `OBSERVED_SCMS_DATA`: Historical ground-truth records.
2. `SYNTHETIC_E9_STATE`: Observable dynamic operational state vectors.
3. `SIMULATED_COUNTERFACTUAL`: Post-action simulated states and trajectories.
4. `SIMULATED_COST`: Synthetic business economic costs computed under explicit parameter models.

**Integrity Mandate**:  
*Never hardcode test assertions, create dummy facades, or present simulated scenario benefits as realized causal claims. Maintain strict human-in-the-loop triage governance.*
