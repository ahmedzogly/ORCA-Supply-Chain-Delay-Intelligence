# Repository Structure Specification — Supply Chain Delay Intelligence System

This document outlines the complete directory layout, module responsibilities, and architectural boundaries of the `delay_intelligence_system` repository.

---

## 1. Directory Tree Overview

```
delay_intelligence_system/
├── ARCHITECTURE.md                  # Comprehensive system architecture & pipeline blueprint
├── README.md                        # Project overview, quickstart, stage guide, verification
├── pyproject.toml                   # PEP 517/518/621 configuration with deferred extras
├── .gitignore                       # Git ignore rules for artifacts, cache, and venvs
├── configs/                         # Centralized YAML configuration files
│   ├── base.yaml                    # Global paths, random seeds, logging configurations
│   ├── data.yaml                    # Data paths, raw source pointers (read-only), adapter configs
│   ├── validation.yaml              # Pandera validation rules, missingness thresholds
│   ├── features.yaml                # Feature engineering parameters, encodings, horizon rules
│   ├── models.yaml                  # Model architectures, hyperparameters, CV fold settings
│   ├── uncertainty.yaml             # Conformal prediction alpha levels, calibration methods
│   ├── causal.yaml                  # DAG specifications, treatment/outcome pairs, DML settings
│   ├── decision.yaml                # Cost-utility matrices, asymmetric penalty parameters
│   └── serving.yaml                 # FastAPI host/port, Streamlit dashboard settings
├── docs/                            # Formal architecture and technical documentation
│   ├── technology_decision_record.md# Complete formal TDR document (TDR-001 to TDR-008)
│   ├── repository_structure.md      # Detailed directory layout and module responsibilities
│   ├── pipeline_specification.md    # In-depth 10-stage pipeline data contracts & quality gates
│   └── data_dictionary.md           # Unified schema definitions for SCMS, Olist, DataCo
├── src/                             # Core Python package source code
│   └── delay_intelligence/          # Main package namespace
│       ├── __init__.py              # Package version and top-level exports
│       ├── core/                    # Cross-cutting utilities, logging, config loader
│       │   ├── __init__.py
│       │   ├── config.py            # Strongly-typed configuration parser
│       │   ├── exceptions.py        # Custom domain exception hierarchy
│       │   └── logging.py           # Structured logging configuration
│       ├── data/                    # Stage 1: Raw data ingestion & adapters
│       │   ├── __init__.py
│       │   ├── adapters/            # Pluggable dataset adapters
│       │   │   ├── __init__.py
│       │   │   ├── base.py          # BaseIngestionAdapter ABC
│       │   │   ├── scms.py          # SCMS dataset adapter (Stage 1)
│       │   │   ├── olist.py         # Olist relational dataset adapter (Future)
│       │   │   └── dataco.py        # DataCo dataset adapter (Future)
│       │   └── loader.py            # Unified raw data loader & parquet staging
│       ├── validation/              # Stage 2 & 3: Schema validation & Leakage audit
│       │   ├── __init__.py
│       ├── features/                # Stage 4: Feature engineering & transformers
│       │   ├── __init__.py
│       ├── evaluation/              # Stage 5: Evaluation & splitters
│       │   ├── __init__.py
│       ├── models/                  # Stage 6: Multi-task modeling engine
│       │   ├── __init__.py
│       ├── uncertainty/             # Stage 7: Uncertainty & conformal prediction
│       │   ├── __init__.py
│       ├── causal/                  # Stage 8: Causal inference & attribution
│       │   ├── __init__.py
│       ├── decision/                # Stage 9: Prescriptive decision engine
│       │   ├── __init__.py
│       ├── api/                     # Stage 10: REST API Service
│       │   ├── __init__.py
│       └── dashboard/               # Stage 10: Interactive Streamlit UI
│           ├── __init__.py
├── tests/                           # Automated test suite
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures and mock dataset generators
│   ├── test_environment.py          # Stage 0: Environment & dependency validation tests
│   ├── test_config.py               # Stage 0: Configuration schema & loading tests
│   ├── test_architecture.py         # Stage 0: Module import & architecture invariant tests
│   └── test_data_immutability.py    # Stage 0: Read-only verification of raw data sources
├── artifacts/                       # Generated outputs and serialized models (.gitignore managed)
│   ├── data/                        # Bronze/Silver/Gold parquet tables
│   ├── models/                      # Trained model weights and joblib serializers
│   ├── uncertainty/                 # Conformal calibration artifacts
│   ├── causal/                      # Causal graphs, DML estimates, SHAP values
│   ├── metrics/                     # Evaluation metrics JSONs and plots
│   └── reports/                     # Generated stage validation reports
└── models/                          # Production-ready model checkpoints
```

---

## 2. Module Responsibilities

### `src/delay_intelligence/core/`
- **`config.py`**: Locates and parses YAML configuration files safely, provides strongly-typed dictionary access, and resolves absolute paths for raw data sources.
- **`logging.py`**: Configures hierarchical, structured logging for console stdout and optional file logs.
- **`exceptions.py`**: Defines domain-specific exception classes (`ConfigurationError`, `DataValidationError`, `LeakageViolationError`, `DataImmutabilityError`, etc.).

### `src/delay_intelligence/data/`
- **`adapters/base.py`**: Defines the `BaseIngestionAdapter` abstract base class with methods `load_raw()`, `standardize_schema()`, `extract_temporal_features()`, and `get_dataset_metadata()`.
- **`adapters/scms.py`**: Concrete adapter for SCMS supply chain dataset, handling composite string fields and date formats.
- **`adapters/olist.py`**: Concrete adapter for Olist e-commerce dataset, joining 9 relational tables.
- **`adapters/dataco.py`**: Concrete adapter for DataCo global supply chain dataset, managing Latin-1 character encodings.

### `src/delay_intelligence/validation/`
- Enforces Pandera schemas on Bronze data to create Silver data.
- Implements strict temporal horizon auditing, rejecting features generated after the prediction cutoff date.

### `src/delay_intelligence/features/`
- Generates cyclical temporal encodings, lead time features, route/carrier encodings, and numerical scalers.
- Assembles Scikit-learn `ColumnTransformer` pipelines with zero out-of-fold data leakage.

### `src/delay_intelligence/evaluation/`
- Implements `PurgedRollingOriginSplitter` for temporal cross-validation with purge and embargo windows.
- Calculates comprehensive classification and regression evaluation metrics.

### `src/delay_intelligence/models/`
- Houses baseline models (Logistic Regression, Ridge) and advanced models (LightGBM, CatBoost, XGBoost, PyTorch Tabular Neural Networks).
- Implements unified model training and serialization orchestrators.

### `src/delay_intelligence/uncertainty/`
- Implements Split Conformal Prediction and Conformalized Quantile Regression (CQR).
- Evaluates empirical coverage, interval sharpness, and conditional coverage gaps.

### `src/delay_intelligence/causal/`
- Builds Structural Causal Model (SCM) DAGs using NetworkX.
- Estimates Average Treatment Effects (ATE) and Conditional Average Treatment Effects (CATE) using Double Machine Learning (DML).
- Executes placebo and common cause refutation tests and TreeSHAP attribution.

### `src/delay_intelligence/decision/`
- Evaluates asymmetric cost-loss matrices.
- Computes optimal classification decision thresholds.
- Recommends prescriptive interventions (standard monitoring, buffer inventory, expedited air freight).

### `src/delay_intelligence/api/` & `dashboard/`
- **`api/`**: Exposes FastAPI endpoints for real-time delay inference, conformal interval generation, and prescriptive action recommendations.
- **`dashboard/`**: Provides an interactive Streamlit UI for shipment risk monitoring and what-if causal simulation.

---

## 3. Configuration Management

Configurations in `configs/` are organized hierarchically:
- `base.yaml`: Global runtime settings, seeds, compute parameters.
- `data.yaml`: Raw dataset paths (relative to workspace), column mappings, storage formats.
- `validation.yaml`: Schema rules, nullness thresholds, leakage check anchors.
- `features.yaml`: Transformer parameters, encodings, lag windows.
- `models.yaml`: Cross-validation hyperparameters and model architectures.
- `uncertainty.yaml`: Conformal alpha levels and calibration splits.
- `causal.yaml`: SCM DAG edges, treatment variables, refutation suites.
- `decision.yaml`: Cost-loss matrices and mitigation policies.
- `serving.yaml`: Host, port, and UI configuration parameters.

---

## 4. Test Suite Structure

The test suite in `tests/` mirrors the modular architecture:
- `test_environment.py`: Validates Python 3.10+ runtime, Windows OS detection, and baseline package availability.
- `test_config.py`: Validates YAML syntax, required keys, and raw data path resolution.
- `test_architecture.py`: Validates package imports, directory presence, and `BaseIngestionAdapter` contracts.
- `test_data_immutability.py`: Validates read-only access and verifies zero file modification in `scms/`, `olist/`, and `dataco/`.
