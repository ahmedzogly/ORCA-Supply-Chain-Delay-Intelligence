# Technology Decision Record (TDR) — Supply Chain Delay Intelligence System

**Document Version**: 1.0.0  
**Project**: Supply Chain Delay Intelligence System  
**Stage**: Stage 0 (Repository, Architecture & Environment Setup)  
**Status**: APPROVED  

---

## 1. Context & Architectural Overview

The Supply Chain Delay Intelligence System is designed to solve predictive, uncertainty quantification, causal attribution, and prescriptive decision-making challenges for complex supply chain logistics.

The raw data sources comprise 13 CSV files across three directories:
- `scms`: 1 file (~3.61 MB), 33 columns.
- `olist`: 9 relational tables (~120.34 MB).
- `dataco`: 3 files (~182.50 MB, 53 columns).
- **Total volume**: ~306.45 MB.

Because the total dataset volume easily resides within single-node workstation RAM (<1.5 GB in-memory representation), the system strictly adheres to **Python-first, local-first, proportional, and reproducible** engineering practices on Windows.

---

## 2. Technology Decision Matrix

| ID | Decision Area | Adopted Technology | Tier / Activation | Explicitly Rejected Technologies | Primary Justification |
|---|---|---|---|---|---|
| **TDR-001** | Core Architecture Pattern | Modular Monolith (Python Package) | Stage 0 | Microservices, Event Mesh | Low operational complexity, zero network overhead, immediate local testing. |
| **TDR-002** | Runtime & Dependency Tiering | PEP 517/518/621 `pyproject.toml` + pip/venv | Stage 0 Baseline + Granular Extras | Monolithic heavy requirements.txt, Poetry lock-in | Fast lean setup; deferred installation per stage prevents environment bloat. |
| **TDR-003** | Local Execution vs Cloud Infrastructure | Windows-Native Single-Node Execution | Stage 0 | Kubernetes, Cloud VMs, GCP BigQuery, AWS Redshift | Data fits in RAM (~306MB); zero cloud cost, 100% offline reproducibility. |
| **TDR-004** | Data Ingestion & Extensibility | Pluggable Abstract Adapter (`BaseIngestionAdapter`) | Stage 0 (Interface) / Stage 1 (Impl) | Monolithic single-dataset loader | Enables multi-dataset support (SCMS, Olist, DataCo) without refactoring core ML pipeline. |
| **TDR-005** | Intermediate Data Serialization | Apache Parquet (Snappy) + Pandas/PyArrow | Stage 1 (`[data]`) | SQLite, MongoDB, Raw CSVs, Apache Spark | Columnar compression, fast vectorized I/O, typed metadata preservation. |
| **TDR-006** | Data Validation & Quality Gates | Pandera + Pydantic | Stage 2 (`[data]`) | Great Expectations (heavy), Custom regex | Lightweight dataframe-native schema assertions with minimal runtime overhead. |
| **TDR-007** | Modeling & Uncertainty Stack | Scikit-Learn + LightGBM + CatBoost + PyTorch + MAPIE | Stage 6-7 (`[ml]`, `[dl]`, `[uncertainty]`) | H2O AutoML, Cloud AutoML, Bayesian MCMC (PyMC/Stan) | GBDTs achieve state-of-the-art tabular accuracy; split conformal prediction guarantees finite-sample coverage. |
| **TDR-008** | Causal, Prescriptive & Serving Stack | DoWhy + EconML + SciPy + FastAPI + Streamlit | Stage 8-10 (`[causal]`, `[decision]`, `[api]`, `[dashboard]`) | Heavy JavaScript SPAs (React), Apache Airflow, CPLEX/Gurobi | 100% Python data-native stack; low-latency async REST API; interactive what-if dashboard. |

---

## 3. Individual Technology Decision Records

### TDR-001: Core Architecture Pattern — Python-First Modular Monolith
- **Status**: Accepted
- **Context**: The project spans 10 interrelated research and operational stages (ingestion, validation, leakage auditing, features, temporal CV, multi-task ML, uncertainty, causality, prescription, serving).
- **Decision**: Structure the system as a clean Python package (`delay_intelligence`) organized into cohesive functional subpackages under `src/delay_intelligence/`.
- **Rationale**: A modular monolith provides strong encapsulation while eliminating network serialization, service mesh complexity, distributed debugging overhead, and deployment fragility.

---

### TDR-002: Dependency Tiering Strategy
- **Status**: Accepted
- **Context**: Installing the entire universe of machine learning, deep learning, causal inference, and dashboarding libraries during Stage 0 violates minimal dependency constraints and increases setup failure risk.
- **Decision**: Define a lean Stage 0 baseline in `pyproject.toml` containing only essential configuration and packaging utilities (`pyyaml>=6.0`, `setuptools>=61.0.0`). Partition all downstream dependencies into PEP 621 `[project.optional-dependencies]` groups:
  - `[dev]`: `pytest`, `pytest-cov`, `ruff`, `black`, `mypy`
  - `[data]`: `pandas`, `numpy`, `pyarrow`, `pandera`, `pydantic`
  - `[ml]`: `scikit-learn`, `lightgbm`, `catboost`, `xgboost`, `optuna`, `joblib`
  - `[dl]`: `torch`
  - `[uncertainty]`: `mapie`, `scikit-learn`
  - `[causal]`: `dowhy`, `econml`, `networkx`, `shap`, `matplotlib`
  - `[decision]`: `scipy`
  - `[api]`: `fastapi`, `uvicorn`, `pydantic`, `httpx`
  - `[dashboard]`: `streamlit`, `plotly`
  - `[all]`: Unified meta-target
- **Rationale**: Ensures fast Stage 0 setup and testing while providing declarative stage-specific package specifications.

---

### TDR-003: Explicit Rejection of Heavyweight & Unproportional Technologies
- **Status**: Accepted
- **Context**: Avoiding architectural over-engineering is a primary project requirement.
- **Explicitly Rejected Technologies & Rationales**:
  1. **Rejected: Microservices Architecture & Kubernetes**:
     - *Rationale*: A microservices architecture with separate services for feature engineering, training, and inference introduces Docker daemon dependencies, Kubernetes ingress configuration, inter-service gRPC serialization, and distributed failure modes for a dataset that processes in sub-seconds in a single Python process.
  2. **Rejected: Cloud Data Warehouses (Google BigQuery / Snowflake / AWS Redshift)**:
     - *Rationale*: Pushing 306 MB of static CSV files into BigQuery introduces cloud authentication requirements, Google Cloud SDK dependencies, network transfer overhead, and billing complexity. Local Pandas/PyArrow execution processes the entire dataset in under 1 second without internet connectivity.
  3. **Rejected: Heavy Workflow Orchestrators (Apache Airflow / Kubeflow Pipelines)**:
     - *Rationale*: Apache Airflow requires a background scheduler, PostgreSQL metadata database, Redis message queue, and Celery workers, and exhibits poor native Windows ergonomics. A lightweight Python script runner (`run_pipeline.py`) provides deterministic, fully reproducible execution.
  4. **Rejected: Distributed Data Processing (Apache Spark / PySpark / Ray Cluster)**:
     - *Rationale*: PySpark introduces JVM memory overhead, multi-node driver-worker coordination overhead, and slow local startup for datasets under 1 GB. Single-node multi-threaded Pandas/NumPy executes significantly faster on standard multi-core CPUs.
  5. **Rejected: Heavyweight JavaScript Frontend SPAs (React / Next.js / Angular / Vue)**:
     - *Rationale*: Developing a React SPA requires Node.js, npm/yarn, TypeScript, Webpack/Vite build pipelines, and dual-language maintenance. Streamlit provides 100% Python-native reactive interfaces, interactive parameter sliders, and Plotly visualization out-of-the-box.
  6. **Rejected: Vector Databases & Unstructured Semantic Search (Milvus / Pinecone / Qdrant / Weaviate)**:
     - *Rationale*: Supply chain delay intelligence is a structured tabular logistics problem (dates, routes, carriers, quantities, prices, INCO terms). Vector embeddings and nearest-neighbor search are irrelevant to structured tabular classification and regression.

---

### TDR-004: Ingestion Adapter Pattern for Multi-Dataset Extensibility
- **Status**: Accepted
- **Context**: While initial modeling focuses on SCMS, the architecture must support Olist (relational) and DataCo (Latin-1 encoded wide table) without modifying core pipeline logic.
- **Decision**: Define `BaseIngestionAdapter` in `src/delay_intelligence/data/adapters/base.py` enforcing standard methods: `load_raw()`, `standardize_schema()`, `extract_temporal_features()`, `get_dataset_metadata()`.
- **Rationale**: Isolates dataset-specific quirks (e.g. DataCo's latin-1 encoding, Olist's 9 relational joins, SCMS's composite string fields) within self-contained adapter modules.

---

### TDR-005: Intermediate Serialization via Apache Parquet
- **Status**: Accepted
- **Context**: Inter-stage data passing requires fast, schema-preserving, compact storage.
- **Decision**: Standardize on Apache Parquet (Snappy compression) for Bronze, Silver, and Gold data layers in `artifacts/data/`.
- **Rationale**: Parquet enforces strict column data types, preserves nullability, supports fast columnar reads, and reduces disk footprint by 70-80% compared to raw CSVs.

---

### TDR-006: Data Validation via Pandera & Pydantic
- **Status**: Accepted
- **Context**: Data quality and schema drift must be caught prior to feature extraction.
- **Decision**: Use Pandera dataframe schemas for tabular validation and Pydantic models for single-record API schemas.
- **Rationale**: Declarative validation rules, rich error diagnostic outputs, and zero external service dependencies.

---

### TDR-007: Predictive Modeling & Conformal Uncertainty Stack
- **Status**: Accepted
- **Context**: Supply chain delays require accurate point estimates and mathematically valid prediction intervals.
- **Decision**:
  - Point Prediction: LightGBM, CatBoost, XGBoost, Scikit-learn baselines, PyTorch Tabular MLP.
  - Uncertainty: Split Conformal Prediction and Conformalized Quantile Regression (CQR) via MAPIE/custom implementations.
- **Rationale**: Gradient boosted trees consistently achieve superior accuracy on tabular logistics data. Conformal prediction guarantees finite-sample coverage ($1-\alpha$) without assuming Gaussianity or homoscedasticity.

---

### TDR-008: Causal Inference, Prescriptive Optimization & Serving Stack
- **Status**: Accepted
- **Context**: Operations leaders require causal treatment evaluations and prescriptive actions under asymmetric cost structures.
- **Decision**:
  - Causal Inference: Structural Causal Model DAGs (`networkx`), Double Machine Learning (DML via `DoWhy`/`EconML`), and TreeSHAP attribution.
  - Prescriptive Optimization: Asymmetric cost-loss threshold optimization via `scipy.optimize`.
  - Operational Serving: FastAPI ASGI REST service and Streamlit interactive dashboard.
- **Rationale**: Bridges predictive ML with causal decision-making in a unified, 100% Python-native operational stack.
