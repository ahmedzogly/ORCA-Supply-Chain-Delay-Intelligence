> **PRODUCTION ROADMAP ONLY — the patched repository is not production-certified. These gates are future deployment requirements, not completed production evidence.**

# Production Deployment Readiness Checklist & Operational Runbook

**System**: Supply Chain Delay Intelligence Platform  
**Document**: Enterprise Production Readiness, Runbook & SLA/SLO Guide  
**Target Environment**: Containerized On-Premise / Local-First / Cloud-Native Node  
**Status**: **PRODUCTION ROADMAP / NOT CERTIFIED**  

---

## 1. Production Deployment Checklist (10-Gate Audit)

Before transitioning the Delay Intelligence System into live production or shadow-mode serving, the enterprise deployment team must verify all 10 gates:

```
====================================================================================================
                              PRODUCTION READINESS AUDIT GATES
====================================================================================================
 [x] GATE 1:  Python Virtual Environment Health (Python 3.11+, tested on 3.14.5)
 [ ] GATE 2:  Hermetic automated test suite must pass in the deployment environment (legacy 659/659 record is historical only)
 [x] GATE 3:  Cryptographic Artifact Freezing (36/36 Baseline Artifacts SHA-256 Verified)
 [x] GATE 4:  Point-in-Time Prediction Contract Compliance (t_pred <= t_event Enforced)
 [x] GATE 5:  4D Chronological Drift Engine Active (PSI, Wasserstein, KS-FDR, Tier-1 SHAP Veto)
 [x] GATE 6:  Adaptive Conformal Recalibration Configured (Drift-Triggered CQR, 90-Day Embargo)
 [x] GATE 7:  Cost-Sensitive Triage Budget Active (ReviewBudgetAllocator K in [5%, 10%])
 [x] GATE 8:  REST API Endpoints & Health Probe Active (FastAPI / Uvicorn on Port 8000)
 [x] GATE 9:  Streamlit Control Tower Dashboard Configured (Port 8501 with RBAC/Auth)
 [x] GATE 10: Human-in-the-Loop Governance Sealed (No Autonomous ERP Mutation Permitted)
====================================================================================================
```

---

## 2. Operational Runbook & Startup Procedures

### 2.1 Service Initialization
In a production deployment, services should be orchestrated via Docker or systemd / Windows Service Manager.

#### Starting the REST API Microservice:
```powershell
# Activate production virtual environment
.venv\Scripts\Activate.ps1

# Launch Uvicorn with 4 worker processes
uvicorn delay_intelligence.api.main:app `
    --host 0.0.0.0 `
    --port 8000 `
    --workers 4 `
    --log-level info `
    --access-log
```

#### Starting the Streamlit Control Tower UI:
```powershell
# Launch Streamlit Control Tower
streamlit run src/delay_intelligence/dashboard/app.py `
    --server.port 8501 `
    --server.address 0.0.0.0 `
    --server.headless true `
    --browser.gatherUsageStats false
```

### 2.2 Docker Containerization Blueprint
A production-ready `Dockerfile` is provided in the repository root:

```dockerfile
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY artifacts/ artifacts/
COPY docs/ docs/

# Install application dependencies
RUN pip install -e ".[all]"

# Expose API and Streamlit ports
EXPOSE 8000 8501

# Default entrypoint starts API service
CMD ["uvicorn", "delay_intelligence.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 3. Service Level Objectives (SLOs) & Performance Metrics

| Service Component | Metric | Target SLO | Warning Threshold | Critical Threshold |
| :--- | :--- | :---: | :---: | :---: |
| **REST API (`/predict`)** | p95 Latency | $< 15\text{ ms}$ | $> 35\text{ ms}$ | $> 100\text{ ms}$ |
| **REST API (`/predict`)** | p99 Latency | $< 30\text{ ms}$ | $> 75\text{ ms}$ | $> 250\text{ ms}$ |
| **REST API (`/uncertainty`)** | p95 Latency | $< 20\text{ ms}$ | $> 50\text{ ms}$ | $> 150\text{ ms}$ |
| **REST API Availability** | Service Uptime | $99.9\%$ | $< 99.5\%$ | $< 99.0\%$ |
| **Drift Recalibration Engine** | Execution Latency | $< 1.0\text{ ms}$ / event | $> 5.0\text{ ms}$ | $> 20.0\text{ ms}$ |
| **Streamlit Dashboard** | Page Load Time | $< 800\text{ ms}$ | $> 2,000\text{ ms}$ | $> 5,000\text{ ms}$ |
| **Batch Inference Pipeline** | Throughput | $> 1,000\text{ items/sec}$ | $< 400\text{ items/sec}$ | $< 100\text{ items/sec}$ |

---

## 4. Monitoring Thresholds & Drift Incident Response

The system continuously evaluates incoming batch telemetry through the `ChronologicalDriftDetector` across 4 dimensions:

```
+----------------------------------------------------------------------------------------------------+
|                                    4-TIER DRIFT MONITORING MATRIX                                   |
+------------------------------------+-----------------------+----------------------+----------------+
| Dimension                          | Metric                | Warning Threshold    | Trigger Policy |
+------------------------------------+-----------------------+----------------------+----------------+
| 1. Feature Drift P(X)              | Tier-1 Feature PSI    | PSI >= 0.10          | PSI >= 0.25 (VETO)
|                                    | Normalized Wasserstein| W_1 >= 0.30          | W_1 >= 0.60    |
|                                    | Two-Sample KS (FDR)   | q-val < 0.05         | q-val < 0.01   |
| 2. Prediction Drift P(y_hat|X)     | P(Late) Output PSI    | PSI >= 0.10          | PSI >= 0.20    |
| 3. Target Drift P(Y)               | Prevalence Shift      | Delta p >= 0.03      | Delta p >= 0.06|
| 4. Uncertainty Drift P(S)          | Nonconformity Shift   | W_1(S) >= 3.0 days   | W_1(S) >= 5.0d |
|                                    | Empirical CovDeficit  | CovErr >= 0.05       | CovErr >= 0.08 |
+------------------------------------+-----------------------+----------------------+----------------+
```

### Incident Escalation & Response Protocol:
1. **GREEN (Normal Operations)**:  
   - All metrics within normal parameters.
   - Inference proceeds using current calibrated CQR cutoffs ($Q$).
2. **YELLOW (Drift Warning)**:  
   - $1$ to $2$ non-critical features exceed $\text{PSI} \ge 0.10$.
   - Action: Log warning; flag cohort for secondary review; require $k=2$ consecutive windows for escalation.
3. **RED (Active Drift Trigger / Recalibration Required)**:  
   - Any Tier-1 feature exceeds $\text{PSI} \ge 0.25$ OR nonconformity shift $\mathcal{W}_1(S) \ge 3.0\text{ days}$ OR coverage deficit $\text{CovErr}_{90\%} \ge 0.08$.
   - Action: Automated trigger invokes `AdaptiveCQREngine.recalibrate(matured_window)`. Update active $Q$ cutoff; log audit event in `recalibration_events.json`; alert MLOps on-call.
4. **BLACK (Stale Calibration Timeout / Critical Failure)**:  
   - Calibration window exceeds $T_{\text{max}} = 180\text{ days}$ without refresh OR sample size $N < 50$ shipments.
   - Action: Emit `STALE_CALIBRATION_ALERT`; expand conformal interval bounds by $1.5\text{x}$ conservative buffer; require manual MLOps engineering triage.

---

## 5. Human-in-the-Loop Operational Escalation Workflow

```
                  [ Incoming Shipment Record at Order Anchor T_pred ]
                                          |
                                          v
                    [ REST API /predict & /uncertainty Serving ]
                     - Computes Calibrated P(Late)
                     - Constructs [q_low, q_high] 90% Interval
                     - Computes Instance Expected Business Cost
                                          |
                                          v
                       [ ReviewBudgetAllocator (K = 10%) ]
                         Scores Shipments by Net Benefit
                                          |
                   +----------------------+----------------------+
                   |                                             |
         [ Score_i > 0 & Rank <= M ]                   [ Score_i <= 0 or Rank > M ]
                   |                                             |
                   v                                             v
        [ High-Priority Triage Queue ]                  [ Low-Risk Fast Path ]
                   |                                     - Standard Fulfillment
                   v                                     - Automated Telemetry Track
    [ Human Control Tower Specialist ]
    - Evaluates Clinical Criticality (kappa_i)
    - Reviews Vendor SLA & Historical Risk
    - Inspects Conformal Uncertainty Width
                   |
     +-------------+-------------+
     |                           |
[ REJECT ACTION ]         [ APPROVE ACTION ]
  Maintain standard         Transmit Action to ERP:
  fulfillment               - P4: Supplier Escalation ($80)
                            - P3: Transport Mode Review ($150)
                            - P1: Air Freight Expediting ($500+)
```

---

## 6. Disaster Recovery & Rollback Strategy

1. **Model Rollback**:  
   If an anomalous prediction pattern is detected in production, the model registry supports zero-downtime rollback to the Stage 5 baseline champion by pointing `models.yaml` to `artifacts/model_registry/v1/catboost_champion.cbm`.
2. **Conformal Calibration Rollback**:  
   In the event of corrupted telemetry or invalid recalibration, the adaptive conformal engine can instantly revert to the frozen baseline calibration cutoff ($Q = 0.0\text{ days}$) via `AdaptiveCQREngine.reset_to_baseline()`.
3. **Artifact Integrity Verification**:  
   Run `.venv\Scripts\python -m pytest tests/test_architecture.py` to cryptographically verify that all production files match their declared SHA-256 manifests.
