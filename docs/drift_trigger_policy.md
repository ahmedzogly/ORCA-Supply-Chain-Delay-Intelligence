# Drift Trigger Policy and Governance Specification (E6.5)

## 1. Governance Framework & Development-Only Calibration

To prevent data leakage, overfitting, and threshold p-hacking, all statistical thresholds and decision rules in the Drift Detection System are calibrated exclusively on the **Development Period Cross-Validation Folds (2006-04-19 to 2014-08-24)**.

### Governance Mandates:
1. **Holdout Quarantine**: The 365-day Final Holdout (2014-08-24 to 2015-08-24, $N = 1,013$ shipments) is completely masked from all development drift thresholding and hyperparameter selection.
2. **Deterministic Reproducibility**: All decision logic is deterministic, stateless across independent evaluations, and machine-verifiable.
3. **Automated Handoff to E7**: The output of the trigger policy (`trigger_recalibration == True`) serves as the direct programmatic signal for the **Adaptive Conformal Recalibration (E7)** engine.

---

## 2. Multi-Metric Decision Matrix & State Machine

```
                              ┌─────────────────────────────┐
                              │    DETECTION WINDOW INPUT   │
                              └──────────────┬──────────────┘
                                             │
                                   [ N_det < N_min (50) ? ]
                                   ├── YES ──► STATUS: INSUFFICIENT_SAMPLE (Suppress Trigger)
                                   └── NO
                                             │
                        ┌────────────────────┴────────────────────┐
                        │                                         │
          [ Stale Calibration: T >= 180d ? ]             [ Critical Shifts Present? ]
          ├── YES ──► STATUS: RED (Trigger: STALE)       ├── YES ──► STATUS: RED (Trigger: DRIFT)
          └── NO                                         └── NO
                                                                   │
                                                      [ Moderate Shifts Present? ]
                                                      ├── YES ──► STATUS: YELLOW (Log Warning)
                                                      └── NO  ──► STATUS: GREEN (Normal Operation)
```

---

## 3. Comprehensive Threshold & Metric Matrix

| Dimension | Monitored Metric | GREEN (Normal) | YELLOW (Warning) | RED (Critical / Trigger) |
|---|---|---|---|---|
| **Feature Drift** | $\text{PSI}(\text{Tier 1 Features})$ | $< 0.10$ | $0.10 \le \text{PSI} < 0.25$ | $\ge 0.25$ (**Veto**) |
| **Feature Drift** | $\widetilde{\mathcal{W}}_{1}(\text{Tier 1 Features})$ | $< 0.15$ | $0.15 \le \widetilde{\mathcal{W}}_1 < 0.30$ | $\ge 0.30$ |
| **Feature Drift** | $\text{JSDist}(\text{Categorical})$ | $< 0.10$ | $0.10 \le \text{JSDist} < 0.20$ | $\ge 0.20$ |
| **Feature Drift** | Weighted Score $S_{feat}$ | $< 0.60$ | $0.60 \le S_{feat} < 1.20$ | $\ge 1.20$ |
| **Prediction Drift**| $\text{PSI}(\hat{p}_{\text{CatBoost}})$ | $< 0.10$ | $0.10 \le \text{PSI} < 0.20$ | $\ge 0.20$ |
| **Prediction Drift**| $\mathcal{W}_1(\hat{p}_{\text{CatBoost}})$ | $< 0.05$ | $0.05 \le \mathcal{W}_1 < 0.10$ | $\ge 0.10$ |
| **Prediction Drift**| $\text{PSI}(\hat{y}_{\text{LightGBM}})$ | $< 0.10$ | $0.10 \le \text{PSI} < 0.25$ | $\ge 0.25$ |
| **Target Drift** | Prevalence Delta $|\Delta \bar{y}|$ | $< 0.03$ | $0.03 \le |\Delta \bar{y}| < 0.07$ | $\ge 0.07$ (and $p_z < 0.01$) |
| **Target Drift** | $\widetilde{\mathcal{W}}_1(\text{Delay\_Days})$ | $< 0.15$ | $0.15 \le \widetilde{\mathcal{W}}_1 < 0.30$ | $\ge 0.30$ |
| **Uncertainty Drift**| Coverage Error $\text{CovErr}_{90\%}$ | $< 0.04$ | $0.04 \le \text{CovErr} < 0.08$ | $\ge 0.08$ (and $p_{binom} < 0.01$) |
| **Uncertainty Drift**| Nonconformity $\mathcal{W}_1(S_{calib}, S_{det})$ | $< 1.5\text{ days}$ | $1.5 \le \mathcal{W}_1 < 3.0\text{ days}$ | $\ge 3.0\text{ days}$ |
| **Uncertainty Drift**| Interval Expansion Ratio $R_w$ | $< 1.25$ | $\ge 1.25$ | $\ge 1.50$ |

---

## 4. Composite Trigger Decision Logic

A **Tier 2 (RED) Drift Recalibration Trigger** is raised if **ANY** of the following composite conditions is met:

1. **Rule 1 — Tier 1 Feature Veto**: Any Tier 1 critical feature exhibits $\text{PSI} \ge 0.25$.
2. **Rule 2 — Multiple Tier 1 Warnings**: At least two Tier 1 features exhibit $\text{PSI} \ge 0.10$.
3. **Rule 3 — Widespread Covariate Shift**: The composite weighted score $S_{feat} \ge 1.20$.
4. **Rule 4 — Uncertainty Validity Breakdown**:
   - Empirical coverage drops by $> 8\%$ ($\text{CovErr} \ge 0.08$, meaning empirical coverage $< 82\%$) **AND** the exact one-sided binomial test is rejected ($p_{binom} < 0.01$), **OR**
   - Nonconformity distribution distance $\mathcal{W}_1(S_{calib}, S_{det}) \ge 3.0 \text{ days}$.
5. **Rule 5 — Severe Prediction Shift**: Model output probability $\text{PSI}(\hat{p}) \ge 0.20$ or $\mathcal{W}_1(\hat{p}) \ge 0.10$.
6. **Rule 6 — Severe Target / Prevalence Shift**: Target prevalence shift $|\Delta \bar{y}| \ge 0.07$ with $p_z < 0.01$, or continuous delay shift $\widetilde{\mathcal{W}}_1(\text{Delay\_Days}) \ge 0.30$.
7. **Rule 7 — Stale Calibration Timeout**: Elapsed duration since last calibration $T_{elapsed} \ge 180 \text{ days}$ or scored volume $V_{elapsed} \ge 1,500 \text{ shipments}$.

---

## 5. Sample Size Rules & Power Regularization

Statistical tests on small samples ($N < 50$) suffer from low test power and high variance in bin frequencies.
- **Hard Minimum Sample Size**: $N_{min} = 50 \text{ shipments}$.
- **Small Batch Policy**: When $N_{det} < N_{min}$:
  1. System status is marked `INSUFFICIENT_SAMPLE`.
  2. Automatic `trigger_recalibration` is **suppressed (`False`)** to prevent false alarms.
  3. Diagnostic metrics are logged, and samples accumulate in a sliding rolling buffer until $N \ge N_{min}$.

---

## 6. Cooldown & Persistence Filtering

### 6.1 Recalibration Cooldown Period ($T_{cooldown}$)
To prevent trigger oscillation / chattering during protracted supply chain disruptions:
- A minimum cooldown of **$T_{cooldown} = 30 \text{ calendar days}$** (or **$N_{cooldown} = 50 \text{ shipments}$**) is enforced following any recalibration event before another automatic drift trigger can fire.
- When cooldown is active, `cooldown_active = True` and automatic triggering is suppressed while maintaining diagnostic status visibility.

### 6.2 Persistence Confirmation ($k = 2$)
- Moderate non-veto warnings require persistence across **$k = 2$ consecutive detection windows** before escalating to an actionable alert.
- Acute veto violations (Tier 1 PSI $\ge 0.25$, severe conformal undercoverage $p_{binom} < 0.01$) bypass persistence and trigger immediately ($k = 1$).

---

## 7. Operational Escalation & E7 Integration Protocol

When a RED trigger is confirmed:
1. **Trigger Serialization**: An immutable JSON payload is emitted to `artifacts/drift/drift_triggers.json`.
2. **Adaptive Recalibration Handoff**: The trigger payload activates the **Adaptive Conformal Recalibration Engine (E7)**, which ingests the most recent validated sliding training block and updates quantile adjustment factors $Q$.
3. **Audit Log Generation**: All trigger reasons, feature metric snapshots, and timestamp bounds are permanently recorded for supply chain auditability.
