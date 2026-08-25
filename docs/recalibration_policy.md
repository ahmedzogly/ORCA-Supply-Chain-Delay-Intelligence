# Recalibration Policy and Governance Specification (E7 Design Freeze)

## 1. Executive Summary & Design Freeze Mandate

In production logistics environments, machine learning systems face severe non-stationary environments, macroeconomic disruptions, port congestions, and vendor performance shifts. Under such distribution shifts, standard Conformalized Quantile Regression (CQR) with fixed calibration parameters suffers catastrophic undercoverage, as proven in Stage 12 where nominal 90% coverage collapsed to **22.95%** on the final holdout.

To resolve this failure mode while preventing data leakage, p-hacking, and arbitrary hyperparameter tuning, this document establishes the **authoritative, frozen Recalibration Policy (E7)**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          E7 FINAL HOLDOUT DESIGN FREEZE                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. FROZEN BEFORE HOLDOUT ACCESS: All drift thresholds, statistical tests, window sizes,│
│    calibration buffer lengths (180d), embargo gaps (90d), stale timeouts (180d/1500),   │
│    and cooldown periods (30d/50) are frozen exclusively on Development CV (Folds 0–4). │
│ 2. STRICT CHRONOLOGICAL INVARIANT: Calibration data strictly precedes evaluation data  │
│    with a mandatory 90-day label maturity embargo (max(t_calib) <= t_eval - 90d).      │
│ 3. SINGLE-PASS FORWARD EVALUATION: The 365-day Final Holdout (1,013 rows) is evaluated │
│    exactly once in forward chronological order with NO RETUNING.                       │
│ 4. THREE-WAY COMPARISON PROTOCOL:                                                      │
│    - Strategy A: Static CQR (Frozen baseline control, never recalibrates).              │
│    - Strategy B: Rolling CQR (Periodic scheduled recalibration every 90 days).          │
│    - Strategy C: Drift-Triggered CQR (Dynamic adaptive recalibration via E6.5 policy). │
│ 5. FIRST-CLASS EFFICIENCY METRICS: Recalibration count, frequency (events/year),       │
│    average days between recalibrations, and computational latency overhead.            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Framework for Conformal Quantile Recalibration

### 2.1 Conformalized Quantile Regression (CQR) Formulation

Let $(X_i, Y_i) \in \mathcal{X} \times \mathbb{R}$ represent shipment feature vectors and true continuous delay days ($Y = \text{Delay\_Days}$). Given base quantile regressors $\hat{q}_{\alpha_{lo}}(X)$ and $\hat{q}_{\alpha_{hi}}(X)$ predicting the nominal lower and upper quantiles (where $\alpha_{lo} = \alpha / 2 = 0.05$ and $\alpha_{hi} = 1 - \alpha / 2 = 0.95$ for nominal $1 - \alpha = 0.90$ coverage):

1. **Nonconformity Score Function**:
   For any calibration observation $(X_i, Y_i)$, the signed nonconformity score $S_i$ measures the magnitude of interval violation:
   $$S_i = \max\left( \hat{q}_{\alpha_{lo}}(X_i) - Y_i, \; Y_i - \hat{q}_{\alpha_{hi}}(X_i) \right)$$
   - If $Y_i \in [\hat{q}_{\alpha_{lo}}(X_i), \hat{q}_{\alpha_{hi}}(X_i)]$, then $S_i \le 0$ (point is covered by base quantiles).
   - If $Y_i < \hat{q}_{\alpha_{lo}}(X_i)$, $S_i = \hat{q}_{\alpha_{lo}}(X_i) - Y_i > 0$ (lower bound violation).
   - If $Y_i > \hat{q}_{\alpha_{hi}}(X_i)$, $S_i = Y_i - \hat{q}_{\alpha_{hi}}(X_i) > 0$ (upper bound violation).

2. **Empirical Conformal Quantile Adjustment $Q$**:
   Given a calibration set $\mathcal{D}_{calib} = \{(X_i, Y_i)\}_{i=1}^n$ of size $n$, the empirical conformal quantile adjustment factor $Q$ is computed at the finite-sample adjusted quantile level:
   $$p_{level} = \min\left(1.0, \; (1 - \alpha)\left(1 + \frac{1}{n}\right)\right)$$
   $$Q = \text{Quantile}\left(\{S_i\}_{i=1}^n, \; p_{level}, \; \text{method='higher'}\right)$$

3. **Conformal Prediction Interval**:
   For any incoming test shipment $X_{n+1}$, the calibrated prediction interval is:
   $$\mathcal{C}(X_{n+1}) = \left[ \hat{q}_{\alpha_{lo}}(X_{n+1}) - Q, \; \hat{q}_{\alpha_{hi}}(X_{n+1}) + Q \right]$$
   Under exchangeability, this guarantees exact marginal coverage:
   $$\mathbb{P}\left( Y_{n+1} \in \mathcal{C}(X_{n+1}) \right) \ge 1 - \alpha$$

---

## 3. The Three Recalibration Strategies

### 3.1 Strategy A: Static CQR (Frozen Baseline Control)
- **Mechanics**: Computes $Q_{static}$ once on the initial historical development calibration block and holds $Q$ strictly fixed for all subsequent evaluation steps.
- **Recalibration Frequency**: Exactly 0 recalibrations ($f_{recalib} = 0.0\text{ yr}^{-1}$).
- **Role in Study**: Represents the unmaintained, production-drift baseline (control group).

### 3.2 Strategy B: Rolling CQR (Periodic / Scheduled Recalibration)
- **Mechanics**: Re-estimates $Q_{rolling}$ on a fixed periodic schedule of $\Delta T_{cadence} = 90\text{ calendar days}$.
- **Calibration Window**: At evaluation step $t$, uses the most recent matured calibration shipments spanning:
  $$\mathcal{W}_{calib}(t) = \left\{ i \;\middle|\; T_{pred}(i) \in [t - \Delta T_{calib} - \Delta T_{embargo}, \; t - \Delta T_{embargo}] \right\}$$
  where $\Delta T_{calib} = 180\text{ days}$ and $\Delta T_{embargo} = 90\text{ days}$.
- **Recalibration Frequency**: Constant periodic cadence ($f_{recalib} \approx 4.0\text{ events/year}$).
- **Role in Study**: Represents standard time-based scheduled operational maintenance.

### 3.3 Strategy C: Drift-Triggered CQR (Dynamic Adaptive Recalibration)
- **Mechanics**: Recalibrates dynamically **ONLY** when an actionable trigger is raised by the multi-dimensional `DriftTriggerPolicy` (E6.5) or upon reaching the stale calibration timeout, provided the cooldown period has elapsed.
- **Monitoring Cadence**: Evaluates drift health at sliding intervals of $\Delta T_{eval} = 30\text{ calendar days}$.
- **Calibration Window**: When triggered at time $t$, ingests the matured calibration window $[t - 270\text{d}, t - 90\text{d}]$ ($180\text{d}$ window with $90\text{d}$ embargo).
- **Recalibration Frequency**: Dynamic, load-dependent, event-driven ($f_{recalib} \le 4.0\text{ events/year}$).
- **Role in Study**: Represents the optimal, resource-efficient intelligent monitoring paradigm.

---

## 4. Multi-Dimensional Trigger Logic & Policy Rules

The Drift-Triggered Engine executes dynamic recalibration when any of the following mutually independent policy rules evaluates to `True`:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        COMPOSITE RECALIBRATION TRIGGER RULES                           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ RULE 1 — Tier 1 Feature Veto: Any Tier 1 critical feature exhibits PSI >= 0.25         │
│ RULE 2 — Multiple Tier 1 Warnings: >= 2 Tier 1 features exhibit PSI >= 0.10             │
│ RULE 3 — Widespread Covariate Shift: Weighted feature drift score S_feat >= 1.20       │
│ RULE 4 — Uncertainty Validity Breakdown:                                               │
│          - Empirical coverage deficit CovErr >= 0.08 (Cov < 82%) AND binom_p < 0.01, OR│
│          - Nonconformity distance W_1(S_calib, S_det) >= 3.0 days                      │
│ RULE 5 — Model Prediction Shift: Model output probability PSI(p_hat) >= 0.20 or        │
│          W_1(p_hat) >= 0.10                                                            │
│ RULE 6 — Target Prevalence Shift: Binary prevalence |Delta y_bar| >= 0.07 (p_z < 0.01)  │
│          or Continuous Delay W_1(Y) >= 0.30                                            │
│ RULE 7 — Stale Calibration Timeout: Elapsed duration T_elapsed >= 180 days OR scored   │
│          volume V_elapsed >= 1,500 shipments                                           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Sample Size Power Regularization Guard ($N_{min} = 50$)
Small observation windows ($N_{det} < 50$) lack statistical power and produce erratic PSI/KS estimates.
- If $N_{det} < 50$, the system state is set to `INSUFFICIENT_SAMPLE`.
- Automatic recalibration is **suppressed (`trigger = False`)** to prevent false-alarm chatter.

### 4.2 Recalibration Cooldown Period ($T_{cooldown} = 30\text{ days}$, $N_{cooldown} = 50$)
Following any recalibration event at time $t_{recalib}$, an operational cooldown is enforced:
- For $t < t_{recalib} + 30\text{ days}$ (or volume $< 50$ shipments), `cooldown_active = True`.
- Drift monitoring continues logging diagnostics, but automatic recalibration is suppressed unless an acute safety veto occurs.

### 4.3 Persistence Filtering ($k = 2$)
- Moderate non-veto warnings (YELLOW) require persistence across $k = 2$ consecutive evaluation windows before triggering escalation.
- Acute safety vetos (Tier 1 PSI $\ge 0.25$, severe conformal undercoverage $p_{binom} < 0.01$) trigger immediately ($k = 1$).

---

## 5. First-Class Adaptive Efficiency Metrics

Recalibration efficiency is evaluated using four primary operational metrics:

1. **Recalibration Event Count ($K_{recalib}$)**:
   Total number of discrete recalibration events executed over the evaluation horizon.
   $$K_{recalib} = \sum_{t=1}^T \mathbb{I}(\text{recalibration\_executed}_t)$$

2. **Annualized Recalibration Frequency ($f_{recalib}$)**:
   Number of recalibration events normalized per 365 calendar days:
   $$f_{recalib} = K_{recalib} \times \frac{365.0}{\text{Total Evaluation Days}}$$

3. **Mean Time Between Recalibrations ($\overline{\Delta T}_{MTBR}$)**:
   Average duration (in calendar days) between successive recalibrations:
   $$\overline{\Delta T}_{MTBR} = \frac{1}{K_{recalib}} \sum_{k=1}^{K_{recalib}} (t_k - t_{k-1})$$

4. **Computational Latency Overhead ($\tau_{overhead}$)**:
   Wall-clock execution time (in milliseconds) required to extract matured calibration nonconformity scores and compute the finite-sample quantile $Q$.

---

## 6. Holdout Isolation & Single-Pass Protocol

To preserve scientific validity and guarantee zero test-set contamination:

1. **Isolation Guarantee**: The 365-day Final Holdout dataset (2014-08-24 to 2015-08-24, $N = 1,013$ shipments) is never accessed during policy definition or parameter tuning.
2. **Chronological Streaming**: During holdout evaluation, data is presented in strict chronological sequence.
3. **Matured Calibration Admissibility**: At evaluation date $t$, any calibration update may ONLY ingest shipments with prediction timestamps $T_{pred} \le t - 90\text{ days}$. Shipments in the active 90-day embargo gap $[t - 90\text{d}, t]$ have unknown delivery outcomes and are strictly excluded.
4. **Zero Retuning**: All hyperparameters in `configs/adaptive_conformal.yaml` remain locked throughout execution.

---

## 7. Governance Approval & Audit Sign-Off

- **Policy Status**: FROZEN & AUTHORITATIVE
- **Configuration Path**: `configs/adaptive_conformal.yaml`
- **Applicable Stages**: Phase 2 Part 1 (E7 Adaptive Conformal Recalibration)
- **Approved by**: Teamwork Autonomous Orchestration & QA Challenger
