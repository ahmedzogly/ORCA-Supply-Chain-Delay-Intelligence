# Adaptive Conformal Recalibration Methodology (E7)

## 1. Introduction and Theoretical Motivation

In mission-critical logistics, knowing *when* and *by how much* a shipment will be delayed requires rigorous prediction intervals. Standard quantile regression produces conditional interval estimates $[\hat{q}_{\alpha/2}(X), \hat{q}_{1 - \alpha/2}(X)]$, but offers no finite-sample coverage guarantees. Conformalized Quantile Regression (CQR) bridges this gap by learning an empirical adjustment factor $Q$ on a calibration set, providing distribution-free finite-sample marginal coverage:

$$\mathbb{P}\left( Y_{n+1} \in \mathcal{C}(X_{n+1}) \right) \ge 1 - \alpha$$

However, the validity of classical conformal prediction rests strictly upon the **exchangeability** of calibration and test data points:

$$(X_1, Y_1), \dots, (X_n, Y_n), (X_{n+1}, Y_{n+1}) \sim_{\text{i.i.d.}} P(X, Y)$$

In global supply chain logistics, exchangeability is routinely violated by temporal non-stationarity, vendor contract shifts, macroeconomic volatility, and transportation modality changes. Under such shifts, static calibration factors $Q_{static}$ fail to adapt, causing severe coverage collapse (as demonstrated in Stage 12, where coverage fell from $89.3\%$ to $22.95\%$).

The **Adaptive Conformal Recalibration (E7)** framework resolves this fundamental vulnerability by introducing chronological rolling and drift-triggered recalibration mechanisms governed by strict temporal safety constraints.

---

## 2. Mathematical Formulations

### 2.1 Base Quantile Predictions & Signed Nonconformity
Let $X \in \mathbb{R}^d$ denote the $d$-dimensional feature vector, and $Y = \text{Delay\_Days} \in \mathbb{R}$ denote the continuous delay outcome. Given base quantile estimators $\hat{q}_{\alpha_{lo}}(X)$ and $\hat{q}_{\alpha_{hi}}(X)$ with $\alpha_{lo} = \alpha / 2$ and $\alpha_{hi} = 1 - \alpha / 2$:

The signed nonconformity score $S_i$ for an observation $(X_i, Y_i)$ is defined as:
$$S_i = \max\left( \hat{q}_{\alpha_{lo}}(X_i) - Y_i, \; Y_i - \hat{q}_{\alpha_{hi}}(X_i) \right)$$

Interpretation:
- $S_i \le 0 \iff Y_i \in [\hat{q}_{\alpha_{lo}}(X_i), \hat{q}_{\alpha_{hi}}(X_i)]$ (Base quantile interval covers $Y_i$).
- $S_i > 0 \iff Y_i$ falls outside the base quantile interval by distance $S_i$.

### 2.2 Finite-Sample Adjusted Quantile Factor
For an admissible calibration sample $\mathcal{D}_{calib} = \{(X_i, Y_i)\}_{i=1}^n$ of size $n$, the empirical conformal quantile adjustment factor $Q$ is computed as:
$$p_{level} = \min\left( 1.0, \; (1 - \alpha)\left(1 + \frac{1}{n}\right) \right)$$
$$Q = \text{Quantile}\left( \{S_1, \dots, S_n\}, \; p_{level}, \; \text{method='higher'} \right)$$

The calibrated prediction interval for any incoming shipment $X_{eval}$ is:
$$\mathcal{C}(X_{eval}) = \left[ \hat{q}_{\alpha_{lo}}(X_{eval}) - Q, \; \hat{q}_{\alpha_{hi}}(X_{eval}) + Q \right]$$

The interval width is:
$$W(X_{eval}) = \left( \hat{q}_{\alpha_{hi}}(X_{eval}) - \hat{q}_{\alpha_{lo}}(X_{eval}) \right) + 2Q$$

---

## 3. The Three Recalibration Strategies

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THREE RECALIBRATION STRATEGIES (E7)                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Strategy A: Static CQR (Control)                                                       │
│   - Initial Q_static learned on development baseline.                                  │
│   - Q is immutable across all future evaluation horizons.                              │
│   - Recalibrations: 0 | Latency Overhead: 0.0 ms.                                      │
│                                                                                        │
│ Strategy B: Rolling CQR (Scheduled / Periodic)                                         │
│   - Re-estimates Q_rolling every Delta T_cadence = 90 calendar days.                   │
│   - Ingests matured sliding calibration buffer [t - 270d, t - 90d] (180d window).      │
│   - Recalibrations: Periodic (~3–4/year) | Latency: Sub-millisecond.                   │
│                                                                                        │
│ Strategy C: Drift-Triggered CQR (Dynamic Adaptive)                                     │
│   - Monitors 4 drift dimensions every Delta T_eval = 30 calendar days.                 │
│   - Evaluates E6.5 DriftTriggerPolicy (Tier 1 SHAP veto, CovErr >= 0.08, stale timeout)│
│   - Recalibrates ONLY when triggered and cooldown (30d / 50 shipments) is elapsed.     │
│   - Ingests matured sliding calibration buffer [t - 270d, t - 90d].                    │
│   - Recalibrations: Event-driven (~3–4/year) | Latency: Sub-millisecond.               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Strategy A: Static CQR (Frozen Baseline Control)
Strategy A acts as the negative control. It freezes $Q_{static}$ from the initial development calibration set and applies it indefinitely. Under stationary conditions, Strategy A achieves nominal coverage. Under temporal distribution shift, its coverage degrades monotonically with the divergence between calibration and evaluation distributions.

### 3.2 Strategy B: Rolling CQR (Periodic / Scheduled Sliding Window)
Strategy B recalibrates at fixed periodic intervals $\Delta T_{cadence} = 90\text{ calendar days}$.
At evaluation timestamp $t$, the admissible calibration window $\mathcal{W}_{calib}(t)$ is extracted:
$$\mathcal{W}_{calib}(t) = \left\{ (X_i, Y_i) \;\middle|\; T_{pred}(i) \in [t - \Delta T_{calib} - \Delta T_{embargo}, \; t - \Delta T_{embargo}] \right\}$$
where $\Delta T_{calib} = 180\text{ days}$ and $\Delta T_{embargo} = 90\text{ days}$.

The adjustment factor $Q_{rolling}$ is updated to the empirical quantile of nonconformity scores in $\mathcal{W}_{calib}(t)$ and remains active until $t + \Delta T_{cadence}$.

### 3.3 Strategy C: Drift-Triggered CQR (Dynamic Adaptive Recalibration)
Strategy C combines the 4-dimensional statistical drift engine (E6.5) with adaptive conformal updating. Every $\Delta T_{eval} = 30\text{ calendar days}$, the system evaluates:

1. **Feature Drift $P(X)$**: Laplace-smoothed PSI and normalized Wasserstein distance across 26 numerical and 13 categorical features.
2. **Prediction Drift $P(\hat{Y}|X)$**: Output probability PSI $\text{PSI}(\hat{p})$ and point prediction shifts.
3. **Target Drift $P(Y)$**: Late delivery prevalence shift $|\Delta \bar{y}|$ (two-proportion $z$-test) and delay days Wasserstein distance.
4. **Uncertainty Drift $P(S)$**: Nonconformity distribution distance $\mathcal{W}_1(S_{calib}, S_{det})$, empirical coverage deficit $\text{CovErr}_{90\%}$, and exact one-sided binomial undercoverage test ($p_{binom} < 0.01$).
5. **Stale Calibration Timeout**: Elapsed days $T_{elapsed} \ge 180\text{ days}$ or volume $V_{elapsed} \ge 1,500\text{ shipments}$.

If `trigger_recalibration == True` and cooldown is inactive ($t \ge t_{last\_recalib} + 30\text{d}$), the engine ingests $\mathcal{W}_{calib}(t) = [t - 270\text{d}, t - 90\text{d}]$ and re-estimates $Q_{drift}$.

---

## 4. Temporal Safety & Label Maturity Embargo

In supply chain logistics, delivery outcomes are not observed instantaneously at prediction time $T_{pred}$. Instead, delivery occurs after transit lead times (e.g., 30–90 days).

```
   T_pred - 270d           T_pred - 90d                     T_pred (Now)
        │                       │                                │
        ├───────────────────────┼────────────────────────────────┤
        │  MATURED CALIBRATION  │   ACTIVE EMBARGO (NO LABELS)   │
        │     WINDOW (180d)     │          BUFFER (90d)          │
        │    (Labels Matured)   │      (In Transit / Unknown)    │
        └───────────────────────┴────────────────────────────────┘
```

### Invariant Equations:
1. **Past $\rightarrow$ Future Strict Ordering**:
   $$\forall (X_{calib}, Y_{calib}) \in \mathcal{W}_{calib}(t), \quad T_{pred}(X_{calib}) < T_{pred}(X_{eval}) - \Delta T_{embargo}$$
2. **Embargo Compliance**:
   $$\max_{(X \in \mathcal{W}_{calib}(t))} T_{pred}(X) \le t - 90\text{ days}$$
3. **Holdout Quarantine**:
   $$\mathcal{W}_{calib}(t) \cap \mathcal{D}_{holdout} = \emptyset \quad \forall t \le t_{holdout\_start}$$

---

## 5. Statistical and Efficiency Metrics Formulations

Let $\mathcal{D}_{eval} = \{(X_j, Y_j)\}_{j=1}^N$ be the evaluation dataset spanning duration $\Delta T_{eval\_total}$ days.

1. **Empirical Coverage ($\text{Cov}_{90\%}$)**:
   $$\text{Cov}_{90\%} = \frac{1}{N} \sum_{j=1}^N \mathbb{I}\left( Y_j \in \left[ \hat{q}_{\alpha/2}(X_j) - Q_j, \; \hat{q}_{1 - \alpha/2}(X_j) + Q_j \right] \right)$$

2. **Coverage Error ($\text{CovErr}$)**:
   $$\text{CovErr} = (1 - \alpha) - \text{Cov}_{90\%}$$
   - $\text{CovErr} \le 0 \implies$ Target coverage met or exceeded (Valid).
   - $\text{CovErr} > 0 \implies$ Undercoverage (Invalid).

3. **Mean Interval Width ($\overline{W}$)**:
   $$\overline{W} = \frac{1}{N} \sum_{j=1}^N W(X_j)$$

4. **Recalibration Event Count ($K_{recalib}$)**:
   Total count of discrete parameter updates executed during the evaluation horizon.

5. **Annualized Recalibration Frequency ($f_{recalib}$)**:
   $$f_{recalib} = K_{recalib} \times \frac{365.0}{\Delta T_{eval\_total}} \quad [\text{events/year}]$$

6. **Mean Time Between Recalibrations ($\overline{\Delta T}_{MTBR}$)**:
   $$\overline{\Delta T}_{MTBR} = \frac{\Delta T_{eval\_total}}{\max(1, K_{recalib})} \quad [\text{days}]$$

7. **Computational Latency Overhead ($\tau_{overhead}$)**:
   $$\tau_{overhead} = \sum_{k=1}^{K_{recalib}} \tau_k \quad [\text{ms}]$$
   where $\tau_k = t_{\text{end\_fit}} - t_{\text{start\_fit}}$ measured via high-precision hardware timers (`time.perf_counter()`).
