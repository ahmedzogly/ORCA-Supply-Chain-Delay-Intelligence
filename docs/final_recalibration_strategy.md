# Operational Guide: Drift-Triggered Adaptive Conformal Recalibration

**System**: Supply Chain Delay Intelligence Platform  
**Module**: Adaptive Conformal Recalibration Engine (`src/delay_intelligence/adaptive_conformal/`)  
**Experiment Reference**: Phase 2 — Experiments E6.5 & E7  
**Audience**: MLOps Engineers, Quantitative Researchers, Production Support  
**Status**: **OPERATIONAL RUNBOOK / ALGORITHMIC SPECIFICATION**  

---

## 1. Executive Summary & Operational Rationale

In static machine learning systems, uncertainty intervals are calibrated once during offline model training. However, as demonstrated in Stage 12 of this project, static Conformal Quantile Regression (CQR) bounds experience catastrophic coverage collapse under real-world non-stationary macro distribution shifts—dropping from **$89.3\%$ empirical coverage in Development CV to $22.95\%$ on the final holdout** (coverage error $+0.6705$).

To eliminate this vulnerability without introducing continuous, high-latency model retraining, the platform implements **Strategy C: Drift-Triggered Adaptive Conformal Recalibration**. This strategy couples a 4-dimensional chronological drift detector with an event-driven CQR calibration manager, restoring empirical coverage to **$93.88\%$** with only **4 discrete recalibration events per year** and negligible annual compute overhead (**$0.512\text{ ms}$ total**).

---

## 2. Mathematical Architecture of Adaptive CQR

Given pinball-loss quantile estimators $\hat{q}_{\alpha/2}(X), \hat{q}_{1 - \alpha/2}(X)$ for nominal coverage $1 - \alpha = 0.90$:

```
                                  [ Incoming Operational Shipments ]
                                                   |
                                                   v
                         [ Chronological Drift Detector (4 Dimensions) ]
                         - Feature Drift PSI & Wasserstein W_1
                         - Nonconformity Score Shift W_1(S_calib, S_det)
                         - Empirical Coverage Deficit CovErr_90%
                                                   |
                        +--------------------------+--------------------------+
                        |                                                     |
               [ NO DRIFT (Green) ]                                  [ DRIFT TRIGGERED (Red) ]
                        |                                                     |
                        v                                                     v
            Maintain Active Cutoff Q                              [ Embargoed Window Ingestion ]
                                                                  W_calib = [t - 270d, t - 90d]
                                                                  N >= 50 matured shipments
                                                                              |
                                                                              v
                                                                  [ Finite-Sample Quantile Re-eval ]
                                                                  p_level = 0.90 * (1 + 1/n)
                                                                  Q_new = Quantile({S_i}, p_level)
                                                                              |
                                                                              v
                                                                  [ Update Serving Cutoff Q ]
                                                                  C(X) = [q_0.05 - Q, q_0.95 + Q]
```

### 2.1 Finite-Sample Conformal Correction Formula
For calibration sample size $n$ drawn from the matured calibration window $\mathcal{W}_{\text{calib}}$:
$$S_i = \max\left( \hat{q}_{0.05}(X_i) - Y_i, \; Y_i - \hat{q}_{0.95}(X_i) \right)$$
$$p_{\text{level}} = \min\left(1.0, \; 0.90 \times \left(1 + \frac{1}{n}\right)\right)$$
$$Q = \text{Quantile}\left(\{S_i\}_{i=1}^n, \; p_{\text{level}}, \; \text{method='higher'}\right)$$
$$\mathcal{C}_{90\%}(X) = \left[ \hat{q}_{0.05}(X) - Q, \; \hat{q}_{0.95}(X) + Q \right]$$

---

## 3. Operational Policy Parameters & Freeze Specifications

All policy rules, window horizons, and trigger thresholds are frozen in `configs/adaptive_conformal.yaml`:

| Parameter | Configuration Key | Frozen Value | Operational Rationale |
| :--- | :--- | :---: | :--- |
| **Nominal Coverage Target** | `alpha` | $0.10$ ($90\%$ nominal) | Standard enterprise risk-hedging level for logistics |
| **Label Maturity Embargo Buffer** | `embargo_days` | $90\text{ days}$ | Mandatory buffer preventing leakage of in-transit deliveries |
| **Calibration Window Length** | `calib_window_days` | $180\text{ days}$ | Balances sample density with temporal locality |
| **Minimum Calibration Sample Size** | `min_sample_size` | $50\text{ shipments}$ | Suppresses noisy re-estimates on low-volume batches |
| **Monitoring Evaluation Cadence** | `eval_step_days` | $30\text{ days}$ | Regular monthly operational audit interval |
| **Recalibration Cooldown Period** | `cooldown_days` | $30\text{ days}$ | Prevents rapid consecutive recalibration churn |
| **Stale Calibration Timeout** | `stale_timeout_days`| $180\text{ days}$ | Mandatory refresh trigger if no event fired in 6 months |
| **Maximum Volume Timeout** | `stale_timeout_volume`| $1,500\text{ items}$ | Volume-based refresh trigger for high-throughput hubs |

---

## 4. 4-Dimensional Drift Trigger Matrix

The `DriftTriggerPolicy` continuously assesses four data streams to decide when recalibration is mandatory:

```
+----------------------------------------------------------------------------------------------------+
|                                  TRIGGER DECISION POLICY MATRIX                                    |
+------------------------+------------------------------------+------------------+-------------------+
| Stream / Dimension     | Evaluated Metric                   | Warning (Yellow) | Action (Red Alert)|
+------------------------+------------------------------------+------------------+-------------------+
| 1. Feature Drift       | Tier-1 SHAP Features PSI           | PSI >= 0.10      | PSI >= 0.25 (VETO)|
|                        | Normalized 1-Wasserstein Distance  | W_1 >= 0.30      | W_1 >= 0.60       |
|                        | Two-Sample KS with FDR Control     | q-val < 0.05     | q-val < 0.01      |
| 2. Prediction Drift    | P(Late) Model Probability Output   | PSI >= 0.10      | PSI >= 0.20       |
| 3. Target Drift        | Delivery Delay Prevalence Shift    | Delta p >= 0.03  | Delta p >= 0.06   |
| 4. Uncertainty Drift   | Nonconformity Score Shift W_1(S)   | W_1 >= 2.0 days  | W_1 >= 3.0 days   |
|                        | Empirical Coverage Deficit CovErr  | CovErr >= 0.05   | CovErr >= 0.08    |
+------------------------+------------------------------------+------------------+-------------------+
```

### Tier-1 SHAP Feature Veto List:
A severe shift ($\text{PSI} \ge 0.25$) in any of the following 11 top-ranked SHAP features immediately emits a `RED_TRIGGER`:
1. `Vendor INCO Term`
2. `Vendor`
3. `Country`
4. `Transit Days`
5. `vendor_hist_volume`
6. `country_hist_volume`
7. `route_hist_volume`
8. `scheduled_month_sin`
9. `scheduled_month_cos`
10. `Pack Price`
11. `Line Item Value`

---

## 5. Audit Trail of Holdout Recalibration Events (365-Day Cohort)

During the single-pass final holdout evaluation ($N=1,013$ shipments, 2014-08-24 to 2015-08-24), Strategy C executed **4 discrete recalibrations**:

| Event # | Timestamp | Triggering Anomaly / Vector | Ingested Matured Calibration Window | Matured Sample ($n$) | Old Cutoff ($Q$) | New Cutoff ($Q$) | Execution Latency |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **2014-10-23** | Tier-1 Veto on `vendor_hist_volume` ($\text{PSI}=7.102$) & `Country` ($\text{PSI}=7.923$); $\mathcal{W}_1(S)=3.99\text{d}$ | `2014-01-26` to `2014-07-25` | $688$ | $0.00\text{d}$ | **$34.00\text{d}$** | $0.110\text{ ms}$ |
| **2** | **2014-12-22** | Persistent destination volume shift (Tier-1 `country_hist_volume` $\text{PSI}=6.749$; $\mathcal{W}_1(S)=8.09\text{d}$) | `2014-03-27` to `2014-09-23` | $869$ | $34.00\text{d}$ | **$33.00\text{d}$** | $0.152\text{ ms}$ |
| **3** | **2015-02-20** | Vendor realignment & prevalence uptick ($\Delta \bar{p}=+0.066$; $\mathcal{W}_1(S)=5.76\text{d}$) | `2014-05-26` to `2014-11-22` | $803$ | $33.00\text{d}$ | **$28.00\text{d}$** | $0.112\text{ ms}$ |
| **4** | **2015-05-18** | Late-stage corridor stabilization ($\mathcal{W}_1(S)=4.87\text{d}$) | `2014-08-21` to `2015-02-17` | $697$ | $28.00\text{d}$ | **$21.00\text{d}$** | $0.139\text{ ms}$ |

- **Total Annual Computational Latency**: **$0.512\text{ ms}$** (Mean latency per recalibration = $0.128\text{ ms}$).
- **Mean Time Between Recalibrations (MTBR)**: **$91.0\text{ days}$** ($4.01\text{ events / year}$).
- **Empirical Coverage Restored**: **$93.88\%$** (vs $80.36\%$ Static and $86.48\%$ Rolling).

---

## 6. Stale-Calibration Fallback Protocol

If no drift triggers fire for $180\text{ days}$ ($T_{\text{max}}$) OR if the active calibration window contains fewer than $N_{\text{min}} = 50$ shipments:
1. **Fallback Status**: The system enters `STALE_CALIBRATION_MODE`.
2. **Conservative Inflation**: Prediction interval widths are inflated by a conservative multiplier of $1.5\text{x}$:
   $$\mathcal{C}_{\text{stale}}(X) = \left[ \hat{q}_{0.05}(X) - 1.5 \cdot Q, \; \hat{q}_{0.95}(X) + 1.5 \cdot Q \right]$$
3. **Operational Notification**: Emit high-priority MLOps alert to audit data ingestion pipelines.

---

## 7. Provenance & Non-Causal Compliance

All nonconformity scores and interval bounds are classified under 4-tier provenance tags:
- `OBSERVED_SCMS_DATA`: Historical ground-truth shipment outcomes $Y_i$.
- `SYNTHETIC_E9_STATE`: Observable dynamic state vectors $S_i(t)$.
- `SIMULATED_COUNTERFACTUAL`: Simulated post-action quantile adjustments.
- `SIMULATED_COST`: Synthetic business cost impact of uncertainty bounds.
