# Stage 2 E6.5 Report — Chronological Drift Detection System

**System**: Supply Chain Delay Intelligence System  
**Milestone**: Milestone 1 (Phase 2 Part 1: E6.5 Drift Detection)  
**Dataset**: USAID / SCMS Delivery History (10,324 shipment line items, 2006–2015)  
**Modeling Cohort**: 8,319 strictly anchored shipments (7,306 Development, 1,013 Final Holdout)  
**Runtime Environment**: Python 3.14.5 / pandas 3.0.5 / scipy 1.17.0 / catboost 1.2.10 / pytest 9.1.1  
**Report Generated**: 2026-08-18  

---

## STATUS: PASS

---

## 1. Executive Summary

Milestone 1 (E6.5) designs, implements, and validates a **4-Dimensional Chronological Drift Detection Engine** for the Supply Chain Delay Intelligence System.

Prior empirical findings in Stage 12 demonstrated that under real-world temporal shift on the final holdout, static Conformalized Quantile Regression (CQR) nominal 90% coverage collapsed from **89.3% in Development CV to 22.95% on the holdout** (coverage error $+0.6705$, mean interval width $4.19$ days), proving the urgent necessity of automated drift detection and adaptive recalibration.

E6.5 provides the mathematical and algorithmic foundation to monitor, diagnose, and trigger recalibration under strict chronological constraints (Past $\rightarrow$ Future ordering, 90-day embargo gaps, zero final holdout contamination).

---

## 2. Four-Dimensional Drift Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CHRONOLOGICAL DRIFT ARCHITECTURE                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. FEATURE DRIFT P(X):                                                                 │
│    - 26 Numerical Features: Laplace-regularized PSI (eps=1e-4), Scale-Normalized       │
│      1-Wasserstein Distance (W_1_norm), Two-sample KS-test with Benjamini-Hochberg     │
│      False Discovery Rate (FDR) control at alpha=0.05.                                 │
│    - 13 Categorical Features: Jensen-Shannon Divergence & Distance ([0,1] bounded),    │
│      Chi-squared Goodness-of-Fit with rare-category pooling (Cochran's rule),          │
│      Categorical PSI.                                                                  │
│                                                                                        │
│ 2. PREDICTION DRIFT P(Y_hat | X):                                                      │
│    - Classifier Output Probabilities: Binned PSI & Wasserstein distance W_1(p_hat).    │
│    - Regressor Point Predictions: Output PSI & Scale-Normalized W_1(y_hat).            │
│    - Quantile Output Shifts: Conditional quantile shifts (q05, q50, q95).              │
│                                                                                        │
│ 3. TARGET / PREVALENCE DRIFT P(Y):                                                     │
│    - Binary Late Delivery Prevalence: Delta prevalence |Delta y_bar|, Two-proportion   │
│      z-test (z-stat, p-value), Binary Target PSI.                                      │
│    - Continuous Delay Days: Normalized Wasserstein W_1(Y), Target PSI.                 │
│    - Severe Delay Proportion: Shift in extreme delays Delta P(Delay_Days > 14).        │
│                                                                                        │
│ 4. UNCERTAINTY DRIFT P(S), P(W):                                                       │
│    - CQR Nonconformity Shift: W_1(S_calib, S_det), delta mean nonconformity, KS-test.  │
│    - Empirical Coverage Deficit: CovErr = (1 - alpha) - Cov_det.                       │
│    - Exact One-Sided Binomial Test for Undercoverage: H0: p >= 1 - alpha vs H1: p <... │
│    - Conformal Interval Widths: W_1(W_calib, W_det), median width delta, width ratio.  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Empirical Results across Development CV Folds (0–4)

Drift evaluation was executed across the 5 historical Development CV folds spanning **2006-04-19 to 2014-08-24** (7,306 records, strictly excluding the 365-day final holdout):

| Fold | Detection Window | Sample Count ($N_{det}$) | Feature Status ($S_{feat}$) | Top Shifted Feature (PSI) | Prediction PSI $\text{PSI}(\hat{p})$ | Target $\Delta \text{Prev}$ ($p_z$) | Uncertainty Coverage ($\text{Cov}_{90\%}$) | Uncertainty Score $\mathcal{W}_1(S)$ | Overall Status | Trigger Recalibration |
|---|---|---|---|---|---|---|---|---|---|---|
| **0** | 2012-03-08 to 2012-08-31 | 598 | RED (140.44) | `T_pred_month` (7.631) | 0.270 (RED) | $-0.076$ ($p=2.9\times 10^{-7}$) | 0.963 ($p_{binom}=1.00$) | 4.23d (RED) | **RED** | **TRUE (Veto)** |
| **1** | 2012-09-05 to 2013-03-01 | 618 | RED (135.71) | `T_pred_year` (9.480) | 0.184 (YELLOW) | $+0.029$ ($p=0.048$) | 0.927 ($p_{binom}=0.99$) | 2.09d (YELLOW) | **RED** | **TRUE (Veto)** |
| **2** | 2013-03-04 to 2013-08-27 | 738 | RED (111.68) | `T_pred_year` (8.164) | 0.165 (YELLOW) | $+0.134$ ($p=9.4\times 10^{-22}$) | 0.908 ($p_{binom}=0.78$) | 3.78d (RED) | **RED** | **TRUE (Veto)** |
| **3** | 2013-08-30 to 2014-02-19 | 606 | RED (130.14) | `T_pred_year` (10.230) | 0.324 (RED) | $+0.060$ ($p=6.5\times 10^{-5}$) | 0.878 ($p_{binom}=0.043$) | 2.35d (YELLOW) | **RED** | **TRUE (Veto)** |
| **4** | 2014-02-26 to 2014-08-21 | 717 | RED (113.34) | `T_pred_year` (9.001) | 0.705 (RED) | $-0.008$ ($p=0.557$) | 0.909 ($p_{binom}=0.81$) | 1.59d (YELLOW) | **RED** | **TRUE (Veto)** |

### Diagnostic Analysis:
1. **Dynamic Macroeconomic Shift**: Top SHAP drivers (`Vendor INCO Term`, `Vendor`, `Country`, `vendor_hist_volume`) consistently exhibited significant distribution changes across 2012–2014, reflecting contract renegotiations and shifting African distribution hubs.
2. **Prevalence Volatility**: Target late delivery prevalence fluctuated heavily (from $6.52\%$ in Fold 0 up to $26.42\%$ in Fold 2), highlighting non-stationary supply chain conditions.
3. **Nonconformity Expansion**: CQR nonconformity score Wasserstein distance $\mathcal{W}_1(S_{calib}, S_{det})$ exceeded $3.0\text{ days}$ in Folds 0 and 2, accurately flagging that static calibration parameters were breaking down.

---

## 4. Policy & Governance Rules Verification

1. **Tier 1 SHAP Feature Veto**: Confirmed. When any of the 11 Tier 1 critical features (`Vendor INCO Term`, `Vendor`, `Country`, `Transit Days`, etc.) reaches $\text{PSI} \ge 0.25$, the policy immediately flags RED and requests recalibration.
2. **Minimum Sample Size Guard ($N_{min} = 50$)**: Confirmed. Small batches with $N < 50$ are marked `INSUFFICIENT_SAMPLE` and automatic triggers are suppressed.
3. **Stale Calibration Timeout ($T_{max} = 180\text{ days}$ / $V_{max} = 1,500\text{ shipments}$)**: Confirmed. Models uncalibrated after 180 days trigger scheduled maintenance.
4. **Recalibration Cooldown Period ($T_{cooldown} = 30\text{ days}$)**: Confirmed. Prevents chattering and oscillatory recalibration loops.
5. **Persistence Confirmation ($k = 2$)**: Confirmed. Moderate warnings require 2 consecutive windows for escalation.

---

## 5. Temporal Safety & Isolation Proof

1. **Zero Holdout Access**: Confirmed. The 1,013 holdout rows ($T_{pred} \ge 2014\text{-}08\text{-}24$) have zero set intersection with training or calibration indices.
2. **Chronological Invariant**: Confirmed. $\max(t \in W_{train}) \le \min(t \in W_{val})$ for all folds, maintaining the configured 90-day embargo gap.
3. **Immutable Baselines**: Confirmed. Frozen artifacts from Stages 0–13 (`catboost_champion.cbm`, `cqr_calibration.json`, `final_holdout_metrics.json`, `fold_manifest.md`) remain 100% bitwise intact.

---

## 6. Test Suite Execution Summary

```powershell
============================= test session starts =============================
platform win32 -- Python 3.14.5, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\Admin\Desktop\try1\delay_intelligence_system
configfile: pyproject.toml
collected 254 items

tests/test_drift_temporal_safety.py::test_holdout_isolation_in_threshold_calibration PASSED
tests/test_drift_temporal_safety.py::test_strict_chronological_ordering_of_windows PASSED
tests/test_drift_temporal_safety.py::test_label_lag_embargo_compliance PASSED
tests/test_drift_temporal_safety.py::test_dev_frozen_threshold_immutability PASSED
tests/test_drift_temporal_safety.py::test_drift_runner_artifacts_temporal_isolation PASSED
tests/test_drift_determinism.py::test_psi_mathematical_properties_and_robustness PASSED
tests/test_drift_determinism.py::test_wasserstein_metric_axioms PASSED
tests/test_drift_determinism.py::test_categorical_metrics_axioms PASSED
tests/test_drift_determinism.py::test_uncertainty_drift_metrics_axioms PASSED
tests/test_drift_determinism.py::test_composite_trigger_state_transitions PASSED
tests/test_drift_determinism.py::test_end_to_end_drift_determinism_across_runs PASSED
... [243 baseline tests from Stages 0-13 PASSED]

======================= 254 passed, 1 warning in 19.18s =======================
```

---

## 7. Deliverables Inventory

| Artifact / Module | Path | Description |
|---|---|---|
| Drift Schemas | `src/delay_intelligence/drift/schemas.py` | Pydantic data models for 4 drift dimensions, tiers, and triggers |
| Drift Metrics | `src/delay_intelligence/drift/metrics.py` | Mathematical formulations: PSI, Wasserstein, KS-FDR, JSD, Chi2, CQR tests |
| Drift Policy | `src/delay_intelligence/drift/policy.py` | 3-tier composite decision engine with Tier 1 SHAP veto and sample guards |
| Drift Detector | `src/delay_intelligence/drift/detector.py` | Chronological multi-dimensional detection orchestrator |
| Drift Runner | `src/delay_intelligence/drift/runner.py` | Historical CV drift evaluation harness across development folds |
| Package Init | `src/delay_intelligence/drift/__init__.py` | Public API exports for drift package |
| Drift Config | `configs/drift.yaml` | YAML configuration for feature tiers, thresholds, and window policies |
| Methodology Doc | `docs/drift_detection_methodology.md` | Comprehensive mathematical, windowing, and algorithmic documentation |
| Feature Analysis Doc | `docs/drift_feature_analysis.md` | Feature taxonomies, SHAP stability rankings, sensitivities, and failure modes |
| Trigger Policy Doc | `docs/drift_trigger_policy.md` | Complete governance, threshold matrices, cooldown, and escalation policy |
| Temporal Safety Tests | `tests/test_drift_temporal_safety.py` | Automated tests verifying holdout quarantine and chronological invariants |
| Determinism Tests | `tests/test_drift_determinism.py` | Automated tests verifying metric axioms, state transitions, and determinism |
| Drift Metrics Artifact | `artifacts/drift/drift_metrics.csv` | Historical fold-by-fold drift dimension metrics |
| Feature Drift Artifact | `artifacts/drift/feature_drift_summary.csv` | Feature-level PSI, Wasserstein, KS, and Chi2 metrics across all folds |
| Trigger Decisions | `artifacts/drift/drift_triggers.json` | JSON audit trail of trigger evaluations and reasons |
| Full CV Summary | `artifacts/drift/cv_drift_summary.json` | Detailed structured reports across all development folds |
| Stage Report | `stage_2_e6_5_report.md` | Formal milestone report stating STATUS: PASS |

---

## 8. Gate Sign-off & Next Stage Recommendation

- **Milestone 1 (E6.5 Drift Detection) Gate**: **PASSED**.
- **Holdout Status**: Strict quarantine verified.
- **Baseline Integrity**: 243/243 baseline tests intact, zero regressions.
- **Recommendation**: Proceed to **Milestone 2 (E7 Design Freeze & Adaptive Conformal Recalibration Protocol)**.
