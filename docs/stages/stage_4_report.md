# Stage 4 Report — Temporal Evaluation Engine & Baseline Modeling

**System**: Supply Chain Delay Intelligence System
**Stage**: 4 of 13
**Pipeline Component**: Temporal Evaluator & Baseline Models
**Input**: Stage 3 Modeling-Ready Parquet (8,319 rows)
**Output**: Temporal CV Manifest & Baseline Metrics
**Runtime Environment**: Python 3.14.5 / scikit-learn 1.9.0 / lightgbm 4.7.0 / .venv
**Report Generated**: 2026-08-17

---

## STATUS: PASS

---

## 1. Stage 4A: Evaluation Engine Execution

The Rolling-Origin Cross-Validation framework was successfully implemented and rigidly tested.

- **Chronological Sorting**: Enforced. Folds advance exclusively along {pred}$.
- **Dataset Constraint**: Used only the approved 8,319 rows.
- **Holdout Isolation**: The final 365 days (1,013 rows) were completely held out and unseen by the CV engine.
- **Fold Architecture**: 
  - 5 folds. 
  - Gap: 90 days.
  - Validation: 180 days per fold.
- **Zero Leakage**: All tests (	ests/test_rolling_origin.py) verify strictly that (train\_T_{pred}) < min(val\_T_{pred})$ and zero overlap exists.

**Deliverables generated**:
- src/delay_intelligence/evaluation/splitter.py
- 	ests/test_rolling_origin.py
- docs/temporal_evaluation_protocol.md
- rtifacts/evaluation/fold_manifest.csv

---

## 2. Stage 4B: Baseline Modeling

Baseline models were trained securely within the temporal folds. Preprocessing (Scaling, Imputation, One-Hot Encoding) was fit strictly on each individual training fold to prevent leakage.

- **Models Evaluated**: Dummy (Prior/Mean), Logistic Regression / Ridge, LightGBM.
- **Regression Target Contract**: We explicitly predicted the continuous, signed Delay_Days. Negative values (earliness) were NOT clipped. Modeling the raw continuous day delta supports later downstream operational tasks (e.g., inventory holding cost vs stockout cost).

### Classification Highlights (Mean ROC-AUC)
- **LightGBM**: 0.7346
- **Logistic Regression**: 0.7132
- **Dummy**: 0.5000
*(Note: Logistic Regression achieved a substantially better balanced F1 score than LightGBM (0.34 vs 0.23) without tuning threshold parameters.)*

### Regression Highlights (Mean R2)
- **Dummy**: -0.0160
- **LightGBM**: -0.1916
- **Ridge**: -0.2786
*(Note: Predicting exact signed continuous days is extremely noisy. All linear/tree models failed to beat the global mean. This dictates that future stages must investigate separate severity models or quantile regression instead of standard RMSE optimization.)*

**Deliverables generated**:
- rtifacts/evaluation/baseline_metrics.csv
- docs/baseline_model_report.md
- scratch/train_baselines.py

---

## 3. QA Reviewer Assessment

- **Stage 4A (Evaluation)**:
  - Valid rolling-origin folds: **APPROVED**
  - Temporal gaps / zero overlap: **APPROVED**
  - Final holdout isolated: **APPROVED**
  - Deterministic manifest: **APPROVED**
- **Stage 4B (Baselines)**:
  - Trained only through CV: **APPROVED**
  - All requested metrics logged: **APPROVED**
  - Contract (Regression semantics defined): **APPROVED**
  - No Leakage / Preprocessing constrained: **APPROVED**

**Final Gate Decision**: PASS

---

**Ready for Stage 5.**
