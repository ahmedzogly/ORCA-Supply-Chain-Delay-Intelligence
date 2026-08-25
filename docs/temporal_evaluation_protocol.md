# Temporal Evaluation Protocol

## 1. Objective
To construct a robust evaluation framework that guarantees chronological integrity, prevents future leakage, and tests model generalization across shifting operational environments.

## 2. Protocol: Expanding-Window Rolling-Origin CV
Standard cross-validation (K-Fold) randomizes data, which causes future data to predict the past—a critical violation in supply chain forecasting. We mandate **Rolling-Origin Cross-Validation**:
- **Chronological Sorting**: The entire 8,319-row modeling cohort is sorted strictly by T_pred.
- **Expanding Window**: Training data strictly precedes evaluation data. For each sequential fold, the training window expands to include more recent data.
- **Temporal Gap**: A configurable gap (90 days) is enforced between the maximum training T_pred and the minimum evaluation T_pred. This mirrors the operational reality where recent ground truth (actual delivery dates) is not yet available for recently dispatched orders.
- **Final Holdout Isolation**: The final 365 days of the dataset (2014-08-24 to 2015-08-24) form an absolute holdout. This dataset is completely unseen during feature engineering and CV.

## 3. Configuration & Enforcement
- **n_folds**: 5
- **gap_days**: 90
- **holdout_duration_days**: 365
- **val_duration_days**: 180
- **min_train_days**: 730 (2 years)

Strict automated testing (	ests/test_rolling_origin.py) continuously verifies:
1. max(train_T_pred) < min(eval_T_pred) for all folds.
2. Gap $\ge$ 90 days.
3. Absolutely no overlap between Train, Validation, and Holdout sets.
