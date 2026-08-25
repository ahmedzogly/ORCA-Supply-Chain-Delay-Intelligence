# Model Selection Protocol

## 1. Objective
To identify the best production-ready model for forecasting supply chain delays, moving beyond raw accuracy/AUC to consider temporal stability, calibration, interpretability, and operational usefulness.

## 2. Methodology
- **Temporal CV**: 5-Fold Expanding Window. Final holdout is strictly isolated.
- **Candidate Models**: Logistic Regression, LightGBM, CatBoost.
- **Tuning & Selection**: Nested/Inner Temporal CV is used to learn decision thresholds and probability calibration functions before the validation fold is evaluated.
- **Primary Metric**: PR-AUC (Precision-Recall Area Under Curve), as this is an imbalanced dataset (~14% minority class). Secondary metrics include F1, ROC-AUC, Brier Score, and inference cost.

## 3. Decision Matrix
A model is selected as Champion if it balances:
1. High PR-AUC and F1.
2. Stability across temporal folds.
3. Good calibration (Brier Score).
4. SHAP compatibility for downstream explainability.
