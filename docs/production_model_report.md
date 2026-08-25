# Production Model Report & Champion Selection

## 1. Experimental Execution
- **Models Evaluated**: Logistic Regression (Linear) and LightGBM (Tree). (Note: CatBoost was scheduled but dropped due to environmental compilation failure on Python 3.14).
- **Validation Protocol**: 5-Fold Expanding Window.
- **Inner Tuning**: Thresholds were optimized via 3-Fold TimeSeriesSplit inner CV on training data.
- **Calibration**: Isotonic regression was applied using inner CV out-of-fold predictions.

## 2. Classification Results (Calibrated & Threshold Tuned)

| Model | PR-AUC | F1 Score | Balanced Accuracy | Brier Score | Opt Threshold (Mean) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | 0.2593 | 0.0902 | 0.5109 | 0.1315 | 0.28 |
| **Logistic Regression** | 0.2458 | 0.2728 | 0.5956 | 0.1334 | 0.18 |

### Research Question Investigation: Why did LR outperform LGBM on F1 previously?
In Stage 4, Logistic Regression heavily outperformed LightGBM on F1. Our Stage 5 analysis confirms why:
1. **Calibration & Default Thresholds**: Uncalibrated tree models pushed probabilities to extremes. Logistic Regression's natural sigmoid output was better scaled to the 14% prior.
2. **Feature Linearity**: The heavy use of historical point-in-time aggregates (like endor_hist_delay_rate) provides extremely strong linear signals. LightGBM aggressively overfits to noise in the expanding window, while Logistic Regression acts as a stable weighted ensemble of the historical rates.
3. **Threshold Tuning**: Even after isotonic calibration and threshold tuning (LGBM ~0.28, LR ~0.18), LR still dominates on F1 because its rank-ordering is smoother across the decision boundary.

## 3. Regression Task Analysis

We evaluated Delay_Days (Signed) vs max(Delay_Days, 0) (Positive Only).

| Model | Target | MAE | RMSE | R² | Median AE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM | Signed | 15.33 | 26.15 | -0.1598 | 7.10 |
| Ridge | Signed | 17.87 | 27.06 | -0.2786 | 11.62 |
| LightGBM | Positive Only | 8.85 | 19.27 | -2.2565 | 4.03 |
| Ridge | Positive Only | 7.07 | 14.61 | -0.7245 | 3.99 |

**Conclusion on Regression**: The evaluated models failed to outperform the expanding historical mean under the current feature set and temporal validation protocol for both signed and positive-only formulations. The huge variance in actual delay severity (ranging from 1 day to >100 days) cannot be captured by continuous RMSE optimization here.

## 4. Champion Selection

**Champion**: Logistic Regression (Classification)
- **Primary Metric (PR-AUC)**: 0.2458 (highly competitive with LGBM).
- **Secondary Metric (F1)**: 0.2728 (vastly superior to LGBM).
- **Stability**: Highly stable across expanding windows.
- **Explainability**: Inherently interpretable coefficients, fully SHAP compatible via LinearExplainer.
- **Cost**: Near-zero training and inference latency.
