# Baseline Model Report

## 1. Experimental Setup
- **Evaluation Strategy**: 5-Fold Expanding-Window Rolling-Origin CV.
- **Data**: 8,319 temporal feature rows engineered in Stage 3.
- **Target (Classification)**: Delay_Flag (1 if Delay_Days > 0, else 0).
- **Target (Regression)**: Signed Delay_Days. We predict the continuous target (including earliness/negative delays) because predicting exact arrival delta supports broader inventory holding cost optimizations, not just stockout prevention.
- **Class Weights**: alanced used for LR and LightGBM classifiers.

## 2. Classification Results (Mean ± Std across 5 Folds)

### Mean Metrics
| model              |   pr_auc |   roc_auc |     f1 |   precision |   recall |   balanced_accuracy |   brier_score |
|:-------------------|---------:|----------:|-------:|------------:|---------:|--------------------:|--------------:|
| Dummy              |   0.1666 |    0.5    | 0      |      0      |   0      |              0.5    |        0.1402 |
| LightGBM           |   0.2935 |    0.7346 | 0.2352 |      0.2542 |   0.242  |              0.5601 |        0.1414 |
| LogisticRegression |   0.3011 |    0.7132 | 0.3479 |      0.285  |   0.5695 |              0.633  |        0.2001 |

### Std Deviation
| model              |   pr_auc |   roc_auc |     f1 |   precision |   recall |   balanced_accuracy |   brier_score |
|:-------------------|---------:|----------:|-------:|------------:|---------:|--------------------:|--------------:|
| Dummy              |   0.0733 |    0      | 0      |      0      |   0      |              0      |        0.0535 |
| LightGBM           |   0.1241 |    0.0601 | 0.191  |      0.1385 |   0.2401 |              0.0665 |        0.0527 |
| LogisticRegression |   0.1529 |    0.079  | 0.1589 |      0.1592 |   0.2525 |              0.0989 |        0.0392 |

### Analysis
- **Logistic Regression** achieved the highest F1 Score (0.3479) and Balanced Accuracy (0.6330), significantly outperforming the Dummy baseline. Its ROC-AUC (0.7132) shows solid ranking capability.
- **LightGBM** achieved slightly better ROC-AUC (0.7346) but worse F1 (0.2352) and Recall. Because LightGBM is tree-based, the expanding window might introduce feature scale/distribution shifts that trees handle well for ranking (ROC) but poorly for calibrated thresholding without further tuning.

## 3. Regression Results (Mean ± Std across 5 Folds)

### Mean Metrics
| model    |     mae |    rmse |      r2 |
|:---------|--------:|--------:|--------:|
| Dummy    | 14.0634 | 24.6103 | -0.016  |
| LightGBM | 15.7178 | 26.4128 | -0.1916 |
| Ridge    | 17.8762 | 27.0694 | -0.2786 |

### Std Deviation
| model    |    mae |   rmse |     r2 |
|:---------|-------:|-------:|-------:|
| Dummy    | 2.4761 | 6.2612 | 0.0253 |
| LightGBM | 5.1195 | 7.1807 | 0.2479 |
| Ridge    | 2.5788 | 5.2291 | 0.2422 |

### Analysis
- **All Models struggled** to beat the Dummy Regressor (predicting the mean). 
- Ridge and LightGBM exhibited **negative ^2$**, meaning they performed worse than simply predicting the historical mean Delay_Days. 
- This indicates the continuous signal for exactly *how many days* early/late a shipment will be is extremely noisy or non-stationary. The regression task requires significantly more complex modeling (e.g., separate models for severity given delay, or quantile regression) in later stages.

## 4. Stability and Cost
- **Temporal Stability**: Both Logistic Regression and LightGBM show noticeable standard deviation across folds, confirming that the supply chain environment is non-stationary. 
- **Training Cost**: LightGBM fits in fractions of a second. Logistic Regression is slightly slower due to the OneHotEncoder expanding the feature space, but still sub-second.
