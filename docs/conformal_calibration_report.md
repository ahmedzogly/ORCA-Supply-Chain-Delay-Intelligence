# Conformal Calibration & Temporal Safety Report

## Method: Conformalized Quantile Regression (CQR)
To guarantee coverage without relying on parametric assumptions, we implemented distribution-free split Conformalized Quantile Regression (CQR).

## Temporal Safety Protocol
1. **Inner Split**: The chronological training fold is further split using a TimeSeriesSplit(n_splits=2).
2. **Base Fit**: LightGBM quantile models are fit on the inner_train partition.
3. **Calibration**: Non-conformity scores (the absolute under-coverage error) are calculated exclusively on the inner_calib partition.
4. **Holdout Protection**: The final 365-day holdout is rigorously excluded from all calibration steps, as enforced by automated tests.

## Validity vs. Sharpness
Results on Delay_Days_Positive_Only across 5 temporal folds:

| Nominal Level | Observed Coverage | Coverage Error | Mean Interval Width |
|---------------|-------------------|----------------|---------------------|
| 80%           | 0.811             | +0.011         | 11.0 days           |
| 90%           | 0.893             | -0.007         | 21.4 days           |
| 95%           | 0.955             | +0.005         | 30.7 days           |

The CQR procedure demonstrated empirical coverage close to nominal targets under the project's temporal evaluation protocol. The resulting intervals are remarkably sharp and operationally viable.

## Unified Risk Representation
The final predictive payload merges Classification and Uncertainty:
- **P(Late)**: Derived from the Stage 5 Calibrated CatBoost Classifier.
- **Median Expected Delay**: Derived from the P50 LightGBM Quantile.
- **Upper Risk Bound**: Derived from the CQR Calibrated P95 upper quantile. 

