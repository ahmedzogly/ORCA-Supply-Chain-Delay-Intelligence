# Probability Calibration Report

## 1. Goal
Evaluate whether output probabilities reflect true delay likelihoods (e.g., if the model predicts 0.8, does the shipment delay 80% of the time?).

## 2. Methodology
- **Calibrator**: CalibratedClassifierCV (Isotonic regression)
- **Inner CV Strategy**: TimeSeriesSplit (3-fold) on the training data.
- **Holdout Protection**: Calibration completely shields the validation fold and final holdout.

## 3. Results
- **Brier Score**: Used to quantify calibration error. Lower is better. 
- Calibrated LightGBM and CatBoost probabilities are significantly more reliable for downstream risk thresholding than uncalibrated raw margins.
