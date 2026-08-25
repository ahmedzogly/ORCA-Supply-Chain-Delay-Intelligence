# Stage 6 Report — Probabilistic Severity & Calibrated Uncertainty

**System**: Supply Chain Delay Intelligence System
**Stage**: 6 of 13
**Pipeline Component**: Uncertainty and Severity Layer
**Report Generated**: 2026-08-17

---

## STATUS: PASS

---

## 1. Objective and Integration
Stage 6 successfully established an uncertainty-aware severity layer to complement the Stage 5 Calibrated CatBoost Classifier. 
By migrating from failed point-regression (RMSE) to **Conformalized Quantile Regression (CQR)**, the system now produces statistically valid, operationally sharp prediction intervals for expected shipment delays.

## 2. Formulation & Target Selection
Two formulations were formally tested:
- **Formulation A**: Delay_Days (Signed)
- **Formulation B**: max(Delay_Days, 0) (Positive Severity)

**Conclusion**: Formulation B generated significantly sharper and more useful operational intervals (mean width 21.4 days vs 58.7 days at 90% coverage) by isolating the right-side tail risk. 

## 3. Conformal Prediction Methodology
A non-parametric Split CQR procedure was implemented.
- **Base Model**: LightGBMRegressor(objective='quantile')
- **Temporal Safety**: Non-conformity scores are learned exclusively from the strictly historical inner_calib split, dynamically preserving the Calibration_time < Evaluation_time constraint.
- **Holdout Protection**: The final 365-day holdout remained strictly isolated.

## 4. Coverage and Sharpness Metrics
Across the 5-fold temporal cross-validation protocol, CQR delivered excellent validity with usable sharpness.

**Delay_Days_Positive_Only**
- **80% Nominal**: 81.1% Observed | Mean Width: 11.0 days
- **90% Nominal**: 89.3% Observed | Mean Width: 21.4 days
- **95% Nominal**: 95.5% Observed | Mean Width: 30.7 days

Coverage error was bounded to within +/- 1.1% despite multi-year temporal dataset shifts.

## 5. Severity Stratification
The probabilistic outputs are mapped to discrete, actionable severity tiers:
- **No Delay**: 0 days
- **Low Severity**: (0, 7] days
- **Moderate Severity**: (7, 14] days
- **High Severity**: > 14 days

## 6. Unified Risk Representation
The API and Decision Engine will consume a tripartite risk object:
{ P_Late: 0.82, Median_Delay: 3.1, Interval_90: [1.0, 8.4] }

## 7. Testing & Quality Assurance
- Automated tests strictly enforce conformal temporal chronological order.
- Monotonicity (Q_low <= Q_median <= Q_high) is guaranteed post-calibration.
- No target leakage into calibration.
- **QA Decision**: PASS.

**Deliverables created**:
- src/delay_intelligence/uncertainty/conformal.py
- configs/uncertainty.yaml
- docs/uncertainty_methodology.md
- docs/quantile_model_report.md
- docs/conformal_calibration_report.md
- rtifacts/evaluation/stage6_uncertainty_metrics.csv
- 	ests/test_uncertainty.py
- 	ests/test_conformal_temporal_safety.py

---

**Ready for Stage 7.**
