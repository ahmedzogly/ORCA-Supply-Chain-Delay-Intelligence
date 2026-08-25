# Cross-Domain Experiment Protocol

1. **Zero-Shot Transfer**: SCMS Calibrated Champion is executed on DataCo and Olist evaluation folds without any modification.
2. **Target-Domain Recalibration**: SCMS Champion is retained, but conformal bounds and Platt scaling are refit using DataCo/Olist development data to correct calibration shift.
3. **Temporal Constraint**: Final SCMS holdout is never accessed.
