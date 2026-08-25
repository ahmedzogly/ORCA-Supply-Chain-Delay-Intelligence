# Final Validation Report

## Holdout Protection
The final SCMS 365-day chronological holdout was preserved flawlessly until Stage 12. No models, thresholds, or causal graphs were tuned on this data.

## Classification Outcome
- **PR-AUC**: 0.181 (degraded from CV mean of 0.287).
- **Status**: MODERATE DEGRADATION.
- **Analysis**: Temporal shift in the final 365 days significantly impacted model calibration. The CatBoost champion successfully retained predictive power (ROC-AUC 0.695) but suffered in the tails.

## Severity & Uncertainty Outcome
- **Empirical Coverage**: 23.0% (target 90.0%).
- **Analysis**: Severe degradation. The conformal calibration from the development period failed to generalize to the final year. This proves that conformal intervals must be continuously recalibrated in production; zero-shot temporal transfer of uncertainty bounds is unsafe.

## Decision Engine
- **Intervention Rate**: 0.49%
- **Analysis**: The system gracefully defaulted to high conservatism (high human-review rate) due to the low-confidence probabilities under drift.

## System Integrity
The end-to-end integration (Features -> FastAPI -> Rules) executed perfectly with zero schema mismatches.
