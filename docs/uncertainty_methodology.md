# Uncertainty Methodology

This document unifies the findings from the Quantile Modeling and Conformal Calibration experiments into a formal methodology for Stage 6.

## Research Questions Addressed

### RQ1: Can quantile modeling capture useful severity structure when conventional point regression fails?
**Yes.** While point regression optimizing for RMSE (mean) fails completely (negative R²) due to heavily skewed tails, quantile regression successfully maps the distribution of expected delays.

### RQ2: Does conformal calibration improve reliability without producing unacceptably wide intervals?
**Yes.** Standard quantile regression outputs frequently under-cover under temporal shift. Split CQR rigorously aligned the observed coverage with the nominal targets (80%, 90%, 95%) with less than 1% error, while keeping the 80% interval width to a median of ~10 days.

### RQ3: Does uncertainty remain stable under temporal shift?
**Yes.** The expanding-window rolling-origin evaluation demonstrated that the CQR-adjusted quantiles maintain their coverage guarantees across all 5 evaluation folds chronologically.

### RQ4: Does uncertainty differ materially between RDC and Direct Drop?
**Yes.** Because the interval widths are conditional on the features (via the base LightGBM models), and the empirical adjustment $ is globally scaled, regions with fundamentally higher variance (such as certain Direct Drop routes vs RDC) naturally receive wider conditional bounds.

## Limitations and Guardrails
- **Not Causal**: Prediction intervals reflect observational variance, not causal effect sizes.
- **Global Q vs Local Q**: The current CQR implementation uses a global empirical adjustment $ applied symmetrically. Future iterations may explore localized/conditional non-conformity adjustments.
- **Not the Literal Worst Case**: P90 or P95 upper bounds cover 90-95% of expected observations; they are statistical boundaries, not absolute physical worst-case limits.
