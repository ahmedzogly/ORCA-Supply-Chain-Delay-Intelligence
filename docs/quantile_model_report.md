# Severity Definition & Quantile Modeling Report

## Formulation A vs Formulation B
We empirically evaluated two representations of severity under quantile modeling:
- **Formulation A (Delay_Days)**: Signed delays. Captures both early deliveries (negative) and late deliveries (positive).
- **Formulation B (max(Delay_Days, 0))**: Positive severity only. Explicitly masks early deliveries as zero delay.

**Result**: Formulation B provides significantly sharper and more operationally useful prediction intervals. At a 90% confidence level, the mean interval width for Signed Delays was **58.7 days**, driven by extreme asymmetry in early vs late tails. In contrast, Positive Severity yielded a mean width of just **21.4 days**. 
Therefore, max(Delay_Days, 0) is the required target for severity modeling.

## Severity Categories
To provide interpretable operational categories alongside continuous probabilistic bounds, we defined the following explicit thresholds on max(Delay_Days, 0):
1. **No Delay**: 0 days (Expected on time or early)
2. **Low Severity**: (0, 7] days (Minor operational delay)
3. **Moderate Severity**: (7, 14] days (Requires planning adjustment)
4. **High Severity**: > 14 days (Severe delay requiring intervention)

## Quantile Model
**Model**: LightGBMRegressor(objective='quantile')
**Quantiles Evaluated**: P025, P05, P10, P50 (Median), P90, P95, P975.
**Temporal Integration**: Embedded within the 5-fold Rolling Origin Splitter.
**Interpretation**: 
- P50 represents the Median Expected Outcome.
- P10 and P90 define the uncalibrated bounds for the 80% coverage interval.
- Uncalibrated intervals naturally under-cover out of sample due to non-stationarity, necessitating Conformal Calibration.
