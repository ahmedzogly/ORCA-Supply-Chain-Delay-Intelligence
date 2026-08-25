# Explainability Methodology

This stage utilizes SHAP (SHapley Additive exPlanations) to explain the predictive behavior of the Calibrated CatBoost Production Champion.
SHAP values represent the marginal contribution of a feature to the model's prediction of P(Late).

## Interpretability Rules
1. **No Causal Claims**: SHAP values are strictly predictive associations. A high SHAP value means the feature strongly contributed to the model's output probability. It does NOT mean the feature caused the delay in the real world.
2. **Temporal Stability**: Feature importance is evaluated across the 5 chronological folds. Features are only considered reliable predictive drivers if their importance rank remains stable.

# SHAP Global Report

Based on the SHAP values computed on the validation sets of all folds, the top predictive drivers of P(Late) are:
1. **Vendor INCO Term**: Controls routing, liability, and transit paths.
2. **Vendor**: Direct historical performance of the specific vendor.
3. **vendor_hist_volume**: Scale of operations for the vendor.
4. **Country**: Destination factors, customs, and last-mile infrastructure.
5. **country_hist_delay_rate**: Baseline historical risk of the destination.

# SHAP Stability Report

Feature ranks across 5 folds show clear segmentation:
- **Stable Drivers**: Vendor INCO Term, Vendor, and Forecast_Horizon_Days show low variance in importance rank across temporal boundaries.
- **Unstable Drivers**: Certain features spike in importance during specific folds, reflecting shifting supply chain dynamics.
- **Emerging Drivers**: Features that steadily climb in importance rank as time progresses.
