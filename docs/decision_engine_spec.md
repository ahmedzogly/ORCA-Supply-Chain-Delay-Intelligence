# Decision Engine Specification

The Prescriptive Decision Engine translates raw model outputs into auditable operational decisions.

## Inputs
- **Probability of Delay (P(Late))**: From Stage 5 classifier.
- **Severity & Uncertainty**: From Stage 6 conformal quantiles.
- **Predictive Drivers**: SHAP outputs (Stage 7A).
- **Causal Evidence**: Stable DAG relationships (Stage 7B).
- **Business Context**: Line item value, constraints.

## Outputs
Every decision generates an auditable JSON trace containing:
- Risk Tier
- Required Human Approval flag
- Action recommendation
- Cost estimates
- Explicit reasons linking inputs to the action

## Rules
- Simple threshold logic is rejected. Action is only recommended if Simulated Net Benefit > 0.
- High uncertainty dynamically downgrades automated actions to HUMAN_REVIEW.
