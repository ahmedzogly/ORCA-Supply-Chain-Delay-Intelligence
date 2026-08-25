# API Specification

## Endpoints

### GET /health
Validates container health and model registry readiness.
Returns: {"status": "ok", "model_version": "v1.0.0"}

### POST /predict
Estimates prediction boundaries.
Returns probability_late, isk_tier, severity_p50, and severity_interval_90.

### POST /explain
Explains a specific request using SHAP and Causal constraints.
Returns 	op_predictive_drivers and causal_candidates.

### POST /recommend
End-to-end wrapper returning the prescriptive decision.
Returns ecommendation, decision_reason, obustness, and human_approval_required.

## Schema Rules
Requests containing forbidden target variables (Delay_Days, Delay_Flag, etc.) are actively rejected with HTTP 422.
