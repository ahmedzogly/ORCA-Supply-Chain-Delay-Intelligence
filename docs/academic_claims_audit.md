# Academic Claims Audit

| Claim | Evidence | Scope | Confidence / Limitation |
|---|---|---|---|
| Historical Data Predicts Delay Risk | ROC-AUC 0.695 on strict 1-year holdout | Global SCMS | High / Excludes unobserved global shocks. |
| CatBoost Outperforms LR on Mixed Types | Superior temporal CV F1 and PR-AUC | Within-Domain | Moderate / May vary if data quality degrades. |
| Conformal Intervals Guarantee Coverage | **Falsified**. Coverage dropped to 23% under temporal shift | Temporal Holdout | Zero-shot conformal prediction fails on long temporal horizons. Continuous recalibration is mandatory. |
| Causal Discovery Identifies Root Causes | **Falsified**. Edges were unstable across domains | Cross-Domain | SHAP/PC Algorithm produce 'Causal Hypotheses' only. |
