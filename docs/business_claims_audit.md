# Business Claims Audit

| Claim | Classification | Description |
|---|---|---|
| Model accurately flags delays ahead of time | Observed | Final holdout retained ROC-AUC of 0.695. |
| System generates  savings | Simulated Assumption | Value is based on cost matrices in decision.yaml. No realized ROI is claimed. |
| Actioning 'Expedite' prevents delay | Scenario-based | Assumes vendor compliance. |
| 90% of delays fall in the severity window | **Falsified** | Temporal shift broke the interval calibration. |
