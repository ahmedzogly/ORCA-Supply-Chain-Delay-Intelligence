# Final Results Source of Truth

**Positioning:** **Research-validated Decision Intelligence Prototype with a Production Roadmap**  
**Purpose:** This is the authoritative interpretation layer for the demo. If a legacy report conflicts with this document, this document controls the presentation claim.

## 1. Evidence taxonomy

| Label | Meaning | Allowed to say | Do not say |
|---|---|---|---|
| **REAL DATA** | Historical SCMS source records | “This record comes from the SCMS dataset.” | “This is a live 2026 shipment.” |
| **MODEL OUTPUT** | Fitted-model prediction/explanation | “The model estimates…” | “This will definitely happen.” |
| **SIMULATED SCENARIO** | Parameterized action/cost/counterfactual | “Under these assumptions, estimated impact is…” | “The company saved this amount.” |
| **EXPLORATORY ONLY** | Legacy causal-discovery hypothesis | “Candidate hypothesis observed across folds.” | “X causes delay.” |
| **NOT VALIDATED** | Portability scaffold with no empirical target test | “Adapter/protocol exists.” | “Validated on DataCo/Olist.” |

## 2. Data used by the patched serving demo — REAL DATA

- Source: SCMS Delivery History Dataset bundled in `data/raw/SCMS_Delivery_History_Dataset.csv`.
- Raw rows: **10,324**.
- Exact SHA-256: `918b992dd3e8d4b64d2a727b2c4ea607603d0c58f19484e73f7b78528c6a8673`.
- Strict modeling cohort: **8,319** rows after prediction-contract/temporal eligibility.
- Patched temporal serving split:
  - training: **6,312** rows, ending before 2013-11-27;
  - 90-day embargo;
  - calibration: **717** rows, 2014-02-25 to before 2014-08-24;
  - untouched final holdout: **1,013** rows, 2014-08-24 to 2015-08-24.

The data are historical and therefore support a reproducible prototype, not a claim of current-2026 production generalization.

## 3. Patched v2 serving validation — MODEL OUTPUT

Artifact: `artifacts/model_registry/v2/serving_validation.json`

### Classification

| Metric | Holdout |
|---|---:|
| PR-AUC | **0.2695669** |
| ROC-AUC | **0.8330262** |
| F1 | **0.3452381** |
| Precision | **0.2710280** |
| Recall | **0.4754098** |
| Balanced accuracy | **0.6967385** |
| Brier score | **0.0499685** |
| Calibrated decision threshold | **0.23** |

These are v2 serving metrics produced after replacing the original proxy/mock registry with a real CatBoost + isotonic-calibration path. The holdout was not used for fitting, probability calibration, threshold selection, or CQR calibration.

### Conditional severity + CQR

Severity is defined as **delay days conditional on the shipment actually being late**.

| Metric | Delayed holdout rows |
|---|---:|
| Evaluation rows | **61** |
| Nominal coverage | **90%** |
| Empirical coverage | **95.08%** |
| Mean interval width | **54.92 days** |
| Median interval width | **37.78 days** |
| Mean q50 prediction | **24.45 days** |

**Required interpretation:** coverage is good on this small delayed-only holdout, but intervals are wide. The result demonstrates uncertainty quantification with a substantial sharpness trade-off; it does not demonstrate precise duration forecasting.

## 4. Original frozen research holdout — MODEL OUTPUT / HISTORICAL BASELINE

Artifact: `artifacts/final/final_holdout_metrics.json`

The original packaged/frozen holdout artifact reported:

- PR-AUC **0.1810468**
- ROC-AUC **0.6950510**
- F1 **0.0606061**
- Precision **0.40**
- Recall **0.0327869**
- Brier **0.0542338**
- original static severity coverage **22.95%**, mean interval width **4.19 days**

These values are retained for provenance. They must not be mixed into the v2 serving table as if they came from the same trained/packaged model.

## 5. Adaptive conformal research artifact — MODEL OUTPUT / RESEARCH EXPERIMENT

Artifact: `artifacts/adaptive_conformal/holdout_adaptive_comparison.json`

The legacy adaptive drift-triggered experiment reports:

- empirical holdout coverage **93.8796%**;
- mean interval width **49.9260 days**;
- median interval width **58.8774 days**;
- **4** recalibration events;
- total measured recalibration latency **0.538 ms** in that artifact.

Use the phrase **“observed empirical coverage”**, not “guaranteed coverage restored.” The width trade-off must be shown with the coverage.

## 6. Business economics — SIMULATED SCENARIO

Artifact: `artifacts/results/e8_final_holdout_metrics.json`

At the **Base** cost scenario and **10% review budget**, the cost-sensitive simulation reports:

- modeled Do-Nothing cost: **$411,378.96**;
- modeled cost-sensitive scenario cost: **$379,889.52**;
- **scenario-based estimated economic impact: $31,489.44**;
- modeled cost reduction: **7.6546%**.

**Mandatory wording:** “scenario-based estimated economic impact under configured assumptions.”  
**Prohibited wording:** “we saved $31,489” or “7.65% realized ROI.”

The Action Center now exposes sliders for assumed delay cost/day, intervention cost, and assumed days reduced so the audience can see that the result depends on assumptions.

## 7. Explainability — MODEL OUTPUT

The patched live `/explain` endpoint now computes **real local CatBoost SHAP values** for each request. SHAP values explain the classifier’s prediction. They do **not** prove that changing a feature will change the outcome.

## 8. Causal analysis — EXPLORATORY ONLY

Legacy PC/Fisher-Z edge stability is retained only as hypothesis generation. Because nominal categorical variables were numerically encoded in the historical discovery experiment, the project does not claim identified causal treatment effects. The API labels any overlap as `exploratory_hypothesis_only`.

## 9. DataCo / Olist — NOT VALIDATED

No empirical external-validation metrics are claimed. The old hard-coded transfer numbers were retired and the evaluator now raises an explicit `ExternalValidationNotPerformed` error if asked for zero-shot/recalibration results without a real target-domain study.

## 10. Live API truth table

| Endpoint | Current implementation | Evidence |
|---|---|---|
| `/predict` | real CatBoost + real isotonic calibration + real LightGBM q05/q50/q95 + real split-CQR adjustment | MODEL OUTPUT |
| `/explain` | real CatBoost local SHAP; optional legacy-edge overlap | MODEL OUTPUT + EXPLORATORY ONLY |
| `/recommend` | decision rules plus parameterized cost/efficacy assumptions | SIMULATED SCENARIO |

## 11. Claims for the committee

### Safe academic claim

> “The contribution is an integrated, leakage-aware decision-intelligence prototype combining temporally calibrated prediction, conditional severity uncertainty, real local model explanations, drift-aware research modules, and explicitly simulated human-governed interventions.”

### Safe business claim

> “The demo prioritizes which shipments merit attention and lets decision-makers stress-test intervention economics; it does not claim realized ROI without prospective operational deployment.”

### Production boundary

The repository is **not production-ready**. Production requires recent prospective data, live enterprise integrations, security/auth, observability, deployment controls, current-domain validation, and measured intervention outcomes.
