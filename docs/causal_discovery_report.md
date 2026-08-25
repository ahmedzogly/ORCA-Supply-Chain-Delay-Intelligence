# Exploratory Causal Analysis

> **Evidence scope: EXPLORATORY ONLY — hypothesis generation, not causal identification.**

The historical experiment used the PC algorithm with a Fisher-Z conditional-independence test across temporal folds. It produced recurring graph edges such as `Fulfill Via -> Delay_Days`. These outputs are retained because they may help generate operational hypotheses for later study.

## Methodological limitation

Several variables are categorical and were encoded numerically for the legacy discovery run. Fisher-Z is fundamentally a partial-correlation test and its assumptions are not naturally satisfied by arbitrary integer encodings of nominal categories. Therefore, edge stability does **not** establish a causal effect, intervention efficacy, or treatment value.

## Allowed interpretation

- “A stable exploratory graph edge was observed across folds.”
- “This relationship is a candidate hypothesis for future causal validation.”
- Predictive SHAP and graph edges may be shown side by side only when clearly labeled as different evidence types.

## Prohibited interpretation

- “The model proved that fulfillment mode causes delay.”
- “Changing the variable will reduce delay by X days.”
- “The causal module validates the financial benefit of an intervention.”

## Production/research upgrade path

A stronger causal study should define a treatment, outcome, confounder set, temporal identification strategy, positivity/overlap checks, and either a credible quasi-experimental design or domain-appropriate estimators/tests for mixed data. Any effect estimate should then be stress-tested with refutation/sensitivity analyses and, where possible, prospective intervention data.
