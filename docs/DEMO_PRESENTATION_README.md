# Delay Intelligence — Demo Presentation Guide

## How to launch

```bash
# Terminal 1: Start the API server
python run.py --api

# Terminal 2: Start the Streamlit dashboard
python run.py --dashboard
```

The dashboard opens at **http://localhost:8501**.

> **Note:** Both the API server and the dashboard must be running simultaneously.
> The dashboard uses a FastAPI TestClient internally, but if using the standalone
> API, launch it on port 8000 first.

## Recommended live demo flow (5 minutes)

### 1. Landing page (30 seconds)
- Show the title: **Delay Intelligence**
- Point out the three evidence badges: REAL DATA, MODEL OUTPUT, SIMULATED SCENARIO
- Highlight the inference pipeline summary
- Emphasize: "Research / Demo Prototype — Not a Production Control Tower"
- Click **Open Executive Control Tower**

### 2. Executive Control Tower (60 seconds)
- Show the portfolio KPIs: 100 shipments monitored, 9 above threshold, 7.5% mean risk
- Walk through the risk distribution chart — most shipments are low risk
- Highlight the **Priority Shipments** table — top 10 by calibrated risk
- Point out: "Highest-risk shipment is **83922** at 46.7%"
- Click **Investigate Shipment 83922**

### 3. Shipment Risk Explorer (90 seconds)
- Show the hero KPI cards: 46.7% Late Probability, FLAG, 15.8 days expected delay
- Walk through the **SHAP visualization** — horizontal bars showing which features increase/decrease risk
- Explain: "SHAP explains the model prediction; it does not establish causation"
- Open the **Real pre-outcome shipment features** expander to show raw data
- Scroll to **Exploratory causal hypotheses** — clearly labeled EXPLORATORY ONLY

### 4. Decision & Action Center (60 seconds)
- Show model assessment: same shipment, 46.7% risk, WATCH tier
- Adjust the **scenario sliders** (delay cost, intervention cost, efficacy)
- Show how the scenario net benefit changes dynamically
- Emphasize: "SIMULATED SCENARIO — configurable planning assumptions, not accounting facts"
- Show the policy recommendation and robustness assessment

### 5. Portfolio Intelligence (30 seconds)
- Show mean late risk, median conditional delay, high-risk count
- Walk through risk by fulfillment channel and shipment mode
- Show the severity distribution and top risk drivers

### 6. Model Evidence (30 seconds)
- Show **Predictive Performance** tab: PR-AUC 0.2696 (4.5× improvement over random)
- Switch to **Uncertainty** tab: 95.1% empirical coverage with 54.9 day intervals
- Note the **Limitations** tab for academic transparency
- Show Historical Baseline in expander — preserved for provenance

## What claims are safe

### Approved academic language
> "The contribution is an integrated, leakage-aware decision-intelligence prototype
> combining temporally calibrated prediction, conditional severity uncertainty,
> real local model explanations, drift-aware research modules, and explicitly
> simulated human-governed interventions."

### Approved business language
> "The demo prioritizes which shipments merit attention and lets decision-makers
> stress-test intervention economics; it does not claim realized ROI without
> prospective operational deployment."

### Positioning statement
> "Research-validated Decision Intelligence Prototype with a Production Roadmap"

## What NOT to claim

- ❌ **No realized ROI claim** — All financial impacts are scenario-based estimates under configurable assumptions
- ❌ **No production deployment claim** — This is a research/demo prototype
- ❌ **No external DataCo/Olist validation claim** — Adapter protocols exist but no empirical validation has been produced
- ❌ **No identified causal effect claim** — PC/Fisher-Z edges are hypothesis-generating only
- ❌ **No "enterprise-grade platform" claim** — Use "research-validated prototype" instead
- ❌ **No "savings achieved" language** — Use "scenario-based estimated economic impact"
- ❌ **No "guaranteed coverage" language** — Use "observed empirical coverage"

## Evidence taxonomy

| Label | Meaning |
|-------|---------|
| **REAL DATA** | Historical SCMS source records |
| **MODEL OUTPUT** | Fitted-model prediction/explanation |
| **SIMULATED SCENARIO** | Parameterized action/cost/counterfactual |
| **EXPLORATORY ONLY** | Legacy causal-discovery hypothesis |
| **NOT VALIDATED** | Adapter/protocol exists but no target-domain empirical test |
