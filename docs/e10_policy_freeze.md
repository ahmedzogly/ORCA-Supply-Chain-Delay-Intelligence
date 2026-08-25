# Phase 2 — Experiment E10: Counterfactual Policy Evaluation Design Freeze

**Status:** FROZEN AND IMMUTABLE  
**Freeze Timestamp (UTC):** 2026-08-22T16:12:00Z  
**Configuration File:** `configs/e10_counterfactual.yaml`  
**Pre-Freeze Manifest:** `artifacts/phase2/e10/e10_pre_freeze_manifest.json`  

---

## 1. Executive Summary & Objective

Experiment E10 conducts a scenario-based counterfactual policy evaluation within the USAID SCMS Supply Chain Delay Intelligence platform. The goal is to determine whether the E8 cost-sensitive champion policy (`E8-C_tuned_gamma`, $\gamma^*=1.20$) remains economically preferable against alternative operational policies under evolving E9 shipment states, dynamic disruption regimes, and uncertain action effects.

To maintain strict scientific integrity:
1. **Zero Retraining / Zero Baseline Alteration:** All Stage 0–13 models, E6.5 drift detectors, E7 adaptive conformal recalibrators, E8 cost models, and E9 digital twin simulation components are frozen.
2. **Strict Chronological Evaluation:** The development cohort ($N=7,306$, $T \le \text{2014-08-24}$) with 90-day embargo periods is used for all preliminary evaluation, parameter checks, and sensitivity grids. The 365-day final holdout ($N=1,013$, $T > \text{2014-08-24}$) is strictly quarantined and accessed exactly once.
3. **No Causal Claims from Observational Data:** Historical SCMS data lacks explicit intervention logs. All evaluated counterfactual transitions and benefits are synthetic simulation estimates parameterized by frozen assumptions.

---

## 2. Scientific Guardrails & Data Provenance

### 2.1 Non-Causal Disclaimer
> **MANDATORY NOTICE:** Historical SCMS supply chain records lack randomized treatment assignments and explicit intervention logs. All counterfactual transitions, risk reductions, and cost savings evaluated in Experiment E10 represent synthetic scenario simulations parameterized by explicit domain assumptions. No observational claims of actual historical intervention efficacy or true causal treatment effects are asserted.

### 2.2 Provenance Tagging Schema
Every record, column, and artifact in E10 is explicitly classified into one of four immutable provenance tiers:

| Provenance Tag | Definition & Scope | Example Fields |
|---|---|---|
| `OBSERVED_SCMS_DATA` | Ground-truth historical shipment features, timestamps, and observational outcomes. | `Line Item Value`, `Country`, `Shipment Mode`, `Delay_Days`, `Delay_Flag` |
| `SYNTHETIC_E9_STATE` | Dynamic operational state variables and telemetry anomalies synthesized across E9 regimes $S_0..S_6$. | `iot_temperature_c`, `iot_route_deviation_km`, `current_ETA`, `ETA_shock_flag` |
| `SIMULATED_COUNTERFACTUAL` | Model-simulated counterfactual transitions and post-intervention shipment states. | $\tilde{D}_i(a)$, $\tilde{p}_i(a)$, $\tilde{W}_i(a)$, `action_selected` |
| `SIMULATED_COST` | Synthetic business economic costs computed under scenario cost parameterizations. | $C_{\text{action}}$, $C_{\text{residual\_delay}}$, $C_{\text{risk}}$, $\mathbb{E}[\text{Cost}(a \mid S_i)]$ |

---

## 3. Mathematical Formulation of State, Transitions, and Costs

### 3.1 Observable Shipment State $S_i(t)$
Each shipment $i$ at prediction anchor $T_{\text{pred}}$ is characterized by an observable state vector:
$$S_i(t) = \left( V_i, \kappa_i, \lambda_{\text{mode}}, \text{channel}_i, \hat{p}_i, \hat{D}_i, W_i, \mathbf{z}_i \right)$$
where:
- $V_i$: Line item value (USD) [`Line Item Value`].
- $\kappa_i$: Clinical criticality multiplier:
  $$\kappa_i = 1.0 + \delta_{\text{first\_line}} \cdot \mathbb{I}(\text{FirstLine}) + \delta_{\text{pediatric}} \cdot \mathbb{I}(\text{Pediatric}) + \delta_{\text{arv}} \cdot \mathbb{I}(\text{ARV})$$
- $\lambda_{\text{mode}}$: Transport mode cost factor (Air: 1.00, Air Charter: 0.90, Truck: 1.10, Ocean: 1.25).
- $\text{channel}_i$: Fulfillment channel (`Direct Drop` vs `From RDC`).
- $\hat{p}_i$: Calibrated probability of delay from Stage 5 CatBoost Champion with Isotonic Regression ($\hat{p}_i \in [0, 1]$).
- $\hat{D}_i$: Expected baseline delay duration (calendar days) from Stage 5/6 regression model.
- $W_i$: Conformal uncertainty interval width ($W_i = \hat{y}_{\text{high}} - \hat{y}_{\text{low}}$) from Stage 6/7 CQR.
- $\mathbf{z}_i$: Real-time IoT monitoring telemetry (if active in E9 simulation; monitoring-only signal).

### 3.2 State Transition Dynamics $f(S_i, a)$
Applying action $a \in \mathcal{A}$ transitions the state deterministically:
- Counterfactual Residual Delay:
  $$\tilde{D}_i(a) = \max\left(0, \hat{D}_i + \Delta D(a)\right)$$
- Counterfactual Residual Delay Probability:
  $$\tilde{p}_i(a) = \text{clip}\left(\hat{p}_i \cdot (1 + \Delta R(a)), 0.0, 1.0\right)$$
- Counterfactual Conformal Uncertainty Width:
  $$\tilde{W}_i(a) = \max\left(0.1, W_i \cdot (1 + \Delta W(a))\right)$$

### 3.3 Expected Realized Cost Formulation
The expected economic cost of taking action $a$ given observable state $S_i$ is:
$$\mathbb{E}[\text{Cost}(a \mid S_i)] = C_{\text{action}}(a, i) + \mathbb{E}[C_{\text{residual\_delay}}(a \mid S_i)] + \mathbb{E}[C_{\text{risk}}(a \mid S_i)]$$

Where the individual components are defined as:
1. **Action Cost $C_{\text{action}}(a, i)$:** Direct operational cost incurred by executing action $a$.
2. **Expected Residual Delay Cost $\mathbb{E}[C_{\text{residual\_delay}}(a \mid S_i)]$:**
   $$\mathbb{E}[C_{\text{residual\_delay}}(a \mid S_i)] = \tilde{p}_i(a) \cdot \left( c_{\text{daily\_base}} \cdot \lambda_{\text{mode}} + \rho_{\text{value}} \cdot V_i \right) \cdot \tilde{D}_i(a)$$
3. **Expected Residual Risk / Stockout Cost $\mathbb{E}[C_{\text{risk}}(a \mid S_i)]$:**
   $$\mathbb{E}[C_{\text{risk}}(a \mid S_i)] = \tilde{p}_i(a) \cdot c_{\text{fixed\_stockout}} \cdot \kappa_i$$

---

## 4. Operational Action Space & Frozen Action Effects

The policy evaluation spans six operational policies ($P_0..P_5$) and one isolated offline benchmark ($P_{\text{oracle}}$):

```
+---------------------------------------------------------------------------------------------------+
| Policy / Action Code         | Action Type       | Frozen Efficacy               | Cost Formula   |
+---------------------------------------------------------------------------------------------------+
| P0: NO_ACTION                | Passive Control   | Delta D=0, Delta R=0, Delta W=0| 0.0 USD        |
| P1: E8_COST_SENSITIVE        | Selective Expedite| If p_i >= tau*_i: Delta D=-3.0| C_triage + C_exp|
| P2: EXPEDITE                 | Express Logistics | Delta D = -3.0 days           | c_exp + gamma*V|
| P3: TRANSPORT_MODE_REVIEW    | Routing / Modal   | Delta D = -2.0 days           | c_mode+beta*lnV|
| P4: SUPPLIER_ESCALATION      | Vendor SLA Mgmt   | Delta R = -0.15 (-15% risk)   | c_esc + c_inq  |
| P5: HUMAN_REVIEW             | Expert Triage     | Delta W = -0.50 (-50% uncert) | c_tri+beta*lnV |
| OFFLINE: ORACLE_POLICY       | Offline Benchmark | Omniscient optimal action     | Isolated       |
+---------------------------------------------------------------------------------------------------+
```

### 4.1 Detailed Policy Definitions

#### P0: `NO_ACTION`
- **Description:** Business-as-usual default fulfillment with zero intervention.
- **Action Cost:** $C_{\text{action}} = 0.0$ USD.
- **State Transition:** $\Delta D = 0.0$, $\Delta R = 0.0$, $\Delta W = 0.0$.
- **Expected Cost:** $\mathbb{E}[\text{Cost}(P_0 \mid S_i)] = \hat{p}_i \cdot \left( c_{\text{daily\_base}} \cdot \lambda_{\text{mode}} + \rho_{\text{value}} \cdot V_i \right) \cdot \hat{D}_i + \hat{p}_i \cdot c_{\text{fixed\_stockout}} \cdot \kappa_i$.

#### P1: `E8_COST_SENSITIVE`
- **Description:** E8 Champion cost-sensitive threshold rule ($\gamma^* = 1.20$). Triggers proactive intervention if calibrated risk exceeds instance-dependent threshold $\tau_i^*$:
  $$\tau_i^* = \frac{\text{FP\_Cost}(i)}{\gamma^* \cdot \text{Net\_Benefit}(i) + \text{FP\_Cost}(i)}$$
  where $\text{FP\_Cost}(i) = C_{\text{triage\_base}} + \beta_{\text{audit}} \cdot \ln(1 + V_i) + C_{\text{inquiry}}(\text{channel}_i)$, and $\text{Net\_Benefit}(i)$ is expected stockout savings from reducing delay by $\Delta_{\text{days\_saved}}$.
- **Action Cost:** If $p_i \ge \tau_i^*$, incurs $C_{\text{action}} = \text{FP\_Cost}(i) + c_{\text{expedite\_base}} + \gamma_{\text{expedite}} \cdot V_i$; else $0.0$.
- **State Transition:** If triggered, $\Delta D = -3.0$ days; else $\Delta D = 0.0$.

#### P2: `EXPEDITE`
- **Description:** Universal express freight and priority carrier booking.
- **Frozen Action Effect:** $\Delta D = -3.0$ calendar days (Source: `docs/e9_simulation_assumptions.json`).
- **Action Cost:** $C_{\text{action}}(P_2, i) = c_{\text{expedite\_base}} + \gamma_{\text{expedite}} \cdot V_i$.

#### P3: `TRANSPORT_MODE_REVIEW`
- **Description:** Formal modal shift / carrier re-routing review across Air, Sea, Truck, and RDC.
- **Frozen Action Effect:** $\Delta D = -2.0$ calendar days (Source: `docs/e9_simulation_assumptions.json`).
- **Action Cost:** $C_{\text{action}}(P_3, i) = c_{\text{mode\_review\_base}} + \beta_{\text{mode}} \cdot \ln(1 + V_i)$.

#### P4: `SUPPLIER_ESCALATION`
- **Description:** Vendor SLA enforcement, direct procurement inquiry, and dedicated batch tracking.
- **Frozen Action Effect:** $\Delta R = -0.15$ (-15% relative risk reduction) (Source: `docs/e9_simulation_assumptions.json`).
- **Action Cost:** $C_{\text{action}}(P_4, i) = c_{\text{escalation\_base}} + c_{\text{inquiry}}(\text{channel}_i)$ (Direct Drop: $c_{\text{direct\_inquiry}}$, RDC: $c_{\text{rdc\_inquiry}}$).

#### P5: `HUMAN_REVIEW`
- **Description:** Control-tower expert investigation and manual audit triage. Resolves epistemic uncertainty.
- **Frozen Action Effect:** $\Delta W = -0.50$ (-50% reduction in conformal uncertainty interval width $W_i$).
- **Action Cost:** $C_{\text{action}}(P_5, i) = c_{\text{triage\_base}} + \beta_{\text{audit}} \cdot \ln(1 + V_i)$.

#### `ORACLE_POLICY` (Offline Reference Only)
- **Description:** Theoretical cost-minimizing oracle action:
  $$a^*_i = \arg\min_{a \in \{P_0, P_2, P_3, P_4, P_5\}} \mathbb{E}[\text{Cost}(a \mid S_i)]$$
- **Isolation Contract:** **STRICTLY OFFLINE POST-FREEZE ONLY.** The Oracle policy must never be imported, referenced, or executed during online policy execution, threshold tuning, or operational selection. It is evaluated post-freeze exclusively to establish theoretical lower bounds, Oracle Gap, and Policy Regret.

---

## 5. Policy Regret & Economic Performance Metrics

For each policy $P_k$ and shipment $i$:
1. **Simulated Net Benefit:**
   $$\text{Benefit}(P_k, i) = \mathbb{E}[\text{Cost}(P_0 \mid S_i)] - \mathbb{E}[\text{Cost}(P_k \mid S_i)]$$
2. **Policy Regret:**
   $$\text{Regret}(P_k, i) = \mathbb{E}[\text{Cost}(P_k \mid S_i)] - \mathbb{E}[\text{Cost}(\text{Oracle} \mid S_i)] \ge 0$$
3. **Oracle Gap:**
   $$\text{Oracle\_Gap}(P_k) = \frac{1}{N} \sum_{i=1}^N \text{Regret}(P_k, i)$$
4. **Policy Switching Rate & Hysteresis:**
   Proportion of shipments where the optimal action changes between standard state $S_0$ and disrupted state $S_j$.

---

## 6. Cost Scenario Parameterizations

The cost parameters are frozen across Low, Base, and High scenarios:

| Parameter | Symbol | Low | Base | High | Unit |
|---|---|---|---|---|---|
| Daily Operational Penalty | $c_{\text{daily\_base}}$ | 50.0 | 150.0 | 350.0 | USD/day |
| Value Holding Rate | $\rho_{\text{value}}$ | 0.0005 | 0.0010 | 0.0020 | 1/day |
| Fixed Stockout Penalty | $c_{\text{fixed\_stockout}}$ | 200.0 | 500.0 | 1500.0 | USD |
| Base Triage Cost | $c_{\text{triage\_base}}$ | 25.0 | 50.0 | 100.0 | USD |
| Value Audit Scaling | $\beta_{\text{audit}}$ | 5.0 | 10.0 | 20.0 | USD/log(USD) |
| Direct Drop Inquiry Cost | $c_{\text{direct\_inquiry}}$ | 15.0 | 30.0 | 60.0 | USD |
| RDC Inquiry Cost | $c_{\text{rdc\_inquiry}}$ | 5.0 | 10.0 | 20.0 | USD |
| Base Expediting Fee | $c_{\text{expedite\_base}}$ | 250.0 | 500.0 | 1000.0 | USD |
| Expedite Value Surcharge | $\gamma_{\text{expedite}}$ | 0.0020 | 0.0050 | 0.0100 | % of Value |
| Base Mode Review Fee | $c_{\text{mode\_review\_base}}$ | 100.0 | 200.0 | 400.0 | USD |
| Mode Review Value Scaling | $\beta_{\text{mode}}$ | 10.0 | 20.0 | 40.0 | USD/log(USD) |
| Base Escalation Fee | $c_{\text{escalation\_base}}$ | 75.0 | 150.0 | 300.0 | USD |
| Assumed Delay Duration | $\bar{D}$ | 10.0 | 12.0 | 15.0 | days |
| Days Saved (Efficacy) | $\Delta_{\text{days\_saved}}$ | 4.0 | 5.0 | 6.0 | days |
| Criticality: First Line | $\delta_{\text{first\_line}}$ | +0.30 | +0.30 | +0.40 | multiplier |
| Criticality: Pediatric | $\delta_{\text{pediatric}}$ | +0.20 | +0.20 | +0.30 | multiplier |
| Criticality: ARV | $\delta_{\text{arv}}$ | +0.10 | +0.15 | +0.20 | multiplier |

---

## 7. Chronological Evaluation & Quarantine Protocol

```
+---------------------------------------------------------------------------------------------------+
| DEVELOPMENT COHORT (N=7,306, T <= 2014-08-24)            | 90-d Embargo | 365-d FINAL HOLDOUT     |
| 5-Fold Expanding-Window Cross-Validation                 | (Purged)     | (N=1,013, STRICT PASS)  |
+---------------------------------------------------------------------------------------------------+
```

1. **Development Cohort ($N=7,306$):**
   - Spans 2006-04-19 through 2014-08-24.
   - Evaluated via 5-fold expanding-window cross-validation with minimum 730-day training history.
   - Used for all preliminary evaluation, sensitivity grids, and budget optimization.
2. **Embargo Gap:** 90 calendar days strictly purged between training origins and test evaluation windows.
3. **Quarantined Final Holdout ($N=1,013$):**
   - Spans 2014-08-24 through 2015-09-30 (365 days).
   - STRICTLY QUARANTINED.
   - Evaluated exactly once in Milestone 5 post-freeze with zero retuning or modification.

---

## 8. Capacity-Constrained Review Budgeting

Under operational capacity limits $K \in \{0.05, 0.10, 0.20\}$ (5%, 10%, 20% of shipment volume):
1. **Prioritization Score:**
   $$\text{Score}_i = \max_{a \in \{P_2, P_3, P_4, P_5\}} \left( \mathbb{E}[\text{Cost}(P_0 \mid S_i)] - \mathbb{E}[\text{Cost}(a \mid S_i)] \right)$$
2. **Budget Allocation:** Top $\lfloor K \cdot N \rfloor$ shipments ranked by $\text{Score}_i$ receive their optimal intervention; remaining shipments default to $P_0$ (NO_ACTION).

---

## 9. Multi-Dimensional Sensitivity Analysis Grid ($3 \times 3$)

To evaluate policy robustness under assumption perturbations, a $3 \times 3$ sensitivity grid is defined:

| Grid Cell | Action Cost Multiplier | Action Efficacy Multiplier | Description |
|---|---|---|---|
| `Cost_Low__Eff_Low` | 0.50x | 0.50x | Cheap actions, weak efficacy ($\Delta D = -1.5\text{d}, \Delta R = -0.075$) |
| `Cost_Low__Eff_Base` | 0.50x | 1.00x | Cheap actions, standard efficacy |
| `Cost_Low__Eff_High` | 0.50x | 1.50x | Cheap actions, strong efficacy ($\Delta D = -4.5\text{d}, \Delta R = -0.225$) |
| `Cost_Base__Eff_Low` | 1.00x | 0.50x | Standard cost, weak efficacy |
| `Cost_Base__Eff_Base` | 1.00x | 1.00x | Standard cost, standard efficacy (Baseline Reference) |
| `Cost_Base__Eff_High` | 1.00x | 1.50x | Standard cost, strong efficacy |
| `Cost_High__Eff_Low` | 2.00x | 0.50x | Expensive actions, weak efficacy (Adversarial stress) |
| `Cost_High__Eff_Base` | 2.00x | 1.00x | Expensive actions, standard efficacy |
| `Cost_High__Eff_High` | 2.00x | 1.50x | Expensive actions, strong efficacy |

---

## 10. Cryptographic Baseline Invariance Audit

All 36 baseline artifacts across Stages 0–13, E6.5, E7, E8, and E9 have been cryptographically audited and verified 100% bitwise invariant against `artifacts/phase2/e9/e9_immutability_manifest.json`:

```json
{
  "catboost_champion.cbm": "261dc20da9ea3eb9fc53dd543c2bb837d9d6f613f8b81b71e13e1e2b99584ea4",
  "cqr_calibration.json": "36f3b10fb80f5691edb41e51251241ce92c0c914729861bd8d1e7c0fe42be284",
  "e8_final_holdout_results.parquet": "e88a7aeb2d182c04ddcd2db452fa9b6ee9417d785e03176e38e96b669be68501",
  "e8_frozen_policy.json": "a5f127c1d433904ce0b31ef5c71ed10b35490ba6d51f82157b5b6d17692a0b3f"
}
```

Full 36-artifact manifest recorded at `artifacts/phase2/e10/e10_pre_freeze_manifest.json`.

---

## 11. Design Freeze Certification

The undersigned implementer certifies that:
1. `configs/e10_counterfactual.yaml` is formally created and frozen.
2. All policy actions ($P_0..P_5$) and offline Oracle contracts are mathematically specified without ambiguity.
3. E9 action effects are frozen without tuning against holdout data.
4. All baseline hashes match historical manifests bit-for-bit.
5. All 558 existing repository tests pass 100%.

**DESIGN FREEZE STATUS: APPROVED & LOCKED**
