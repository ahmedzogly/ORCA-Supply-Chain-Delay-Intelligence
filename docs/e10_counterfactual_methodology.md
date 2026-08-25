# Phase 2 — Experiment E10: Counterfactual Policy Evaluation Methodology

**Status:** IMPLEMENTED & VALIDATED ON DEVELOPMENT COHORT  
**Target:** USAID SCMS Supply Chain Delay Intelligence  
**Configuration Reference:** `configs/e10_counterfactual.yaml`  
**Development Cohort:** $N=7,306$ records ($T_{\text{pred}} \le \text{2014-08-24}$, 90-day embargo gap)  
**Quarantined Final Holdout:** $N=1,013$ records ($T_{\text{pred}} > \text{2014-08-24}$, strictly unaccessed)

---

## 1. Executive Overview & Problem Formulation

Experiment E10 evaluates whether the E8 cost-sensitive champion policy (`E8-C_tuned_gamma`, $\gamma^*=1.20$) remains economically preferable against alternative operational policies under evolving shipment states, dynamic operational friction, and uncertain action effects.

### 1.1 Mandatory Scientific Non-Causal Guardrails
> **SCIENTIFIC NOTICE:** Historical SCMS supply chain records lack randomized treatment assignments and explicit intervention logs. All counterfactual transitions, risk reductions, and cost savings evaluated in Experiment E10 represent synthetic scenario simulations parameterized by explicit domain assumptions. No observational claims of actual historical intervention efficacy or true causal treatment effects are asserted.

All artifacts, states, and data pipelines strictly adhere to four immutable provenance tiers:
1. `OBSERVED_SCMS_DATA`: Historical shipment features, timestamps, and observational outcomes.
2. `SYNTHETIC_E9_STATE`: Observable dynamic operational state variables and telemetry alerts.
3. `SIMULATED_COUNTERFACTUAL`: Simulated post-intervention states and deterministic transition outcomes.
4. `SIMULATED_COST`: Synthetic business economic costs computed under scenario parameterizations.

---

## 2. State Space and Transition Dynamics

### 2.1 Observable Shipment State $S_i(t)$
Each shipment $i$ at prediction anchor $T_{\text{pred}}$ is characterized by an immutable observable state vector:
$$S_i(t) = \left( V_i, \kappa_i, \lambda_{\text{mode}}, \text{channel}_i, \hat{p}_i, \hat{D}_i, W_i, \mathbf{z}_i \right)$$

- **Monetary Value ($V_i$):** Line item value in USD.
- **Clinical Criticality ($\kappa_i$):** Domain priority multiplier:
  $$\kappa_i = 1.0 + \delta_{\text{first\_line}} \cdot \mathbb{I}(\text{FirstLine}) + \delta_{\text{pediatric}} \cdot \mathbb{I}(\text{Pediatric}) + \delta_{\text{arv}} \cdot \mathbb{I}(\text{ARV})$$
- **Transport Mode Friction ($\lambda_{\text{mode}}$):** Air (1.00), Air Charter (0.90), Truck (1.10), Ocean (1.25).
- **Fulfillment Channel ($\text{channel}_i$):** `Direct Drop` vs `From RDC`.
- **Delay Risk ($\hat{p}_i$):** Calibrated delay probability from Stage 5 CatBoost Champion with Isotonic Calibration ($\hat{p}_i \in [0, 1]$).
- **Expected Baseline Delay ($\hat{D}_i$):** Estimated unmitigated delay duration (calendar days).
- **Conformal Uncertainty Width ($W_i$):** CQR 80% prediction interval width ($\hat{y}_{\text{high}} - \hat{y}_{\text{low}}$).
- **IoT Telemetry Signals ($\mathbf{z}_i$):** Real-time continuous monitoring (temperature, route deviation).

**Zero Hidden Scenario Leakage:** The state structure contains *zero* hidden regime labels (e.g. $S_0..S_6$ from E9). Decision policies operate strictly on observable attributes.

### 2.2 Deterministic State Transitions
Applying operational action $a \in \mathcal{A}$ deterministically transitions state $S_i$:
- **Residual Delay:** $\tilde{D}_i(a) = \max(0, \hat{D}_i + \Delta D(a) \cdot e_{\text{mult}})$
- **Residual Risk:** $\tilde{p}_i(a) = \text{clip}\left(\hat{p}_i \cdot (1 + \Delta R(a) \cdot e_{\text{mult}}), 0.0, 1.0\right)$
- **Residual Uncertainty:** $\tilde{W}_i(a) = \max(0.1, W_i \cdot (1 + \Delta W(a) \cdot e_{\text{mult}}))$

---

## 3. Operational Policy Suite (P0–P5)

| Policy ID | Policy Name | Action Code | Frozen Action Effect | Cost Formula |
|---|---|---|---|---|
| **P0** | Default Control | `NO_ACTION` | $\Delta D=0, \Delta R=0, \Delta W=0$ | $\$0.00$ |
| **P1** | E8 Cost-Sensitive | `E8_COST_SENSITIVE` | If $\hat{p}_i \ge \tau_i^*$: $\Delta D=-3.0\text{d}$ | $C_{\text{triage}} + C_{\text{expedite}}$ |
| **P2** | Proactive Expediting | `EXPEDITE` | $\Delta D=-3.0\text{d}$ | $c_{\text{exp\_base}} + \gamma_{\text{exp}} \cdot V_i$ |
| **P3** | Mode Review | `TRANSPORT_MODE_REVIEW` | $\Delta D=-2.0\text{d}$ | $c_{\text{mode\_base}} + \beta_{\text{mode}} \cdot \ln(1 + V_i)$ |
| **P4** | Supplier Escalation | `SUPPLIER_ESCALATION` | $\Delta R=-15\%$ | $c_{\text{esc\_base}} + C_{\text{inquiry}}$ |
| **P5** | Human Review | `HUMAN_REVIEW` | $\Delta W=-50\%$ | $c_{\text{triage\_base}} + \beta_{\text{audit}} \cdot \ln(1 + V_i)$ |
| **Oracle** | Offline Omniscient | `ORACLE_POLICY` | $a^* = \arg\min_a \mathbb{E}[\text{Cost}(a \mid S_i)]$ | Ex-post isolated benchmark |

### 3.1 Offline Isolated Oracle
The `OfflineOraclePolicy` evaluates the omniscient lower bound:
$$a^*_i = \arg\min_{a \in \{P_0, P_2, P_3, P_4, P_5\}} \mathbb{E}[\text{Cost}(a \mid S_i)]$$
**Strict Isolation Contract:** The Oracle engine is strictly isolated and evaluated ex-post exclusively to establish theoretical lower bounds, Oracle Gap, and Policy Regret:
$$\text{Regret}(P_k, i) = \mathbb{E}[\text{Cost}(P_k \mid S_i)] - \mathbb{E}[\text{Cost}(\text{Oracle} \mid S_i)] \ge 0$$
$$\text{Oracle\_Gap}(P_k) = \frac{1}{N} \sum_{i=1}^N \text{Regret}(P_k, i)$$

---

## 4. Expected Realized Cost Formulation

The expected business cost of selecting action $a$ given observable state $S_i$ is:
$$\mathbb{E}[\text{Cost}(a \mid S_i)] = C_{\text{action}}(a, i) + \mathbb{E}[C_{\text{residual\_delay}}(a \mid S_i)] + \mathbb{E}[C_{\text{risk}}(a \mid S_i)]$$

Where:
1. $C_{\text{action}}(a, i)$: Direct operational expenditure.
2. $\mathbb{E}[C_{\text{residual\_delay}}(a \mid S_i)] = \tilde{p}_i(a) \cdot \left( c_{\text{daily\_base}} \cdot \lambda_{\text{mode}} + \rho_{\text{value}} \cdot V_i \right) \cdot \tilde{D}_i(a)$
3. $\mathbb{E}[C_{\text{risk}}(a \mid S_i)] = \tilde{p}_i(a) \cdot c_{\text{fixed\_stockout}} \cdot \kappa_i$

---

## 5. Temporal Development Backtest Results ($N=7,306$)

Across the 5-fold expanding-window cross-validation with 90-day embargo gap:

| Policy | Mean Expected Cost ($) | Mean Net Benefit ($) | Total Oracle Gap ($) | Mean Regret ($) | Intervention Rate (%) | Hysteresis Stability (%) |
|---|---|---|---|---|---|---|
| **P0: NO_ACTION** | $544.41 | $0.00 | $73,469.94 | $24.14 | 0.0% | 100.0% |
| **P1: E8_COST_SENSITIVE** | $759.79 | -$215.37 | $728,854.78 | $239.52 | 53.4% | 53.1% |
| **P2: EXPEDITE** | $544.41 | $0.00 | $73,469.94 | $24.14 | 0.0% | 100.0% |
| **P3: MODE_REVIEW** | $544.39 | +$0.02 | $73,401.65 | $24.12 | 0.3% | 100.0% |
| **P4: SUPPLIER_ESCALATION** | $541.44 | +$2.97 | $64,422.75 | $21.17 | 1.9% | 100.0% |
| **P5: HUMAN_REVIEW** | $546.93 | -$2.52 | $81,135.70 | $26.66 | 3.4% | 100.0% |
| **Oracle (Benchmark)** | **$520.27** | **+$24.14** | **$0.00** | **$0.00** | **17.6%** | **100.0%** |

### Key Methodological Insights:
1. **Unconstrained Blanket Expediting Inefficiency:** Universal expediting without capacity constraints or risk calibration incurs excessive action costs ($C_{\text{expedite}}$) on false-positive shipments.
2. **Targeted Supplier Escalation Value:** P4 (Supplier Escalation) demonstrates positive net economic benefit (+$2.97/shipment) because targeted pre-dispatch vendor SLA enforcement on Direct Drop shipments has low administrative friction relative to stockout avoidance.
3. **Oracle Frontier:** The theoretical optimal policy achieves an intervention rate of 17.6% with mean cost of $520.27/shipment, yielding +$24.14/shipment net savings.

---

## 6. Review Budget Prioritization ($K \in \{5\%, 10\%, 20\%\}$)

Under capacity constraints ranking shipments by Expected Net Benefit:
$$\text{Score}_i = \max_{a \in \mathcal{A}_{\text{intervene}}} \left( \mathbb{E}[\text{Cost}(\text{NO\_ACTION} \mid S_i)] - \mathbb{E}[\text{Cost}(a \mid S_i)] \right)$$

On the full development cohort ($N=7,306$):
- **$K = 5\%$ Review Budget ($M=365$ shipments):** $135,344.82 USD total net economic savings (100.0% utilization).
- **$K = 10\%$ Review Budget ($M=730$ shipments):** $198,047.40 USD total net economic savings (100.0% utilization).
- **$K = 20\%$ Review Budget ($M=1,461$ capacity):** $228,209.93 USD total net economic savings (1,301 allocated, 89.0% utilization, saturated when positive net benefit instances are exhausted).

---

## 7. Artifacts & Invariance Summary

- `artifacts/phase2/e10/e10_dev_evaluation_results.parquet`: 217,329 evaluation records across 5 CV folds and full dev cohort.
- `artifacts/phase2/e10/e10_sensitivity_grid_results.parquet`: 189 sensitivity grid evaluation summary rows.
- Baseline models and datasets verified 100% SHA-256 invariant.
- Final holdout ($N=1,013$, $T_{\text{pred}} > \text{2014-08-24}$) verified strictly quarantined.
