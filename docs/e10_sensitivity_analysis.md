# Phase 2 — Experiment E10: Multi-Dimensional Sensitivity Analysis

**Status:** EVALUATED & SYNTHESIZED ON DEVELOPMENT COHORT  
**Target:** USAID SCMS Supply Chain Delay Intelligence  
**Dataset:** Development Cohort ($N=7,306$, $T_{\text{pred}} \le \text{2014-08-24}$)  
**Artifact:** `artifacts/phase2/e10/e10_sensitivity_grid_results.parquet`

---

## 1. Multi-Dimensional Grid Design ($3 \times 3$)

To rigorously evaluate policy stability and economic robustness against modeling assumption errors and operational friction, Experiment E10 implements a $3 \times 3$ sensitivity grid combining:

1. **Action Cost Multipliers ($c_{\text{mult}}$):**
   - **Low (0.50x):** -50% reduction in intervention fees (carrier discounts, automated triage).
   - **Base (1.00x):** Calibrated baseline operational costs.
   - **High (2.00x):** +100% inflation in intervention fees (severe supply chain capacity crunch, emergency freight surcharges).

2. **Action Efficacy Multipliers ($e_{\text{mult}}$):**
   - **Low (0.50x):** -50% degradation in mitigation power ($\Delta D = -1.5\text{d}, \Delta R = -7.5\%$).
   - **Base (1.00x):** Calibrated standard efficacy ($\Delta D = -3.0\text{d}, \Delta R = -15.0\%$).
   - **High (1.50x):** +50% enhancement in mitigation power ($\Delta D = -4.5\text{d}, \Delta R = -22.5\%$).

---

## 2. Sensitivity Results Across the $3 \times 3$ Grid (Base Cost Scenario, $N=7,306$)

The table below reports policy performance for the 9 grid cells on the development cohort under the Base Cost Scenario:

| Grid Cell | Cost Mult | Eff Mult | Oracle Cost ($) | Oracle Benefit ($) | Oracle Interv (%) | P4 Supplier Net Benefit ($) | P1 Cost-Sensitive Net Benefit ($) |
|---|---|---|---|---|---|---|---|
| `Cost_Low__Eff_Low` | 0.50x | 0.50x | $514.88 | +$29.53 | 20.4% | +$6.82 | -$68.45 |
| `Cost_Low__Eff_Base` | 0.50x | 1.00x | $499.52 | +$44.89 | 24.1% | +$7.81 | -$59.20 |
| `Cost_Low__Eff_High` | 0.50x | 1.50x | $486.21 | +$58.20 | 31.5% | +$8.80 | -$49.96 |
| `Cost_Base__Eff_Low` | 1.00x | 0.50x | $530.12 | +$14.29 | 14.3% | +$2.00 | -$242.84 |
| `Cost_Base__Eff_Base` | 1.00x | 1.00x | **$520.27** | **+$24.14** | **17.6%** | **+$2.97** | **-$215.37** |
| `Cost_Base__Eff_High` | 1.00x | 1.50x | $509.30 | +$35.11 | 20.8% | +$3.94 | -$187.89 |
| `Cost_High__Eff_Low` | 2.00x | 0.50x | $540.85 | +$3.56 | 5.2% | -$7.64 | -$591.62 |
| `Cost_High__Eff_Base` | 2.00x | 1.00x | $536.60 | +$7.81 | 7.9% | -$6.70 | -$527.70 |
| `Cost_High__Eff_High` | 2.00x | 1.50x | $531.93 | +$12.48 | 11.2% | -$5.75 | -$463.78 |

---

## 3. Key Findings and Sensitivity Insights

### 3.1 Dynamic Intervention Frontier
1. **Elasticity to Action Cost:** As action costs double (from 0.50x to 2.00x), the Oracle's optimal intervention rate contracts sharply from 31.5% down to 5.2%. Under expensive logistics, intervention is justified only for ultra-high-value clinical shipments ($V_i > \$250,000$).
2. **Elasticity to Efficacy:** Increasing efficacy from 0.50x to 1.50x nearly doubles the Oracle net economic benefit across all cost tiers (e.g., from +$14.29 to +$35.11 in the Base Cost tier).

### 3.2 Robustness of Targeted Operational Policies
- **P4 (Supplier Escalation):** Robustly positive under Low and Base action costs (+$2.00 to +$8.80 net savings per shipment). Only turns negative under severe 2.00x administrative cost inflation (-$6.70/shipment).
- **P1 (E8 Cost-Sensitive Expediting):** Shows extreme sensitivity to action cost inflation when unconstrained. Because threshold $\tau_i^*$ triggers expediting at relatively high frequency (~53% under base assumptions), cost spikes dramatically increase total realized cost if applied universally without a strict capacity-budget constraint.
- **Budgeting Synergy:** When P1 or Oracle policies are combined with the review budget allocator ($K=10\%$), performance remains robust across all 9 sensitivity grid cells, capturing over $198,000 USD in net savings on the development cohort.

---

## 4. Policy Governance Recommendations

1. **Mandate Capacity Budgets:** Never deploy unconstrained expediting policies into automated production. Always couple cost-sensitive decision rules with a strict Control-Tower Review Budget ($K \in [5\%, 10\%]$).
2. **Prioritize Sourcing Escalations on Direct Drop:** Vendor SLA enforcement has high ROI and low baseline friction; integrate P4 as a primary early-warning intervention prior to cargo dispatch.
3. **Continuous Sensitivity Monitoring:** Track carrier freight rates and expediting premiums in real-time; if freight surcharges exceed +50%, automatically tighten the review budget from $K=10\%$ to $K=5\%$.
