# STAGE 8 REPORT

STATUS: PASS

## Stage 8: Prescriptive Decision Engine

The project has successfully bridged predictive explanations and causal candidates into an actionable, human-in-the-loop decision engine.

### Key Capabilities Evaluated
1. **Uncertainty-Aware Decisions (RQ1)**: High uncertainty intervals dynamically veto automated actions, successfully defaulting to HUMAN_REVIEW to protect operations from volatile confidence bounds.
2. **VALUE_ONLY Benchmarking (RQ2)**: The Prescriptive Engine outperforms the naive financial-priority baseline by ~5x in expected delay days captured under the same review budget. It also optimizes beyond naive probability (RISK_ONLY) by ignoring delays where interventions cost more than the expected residual benefit.
3. **Cost-Sensitivity Analysis (RQ3)**: Implemented. The engine classifies recommendations as ROBUST, SENSITIVE, or UNSUPPORTED against simulated cost variations, explicitly avoiding false ROI guarantees.
4. **Causal vs Predictive (RQ5)**: Rule sets combine predictive importance (SHAP) and stable causal candidates (e.g., Shipment Mode) to identify precise intervention routes (e.g., TRANSPORT_MODE_REVIEW), avoiding spurious actions.

### Implementation Checklist
- [x] Decision engine mapping Probability + Severity + Uncertainty + Causal driver.
- [x] Explicit Cost Assumptions defined in configs/decision.yaml.
- [x] Traceable decision outputs mapped to expected impact and reasons.
- [x] Human-in-the-loop strictly enforced (auto-execution blocked for interventions).
- [x] Final chronological holdout protected.
- [x] All automated safety, logic, and traceability tests passing.
