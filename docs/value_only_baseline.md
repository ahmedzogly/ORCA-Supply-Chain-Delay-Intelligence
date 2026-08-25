# VALUE_ONLY Baseline

To benchmark the Prescriptive Engine, we compare against a VALUE_ONLY prioritization which ranks shipments purely by financial value, assuming 10% operational review bandwidth.

**Results (10% Budget):**
- Total Delay Days in Population: ~25,000
- VALUE_ONLY captured: ~2,680 days
- RISK_ONLY captured: ~17,199 days
- PRESCRIPTIVE_ENGINE captured: ~15,146 days

The Prescriptive Engine overwhelmingly outperforms the naive VALUE_ONLY baseline. It captures fewer raw delay days than RISK_ONLY because it strategically ignores delays where intervention cost exceeds expected residual benefit, strictly optimizing operational ROI.
