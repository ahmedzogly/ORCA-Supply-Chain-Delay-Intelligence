# Severity Definition

## Empirical Justification
To provide interpretable operational categories alongside continuous probabilistic bounds, we defined the following Project-defined operational severity tiers on positive severity (max(Delay_Days, 0)):

1. **No Delay**: 0 days
   - *Operational Meaning*: Shipment expected on time or early.

2. **Low Severity**: (0, 7] days
   - *Operational Meaning*: Minor operational delay. Usually absorbable by safety stock or standard buffer times.

3. **Moderate Severity**: (7, 14] days
   - *Operational Meaning*: Significant delay requiring planning adjustment or proactive communication.

4. **High Severity**: > 14 days
   - *Operational Meaning*: Severe disruption requiring direct intervention (e.g., rerouting, expediting, or alternative sourcing).

These thresholds are static post-evaluation limits applied to the max(Delay_Days, 0) target definition to translate continuous risk intervals into discrete planning alerts.

