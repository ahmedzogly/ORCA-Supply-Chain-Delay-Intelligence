# Prescriptive Policy

This policy governs action mappings. 

## Action Taxonomy
- NO_ACTION: Auto-eligible. Safe to ignore.
- MONITOR: Auto-eligible. Keep on watchlists.
- EXPEDITE: High cost, high efficacy. Reserved for high-value critical delays.
- SUPPLIER_ESCALATION: Triggered when Vendor is a predictive driver.
- TRANSPORT_MODE_REVIEW: Triggered when Shipment Mode is a causal driver.
- HUMAN_REVIEW: Default for edge cases, high uncertainty, or negative net-benefit scenarios.

## Human-in-the-Loop
The engine does NOT autonomously execute operational changes. Any action outside NO_ACTION or MONITOR requires explicit human approval.
