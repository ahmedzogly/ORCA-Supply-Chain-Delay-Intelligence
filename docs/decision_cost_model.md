# Decision Cost Model

Calculates Expected Net Benefit for an action $.

Expected Base Cost = Expected Severity * (Base Daily Cost + Value Multiplier)
Residual Cost(A) = max(0, Expected Severity - Efficacy(A)) * (Daily Cost)
Total Cost(A) = Action Cost(A) + Residual Cost(A)
Net Benefit(A) = Expected Base Cost - Total Cost(A)

*Assumptions*: 
- Base delay cost = /day
- Value Multiplier = 0.1% of Line Item Value per day
- Expedite Cost =  (reduces delay by 5 days)

**Note:** All costs are simulated scenario estimates for policy evaluation. They do not represent realized historical ROI.
