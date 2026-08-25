# Causal Ontology

To prevent unidentifiable or illogical causal structures, we define a common causal ontology mapped specifically to the SCMS dataset.

## Tiers
1. **Origin / Strategic (Tier 0)**: Fulfill Via, Country, Product Group
2. **Order Characteristics (Tier 1)**: Line Item Quantity, Line Item Value
3. **Logistics (Tier 2)**: Shipment Mode
4. **Outcome (Tier 3)**: Delay_Days

## Expert Constraints
- Temporal ordering applies (Tier N cannot cause Tier N-1).
- Delay_Days is the terminal outcome and cannot cause any upstream variable.
- Any edge violating these temporal constraints is forbidden during causal discovery.
