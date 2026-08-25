# Olist Data Audit

- **Entity**: Order
- **Eligibility**: Only delivered orders.
- **Leakage Risk**: order_delivered_carrier_date occurs post-prediction.
- **Semantic Mapping**: Severe geographic domain shift (Brazil-only). Sparse features require table joins (e.g. order_items) to derive Line Item Value.
