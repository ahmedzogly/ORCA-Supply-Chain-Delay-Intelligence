# DataCo Data Audit

- **Entity**: Order Item
- **Eligibility**: Canceled orders excluded.
- **Leakage Risk**: shipping date (DateOrders) occurs post-prediction and must be stripped before inference.
- **Semantic Mapping**: Shipping Mode maps cleanly to Shipment Mode.
