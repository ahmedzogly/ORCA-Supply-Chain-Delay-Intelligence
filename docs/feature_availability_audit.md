# Feature Lineage & Availability Audit

| Feature | Source Column(s) | Transformation | Availability Event | Available at T_pred? | Leakage Risk | Status |
| ------- | ---------------- | -------------- | ------------------ | -------------------- | ------------ | ------ |
| Country | Country | None | M0 (Project Inception) | Yes | None | APPROVED |
| Vendor | Vendor | None | M2 (Order Commitment) | Yes | None | APPROVED |
| Manufacturing Site | Manufacturing Site | None | M2 (Order Commitment) | Yes | None | APPROVED |
| Product Group | Product Group | None | M1 (Quotation) | Yes | None | APPROVED |
| Sub Classification | Sub Classification | None | M1 (Quotation) | Yes | None | APPROVED |
| Molecule/Test Type | Molecule/Test Type | None | M1 (Quotation) | Yes | None | APPROVED |
| Brand | Brand | None | M1 (Quotation) | Yes | None | APPROVED |
| Dosage | Dosage | None | M1 (Quotation) | Yes | None | APPROVED |
| Dosage Form | Dosage Form | None | M1 (Quotation) | Yes | None | APPROVED |
| First Line Designation | First Line Designation | None | M1 (Quotation) | Yes | None | APPROVED |
| Line Item Quantity | Line Item Quantity | Log1p | M2 (Order Commitment) | Yes | None | TRANSFORM_REQUIRED |
| Line Item Value | Line Item Value | Log1p | M2 (Order Commitment) | Yes | None | TRANSFORM_REQUIRED |
| Pack Price | Pack Price | Log1p | M2 (Order Commitment) | Yes | None | TRANSFORM_REQUIRED |
| Unit Price | Unit Price | Log1p | M2 (Order Commitment) | Yes | None | TRANSFORM_REQUIRED |
| Line Item Insurance (USD) | Line Item Insurance (USD) | Fill NA 0 -> Log1p | M2 (Order Commitment) | Yes | None | TRANSFORM_REQUIRED |
| Unit of Measure (Per Pack) | Unit of Measure (Per Pack) | None | M1 (Quotation) | Yes | None | APPROVED |
| Fulfill Via | Fulfill Via | None | M1/M2 (Routing) | Yes | None | APPROVED |
| Shipment Mode | Shipment Mode | Fill NA 'Missing' | M2 (Order Commitment) | Yes | None | APPROVED_WITH_RISK (May reflect actual instead of planned) |
| Vendor INCO Term | Vendor INCO Term | None | M2 (Order Commitment) | Yes | None | APPROVED |
| Scheduled_Transit_Days | Scheduled Delivery Date, T_pred | (Scheduled - T_pred).days | M2 (Order Commitment) | Yes | None | APPROVED (Renamed Forecast_Horizon_Days) |
| PQ_to_PO_Days | PO Sent Date, PQ Sent Date | (PO - PQ).days | M2 (Order Commitment) | Yes | None | APPROVED |
| T_pred_year, month, quarter, dayofweek | T_pred | Date component extraction | M2 (Order Commitment) | Yes | None | APPROVED |
| is_rdc_fulfillment | Fulfill Via | Boolean logic | M2 (Order Commitment) | Yes | None | APPROVED |
| is_pre_pq_process | PQ First Sent Date | Boolean logic | M1 (Quotation) | Yes | None | APPROVED |
| po_sent_is_date, pq_is_date | PO/PQ Dates | Boolean logic | M2 (Order Commitment) | Yes | None | APPROVED |
| weight_is_numeric | Weight (Kilograms) | Boolean logic | M3 (Consignment) | No (Post-PO) | Mod-High | EXCLUDED (Dropped to prevent leakage) |
| freight_is_numeric | Freight Cost (USD) | Boolean logic | M3/M5 (Consignment) | No (Post-PO) | Mod-High | EXCLUDED (Dropped to prevent leakage) |
| vendor_hist_delay_rate | Vendor, Delay_Flag, Delivered Date | Point-in-time expanding mean | T_pred | Yes (Strictly < T_pred) | Managed | APPROVED |
| vendor_hist_delay_median | Vendor, Delay_Days, Delivered Date | Point-in-time expanding median | T_pred | Yes (Strictly < T_pred) | Managed | APPROVED |
| country_hist_delay_rate | Country, Delay_Flag, Delivered Date | Point-in-time expanding mean | T_pred | Yes (Strictly < T_pred) | Managed | APPROVED |
| country_hist_delay_median | Country, Delay_Days, Delivered Date | Point-in-time expanding median | T_pred | Yes (Strictly < T_pred) | Managed | APPROVED |
| site_hist_delay_rate | Manufacturing Site, Delay_Flag | Point-in-time expanding mean | T_pred | Yes (Strictly < T_pred) | Managed | APPROVED |
| ID | ID | None | System Ingestion | Yes | High (Memorization) | EXCLUDED |
| ASN/DN # | ASN/DN # | None | M3 (Consignment) | No | High | EXCLUDED |
| PQ #, PO / SO # | PQ #, PO / SO # | None | M1, M2 | Yes | High (Surrogate) | EXCLUDED |
| Item Description | Item Description | None | M1 (Quotation) | Yes | High (Cardinality/NLP) | EXCLUDED (Pending NLP) |
| Weight (Kilograms) | Weight (Kilograms) | None | M3 (Consignment Weigh-in) | No | Severe | EXCLUDED |
| Freight Cost (USD) | Freight Cost (USD) | None | M3/M5 (Carrier Invoicing) | No | Severe | EXCLUDED |
| Delivered to Client Date | Delivered to Client Date | None | M4 (Delivery) | No | 100% Target | TARGET |
| Delivery Recorded Date | Delivery Recorded Date | None | M5 (ERP Log) | No | Post-outcome | EXCLUDED |
| Delay_Flag, Delay_Days | Delivered, Scheduled | Subtraction | M4 (Delivery) | No | 100% Target | TARGET |
| is_temporal_anomaly | Delivered, Scheduled | Boolean logic | M4 (Delivery) | No | Target-derived | EXCLUDED (Filter only) |
