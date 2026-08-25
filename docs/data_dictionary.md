# Data Dictionary — Multi-Dataset Supply Chain Intelligence

This document details the schema definitions, column descriptions, data types, and target variable mappings for the three supported logistics datasets.

---

## 1. SCMS Delivery History Dataset

- **File Path**: `scms/SCMS_Delivery_History_Dataset.csv`
- **Volume**: ~3.61 MB (3,785,904 bytes), 10,324 records, 33 columns
- **Domain**: International public health logistics and pharmaceutical delivery history.
- **Encoding**: UTF-8 / Standard ASCII
- **Primary Key**: `ID`

### Column Inventory & Descriptions

| Column Name | Raw Type | Target Ingestion Type | Category | Description |
|---|---|---|---|---|
| `ID` | Integer | `int64` | Identifier | Unique shipment transaction identifier. |
| `Project Code` | String | `string` | Categorical | Project identification code (e.g. 100-CI-T01). |
| `PQ #` | String | `string` | Categorical | Price Quote number. |
| `PO / SO #` | String | `string` | Categorical | Purchase Order / Sales Order identifier. |
| `ASN/DN #` | String | `string` | Categorical | Advanced Shipping Notice / Delivery Note identifier. |
| `Country` | String | `string` | Categorical | Destination country for the medical supply shipment. |
| `Managed By` | String | `string` | Categorical | Managing entity (e.g. PMO - US). |
| `Fulfill Via` | String | `string` | Categorical | Fulfillment channel (Direct Drop, From RDC). |
| `Vendor INCO Term` | String | `string` | Categorical | International Commercial Terms (e.g. EXW, CIP, DDP, FOB). |
| `Shipment Mode` | String | `string` | Categorical | Transport mode (Air, Truck, Ocean, Air Charter). |
| `PQ First Sent to Client Date` | String | `datetime64[ns]` | Milestone | Date price quote was initially transmitted. |
| `PO Sent to Vendor Date` | String | `datetime64[ns]` | Milestone | Date purchase order was dispatched to supplier (**Prediction Anchor**). |
| `Scheduled Delivery Date` | String | `datetime64[ns]` | Milestone | Contracted target delivery date. |
| `Delivered to Client Date` | String | `datetime64[ns]` | Post-Event | Actual physical delivery date to recipient (**Target Source**). |
| `Delivery Recorded Date` | String | `datetime64[ns]` | Post-Event | Date delivery was logged in central ERP. |
| `Product Group` | String | `string` | Categorical | Broad medical product category (ARV, HRDT, ACT, ANTM). |
| `Sub Classification` | String | `string` | Categorical | Granular classification (e.g. Pediatric, Adult, HIV test). |
| `Vendor` | String | `string` | Categorical | Manufacturing / supplying vendor name. |
| `Item Description` | String | `string` | Text | Full item description text. |
| `Molecule/Test Type` | String | `string` | Categorical | Active pharmaceutical molecule or diagnostic assay type. |
| `Brand` | String | `string` | Categorical | Commercial brand name. |
| `Dosage` | String | `string` | Categorical | Pharmaceutical dosage concentration. |
| `Dosage Form` | String | `string` | Categorical | Physical form (Tablet, Capsule, Test kit, Solution). |
| `Unit of Measure (Per Pack)` | Integer | `int32` | Numeric | Number of units per packaged box. |
| `Line Item Quantity` | Integer | `int64` | Numeric | Total quantity of units ordered. |
| `Line Item Value` | Float | `float64` | Numeric | Total monetary value of order line in USD. |
| `Pack Price` | Float | `float64` | Numeric | Price per unit pack in USD. |
| `Unit Price` | Float | `float64` | Numeric | Price per individual unit in USD. |
| `Manufacturing Site` | String | `string` | Categorical | Production plant location. |
| `First Line Designation` | String | `string` | Categorical | Flag indicating essential first-line drug therapy (Yes/No). |
| `Weight (Kilograms)` | Mixed | `float64` | Numeric | Shipment weight in kilograms (requires cleaning mixed text flags). |
| `Freight Cost (USD)` | Mixed | `float64` | Numeric | Freight transport cost in USD (requires cleaning mixed text flags). |
| `Line Item Insurance (USD)` | Float | `float64` | Numeric | Insurance fee for order line in USD. |

### Derived Target Definitions
1. **`Delay_Flag` (Binary)**:
   $$\text{Delay\_Flag} = \begin{cases} 1 & \text{if } \text{Delivered to Client Date} > \text{Scheduled Delivery Date} \\ 0 & \text{otherwise} \end{cases}$$
2. **`Delay_Days` (Continuous)**:
   $$\text{Delay\_Days} = \max(0, (\text{Delivered to Client Date} - \text{Scheduled Delivery Date})_{\text{days}})$$

---

## 2. Olist Brazilian E-Commerce Dataset

- **Directory**: `olist/`
- **Volume**: ~120.34 MB (126,183,944 bytes) across 9 relational CSV files.
- **Domain**: B2C E-commerce marketplace orders and multi-carrier logistics across Brazil.
- **Primary Tables**:
  - `olist_orders_dataset.csv` (Primary order table, timestamps: purchase, approved, carrier delivery, customer delivery, estimated delivery).
  - `olist_order_items_dataset.csv` (Line items, product IDs, seller IDs, prices, freight values).
  - `olist_products_dataset.csv` (Product categories, dimensions, weights).
  - `olist_customers_dataset.csv` & `olist_sellers_dataset.csv` (Zip codes, cities, states).
  - `olist_order_reviews_dataset.csv` (Post-delivery review scores and comments).
  - `olist_order_payments_dataset.csv` (Payment types and installment structures).
  - `olist_geolocation_dataset.csv` (Lat/long coordinates for zip codes).

---

## 3. DataCo Global Supply Chain Dataset

- **File Path**: `dataco/DataCoSupplyChainDataset.csv`
- **Volume**: ~91.47 MB (95,910,149 bytes), 180,519 rows, 53 columns.
- **Domain**: Omnichannel retail and logistics with global distribution.
- **Encoding**: `latin1` / `ISO-8859-1`
- **Key Columns**:
  - `Days for shipping (real)` & `Days for shipment (scheduled)`
  - `Delivery Status` (Advance shipping, Late delivery, Shipping on time, Shipping canceled)
  - `Late_delivery_risk` (Binary target flag)
  - `Shipping date (DateOrders)` & `order date (DateOrders)`
  - `Shipping Mode` (Standard Class, First Class, Second Class, Same Day)
  - `Sales per customer`, `Order Item Total`, `Order Profit Per Order`
  - Geographic attributes (Customer Country/City, Order Region/Market)
