# Supply Chain Delay Intelligence System
# Feature Availability Matrix Across Operational Milestones

**System**: Supply Chain Delay Intelligence System  
**Milestone**: Stage 2 — Prediction Contract, Target Definition & Leakage Specification  
**Document**: Feature Availability Matrix (`docs/feature_availability_matrix.md`)  
**Status**: AUTHORITATIVE SPECIFICATION  
**Dataset Reference**: USAID / SCMS Delivery History Dataset ($N = 10,324$)  

---

## 1. Operational Lifecycle Milestones

In international public health supply chains, order fulfillment transitions through six discrete operational milestone events:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SUPPLY CHAIN LIFECYCLE MILESTONES                              │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────┬────────────┤
│  Milestone M0   │  Milestone M1   │  Milestone M2   │  Milestone M3   │Milestone M4 │Milestone M5│
│  Project Setup  │ Price Quotation │Order Commitment │Consignment/ASN  │  Delivery   │ ERP Logged │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────┼────────────┤
│• Project Code   │• PQ #           │• PO / SO #      │• ASN/DN #       │• Delivered  │• Delivery  │
│• Country        │• PQ Sent Date   │• PO Sent Date   │• Actual Weight  │  to Client  │  Recorded  │
│• Managed By     │• Product Specs  │• Scheduled Date │• Carrier Freight│  Date       │  Date      │
│                 │• Unit/Pack Price│• Vendor         │  Airway Bill    │ (Target     │ (Post-     │
│                 │• Order Quantity │• Manufacturing  │  / Invoice      │  Milestone) │  Outcome)  │
│                 │• Line Item Value│  Site           │• Mode switches  │             │            │
│                 │• Clinical Regim │• INCO Term      │                 │             │            │
│                 │                 │• Planned Mode   │                 │             │            │
│                 │                 │• Insurance USD  │                 │             │            │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────┴────────────┘
         ▲                                   ▲                                 ▲
         │                                   │                                 │
     Pre-Quote                      PREDICTION BOUNDARY                  Target Event
                                    (T_pred <= T_PO)                     (T_outcome)
```

### Milestone Stage Definitions:
1. **$M_0$ (Project Inception & Governance)**: Country health program bilateral agreement signed; recipient geography and managing PMO office established.
2. **$M_1$ (Price Quotation & Requisition — $T_{\text{PQ}}$)**: Health facility submits drug order requisition; SCMS issues formal Price Quotation (`PQ #`, `PQ First Sent to Client Date`) detailing clinical formulations, dosage strengths, quantities, unit prices, and commodity valuations.
3. **$M_2$ (Order Commitment & Supplier Contracting — $T_{\text{PO}} / T_{\text{pred}}$)**: Formal Purchase Order (`PO / SO #`) issued to pharmaceutical vendor (Direct Drop) or warehouse stock allocated (From RDC). Contractual `Scheduled Delivery Date`, `Vendor INCO Term`, `Manufacturing Site`, and planned `Shipment Mode` are finalized. **This is the canonical operational prediction boundary ($T_{\text{pred}}$).**
4. **$M_3$ (Warehouse Packing & Freight Consignment — $T_{\text{Consign}}$)**: Pharmaceutical batch manufactured, packed into shipping cartons/pallets, and tendered to carrier. Advance Shipping Notice (`ASN/DN #`) created; gross physical weight weighed on scales; airway bill / bill of lading generated.
5. **$M_4$ (Physical Client Delivery & Target Realization — $T_{\text{outcome}}$)**: Cargo arrives at destination country central warehouse (`Delivered to Client Date`). Ground-truth delivery performance ($Y = \text{Delivered} - \text{Scheduled}$) is realized.
6. **$M_5$ (Post-Delivery Administrative ERP Logging — $T_{\text{Record}}$)**: Stamped delivery proof receipts received from local clinics and administratively entered into the USAID SCMS web portal (`Delivery Recorded Date`).

---

## 2. Milestone Feature Availability Matrix

The matrix below documents the exact operational availability status of every attribute across all six milestones:

| Attribute / Feature Name | $M_0$: Project Setup | $M_1$: Price Quotation ($T_{\text{PQ}}$) | $M_2$: Order Commitment ($T_{\text{PO}} / T_{\text{pred}}$) | $M_3$: Consignment Dispatch ($T_{\text{Consign}}$) | $M_4$: Physical Delivery ($T_{\text{Deliv}}$) | $M_5$: Post-Delivery ERP ($T_{\text{Record}}$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `ID` (Primary Key) | 🔒 Database Key | 🔒 Database Key | 🔒 Database Key | 🔒 Database Key | 🔒 Database Key | 🔒 Database Key |
| `Project Code` | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Country` | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Managed By` | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `PQ #` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `PQ First Sent to Client Date` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Product Group` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Sub Classification` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Item Description` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Molecule/Test Type` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Brand` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Dosage` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Dosage Form` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Unit of Measure (Per Pack)` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Line Item Quantity` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Line Item Value` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Pack Price` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Unit Price` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `First Line Designation` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `is_pre_pq_process` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `pq_first_sent_is_date` | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `PO / SO #` | ❌ Unavailable | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Fulfill Via` & `is_rdc_fulfillment` | ❌ Unavailable | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Vendor` | ❌ Unavailable | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Manufacturing Site` | ❌ Unavailable | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Vendor INCO Term` | ❌ Unavailable | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Shipment Mode` (Planned) | ❌ Unavailable | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `PO Sent to Vendor Date` | ❌ Unavailable | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `po_sent_is_date` | ❌ Unavailable | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Scheduled Delivery Date` | ❌ Unavailable | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Scheduled_Transit_Days` | ❌ Unavailable | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `PQ_to_PO_Days` | ❌ Unavailable | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `Line Item Insurance (USD)` | ❌ Unavailable | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `weight_is_numeric` | ❌ Unavailable | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `freight_is_numeric` | ❌ Unavailable | ❌ Unavailable | ✅ Available | ✅ Available | ✅ Available | ✅ Available |
| `ASN/DN #` | ❌ Unavailable | ❌ Unavailable | ❌ **FORBIDDEN** | ✅ Available | ✅ Available | ✅ Available |
| `Weight (Kilograms)` (Actual Weighed) | ❌ Unavailable | ❌ Unavailable | ❌ **FORBIDDEN** | ✅ Available | ✅ Available | ✅ Available |
| `Freight Cost (USD)` (Actual Invoiced) | ❌ Unavailable | ❌ Unavailable | ❌ **FORBIDDEN** | ⚠️ Partial Invoice | ✅ Final Invoice | ✅ Settled |
| `Delivered to Client Date` | ❌ Unavailable | ❌ Unavailable | ❌ **FORBIDDEN** | ❌ **FORBIDDEN** | 🎯 **TARGET EVENT** | ✅ Historic |
| `Delay_Flag` (Target) | ❌ Unavailable | ❌ Unavailable | ❌ **FORBIDDEN** | ❌ **FORBIDDEN** | 🎯 **TARGET REALIZED** | ✅ Historic |
| `Delay_Days` (Target) | ❌ Unavailable | ❌ Unavailable | ❌ **FORBIDDEN** | ❌ **FORBIDDEN** | 🎯 **TARGET REALIZED** | ✅ Historic |
| `Delivery Recorded Date` | ❌ Unavailable | ❌ Unavailable | ❌ **FORBIDDEN** | ❌ **FORBIDDEN** | ❌ **FORBIDDEN** | ⚠️ **POST-OUTCOME** |
| `is_temporal_anomaly` | ❌ Unavailable | ❌ Unavailable | ❌ **FORBIDDEN** | ❌ **FORBIDDEN** | ❌ **FORBIDDEN** | ⚠️ **FILTER ONLY** |

---

## 3. Transition Rules & Stage-Gate Boundaries

1. **Prediction Gate ($M_2$)**: All features utilized by the predictive inference pipeline MUST be marked `✅ Available` at $M_2$ (Order Commitment). Any feature marked `❌ Unavailable` or `❌ FORBIDDEN` at $M_2$ must be strictly rejected from the input feature vector.
2. **Consignment Transition ($M_2 \to M_3$)**: The generation of `ASN/DN #`, actual gross weight scale tickets, and carrier invoices occurs strictly after order commitment. These attributes cannot be back-propagated to the $M_2$ prediction time.
3. **Outcome Realization Gate ($M_4$)**: Physical arrival at the destination warehouse triggers the realization of the target variables (`Delay_Flag`, `Delay_Days`). These variables provide ground-truth supervision for historical training sets but are forbidden from the feature set.
4. **Administrative Audit Gate ($M_5$)**: Field office ERP data entry creates `Delivery Recorded Date`. This administrative artifact is strictly post-outcome and forbidden from modeling.
