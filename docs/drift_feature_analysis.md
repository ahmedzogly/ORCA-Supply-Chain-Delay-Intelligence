# Feature Sensitivity and Drift Criticality Analysis (E6.5)

## 1. Feature Inventory & Taxonomy

The SCMS Delay Intelligence System utilizes 39 modeling features partitioned into four structural types:

| Feature Name | Type | Physical Dimension / Transform | Group |
|---|---|---|---|
| `Vendor INCO Term` | Categorical | Trade incoterm contract level (EXW, CIP, DDP, FOB, etc.) | Operational |
| `Vendor` | Categorical | Supplier entity ID | Static Master |
| `vendor_hist_volume` | Numerical | Expanding historical shipment count | Historical Aggregate |
| `Country` | Categorical | Destination country code | Static Master |
| `country_hist_delay_rate` | Numerical | Point-in-time country historical delay frequency $[0, 1]$ | Historical Aggregate |
| `vendor_hist_delay_rate` | Numerical | Point-in-time supplier historical delay frequency $[0, 1]$ | Historical Aggregate |
| `country_hist_volume` | Numerical | Expanding country historical shipment volume | Historical Aggregate |
| `Scheduled_Transit_Days` | Numerical | Contracted transit lead time (days) | Temporal |
| `Forecast_Horizon_Days` | Numerical | Days between prediction timestamp and scheduled delivery | Temporal |
| `Line Item Insurance (USD)`| Numerical | $\log(1 + \text{USD})$ insurance coverage value | Transaction |
| `Line Item Quantity` | Numerical | $\log(1 + \text{units})$ shipment order volume | Transaction |
| `Line Item Value` | Numerical | $\log(1 + \text{USD})$ total line item cost | Transaction |
| `country_hist_delay_median`| Numerical | Historical median delay days in country | Historical Aggregate |
| `PQ_to_PO_Days` | Numerical | Procurement lead time interval (days) | Temporal |
| `site_hist_delay_rate` | Numerical | Historical manufacturing site delay frequency | Historical Aggregate |
| `Unit Price` | Numerical | $\log(1 + \text{USD})$ per-unit pack price | Transaction |
| `is_rdc_fulfillment` | Binary | RDC warehouse vs Direct Drop channel indicator | Missingness / Structural |
| `Pack Price` | Numerical | $\log(1 + \text{USD})$ unit pack purchase cost | Transaction |
| `Shipment Mode` | Categorical | Air, Ocean, Truck, Air Charter | Operational |
| `Unit of Measure (Per Pack)`| Numerical | Discrete tablet/pack count | Transaction |
| `T_pred_month` | Numerical | Calendar month of prediction $[1, 12]$ | Temporal |
| `freight_is_numeric` | Binary | Indication of numeric freight charge availability | Missingness / Structural |
| `First Line Designation` | Categorical | Primary / Secondary therapy designation | Static Master |
| `Sub Classification` | Categorical | Therapeutic sub-category | Static Master |
| `T_pred_dayofweek` | Numerical | Day of week $[0, 6]$ | Temporal |
| `Dosage Form` | Categorical | Tablet, Capsule, Oral Solution, Test Kit | Static Master |
| `vendor_hist_delay_median` | Numerical | Historical median supplier delay days | Historical Aggregate |
| `Product Group` | Categorical | ARV, Malaria, HIV Test, ACT | Static Master |
| `Molecule/Test Type` | Categorical | Chemical compound or diagnostic test type | Static Master |
| `T_pred_quarter` | Numerical | Calendar quarter $[1, 4]$ | Temporal |
| `T_pred_year` | Numerical | Calendar year of prediction anchor | Temporal |
| `weight_is_numeric` | Binary | Indication of physical weight availability | Missingness / Structural |
| `po_sent_is_date` | Binary | Indicator for valid PO transmission date | Missingness / Structural |
| `Dosage` | Categorical | Concentration / strength specification | Static Master |
| `Fulfill Via` | Categorical | Delivery channel (`From RDC` vs `Direct Drop`) | Operational |
| `Brand` | Categorical | Pharmaceutical brand identifier | Static Master |
| `Manufacturing Site` | Categorical | Production plant location | Static Master |
| `is_pre_pq_process` | Binary | Flag for legacy procurement process | Missingness / Structural |
| `pq_first_sent_is_date` | Binary | Indicator for valid client quote date | Missingness / Structural |

---

## 2. Empirical SHAP Stability & Feature Criticality Hierarchy

Cross-fold TreeSHAP attribution analysis on the Stage 5 CatBoost Champion (`artifacts/explainability/shap_stability.csv`) established consistent feature rankings across all 5 chronological Development folds:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                     FEATURE CRITICALITY HIERARCHY                              │
├────────────────────────────────────────────────────────────────────────────────┤
│ TIER 1: CRITICAL DRIVERS (Weight = 3.0, VETO POWER)                            │
│  1. Vendor INCO Term (Mean Rank: 1.4, Var: 0.64)                               │
│  2. Vendor (Mean Rank: 2.6, Var: 0.64)                                         │
│  3. vendor_hist_volume (Mean Rank: 3.6, Var: 7.84)                             │
│  4. Country (Mean Rank: 5.6, Var: 4.64)                                        │
│  5. country_hist_delay_rate (Mean Rank: 6.2, Var: 4.16)                        │
│  6. vendor_hist_delay_rate (Mean Rank: 6.4, Var: 14.64)                        │
│  7. country_hist_volume (Mean Rank: 8.0, Var: 12.00)                           │
│  8. Scheduled_Transit_Days (Mean Rank: 8.8, Var: 7.76)                         │
│  9. Forecast_Horizon_Days (Mean Rank: 9.0, Var: 2.40)                          │
│ 10. Line Item Insurance (USD) (Mean Rank: 9.6, Var: 11.84)                     │
│ 11. Line Item Quantity (Mean Rank: 10.0, Var: 11.60)                           │
├────────────────────────────────────────────────────────────────────────────────┤
│ TIER 2: HIGH/MEDIUM PREDICTORS (Weight = 1.5)                                  │
│ 12. Line Item Value (Mean Rank: 14.2)                                          │
│ 13. country_hist_delay_median (Mean Rank: 16.6)                               │
│ 14. PQ_to_PO_Days (Mean Rank: 16.8)                                            │
│ 15. site_hist_delay_rate (Mean Rank: 17.2)                                     │
│ 16. Unit Price (Mean Rank: 17.8)                                               │
│ 17. is_rdc_fulfillment (Mean Rank: 18.4)                                      │
│ 18. Pack Price (Mean Rank: 18.6)                                               │
│ 19. Shipment Mode (Mean Rank: 19.0)                                            │
│ 20. Unit of Measure (Per Pack) (Mean Rank: 21.2)                               │
│ 21. T_pred_month (Mean Rank: 21.4)                                             │
├────────────────────────────────────────────────────────────────────────────────┤
│ TIER 3: CONTEXTUAL / METADATA FEATURES (Weight = 0.5)                          │
│ 22-39. freight_is_numeric, Sub Classification, First Line Designation,         │
│        Dosage Form, Product Group, Molecule, Dosage, Brand, Site, etc.         │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Critical Policy Implication:
- **Tier 1 Features possess Veto Power**: A severe shift ($\text{PSI} \ge 0.25$) in any single Tier 1 feature immediately forces the system state to **RED (Recalibration Required)**, regardless of the stability of lower-ranked features.
- **Tier 3 Features are Suppressed from Unilateral Veto**: Contextual metadata features with localized shifts (e.g. slight seasonal changes in `Dosage Form`) cannot trigger false positive model recalibrations.

---

## 3. Feature Sensitivity & Elasticity Analysis

Elasticity measures the expected relative shift in model delay probability $\hat{p}$ per unit shift in normalized feature value $X_j$:
$$\mathcal{E}(X_j) = \mathbb{E}\left[ \left| \frac{\partial \hat{p}}{\partial X_j} \right| \cdot \frac{\sigma(X_j)}{\sigma(\hat{p})} \right]$$

### Empirical Sensitivity Findings:
1. **`Vendor INCO Term` ($\mathcal{E} \approx 0.42$)**: Changing freight terms from `EXW` (Ex Works, where buyer handles origin shipping) to `DDP` (Delivered Duty Paid, where seller guarantees destination delivery) shifts predicted delay probability by up to $-14.2\%$.
2. **`vendor_hist_delay_rate` ($\mathcal{E} \approx 0.38$)**: A $+10\%$ increase in a supplier's historical delay rate inflates delay odds by $+1.45\times$.
3. **`Scheduled_Transit_Days` ($\mathcal{E} \approx 0.31$)**: Short transit horizons ($< 30\text{ days}$) under international air corridors show steep nonlinear risk elasticity due to lack of buffer time for customs clearance.

---

## 4. Domain-Specific Drift Vulnerabilities in Supply Chains

### 4.1 INCO Term Structural Recontracting
Global procurement agreements undergo multi-year contract renewals. A programmatic transition in contractual terms (e.g. shifting bulk pharmaceutical shipments from CIP to FCA) fundamentally alters supply chain risk allocation, creating sudden covariate and concept drift.

### 4.2 Destination Infrastructure Bottlenecks
Port congestion, customs documentation shifts, or regional road infrastructure degradation in specific destination countries (e.g. Nigeria, Côte d'Ivoire) causes localized surges in `country_hist_delay_rate` and `country_hist_delay_median`.

### 4.3 Air vs Ocean Freight Substitution
During global disruptions (volcanic ash clouds, pandemic flight reductions, port strikes), logistics managers substitute ocean freight for air freight, altering `Shipment Mode`, `Scheduled_Transit_Days`, and `Line Item Insurance (USD)`.

---

## 5. Cold-Start and Unseen Categorical Level Protocol

When new suppliers, manufacturing plants, or country codes appear in a detection window:
1. **Laplace Regularization**: Categorical PSI and JSD allocate non-zero probability mass $\tilde{q} = \epsilon / (N_{det} + K\epsilon)$ to unobserved categories without crashing.
2. **Global Prior Fallback**: Historical aggregate features (`vendor_hist_delay_rate`, etc.) substitute the global expanding training prior for unseen entities, preventing artificial spikes in nonconformity scores.
3. **Cochran Pooling**: Rare levels are merged into `'__OTHER__'` before $\chi^2$ testing, ensuring low-frequency new categories do not trigger spurious statistical rejections.
