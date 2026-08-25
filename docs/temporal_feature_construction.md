# Temporal Feature Construction & Feature Engineering Spec

## Objective
Convert the Bronze SCMS dataset into a Modeling-Ready Feature Dataset while strictly enforcing the Prediction Contract ({\text{pred}}$). This involves preventing data leakage by ensuring every feature's availability is theoretically and practically guaranteed at or before {\text{pred}}$.

## 1. Feature Grouping

Features were categorized into five domains:
1. **Static / Master Data**: Intrinsic properties of the order and product (e.g., Country, Vendor, Product Group). Kept as raw categorical.
2. **Transaction Features**: Financial and quantity metrics (e.g., Line Item Quantity, Pack Price). Log1p transformations applied to stabilize right-skewed distributions.
3. **Temporal Components**: Derived strictly from {\text{pred}}$ (Year, Month, Quarter) and lead time (Forecast Horizon Days).
4. **Operational**: Logistics routing variables (e.g., Shipment Mode, INCO Term).
5. **Historical Point-in-Time (PIT) Aggregates**: Expanding window moving averages (delay rates, medians) bounded strictly by {\text{pred}}$.

## 2. Point-In-Time Historical Aggregates (The Engine)

Historical features pose the highest leakage risk. We engineered a robust PIT extraction system:
- **Rule**: For a prediction row $, the history subset is all rows $ where {\text{outcome}}(j) < T_{\text{pred}}(i)$.
- **No Global Leakage**: At no point is group_mean(target) applied globally.
- **Implemented Features**: endor_hist_delay_rate, endor_hist_delay_median, country_hist_delay_rate, country_hist_delay_median, site_hist_delay_rate.

## 3. Cold Start Policy

For new entities (Vendors, Countries) that have zero historical record strictly prior to a given {\text{pred}}$, they suffer a "Cold Start".
- **Fallback**: The pipeline computes the global expanding average/median up to {\text{pred}}$ and fills NaNs for unseen entities with this global point-in-time prior.
- **Result**: No artificial zeros are introduced that would skew model semantics.

## 4. Exclusion & Ambiguity

- **Weight and Freight**: As specified in the Stage 2 contract, these variables reflect actual gross physical dispatch metrics and downstream carrier invoicing. They remain **EXCLUDED** from the Stage 3 feature set.
- **Surrogate Identifiers**: PO / SO #, PQ #, and ID were audited and dropped due to high memorization risk.
- **High Cardinality**: Item Description dropped in favor of the structured hierarchical categoricals (Product Group, Molecule/Test Type).

## 5. Artifact Output
- **Pipeline Implementation**: src/delay_intelligence/features/builder.py
- **Output Artifact**: rtifacts/data/scms_modeling_features.parquet
- **Rows**: 8,319
- **Columns**: 46 (including targets reserved for downstream splitting/evaluation).
