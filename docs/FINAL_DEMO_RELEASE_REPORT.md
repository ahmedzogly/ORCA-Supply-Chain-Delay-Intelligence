# Final Demo Release Report

**Date:** 2026-08-23
**Status:** **READY**

## Test Audit Results

| Test Suite | Status | Notes |
|---|---|---|
| Dashboard UI tests (`test_dashboard_safety.py`, etc.) | ✅ PASS | 4/4 tests passed |
| Prediction contract tests (`test_prediction_contract.py`) | ✅ PASS | 13/13 tests passed |
| Architecture tests (`test_architecture.py`) | ✅ PASS | 36/36 tests passed |
| Closure/Serving integrity (`run.py --verify`) | ✅ PASS | All 36 SHA-256 invariants verified |
| Full pytest suite | ⚠️ MIXED | 641 passed, 6 skipped, 4 failed, 9 errors. Critical serving/dashboard tests pass. Legacy failures due to environment issues (WinError 5 temp directory access, missing setuptools, hardcoded file paths like `C:/Users/Admin/Downloads/DataCoSupplyChainDataset.csv`). |
| Dashboard Launch | ✅ PASS | Starts immediately, no absolute path errors, no missing artifacts |

## Launch Command

```bash
python run.py --api
# In a separate terminal
python run.py --dashboard
```

**Tested Environment:** Python 3.14.5, Streamlit 1.62.0, Windows OS.

## Model Integrity Verification

The UI refactor strictly preserved the underlying ML artifacts and serving pipelines. 

A programmatic smoke test confirmed that a control shipment produces identical numeric output compared to pre-refactor states:
- **Prediction:** `0.0077` (unchanged)
- **Decision Tier:** `LOW_RISK`
- **Models:** CatBoost + isotonic + LightGBM Q05/Q50/Q95 + CQR
- **Threshold:** `0.23`

No models were retrained.

## Default Demo Shipment and Scope

- **Default Demo Shipment ID:** `83922`
- **Scope:** Highest-risk shipment in the current 100-row demo portfolio (`artifacts/demo/demo_shipments.csv`).
- **Wording:** The dashboard explicitly says "Highest-risk shipment in the current demo portfolio".

## Academic and Business Integrity Audit

All metrics displayed on the Model Evidence page exactly match the canonical `docs/FINAL_RESULTS_SOURCE_OF_TRUTH.md` and `artifacts/model_registry/v2/serving_validation.json` outputs.

An evidence-label audit was performed on the active dashboard. 

**Safe Claims in UI:**
- `Research-validated Decision Intelligence Prototype with a Production Roadmap`
- `SIMULATED SCENARIO`
- `EXPLORATORY ONLY`
- `NOT VALIDATED` (for transferability)
- `Highest-risk shipment in the current demo portfolio`

**Forbidden Claims (Successfully Eradicated from Dashboard):**
- Enterprise-grade platform
- Realized ROI / Guaranteed Savings
- Guaranteed Coverage
- Causal Effect (replaced with hypothesis generation / exploratory only)

*(Note: Some legacy report documents like `docs/reports/FINAL_REPORT.md` retain legacy wording such as "enterprise-grade" to preserve history, but these words do not appear in the active demo).*

## Remaining Blockers
- **None.** The system is ready for the live presentation.
