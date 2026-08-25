# Demo Credibility Closure — Patch Changelog

## Completed

1. **Removed placeholder serving behavior from the live API**
   - v2 CatBoost classifier is loaded from the registry.
   - isotonic probability calibration is applied online.
   - the calibrated decision threshold is used online.
   - LightGBM q05/q50/q95 severity models are loaded online.
   - real split-CQR calibration is applied online.
   - local CatBoost SHAP values are computed at request time.

2. **Unified offline build and online serving**
   - `scripts/build_serving_registry.py` rebuilds the exact v2 serving artifacts from the exact-hash SCMS CSV using an isolated temporal train/calibration/holdout design.
   - `artifacts/model_registry/v2/serving_validation.json` reports the holdout validation for that serving registry.

3. **Dashboard cleanup**
   - removed five duplicate/stale pages;
   - removed broken `APIClient` imports and syntax-error page;
   - dashboard uses the generated real holdout demo sample;
   - evidence labels appear in each workflow;
   - Action Center includes configurable scenario-economics sliders.

4. **External-validation claim correction**
   - DataCo/Olist hard-coded metrics retired;
   - evaluator now raises `ExternalValidationNotPerformed` rather than fabricate results;
   - docs call these portability prototypes/protocols only.

5. **Causal claim correction**
   - renamed presentation framing to **Exploratory Causal Analysis**;
   - documented the PC/Fisher-Z + encoded-categorical limitation;
   - API returns only `exploratory_hypothesis_only` overlap, never a validated causal effect.

6. **Single results source of truth**
   - added `docs/FINAL_RESULTS_SOURCE_OF_TRUTH.md`;
   - separates legacy frozen research artifacts from v2 serving validation;
   - defines allowed/prohibited language by evidence label.

7. **Reproducibility**
   - bundled the exact-hash SCMS raw CSV under `data/raw/`;
   - changed active data paths to repository-relative paths;
   - removed the fake v1 git hash and explicitly deprecated v1 for serving;
   - added v2 reproducibility verification and closure-manifest generation;
   - expanded demo/runtime dependency declarations.

8. **Business claims**
   - `$31,489.44` is presented only as **scenario-based estimated economic impact** under the historical base/10%-budget assumptions;
   - no realized ROI/savings claim;
   - demo sliders expose sensitivity to cost, intervention cost, and assumed efficacy.

9. **Positioning**
   - current wording: **Research-validated Decision Intelligence Prototype with a Production Roadmap**;
   - production-readiness documents are marked as roadmap/legacy rather than certification.

## Validation performed in the patch environment

- API/dashboard contract/safety/end-to-end critical tests: passed.
- Cross-domain protocol tests: passed and now check that fabricated results cannot be returned.
- `scripts/verify_closure_reproducibility.py`: passed.
- Python compile check for `src/` and `scripts/`: passed.
- Full legacy suite was also attempted. It collected 659 tests and, in this environment, produced **576 passed, 44 failed, 6 skipped, 34 errors**. Major blockers include the absence of a Parquet engine (`pyarrow`/`fastparquet`) and legacy tests that still assume the original out-of-repository `../scms`, DataCo, and Olist source layout. This patch therefore does **not** claim 659/659 current-environment closure.

See `artifacts/closure_manifest.json` for the final machine-readable verification record.
