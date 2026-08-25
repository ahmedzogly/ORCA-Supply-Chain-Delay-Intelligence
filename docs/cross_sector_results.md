# Cross-Domain Portability Protocol — NOT External Validation

> **Evidence status: NOT VALIDATED**

The repository contains schema/adaptor prototypes for **DataCo** and **Olist**, but it does **not** contain a completed empirical external-validation run on either dataset. Previous hard-coded zero-shot and recalibration numbers were illustrative demo values and have been retired. They must not be quoted as experimental results.

## What is actually implemented

- Dataset-specific adapter / prediction-contract concepts.
- Separation of source and target domains.
- A protocol scaffold in `src/delay_intelligence/evaluation/cross_domain.py`.

## What is not established

- No measured SCMS → DataCo PR-AUC/F1.
- No measured SCMS → Olist PR-AUC/F1.
- No measured target-domain recalibration improvement.
- No demonstrated causal transfer.
- No demonstrated conformal coverage transfer.

## Required future external-validation protocol

1. Obtain the real target-domain datasets under an appropriate license/authorization.
2. Freeze a target prediction timestamp, outcome definition, and leakage exclusions.
3. Map only semantically compatible features; do not use ordinal encodings as semantic equivalence.
4. Evaluate the frozen SCMS model zero-shot on a chronological target holdout.
5. If recalibration is tested, isolate a target calibration window from the target test period.
6. Report PR-AUC relative to target prevalence, ROC-AUC, calibration/Brier score, subgroup performance, and uncertainty coverage/width.

Until those steps are run, DataCo/Olist are **portability prototypes**, not evidence of external generalization.
