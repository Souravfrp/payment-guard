# Reproducing my saved calibration predictions

## Why I added a second version

My first chronological calibration experiment (`models/calibration_temporal_v1`) saved February and January predictions, metrics, reliability bins and artifact hashes, but **not** the fitted logistic pipelines and calibration functions. Verifying file hashes does not reproduce fitted predictions. I am keeping that initial run unchanged and adding a **new versioned run** that saves and reloads six locally generated fitted objects: two full-feature logistic pipelines, two nonnegative-slope sigmoid calibrators and two isotonic calibrators.

The original January selection rule remains frozen: minimum January log loss, with Brier score resolving differences of no more than 1e-6 and a fixed simplicity order for remaining ties. February is a historical out-of-time comparison, **not** a new method selection period. The March–April holdout remains unscored.

## What the verifier tests

For each of the two stages, I reconstruct probabilities from the saved logistic pipeline's `predict_proba` and the reloaded calibration mappings. I check the recorded SHA-256 artifact digests (accidental changes, **not** provenance or authenticity), the original prepared-file fingerprint, aligned source rows, timestamps, labels and saved probabilities. I recompute and compare all saved probability and fixed-budget metrics from these probability vectors and verify January's declared selection against the saved metrics. A separate opt-in cross-run comparison matches v2 against the original v1 row-level scores and numerical metrics. Each run has its own unique output folder. No generated model or row-level predictions will be committed to Git.

This is a local reproducibility check, conditional on trusting my own joblib files and the prepared dataset. Joblib files can execute Python objects during deserialization, so I never load serialized models from untrusted sources.

## Chronological protocol

| Stage | Logistic fitting | Calibrator fitting | Evaluation |
|---|---|---|---|
| A | October–November 2025 | December 2025 | January 2026 |
| B | October–December 2025 | January 2026 | February 2026 |

Two models are fitted, not just one model rescored: the training period changes between A and B. Comparing calibrated to uncalibrated scores **within Stage B** holds that base fit constant. Comparing to the previous October–January baseline does not isolate calibration effects, because its base fitting window differs.

## Mathematics

For a fitted logistic score \(p\), the uncalibrated output is \(p\). Sigmoid adjustment takes \(\sigma(a\,\operatorname{logit}(p)+b)\) with \(a\ge 0\) fitted on the later calibration month. Isotonic adjustment fits a nondecreasing piecewise mapping \(f(p)\) to labels on that month. Coefficients and monotonic maps are fitted independently for Stages A and B. Their model scores are ranked at the previously frozen review budgets. A nondecreasing mapping can introduce ties and thus may alter top-k selection at boundaries even though it does not reverse any strictly ranked pair.

## Run commands on my Mac

With my `payment-guard-verify` environment active, from the repository root:

```bash
PYTHONPATH=src python -m pytest -q tests/test_calibration_saved_v2.py
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m payment_guard.calibration_saved_v2 \
  --compare-v1 models/calibration_temporal_v1
```

This creates `models/calibration_saved_v2` **only if that folder does not already exist**. The v1 output folder is only read. After it completes, the verification can be repeated without retraining:

```bash
PYTHONPATH=src python -m payment_guard.calibration_saved_v2 \
  --verify-only --compare-v1 models/calibration_temporal_v1
```

If either verification reports a mismatch, I will inspect it before interpreting the new results. I will not delete the original run, silently change preprocessing or claim reproducibility based only on matching summary CSVs. The public repository can include this implementation and synthetic-fixture tests, but readers still need the independently obtained licensed data to repeat the full-data study.

## Limits and publication plan

Both v1 and v2 are on synthetic data. File hashes, fitted-object reconstruction and close agreement to prior local predictions do not establish statistical validity, external reproduction or operational payment-fraud performance. The changed frequency of positive labels between months is one reason to investigate probability calibration, not a proof of concept drift.

After local validation, I will write a compact aggregate results table, annotate the corrected *shared-transaction* reliability plots and explain the differences between January's chosen method and February's observed performance. Only then will I commit the new code and research narrative; the original holdout remains untouched until the modelling and review-policy choices have been frozen.
