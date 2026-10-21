# Running the frozen March–April holdout exactly once

**Do not score the holdout until the fixture tests and pre-score fingerprint check pass.**

The prior frozen choices are recorded in [final_holdout_frozen_protocol.md](final_holdout_frozen_protocol.md)
and [final_holdout_freeze_v1.json](../config/final_holdout_freeze_v1.json).
This script implements those commitments; it does not change model selection or review budgets.

## 1. Pull and test WITHOUT evaluating holdout records

From the payment-guard-verify environment on my Mac:

    cd "$HOME/Documents/payment-guard"
    git pull --ff-only
    PYTHONPATH=src python -m pytest -q tests/test_evaluate_frozen_holdout.py
    PYTHONPATH=src python -m pytest -q
    python scripts/evaluate_frozen_holdout.py --preflight-only
    git status --short --branch

The eight new fixture-only tests use synthetic records. The preflight rechecks the
previously created private file models/final_holdout_preflight_v1.json by hashing
frozen code, prepared-data bytes, existing run manifests and fitted artifacts.
Neither command produces predictions from the reserved final holdout. If a test,
fingerprint or dependency check fails, stop before scoring.

## 2. Explicit FIRST holdout scoring (only after all the checks)

The following command CONSUMES the March–April final holdout. Do not run it
until the fixture tests, full project suite, and no-scoring preflight have passed:

    python scripts/evaluate_frozen_holdout.py --score-frozen-holdout

It refuses to score if the original private fingerprint is missing, different,
or the immutable output already exists. Only test-split rows are loaded for
scoring, with expected 253,190 rows dated March–April 2026 and timestamp +
source_row ordering. Pre-existing fitted XGBoost/logistic objects and maps are
used; no fit or calibration training occurs. Primary pooled endpoint: top 1%
capture difference of uncalibrated full XGBoost minus uncalibrated full logistic.
The seeded tie rule, other review budgets, sensitivity models, calibration
diagnostics and March/April descriptive checks follow the predeclared freeze.

The script writes only inside the Git-ignored local models/final_holdout_v1:

- primary_summary.json: primary pooled counts and caveat
- metrics.csv: pooled and month-specific aggregate metrics
- paired_overlap.csv: fraud-capture disagreement and total review-list overlap
- pooled_primary_review_budgets.png and pooled_primary_disagreement.png
- private_predictions.parquet: confidential row-level scores, NEVER commit
- run.json: local immutable provenance and artifact-hash manifest, NEVER commit

A partial first run should be investigated and documented, not silently overwritten.
Never alter the model or reporting rules after seeing any holdout outcome.

## 3. Read-only result verification

After a successful first scoring run:

    python scripts/evaluate_frozen_holdout.py --verify-only

This recomputes the counts from the **original saved private predictions**
and checks the model/data/score-file fingerprints. It does not call the
models for a second scoring pass or select a new winner.

## 4. Public reporting

Only copy aggregate tables, charts and the candid research interpretation
into tracked results/ after the read-only verification passes. Do not upload
raw/prepared data, trained models, original predictions, or the entire models
folder. Include the uncertainties around synthetic data, partial April and
the unproven real-world decision-time provenance of category/type.

The scorer and these instructions are committed **after the protocol freeze but
before the first holdout score**. The scorer's exact source SHA-256 is included
in the first local holdout-run manifest.
