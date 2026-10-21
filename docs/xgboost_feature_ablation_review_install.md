# Aggregate-only XGBoost ablation publication check

This ZIP adds an independent **public** aggregate audit and results narrative after the already completed private local model fitting. It does not retrain or access March–April data.

## Install

From the Payment Guard repository root, after downloading this package into `~/Downloads`:

```bash
unzip -t "$HOME/Downloads/payment_guard_xgboost_ablation_aggregate_patch_v1.zip"
unzip -n "$HOME/Downloads/payment_guard_xgboost_ablation_aggregate_patch_v1.zip" -d .
```

The package includes the two user-supplied aggregate CSVs, the original submitted 1%-budget chart, a complementary AP chart, a documented analysis, a reproducible aggregate-only script and ten isolated tests. The copied original CSVs and original chart retain their uploaded bytes and expected SHA-256 digests.

## Check before staging

```bash
PYTHONPATH=src python -m pytest -q tests/test_xgboost_ablation_aggregate_review.py
PYTHONPATH=src python -m pytest -q
python scripts/plot_xgboost_ablation_aggregate_review.py --check-only
python scripts/plot_xgboost_ablation_aggregate_review.py
open models/xgboost_ablation_aggregate_review_v1
git status --short --branch
```

The `--check-only` option is **read-only**; it audits both committed aggregate tables, their 48-row consistency, each budget and confusion matrix, feature identity, AP consistency and all deltas, and verifies public input hashes. The ordinary command also generates two charts from the public aggregate rows under `models/`, which is ignored by Git. It deliberately **refuses to overwrite** that output folder if already present.

Publication should contain only the explicit script, test, documentation, aggregate CSVs and PNGs. Never stage `models/`, private row-level Parquet files, joblib objects or raw AMLNet transactions. The full local suite before this patch had 161 passing tests; these ten tests should bring it to **171**, assuming no other changes.

The review cannot substitute for the existing `PYTHONPATH=src python -m payment_guard.xgboost_feature_ablation --verify-only` saved-model reconstruction, which must also pass on the Mac.
