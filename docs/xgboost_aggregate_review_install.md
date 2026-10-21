# Mac installation and verification — XGBoost aggregate review patch

This patch adds only **four aggregate result tables, five publication-review PNG figures, a first-person mathematical results document, one aggregate-only plotting/auditing script, and eight focused tests**. It does not replace the previous XGBoost experiment code, saved models, logistic baseline, or any March–April holdout artifacts. Installation is deliberately separate from committing to GitHub.

From Terminal, with `payment-guard-verify` activated:

```bash
cd "$HOME/Documents/payment-guard"
git status --short --branch
unzip -t "$HOME/Downloads/payment_guard_xgboost_aggregate_review_patch_v1.zip"
unzip -n "$HOME/Downloads/payment_guard_xgboost_aggregate_review_patch_v1.zip" -d .
PYTHONPATH=src python -m pytest -q tests/test_xgboost_aggregate_review.py
PYTHONPATH=src python -m pytest -q
python scripts/plot_xgboost_aggregate_review.py
python scripts/plot_xgboost_aggregate_review.py --check-only
open models/xgboost_aggregate_review_v2
```

Expected: all **eight new aggregate tests** pass; the full suite should increase from 133 to **141** passing tests in the same environment if no new changes are present. The review script writes five *local ignored* regenerated figures plus SHA-256 manifest to `models/xgboost_aggregate_review_v2/`, refuses overwrite, and its `--check-only` verifies saved input/figure hashes. The ZIP also includes the five preview public PNGs under `results/figures/xgboost_benchmark_v1/`, but these should not be committed before visual inspection and the local tests.

The new source tables in `results/tables/xgboost_benchmark_v1` were copied **byte for byte** from the user-provided aggregate ZIP. No row-level predictions, prepared Parquet, trained joblib, JSON model exports or final holdout records are present.

**Send the full test output and script verification output** before creating any commits. Do not run the original full-data XGBoost fitting command again: the original `models/xgboost_benchmark_v1` is already complete and verified.
