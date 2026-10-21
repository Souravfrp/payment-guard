PAYMENT GUARD — FEATURE-FAMILY ABLATION PATCH v1

1. Download this ZIP into ~/Downloads on your Mac.
2. From the payment-guard repo root on research/xgboost-chronological-benchmark:

   unzip -t "$HOME/Downloads/payment_guard_xgboost_feature_ablation_patch_v1.zip"
   unzip -n "$HOME/Downloads/payment_guard_xgboost_feature_ablation_patch_v1.zip" -d .

3. Run synthetic-only tests:

   PYTHONPATH=src python -m pytest -q tests/test_xgboost_feature_ablation.py
   PYTHONPATH=src python -m pytest -q

4. Run on your existing local AMLNet prepared development data only:

   PYTHONPATH=src python -m payment_guard.xgboost_feature_ablation
   PYTHONPATH=src python -m payment_guard.xgboost_feature_ablation --verify-only
   open models/xgboost_ablation_v1

5. Send the printed January/February 1% result table and the aggregate
   metrics.csv + changes_vs_full.csv and ablation_at_one_percent.png for review.
   Do NOT upload *predictions.parquet, *.joblib, prepared transaction data, or
   the complete models/ directory.

The final March-April 2026 holdout is filtered out and not scored. This is
pre-final historical research only; it does not select a final model.

No GitHub files are modified by extracting this ZIP beyond the three new
untracked source/test/protocol files. Do not commit until the run is verified.
