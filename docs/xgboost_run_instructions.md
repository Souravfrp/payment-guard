# Running and verifying my XGBoost benchmark

```text
Payment Guard XGBoost research checkpoint — installation and verification

This is a development experiment. March/April 2026 remain reserved.
It is NOT permitted to publish hypothetical January/February results.

The GitHub research branch contains this code and tests. On your Mac:

cd "$HOME/Documents/payment-guard"
git switch main
git pull --ff-only origin main
git fetch origin research/xgboost-chronological-benchmark
git switch --track origin/research/xgboost-chronological-benchmark

# Confirm the downloaded ZIP, then add only its new research files.
unzip -t "$HOME/Downloads/payment_guard_xgboost_checkpoint_v1.zip"
unzip -n "$HOME/Downloads/payment_guard_xgboost_checkpoint_v1.zip" -d .

# In the already working conda environment:
python -m pip install 'xgboost>=2.1,<4'
PYTHONPATH=src python -m pytest -q tests/test_xgboost_chronological.py
PYTHONPATH=src python -m pytest -q

# Run only if both suites pass. Expects models/calibration_saved_v2 locally.
PYTHONPATH=src python -m payment_guard.xgboost_chronological

# Independent read-only audit of this completed local run:
PYTHONPATH=src python -m payment_guard.xgboost_chronological --verify-only

# Review figures:
open models/xgboost_benchmark_v1/january_family_comparison.png
open models/xgboost_benchmark_v1/february_family_metrics.png
open models/xgboost_benchmark_v1/february_precision_recall.png
open models/xgboost_benchmark_v1/february_shared_cohort_reliability.png
open models/xgboost_benchmark_v1/stage_b_feature_importance.png

# Show summary and branch state:
cat models/xgboost_benchmark_v1/december_candidates.csv
python -c 'import pandas as pd; d=pd.read_csv("models/xgboost_benchmark_v1/metrics.csv"); print(d.loc[d.review_fraction.eq(.01), ["stage", "series", "true_positives", "false_positives", "average_precision", "log_loss", "brier_score"]].to_string(index=False))'
git status --short --branch

The model folder is ignored by Git. DO NOT add it to the repository.
Send the test output, run log, manifest and four figures for interpretation before publication.
```
