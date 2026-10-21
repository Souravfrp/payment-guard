# Chronological XGBoost feature-family ablations — locked protocol

**Status: protocol and executable code prepared; real-data results are not available until the local run is verified.** I will not choose a final model, tune on February, or touch the March–April 2026 holdout during these experiments.

## Research question

The earlier XGBoost benchmark modestly exceeded the matched logistic ranking on February's historical 1% budget (198 versus 197 fraud-labelled transactions caught). Its normalized average-split-gain feature grouping was concentrated in `category` (66.64%) and `type` (17.41%). I want to test how much the apparent model-family advantage changes when either original source feature group is unavailable. The grouped gain is descriptive, not causal and not a percentage of total model gain. Strong synthetic features may exploit specific generator relationships; I will not assume that they establish real-world suitability.

## Locked and comparable design

| Variant | Removed source columns | Kept original source columns |
| --- | --- | ---: |
| `full` | Nothing | 13 |
| `without_type` | `type` | 12 |
| `without_category` | `category` | 12 |
| `without_both` | `type`, `category` | 11 |

For **both logistic regression and XGBoost**, I use the same original source predictors, excluded only as declared above. The existing preprocessing pipeline, including `log1p(amount)`, numeric scaling, calendar cycles and one-hot encoding, remains the same except for the removed categorical group. The full models are reused from the saved, locally verified benchmark; they are **not retrained**. I refit the three reduced-feature variants for each family and historical period.

- **January:** fit reduced-feature models only on October–November 2025 and evaluate on January 2026. XGBoost's one existing selected configuration came from December development AP.
- **February:** fit reduced-feature models on October–December 2025 and evaluate on February 2026. January labels do not select these models or hyperparameters.
- XGBoost keeps the **December-selected** `depth3_240` configuration with all its previous constants: depth 3, 240 rounds, learning rate 0.05, unit class weights, CPU histogram algorithm, regularization and random seed. No new search, early stopping, class reweighting or synthetic oversampling.
- Logistic regression retains its earlier regularization, solver and maximum iterations. Different fitted windows are not pooled for a causal month-on-month claim.
- For each fitted family/variant/period, score uncalibrated probabilities. I compare **fixed review budgets** of 0.1%, 0.5% and 1%; choose the top `ceil(budget × N)` transactions by score with the existing label-independent tie priority. I also report average precision, log loss and Brier score. Because sigmoid/isotonic maps were trained on different windows, the feature-ablation comparison holds those maps **out** rather than retuning them.

The main effects are *within the same month, same family and same budget*. For example:

\[
\Delta_{\mathrm{cat},\mathrm{XGB},\mathrm{Feb}} = \mathrm{TP@1\%}(\mathrm{XGB\ without\ category})-\mathrm{TP@1\%}(\mathrm{XGB\ full}).
\]

A second model-family comparison uses XGBoost minus logistic **with the same feature set**. Neither contrast is a causal estimate of individual feature effects: dropping a feature and refitting can change the other coefficients, trees and score ranks.

## Reproducibility and protected data

The code uses the project's verified `load_windows`, which filters the prepared Parquet to training and February validation and rejects out-of-period rows. A whole-file SHA-256 fingerprints the source file but does not imply holdout scoring. The script runs the pre-existing XGBoost saved-artifact audit before fitting, checks the original logistic run and the exact January/February source-row alignment, fixed selection criterion, population counts and tie permutation. It never imports March–April observations into training or evaluation.

The output folder is deliberately under ignored `models/xgboost_ablation_v1`. It contains **local-only** fitted pipelines and row-level predictions, plus an aggregate metric table, within-family deltas, chart, manifest and hashes. The optional `--verify-only` command reopens **trusted local** fitted objects, reconstructs the predictions, checks the original full-model predictions and recomputes every saved metric without retraining or rewriting. Do not upload the whole `models/` folder or share its row-level Parquet contents.

After the local run, I will audit the actual metrics and generate a separate *aggregate-only* public results package. I will document findings, limitations and references before making any model-family choice. April and March must remain outside evaluation until a written, frozen selection and calibration protocol is committed.

## Run on the Mac

In the existing `payment-guard-verify` environment and `~/Documents/payment-guard`:

```bash
PYTHONPATH=src python -m pytest -q tests/test_xgboost_feature_ablation.py
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m payment_guard.xgboost_feature_ablation
PYTHONPATH=src python -m payment_guard.xgboost_feature_ablation --verify-only
open models/xgboost_ablation_v1
```

**Do not rerun** the main command if the versioned output already exists. The default is fail-safe against overwriting previous analysis.

## Sources

- Chen, T. & Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754).
- XGBoost. [Model parameters and importance types](https://xgboost.readthedocs.io/en/stable/parameter.html).
- scikit-learn. [Average precision score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
- scikit-learn. [Permutation and feature importance caveats](https://scikit-learn.org/stable/modules/permutation_importance.html).
- Original project source: `docs/xgboost_chronological_protocol.md`, `docs/xgboost_first_aggregate_results.md` and `docs/xgboost_paired_disagreement_protocol.md`.
