# Frozen protocol for the March–April 2026 holdout — v1

**Status: research protocol frozen on GitHub before holdout scoring.** This document locks the *analysis choices*, not the eventual results. The March–April holdout has not been scored by this workflow. The source's full-period labels were audited during initial preparation; I do **not** describe those rows as never seen in any sense.

The machine-readable specification is [`config/final_holdout_freeze_v1.json`](../config/final_holdout_freeze_v1.json). The specification pins the Git blob IDs of the existing modeling and verification source files to the pre-freeze research state (`84dd8c114e57c6b58876a8d03c41bea5fe9591c4`). Modifying any pinned implementation requires a documented protocol amendment **before** the holdout is opened; it must not be a response to holdout outcomes.

## Research question and historical basis

I want to know whether XGBoost's modest historical advantage over the same-window logistic baseline survives a final chronological evaluation on the synthetic AMLNet v2 dataset. I will not interpret a good result as evidence of performance on actual payment systems.

I compared the models on previously explored January and February periods with the same 13 permitted source features. In January, at 1% reviews, both families caught 271 of 281 positive labels. In February, XGBoost caught 198 of 202 and logistic regression 197 of 202; their paired fraud-capture overlap was 196. Matched feature removal showed that both families depend substantially on `category`. I choose the full-feature XGBoost **as the primary candidate before the final test**, while keeping the already fitted logistic regression as a mandatory reference. This is a proposed candidate, not a claim that XGBoost is statistically superior.

The project feature policy retains the top-level `category` and `type` provisionally as decision-time candidates. Their exact synthetic generation mechanism and their real-world availability before authorization have not been independently verified. That assumption is a material limitation of my primary result; the sensitivity comparisons below are therefore required.

## Frozen data and model identity

| Component | Frozen choice |
| --- | --- |
| Raw source | AMLNet v2 synthetic data, single prepared Parquet file already used in development |
| Holdout | Only rows in the prepared `test` split, dated March–April 2026; April is partial |
| Chronology | Rows in ascending transaction timestamp, then `source_row`; unique source references |
| Expected holdout size | 253,190 rows from the previously documented split audit; do not use this to select a model |
| Primary model | **Existing October–December 2025 fitted full-feature XGBoost**, `models/xgboost_benchmark_v1/B_xgboost.joblib` |
| Main comparator | **Existing October–December 2025 fitted full-feature logistic regression**, `models/calibration_saved_v2/B_base.joblib` |
| XGBoost choice | December-selected `depth3_240`: depth 3, 240 boosting rounds, learning rate 0.05, unchanged XGBoost constants |
| Source features | The same 13 features and train-fitted encoders, amount transformation and calendar features already verified |
| Primary score | **Uncalibrated predicted probability** from each saved fitted model |
| Primary capacity | Top **1%** of the combined March–April holdout, using `ceil(0.01 × N)` reviews per model; for 253,190 rows, 2,532 reviews |
| Tie treatment | Identical, label-independent seeded priority `np.random.default_rng(20261005).permutation(N)`, ranked with `np.lexsort((ties, -scores))` |
| Primary comparison | Paired difference in positive labels captured at the 1% budget, **TP_XGBoost − TP_logistic**, with each model's TP, FP, precision and recall also shown |
| Code boundary | The five implementation files and their Git blob IDs recorded in the machine-readable lock |
| Changes after scoring | **No model, threshold, feature, calibration, budget, or reporting-rule changes chosen from holdout results** |

Both the primary XGBoost and logistic models must use their existing trained preprocessing. Do not retrain on January or February, refit preprocessing on the test rows, tune hyperparameters, adjust review capacity, or choose a threshold using test labels. The original stage-B experiments fitted on October–December; their January calibration maps remain separate saved objects.

## Predeclared secondary analyses

All of the following are reported even if they disagree with the primary result. None can replace the primary comparison or motivate another round of test-based model selection.

1. **Other budgets:** capture/precision/recall at 0.1% and 0.5% using the same ranking and ceiling rule. Report average precision over the whole combined holdout.
2. **Probability diagnostics:** log loss and Brier score on the two raw model scores. Also evaluate the already fitted XGBoost and logistic sigmoid and isotonic mappings on the same rows as separately labeled exploratory calibration diagnostics. Their February performance must not be used to retroactively call a single mapping the chosen final winner; no calibrator is selected by holdout outcomes.
3. **Feature sensitivity:** compare saved February-stage **uncalibrated** full models against the **existing October–December fitted** `without_category` and `without_both` ablation models for both families. These are historical refits—not fresh holdout-specific fits. They test robustness to the top-level category/type assumptions, not causal feature effects.
4. **Paired disagreements and workload:** for each budget, count positive labels found by both families, only XGBoost, only logistic, and neither. Separately report the overlap of *all* reviewed transactions to describe queue stability; do not confuse these two overlaps.
5. **Temporal breakdown:** report March and partial April separately as descriptive checks on the same fixed scores. Apply each month's top-1%-of-that-month capacity when reporting month-specific review counts; this is distinct from and cannot replace the pooled primary result. Include positive label counts and class prevalence in each month.

The pooled final result is the single primary comparison. A changed March-versus-April pattern will be documented, not turned into a new training or calibration choice. No multiple-comparison adjusted success claim is predeclared; any confidence or significance discussion must recognize the rare positives, synthetic design, and possible temporal dependence.

## Holdout access gate — do not score until all pass

Before running any holdout evaluation, record a **read-only preflight** with:

- This protocol and JSON freeze committed on the research branch, before the first holdout-model evaluation.
- The pre-freeze source-code Git blob IDs matching the JSON lock; deviations require a separately documented amendment made *without* consulting holdout outcomes.
- The prepared dataset SHA-256 unchanged from the already saved logistic, XGBoost and ablation run manifests; the hashes and manifest references agree.
- Fitted `joblib` artifacts, existing calibration maps, the six relevant February-stage ablation models, and saved run manifests present locally and matching their existing artifact digests. Never load untrusted `joblib` files.
- Previous development-only verification commands passing. Nothing in the preflight computes predictions on `split=test` or reads test labels. Hashing the complete prepared file for integrity reads bytes, not scored records.

The existing full, logistic, and ablation manifests are **local** and are not GitHub-published artifact locks. Because GitHub does not contain those private model files, I cannot truthfully claim their individual hashes have already been pinned publicly. A separate **pre-score artifact fingerprint** must be recorded locally (and its non-sensitive hashes may be committed as a pre-score amendment) before the March–April score is produced.

## Execution and reporting rule

After the gate passes, use a single dedicated holdout scoring invocation that loads **only the test split for scoring**, validates timestamp boundaries, identities and feature schema, and uses only the frozen fitted models. That evaluation command has **not been implemented or run by this documentation commit**. I will verify its fixture-only tests and read-only preflight first. The first scoring run will save an immutable versioned local manifest and aggregate outputs; a later **read-only verification** of identical outputs is permitted, but testing revised candidate choices against the holdout is not.

The final public report must include all primary and predeclared secondary counts, class prevalence, full review workloads, calibration caveats, feature-assumption limitations, and any adverse or surprising results. Only aggregate tables and figures are appropriate for the public research repository; prepared records, trained model files, and individual predictions remain local.

**References:** [original XGBoost benchmark](xgboost_chronological_protocol.md), [January–February ablations](xgboost_feature_ablation_first_results.md), [feature policy](../results/tables/amlnet_v2_initial_model_feature_policy.csv), [training category investigation](training_category_analysis.md), [paired disagreements](xgboost_paired_disagreement_protocol.md).
