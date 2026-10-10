# Review-budget methodology and mathematical formulation

## Research question

I want to know whether a simple, interpretable model can rank synthetic payment transactions for investigation more effectively than reviewing the largest amounts first, and what that ranking costs when investigation capacity is limited.

## Data, time boundary and features

AMLNet v2 contains 1,090,000 synthetic transactions and 1,411 positive `isFraud` labels. The prepared chronological partitions contain 668,820 training rows (931 positives), 167,990 February 2026 validation rows (202 positives), and 253,190 later holdout rows (278 positives). The last partition was not scored in this experiment; earlier exploratory audits did inspect the broader dataset, so it is not an entirely untouched dataset.

The model uses 13 source predictors, including amount, origin balance, calendar fields, payment type and category, payment method, device, and location. Supplied risk scores, duplicated target information, post-transaction balances, and target-revealing annotations are excluded. Device/location availability at authorization time is an assumption, not verified deployment evidence.

## Model

For transformed feature vector x, logistic regression estimates p(y=1|x)=sigmoid(b+w^T x), where sigmoid(z)=1/(1+exp(-z)). The fitted objective minimizes binary cross-entropy with L2 regularization, using the implementation's C=1.0 setting, an intercept, and no class weighting or resampling. This describes the optimization family; exact solver conventions are specified in the training implementation.

The raw model uses transaction amount directly; the log model uses log(1+amount). Both standardize monetary predictors with training-only statistics, use training-only one-hot encodings, and use fixed cyclical calendar features. All eight logistic variants are refitted: full raw/log, without type, without category, and without both. Removing a feature and refitting is a predictive ablation, not a causal intervention. The two reference rankings are constant training prevalence (seeded, label-independent tie order) and descending transaction amount.

## Retrospective decision rule and metrics

For validation scores s_i, select the top k by descending score, breaking ties with the same saved label-independent priority. The predeclared review fractions 0.1%, 0.5%, and 1% correspond to k=168, 840, and 1,680. With P=202 positives, caught=TP, missed=P-TP, false alarms=k-TP, precision=TP/k, and recall=TP/P. Average precision evaluates ranking over cutoffs; log loss and Brier score describe probability predictions but do not establish calibration.

This is a whole-month retrospective top-k analysis. A live policy would need a time-respecting capacity mechanism, available-at-decision-time features, and separate calibration and drift checks.

## Exploratory economic scenarios

For a review cost c_r and assumed per-missed-positive cost c_m, the illustrative objective is C(k,m)=c_r*k+c_m*FN(k,m). This is not an estimate of actual bank losses: costs are hypothetical, false positives incur review cost rather than an additional specified penalty, and there is no cost for delays, customer harm, or blocking. The cost comparison searches only the three existing budgets and ten existing rankings; it cannot identify a globally optimal budget or prove financial savings. The INR amounts are scenario assumptions, not currency conversions from transaction amounts.

The transaction-value diagnostic sums the amounts of fraud-labelled transactions selected within a budget, divided by total fraud-labelled transaction amount. This is *labelled transaction-value capture*, not prevented loss, recovered money, or expected loss. Report transaction units as recorded by the dataset, with source documentation, rather than silently treating them as INR.

## Evidence and reproducibility boundaries

The committed aggregate CSV `results/tables/logistic_ablation_summary.csv` supports the model-level counts. Local files `models/logistic_ablation_v1/validation_predictions.parquet` and `data/processed/amlnet_v2_prepared.parquet` are required for row-level value capture, missed-case investigation, and within-band comparisons. Scripts under `scripts/analyze_*.py` reconstruct those diagnostics; publishing the scripts does not mean this GitHub update independently reran them. An earlier exploratory amount-band fraud-rate calculation overflowed because the grouped fraud count had `int8` dtype. The owner reran it using floating-point arithmetic and confirmed the corrected rates; see the [follow-up table](review_cost_followup.md#corrected-validation-amount-band-fraud-rates). This correction concerns exploratory rates, not the committed ranking metrics.

## Interpretation and limits

The full log model caught 196/202 positives at 1,680 reviews, versus 188/202 for the raw model, but 1,484 of the log model's selected transactions were non-fraud-labelled. Removing category reduced the log model to 130/202, showing sensitivity to synthetic category patterns. The log transformation's apparent advantage is conditional on these refitted multifeature models, this dataset, and this validation month. No confidence interval, significance claim, production suitability, or untouched holdout performance is established.

## References

- AMLNet v2 dataset and license: https://doi.org/10.5281/zenodo.21237971
- scikit-learn LogisticRegression documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- scikit-learn average precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- scikit-learn Brier score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
- scikit-learn log loss: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
- Internal preregistered plan: [baseline_experiment_plan.md](baseline_experiment_plan.md)
- Aggregate ablation results: [feature_ablation_results.md](feature_ablation_results.md)
