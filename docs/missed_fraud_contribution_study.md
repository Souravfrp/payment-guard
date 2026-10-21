# Why the full logistic model missed six labelled fraud transactions

## Research question

I have already checked that the saved February validation predictions reproduce the published logistic-regression and feature-ablation results. My next question is narrower: what explains the fitted score *within the model* for each of the six fraud-labelled transactions below the 1,680-review cutoff?

I do not want to select a more complex model merely because it is commonly used for fraud detection. First, I want to understand which features contribute to the existing model's ranking, and whether the misses reveal a specific limitation that can be tested fairly in later experiments.

## Data boundary and reproducibility

I use only the February 2026 validation slice and the saved `logistic_log` pipeline from `models/logistic_ablation_v1`. I do not refit the model, change its preprocessing, recompute its coefficients, or score March–April holdout transactions. I join predictions to the prepared validation records using `source_row`, verify labels and timestamps, and re-create the ranking with the saved label-independent tie priority. I also check that the six missed source rows, total missed amount, and the 196 caught positives match the previously recorded experiment.

I check that the saved probabilities are reproduced by the original pipeline. These checks are conditional on the trusted local artifacts and the existing fitted model; their full verification was handled separately in the reproducibility checkpoint.

## Mathematical reconstruction

The fitted logistic score is

\[
z_i = \beta_0 + \boldsymbol{\beta}^{\mathsf T}\widetilde{\boldsymbol{x}}_i,
\qquad
p_i=\sigma(z_i)=\frac{1}{1+e^{-z_i}}.
\]

Here \(\widetilde{\boldsymbol{x}}_i\) is the vector **after** applying the saved log transformation, training-fitted standardization, sine/cosine construction and one-hot encoding. The contribution of the encoded coordinate \(j\) is \(\beta_j\widetilde{x}_{ij}\). I sum the contributions of one-hot indicators belonging to the same original categorical field and of sine/cosine coordinates belonging to the same calendar cycle. This recovers 13 source-feature groups. I keep the intercept separate because it is shared by all examples.

For a missed transaction \(m\) and a comparator \(c\), their difference is

\[
z_m-z_c=\sum_g\left(C_g(m)-C_g(c)\right).
\]

The intercept cancels exactly, and the script checks numerically that the grouped differences sum to the reconstructed log-odds gap. Because \(\sigma\) is strictly increasing, this log-odds comparison preserves the probability ranking. It does not require treating the probabilities as calibrated frequencies.

## Comparator selection and alternatives

For every missed labelled positive, I select a **caught positive with the same transaction type and category**. Among these eligible comparators, I minimize Euclidean distance in the pair consisting of log-transformed amount and origin balance, using scaling parameters learned from training data:

\[
d(m,c)^2=
\left(\frac{\log(1+a_m)-\log(1+a_c)}{s_a}\right)^2 +
\left(\frac{b_m-b_c}{s_b}\right)^2.
\]

Here \(a\) is amount, \(b\) is origin balance, and \(s_a,s_b\) are the fitted training standard deviations. Matching uses these two monetary dimensions rather than the final model probability; matching on the final score would largely restate the very difference I am trying to investigate. Ties are broken deterministically by source-row order. One caught transaction may be reused for multiple misses; the pairs are not independent matched samples.

This is **not** propensity-score matching, causal inference, or an estimate of a treatment effect. I am inspecting already-fitted model outputs to identify descriptive contrasts. Holding the observed type/category equal helps with interpretation, but the fitted model still uses other features and interactions arising from its encoding. The nearest monetary comparator is not necessarily close in every other feature.

I considered fitting a tree model or applying model-agnostic SHAP at this stage. Neither is required to explain a *linear model in its own transformed coordinate system*: exact coefficient times transformed-value contributions already reconstruct its log-odds. A different model can be considered later if these diagnostics motivate a concrete hypothesis, and it must then be evaluated with the same time separation and review-capacity constraints.

## Algorithmic viewpoint and complexity

1. Read validation predictions and the February-only prepared records; verify a one-to-one join and the frozen validation totals.
2. Sort \(N\) validation scores with saved tie priorities. Identify the six positives below the top-\(k\) boundary.
3. For each missed positive, search through at most \(P\) caught positives for a same-type/category monetary nearest neighbor.
4. Transform the 12 selected rows with the saved pipeline and compute \(\widetilde{X}\odot\beta\). Aggregate one-hot and cyclical terms by original feature group.
5. Write the matched-pair and contribution tables and two chart files, without overwriting an existing output.

Sorting requires \(O(N\log N)\) time. The matching stage requires \(O(MP)\) comparisons for \(M\) missed and \(P\) caught positives, using constant-dimensional distances. Contribution reconstruction takes \(O((2M)D)\) arithmetic operations for \(D\) transformed columns, excluding sparse-to-dense conversion overhead. The implementation materializes dense transformed values only for the 12 selected records, not the full validation dataset.

## Saved outputs and how to read them

- `results/tables/missed_fraud_matched_pairs.csv` records miss/comparator IDs, ranks, probabilities, log-odds, cutoff and standardized matching distance.
- `results/tables/missed_fraud_group_contributions.csv` records the original source-feature-group contributions for each case, the total log-odds, and the saved probability.
- `results/tables/missed_fraud_contribution_differences.csv` records each group contribution difference from missed to matched caught and should add up to the overall log-odds difference.
- `results/figures/missed_fraud_score_comparison.png` compares six missed scores with their matched caught scores and the review cutoff.
- `results/figures/missed_fraud_feature_differences.png` visualizes the group-level differences. The color scale encodes **log-odds differences**, not probabilities or causal feature effects.

These outputs are *not populated in GitHub until the local run has passed*. I will report actual observed feature differences only after I see that run's printed results and figures.

## Limits and next experiment

All data are synthetic. The category field already exhibited an unusually strong predictive association in the ablation experiment, so apparent patterns may be artifacts of the data-generation mechanism. Six misses are too few to support general statements about fraud behavior. Local explanation does not answer whether a feature is available at transaction decision time, whether probabilities are calibrated, or whether the ranking generalizes beyond February.

After validating these six examples, I will study probability calibration and temporal stability before making any decision to change the model. I will not use the final holdout to decide which features or model to try next.

## Commands

From the project root, with `payment-guard-verify` active:

```bash
PYTHONPATH=src python -m pytest -q tests/test_missed_fraud_contributions.py

PYTHONPATH=src python scripts/analyze_missed_fraud_contributions.py
```

The script writes five **new** files and refuses to replace existing outputs. It reads only existing project-generated local model artifacts; serialized joblib models from an untrusted source should never be loaded.
