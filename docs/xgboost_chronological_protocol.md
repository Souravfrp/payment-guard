# Why I am testing XGBoost after the logistic baseline

**Status: design and implementation before the full AMLNet run. No XGBoost performance results are claimed here.**

## Research question and restrictions

My first fraud-ranking model was a regularized logistic regression, with 13 input features and a log transform of amount. It is an explainable and useful benchmark, not evidence that a linear decision surface is optimal. I am adding one nonlinear model family—XGBoost—not because it has to win, but to test whether interactions and conditional thresholds add measurable value. The existing logistic artifacts and calibration decision remain frozen, and the older February findings remain historical development evidence.

All fitting, configuration selection, calibration and comparison use **only October 2025 through February 2026**. The prepared-file reader explicitly requests the `train` and `validation` split labels; its audited time checks exclude March–April. The SHA-256 checksum of the original Parquet file covers its bytes, including any embedded holdout bytes, but that is **not** examining, fitting on or scoring holdout rows. No March–April figures, labels, metrics or predictions may be created. The study is based on synthetic AMLNet v2; it cannot establish deployment performance.

## The two model families and the experimental sequence

| Model family | Source features | Training fit | Hyperparameters | Jan/Feb task |
|---|---|---|---|---|
| Logistic regression | Original 13, log amount; L2, C=1 | Oct–Nov for A; Oct–Dec for B | Unchanged, frozen baseline | Jan ranking / Feb saved calibration reference |
| XGBoost gradient-boosted trees | **Exactly the same original 13**, log amount, fitted preprocessing and category encoding | Oct–Nov for December model selection and A; Oct–Dec for B | Exactly three locked configurations selected on December **average precision**, see below | Jan ranking / Jan calibration and Feb model comparison |

1. **December selection (development):** Fit each of the three XGBoost candidates on Oct–Nov, score December and rank candidates by average precision; near-ties within `1e-6` resolve using smaller log loss then fixed candidate order. No January/February model comparison influences this configuration choice.
2. **January historical model-family diagnostic:** With the December-chosen XGBoost specification, fit XGBoost and logistic **both** on Oct–Nov and compare *uncalibrated ranking* on January. December is **not** used to fit a Stage-A probability calibrator because its labels were just used to choose XGBoost hyperparameters. This is an intentional departure from the earlier logistic calibration Stage A; do not compare January XGBoost uncalibrated probabilities with the earlier logistic's calibrated January claim as if the protocols were identical.
3. **February historical model-family and calibration comparison:** Fit the chosen XGBoost configuration on Oct–Dec, fit sigmoid and isotonic calibration mappings on XGBoost's January scores (no January training overlap), and apply them to February. Compare against the **unaltered** saved logistic Stage-B February predictions from `models/calibration_saved_v2` with identical Oct–Dec fitting, January calibration, 13 source features and tie-breaking priority. Verify existing logistic artifact hashes and row identity before reusing them.
4. **Model choice not yet frozen:** This historical development study does not choose a production winner or authorize final holdout access. Follow with paired uncertainty and temporal stability analyses and a realistic review-capacity protocol *before* a one-time March–April evaluation.

We do **not** perform random train/test splitting, synthetic minority oversampling, sample reweighting or a broad automatic search. An unweighted baseline permits a cleaner first probability comparison. A weighted variant (`scale_pos_weight`) could be a new **explicitly versioned** sensitivity study, but weighted logistic loss can distort probability interpretation; the class imbalance alone does not justify an undocumented weight.

## Locked XGBoost candidate table

| Name | Maximum depth | Trees | Shared learning rate | Minimum child weight | Weights |
|---|---:|---:|---:|---:|---|
| `depth2_120` | 2 | 120 | 0.05 | 5.0 | 1:1 |
| `depth2_240` | 2 | 240 | 0.05 | 5.0 | 1:1 |
| `depth3_240` | 3 | 240 | 0.05 | 5.0 | 1:1 |

Common settings: objective `binary:logistic`, histogram tree building, CPU, random seed `20261005`, four threads, `subsample=1`, `colsample_bytree=1`, `gamma=0`, L2 coefficient `lambda=1`, L1 coefficient `alpha=0`, `base_score=0.5`. With **no early stopping**, December is used only for one fixed candidate comparison. Fitted preprocessing uses the training period only and is not reused from a fit that saw future data. A search over three shallow variants is not an exhaustive optimum. We can justify it for reproducibility and overfit control under 200–300 positive examples per calendar month.

## Mathematical model and why it differs

For the transformed feature vector `x` and fitted coefficient vector `beta`, logistic regression uses:

\[
  p_{\mathrm{LR}}(y=1\mid x)=\sigma(\beta_0+\beta^\top x),\qquad
  \sigma(z)=(1+e^{-z})^{-1}.
\]

This learns an additive linear log-odds function in the chosen transformed features, including one-hot indicators and cyclic time terms. The log transformation and one-hot encoding do not make it a fully nonlinear interaction model.

XGBoost instead adds shallow trees to the *logit* score, with each tree making conditional splits:

\[
  F_T(x)=F_0(x)+\eta\sum_{t=1}^{T} f_t(x),\qquad
  p_{\mathrm{XGB}}(y=1\mid x)=\sigma(F_T(x)).
\]

Its regularized training objective is approximately

\[
  \mathcal L = \sum_{i=1}^n\ell(y_i,F_T(x_i))+
  \sum_{t=1}^T \left(\gamma L_t+\frac{\lambda}{2}\sum_{j=1}^{L_t} w_{tj}^2\right),
\]

where `L_t` is the number of leaves in tree `t` and `w_tj` its leaf scores. The second-order Taylor approximation uses gradients `g_i` and Hessians `h_i` of logistic loss. For a leaf with gradient sum `G` and curvature sum `H`, the regularized optimal update is

\[
  w^*=-\frac{G}{H+\lambda}.
\]

For a candidate split into left and right groups, approximate improvement is

\[
 \text{Gain}=\tfrac12\left[
 \frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}
 -\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma.
\]

This allows interaction effects without specifying interaction terms manually, but deeper trees or too many boosting rounds increase variance and risks of learning synthetic shortcuts. Histogram binning accelerates approximate split searches, but the resulting leaf partitions can be sensitive to rare categories and temporal changes.

## Evaluation, diagnostics and uncertainty

**Primary XGBoost hyperparameter-selection measure (December)** is average precision (AP). With precision `P_k` and recall `R_k` at ranked threshold `k`, scikit-learn uses a weighted, non-interpolated sum `AP = sum_k (R_k - R_(k-1)) P_k`. Unlike ROC-AUC, precision–recall analysis foregrounds the rare positive class. The fixed historical review budgets are **0.1%, 0.5%, 1%** of a month, with `k=ceil(fraction*n)`, and fixed seeded *label-independent* tie priority. For 1%, report positives caught, false alarms, recall and precision.

For a probability `p_i`, Brier score is `mean((p_i-y_i)^2)` and log loss is `-mean(y_i log p_i + (1-y_i)log(1-p_i))`. These measure different aspects than ranking and are not suitable for choosing a model merely because its AP or top-k capture is high. Sigmoid and isotonic calibrators are **fitted only on January Stage-B scores**, and compared on February without changing the hyperparameters selected on December.

Output graphs include January AP/positives comparison, February AP/positives/log-loss grouped bars, February raw-model precision–recall curves, an observed-versus-predicted February calibration graph, and a Stage-B original-feature-group split-gain chart. The latter uses **the same ten cohorts defined by the unchanged logistic *uncalibrated* score** for both families: these are apples-to-apples *group summaries*, but XGBoost's own high-risk ranking is not represented by those cohorts. The feature-group chart sums normalized split gain across one-hot and cyclic transforms; it is an explanatory diagnostic, **not** SHAP, a causal effect, or a guarantee the most-used features will generalize. Large ties and severe class rarity must be explicitly counted. Use actual labels, units, legible navy/orange contrasts and different line styles/markers, not colour alone. No graphs of March–April are permitted.

**Statistical limitations:** December's AP choice can itself overfit December; January/February have already been seen in the project; one or two historical months cannot establish time robustness or a confidence interval. It is not appropriate to report an XGBoost "winner" without paired block-aware uncertainty, sensitivity to changes in training months, realistic daily review constraints and a true one-time final holdout. The data-generation process may make fraud artificially separable; feature availability at decision time has not yet been independently verified.

## Reproducibility and data exclusions

All final outputs under `models/xgboost_benchmark_v1/` are local ignored artifacts. The run records packages, features, candidate configurations, source checksum, both fitted XGBoost pipelines, two calibrators, JSON exports of the two tree ensembles, row-aligned January and February predictions, metrics, feature-group importance summaries, five plots, artifact hashes and a completion manifest. The verifier reopens locally **trusted** saved objects, regenerates XGBoost scores and reports and checks hashes, chronology, and alignment. Joblib files are Python-pickle-based: never load artifacts from an untrusted source.

The old logistic results are checked against their run manifest hash and February transaction identity before comparison. Neither original logistic outputs nor raw prepared Parquet are overwritten. A GitHub clone does not contain the local AMLNet synthetic dataset or individual predictions, so passing unit tests in CI is not equivalent to reproducing all 1,090,000 source rows.

## References

1. Chen, T. & Guestrin, C. (2016), **XGBoost: A Scalable Tree Boosting System**, KDD '16, pp. 785–794. https://doi.org/10.1145/2939672.2939785 ; open preprint https://arxiv.org/abs/1603.02754 . Formal tree objective, second-order split scoring, regularization and efficient tree construction.
2. XGBoost maintainers, **XGBoost Parameters**. https://xgboost.readthedocs.io/en/stable/parameter.html . Parameter definitions, `binary:logistic`, `tree_method=hist` and class-balance weighting cautions.
3. XGBoost maintainers, **Using the Scikit-Learn Estimator Interface**. https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html . API behaviour, evaluation sets and early stopping, which I deliberately do not activate.
4. scikit-learn developers, **Model Evaluation**. https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics . Non-interpolated average precision and rare-event ranking metrics.
5. scikit-learn developers, **Probability Calibration**. https://scikit-learn.org/stable/modules/calibration.html . Calibration, sigmoid/isotonic assumptions and how isotonic creates ties.

The references provide methodological context; numerical AMLNet findings will come **only** from the local run and its audit, not from these external papers.
