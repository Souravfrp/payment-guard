# My chronological probability-calibration experiment

**Status: pre-run protocol.** No numerical calibration results are claimed in this document. I wrote the experimental design before fitting the new models. The original completed logistic regression and ablation results remain frozen and are not overwritten.

## Research question

The full log-amount logistic model caught 196 of 202 synthetic fraud-labelled February transactions when I retrospectively reviewed the top 1,680 scores. This was a useful ranking result, but it does **not** mean that a predicted score of 0.2 corresponds to fraud occurring in one out of five similar cases. I want to check reliability of the estimated probabilities separately from the top-k ranking.

The training-month positive-label rates varied: October 2025 226/120,879 (0.1870%), November 226/171,789 (0.1316%), December 198/190,055 (0.1042%), and January 2026 281/186,097 (0.1510%). The months are not exchangeable by assumption. October begins on October 13, so it is also a partial month. These sample sizes motivate forward-only calibration and transparent bin counts, not a claim that isotonic regression is automatically safe.

## Predefined stages and source separation

| Stage | Fit log-amount logistic model | Fit calibrator on predictions | Evaluate method |
|---|---|---|---|
| A: choose a calibration method | October–November 2025, 292,668 rows/452 positives | December 2025, 190,055 rows/198 positives | January 2026, 186,097 rows/281 positives |
| B: historical validation comparison | October–December 2025, 482,723 rows/650 positives | January 2026, 186,097 rows/281 positives | February 2026, 167,990 rows/202 positives |

I do not reuse the original October–January fitted logistic pipeline for January calibration: January was part of that model's training period. Both new base fits keep the previously declared 13 predictors, log amount, L2 penalty and `C=1.0`; preprocessing is fitted only on each stage's fit period. For each stage I fit the calibrators on **scores from a later month not used in the base fit**.

I choose a method by **January evaluation log loss in Stage A**. If two candidates' log losses differ by no more than `1e-6`, I select the one with lower January Brier score; a remaining exact tie favours uncalibrated, then sigmoid, then isotonic. This rule is specified before seeing new January predictions. I then refit the base model on October–December and fit the chosen type of calibrator on January scores (Stage B). February reports all three methods for transparency, but I will not switch the selected method based on February. Stage B's February results are a **historical validation comparison**, not an untouched out-of-sample final estimate, because February has already been used in earlier analyses. I will not score the March–April reserved holdout during this experiment.

## Mathematical methods

The fixed baseline logistic pipeline produces a probability `p` and log-odds `z = log(p/(1-p))`. The three candidates are:

1. **No recalibration:** return the original `p` unchanged.
2. **Monotone sigmoid/Platt-style adjustment:** fit `q(p) = sigmoid(a logit(p) + b)` on the calibration month by minimizing average binary negative log likelihood with **a >= 0**. A nonnegative slope rules out reversing the learned ranking. I bound `a` in `[0,20]` and `b` in `[-30,30]` for numerical stability and record this choice. If `a=0`, all adjusted scores tie; monotonicity alone does not guarantee identical top-k behavior in that degenerate case. Very extreme base scores are clipped to `[1e-12,1-1e-12]` before the logit transform, so some extreme ordering can also become tied.
3. **Isotonic regression:** fit a nondecreasing function `f` mapping original probabilities to empirical labels by minimizing the sum of squared residuals, constrained to `[0,1]`. Isotonic calibration may create large groups of ties, particularly with rare events.

These choices compare parametric and flexible calibration against no correction without claiming either always helps. Sigmoid and isotonic maps are nondecreasing, so ordinary strictly ordered scores normally keep their order, but isotonic can create ties. The original label-independent seeded tie priority is used for retrospective top-k reporting.

## Evaluation and interpretation

For each method I report log loss, Brier score and average precision. At the three previously declared monthly review fractions (0.1%, 0.5%, 1%), I report selected count, positive labels caught, false alerts, precision, recall and boundary ties. **Neither precision at k nor recall at k is a probability-calibration metric.** A method can improve log loss while leaving top-k outcomes unchanged. The fixed monthly review budgets remain retrospective exercises, not evidence of an operating daily review service.

Reliability bins group predicted scores by up to ten quantile intervals while keeping exact ties together. Each saved bin contains its number of transactions, positive count, mean probability and observed positive-label fraction. Small bin counts and zero-event bins should be read cautiously. Plots display percentage rates, not model accuracy in a causal sense. No unrequested sample weighting or artificial resampling is introduced.

The original full logistic baseline was fitted on **October–January**, whereas Stage B's fitted logistic base model uses **October–December**. Comparing those two raw-model results is not an isolated test of calibration: the base-model training period differs. The scientific comparison is between the three Stage B calibration choices **applied to the identical October–December fitted model**.

## Reproducibility boundaries and algorithmic cost

The script loads only prepared rows whose saved split labels are `train` or `validation`, enforces the known month counts, checks chronological fit/calibration/evaluation boundaries and refuses an existing output path. It writes a local `models/calibration_temporal_v1` folder with aggregate metrics, reliability bins, January and February row-level predictions, figures and a manifest. The model and prediction records are not automatically added to Git. The prepared file is hashed as a whole for provenance, which is not the same as scoring its reserved test rows.

Two fitted logistic models cost their usual iterative optimization time; each requires feature encoding learned on its fit window. Calibrator fitting is comparatively small, though isotonic sorting on the calibration window costs `O(C log C)` for `C` calibration rows. Evaluating the ranking of `N` records costs `O(N log N)` for each method; reading and hashing the prepared Parquet may dominate I/O. A later reviewer cannot repeat the full-data study without separately obtaining and preparing the same licensed synthetic dataset.

## Limitations and next decision

A month with about 200 positive examples is enough to try both methods, not to guarantee stable fine-grained calibration. The synthetic category structure may dominate predictions and may change across time. Labels and predictors are assumed to be available as recorded; no detection latency, production costs or real payment fraud claims follow. The March–April holdout is reserved until the modelling and capacity-policy choices are frozen. I will inspect the Stage A outcome and numerical diagnostics before committing any claims about Stage B performance.

## Local commands

```bash
PYTHONPATH=src python -m pytest -q tests/test_calibration_temporal.py
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m payment_guard.calibration_temporal
```

The last command writes `models/calibration_temporal_v1`. It never replaces an existing directory; an interrupted or failed run should be inspected rather than silently overwritten. **Do not rerun the full command unless the tests pass and the protocol has been reviewed.**
