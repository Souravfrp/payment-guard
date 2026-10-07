# My first logistic-regression experiment

Status: the first two full-feature fits completed on 6 October 2026.
The six feature-removal fits are implemented; their full-data run remains pending.
See the [ablation run instructions](running_feature_ablation.md). The plan below was written
before the first run; [results are recorded separately](logistic_baseline_first_results.md).

## What I want to find out

My SQL analysis showed that fraud frequency varies with amount and type.
I now want to check whether a fitted model improves on a simple amount
ranking in the later validation period. I will compare raw amount with
log-transformed amount, then check how much the results depend on type
and category. I am writing these choices down before seeing model results.

I am starting with logistic regression because I can explain its weighted
score, loss function and regularization. This is a starting point, not a
claim that it is the best model. A tree comparison will follow if there
is a useful question about nonlinear effects or interactions.

## Data and feature boundaries

I will read the verified prepared Parquet table. Training ends before
1 February 2026; validation covers February 2026. I will exclude test
rows from this experiment. The test period was included in earlier
exploratory auditing, so it remains a chronological holdout rather than
an entirely unseen dataset.

The target is `isFraud`. I will use an explicit feature list based on the
[existing feature policy](../results/tables/amlnet_v2_initial_model_feature_policy.csv):

- Amount and origin balance: `amount`, `oldbalanceOrg`.
- Calendar: `hour`, `day_of_week`, `day_of_month`, `month`.
- Payment information: `type`, `category`, `metadata.payment_method`.
- Device: `metadata.device_info.os`, `metadata.device_info.type`.
- Location: `metadata.location.city`, `metadata.location.state`.

The four device/location fields retain the documented assumption that they
are available before authorization. Inclusion in this experiment does not
prove that assumption.

Source-row references, split labels and raw timestamps are not predictors.
Neither are duplicate labels, post-transaction balances, supplied risk scores
or target-revealing metadata. I will not add historical aggregates in this
first experiment.

## Preprocessing choices

For amount I will compare raw values against the natural logarithm
`log1p(amount)`. Origin balance stays raw in both versions so that this
comparison changes only the amount transformation.

I will standardize both monetary features using training means and standard
deviations, after any transformation. Those same fitted transformations
will be applied to validation. A constant training feature will be handled
without division by zero.

I will use fixed sine/cosine pairs for hour, weekday and month, with periods
24, 7 and 12. These preserve wraparound and avoid treating a new month in
validation as an unknown category. Day of month will be standardized as a
numerical feature; this simple representation does not model varying month
lengths. Calendar treatment will stay identical across variants.

I will one-hot encode the remaining categorical fields using training
categories only. Unseen validation categories will receive no active
indicator in that field's block, and their counts will be reported.
I will fail on unexpected missing values rather than silently invent an
imputation policy for this already validated dataset.

## Comparisons

First I will run the full-feature raw and log amount versions. Then I will
repeat both versions with type removed, category removed, and both removed:
eight logistic fits altogether. Separate removals help distinguish dependence
on one field from dependence on the pair.

I will also report two simple references:

- A constant probability equal to the training fraud fraction, for probability
  losses and a no-information reference.
- Ranking by descending amount, for comparison at the same review workload.

A strictly increasing log transform preserves the amount-only ordering.
It can change a fitted multifeature model, but it cannot improve an
amount-only ranking merely by changing its scale.

The initial logistic fits will use the same L2 regularization setting
(`C=1.0` in the planned implementation), an intercept and no class weighting
or resampling. This keeps the initial comparison small. It does not mean
that this regularization strength is optimal. Solver settings, dependency
versions and convergence status will be recorded when code is implemented.
I will resolve convergence failures before interpreting results and record
any settings changed for that purpose.

## Evaluation rules

My primary comparison will be recall at a fixed 1% validation review fraction.
This is an experimental workload, not a known business capacity. I will also
report 0.1% and 0.5% as sensitivity checks, without choosing whichever
budget makes a model look best.

For each fraction q, I will select the top ceil(q * N) validation scores.
With 167,990 validation transactions, these budgets are 168, 840 and 1,680.
I will report the exact selected counts, true positives, false positives,
false negatives, precision and recall. Score ties will be broken by a fixed
seeded random order independent of labels, shared across models. I will
report ties crossing the cutoff; they can make a top-k result sensitive
to tie handling.

This is retrospective ranking of a whole validation batch. It is not a
deployable real-time threshold: a live system cannot see all future scores.
Thresholds and daily review capacity require a later decision-policy study.

I will also report average precision (without treating it as identical to
trapezoidal PR area), log loss and Brier score. Probability losses provide
a first check, not a complete calibration assessment. A reliability plot
and any fitted recalibration will belong to a later, separately designed step.

On a fixed dataset and at fixed k, precision and recall are both determined
by the number of caught fraud cases: precision = TP/k and recall = TP/P,
where P is the total positive count. They therefore rank models identically
at that workload, although they communicate different practical questions.

I will report all eight fits. If two fits catch the same number of fraud
cases at the primary budget, I will report a tie on the primary criterion.
Secondary metrics can describe differences but will not establish a
statistically significant winner. Validation comparisons are exploratory
model selection; they are not an unbiased final performance estimate.

## What I will save and check

I will save aggregate validation metrics and a run record containing the
input fingerprint, Git revision, feature lists, transformations, settings,
package versions, timings, seed and convergence information. Local prediction
files will retain source references for later SQL joins, but row-level data
and model artifacts will stay outside Git under the project's data policy.

Tests will check that preprocessing fits only on training rows, forbidden
columns never enter predictors, unknown categories are handled, fixed-budget
counts and ties are reproducible, and metrics agree with hand-calculated
small examples. I will inspect the first run before interpreting a table
of scores.

No outcome is assumed. If log amount or additional features fail to improve
the baseline, I will record that result. Claims about actual fraud losses,
calibration, production use or an approve/review/block policy remain outside
this first experiment.


## Implementation checkpoint

The first two full-feature fits and both reference scores are implemented in
`src/payment_guard/train_logistic_baseline.py`. The [run instructions](running_logistic_baseline.md)
record the solver settings, local artifacts and small-fixture test evidence.
The first full-data run completed on the Mac. The `--feature-ablation` option
now adds the six planned removal variants and refits both full-feature controls
in a separate run folder. The full-data ablation outcomes remain pending.
