# Running my first two logistic baselines

I have implemented the first stage of my [experiment plan](baseline_experiment_plan.md):
full-feature logistic regression with raw amount, and the same model with
log-transformed amount. The type/category removal comparisons remain pending.
The first full-data run completed on my Mac on 6 October 2026. My
[first results](logistic_baseline_first_results.md) record the terminal evidence
and initial interpretation; inspection of the saved run artifacts is pending.

## What the command does

The command reads training and validation rows from my verified prepared
Parquet file. It checks the audited counts, chronological boundaries,
source references, calendar fields, labels and required values. It does not
score test rows. The input file hash identifies the whole prepared file;
hashing its bytes is separate from using test observations in modelling.

Both models use the same 13 permitted source features. Calendar features
become fixed sine/cosine pairs for hour, weekday and month, with numerical
day of month. Monetary features and day of month are standardized using
training statistics. The categorical encoder learns only training categories.
Unseen validation categories get all-zero indicators for that field, and
I record how often this happens.

The model settings are L2 regularization, C=1.0, an intercept, no class
weighting, the lbfgs solver, tolerance 1e-6 and a maximum of 2,000 iterations.
The stricter tolerance is intended to avoid comparing very loosely fitted
models. Both full-data fits completed without a surfaced convergence error;
I still need to inspect their recorded iteration counts. The implementation
uses the library's default L2 penalty to work across the old and new penalty
interfaces. Installed dependency versions are saved with every completed run.

A convergence warning stops the experiment before it saves a completed run.
It does not quietly increase the iteration limit or change regularization.
If needed, an explicit --max-iter override is recorded in the run metadata.

## Commands on my Mac

From the repository root, with payment-guard-verify active:

```bash
PYTHONPATH=src python -m pytest -q
```

After the tests pass:

```bash
PYTHONPATH=src python -m payment_guard.train_logistic_baseline
```

I should leave Terminal running until it prints the validation comparison
and the completed output path. No GPU or database rebuild is required.
The first full Mac run completed; its exact training timings and iteration
counts have not yet been inspected from the saved manifest. The experiment loads training/validation frames and transformed
matrices into memory; it is not an out-of-core training procedure.

The default output is models/logistic_baseline_v1/, which is excluded from
Git. A run never overwrites an existing output directory. An intentional
rerun needs a new path, for example:

```bash
PYTHONPATH=src python -m payment_guard.train_logistic_baseline --output models/logistic_baseline_v2
```

I should inspect a failed run rather than delete its files automatically.
A directory is complete only when run.json can be read and says complete=true.
The manifest is written last; ordinary save failures or interruptions can
leave an incomplete directory. This is not a crash-durability guarantee.

## Reading the output

The terminal prints the 1% validation review-budget comparison. With 167,990
validation transactions, each approach selects exactly 1,680 rows.
The metrics CSV also includes budgets of 168 and 840 rows.

| Output | Purpose |
| --- | --- |
| validation_metrics.csv | Counts, precision, recall, average precision, probability losses and cutoff ties for all four approaches |
| validation_predictions.parquet | Validation source references, timestamps, labels, scores and shared tie priorities for later SQL analysis |
| logistic_raw.joblib / logistic_log.joblib | Fitted preprocessing and logistic models |
| run.json | Input and artifact hashes, Git state, feature names, coefficients, preprocessing statistics, categories, convergence, timings and versions |

The four approaches are the two learned models, a constant probability equal
to training prevalence, and descending transaction amount. Amount is only
a ranking score, so its log loss and Brier score are left empty. The constant
reference ties all scores; its top-k result depends on the seeded random
tie ordering and is not an expected random-performance estimate.

Exact score ties use the same seeded, label-independent permutation across
approaches, after sorting validation rows by source reference. I report how
many tied observations straddle a cutoff. A fixed seed makes the result
repeatable but does not remove uncertainty from those ties.

This top-k comparison ranks a whole validation batch retrospectively. It
does not implement a live approval, review or blocking threshold. The primary
metric is recall at 1% review, as written in the plan before seeing results.
I will show all comparisons, including negative results.

The saved pipelines expect the fixed feature construction performed by
`design_frame(..., log_amount=False/True)` in the training module. This
transformation is needed before predicting with a reloaded pipeline; passing
the original raw columns directly is not the intended interface.

Row-level predictions and model files stay local. Only reviewed aggregate
results and their interpretation should later be committed. Model outputs
are not proof of calibrated probabilities or production fraud performance.

## Verification performed before the full run

The expanded suite passed 61 tests on Linux/Python 3.12. The 14 new cases
cover exclusion of test rows and forbidden predictors, rejected invalid
inputs, training-only scaling and categories, constant numerical features,
unknown validation categories, hand-calculated evaluation metrics, cutoff
ties, convergence failure, and a real Parquet/model-file round trip through
the command entry point using small fixture counts. An existing run is
preserved when a repeat command tries to use its output path.

On 6 October 2026, my Mac terminal at commit `28f1de7` reported 61 tests
passed in 68.18 seconds, followed by successful completion of the full
baseline command. The [first results](logistic_baseline_first_results.md)
separate that full-data evidence from the small-fixture tests.

## Library references

I use scikit-learn's implementations rather than writing a custom optimizer.
The mathematical reasoning and comparison choices are in the experiment plan;
these references specify the library behavior used by the code:

- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

The stable documentation can change. The exact package versions used for
my experiment are recorded in run.json, so results can be interpreted against
the installed implementation.
