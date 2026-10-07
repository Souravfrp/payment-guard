# Checking how much my model depends on type and category

Status: the full-data ablation run completed on the Mac on 7 October 2026.
The [results account](feature_ablation_results.md) records all variants and
budgets, convergence evidence and limitations. The original baseline artifacts
are unchanged.

## Why I am doing this

My training SQL exploration found unusually strong label patterns in type and
category. For example, all 42 training PAYMENT transactions were fraud-labelled.
I want to measure how much my model relies on these fields before interpreting
its high validation recall as a robust result.

An ablation means removing a feature and fitting the model again. I do not
just set its fitted coefficient to zero: the remaining coefficients need the
opportunity to adapt. This is a predictive comparison, not a causal experiment.
A performance drop would demonstrate dependence in this dataset, not prove
that the removed field leaked the target. Little change would not prove that
the field was irrelevant; other fields might carry related information.

## The controlled comparison

| Feature group | Raw amount | Log amount | Source predictors |
| --- | --- | --- | ---: |
| Full | logistic_raw | logistic_log | 13 |
| Removed type | logistic_raw_without_type | logistic_log_without_type | 12 |
| Removed category | logistic_raw_without_category | logistic_log_without_category | 12 |
| Removed both | logistic_raw_without_both | logistic_log_without_both | 11 |

There are six new removal variants. The command also refits the original two
full-feature models as controls, so all eight fits use the same input file,
installed packages and experiment settings. It does not load, modify or replace
the old baseline artifacts. Constant training prevalence and amount ranking
are recalculated as references, giving ten approaches and 30 metric rows.

Only the specified categorical fields are removed. The input is still the
complete verified prepared table, and its required columns are checked.
Excluded fields do not enter feature construction, encoding or model fitting.
Amount transformation, balance, calendar features, scaling, solver, C=1.0,
tolerance, class weighting and the chronological boundaries remain unchanged.
This measures removals at a fixed regularization setting, not the best possible
retuned model for each feature set. All preprocessing is fitted on training
data. Test rows are neither fitted nor scored.

## Run on my Mac

From the repository root with payment-guard-verify active:

```bash
git pull --ff-only
PYTHONPATH=src python -m pytest -q
```

After the tests pass, run:

```bash
PYTHONPATH=src python -m payment_guard.train_logistic_baseline --feature-ablation
```

The command prints eight fitting messages, the primary comparison and the
completed output path. It can take longer than the first two-model run.
No GPU or SQLite rebuild is required. I should keep Terminal running until
the command finishes or reports an error.

The default output is `models/logistic_ablation_v1/`. The original
`models/logistic_baseline_v1/` is preserved. Both are excluded from Git.
If the ablation output already exists, the command stops before fitting.
For an intentional rerun, supply a new `--output` directory; first inspect
why the earlier run needs repeating. A convergence warning aborts the run
without marking it complete. Interrupted writes can leave an incomplete folder;
`run.json` is written last with `complete=true` only on successful completion.

## Read the results without moving the goalposts

The primary budget remains 1% of 167,990 validation rows: 1,680 reviews.
The other declared budgets remain 168 and 840. Exact ties use the same seeded,
label-independent priority order for every approach. Compare a raw removal
with the raw full model, and a log removal with the log full model.

For a fixed review count k and positive-label count P=202:

```text
precision = TP / k
recall = TP / P
delta recall = recall_removed - recall_full
delta TP = TP_removed - TP_full
delta false alarms = -delta TP   (because k is unchanged)
```

A negative delta recall is a decline. Multiply the recall difference by 100
to express percentage points; do not call it a percentage relative change.
At this validation size, one additional caught positive changes recall by
100/202, or about 0.495 percentage points. Equal caught counts are a tie on
the primary criterion, even if average precision or probability losses differ.
Equal counts also need not mean the same individual transactions were caught.

I will show all variants and all budgets, with false alarms, precision and
recall. Average precision, log loss and Brier score provide further descriptions.
They do not turn a single validation month into independent repeat experiments
or prove calibration. I have not predeclared a statistical significance test.

First compare the full controls with the original run (raw: 188; log: 196
caught at 1,680 reviews). These are historical observations, not hard-coded
pass conditions. If they differ, inspect input hashes, packages, convergence
and settings before attributing any change to feature removal.

## Saved evidence and checks

The output contains `validation_metrics.csv`, aligned
`validation_predictions.parquet`, eight fitted joblib pipelines and `run.json`.
The manifest records included and excluded source features for each model,
encoded names, training categories, scaling statistics, coefficients,
unknown-category counts, convergence, iterations, timings, versions, Git state
and input/artifact hashes. The top-level feature list describes the complete
input candidates; each model's `source_features` is its actual retained set.

Reloaded pipelines require the same fixed construction:

```python
frame = design_frame(validation, log_amount=record['log_amount'],
                     excluded=record['excluded_features'])
scores = fitted.predict_proba(frame)[:, 1]
```

The added tests check that changing removed fields cannot affect the fitted
model, prediction works without those fields, full controls match the original
experiment, ties are shared, saved pipelines reproduce scores, hashes match,
test rows are excluded and existing run folders are preserved. These are
small-fixture software tests; full-data outcomes must come from the Mac run.

Before publication on the feature branch, the expanded suite passed all 72
tests on Linux (61 existing cases and 11 added cases). On 7 October 2026,
the Mac suite passed all 72 cases in 2.75 seconds, followed by completion of
the full-data ablation run. The test time is not a training benchmark.

The existing four-model plotting script is specific to the first baseline
checkpoint. Do not pass the 30-row ablation file to it. A separate comparison
figure will follow inspection of the full-data ablation results.

The data are synthetic and the final period was previously audited. Calibration,
cost optimization, live decisions and final holdout evaluation remain separate
future steps. See the [original experiment plan](baseline_experiment_plan.md).
