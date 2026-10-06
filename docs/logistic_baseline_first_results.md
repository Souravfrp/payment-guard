# My first full-data logistic baseline results

## Run evidence

On 6 October 2026 (Asia/Kolkata), I ran the code at commit `28f1de7`
in my `payment-guard-verify` environment on the Mac. The expanded tests
reported **61 passed in 68.18 seconds**. That is the test-suite duration,
not the model training time.

I then ran:

```bash
PYTHONPATH=src python -m payment_guard.train_logistic_baseline
```

The terminal confirmed 668,820 training rows and 167,990 validation rows,
with test rows excluded. Both full-feature fits completed and the command
reported the saved run at `models/logistic_baseline_v1`.

This account is based on the terminal output I supplied. The saved metrics,
run manifest and row-level predictions have not yet been independently
inspected here. In particular, exact iteration counts, package versions,
fit timings, cutoff ties and probability losses are not inferred from
the completion message.

## What happened at the planned review budget

The primary comparison selected 1,680 transactions from the February
validation period for each approach. There are 202 positive labels in
that period. Missed fraud below is calculated as 202 minus caught fraud.

| Approach | Selected | Caught fraud | False alarms | Missed fraud | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant training rate | 1,680 | 4 | 1,676 | 198 | 0.24% | 1.98% |
| Descending amount | 1,680 | 109 | 1,571 | 93 | 6.49% | 53.96% |
| Logistic, raw amount | 1,680 | 188 | 1,492 | 14 | 11.19% | 93.07% |
| Logistic, log amount | 1,680 | 196 | 1,484 | 6 | 11.67% | 97.03% |

Both fitted models caught more fraud than amount ranking at the same
workload. The log-amount version caught eight more fraud cases in total
than the raw version, and 87 more than amount ranking. I have not compared
the individual flagged sets yet, so this does not establish that one set
of detected fraud cases contains the other.

The constant score assigns the same probability to every row. Its four
caught cases came from the fixed random ordering used for score ties.
This is one reproducible no-information reference, not an estimate of the
expected number caught by random selection.

## What precision and recall mean here

The log-amount model caught 196 of the 202 positive labels, giving about
97.03% recall. But only 196 of its 1,680 flags were fraud-labelled, giving
11.67% precision. Most of its flags, 1,484, were non-fraud.

So I cannot describe this result as 97% accuracy or say that 97% of flags
were correct. The useful result is high fraud coverage at a specified
review workload, with a substantial false-alarm burden still present.

Compared with the raw model, the increase in recall is about 3.96 percentage
points. This is an observed difference on one validation month, not a
statistical significance claim or proof that the log model is universally
better.

## What I still need to examine

I will inspect the saved metrics at the other two declared review budgets,
average precision, log loss, Brier score, convergence records and cutoff
ties. The full run manifest also records the input hash and dependency
versions needed to identify this experiment.

Next I will run the planned type/category removal comparisons. The earlier
SQL exploration showed unusually strong label patterns in those synthetic
fields, so the current scores are not enough to establish robustness.

No test-period evaluation, probability recalibration, cost optimization or
live approve/review/block policy has been completed. The 1% budget is a
retrospective batch-ranking experiment, not a measured business capacity.
The synthetic-data limitation and the prior exploratory auditing of the
test period remain unchanged.

The [experiment plan](baseline_experiment_plan.md) was written before these
results. The [run instructions](running_logistic_baseline.md) describe the
implementation and local artifacts. The fitted models and row-level data
remain outside Git.
