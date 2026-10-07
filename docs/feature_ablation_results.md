# What I learned from removing type and category

## Completed run and evidence

On 7 October 2026 (Asia/Kolkata), I pulled commit
`055bc242fd0935a6384c7e6acdf7aed475121fda` and ran the expanded suite on my
Mac in `payment-guard-verify`: **72 passed in 2.75 seconds**. This is the
test-suite time, not a model-training benchmark.

I then ran:

```bash
PYTHONPATH=src python -m payment_guard.train_logistic_baseline --feature-ablation
```

The command fitted eight logistic models using 668,820 training rows and
167,990 February validation rows. Test rows were excluded. It saved the
completed run to `models/logistic_ablation_v1`, separately from my original
baseline. I supplied all 30 rows of printed metrics and a manifest excerpt.
The manifest reported complete=true and a clean working tree at the commit
above. All eight fits reported convergence; all retained categorical fields
had zero unknown validation categories.

The reported versions were NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1,
scikit-learn 1.9.1, PyArrow 24.0.0 and joblib 1.6.0. Input/artifact hashes,
individual predictions, per-model timings and the complete saved manifests
have not been independently inspected. This account uses the supplied
terminal evidence, not a second full-data run.

## The main finding

At the predeclared 1% budget (1,680 reviews), removing category reduced
caught positives from 188 to 131 in the raw model and from 196 to 130 in
the log model. Removing type produced smaller declines, to 181 and 193.
Removing both reduced these counts to 111 and 113, compared with 109 for
amount ranking.

For the log model, removing category changed recall from 97.03% to 64.36%:
a decline of **32.67 percentage points**, or 66 fewer caught positives.
Removing type reduced recall by 1.49 percentage points, or three positives.
Removing both reduced recall by 41.09 percentage points, or 83 positives.
At a fixed review count, every lost true positive adds one false alarm.

This shows much greater dependence on category than on type in these
fixed-setting fits and this validation month. It does not prove category
is leakage, that it is causally important, or that the remaining fields
are independent of it. Small marginal effects can arise when fields carry
overlapping information. The earlier audit remains the reason for caution
about synthetic patterns.

## Every declared workload

There are 202 positive labels in the same validation period for all rows.
Precision is caught/reviews; recall is caught/202. Missed is 202-caught.
Percentages are rounded here, not before evaluation.

| Approach | Reviews | Caught | False alarms | Missed | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant_train_rate | 168 | 0 | 168 | 202 | 0.00% | 0.00% |
| constant_train_rate | 840 | 2 | 838 | 200 | 0.24% | 0.99% |
| constant_train_rate | 1,680 | 4 | 1,676 | 198 | 0.24% | 1.98% |
| amount_ranking | 168 | 68 | 100 | 134 | 40.48% | 33.66% |
| amount_ranking | 840 | 99 | 741 | 103 | 11.79% | 49.01% |
| amount_ranking | 1,680 | 109 | 1,571 | 93 | 6.49% | 53.96% |
| logistic_raw | 168 | 157 | 11 | 45 | 93.45% | 77.72% |
| logistic_raw | 840 | 185 | 655 | 17 | 22.02% | 91.58% |
| logistic_raw | 1,680 | 188 | 1,492 | 14 | 11.19% | 93.07% |
| logistic_log | 168 | 166 | 2 | 36 | 98.81% | 82.18% |
| logistic_log | 840 | 194 | 646 | 8 | 23.10% | 96.04% |
| logistic_log | 1,680 | 196 | 1,484 | 6 | 11.67% | 97.03% |
| logistic_raw_without_type | 168 | 145 | 23 | 57 | 86.31% | 71.78% |
| logistic_raw_without_type | 840 | 178 | 662 | 24 | 21.19% | 88.12% |
| logistic_raw_without_type | 1,680 | 181 | 1,499 | 21 | 10.77% | 89.60% |
| logistic_log_without_type | 168 | 164 | 4 | 38 | 97.62% | 81.19% |
| logistic_log_without_type | 840 | 193 | 647 | 9 | 22.98% | 95.54% |
| logistic_log_without_type | 1,680 | 193 | 1,487 | 9 | 11.49% | 95.54% |
| logistic_raw_without_category | 168 | 93 | 75 | 109 | 55.36% | 46.04% |
| logistic_raw_without_category | 840 | 120 | 720 | 82 | 14.29% | 59.41% |
| logistic_raw_without_category | 1,680 | 131 | 1,549 | 71 | 7.80% | 64.85% |
| logistic_log_without_category | 168 | 91 | 77 | 111 | 54.17% | 45.05% |
| logistic_log_without_category | 840 | 119 | 721 | 83 | 14.17% | 58.91% |
| logistic_log_without_category | 1,680 | 130 | 1,550 | 72 | 7.74% | 64.36% |
| logistic_raw_without_both | 168 | 72 | 96 | 130 | 42.86% | 35.64% |
| logistic_raw_without_both | 840 | 104 | 736 | 98 | 12.38% | 51.49% |
| logistic_raw_without_both | 1,680 | 111 | 1,569 | 91 | 6.61% | 54.95% |
| logistic_log_without_both | 168 | 72 | 96 | 130 | 42.86% | 35.64% |
| logistic_log_without_both | 840 | 100 | 740 | 102 | 11.90% | 49.50% |
| logistic_log_without_both | 1,680 | 113 | 1,567 | 89 | 6.73% | 55.94% |

The two full-feature controls and the two references match the previously
supplied aggregate metrics at all three budgets, including ranking and
probability losses. This supports consistency of the comparison; matching
aggregates do not establish that every individual prediction is identical.

Removing category reduced caught counts at all three budgets for both amount
versions. Without category, raw amount caught more than log amount at every
budget: 93 versus 91, 120 versus 119, and 131 versus 130. I therefore cannot
say that log amount always improved the model.

Without both fields, raw and log tied at 168 reviews (72 caught). Raw caught
more at 840 reviews (104 versus 100), while log caught more at the primary
1,680 budget (113 versus 111). These are metric/workload-specific observations,
not evidence of a universally better transformation.

For log without type, increasing reviews from 840 to 1,680 caught no additional
positive labels: both counts were 193. The extra 840 flags were all non-fraud
according to the validation labels. This does not establish an optimal budget
without valuing review effort and missed fraud.

## Ranking, probability losses and fitting

Higher average precision is better; lower log loss and Brier score are better.
These values repeat across each approach's three metric rows because its score
vector is unchanged. They are not three independent runs. Amount ranking has
no probability losses because transaction amounts are not probabilities.

| Approach | Average precision | Log loss | Brier score | Iterations |
| --- | ---: | ---: | ---: | ---: |
| constant_train_rate | 0.00120245 | 0.00929984 | 0.00120104 | N/A |
| amount_ranking | 0.32553202 | N/A | N/A | N/A |
| logistic_raw | 0.88380035 | 0.00210094 | 0.00036204 | 90 |
| logistic_log | 0.94677814 | 0.00105914 | 0.00016918 | 59 |
| logistic_raw_without_type | 0.82474795 | 0.00302059 | 0.00046742 | 80 |
| logistic_log_without_type | 0.92072907 | 0.00140914 | 0.00020932 | 62 |
| logistic_raw_without_category | 0.47571979 | 0.00454262 | 0.00079658 | 95 |
| logistic_log_without_category | 0.47324143 | 0.00493258 | 0.00089221 | 79 |
| logistic_raw_without_both | 0.35104532 | 0.00594396 | 0.00093821 | 54 |
| logistic_log_without_both | 0.34958421 | 0.00640949 | 0.00097770 | 44 |

The full log model had the highest average precision and lowest probability
losses among these approaches on this period. Without category, raw had better
average precision and both probability losses than log. Without both fields,
raw also had better average precision and both losses, despite catching two
fewer positives at the primary budget. A fixed-budget result and a whole-ranking
metric answer different questions.

Convergence means the solver satisfied its stopping conditions; it does not
mean that the feature set is appropriate, probabilities are calibrated or
the model will generalize. Zero unknown categories establishes coverage of
the observed category values, not stability of their frequencies or label rates.

All amount-ranking and logistic cutoffs had a single observation at the cutoff,
with no crossing tie. The constant reference tied all 167,990 rows at every
budget and used the shared seeded, label-independent tie ordering.

## Aggregate evidence and scope

The [aggregate snapshot](../results/tables/logistic_ablation_summary.csv)
contains all ten approaches and three workloads. Caught counts and probability
metrics were transcribed from the supplied output. False alarms, missed labels,
true negatives, precision and recall were reconstructed from those counts and
the fixed validation totals, and reconcile with the printed metrics. It is a
curated summary, not a copy of the original metrics file: cutoff details and
review-fraction fields remain in the original local CSV.

This experiment refitted every model with the same C=1.0 and other declared
settings. It did not tune a separate regularization strength for each feature
set. The result is predictive dependence under this protocol, not a causal
estimate, proof of leakage, or a statistical significance claim.

I have now completed the planned eight-fit comparison. Before treating a model
as suitable for use, I still need to examine errors and category patterns,
then separately address calibration, decision costs and final holdout
evaluation. No test-period model evaluation, real-payment validation or live
approval/review/block policy was performed. Earlier exploratory auditing
included the final period, so it is not an entirely unseen dataset.

The [experiment plan](baseline_experiment_plan.md) and
[run instructions](running_feature_ablation.md) explain the design. The original
baseline account remains a record of the earlier checkpoint.

## Follow-up training SQL investigation

The [category analysis](training_category_analysis.md) records the completed
category and monthly summaries. Its counts reconcile to 668,820 training rows
and 931 positives. Six categories had no observed fraud in any training month;
Other contained 801 positives. These descriptive patterns help interpret the
ablation, but do not prove leakage or establish real-world risk rules.
