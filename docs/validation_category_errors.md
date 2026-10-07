# What I learned from the validation errors

## Scope

I examined saved predictions from logistic_ablation_v1 at the
predeclared 1,680-review budget. These are February validation
results on synthetic data, not final test results.

I joined predictions to prepared transactions using source_row,
checked one-to-one matching and agreement of labels, and reproduced
the score ordering using the saved tie_priority.

## Where the errors occurred

| Category | Full model caught | Without category caught | Full model false alarms | Without category false alarms |
| --- | ---: | ---: | ---: | ---: |
| Other | 171 | 109 | 1346 | 0 |
| Recreation | 16 | 12 | 53 | 0 |
| Shell Company | 9 | 9 | 59 | 31 |
| Housing | 0 | 0 | 22 | 1519 |
| Education | 0 | 0 | 4 | 0 |

Food, Healthcare, Transport and Utilities had no reviewed
transactions or positive labels in either breakdown.

The full log model caught 196 positives and missed six.
Without category, it caught 130 and missed 72.
Both reviewed 1,680 transactions.

The net decline of 66 caught positives came from Other (62)
and Recreation (4). Equal caught counts do not establish that
the models caught the same individual transactions.

Without category, Housing consumed 1,519 reviews, about 90.4%
of the budget. All were false alarms according to the labels.

## Housing transfers

All 1,519 selected Housing transactions were TRANSFERs.
Another 6,854 Housing transfers were not selected.

| Housing transfers | Selected | Not selected |
| --- | ---: | ---: |
| Transactions | 1519 | 6854 |
| Median amount | 3616.64 | 1937.72 |
| Median origin balance | 237152.49 | 298499.09 |

The model selected about 18.1% of Housing transfers. Selected
transfers had higher median amounts and lower median origin
balances than unselected Housing transfers.

These summaries describe associations. They do not establish
which features caused the high scores. Other retained features
also contribute, and removing category refitted the whole model.

## Evidence and remaining work

These figures came from terminal diagnostics run on my Mac on
7 October 2026. The checks reproduced the previously reported
confusion counts. Artifact hashes were not independently checked.

The next explanatory step is to inspect fitted coefficients
and contributions to individual scores. I have not completed
that analysis or introduced a Housing-based decision rule.
