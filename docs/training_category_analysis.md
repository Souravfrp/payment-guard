# What category tells me in the training data

## Why I returned to SQL

The ablation experiment showed a large decline when I removed category.
Before trying to improve the score, I wanted to understand the pattern
behind that result. I grouped training transactions by category, then by
category and month. These are descriptive checks after viewing validation
results; they are not an independent confirmation or a new predeclared test.

I ran [the category query](../sql/explore_train_categories.sql) and
[the monthly query](../sql/explore_train_monthly_categories.sql) in DB Browser
for SQLite on 7 October 2026. Both explicitly use `WHERE split = 'train'`.
The queries were committed as `83ed8a9` and `a9caa85`, respectively.

## Category totals

| Category | Transactions | Fraud labels | Fraud percentage |
| --- | ---: | ---: | ---: |
| Cryptocurrency | 1 | 1 | 100.000% |
| Education | 26,724 | 0 | 0.000% |
| Food | 113,606 | 0 | 0.000% |
| Healthcare | 40,003 | 0 | 0.000% |
| Housing | 132,866 | 0 | 0.000% |
| Other | 147,051 | 801 | 0.545% |
| Property Investment | 13 | 13 | 100.000% |
| Recreation | 87,295 | 49 | 0.056% |
| Shell Company | 739 | 67 | 9.066% |
| Transport | 100,455 | 0 | 0.000% |
| Utilities | 20,067 | 0 | 0.000% |

The counts reconcile to **668,820 training transactions and 931 positive
labels**. The six zero-positive categories together contain 433,721
transactions, about 64.85% of training rows. In contrast, Other contains
801 of the 931 positive labels, about 86.04%.

These are different conditional proportions:

- Fraud rate within Other = 801 / 147,051 = about 0.545%.
- Share of fraud labels in Other = 801 / 931 = about 86.04%.

The first asks how often an Other transaction is fraud-labelled. The second
asks where the fraud labels occur. Most Other transactions are still
non-fraud. A large share of positive labels does not imply high precision
for flagging every transaction in that category.

Property Investment and Cryptocurrency are 100% fraud-labelled in training,
but together have only 14 transactions. They account for only about 1.50%
of training positives. These tiny groups cannot by themselves explain
coverage of most fraud labels. The category pattern also includes separation
among much larger groups.

## Across training months

The table reports within-category fraud percentages. October is partial.

| Category | October 2025 | November 2025 | December 2025 | January 2026 |
| --- | ---: | ---: | ---: | ---: |
| Other | 0.666% | 0.529% | 0.406% | 0.622% |
| Recreation | 0.171% | 0.018% | 0.040% | 0.033% |
| Shell Company | 13.768% | 9.836% | 6.452% | 7.960% |

Education, Food, Healthcare, Housing, Transport and Utilities had no observed
positive labels in any of the four training months. Property Investment
had 4, 4, 4 and 1 transactions, respectively, all positive. Cryptocurrency
had one transaction, in January, also positive. There are no Cryptocurrency
rows in the other training months: an absent group is not a measured 0%
fraud rate.

Other contained the majority of positive labels each month. Shell Company
had a higher within-category rate, but only 138, 183, 217 and 201 transactions.
Rates change across months; the table alone does not establish statistical
significance or the cause of those changes. Partial October also limits
comparisons of raw monthly counts and may affect the composition of transactions.

| Month | Transactions | Positive labels |
| --- | ---: | ---: |
| 2025-10 (partial) | 120,879 | 226 |
| 2025-11 | 171,789 | 226 |
| 2025-12 | 190,055 | 198 |
| 2026-01 | 186,097 | 281 |
| Total | 668,820 | 931 |

## What the SQL calculates

`WHERE` chooses training rows before aggregation. `GROUP BY category`
forms one group per observed category. Adding
`SUBSTR(timestamp, 1, 7)` creates month-category groups from the stored
ISO-format timestamps. `COUNT(*)` counts all rows in each group.
Because the target was validated as nonmissing binary 0/1, `SUM(isFraud)`
counts positives. Multiplication by `100.0` gives a percentage using
floating-point division; `ROUND(..., 3)` controls display precision.

For group g, let n_g be its transaction count and f_g its positive-label count:

```text
n_g = number of rows in group g
f_g = sum of isFraud over rows in group g
empirical fraud percentage = 100 * f_g / n_g
overall training fraud percentage = 100 * sum(f_g) / sum(n_g)
```

The overall percentage is weighted by transaction counts; it is not the
unweighted average of the displayed category percentages. `ORDER BY`
sorts the output for reading and changes none of these counts. The category
query orders by rounded percentage, then group size; equal displayed
percentages need not imply equal unrounded rates.

## Connection to the ablation

At 1,680 validation reviews, the log model caught 196 positives with all
features, 193 without type, 130 without category and 113 without both.
The training summaries offer a plausible explanation: category separates
large groups with very different observed label rates. A one-hot encoded
logistic model can learn different contributions for those groups.

This is consistent with category dependence; it does not identify the
individual validation errors or establish causality. The model also uses
other inputs, correlated fields can substitute for each other, and all
variants kept C=1.0 rather than being individually retuned.

I cannot infer that these category names are reliable risk rules outside
AMLNet. Zero observed positives do not mean zero future risk. Nor does
100% in a tiny synthetic group establish a universal rule. Whether a field
constitutes leakage depends on how it was generated and when its value
would be available, not only on its predictive strength.

## Evidence and next step

The [monthly aggregate snapshot](../results/tables/training_monthly_categories.csv)
was transcribed from the supplied SQL output. The 41 observed groups reconcile
with the category screenshot and the audited training totals. Percentages in
the snapshot were recalculated from supplied integer counts to three decimals.
This is an arithmetic reconciliation of supplied results, not a fresh
independent database query. Missing month-category combinations were not
invented or filled with zeros.

The [ablation results](feature_ablation_results.md) remain the source for model
performance. A useful next diagnostic is to join saved validation predictions
to category by source reference and inspect false alarms and missed positives
at the already declared budget. That diagnostic has not been run yet and
would remain validation exploration, not final test evaluation.
