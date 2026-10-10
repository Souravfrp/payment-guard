# What I learned from the first SQL exploration

## Why I started with these questions

I had not used SQL as a regular part of a project before PaymentGuard. After
checking that the SQLite table matched my prepared data, I used DB Browser
to ask a few small questions before moving to a model: do fraud-labelled
transactions have different amounts, does the transaction type matter, and
does the amount pattern remain when I look at transfers alone?

I ran these queries on the training period only. There were 668,820
transactions, including 931 fraud labels. These results describe the supplied
labels in synthetic AMLNet data. They are not measurements of fraud in a
payment company.

## Amount and transaction type

My first comparison gave an average amount of 636.73 for non-fraud
transactions and 10,105.27 for fraud-labelled transactions. The difference
was large, but an average can be pulled upward by a few large observations.
It did not tell me how the labels were distributed across amounts.

I then divided the amounts into four ranges:

| Amount range | Transactions | Fraud labels | Fraud percentage |
| --- | ---: | ---: | ---: |
| Below 100 | 141,881 | 17 | 0.012% |
| 100 to below 1,000 | 396,781 | 126 | 0.032% |
| 1,000 to below 10,000 | 129,804 | 542 | 0.418% |
| 10,000 and above | 354 | 246 | 69.492% |

These were convenient ranges for exploration, not thresholds selected by
an optimization procedure. The highest range had the highest fraud rate,
but most of the fraud labels occurred below 10,000.

Next I grouped by transaction type. All 42 PAYMENT transactions were
fraud-labelled. TRANSFER contained 167,322 transactions and 889 fraud
labels, a rate of 0.531%. The remaining six types contained 501,456
transactions and no fraud labels.

I cannot turn these observations into statements that every future PAYMENT
will be fraud or that the other types are safe. The pattern may reflect how
the synthetic data were constructed. It gives me a reason to compare models
with and without the type field, as already planned in the feature policy.

## Looking within transfers

The amount comparison mixed different transaction types. I repeated it
with both `split = 'train'` and `type = 'TRANSFER'`:

| Transfer amount range | Transactions | Fraud labels | Fraud percentage |
| --- | ---: | ---: | ---: |
| Below 100 | 35,169 | 17 | 0.048% |
| 100 to below 1,000 | 99,075 | 126 | 0.127% |
| 1,000 to below 10,000 | 32,846 | 541 | 1.647% |
| 10,000 and above | 232 | 205 | 88.362% |

The rate still increased across these ranges. This means that the overall
amount pattern was not explained entirely by mixing transaction types.
It does not show that increasing an amount causes fraud, or that amount
will remain equally useful in a later period.

## Checking the training months

I also grouped the training transactions by month using
`SUBSTR(timestamp, 1, 7)`. I wanted to check whether the overall training
fraud rate hid differences between months.

| Month | Transactions | Fraud labels | Fraud percentage |
| --- | ---: | ---: | ---: |
| October 2025 (partial month) | 120,879 | 226 | 0.187% |
| November 2025 | 171,789 | 226 | 0.132% |
| December 2025 | 190,055 | 198 | 0.104% |
| January 2026 | 186,097 | 281 | 0.151% |

The observed rate fell through December and rose in January. October and
November each had 226 fraud labels, but November had more transactions,
so its rate was lower. This was another example of why the denominator
matters. October covers only part of a month, so I cannot compare its
transaction count with a full month as though the observation periods
were equal.

The four rows add up to 668,820 transactions and 931 fraud labels, matching
the training totals. These results support evaluating performance over
time. They do not establish a statistically significant change, identify
its cause, or prove that a model will deteriorate. I have not fitted a
model or performed a significance test for these monthly differences.

## The probability calculation behind the SQL

Let $Y_i=1$ denote a fraud label and $Y_i=0$ a non-fraud label.
For a group $B$, such as transfers of at least 10,000, the observed fraction is

$$
\widehat p_B =
\frac{\sum_{i\in B}Y_i}{|B|}.
$$

In the query, `SUM(isFraud)` gives the numerator and `COUNT(*)` gives the
denominator. Multiplying by 100 expresses the fraction as a percentage.
For large transfers, this is $205/232\approx 0.88362$.

This is an empirical conditional proportion: among transactions satisfying
the group conditions, what fraction had a fraud label? It is not yet a
calibrated probability for a new transaction.

The group size matters too. A 100% observed rate from 42 transactions is not
a proof of a population probability of one. I have not attached confidence
intervals or significance claims here. Such claims would require assumptions
about sampling and dependence; transactions ordered in time should not
automatically be treated as independent, identically distributed trials.

## Why the denominator changes the conclusion

Suppose I flag every training transaction with an amount of at least 10,000.
From the first table, I would flag 354 transactions: 246 fraud-labelled and
108 non-fraud. I would miss the remaining 685 fraud labels.

$$
\text{Precision}=\frac{TP}{TP+FP}=\frac{246}{354}\approx69.49\%,
\qquad
\text{Recall}=\frac{TP}{TP+FN}=\frac{246}{931}\approx26.42\%.
$$

Here TP counts flagged fraud, FP counts flagged non-fraud, and FN counts
unflagged fraud. Precision asks how often a flag is correct. Recall asks
how much of the fraud I find. Reversing the conditioning changes the question:
the fraction of flagged transactions that are fraud is not the fraction of
fraud transactions that are flagged.

If I flag only transfers of at least 10,000, precision rises to
$205/232=88.36\%$. Recall is $205/889=23.06\%$ **within transfers**,
but only $205/931=22.02\%$ across **all training transactions**.
I need to state which population I am evaluating.

Even predicting non-fraud for every training transaction would give
$667889/668820\approx99.86\%$ accuracy, with zero fraud recall.
This is why accuracy alone will not be an adequate baseline measure.

These are calculations for illustrative rules on the data I explored.
I have not selected a deployment threshold or evaluated these rules on a
later period.

## The algorithm behind the grouped counts

I can reproduce the amount-band calculation with a simple loop. For each
transaction, I check the training condition, assign one of four bands, add
one to that band's count, and add its binary label to that band's fraud
count. For the transfer query I also check its type.

For $N$ input rows and $K$ groups, this counting algorithm takes
$O(N)$ time and $O(K)$ space for its counters. For the four fixed
amount bands, $K=4$. This explains why I can compute these summaries
without retaining all matching transactions in memory.

That is a complexity statement about the counting algorithm, not a
benchmark or a claim about SQLite's execution plan. A database may use
sorting, temporary storage or indexes. I have not measured or compared
those plans in this stage.

## What this changes for the model

I now have specific comparisons to make. I will compare a simple baseline
with a model using the permitted features, and measure how results change
when type and category are removed. I will fit preprocessing on training
data and use the later validation period for model and threshold choices.

The next mathematical question is decision cost: how should missed fraud,
false alarms and limited review capacity affect a decision? I have not
assigned those costs yet. Before optimizing anything, I need to state the
cost assumptions and distinguish transaction amount from actual loss.
Calibration, expected costs and constrained decisions belong to that next
stage, once there are model scores to examine.

The test period remains a chronological holdout after earlier exploratory
auditing; it is not an entirely unseen dataset. These four SQL queries
did not use validation or test rows.

## Evidence and reproduction

The tables above were transcribed from the DB Browser results I obtained
on 5 October 2026. Their counts reconcile to 668,820 training transactions
and 931 fraud labels, and to 167,322 transfers and 889 transfer fraud labels.

The saved queries are:

- [Amount bands](../sql/explore_train_amount_bands.sql)
- [Transaction types](../sql/explore_train_transaction_types.sql)
- [Amount bands within transfers](../sql/explore_train_transfer_amount_bands.sql)
- [Monthly training fraud](../sql/explore_train_monthly_fraud.sql)

The initial average-amount query was run interactively; it is not one of
these four saved queries. The [SQLite validation account](sql_validation.md)
records the separate full comparison against the prepared Parquet file.
No model was trained during this SQL exploration.


## Subsequent milestone note

This page preserves the training-only SQL questions and conclusions as recorded before model evaluation. The later [baseline and feature-ablation results](feature_ablation_results.md) and [exploratory review-cost analysis](review_cost_followup.md) address some of the questions posed here. Probability calibration, temporal robustness and final holdout evaluation are not complete.
