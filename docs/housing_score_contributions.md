# What the Housing cutoff examples showed me

## Why I checked this

The log-amount model without category used 1,519 of its 1,680
validation reviews on Housing transactions. All were labelled
non-fraud. I wanted to understand the actual score calculation,
rather than explain it only through group averages.

## How I checked it

I used the saved logistic_ablation_v1 model and validation
predictions. I matched transactions using source_row, checked
one-to-one matching and agreement of labels and timestamps,
and reproduced the ranking using scores and saved tie priorities.

I selected the Housing transfer nearest the cutoff on each side.
In this run, these were also ranks 1,680 and 1,681 overall.

I applied the original log-amount and calendar transformations,
then the fitted preprocessing. For each transformed feature,
I multiplied its value by its coefficient.

The intercept plus these contributions gave the log-odds.
Applying the sigmoid function reproduced both saved probabilities.

## Results

| Quantity | Selected | Not selected |
| --- | ---: | ---: |
| Source row | 671963 | 706036 |
| Rank | 1680 | 1681 |
| Amount | 5572.37 | 4017.28 |
| Origin balance | 387269.34 | 336332.37 |
| Amount contribution | 5.806329 | 5.191004 |
| TRANSFER contribution | 3.694723 | 3.694723 |
| Final log-odds | -2.877067 | -2.877071 |
| Model probability | 0.0532989135 | 0.0532987186 |
| Actual label | 0 | 0 |

Amount and transfer type were the largest positive contributions
in both examples. The selected transaction received about 0.615
more from amount, but the remaining contributions almost entirely
offset this difference.

The selected transaction had a higher origin balance in this
pair. This reminded me that a difference in group medians does
not have to hold for every individual comparison.

## What the cutoff means

The policy reviewed the 1,680 highest scores. It did not require
a probability above 50%. These two transactions had almost
identical scores, but the capacity limit admitted only one.

This is a ranking boundary, not evidence of a sudden change in
risk. The probabilities have not been established as calibrated
real-world fraud frequencies.

## Limits

These are two illustrative validation examples from synthetic
data, not a matched causal comparison or an explanation of all
Housing false alarms.

The contributions explain this fitted model in its existing
encoding. They do not establish causal effects of the features.
No model was refitted and no Housing-specific rule was added.

The checks reproduced aggregate confusion counts and these two
predictions. They did not independently verify artifact hashes
or every saved probability. The final test set was not used.

## Reproduce

From the repository root, with the local prepared data and saved
ablation artifacts available:

```bash
PYTHONPATH=src python scripts/explain_housing_cutoff.py
