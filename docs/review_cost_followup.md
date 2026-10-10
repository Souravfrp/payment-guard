# Cost, transaction-value and missed-case follow-up

## Status

This is an exploratory follow-up to the completed February validation ablation, not a new model fit or holdout evaluation. The scripts were supplied for code review; their full-data execution was **not** independently reproduced during this GitHub update because row-level predictions and prepared Parquet data remain local.

## Established aggregate comparison

At 1,680 reviews (1% of validation transactions):

| Ranking | Fraud labels caught | Missed | False alarms | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Amount descending | 109 | 93 | 1,571 | 53.96% | 6.49% |
| Logistic, raw amount | 188 | 14 | 1,492 | 93.07% | 11.19% |
| Logistic, log amount | 196 | 6 | 1,484 | 97.03% | 11.67% |
| Logistic, log amount without category | 130 | 72 | 1,550 | 64.36% | 7.74% |

These figures are reconciled with the committed `logistic_ablation_summary.csv`. They describe a fixed retrospective workload, not real-time alert quality.

## What the four scripts answer

- `analyze_review_costs.py` applies hypothetical INR 100 per review and INR 50,000 per missed fraud label to the existing 30 model/budget rows.
- `analyze_cost_sensitivity.py` varies the ratio of missed-case cost to review cost. It compares only the three declared budgets and available rankings.
- `analyze_fraud_value_capture.py` joins saved validation predictions to transaction amounts by unique source row, checks labels and reconstructed top-k caught counts, and compares count recall with the share of fraud-labelled transaction amount selected.
- `analyze_missed_fraud.py` joins the saved log-model scores to validation transaction details and reports positives below rank 1,680. It asserts six misses and a summed transaction amount of 5,052.04 units, so it will fail if the underlying saved run differs.

The earlier local diagnostic reported 2,109,203.43 total fraud-labelled validation transaction-amount units and 2,104,151.39 captured by the full log model at 1,680 reviews (99.76%). Those figures are **previously reported local outputs**, not independently verified by this update. Amount capture is not loss prevention.

## Research questions answered and unanswered

The fixed-budget comparison quantifies how many positive labels are recovered per available investigation slot. The missed-case script can identify which labelled positives lie below the cutoff. The cost scenarios show how rankings and budgets would compare under specified assumptions. They do **not** establish real bank costs, optimal thresholds, calibration, causal feature importance, generalization to real transactions, or final holdout performance.

A separate within-amount-band diagnostic was reported locally; because its generating script and full input evidence were not supplied in this ZIP, I have not elevated its detailed band-level counts to independently verified repository results. Earlier band-rate output also requires a fresh denominator check.

## Run locally

From the repository root with the saved validation predictions and prepared dataset present:

```bash
python scripts/analyze_review_costs.py
python scripts/analyze_cost_sensitivity.py
python scripts/analyze_fraud_value_capture.py
python scripts/analyze_missed_fraud.py
```

Use the committed aggregate summary for the first two scripts. The latter two need the uncommitted local Parquet inputs. Preserve the printed output and environment information when citing new results. See [methodology](review_budget_methodology.md) for definitions and limitations.
