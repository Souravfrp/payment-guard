# Cost, transaction-value and missed-case follow-up

## Status

This is an exploratory follow-up to the completed February validation ablation, not a new model fit or holdout evaluation. The scripts were reviewed and subsequently run by the project owner in the local `payment-guard` environment. The owner shared terminal outputs for the four scripts and a test run of **72 passed, 70 dependency deprecation warnings**. These are user-reported local executions, not an independent execution by the GitHub documentation updater. The saved row-level Parquet files remain local.

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

The earlier local diagnostic reported 2,109,203.43 total fraud-labelled validation transaction-amount units and 2,104,151.39 captured by the full log model at 1,680 reviews (99.76%). Those figures were **confirmed by the owner's subsequent local script execution**: all 27 model-budget caught counts matched the committed aggregate summary. The row-level files were not uploaded for an independent rerun. Amount capture is not loss prevention.

## Locally checked economic sensitivity

At a review cost of INR 100 and a missed-label cost of INR 50,000, the lowest hypothetical total among the 30 model-budget combinations is **INR 468,000** for the full log model at 1,680 reviews (196 caught, six missed, 1,484 false alarms). The same model at 840 reviews costs **INR 484,000** (194 caught, eight missed). These are assumed costs, not measured losses or savings.

The owner also ran the cost-ratio sensitivity script. For tested missed-to-review cost ratios **1, 10, 24**, the preferred option was the log model at **168** reviews; for **50, 100, 200, 420**, it was the log model at **840** reviews; and for **500, 1,000**, it was the log model at **1,680** reviews. These are tested points, not complete intervals. For the full log model, C(840)=840c_r+8c_m and C(1680)=1680c_r+6c_m; they tie at c_m/c_r=420. The script favors the smaller budget in a tie.

## Locally checked transaction-value results

The owner's validation run printed **2,109,203.43** total fraud-labelled transaction-amount units. At 1,680 reviews, the amount-ranking reference caught 109 labels and **90.93%** of fraud-labelled amount; the raw-amount logistic model caught 188 and **99.26%**; and the log-amount logistic model caught 196 and **99.76%** (**2,104,151.39** units). The remaining six positives sum to **5,052.04** units. This is labelled transaction amount, **not** observed losses prevented or money recovered.

## Missed positives under the full log model at 1,680 reviews

The owner's `analyze_missed_fraud.py` run produced the following source-row IDs and ranks:

| Source row | Rank | Amount (dataset units) | Type | Category |
| ---: | ---: | ---: | --- | --- |
| 830467 | 1,799 | 2,421.62 | TRANSFER | Shell Company |
| 830408 | 2,457 | 2,213.78 | TRANSFER | Shell Company |
| 719998 | 3,969 | 231.69 | TRANSFER | Other |
| 722735 | 19,569 | 69.52 | TRANSFER | Other |
| 722740 | 27,862 | 41.37 | TRANSFER | Other |
| 816607 | 147,092 | 74.05 | TRANSFER | Shell Company |

All six are transfers; three belong to each of the two listed categories. This describes the errors but does not establish why the model ranked them poorly. Feature-level comparisons and more data would be needed for an explanation.

## Corrected validation amount-band fraud rates

A local check found an exploratory percentage-calculation error: the grouped `fraud_cases` column had NumPy/pandas dtype `int8`, and evaluating `100 * fraud_cases / transactions` overflowed before division. The owner reran the calculation after casting counts to `float64`, confirming the following figures:

| Transaction amount band | Validation transactions | Fraud-labelled cases | Fraud rate |
| --- | ---: | ---: | ---: |
| <100 | 36,712 | 3 | 0.008172% |
| 100–<1,000 | 98,614 | 23 | 0.023323% |
| 1,000–<10,000 | 32,580 | 124 | 0.380602% |
| ≥10,000 | 84 | 52 | 61.904762% |
| **Total** | **167,990** | **202** | **0.120245%** |

Use `100.0 * fraud_cases.astype("float64") / transactions.astype("float64")` (or divide before multiplying). The invalid prior rates, including a negative rate, must not be cited. This issue was found in a separate exploratory aggregation, not in the committed model ranking metrics. The highest band has only 84 observations; its observed fraud rate should not be generalized to real payments.

## Research questions answered and unanswered

The fixed-budget comparison quantifies how many positive labels are recovered per available investigation slot. The missed-case script can identify which labelled positives lie below the cutoff. The cost scenarios show how rankings and budgets would compare under specified assumptions. They do **not** establish real bank costs, optimal thresholds, calibration, causal feature importance, generalization to real transactions, or final holdout performance.

A separate within-amount-band diagnostic was reported locally; because its generating script and full input evidence were not supplied in this ZIP, I have not elevated its detailed band-level counts to independently verified repository results. The earlier band-rate output was corrected using floating-point arithmetic; see the verified table above.

## Run locally

From the repository root with the saved validation predictions and prepared dataset present:

```bash
python scripts/analyze_review_costs.py
python scripts/analyze_cost_sensitivity.py
python scripts/analyze_fraud_value_capture.py
python scripts/analyze_missed_fraud.py
```

Use the committed aggregate summary for the first two scripts. The latter two need the uncommitted local Parquet inputs. Preserve the printed output and environment information when citing new results. See [methodology](review_budget_methodology.md) for definitions and limitations.
