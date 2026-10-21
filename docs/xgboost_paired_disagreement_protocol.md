# January–February paired disagreement study

I compare **uncalibrated** logistic regression and XGBoost using the saved
predictions for **identical transactions** in each month. Both models use
October–November fitted data for January and October–December fitted data for
February. January and February are *historical development periods*, not an
untouched final test. March–April remains unscored.

For each month and each review fraction \(q\in\{0.001,0.005,0.01\}\), I
select the top \(k=\lceil qn\rceil\) rows using each model's score in descending
order. Exact ties use the **same fixed, label-independent tie priority** as
the earlier evaluations. For the observed positive labels, I partition:

- **Both:** captured by logistic regression and XGBoost.
- **Only logistic:** captured by logistic regression, not by XGBoost.
- **Only XGBoost:** captured by XGBoost, not by logistic regression.
- **Neither:** captured by neither model.

These four groups sum to the month's positive-label count. Each model's
captured positives are reconstructed from the shared and model-only groups,
so this measures paired differences rather than comparing two aggregate
recall counts alone.

The output contains one aggregate-only CSV and one PNG combining six full
positive-label allocation bars and a model-only-detail row. The script
verifies both saved-run manifests, input identity, prediction file SHA-256
hashes, row identities and ordering, dates, counts, tie priority, and the
original logistic scores. It does not open the prepared parquet dataset,
refit a model, load joblib objects, write row-level outputs or access
March–April. Run from the repo root:

```bash
PYTHONPATH=src python -m pytest -q tests/test_xgboost_paired_disagreement.py
PYTHONPATH=src python scripts/plot_xgboost_paired_disagreement.py
open models/xgboost_disagreement_v1
```

**Interpretation:** A greater number of `only_xgboost` than `only_logistic`
means XGBoost captured more positives at that one budget in that historical
month. It does not prove out-of-sample superiority. Pair-wise disagreement,
operational alert churn (the scripts also report overlap across *all* selected
transactions), feature ablation, uncertainty and time stability should be
considered before freezing a model. Both data and labels are synthetic.

Keep the original individual-transaction predictions and fitted models local.
For public review, inspect the aggregate CSV and PNG for disclosive content
before explicitly staging those two files. Do not commit them before checking.


## Observed historical results

The completed January and February comparison found:

| Month | Review budget | Both caught | Only logistic | Only XGBoost | Neither |
| --- | ---: | ---: | ---: | ---: | ---: |
| January | 0.1% | 173 | 14 | 14 | 80 |
| January | 0.5% | 265 | 1 | 2 | 13 |
| January | 1.0% | 268 | 3 | 3 | 7 |
| February | 0.1% | 160 | 5 | 8 | 29 |
| February | 0.5% | 194 | 0 | 3 | 5 |
| February | 1.0% | 196 | 1 | 2 | 3 |

At February's 1% budget, both models captured 196 of the
same positive-labelled transactions, but their complete
review lists overlapped on only 828 of 1,680 transactions.
This distinction matters when considering review-queue
stability and operational workload.

These findings show a modest historical ranking advantage
for XGBoost, not statistical proof of future superiority.

Public aggregate results:

- [Paired disagreement CSV](../results/tables/xgboost_benchmark_v1/january_february_paired_disagreements.csv)
- [Combined disagreement figure](../results/figures/xgboost_benchmark_v1/january_february_paired_disagreements.png)

Individual transaction predictions remain local.
March–April remains unscored.
