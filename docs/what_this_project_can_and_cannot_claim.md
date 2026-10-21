# Payment Guard: what I can and cannot claim

*A short guide for recruiters and technical reviewers — 21 October 2026*

## What this project is about

I built Payment Guard to investigate a practical ranking problem: **when a payment-review team can inspect only a small fraction of transactions, which ones should a model put first?** I used the synthetic AMLNet v2 dataset to work through data checks, chronological splits, SQL exploration, interpretable logistic baselines, XGBoost, model calibration, feature-removal tests, and a final frozen holdout comparison.

The goal was to make the evaluation defensible, not to claim that I had built a bank-ready fraud-detection service. I kept the research code and aggregate results public while excluding the transaction-level data, predictions, fitted models and private manifests.

## What the completed experiment demonstrates

- **A controlled comparison at a fixed workload.** Before the completed March–April 2026 final scoring run, I fixed the models, pooled holdout period, top-1% review budget, tie handling and primary comparison. I evaluated the saved models without retraining or selecting a winner afterward. An earlier command *opened and validated* holdout rows but failed on a Python import before producing scores; that interruption is disclosed in the [execution record](final_holdout_execution.md).
- **An observed advantage for XGBoost on the declared primary measure.** Among **253,190 synthetic transactions** with **278 positive labels**, each model selected **2,532** transactions for review. Uncalibrated XGBoost captured **267** labelled positives (96.04% recall); logistic regression captured **254** (91.37%). That is a **+13 positive-label difference**, not a 13% accuracy gain.
- **A different result when review capacity changes.** At the smaller **0.1%** budget, logistic regression captured **225** positives and XGBoost **222**. XGBoost led at 0.5% and at the declared 1% budget. I therefore cannot say one model always ranks better for every operating capacity.
- **Different review decisions even when positive-label capture overlaps.** At 1%, both models found **252** of the same positive labels. XGBoost alone found **15**, logistic alone found **2**, and neither found **9**. But the *entire* review lists overlapped on only **1,210 of 2,532 transactions**. Similar-looking headline metrics can conceal different workload choices.
- **Sensitivity to the feature set.** At 1%, XGBoost captured **169** positives when `category` was excluded and **141** when both `category` and `type` were excluded, compared with **267** using all features. These are comparisons using separately refitted, previously prepared feature-ablation models; they are **not causal effects** of individual features. In both reduced-feature comparisons, logistic regression had higher whole-ranking average precision.
- **An auditable published result.** The publicly committed [42 metric records](../results/tables/final_holdout_v1/metrics.csv), [nine paired-overlap records](../results/tables/final_holdout_v1/paired_overlap.csv) and [primary summary](../results/tables/final_holdout_v1/primary_summary.json) passed aggregate checksum and arithmetic checks. The full **186-test suite passed locally** on 21 October 2026. The public verifier does not load private predictions or fitted models, and a local test pass is **not** a GitHub CI pass.

## What I cannot conclude

1. **“XGBoost is always superior.”** No: logistic regression did better at the smallest tested review capacity and in average precision for the two reduced-feature variants.
2. **“The model detects 96% of fraud in real banking traffic.”** No: **96.04% is recall of labelled positives in this synthetic holdout** at a retrospective, pooled 1% review budget. Real-world labels, transaction mix, timing and fraud behaviour may differ substantially.
3. **“The feature set is proven usable at payment-decision time.”** No: `category` and `type` were provisionally treated as available, but their source-generation mechanisms and real-world decision-time availability were not established. Feature dependence does not, by itself, prove leakage or validity.
4. **“The holdout proves statistical superiority or generalization.”** No: +13 is an observed paired difference, not a significance test or uncertainty interval. The holdout came from the same synthetic data source, and exploratory dataset audits predated the frozen scoring experiment. It was not an entirely untouched dataset in every sense.
5. **“The model is production-ready or saves a known amount of money.”** No: no live deployment, approval/review/block decision policy, measured review costs, measured fraud losses avoided, latency/service reliability test, or model monitoring study was completed. Review budgets are experimental choices, not documented bank staffing limits.
6. **“Anyone can rerun every reported score from the public files alone.”** No: the public repository includes code, aggregate figures and consistency tests, but full score reproduction requires separately obtaining the data and rebuilding or verifying locally stored historical fitted artifacts. Transaction-level predictions and models have deliberately not been published.
7. **“Final-holdout calibration was optimized.”** No: the calibration mappings were fixed earlier. Probability losses were reported as secondary diagnostics; I did not change the selected model, calibration method or features using March–April results.

## How I would describe this work in an interview

> I built a synthetic-transaction fraud-review ranking study, starting from dataset validation and SQL analysis and moving through logistic regression, XGBoost, calibration and feature sensitivity. I froze a chronological final comparison at a 1% review capacity before scoring. XGBoost captured 267 of 278 labelled positives versus 254 for logistic regression, although logistic did better at a 0.1% review capacity. Removing two synthetic categorical fields weakened performance substantially, so I treat this as evidence about the dataset and the experimental protocol, not as proof that the model would work unchanged in a bank.

## What I would investigate next

I would first verify which source fields exist *before* a real payment decision and how they are generated. I would then use suitably governed real-world, time-stamped data to measure stability and uncertainty, account for delayed labels, choose a review threshold based on documented capacity and costs, and evaluate monitoring and failure modes. Those steps are **future work**, not completed outcomes.

For full definitions, monthly results, probability metrics and numeric evidence, see the [final holdout report](final_holdout_first_results.md), [frozen methodology](final_holdout_frozen_protocol.md), and [public aggregate audit script](../scripts/audit_final_holdout_public.py).
