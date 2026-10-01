# PaymentGuard Data Workspace

## Purpose and Scope

PaymentGuard is a one-month project in which I study a practical problem faced by payment companies. When a new transaction arrives, the company must decide whether to approve it, send it for manual review, or block it.

My main question is: how can these decisions reduce fraud losses without rejecting too many genuine transactions or sending more cases for review than the available team can handle?

This document records how I design and organize the transaction data for the project. It explains what one observation represents, what the variables mean, where the data come from, which information is available when a decision is made, and how the data will be validated. I will clearly label any synthetic data and will not present it as real or confidential company data.

## Unit of Observation

One row represents one payment transaction request at the time recorded in `metadata.timestamp`. The row contains the transaction amount, transaction type, category, origin and destination accounts, balance information, calendar fields, nested metadata and supplied outcome labels.

The released dataset does not contain a reliable transaction identifier. The `step` column is zero in every row, so I cannot use it to identify transactions or order them in time. I also do not treat the CSV row number as a business identifier. The selected outcome is `isFraud`, which is used only as the target.

## Transaction Schema

The raw CSV contains 17 top-level columns. I also found 47 paths inside the nested `metadata` field. I audited these fields according to whether they could reasonably be available before an approve, review or block decision.

For the initial baseline, I include `amount`, `oldbalanceOrg`, `hour`, `day_of_week`, `day_of_month`, `month` and `metadata.payment_method`. I include device operating system, device type, city and state only under the documented assumption that they are captured before authorization.

I retain `type` and the top-level `category` as decision-time candidates, but they require an ablation comparison. Some rare values in these fields perfectly separate fraud-labelled transactions in this synthetic dataset, so I must measure how much model performance depends on them.

I reserve `nameOrig`, `nameDest`, device IP address and merchant ID for historical features calculated only from earlier transactions. I exclude raw postcode from the first baseline because its 5,000 values can encourage memorization. I use `metadata.timestamp` for chronological ordering and time-derived features, while `isFraud` remains the target.

I exclude post-transaction balances, duplicate labels, outcome descriptions, supplied probabilities, precomputed risk fields and metadata groups that reveal the target. The complete decision table is recorded in [`amlnet_v2_initial_model_feature_policy.csv`](../results/tables/amlnet_v2_initial_model_feature_policy.csv).

## Data Sources and Provenance

The raw source is AMLNet Version 2.0, a synthetic anti-money-laundering benchmark dataset published through Zenodo. The local file is `data/raw/amlnet_v2/AMLNet_v2_transactions.csv`. It contains 1,090,000 transactions dated from 13 October 2025 to 27 April 2026.

I verified the downloaded file against Zenodo's published MD5 checksum and also calculated its SHA-256 checksum. The raw CSV is excluded from Git because it is approximately 729 MiB. Its source, licence, checksum values and reproducible download procedure are recorded in the [raw-data documentation](../data/raw/amlnet_v2/README.md).

## Temporal Integrity Rules

I extracted all 1,090,000 timestamps without a parsing failure. The separate calendar fields agree with the extracted timestamp. However, I found 4,527 chronological decreases in the original CSV row order. Therefore, I cannot assume that file order represents time.

Before chronological validation, I must sort transactions by the extracted timestamp. I will not use a random train-test split. Any categorical encoding or numerical transformation must be fitted on the training period only. Historical account, IP or merchant features must use information strictly earlier than the transaction being scored.

October 2025 begins on 13 October and April 2026 ends on 27 April, so these are partial months. The observed monthly fraud rate changes across the available period, from approximately 0.187% in October 2025 to 0.080% in April 2026, with an increase in January. I treat this as evidence that chronological behaviour must be examined rather than assuming a stable distribution.

The monthly evidence is saved in [`amlnet_v2_temporal_risk_profile.csv`](../results/tables/amlnet_v2_temporal_risk_profile.csv) and visualized in [`amlnet_v2_temporal_risk_profile.png`](../results/figures/amlnet_v2_temporal_risk_profile.png).

## Validation Rules

I validated the complete CSV in chunks of 50,000 rows. The checks confirmed 1,090,000 rows, 17 expected columns, valid calendar ranges and no missing required metadata keys. The `fraud_probability` field contains 10,000 missing values, but it is excluded from predictive features.

The dataset contains 1,411 fraud-labelled rows, giving an overall fraud rate of approximately 0.12945%. The `isFraud` and `isMoneyLaundering` columns agree in every row, so I do not treat them as separate prediction targets.

The leakage audit showed that `integration_info`, `layering` and `structuring` appear only in fraud-labelled rows and together identify all 1,411 positive cases. I exclude these paths and their related annotations. Candidate profiling also found rare merchant-average, merchant-category and merchant-risk groups in which all 58 observed rows are fraud-labelled.

The reproducible audit outputs are:

- [`amlnet_v2_leakage_audit.csv`](../results/tables/amlnet_v2_leakage_audit.csv)
- [`amlnet_v2_candidate_metadata_profile.csv`](../results/tables/amlnet_v2_candidate_metadata_profile.csv)
- [`amlnet_v2_initial_model_feature_policy.csv`](../results/tables/amlnet_v2_initial_model_feature_policy.csv)

These checks validate the raw-data structure and define modelling restrictions. They do not mean that the dataset has already been cleaned or made model-ready.

## Privacy and Synthetic Data Policy

AMLNet Version 2.0 is fully synthetic. I do not present its customers, accounts, merchants, devices, IP addresses, transactions or labels as records of real people or financial institutions.

I still treat account names, merchant IDs, IP addresses and location fields as identifiers when designing features and publishing outputs. The raw CSV remains outside Git, and published tables and figures contain aggregated evidence rather than individual transaction records.

The dataset is published under the Creative Commons Attribution-NonCommercial 4.0 International licence. PaymentGuard uses it as an independent educational and non-commercial portfolio project, with the source and licence recorded in the repository.
