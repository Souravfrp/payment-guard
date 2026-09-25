# AMLNet Version 2.0 Raw Data

## Source

The raw-data source selected for PaymentGuard is AMLNet Version 2.0.

- Dataset: AMLNet Synthetic Anti-Money Laundering Benchmark Dataset
- Version: 2.0
- Publisher: Zenodo
- DOI: https://doi.org/10.5281/zenodo.21237971
- Source record: https://zenodo.org/records/21237971
- Published: 7 July 2026
- Raw file: `AMLNet_v2_transactions.csv`

## Why I Selected This Dataset

While selecting the data for PaymentGuard, I wanted the transaction period to be close to the present time. AMLNet Version 2.0 contains transactions from 13 October 2025 to 27 April 2026, which matches this requirement.

It contains around 1.09 million transactions, with information about the transaction amount, payment type, accounts, merchants, balances, time, location and device. Fraud cases form only a small part of the data. This gives me a practical dataset for studying how transactions can be approved, reviewed or blocked when fraud is rare and review capacity is limited.

## Synthetic Data and License

AMLNet Version 2.0 is fully synthetic. Its customers, accounts, merchants, transactions and labels do not represent real people or financial institutions.

The dataset is published under the Creative Commons Attribution-NonCommercial 4.0 International license. PaymentGuard uses it as an educational and non-commercial project.

## Download and Integrity

I downloaded the raw CSV using `src/payment_guard/download_amlnet_v2_data.py`. The script retrieves the file from the fixed Zenodo record, supports retrying an interrupted download, and checks the file before treating it as complete.

The downloaded file is approximately 729 MiB on my Mac. Its MD5 checksum matches the value published by Zenodo:

- MD5: `0d3702e54495385bb8f01a688466a4e3`
- SHA-256: `09328eb063d525a23a1ce9be663615fae4afdfa082f2b622c462b7d0fe0eab2d`

The raw CSV is excluded from Git because of its size. The download script and this README remain in the repository so that the data source and download process can be reproduced.

## Validation Findings

I validated the complete CSV in chunks of 50,000 rows so that I did not need to load the entire file into memory at once. The dataset contains 1,090,000 transactions and 17 columns. There are 1,411 positively labelled transactions, giving a positive-label rate of approximately 0.12945%.

The validation also identified three limitations. The `step` column is zero in every row, so I cannot use it as a transaction identifier or time variable. The `isFraud` and `isMoneyLaundering` columns are identical in every row, so I use `isFraud` as the single target instead of treating them as two different prediction problems. The `fraud_probability` column contains 10,000 missing values and is closely connected to the supplied target, so I exclude it from predictive features.

I also exclude `laundering_typology`, `risk_indicators`, and `network_metrics` from the initial model because they may contain target-derived or future information. I will use only information that can reasonably be treated as available when the transaction decision is made.
