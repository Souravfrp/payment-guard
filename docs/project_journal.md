# PaymentGuard Project Journal

## September 2026

### Repository and Environment Setup

I created a separate Conda environment for PaymentGuard because, after completing my master's at TIFR, I spent time looking at how data-science projects are organized. I understood that different projects may require different packages and versions. A separate environment lets me install and verify the tools needed for PaymentGuard without affecting my other Python projects.

I had already used Anaconda, Jupyter, Python, GitHub, SourceTree, and Overleaf. However, I had not previously used SQL as a complete part of a project. For PaymentGuard, I installed and verified SQLite and SQLAlchemy support because SQL analysis will be an important component of the project.

Previously, I mainly used SourceTree for Git operations. During this setup, I began learning how to perform the same operations directly in Terminal: checking the repository status, staging selected files, creating commits, and pushing them to GitHub. I also learned that Terminal and SourceTree are two ways of working with the same Git repository. A clear commit history allows other people, including recruiters, to follow how the project developed. My goal is to become comfortable using Python, Git, GitHub, and SourceTree together in one reproducible workflow.

In my earlier work, I often kept most of the analysis inside Jupyter notebooks. For PaymentGuard, I wanted to understand how a complete data-science project is organized, so I created separate folders for different types of work. The `data` folder keeps the original and processed datasets separate, while `src` contains Python code that can be reused outside a notebook. I will keep SQL queries in `sql`, correctness checks in `tests`, and explanations of my decisions in `docs`. The `dashboard` folder will contain the final interactive application, and `results` will store selected tables and figures. This structure should make the project easier for me to understand, reproduce, test, and explain to someone else.

## October 2026

### Schema, Leakage and Temporal Risk Audit

Before starting data cleaning or modelling, I examined which fields could safely be used for a transaction decision. Reading only the column names was not enough because the dataset contains nested metadata, supplied risk information and fields that describe the outcome.

I scanned all 1,090,000 transactions in chunks of 50,000 rows. This allowed me to examine the complete 729 MiB CSV without loading the whole file into memory at once. I confirmed that the timestamps could be extracted, but I found 4,527 decreases in the original row order. Therefore, I will sort the data by timestamp and use chronological validation instead of assuming that CSV order is chronological.

The leakage audit identified a serious synthetic-data limitation. The `integration_info`, `layering` and `structuring` metadata groups occur only in fraud-labelled rows and together identify every positive case. I excluded these groups and their related annotations instead of allowing a model to learn the supplied answer.

I then profiled the unresolved device, location and merchant fields. Device operating system, device type, city and state have complete coverage and low cardinality, but I will include them only under the assumption that they are available before authorization. Country is constant, postcode has high cardinality, and several merchant fields contain rare values that perfectly separate 58 fraud-labelled rows. These findings led me to exclude or restrict those fields.

I also measured fraud rates across the seven observed months. The overall rate changed during the period, while some rare transaction types and categories showed deterministic or unstable behaviour. I kept the top-level `type` and `category` fields as decision-time candidates, but I will compare models with and without them so that I can measure how much performance depends on the synthetic data structure.

Finally, I combined the audit, candidate-field profile and temporal evidence into an initial model feature policy. The policy records which fields can be included, which require an assumption or ablation test, which may only be used to derive historical features, and which must be excluded. This stage defines the rules for the first baseline; it does not yet claim that the data are cleaned or model-ready.

### Transaction Preparation and Chronological Splits

I turned the audited feature policy into a preparation pipeline.
Before the full run, I measured a small sample to assess memory,
checked metadata extraction against the earlier timestamp audit,
and added tests for invalid inputs and chronological boundaries.

The full run reconciled all 1,090,000 transactions and 1,411 positive
labels. I then reloaded the saved output and checked source references,
columns, labels and split assignments independently.

I recorded why each step was needed, including the additional
investigations prompted by leakage, unresolved metadata and
nonchronological file order, in my
[data-preparation account](data_preparation.md).

This completes preparation for the first baseline, not modelling
or evaluation of transaction decisions.


### SQLite Import and SQL Validation

I imported the prepared data into SQLite and practised SELECT, WHERE,
COUNT, SUM, DISTINCT, GROUP BY and timestamp summaries in DB Browser.
The import totals matched 1,090,000 transactions and 1,411 positive labels.
I saved four SQL checks for totals, splits, source references, labels,
amounts and missing timestamps, and committed them to the recovery branch.

I then added a reusable converter with exclusive publication and read-only
comparison against the prepared Parquet file. The automated tests use small
fixtures. On 5 October 2026 (Asia/Kolkata), I ran the read-only verification
at commit `1fb4602` in `payment-guard-verify` on my Mac. It confirmed every
imported value against the prepared data, with 1,090,000 transactions and
1,411 positive labels. The [SQL validation account](sql_validation.md) records
the evidence and scope. No model has been trained yet.

### First Training-Data SQL Exploration — 5 October 2026

I used DB Browser to compare transaction amounts and types in the training
period. I then repeated the amount comparison within transfers to see whether
the pattern remained within one type. I saved all three grouped queries.

The larger amount bands had higher observed fraud rates, including within
transfers. But a rule that flagged only large transactions would miss most
fraud labels. Calculating precision and recall helped me see why these two
questions have different denominators. I also found that all 42 training
PAYMENT transactions were fraud-labelled, which supports the planned comparison
of models with and without type and category.

I recorded the results and their limits in my [training SQL exploration
notes](training_sql_exploration.md), including how the grouped counts connect
to empirical conditional proportions and a simple counting algorithm.
These were training-data observations on synthetic data; no model or
transaction decision policy was fitted at this stage.
