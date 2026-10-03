# How I Handled and Prepared the Transaction Data

## Why this work comes before modelling

My objective is to study how a payment company can approve, review
or block a transaction using information available when it arrives.
The eventual decision must account for fraud losses, false rejections
and review constraints.

Before fitting a model, I needed to establish what the data represent,
which fields I can use, and how I can evaluate predictions over time.
This document connects my earlier acquisition and audit work with
the preparation stage. It records the reasons for additional checks
rather than presenting the process as an uninterrupted sequence.

## 1. Establish the source and preserve the raw file

I selected AMLNet Version 2.0 for its transaction detail, recent
synthetic transaction period and rare positive labels. The dataset
provenance, licence and selection rationale are recorded in the
[raw-data README](../data/raw/amlnet_v2/README.md).

I kept the raw CSV separate from processed outputs. This lets me
reproduce preparation and investigate a result against its source.
Recent synthetic dates do not establish similarity to current
production payments.

## 2. Verify file integrity

I used the downloader to check the published MD5 and record SHA-256.
During preparation, I checked SHA-256 again against the recorded value.

A matching checksum establishes byte-level agreement with the recorded
file. It does not establish realism, correct labels or safe predictors.
That distinction required a separate content-validation stage.

## 3. Validate the contents in chunks

The raw file is approximately 729 MiB. I validated it in 50,000-row
chunks rather than loading the entire raw table at once.

The earlier validation recorded 1,090,000 rows, 17 columns and 1,411
positive labels. It also identified an unusable step column,
identical fraud and money-laundering labels, and missing values in
the supplied fraud-probability field.

I therefore use isFraud as one supplied binary outcome. I do not
claim to have separate fraud and money-laundering prediction tasks.

## 4. Check what information can enter a decision

Column names alone were insufficient because metadata contains
nested fields, supplied scores and outcome annotations.

I audited 17 top-level columns and 47 metadata paths. The
integration_info, layering and structuring groups together identified
every positive-labelled record. I excluded these annotations.

I also excluded post-transaction balances and supplied scores whose
construction or decision-time availability was not established.
A model using those fields could produce misleading performance.

## 5. Investigate unresolved fields

The first audit left device, location and merchant fields unresolved.
I added a profiling step to examine coverage, distinct values and
label patterns before deciding how to use them.

Country was constant, postcode had high cardinality, and some merchant
groups perfectly separated 58 positive-labelled records. I excluded
or restricted these fields rather than automatically using every column.

Device operating system, device type, city and state remain conditional
on the assumption that they are captured before authorization.
The dataset does not independently prove that availability.

## 6. Check time ordering and temporal behaviour

I found 4,527 decreases in timestamps in the original CSV order.
Therefore, reading rows sequentially does not reproduce event order.
This finding required global sorting before chronological splitting.

I also examined monthly label rates and transaction groups.
Some type and category values had deterministic synthetic patterns.
Strong association alone is not proof of leakage, so I retained those
two fields provisionally and required comparisons with and without them.

## 7. Freeze an explicit feature policy

I combined the schema audit, candidate profiling and temporal evidence
into the [feature policy](../results/tables/amlnet_v2_initial_model_feature_policy.csv).

The initial candidate set has seven directly included fields,
four fields with availability assumptions and two requiring ablation.
The preparation code checks that its 13 predictors match this policy.

Identifiers reserved for later historical features are not direct
predictors and are not retained in this initial table. Building those
features later will require a separate extraction from the raw source.

## 8. Measure resources before choosing the implementation

My Mac has 16 GiB of physical memory. A 5,000-row pilot estimated
approximately 0.16 GiB for the initial compact full table.

This was a linear estimate from a sample, not a peak-memory guarantee.
It excluded raw chunks, intermediate copies and sorting structures.
I used it to support chunked extraction followed by an in-memory sort,
then measured the actual full run.

## 9. Parse approved metadata and validate each chunk

The metadata contains Python-style datetime expressions rather than
ordinary JSON. I parsed its syntax tree without executing those
expressions and extracted only permitted fields.

I required nonempty categorical strings, finite numerical values,
nonnegative amounts, binary labels and agreement between calendar
fields and timestamps. I did not assume that negative balances are
invalid without establishing a rule for them.

The preparation pipeline stops on invalid required records.
This is stricter than profiling scripts that count parsing failures
and continue collecting diagnostic evidence.

I retained timestamp microseconds and did not invent a timezone for
the dataset's timezone-naive timestamps.

## 10. Preserve traceability, sort and assign periods

I assigned each record a source_row reference, starting at one after
the CSV header. It identifies a source record, not a business
transaction, and it is not a predictor.

I sorted by timestamp and source_row. The second key makes ties
reproducible; it does not establish an event sequence within a tie.
Future historical features must use strictly earlier timestamps.

I assigned transactions before 1 February 2026 to training,
February to validation, and transactions from 1 March 2026 to testing.

| Split | Transactions | Positive labels | Positive rate |
|---|---:|---:|---:|
| train | 668,820 | 931 | 0.1392% |
| validation | 167,990 | 202 | 0.1202% |
| test | 253,190 | 278 | 0.1098% |

No transformation or model was fitted during preparation.
I will fit transformations on training data only and use validation
for model and policy choices before final test evaluation.

### Chronological split visualization

![Chronological splits and positive-label rates](../results/figures/amlnet_v2_chronological_splits.png)

I use the upper panel to show the observed split periods and the lower panel to compare empirical positive-label rates. These describe synthetic data; they do not establish statistical significance or model performance.

I reproduce the figure from the aggregate split summary using `PYTHONPATH=src python -m payment_guard.plot_amlnet_v2_splits`.

## 11. Reconcile counts and examine repeated records

The full preparation matched 1,090,000 rows and
1,411 positive labels from the earlier audit.

I found 0 rows in repeated
groups defined by equality of the selected predictors, timestamp
and outcome. This is not a full raw-row duplicate audit or proof
of business transaction identity. I retained every record.

## 12. Verify the saved artifact separately

I saved the prepared table as Parquet and independently reloaded it.
The saved-output check confirmed expected columns, no missing values,
binary labels, complete unique source references, chronological order,
split assignments and agreement with the run report.

I also verified strict separation between the three time periods.
The 32 automated tests cover parser behaviour, invalid chunk inputs,
source references, feature-policy enforcement and split boundaries.
They do not cover every possible full-run or output-writing failure.

## Mathematics and computational reasoning

For record i, I separate timestamp t_i, permitted predictors x_i
and supplied outcome y_i. I require y_i to belong to {0, 1}.

For a split with n records and k positive labels, its empirical
positive rate is k / n. This describes synthetic labels, not a
measured production fraud probability.

Temporal separation requires:

- max(training timestamps) < min(validation timestamps)
- max(validation timestamps) < min(test timestamps)

For sample memory M_n across n rows, I used M_n * N / n to estimate
the table memory for N rows. This assumes representative record sizes.

Let B denote input bytes and N the transaction count. Checksum
verification and parsing scan the input. With fixed-format records,
their work is O(B); comparison-based sorting takes O(N log N)
comparisons. Fixed-column validation and split assignment take O(N).

For C rows per chunk, raw-chunk storage grows with C. However,
I accumulate all prepared records for sorting, so overall memory
also grows with N. Chunked reading does not make this pipeline
constant-memory. These complexity statements describe the principal
operations, not guaranteed performance of every library operation.

## Measured full-run performance

The local run report records:

- Prepared-table memory: 154.81 MiB
- Process peak resident memory: 925.14 MiB
- Extraction: 58.42 seconds
- Sorting: 0.33 seconds
- Parquet writing: 1.64 seconds
- Elapsed before output publication: 61.01 seconds

The Terminal reported 61.01 seconds total for this run.
These are measurements from one execution on my Mac.
Peak process memory includes more than the final table.

## Limits and next work

My earlier feature audit used full-period label patterns.
The test period is therefore a chronological holdout after exploratory
auditing, not a completely unseen dataset.

AMLNet is synthetic, and its two supplied outcome labels are identical.
This preparation does not establish effectiveness on real payment fraud.

I have not trained a model or evaluated an approve, review or block
policy. Historical features, feature ablations, calibration and
explicit economic assumptions remain to be implemented.

I added PyArrow to the environment specification because the
preparation code explicitly uses it for Parquet writing.
Recreating the environment from scratch remains to be checked.
The output-writing workflow also needs further failure-mode checks:
the table and report are published separately, not as one transaction.

## Evidence and sources

- [Project objective](../README.md)
- [Dataset provenance, licence and checksums](../data/raw/amlnet_v2/README.md)
- [Data workspace and audit reasoning](data_workspace.md)
- [Project journal](project_journal.md)
- [Leakage audit](../results/tables/amlnet_v2_leakage_audit.csv)
- [Candidate profiling](../results/tables/amlnet_v2_candidate_metadata_profile.csv)
- [Temporal evidence](../results/tables/amlnet_v2_temporal_risk_profile.csv)
- [Feature policy](../results/tables/amlnet_v2_initial_model_feature_policy.csv)
- [Preparation implementation](../src/payment_guard/prepare_amlnet_v2_data.py)
- [Automated tests](../tests/test_prepare_amlnet_v2_data.py)

The aggregate split summary is recorded in
[amlnet_v2_split_summary.csv](../results/tables/amlnet_v2_split_summary.csv).
Raw and processed transaction files remain excluded from Git.
