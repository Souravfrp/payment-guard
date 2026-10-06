# SQLite conversion and validation

The prepared Parquet file is the source for the SQLite `transactions` table.
SQLite lets me practise SQL and independently summarize the prepared data.
It does not train a model or establish fraud-detection performance.

## Recorded manual checkpoint

The initial conversion ran on Sourav's Mac. Its terminal output confirmed
1,090,000 transactions and 1,411 positive labels. The SQL checks saved in
`sql/` produced the following results in DB Browser for SQLite:

| Check | Observed result |
| --- | ---: |
| Total transactions | 1,090,000 |
| Positive labels | 1,411 |
| Positive percentage | 0.12945% |
| Nonmissing source references | 1,090,000 |
| Distinct source references | 1,090,000 |
| Invalid or missing binary labels | 0 |
| Missing or negative amounts | 0 |
| Missing or unrecognized split names | 0 |
| Missing or blank timestamps | 0 |

| Split | Transactions | Positive labels |
| --- | ---: | ---: |
| train | 668,820 | 931 |
| validation | 167,990 | 202 |
| test | 253,190 | 278 |

The split timestamp ranges matched the saved Python split summary.
These checks verify the named properties; they do not prove that every
imported cell matches the source. The original import used pandas-inferred
SQL types, without explicit primary-key or CHECK constraints.

## Reusable conversion and full comparison

From the repository root in the project environment, check the existing
Mac database without modifying it:

```bash
PYTHONPATH=src python -m payment_guard.build_amlnet_v2_sqlite --verify-only
```

If no database exists, build it using:

```bash
PYTHONPATH=src python -m payment_guard.build_amlnet_v2_sqlite
```

Defaults are `data/processed/amlnet_v2_prepared.parquet` and
`data/processed/payment_guard_data.sqlite`. Optional `--input` and `--output`
arguments accept other paths. The command requires the audited full-dataset
totals. It preserves six timestamp decimal places and refuses timezone-aware
or finer-than-microsecond timestamps rather than silently changing them.

The converter checks the prepared column set, source references, labels,
amounts and split boundaries. It compares every SQLite column and value
against the prepared data, ordered by source reference. It uses pandas-inferred
SQL types, so constraints are checked by the program rather than enforced
against later database edits. The read-only verification works with the
original manually created database too.

A new database is built in a temporary directory on the destination filesystem.
Only after the comparison succeeds is it published using an exclusive hard
link. Existing destinations are never replaced. Ordinary failures clean up
staging; forced termination can leave a staging directory. This is not a
power-loss durability guarantee. The filesystem must support hard links.
Keep the database closed in DB Browser during conversion or verification.

The Parquet file and sorted comparison frame are loaded into memory; SQLite
writes and reads are chunked. The new converter has automated small-fixture
tests.

### Completed full-dataset comparison

On 5 October 2026 (Asia/Kolkata), Sourav supplied terminal output from the
`payment-guard-verify` environment after pulling commit `1fb4602` and running
`--verify-only`. It reported:

```text
Verified transactions: 1,090,000
Verified positive labels: 1,411
Verified every imported value against prepared data.
Database: /Users/souravroy/Documents/payment-guard/data/processed/payment_guard_data.sqlite
```

This completed the full-dataset comparison of the existing SQLite table with
the prepared Parquet file without rebuilding the database. It is separate
from the earlier manual SQL summaries. It does not demonstrate a full-dataset
run of the new database-building path, and it does not validate the realism
of the synthetic labels. The expanded Mac test suite was run separately,
as recorded below.

## Interpretation and next work

The data are synthetic. Positive labels are rare, and a model that predicts
no positives could have high accuracy while detecting none. The [training-period exploration](training_sql_exploration.md) and
[first baseline comparison](logistic_baseline_first_results.md) are now complete.
The baselines report precision and recall at fixed review budgets; a live
decision threshold remains future work. Preprocessing is fitted only on
training data. The test period is a chronological holdout after exploratory
auditing, not an entirely unseen dataset.

## Expanded suite on the Mac — 5 October 2026

After pulling commit `e510a6b`, I ran the expanded suite in my existing
`payment-guard-verify` environment:

```bash
PYTHONPATH=src python -m pytest -q
```

The terminal reported:

```text
47 passed in 1.13s
```

This covered preparation, publication recovery, SQLite conversion and
verification tests, including execution of the saved SQL on small fixtures.
The timing is a test-suite duration, not a full-dataset processing benchmark.
I did not recreate the Conda environment or rebuild the full database in
this run. The full-dataset read-only comparison was verified separately.
