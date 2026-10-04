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
tests. Full-dataset, value-by-value verification on the Mac is still pending;
the earlier manual totals must not be presented as that new verification.

## Interpretation and next work

The data are synthetic. Positive labels are rare, and a model that predicts
no positives could have high accuracy while detecting none. The next stage
is training-period exploration and a simple baseline, followed by validation
of precision, recall and decision thresholds. Fit preprocessing only on
training data. The test period is a chronological holdout after exploratory
auditing, not an entirely unseen dataset.
