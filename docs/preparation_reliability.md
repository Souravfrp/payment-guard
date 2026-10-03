# Preparation reliability checkpoint

The preparation pipeline previously published the Parquet file and JSON report
using two replacements. If the second replacement failed, the table remained
and the next run refused to overwrite it. Replacement also allowed a concurrent
run to overwrite a destination created after the initial existence check.

Publication now creates hard links from staged files on the same filesystem.
Creating a destination that already exists fails instead of overwriting it.
If report publication raises an error, the table published by this call is
removed and the original error is raised. Each run uses its own temporary
directory, which is cleaned up when the write block exits.

This is recovery from raised errors, not an atomic two-file transaction.
Readers must require both final files. A forced process termination, power
failure, or a failure during rollback can still leave an incomplete pair.
Inspect any incomplete pair before moving it aside and rerunning; do not
automatically delete existing outputs. Filesystems without hard-link support
will reject publication. Concurrent removal or modification by unrelated
programs is outside this recovery guarantee.

## Verified in the review environment

- Original preparation suite: 32 passed.
- Expanded preparation and publication suite: 38 passed.
- New checks use actual PyArrow Parquet writing and reading.
- Injected failures cover both publication steps, preservation of either
  pre-existing destination, missing staged reports, and successful retry.
- Tests run on Linux with Python 3.12; these do not recreate the Mac environment.
- No full AMLNet preparation was rerun, because raw data are excluded from Git.

## Fresh Conda environment check on the Mac

Run from the project root after checking out this change. Use a separate
environment so the existing payment-guard environment remains available:

```bash
conda env create --name payment-guard-verify --file environment.yml
conda run --name payment-guard-verify python -c "import pandas, numpy, pyarrow, sklearn, sqlalchemy, sqlite3; print('Core imports OK')"
conda run --name payment-guard-verify env PYTHONPATH=src python -m pytest -q
```

Expected test result for this change: 38 passed. Record actual output before
marking fresh-environment verification complete. No Conda executable was
available in the review environment, so this remains pending.

The existing prepared data need not be regenerated for these tests.

