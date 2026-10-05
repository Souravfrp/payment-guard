"""Build or read-only verify the SQLite copy of prepared AMLNet data."""

from __future__ import annotations

import argparse
from contextlib import closing
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from payment_guard.prepare_amlnet_v2_data import METADATA_FIELDS, TOP_LEVEL_FEATURES


def load_expected(source: Path) -> pd.DataFrame:
    """Load prepared data and serialize naive timestamps without precision loss."""
    data = pd.read_parquet(source)
    columns = TOP_LEVEL_FEATURES + list(METADATA_FIELDS) + [
        "source_row", "timestamp", "isFraud", "split"
    ]
    if not data.columns.is_unique or set(data.columns) != set(columns):
        raise ValueError("Unexpected prepared columns.")
    if data.empty or data.isna().any().any():
        raise ValueError("Prepared data must be nonempty and contain no missing values.")
    refs = data["source_row"]
    if not pd.api.types.is_integer_dtype(refs) or refs.min() != 1 or refs.max() != len(data) or not refs.is_unique:
        raise ValueError("source_row must contain each integer from 1 to the row count.")
    if not data["isFraud"].isin([0, 1]).all():
        raise ValueError("Invalid labels.")
    if not np.isfinite(data["amount"]).all() or (data["amount"] < 0).any():
        raise ValueError("Invalid amounts.")
    times = data["timestamp"]
    if not pd.api.types.is_datetime64_dtype(times.dtype):
        raise ValueError("Expected timezone-naive datetime timestamps.")
    if not times.eq(times.dt.floor("us")).all():
        raise ValueError("Timestamps must not lose sub-microsecond precision.")
    splits = np.where(times < pd.Timestamp("2026-02-01"), "train",
                      np.where(times < pd.Timestamp("2026-03-01"), "validation", "test"))
    if not data["split"].eq(splits).all():
        raise ValueError("Splits disagree with chronological boundaries.")
    data = data.sort_values("source_row").reset_index(drop=True)
    data["timestamp"] = data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    return data


def verify_database(expected: pd.DataFrame, destination: Path) -> None:
    """Compare all columns and values without modifying or creating a database."""
    uri = destination.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        columns = [row[1] for row in connection.execute('PRAGMA table_info("transactions")')]
        if len(columns) != len(expected.columns) or set(columns) != set(expected.columns):
            raise ValueError("SQLite columns do not match prepared data.")
        offset = 0
        for actual in pd.read_sql_query(
            'SELECT * FROM "transactions" ORDER BY "source_row"',
            connection, chunksize=50_000,
        ):
            reference = expected.iloc[offset:offset + len(actual)].reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(
                    actual[list(expected.columns)], reference,
                    check_dtype=False, check_exact=True,
                )
            except AssertionError as error:
                raise ValueError(f"SQLite values differ near row {offset + 1}.") from error
            offset += len(actual)
        if offset != len(expected):
            raise ValueError("SQLite row count does not match prepared data.")


def build_database(expected: pd.DataFrame, destination: Path) -> None:
    """Publish a verified file exclusively; never overwrite an existing path."""
    if destination.exists():
        raise FileExistsError(f"Database already exists: {destination}; use --verify-only.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".sqlite-build-", dir=destination.parent) as staging:
        temporary = Path(staging) / "database.sqlite"
        with closing(sqlite3.connect(temporary)) as connection:
            expected.to_sql("transactions", connection, index=False, if_exists="fail", chunksize=10_000)
        verify_database(expected, temporary)
        os.link(temporary, destination)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/processed/amlnet_v2_prepared.parquet")
    parser.add_argument("--output", type=Path, default=root / "data/processed/payment_guard_data.sqlite")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print("Reading prepared transactions...", flush=True)
    expected = load_expected(args.input)
    if len(expected) != 1_090_000 or int(expected["isFraud"].sum()) != 1_411:
        raise ValueError("Prepared totals disagree with the audited AMLNet checkpoint.")
    if args.verify_only:
        verify_database(expected, args.output)
    else:
        build_database(expected, args.output)
    print(f"Verified transactions: {len(expected):,}")
    print(f"Verified positive labels: {int(expected['isFraud'].sum()):,}")
    print("Verified every imported value against prepared data.")
    print(f"Database: {args.output.resolve()}")


if __name__ == "__main__":
    main()
