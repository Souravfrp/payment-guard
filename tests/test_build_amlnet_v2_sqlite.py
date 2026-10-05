"""Exercise real Parquet/SQLite files and publication failure recovery."""

import sqlite3

import pandas as pd
import pytest

from payment_guard import build_amlnet_v2_sqlite as module


@pytest.fixture
def prepared(tmp_path):
    data = pd.DataFrame({name: [1, 2, 3] for name in module.TOP_LEVEL_FEATURES})
    for name in module.METADATA_FIELDS:
        data[name] = ["a", "b", "c"]
    data["type"] = ["TRANSFER"] * 3
    data["category"] = ["payment"] * 3
    data["source_row"] = [3, 1, 2]
    data["isFraud"] = [0, 1, 0]
    data["timestamp"] = pd.to_datetime([
        "2026-01-31 23:59:59.123456", "2026-02-01 00:00:00.000001",
        "2026-03-01 00:00:00.636449",
    ])
    data["split"] = ["train", "validation", "test"]
    path = tmp_path / "prepared.parquet"
    data.to_parquet(path, index=False)
    return path


def test_round_trip_and_saved_sql(prepared, tmp_path):
    expected = module.load_expected(prepared)
    destination = tmp_path / "copy.sqlite"
    module.build_database(expected, destination)
    module.verify_database(expected, destination)
    with sqlite3.connect(destination) as connection:
        assert connection.execute("SELECT timestamp FROM transactions WHERE source_row=3").fetchone()[0] == "2026-01-31 23:59:59.123456"
        from pathlib import Path
        sql_root = Path(__file__).resolve().parents[1] / "sql"
        for file in sql_root.glob("*.sql"):
            for statement in file.read_text().split(";"):
                if statement.strip():
                    connection.execute(statement).fetchall()


def test_existing_destination_unchanged(prepared, tmp_path):
    destination = tmp_path / "copy.sqlite"
    destination.write_bytes(b"existing content")
    with pytest.raises(FileExistsError):
        module.build_database(module.load_expected(prepared), destination)
    assert destination.read_bytes() == b"existing content"


def test_same_count_corruption_detected(prepared, tmp_path):
    expected = module.load_expected(prepared)
    destination = tmp_path / "copy.sqlite"
    module.build_database(expected, destination)
    with sqlite3.connect(destination) as connection:
        connection.execute("UPDATE transactions SET amount=99 WHERE source_row=1")
    with pytest.raises(ValueError, match="values differ"):
        module.verify_database(expected, destination)


def test_read_only_missing_file(prepared, tmp_path):
    destination = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        module.verify_database(module.load_expected(prepared), destination)
    assert not destination.exists()


def test_failed_verification_leaves_no_output_and_allows_retry(prepared, tmp_path, monkeypatch):
    expected = module.load_expected(prepared)
    destination = tmp_path / "copy.sqlite"
    original = module.verify_database
    def fail(*args):
        raise ValueError("injected failure")
    monkeypatch.setattr(module, "verify_database", fail)
    with pytest.raises(ValueError, match="injected"):
        module.build_database(expected, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".sqlite-build-*"))
    monkeypatch.setattr(module, "verify_database", original)
    module.build_database(expected, destination)


@pytest.mark.parametrize("column,value", [("isFraud", 2), ("source_row", 0), ("split", "test"), ("amount", -1)])
def test_invalid_prepared_values_rejected(prepared, column, value):
    data = pd.read_parquet(prepared)
    data.loc[0, column] = value
    data.to_parquet(prepared, index=False)
    with pytest.raises(ValueError):
        module.load_expected(prepared)
