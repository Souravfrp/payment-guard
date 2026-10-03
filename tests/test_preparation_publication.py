"""Output recovery checks using real Parquet bytes and injected I/O failures."""

import json
import os

import pandas as pd
import pytest

from payment_guard.prepare_amlnet_v2_data import publish_output_pair


@pytest.fixture
def outputs(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    table = staging / "table.parquet"
    report = staging / "report.json"
    frame = pd.DataFrame({"source_row": [1, 2], "isFraud": [0, 1]})
    frame.to_parquet(table, engine="pyarrow", index=False)
    report.write_text(json.dumps({"rows": 2, "positive_labels": 1}))
    return table, tmp_path / table.name, report, tmp_path / report.name


def test_publish_preserves_readable_parquet_and_report(outputs):
    publish_output_pair(*outputs)
    table, final_table, report, final_report = outputs
    # Published files remain usable after staging cleanup.
    table.unlink()
    report.unlink()
    frame = pd.read_parquet(final_table)
    summary = json.loads(final_report.read_text())
    assert len(frame) == summary["rows"] == 2
    assert int(frame.isFraud.sum()) == summary["positive_labels"] == 1


@pytest.mark.parametrize("position", [1, 3])
def test_never_overwrites_existing_output(outputs, position):
    outputs[position].write_bytes(b"existing output")
    with pytest.raises(FileExistsError):
        publish_output_pair(*outputs)
    assert outputs[position].read_bytes() == b"existing output"
    assert not outputs[3 if position == 1 else 1].exists()


@pytest.mark.parametrize("failing_call", [1, 2])
def test_failed_publication_can_be_retried(outputs, monkeypatch, failing_call):
    original = os.link
    calls = 0

    def fail_once(source, destination):
        nonlocal calls
        calls += 1
        if calls == failing_call:
            raise OSError("injected publication failure")
        return original(source, destination)

    monkeypatch.setattr(os, "link", fail_once)
    with pytest.raises(OSError, match="injected"):
        publish_output_pair(*outputs)
    assert not outputs[1].exists()
    assert not outputs[3].exists()
    monkeypatch.setattr(os, "link", original)
    publish_output_pair(*outputs)
    assert outputs[1].exists() and outputs[3].exists()


def test_missing_staged_report_rolls_back_table(outputs):
    outputs[2].unlink()
    with pytest.raises(FileNotFoundError):
        publish_output_pair(*outputs)
    assert not outputs[1].exists()
    assert not outputs[3].exists()
