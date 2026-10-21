"""Fixture checks for versioned local serialization and verification protocol."""
from __future__ import annotations
import json

import numpy as np
import pandas as pd
import pytest

from payment_guard.calibration_saved_v2 import (
    OBJECT_FILES, audit_run, compare_saved_predictions, compare_previous_run,
    load_windows, run,
)
from payment_guard.calibration_temporal import fit_calibrators, calibrated_scores


def test_saved_objects_list_six_unique_names():
    assert len(OBJECT_FILES) == 6
    assert len(set(OBJECT_FILES.values())) == 6
    assert all(n.endswith(".joblib") for n in OBJECT_FILES.values())


def test_prediction_compare_passes_aligned_scores():
    p = np.array([0.01, 0.04, 0.9])
    actual = {k: p for k in ("uncalibrated", "sigmoid", "isotonic")}
    compare_saved_predictions(actual, {k: p.copy() for k in actual}, label="toy")


def test_prediction_compare_rejects_changed_values():
    p = np.array([0.01, 0.04, 0.9])
    a = {k: p.copy() for k in ("uncalibrated", "sigmoid", "isotonic")}
    b = {k: p.copy() for k in a}
    b["isotonic"][1] += 0.001
    with pytest.raises(AssertionError):
        compare_saved_predictions(a, b, label="toy")


def test_prediction_compare_rejects_missing_method():
    with pytest.raises(ValueError):
        compare_saved_predictions({"uncalibrated": [0.1]}, {"uncalibrated": [0.1]}, label="toy")


def test_versioned_run_refuses_to_replace_preexisting_path(tmp_path):
    existing = tmp_path / "v2"
    existing.mkdir()
    (existing / "sentinel.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        run(tmp_path / "nonexistent.parquet", existing)
    assert (existing / "sentinel.txt").read_text() == "keep"


def test_missing_prepared_file_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_windows(tmp_path / "absent.parquet")


def test_positive_sigmoid_and_isotonic_serialization(tmp_path):
    import joblib
    p = np.linspace(0.001, 0.999, 120)
    y = np.array([0] * 80 + [1] * 40)
    cal = fit_calibrators(p, y)
    before = calibrated_scores(p, cal)
    for name, model in cal.items():
        path = tmp_path / f"{name}.joblib"
        joblib.dump(model, path)
        loaded = joblib.load(path)
        from_loaded = loaded.predict(p)
        np.testing.assert_allclose(from_loaded, before[name], rtol=0, atol=0)


def test_v1_comparison_rejects_changed_data_fingerprint(tmp_path):
    first, second = tmp_path / "v1", tmp_path / "v2"
    first.mkdir()
    second.mkdir()
    (first / "run.json").write_text(json.dumps({"complete": True, "input_sha256": "a", "selected_method_on_january": "isotonic"}))
    (second / "run.json").write_text(json.dumps({"complete": True, "input_sha256": "b", "selected_method_on_january": "isotonic"}))
    with pytest.raises(ValueError, match="fingerprint"):
        compare_previous_run(second, first)


def test_saved_run_rejects_tampered_model_before_deserialization(tmp_path):
    folder = tmp_path / "model_run"
    folder.mkdir()
    hashes = {n: "0" * 64 for n in OBJECT_FILES.values()}
    for filename in hashes:
        (folder / filename).write_bytes(b"not a trusted joblib artifact")
    (folder / "run.json").write_text(json.dumps({
        "complete": True, "march_april_scored": False,
        "artifact_sha256": hashes, "selected_method_on_january": "isotonic",
    }))
    with pytest.raises(ValueError, match="hash invalid"):
        audit_run(folder, {})


def test_complete_small_fixture_roundtrip(tmp_path, monkeypatch):
    """Fit, serialize, reload, and verify on synthetic records; never real data."""
    pytest.importorskip("pyarrow")
    import payment_guard.calibration_saved_v2 as module
    from payment_guard.train_logistic_baseline import CATEGORICAL

    rng = np.random.default_rng(42)
    months = {}
    source_row = 1
    for month, key in (("2025-10", "oct"), ("2025-11", "nov"), ("2025-12", "dec"),
                       ("2026-01", "jan"), ("2026-02", "feb")):
        n = 100
        timestamps = pd.date_range(month + "-01", periods=n, freq="h")
        amount = rng.uniform(0, 1000, n)
        frame = pd.DataFrame({
            "timestamp": timestamps, "source_row": np.arange(source_row, source_row + n),
            "isFraud": (amount > 800).astype(int),
            "split": "validation" if key == "feb" else "train",
            "amount": amount, "oldbalanceOrg": rng.uniform(0, 500, n),
            "hour": timestamps.hour, "day_of_week": timestamps.dayofweek,
            "day_of_month": timestamps.day, "month": timestamps.month,
        })
        for col in CATEGORICAL:
            frame[col] = np.where(np.arange(n) % 3 == 0, "A", "B")
        months[key] = frame
        source_row += n
    windows = {
        "octnov": pd.concat([months["oct"], months["nov"]]),
        "dec": months["dec"], "jan": months["jan"],
        "octdec": pd.concat([months["oct"], months["nov"], months["dec"]]),
        "feb": months["feb"],
    }
    monkeypatch.setattr(module, "load_windows", lambda _: (windows, "toy-input"))
    output = tmp_path / "fitted_v2"
    module.run(tmp_path / "toy.parquet", output)
    assert len(list(output.glob("*.joblib"))) == 6
    module.audit_run(output, windows)
    module.compare_previous_run(output, output)
