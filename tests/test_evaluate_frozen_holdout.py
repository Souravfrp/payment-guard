"""Fixture-only checks for the final holdout evaluator. No AMLNet rows read."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCATION = ROOT / "scripts/evaluate_frozen_holdout.py"
LOADER = importlib.util.spec_from_file_location("frozen_holdout_fixture", LOCATION)
assert LOADER is not None and LOADER.loader is not None
m = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(m)


def specification(n=12):
    return {
        "data": {"expected_holdout_rows": n, "holdout_split": "test",
                 "holdout_start_inclusive": "2026-03-01T00:00:00",
                 "holdout_end_exclusive": "2026-05-01T00:00:00",
                 "prepared_parquet": "data/fake.parquet"},
        "source_features": [
            "amount", "oldbalanceOrg", "hour", "day_of_week", "day_of_month",
            "month", "type", "category", "metadata.payment_method",
            "metadata.device_info.os", "metadata.device_info.type",
            "metadata.location.city", "metadata.location.state"],
        "primary": {"local_artifact": "models/xgb/B_xgboost.joblib"},
        "comparator": {"local_artifact": "models/log/B_base.joblib"},
        "fixed_sensitivity_artifacts": [
            {"family": family, "variant": variant,
             "local_artifact": f"models/abl/february_{family}_{variant}.joblib"}
            for family in ("xgboost", "logistic")
            for variant in ("without_category", "without_both")],
        "secondary_calibration_artifacts": [
            f"models/{'xgboost_benchmark_v1' if family=='xgboost' else 'calibration_saved_v2'}/B_{family}_{variant}.joblib"
            for family in ("xgboost", "logistic")
            for variant in ("sigmoid", "isotonic")],
        "ranking": {"tie_seed": 20261005, "primary_budget_fraction": 0.01},
    }


def toy_rows(n=12):
    dates = pd.date_range("2026-03-03", periods=n//2, freq="h").append(
        pd.date_range("2026-04-03", periods=n-n//2, freq="h"))
    vals = np.arange(n)
    df = pd.DataFrame({"timestamp": dates, "source_row": vals+1,
                       "split": "test", "isFraud": (vals % 3 == 0).astype(int)})
    for col in specification(n)["source_features"]:
        df[col] = "A" if col in ("category", "type") else 1.0
    return df


def test_projection_and_parquet_test_filter(tmp_path):
    source = tmp_path / "fake.parquet"
    source.write_bytes(b"fixture")
    kwargs_seen = []
    def reader(_source, **kwargs):
        kwargs_seen.append(kwargs)
        return toy_rows()
    output = m.holdout_frame(source, specification(), reader=reader)
    assert len(output) == 12
    assert kwargs_seen[0]["filters"] == [("split", "==", "test")]
    assert set(kwargs_seen[0]["columns"]) == set(toy_rows().columns)


def test_bad_holdout_membership_labels_and_duplicates_are_rejected():
    good = toy_rows()
    assert len(m.validate_frame(good, specification())) == 12
    for column, value in (
        ("split", "train"), ("timestamp", pd.Timestamp("2026-02-28")),
        ("source_row", 2), ("isFraud", 3)):
        broken = good.copy()
        broken.loc[0, column] = value
        with pytest.raises(ValueError):
            m.validate_frame(broken, specification())
    with pytest.raises(ValueError, match="row count"):
        m.validate_frame(good.iloc[:-1], specification())


def test_ceiling_and_shared_label_independent_ties():
    scores = np.array([.9, .9, .8, .7, .6])
    order = np.array([2, 0, 3, 4, 1])
    assert len(m.selected_indices(scores, order, .01)) == 1
    assert list(m.selected_indices(scores, order, .4)) == [1, 0]
    with pytest.raises(ValueError, match="permutation"):
        m.selected_indices(scores, np.zeros(5, dtype=int), .2)


def test_paired_fraud_capture_and_review_queue_overlap_reconcile():
    a = np.array([.9, .4, .3, .72, .2, .1, .88, .51, .4, .3, .1, .1])
    b = np.array([.6, .5, .4, .95, .3, .2, .9, .1, .55, .3, .2, .1])
    metric, overlap = m.report_group(toy_rows(), {m.PRIMARY: a, m.CONTROL: b}, "Pooled")
    assert len(metric) == 6 and len(overlap) == 3
    assert (overlap.both_caught + overlap.only_xgboost
            + overlap.only_logistic + overlap.neither_caught).eq(4).all()
    assert (overlap.xgboost_caught-overlap.logistic_caught).eq(
        overlap.primary_delta_caught).all()
    assert (metric.true_positives+metric.false_negatives).eq(4).all()


def test_invalid_nonprobability_and_nan_scores_rejected():
    for value in (np.nan, -.1, 1.1):
        with pytest.raises(ValueError):
            m.check_scores([.1, value], 2)


def test_private_lock_required_before_scoring(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/final_holdout_freeze_v1.json").write_text(
        json.dumps(specification()))
    monkeypatch.setattr(m, "freeze_module",
                        lambda root: types.SimpleNamespace(validate_spec=lambda s: None))
    with pytest.raises(FileNotFoundError, match="fingerprint missing"):
        m.verified_pre_score_gate(tmp_path)


def test_output_cannot_be_outside_ignored_directory_or_overwritten(tmp_path):
    folder = tmp_path / "models/final_holdout_v1"
    folder.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        m.evaluate_once(tmp_path, folder)
    with pytest.raises(ValueError, match="output must be"):
        m.evaluate_once(tmp_path, tmp_path/"results/public")


class FakeModel:
    def __init__(self, values):
        self.p = values
    def predict_proba(self, design):
        p = self.p[:len(design)]
        return np.column_stack([1-p, p])


class FakeCalibrator:
    def predict(self, raw):
        return np.clip(.75*raw+.01, 0, 1)


def test_synthetic_first_run_read_only_audit_and_corruption(tmp_path, monkeypatch):
    spec = specification()
    output = tmp_path/"models/final_holdout_v1"
    checks = {"freeze_spec_sha256": "freeze", "prepared_parquet_sha256": "source",
              "existing_run_manifest_sha256": {"manifest": "hash"},
              "selected_model_and_map_sha256": {"model": "hash"},
              "source_git_blobs": {"code": "git-hash"}}
    monkeypatch.setattr(m, "verified_pre_score_gate",
                        lambda root: (spec, checks, "original-lock"))
    monkeypatch.setattr(m, "holdout_frame", lambda path, frozen: toy_rows())
    fixture_scores = np.array([.95,.5,.3,.80,.1,.25,.9,.25,.3,.86,.2,.1])
    monkeypatch.setattr(m.joblib, "load",
        lambda p: FakeCalibrator() if "sigmoid" in str(p) or "isotonic" in str(p)
        else FakeModel(fixture_scores))
    package = types.ModuleType("payment_guard")
    training = types.ModuleType("payment_guard.train_logistic_baseline")
    training.FEATURES = spec["source_features"]
    training.design_frame = lambda df, log_amount, excluded=(): (
        df.drop(columns=[x for x in excluded if x in df]))
    monkeypatch.setitem(sys.modules, "payment_guard", package)
    monkeypatch.setitem(sys.modules, "payment_guard.train_logistic_baseline", training)
    # Fixture-only stand-in for pyarrow; real code continues to use Parquet.
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, p, **kw:self.to_pickle(p))
    monkeypatch.setattr(pd, "read_parquet", lambda p, **kw:pd.read_pickle(p))

    m.evaluate_once(tmp_path, output)
    assert json.loads((output/"run.json").read_text())["first_scored_holdout"]
    assert (output/"private_predictions.parquet").is_file()
    assert len(pd.read_csv(output/"metrics.csv")) == 42
    assert len(pd.read_csv(output/"paired_overlap.csv")) == 9
    m.verify_only(tmp_path, output)
    with pytest.raises(FileExistsError):
        m.evaluate_once(tmp_path, output)
    (output/"metrics.csv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum changed"):
        m.verify_only(tmp_path, output)
