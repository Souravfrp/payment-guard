"""Small examples check leakage boundaries, ranking arithmetic and saved runs."""
import json
import math
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from payment_guard import train_logistic_baseline as baseline


@pytest.fixture
def prepared():
    times = pd.Series(list(pd.date_range("2026-01-20", periods=200, freq="h"))
                      + list(pd.date_range("2026-02-01", periods=40, freq="h"))
                      + list(pd.date_range("2026-03-01", periods=3, freq="h")))
    n = len(times)
    frame = pd.DataFrame({
        "source_row": np.arange(1, n + 1), "timestamp": times,
        "amount": (np.arange(n) % 11 + 1) * 10.0,
        "oldbalanceOrg": np.full(n, 100.0),
        "hour": times.dt.hour, "day_of_week": times.dt.dayofweek,
        "day_of_month": times.dt.day, "month": times.dt.month,
        "isFraud": (np.arange(n) % 7 == 0).astype(int),
        "split": ["train"] * 200 + ["validation"] * 40 + ["test"] * 3,
    })
    for field in baseline.CATEGORICAL:
        frame[field] = np.where(np.arange(n) % 2, "one", "two")
    return frame


def test_period_selection_excludes_test_and_sorts(prepared):
    prepared.loc[prepared.split.eq("test"), "amount"] = np.nan
    prepared.loc[prepared.split.eq("test"), "isFraud"] = 999
    train, validation = baseline.select_periods(prepared.sample(frac=1, random_state=1), enforce_counts=False)
    assert len(train) == 200 and len(validation) == 40
    assert train.source_row.is_monotonic_increasing
    assert validation.source_row.is_monotonic_increasing
    with pytest.raises(ValueError, match="Audited counts"):
        baseline.select_periods(prepared)


@pytest.mark.parametrize("column,value,message", [
    ("split", "validation", "boundary"),
    ("source_row", 2, "Source references"),
    ("amount", -1, "numerical"),
    ("amount", np.nan, "missing"),
    ("hour", 99, "Calendar"),
    ("isFraud", 2, "binary"),
])
def test_bad_training_rows_rejected(prepared, column, value, message):
    prepared.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        baseline.select_periods(prepared, enforce_counts=False)


def test_features_exclude_target_and_ids_and_change_only_amount(prepared):
    prepared["fraud_probability"] = prepared.isFraud
    prepared["newbalanceOrig"] = 123
    raw = baseline.design_frame(prepared, log_amount=False)
    logged = baseline.design_frame(prepared, log_amount=True)
    assert not set(["isFraud", "source_row", "split", "timestamp", "fraud_probability", "newbalanceOrig"]) & set(raw)
    pd.testing.assert_frame_equal(raw.drop(columns="amount"), logged.drop(columns="amount"))
    np.testing.assert_allclose(logged.amount, np.log1p(raw.amount))
    assert "month" not in raw and "month_sin" in raw


def test_scaling_and_encoding_fit_training_only(prepared):
    train, validation = baseline.select_periods(prepared, enforce_counts=False)
    validation.loc[:, "amount"] = 1e9
    validation.loc[:, "type"] = "never_seen_in_training"
    for log_amount in (False, True):
        model = baseline.fit_model(train, log_amount=log_amount)
        prep = model.named_steps["preprocess"]
        scaler = prep.named_transformers_["scaled"]
        expected = baseline.design_frame(train, log_amount=log_amount)
        np.testing.assert_allclose(scaler.mean_, expected[baseline.SCALED].mean())
        assert scaler.scale_[1] == 1.0  # constant balance is safe
        saved_mean = scaler.mean_.copy()
        assert np.isfinite(model.predict_proba(baseline.design_frame(validation, log_amount=log_amount))).all()
        np.testing.assert_array_equal(saved_mean, scaler.mean_)
        encoder = prep.named_transformers_["categorical"]
        assert "never_seen_in_training" not in encoder.categories_[0]
        encoded = encoder.transform(validation[baseline.CATEGORICAL]).toarray()
        assert (encoded[:, :len(encoder.categories_[0])] == 0).all()


def test_hand_calculated_metrics_and_exact_budgets():
    # At 1%, two rows selected: one true fraud, one false alarm; one fraud missed.
    y = np.zeros(200, dtype=int)
    y[[0, 2]] = 1
    scores = np.zeros(200)
    scores[:3] = [0.9, 0.8, 0.7]
    rows = baseline.evaluate_scores(y, scores, np.arange(200), probabilities=True)
    row = rows[-1]
    assert [r["selected"] for r in rows] == [1, 1, 2]
    assert (row["true_positives"], row["false_positives"], row["false_negatives"], row["true_negatives"]) == (1, 1, 1, 197)
    assert row["precision"] == row["recall"] == 0.5
    assert row["average_precision"] == pytest.approx((1 + 2 / 3) / 2)
    assert row["brier_score"] == pytest.approx((0.1**2 + 0.8**2 + 0.3**2) / 200)
    assert row["log_loss"] == pytest.approx(-(math.log(0.9) + math.log(0.2) + math.log(0.7)) / 200)
    assert not row["tie_crosses_cutoff"]


def test_tied_scores_use_shared_label_independent_priority():
    y = np.array([1, 0] * 100)
    ties = np.random.default_rng(baseline.SEED).permutation(200)
    scores = np.ones(200)
    row = baseline.evaluate_scores(y, scores, ties, probabilities=True)[-1]
    expected = y[np.argsort(ties)[:2]].sum()
    assert row["true_positives"] == expected
    assert row["cutoff_tie_count"] == 200 and row["selected_from_cutoff_tie"] == 2
    assert row["tie_crosses_cutoff"]
    reversed_labels = baseline.evaluate_scores(1 - y, scores, ties, probabilities=True)[-1]
    assert row["true_positives"] + reversed_labels["true_positives"] == 2
    assert baseline.evaluate_scores(y, scores, ties, probabilities=True)[-1] == row


def test_amount_ranking_has_no_probability_losses():
    rows = baseline.evaluate_scores([0, 1, 0, 1], [10, 20, 30, 40], np.arange(4), probabilities=False)
    assert all(r["log_loss"] is None and r["brier_score"] is None for r in rows)
    with pytest.raises(ValueError, match="Probabilities"):
        baseline.evaluate_scores([0, 1], [10, 20], np.arange(2), probabilities=True)


def test_convergence_warning_stops_fit(prepared, monkeypatch):
    import warnings
    def failed(*args, **kwargs):
        warnings.warn("injected convergence failure", ConvergenceWarning)
    monkeypatch.setattr(baseline.Pipeline, "fit", failed)
    with pytest.raises(RuntimeError, match="did not converge"):
        baseline.fit_model(prepared.iloc[:200], log_amount=False)


def test_cli_roundtrip_with_fixture_counts(prepared, tmp_path, monkeypatch):
    import joblib
    source = tmp_path / "prepared.parquet"
    prepared.to_parquet(source, index=False)
    output = tmp_path / "run"
    monkeypatch.setattr(baseline, "EXPECTED_COUNTS", {
        name: (len(group), int(group.isFraud.sum()))
        for name, group in prepared[prepared.split.ne("test")].groupby("split")
    })
    # Preserve chronological return ordering even if groupby orders change.
    baseline.EXPECTED_COUNTS = {k: baseline.EXPECTED_COUNTS[k] for k in ("train", "validation")}
    monkeypatch.setattr(sys, "argv", ["baseline", "--input", str(source), "--output", str(output)])
    baseline.main()
    report = json.loads((output / "run.json").read_text())
    metrics = pd.read_csv(output / "validation_metrics.csv")
    predictions = pd.read_parquet(output / "validation_predictions.parquet")
    assert report["complete"] and len(metrics) == 12 and len(predictions) == 40
    assert report["input_sha256"] == baseline.sha256(source)
    assert set(predictions.source_row) == set(prepared.loc[prepared.split.eq("validation"), "source_row"])
    for file, digest in report["artifacts_sha256"].items():
        assert baseline.sha256(output / file) == digest
    validation = prepared[prepared.split.eq("validation")]
    for name, log_amount in (("logistic_raw", False), ("logistic_log", True)):
        fitted = joblib.load(output / f"{name}.joblib")
        np.testing.assert_allclose(fitted.predict_proba(baseline.design_frame(validation, log_amount=log_amount))[:, 1], predictions[name])
    original = (output / "run.json").read_bytes()
    with pytest.raises(SystemExit):
        baseline.main()
    assert (output / "run.json").read_bytes() == original
