"""Small-fixture tests: no project Parquet, original saved models, or holdout needed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from payment_guard.calibration_temporal import (
    METHODS, PositiveSigmoid, calibrated_scores, dated_windows,
    fit_calibrators, reliability_by_quantile, select_method,
)


def tiny_periods():
    dates = ["2025-10-15", "2025-10-20", "2025-11-03", "2025-11-18",
             "2025-12-03", "2025-12-24", "2026-01-05", "2026-01-21"]
    train = pd.DataFrame({"split": "train", "timestamp": pd.to_datetime(dates),
                          "source_row": np.arange(1, 9), "isFraud": [1, 0, 0, 1, 0, 1, 1, 0]})
    feb = pd.DataFrame({"split": ["validation", "validation"],
                        "timestamp": pd.to_datetime(["2026-02-03", "2026-02-18"]),
                        "source_row": [9, 10], "isFraud": [1, 0]})
    return train, feb


def test_stage_windows_are_forward_only_and_disjoint():
    train, feb = tiny_periods()
    w = dated_windows(train, feb, enforce_counts=False)
    assert set(w) == {"octnov", "dec", "jan", "octdec", "feb"}
    assert list(w["octnov"].source_row) == [1, 2, 3, 4]
    assert list(w["octdec"].source_row) == [1, 2, 3, 4, 5, 6]
    assert list(w["jan"].source_row) == [7, 8]
    assert w["dec"].timestamp.max() < w["jan"].timestamp.min()
    assert w["jan"].timestamp.max() < w["feb"].timestamp.min()


def test_strict_full_month_counts_reject_toy_data():
    train, feb = tiny_periods()
    with pytest.raises(ValueError, match="Training month"):
        dated_windows(train, feb, enforce_counts=True)


def test_stage_checks_disallow_holdout_and_overlaps():
    train, feb = tiny_periods()
    changed = feb.copy()
    changed.loc[0, "timestamp"] = pd.Timestamp("2026-03-01")
    with pytest.raises(ValueError, match="February"):
        dated_windows(train, changed, enforce_counts=False)
    changed = feb.copy()
    changed.loc[0, "source_row"] = 1
    with pytest.raises(ValueError, match="overlap"):
        dated_windows(train, changed, enforce_counts=False)
    changed_train = train.copy()
    changed_train.loc[0, "timestamp"] = pd.Timestamp("2026-04-01")
    with pytest.raises(ValueError, match="months"):
        dated_windows(changed_train, feb, enforce_counts=False)


def synthetic_calibration():
    rng = np.random.default_rng(20261021)
    p = np.linspace(0.005, 0.8, 400)
    y = rng.binomial(1, np.sqrt(p) / 2).astype(int)
    assert len(np.unique(y)) == 2
    return p, y


def test_monotone_calibrators_produce_valid_probabilities():
    p, y = synthetic_calibration()
    fitted = fit_calibrators(p, y)
    result = calibrated_scores(p, fitted)
    assert list(result) == list(METHODS)
    for values in result.values():
        assert values.shape == p.shape
        assert np.isfinite(values).all()
        assert np.all((0 <= values) & (values <= 1))
        assert np.all(np.diff(values) >= -1e-12)
    assert fitted["sigmoid"].slope >= 0


def test_sigmoid_requires_two_classes_and_fit_before_predict():
    with pytest.raises(ValueError, match="not fitted"):
        PositiveSigmoid().predict(np.array([0.5]))
    with pytest.raises(ValueError, match="both classes"):
        PositiveSigmoid().fit(np.array([0.1, 0.2]), np.array([0, 0]))
    with pytest.raises(ValueError, match="probabilities"):
        fit_calibrators(np.array([0.2, 1.3]), np.array([0, 1]))


def test_reliability_preserves_all_cases_and_tied_scores():
    p = np.array([0.02, 0.02, 0.02, 0.3, 0.3, 0.8])
    y = np.array([0, 0, 1, 0, 0, 1])
    scores = {method: p for method in METHODS}
    table = reliability_by_quantile(y, scores, np.arange(6) + 1, stage="toy")
    for method in METHODS:
        rows = table.loc[table.method.eq(method)]
        assert rows.rows.sum() == 6
        assert rows.positives.sum() == 2
        # Repeated score values must not be split into separate quantile bins.
        assert (rows.rows >= 1).all()
        assert len(rows) <= len(np.unique(p))


def test_january_log_loss_selects_method_before_february():
    rows = []
    for method, loss, brier in (("uncalibrated", 0.25, 0.13),
                                ("sigmoid", 0.19, 0.14),
                                ("isotonic", 0.20, 0.10)):
        rows.extend({"method": method, "review_fraction": budget,
                     "log_loss": loss, "brier_score": brier} for budget in (0.001, 0.005, 0.01))
    assert select_method(pd.DataFrame(rows)) == "sigmoid"
    changed = pd.DataFrame(rows)
    changed.loc[changed.method.eq("isotonic"), "log_loss"] = 0.1900005
    assert select_method(changed) == "isotonic"  # near-tie resolved using Brier
