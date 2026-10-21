"""Small synthetic checks for missed-case score reconstructions; no full dataset."""
import numpy as np
import pandas as pd
import pytest

from payment_guard.train_logistic_baseline import CATEGORICAL, make_pipeline, design_frame
from scripts import analyze_missed_fraud_contributions as study


def make_rows(count, start, source_offset=0):
    times = pd.date_range(start, periods=count, freq="h")
    frame = pd.DataFrame({
        "source_row": np.arange(source_offset + 1, source_offset + count + 1, dtype=int),
        "timestamp": times,
        "amount": np.linspace(20, 4500, count) * (1 + 0.05 * np.sin(np.arange(count))),
        "oldbalanceOrg": np.linspace(100, 9000, count),
        "hour": times.hour,
        "day_of_week": times.dayofweek,
        "day_of_month": times.day,
        "month": times.month,
        "isFraud": (np.arange(count) % 5 == 0).astype(int),
        "split": "train" if start.startswith("2026-01") else "validation",
    })
    for field in CATEGORICAL:
        frame[field] = "constant"
    frame["type"] = "TRANSFER"
    frame["category"] = "Other"
    return frame


@pytest.fixture
def fitted_and_validation():
    train = make_rows(90, "2026-01-05 00:00:00")
    val = make_rows(40, "2026-02-01 00:00:00", source_offset=1000)
    model = make_pipeline()
    model.fit(design_frame(train, log_amount=True), train.isFraud)
    scores = model.predict_proba(design_frame(val, log_amount=True))[:, 1]
    # Construct known positives on both sides of a fixed review budget. Labels
    # are for the synthetic test only; they do not change fitted predictions.
    order = np.argsort(-scores, kind="stable")
    val["isFraud"] = 0
    val.loc[order[[0, 1, 2, 3, 17, 26]], "isFraud"] = 1
    pred = val[["source_row", "timestamp", "isFraud"]].copy()
    pred["tie_priority"] = np.arange(len(val))
    pred[study.MODEL] = scores
    return model, val, pred


def test_grouping_reproduces_every_score(fitted_and_validation):
    model, val, _ = fitted_and_validation
    intercept, grouped, log_odds, probs = study.grouped_contributions(model, val.iloc[:8])
    assert len(grouped.columns) == 13
    np.testing.assert_allclose(intercept + grouped.sum(axis=1), log_odds)
    np.testing.assert_allclose(probs, model.predict_proba(design_frame(val.iloc[:8], log_amount=True))[:, 1])


def test_ranking_matches_and_log_odds_gaps(fitted_and_validation):
    model, val, pred = fitted_and_validation
    ranked, missed, caught = study.prepare_ranked_validation(val, pred, expected_n=40, expected_p=6,
                                                             budget=8, expected_caught=4)
    assert len(missed) == 2 and len(caught) == 4
    pairs, details, deltas, cutoff = study.build_analysis(ranked, missed, caught, model, budget=8)
    assert len(pairs) == len(deltas) == 2
    assert len(details) == 4
    assert (pairs["missed_score"] < pairs["caught_score"]).all()
    assert (pairs["delta_log_odds_missed_minus_caught"] < 0).all()
    np.testing.assert_allclose(deltas.drop(columns="missed_source_row").sum(axis=1),
                               pairs.delta_log_odds_missed_minus_caught)
    assert np.isfinite(cutoff)


def test_rejects_holdout_row(fitted_and_validation):
    _, val, pred = fitted_and_validation
    val.loc[val.index[0], "split"] = "test"
    with pytest.raises(ValueError, match="Nonvalidation"):
        study.prepare_ranked_validation(val, pred, expected_n=40, expected_p=6,
                                        budget=8, expected_caught=4)


def test_rejects_label_disagreement(fitted_and_validation):
    _, val, pred = fitted_and_validation
    pred.loc[pred.index[0], "isFraud"] = 1 - pred.loc[pred.index[0], "isFraud"]
    with pytest.raises(ValueError, match="label disagreement"):
        study.prepare_ranked_validation(val, pred, expected_n=40, expected_p=6,
                                        budget=8, expected_caught=4)


def test_missing_same_group_comparator_is_reported(fitted_and_validation):
    model, val, pred = fitted_and_validation
    ranked, missed, caught = study.prepare_ranked_validation(val, pred, expected_n=40, expected_p=6,
                                                             budget=8, expected_caught=4)
    caught.loc[:, "category"] = "Housing"
    with pytest.raises(ValueError, match="No same-type/category"):
        study.select_matched_caught(missed, caught, model)


def test_figure_generation(fitted_and_validation, tmp_path):
    model, val, pred = fitted_and_validation
    ranked, missed, caught = study.prepare_ranked_validation(val, pred, expected_n=40, expected_p=6,
                                                             budget=8, expected_caught=4)
    pairs, _, delta, cutoff = study.build_analysis(ranked, missed, caught, model, budget=8)
    study.draw_figures(pairs, delta, cutoff, tmp_path)
    assert (tmp_path / "missed_fraud_score_comparison.png").stat().st_size > 1000
    assert (tmp_path / "missed_fraud_feature_differences.png").stat().st_size > 1000
