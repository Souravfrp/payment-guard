"""Check actual field removal, comparable controls, and isolated saved runs."""
import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from payment_guard import train_logistic_baseline as baseline
from test_train_logistic_baseline import prepared  # shared prepared-data fixture


@pytest.mark.parametrize("excluded", [("type",), ("category",), ("type", "category")])
@pytest.mark.parametrize("log_amount", [False, True])
def test_removed_fields_cannot_affect_fit_or_prediction(prepared, excluded, log_amount):
    train, validation = baseline.select_periods(prepared, enforce_counts=False)
    changed = train.copy()
    for field in excluded:
        changed[field] = changed.isFraud.map({0: "target_zero", 1: "target_one"})
    first = baseline.fit_model(train, log_amount=log_amount, excluded=excluded)
    second = baseline.fit_model(changed, log_amount=log_amount, excluded=excluded)
    # Entirely absent fields must be accepted by feature construction and prediction.
    frame = baseline.design_frame(validation.drop(columns=list(excluded)),
                                  log_amount=log_amount, excluded=excluded)
    assert not set(excluded) & set(frame)
    assert not set(excluded) & set(first.feature_names_in_)
    np.testing.assert_array_equal(first.named_steps["model"].coef_, second.named_steps["model"].coef_)
    np.testing.assert_array_equal(first.predict_proba(frame), second.predict_proba(frame))
    prep = first.named_steps["preprocess"]
    expected = baseline.design_frame(train, log_amount=log_amount, excluded=excluded)
    np.testing.assert_allclose(prep.named_transformers_["scaled"].mean_, expected[baseline.SCALED].mean())


@pytest.mark.parametrize("excluded", [("amount",), ("isFraud",), ("type", "type")])
def test_unplanned_removals_rejected(excluded):
    with pytest.raises(ValueError, match="type/category"):
        baseline.make_pipeline(excluded=excluded)


def test_controls_and_tie_order_match_original_experiment(prepared):
    train, validation = baseline.select_periods(prepared, enforce_counts=False)
    old_metrics, old_predictions, _, _ = baseline.run_experiment(train, validation)
    metrics, predictions, records, models = baseline.run_experiment(train, validation, feature_ablation=True)
    assert len(models) == 8 and len(metrics) == 30
    assert len(records) == 8
    pd.testing.assert_frame_equal(old_predictions, predictions[old_predictions.columns])
    pd.testing.assert_frame_equal(old_metrics, metrics[metrics.model.isin(old_metrics.model)].reset_index(drop=True))
    for name, record in records.items():
        excluded = record["excluded_features"]
        assert not set(excluded) & set(record["source_features"])
        assert not set(excluded) & set(record["training_categories"])
        assert set(record["unknown_validation_categories"]) == set(baseline.CATEGORICAL) - set(excluded)
        assert record["converged"]


def test_ablation_cli_saved_models_hashes_and_no_overwrite(prepared, tmp_path, monkeypatch):
    source = tmp_path / "prepared.parquet"
    # Bad test rows must remain outside the experiment's validation and fitting.
    prepared.loc[prepared.split.eq("test"), "amount"] = np.nan
    prepared.loc[prepared.split.eq("test"), "isFraud"] = 999
    prepared.to_parquet(source, index=False)
    monkeypatch.setattr(baseline, "EXPECTED_COUNTS", {
        name: (len(prepared[prepared.split.eq(name)]), int(prepared.loc[prepared.split.eq(name), "isFraud"].sum()))
        for name in ("train", "validation")
    })
    output = tmp_path / "ablation"
    monkeypatch.setattr(sys, "argv", ["baseline", "--feature-ablation", "--input", str(source), "--output", str(output)])
    baseline.main()
    report = json.loads((output / "run.json").read_text())
    predictions = pd.read_parquet(output / "validation_predictions.parquet")
    assert report["complete"] and report["feature_ablation"]
    assert report["input_sha256"] == baseline.sha256(source)
    assert len(pd.read_csv(output / "validation_metrics.csv")) == 30
    validation = prepared[prepared.split.eq("validation")]
    assert set(predictions.source_row) == set(validation.source_row)
    for file, digest in report["artifacts_sha256"].items():
        assert baseline.sha256(output / file) == digest
    for name, record in report["models"].items():
        fitted = joblib.load(output / f"{name}.joblib")
        frame = baseline.design_frame(validation, log_amount=record["log_amount"], excluded=record["excluded_features"])
        np.testing.assert_allclose(fitted.predict_proba(frame)[:, 1], predictions[name])
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(SystemExit):
        baseline.main()
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
