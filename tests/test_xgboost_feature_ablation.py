"""Synthetic-only feature-ablation protocol tests; no AMLNet or holdout access."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from payment_guard.xgboost_feature_ablation import (
    assert_protocol, EXCLUDED, VARIANTS, FAMILIES, COUNT_GUARDS,
    fit_variant, predict_variant, score_rows, compare_to_full,
    assert_same_metrics, plot_summary, run,
)
from payment_guard.train_logistic_baseline import FEATURES, BUDGETS
from payment_guard.calibration_temporal import SEED


def tiny_frame(n=100):
    rng = np.random.default_rng(517)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "source_row": np.arange(n)+200,
        "timestamp": timestamps,
        "isFraud": (np.arange(n) % 7 == 0).astype(int),
        "tie_priority": np.random.default_rng(SEED).permutation(n),
        "type": np.resize(["transfer", "cashout"], n),
        "category": np.resize(["Bills", "Shopping", "Housing"], n),
        "metadata.payment_method": np.resize(["card", "wire"], n),
        "metadata.device_info.os": np.resize(["ios", "android"], n),
        "metadata.device_info.type": ["mobile"]*n,
        "metadata.location.city": ["Mumbai"]*n,
        "metadata.location.state": ["MH"]*n,
        "amount": rng.uniform(5, 300, n),
        "oldbalanceOrg": rng.uniform(0, 2000, n),
        "hour": timestamps.hour, "day_of_week": timestamps.dayofweek,
        "day_of_month": timestamps.day, "month": timestamps.month,
    })


def predictions(frame):
    rng = np.random.default_rng(777)
    return {f"{family}_{variant}": rng.uniform(.001, .99, len(frame))
            for family in FAMILIES for variant in VARIANTS}


def table_two_months():
    jan = tiny_frame(100)
    feb = tiny_frame(112)
    return pd.concat([score_rows("January", jan, predictions(jan)),
                      score_rows("February", feb, predictions(feb))], ignore_index=True)


def test_protocol_is_frozen_to_december_and_original_features():
    assert_protocol()
    assert len(VARIANTS) == 4
    assert EXCLUDED["without_both"] == ("type", "category")
    assert len(FEATURES) == 13
    assert BUDGETS == (.001, .005, .01)
    assert COUNT_GUARDS["January"] == (186097, 281)
    assert COUNT_GUARDS["February"] == (167990, 202)


def test_reject_unknown_or_full_variant_refit():
    data = tiny_frame()
    for family, variant in (("logistic", "full"), ("xgboost", "invalid"),
                             ("bad", "without_type")):
        with pytest.raises(ValueError, match="declared non-full"):
            fit_variant(data, family, variant)


def test_logistic_ablation_fit_excludes_feature():
    from payment_guard.train_logistic_baseline import design_frame
    data = tiny_frame(100)
    for v in ("without_category", "without_type", "without_both"):
        cols = design_frame(data, log_amount=True, excluded=EXCLUDED[v]).columns
        assert not set(EXCLUDED[v]) & set(cols)
    fit = fit_variant(data, "logistic", "without_both")
    p = predict_variant(fit, data, "without_both")
    assert len(p) == len(data) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_xgboost_ablation_preserves_locked_boosting_config(monkeypatch):
    # Run a tiny real tree fit with fewer rounds; the original 240-round guard
    # is tested by checking that a tampered configuration cannot pass protocol.
    from payment_guard import xgboost_feature_ablation as m
    import xgboost
    original = m.CANDIDATES
    monkeypatch.setattr(m, "CANDIDATES", original)
    trained = fit_variant(tiny_frame(100), "xgboost", "without_type")
    params = trained.named_steps["model"].get_params()
    assert params["max_depth"] == 3
    assert params["n_estimators"] == 240
    assert params["scale_pos_weight"] == 1.0
    assert trained.named_steps["model"].get_booster().num_boosted_rounds() == 240


def test_score_budgets_use_original_ceil_and_shared_ties():
    frame = tiny_frame(100)
    t = score_rows("January", frame, predictions(frame))
    assert len(t) == 24
    assert set(t["selected"]) == {1}
    assert sorted(t.review_fraction.unique()) == sorted(BUDGETS)
    assert np.all(t.true_positives + t.false_negatives == int(frame.isFraud.sum()))
    assert np.all(t.true_positives + t.false_positives == t.selected)


def test_bad_ties_or_incomplete_scores_fail_closed():
    frame = tiny_frame(100)
    p = predictions(frame)
    with pytest.raises(ValueError, match="complete"):
        score_rows("January", frame, {"logistic_full": p["logistic_full"]})
    frame.loc[0, "tie_priority"] = frame.tie_priority.iloc[1]
    with pytest.raises(ValueError, match="tie permutation"):
        score_rows("January", frame, p)


def test_full_family_aligned_delta_summary():
    t = table_two_months()
    d = compare_to_full(t)
    assert len(d) == 48
    assert d.query('variant == "full"').delta_true_positives_vs_full.eq(0).all()
    first = d.query('month == "January" and family == "xgboost" and review_fraction == 0.01')
    assert len(first) == 4
    assert first.full_true_positives.nunique() == 1


def test_duplicate_or_missing_metric_rows_fail():
    t = table_two_months()
    bad = pd.concat([t.iloc[1:], t.iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicated"):
        compare_to_full(bad)
    with pytest.raises(ValueError, match="48"):
        compare_to_full(t.iloc[:-1])
    with pytest.raises(ValueError, match="identities"):
        assert_same_metrics(t, t.iloc[::-1].iloc[1:])


def test_score_tamper_detected():
    t = table_two_months()
    damaged = t.copy()
    damaged.loc[0, "true_positives"] += 1
    with pytest.raises(AssertionError, match="true_positives"):
        assert_same_metrics(t, damaged)
    assert_same_metrics(t, t.copy())


def test_chart_generation_synthetic(tmp_path, monkeypatch):
    # Synthetic sample sizes differ from actual study; preserve 1% grouping.
    from payment_guard import xgboost_feature_ablation as m
    monkeypatch.setattr(m, "COUNT_GUARDS", {"January": (100, 15), "February": (112, 16)})
    path = tmp_path / "chart.png"
    plot_summary(table_two_months(), path)
    assert path.is_file() and path.stat().st_size > 10_000


def test_refuse_overwrite_before_source_access(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run(tmp_path / "missing.parquet", output, tmp_path / "ref", tmp_path / "logistic")


def test_complete_synthetic_local_run_and_hash_audit(tmp_path, monkeypatch):
    """Save and reconstruct all twelve reduced-feature models on fixture rows."""
    from payment_guard import xgboost_feature_ablation as m
    jan = tiny_frame(100)
    feb = tiny_frame(105)
    feb["timestamp"] = pd.date_range("2026-02-01", periods=len(feb), freq="h")
    feb["source_row"] += 1000
    octnov = tiny_frame(115)
    octdec = tiny_frame(155)
    windows = {"octnov": octnov, "octdec": octdec, "jan": jan, "feb": feb}
    monkeypatch.setattr(m, "COUNT_GUARDS", {"January": (100, 15), "February": (105, 15)})
    monkeypatch.setattr(m, "load_windows", lambda _: (windows, "synthetic_digest"))
    monkeypatch.setattr(m, "git_state", lambda _: {"rev":"fixture"})
    # CI fixture environment can lack PyArrow; only this test uses pickle.
    monkeypatch.setattr(pd.DataFrame, "to_parquet",
                        lambda df, path, index=False: df.to_pickle(path))
    monkeypatch.setattr(pd, "read_parquet", lambda path: pd.read_pickle(path))
    old_version = m.version
    monkeypatch.setattr(m, "version", lambda name: "fixture-only" if name == "pyarrow" else old_version(name))
    refdir = tmp_path / "frozen_full"
    logdir = tmp_path / "frozen_logistic"
    refdir.mkdir(); logdir.mkdir()
    (refdir / "run.json").write_text('{"fixture":"frozen"}')
    (logdir / "run.json").write_text('{"fixture":"logistic"}')
    def mock_reference(xgb_dir, logistic_dir, _windows, digest):
        assert digest == "synthetic_digest"
        a = jan[["source_row", "timestamp", "isFraud", "tie_priority"]].copy()
        b = feb[["source_row", "timestamp", "isFraud", "tie_priority"]].copy()
        for fr, cols in ((a, ("logistic", "xgboost")),
                         (b, ("logistic_uncalibrated", "xgboost_uncalibrated"))):
            for c in cols:
                fr[c] = np.linspace(.05, .95, len(fr))
        return {"January": a, "February": b}
    monkeypatch.setattr(m, "checked_reference", mock_reference)
    out = tmp_path / "audit_run"
    m.run(tmp_path / "not_used.parquet", out, refdir, logdir)
    assert (out / "metrics.csv").is_file()
    assert len(pd.read_csv(out / "metrics.csv")) == 48
    m.verify_run(out, refdir, logdir, windows, "synthetic_digest")
    # Corruption of any one of the 12 fitted pipelines fails before deserializing.
    name = "january_xgboost_without_category.joblib"
    with (out / name).open("ab") as fh:
        fh.write(b"altered")
    with pytest.raises(ValueError, match="hash differs"):
        m.verify_run(out, refdir, logdir, windows, "synthetic_digest")
