"""Aggregate-only arithmetic and tamper tests; no AMLNet or saved models needed."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ablation_aggregate_review", ROOT/"scripts/plot_xgboost_ablation_aggregate_review.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def sources():
    base = ROOT / review.TABLE_FOLDER
    return (pd.read_csv(base/"feature_ablation_metrics.csv"),
            pd.read_csv(base/"feature_ablation_changes_vs_full.csv"))


def test_unmodified_published_aggregates_reconcile():
    review.verify_published_inputs(ROOT)
    review.audit_metrics(*sources())


def test_one_extra_or_missing_record_rejected():
    m, d = sources()
    with pytest.raises(ValueError, match="48"):
        review.audit_metrics(m.iloc[:-1], d)
    with pytest.raises(ValueError, match="48"):
        review.audit_metrics(m, pd.concat([d, d.iloc[[0]]], ignore_index=True))


def test_duplicate_budget_rejected():
    m, d = sources()
    m.loc[1, "review_fraction"] = .001
    with pytest.raises(ValueError, match="rows"):
        review.audit_metrics(m, d)


def test_negative_counts_rejected():
    m, d = sources()
    m.loc[0, "false_positives"] = -1
    d.loc[0, "false_positives"] = -1
    with pytest.raises(ValueError, match="confusion"):
        review.audit_metrics(m, d)


def test_wrong_feature_count_rejected():
    m, d = sources()
    i = m.index[m.variant.eq("without_category")][0]
    m.loc[i, "n_source_features"] = 13
    d.loc[i, "n_source_features"] = 13
    with pytest.raises(ValueError, match="feature"):
        review.audit_metrics(m, d)


def test_wrong_delta_rejected():
    m, d = sources()
    d.loc[9, "delta_true_positives_vs_full"] -= 1
    with pytest.raises(ValueError, match="delta"):
        review.audit_metrics(m, d)


def test_wrong_baseline_rejected():
    m, d = sources()
    d.loc[9, "full_average_precision"] = .5
    with pytest.raises(ValueError, match="reference"):
        review.audit_metrics(m, d)


def test_tampered_published_file_rejected(tmp_path):
    for folder in (review.TABLE_FOLDER, review.FIG_FOLDER):
        target = tmp_path/folder
        target.mkdir(parents=True)
        for name in (review.TABLE_DIGESTS if folder==review.TABLE_FOLDER else review.FIG_DIGESTS):
            shutil.copy2(ROOT/folder/name, target/name)
    review.verify_published_inputs(tmp_path)
    with (tmp_path/review.TABLE_FOLDER/"feature_ablation_metrics.csv").open("ab") as f:
        f.write(b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        review.verify_published_inputs(tmp_path)


def test_check_only_does_not_create_output(tmp_path):
    # Audit public files by using their real location and do not create an output directory.
    review.verify_published_inputs(ROOT)
    review.audit_metrics(*sources())
    assert not (tmp_path/"generated").exists()


def test_figure_regeneration_in_isolated_folder(tmp_path):
    m, d = sources()
    review.audit_metrics(m, d)
    review.plot_comparison(m, tmp_path/"captured.png", "true_positives")
    review.plot_comparison(m, tmp_path/"ap.png", "average_precision")
    assert (tmp_path/"captured.png").stat().st_size > 10000
    assert (tmp_path/"ap.png").stat().st_size > 10000
