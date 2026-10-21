"""Tests for aggregate-only results publication; no model fitting or holdout data."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/plot_xgboost_aggregate_review.py"
spec=importlib.util.spec_from_file_location("xgb_aggregate_review",SCRIPT)
assert spec and spec.loader
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PUBLIC=ROOT/"results/tables/xgboost_benchmark_v1"


def sample(tmp_path):
    base=tmp_path/"input"
    base.mkdir()
    for f in mod.FILES:
        shutil.copyfile(PUBLIC/f,base/f)
    return base


def test_original_data_consistent_and_protects_holdout():
    t=mod.load_and_validate(PUBLIC)
    assert len(t["metrics.csv"])==24
    assert len(t["february_shared_cohort_bins.csv"])==60
    assert sum(t["stage_b_feature_importance.csv"].normalized_gain[:2])==pytest.approx(.8404587423)


def test_candidate_order_changed_rejected(tmp_path):
    base=sample(tmp_path)
    p=base/"december_candidates.csv"
    t=mod.load_and_validate(base)[p.name].iloc[::-1]
    t.to_csv(p,index=False)
    with pytest.raises(ValueError, match="Candidate identities"):
        mod.load_and_validate(base)


def test_candidate_choice_changed_requires_new_version(tmp_path):
    base=sample(tmp_path)
    p=base/"december_candidates.csv"
    t=mod.load_and_validate(base)[p.name]
    t.loc[0,"average_precision"]=.999
    t.to_csv(p,index=False)
    with pytest.raises(ValueError, match="December choice changed"):
        mod.load_and_validate(base)


def test_confusion_count_tampered_rejected(tmp_path):
    base=sample(tmp_path)
    p=base/"metrics.csv"
    t=mod.load_and_validate(base)[p.name]
    t.loc[0,"true_positives"]+=1
    t.to_csv(p,index=False)
    with pytest.raises(ValueError, match="Invalid selected count"):
        mod.load_and_validate(base)


def test_future_stage_rejected(tmp_path):
    base=sample(tmp_path)
    p=base/"metrics.csv"
    t=mod.load_and_validate(base)[p.name]
    t.loc[0,"stage"]="C_april"
    t.to_csv(p,index=False)
    with pytest.raises(ValueError, match="Unexpected stage"):
        mod.load_and_validate(base)


def test_mismatched_shared_cohort_rejected(tmp_path):
    base=sample(tmp_path)
    p=base/"february_shared_cohort_bins.csv"
    t=mod.load_and_validate(base)[p.name]
    t.loc[0,"positives"]+=1
    t.to_csv(p,index=False)
    with pytest.raises(ValueError, match="Matched cohort sums"):
        mod.load_and_validate(base)


def test_feature_aggregation_tampered_rejected(tmp_path):
    base=sample(tmp_path)
    p=base/"stage_b_feature_importance.csv"
    t=mod.load_and_validate(base)[p.name]
    t.loc[0,"normalized_gain"]=-0.3
    t.to_csv(p,index=False)
    with pytest.raises(ValueError, match="Importance weights"):
        mod.load_and_validate(base)


def test_plot_output_manifest_and_tamper_detection(tmp_path):
    out=tmp_path/"plots"
    mod.produce(PUBLIC,out)
    assert len(list(out.glob("*.png")))==5
    mod.check_only(PUBLIC,out)
    with pytest.raises(FileExistsError,match="Refusing to overwrite"):
        mod.produce(PUBLIC,out)
    image=next(out.glob("*.png"))
    image.write_bytes(image.read_bytes()+b"tampered")
    with pytest.raises(ValueError,match="Figure changed"):
        mod.check_only(PUBLIC,out)
