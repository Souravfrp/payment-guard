"""Offline, deterministic fixtures only. No AMLNet data and no March-April rows."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from payment_guard.xgboost_chronological import (
    CANDIDATES, COMMON, build_booster, fit_booster, probs,
    locked_candidate_choice, ranking_deciles, _assert_decile_report,
    model_report, same_metric_rows, aligned_logistic_reference, audit_saved,
)
from payment_guard.train_logistic_baseline import FEATURES, CATEGORICAL


def fixture_rows(n=72):
    rng = np.random.default_rng(456)
    dates = pd.date_range("2025-10-15", periods=n, freq="h")
    d = pd.DataFrame({
        "source_row": np.arange(1,n+1), "timestamp": dates,
        "split": "train", "isFraud": (np.arange(n)%7 == 0).astype(int),
        "type": ["transfer", "cashout"]*(n//2),
        "category": ["A", "B", "C"]*(n//3),
        "metadata.payment_method": ["card"]*n,
        "metadata.device_info.os": ["ios", "android"]*(n//2),
        "metadata.device_info.type": ["mobile"]*n,
        "metadata.location.city": ["Mumbai"]*n,
        "metadata.location.state": ["MH"]*n,
        "amount": rng.uniform(1,300,n), "oldbalanceOrg": rng.uniform(0,1000,n),
    })
    d["hour"] = d.timestamp.dt.hour
    d["day_of_week"] = d.timestamp.dt.dayofweek
    d["day_of_month"] = d.timestamp.dt.day
    d["month"] = d.timestamp.dt.month
    return d


def test_candidate_search_locked_in_order_without_january():
    tab = pd.DataFrame({"candidate":[k["name"] for k in CANDIDATES],
                        "average_precision":[.18,.32,.30],"log_loss":[.08,.09,.06]})
    assert locked_candidate_choice(tab) == CANDIDATES[1]["name"]
    tab.loc[0,"average_precision"] = .32-5e-7
    tab.loc[0,"log_loss"] = .01
    assert locked_candidate_choice(tab) == CANDIDATES[0]["name"]
    with pytest.raises(ValueError, match="incomplete"):
        locked_candidate_choice(tab.iloc[:2])


def test_candidates_are_simple_non_weighted_repeatable():
    assert len(CANDIDATES)==3
    assert COMMON["scale_pos_weight"]==1.0
    assert COMMON["tree_method"]=="hist"
    assert COMMON["random_state"]==20261005
    with pytest.raises(ValueError,match="unregistered"):
        build_booster({"name":"invalid", "max_depth":6, "n_estimators":500})


def test_fit_and_model_json_roundtrip(tmp_path):
    frame = fixture_rows()
    small = CANDIDATES[0]
    pipe = fit_booster(frame, small)
    p = probs(pipe, frame)
    assert len(p)==len(frame)
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    import joblib
    fname=tmp_path/"trusted_locally_fitted.joblib"
    joblib.dump(pipe, fname)
    np.testing.assert_allclose(probs(joblib.load(fname),frame),p,atol=0,rtol=0)
    assert pipe.named_steps["model"].get_booster().num_boosted_rounds()==small["n_estimators"]


def test_decile_uses_identical_members_for_six_variants():
    n=120
    y=(np.arange(n)%11==0).astype(int)
    base=np.linspace(.01,.95,n)
    predictions={f"{family}_{method}":np.roll(base,i)/2 for i,(family,method) in
                 enumerate((f,m) for f in ("logistic","xgboost") for m in
                           ("uncalibrated","sigmoid","isotonic"))}
    ties=np.arange(n)
    table=ranking_deciles(y,np.arange(n)+1,ties,base,predictions)
    _assert_decile_report(table,n,int(y.sum()))
    assert len(table)==60
    for decile, group in table.groupby("reference_decile"):
        assert group.positives.nunique()==1
        assert group.observed_rate.nunique()==1


def test_corrupted_probability_and_ties_rejected():
    x=np.arange(20,dtype=float)/20
    y=(np.arange(20)%3==0).astype(int)
    with pytest.raises(ValueError, match="probabilities"):
        ranking_deciles(y,np.arange(20),np.arange(20),x,{"a":np.ones(20)*1.1})
    with pytest.raises(ValueError,match="tie priority"):
        ranking_deciles(y,np.arange(20),np.zeros(20),x,{"a":x})


def test_report_is_consistent_and_change_is_detected():
    y=np.array([0,1,0,0,1,0,1,0,0,0]*5)
    p=np.linspace(.02,.98,len(y))
    record=model_report(y,{"logistic":p,"xgboost":p[::-1]},np.arange(len(y)),"B_february")
    assert len(record)==6
    same_metric_rows(record,record.copy())
    broken=record.copy()
    broken.loc[0,"true_positives"]+=1
    with pytest.raises(AssertionError,match="Metric verification"):
        same_metric_rows(record,broken)


def test_bad_logistic_reference_fails_before_reading_predictions(tmp_path):
    p=tmp_path/"ref"
    p.mkdir()
    (p/"run.json").write_text(json.dumps({"complete":True,"march_april_scored":False,"input_sha256":"wrong"}))
    feb=fixture_rows()
    with pytest.raises(ValueError,match="provenance"):
        aligned_logistic_reference(p,feb,"expected",np.arange(len(feb)))


def test_audit_incomplete_file_does_not_load_model(tmp_path):
    (tmp_path/"run.json").write_text(json.dumps({"complete":False,"march_april_scored":False}))
    with pytest.raises(ValueError,match="incomplete"):
        audit_saved(tmp_path,{},"hash")


def test_complete_synthetic_run_with_reconstruction_and_tamper_detection(tmp_path, monkeypatch):
    """Exercise write/plot/verify without a project dataset or any holdout rows.

    Local PyArrow-free check substitutes pickle for the fixture-only Parquet files.
    The actual production path remains pyarrow read_parquet with split filters.
    """
    from payment_guard import xgboost_chronological as m
    from payment_guard.calibration_temporal import dated_windows, SEED, fit_calibrators, calibrated_scores
    from payment_guard.train_logistic_baseline import fit_model, design_frame
    from payment_guard.calibration_temporal import sha256

    def one_month(first, n, offset):
        d=fixture_rows(n).copy()
        d["source_row"]+=offset
        d["timestamp"]=pd.date_range(first,periods=n,freq="h")
        d["hour"]=d.timestamp.dt.hour
        d["day_of_week"]=d.timestamp.dt.dayofweek
        d["day_of_month"]=d.timestamp.dt.day
        d["month"]=d.timestamp.dt.month
        return d
    october=one_month("2025-10-15",72,0)
    november=one_month("2025-11-15",72,72)
    december=one_month("2025-12-15",72,144)
    january=one_month("2026-01-15",72,216)
    february=one_month("2026-02-15",72,288)
    february["split"]="validation"
    train=pd.concat([october,november,december,january],ignore_index=True)
    windows=dated_windows(train,february,enforce_counts=False)
    # This single stand-in is the only point where full-data reading is faked.
    monkeypatch.setattr(m,"load_windows",lambda src:(windows,"synthetic_source_hash"))
    original_version=m.version
    monkeypatch.setattr(m,"version",lambda name: "synthetic" if name=="pyarrow" else original_version(name))
    monkeypatch.setattr(pd.DataFrame,"to_parquet",lambda self,path,index=False: self.to_pickle(path))
    monkeypatch.setattr(pd,"read_parquet",lambda path,*a,**kw: pd.read_pickle(path))
    logistic_dir=tmp_path/"local_existing_logistic_v2"
    logistic_dir.mkdir()
    model=fit_model(windows["octdec"],log_amount=True)
    def score(f): return model.predict_proba(design_frame(f,log_amount=True))[:,1]
    cal=fit_calibrators(score(windows["jan"]),windows["jan"].isFraud.to_numpy())
    pred=calibrated_scores(score(windows["feb"]),cal)
    saved=windows["feb"][["source_row","timestamp","isFraud"]].reset_index(drop=True).copy()
    saved["tie_priority"]=np.random.default_rng(SEED).permutation(len(saved))
    for method,p in pred.items():saved[method]=p
    target=logistic_dir/"B_february_predictions.parquet"
    saved.to_pickle(target)
    (logistic_dir/"run.json").write_text(json.dumps({"complete":True,"march_april_scored":False,
         "input_sha256":"synthetic_source_hash",
         "artifact_sha256":{target.name:sha256(target)}}))
    output=tmp_path/"xgb_run"
    m.run(tmp_path/"unused_source",output,logistic_dir)
    assert json.loads((output/"run.json").read_text())["march_april_scored"] is False
    assert len(pd.read_csv(output/"december_candidates.csv"))==3
    assert len(pd.read_csv(output/"metrics.csv"))==24  # 2 Jan + 6 Feb variants, x3 review budgets
    assert len(pd.read_csv(output/"february_shared_cohort_bins.csv"))==60
    imp=pd.read_csv(output/"stage_b_feature_importance.csv")
    assert len(imp)==len(FEATURES)
    assert (imp.normalized_gain>=0).all()
    for name in ("january_family_comparison.png","february_family_metrics.png",
                 "february_shared_cohort_reliability.png","february_precision_recall.png",
                 "stage_b_feature_importance.png"):
        assert (output/name).stat().st_size > 3000
    m.audit_saved(output,windows,"synthetic_source_hash",logistic_dir)
    (logistic_dir/"run.json").write_text("untrusted different manifest")
    with pytest.raises(ValueError,match="reference manifest changed"):
        m.audit_saved(output,windows,"synthetic_source_hash",logistic_dir)
    with pytest.raises(FileExistsError,match="overwrite"):
        m.run(tmp_path/"unused_source",output,logistic_dir)
    (output/"december_candidates.csv").write_text("tamper")
    with pytest.raises(ValueError,match="SHA-256"):
        m.audit_saved(output,windows,"synthetic_source_hash",logistic_dir)
