"""Verify published final-holdout aggregate CSVs without private model/row access.

Input: three Git-tracked aggregate files only. No model import, Parquet read, or
holdout scoring. The expected SHA-256 values are from the uploaded five-file
aggregate package audited before this publication.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

EXPECTED = {
    "metrics.csv": "540ad8c54083a0a784f1681f1d0c0fc382f5d6ef5fb01dcd10af1600596aa0ca",
    "paired_overlap.csv": "bf7a2821b855be595803fc9b7d65c27b1d2f5de051aa329092099cdd20275ff7",
    "primary_summary.json": "6d2abb6c7f792522a9d29e96d0037fbc137bd1d0a0c3f868bf404a1e478c0b0c",
}
COUNTS = {"Pooled": (253190, 278), "March": (186045, 224), "April": (67145, 54)}
BUDGETS = (0.001, 0.005, 0.01)
PRIMARY = "xgboost_uncalibrated"
CONTROL = "logistic_uncalibrated"
POOLED_SERIES = {
    "xgboost_uncalibrated", "logistic_uncalibrated",
    "xgboost_without_category", "logistic_without_category",
    "xgboost_without_both", "logistic_without_both",
    "xgboost_sigmoid", "xgboost_isotonic",
    "logistic_sigmoid", "logistic_isotonic",
}


def require(ok: bool, reason: str):
    if not ok:
        raise ValueError(reason)


def verified_input(base: Path, name: str) -> Path:
    p = base / name
    require(p.is_file(), "Missing public aggregate: " + name)
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    require(actual == EXPECTED[name], "Published aggregate hash differs: " + name)
    return p


def read_csv(p: Path):
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    require(bool(reader.fieldnames), "Empty CSV: " + p.name)
    return rows


def f(row, key):
    val = float(row[key])
    require(math.isfinite(val), "Non-finite " + key)
    return val


def integer(row, key):
    value = f(row, key)
    require(value.is_integer(), "Non-integer " + key)
    return int(value)


def audit(base: Path):
    metrics = read_csv(verified_input(base, "metrics.csv"))
    overlap = read_csv(verified_input(base, "paired_overlap.csv"))
    summary = json.loads(verified_input(base, "primary_summary.json").read_text())
    require(len(metrics) == 42 and len(overlap) == 9, "Metric/overlap row count changed")
    seen = set()
    family = {}
    for row in metrics:
        period, model = row["period"], row["series"]
        require(period in COUNTS and
                model in (POOLED_SERIES if period == "Pooled" else {PRIMARY, CONTROL}),
                "Unknown period/model")
        budget = f(row, "review_fraction")
        require(budget in BUDGETS, "Unexpected review budget")
        key = (period, model, budget)
        require(key not in seen, "Duplicate metric row")
        seen.add(key)
        n, positive = COUNTS[period]
        reviews = math.ceil(n * budget)
        tp, fp, fn, tn = (integer(row, col) for col in
                          ("true_positives", "false_positives",
                           "false_negatives", "true_negatives"))
        require(integer(row, "transactions") == n
                and integer(row, "fraud_labels") == positive
                and integer(row, "selected") == reviews, "Frozen population or workload changed")
        require(tp+fn == positive and tp+fp == reviews
                and fp+tn == n-positive and tp+fp+fn+tn == n,
                "Confusion counts disagree: " + str(key))
        require(math.isclose(f(row,"precision"),tp/reviews,abs_tol=1e-12)
                and math.isclose(f(row,"recall"),tp/positive,abs_tol=1e-12),
                "Precision/recall disagree")
        require(0 <= f(row,"average_precision") <= 1
                and f(row,"log_loss") >= 0 and f(row,"brier_score") >= 0,
                "Out-of-range ranking/loss metric")
        stats = (f(row,"average_precision"),f(row,"log_loss"),f(row,"brier_score"))
        group = (period, model)
        require(group not in family or family[group] == stats,
                "Metric repeated differently across review budgets")
        family[group] = stats
    require(len(seen) == 42 and len(family) == 14, "Incomplete fixed series/period combinations")
    indexed = {(r["period"],r["series"],f(r,"review_fraction")):r for r in metrics}
    seen_overlap=set()
    for row in overlap:
        period = row["period"]
        require(period in COUNTS, "Unknown paired period")
        budget = f(row,"review_fraction")
        require(budget in BUDGETS, "Unknown paired budget")
        key=(period,budget)
        require(key not in seen_overlap, "Duplicate paired comparison")
        seen_overlap.add(key)
        n, positive=COUNTS[period]
        k=math.ceil(n*budget)
        require(integer(row,"selected_each_model")==k, "Wrong paired workload")
        both = integer(row,"both_caught")
        xonly = integer(row,"only_xgboost")
        lonly = integer(row,"only_logistic")
        neither = integer(row,"neither_caught")
        require(both+xonly+lonly+neither == positive, "Paired positives do not partition")
        require(integer(row,"xgboost_caught")==both+xonly
                and integer(row,"logistic_caught")==both+lonly
                and integer(row,"primary_delta_caught")==xonly-lonly,
                "Paired model catches/difference disagree")
        require(integer(row,"shared_reviewed_transactions")
                +integer(row,"unique_reviews_per_model")==k,
                "Paired review lists do not reconcile")
        require(0<=integer(row,"shared_reviewed_transactions")<=k,
                "Impossible review overlap")
        require(integer(row,"xgboost_caught")==integer(indexed[period,PRIMARY,budget],"true_positives")
                and integer(row,"logistic_caught")==integer(indexed[period,CONTROL,budget],"true_positives"),
                "Paired captures disagree with metric table")
    require(len(seen_overlap)==9,"Missing paired results")
    require(summary["review_fraction"]==0.01 and
            summary["review_count_each_model"]==2532 and
            summary["positive_labels"]==278 and
            summary["primary_model"]==PRIMARY and
            summary["comparator"]==CONTROL, "Frozen primary endpoint changed")
    selected = next(r for r in overlap if r["period"]=="Pooled"
                    and f(r,"review_fraction")==0.01)
    require(summary["xgboost_caught"]==integer(selected,"xgboost_caught")
            and summary["logistic_caught"]==integer(selected,"logistic_caught")
            and summary["paired_xgboost_minus_logistic"]==integer(selected,"primary_delta_caught"),
            "Primary summary does not reconcile")
    print("PASS: three original aggregate SHA-256 digests, 42 metric rows, 9 paired rows,")
    print("      all population/ceiling/confusion/recall checks and the frozen +13 primary result.")
    print("No private transaction or fitted model was opened.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    audit(args.root/"results/tables/final_holdout_v1")


if __name__=="__main__":
    main()
