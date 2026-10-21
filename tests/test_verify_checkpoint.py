"""No-dataset tests for consistency auditing and failure diagnostics."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from payment_guard import verify_checkpoint as audit


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_tables(tmp_path: Path) -> Path:
    base = tmp_path / "results" / "tables"
    splits = []
    for index, (part, (n, positives)) in enumerate(audit.EXPECTED.items()):
        splits.append({
            "split": part, "rows": n, "positive_labels": positives,
            "positive_rate": positives / n,
            "earliest": f"2026-0{index + 1}-01T00:00:00",
            "latest": f"2026-0{index + 1}-28T23:59:59",
        })
    write_rows(base / "amlnet_v2_split_summary.csv", splits)
    baseline, ablation = [], []
    for model in audit.ALL_MODELS:
        for reviews in audit.BUDGETS:
            caught = {168: 1, 840: 10, 1680: 20}[reviews]
            counts = {
                "model": model, "selected": reviews, "true_positives": caught,
                "false_positives": reviews - caught,
                "false_negatives": 202 - caught,
                "true_negatives": 167_990 - 202 - (reviews - caught),
            }
            if model in audit.REFERENCE_MODELS:
                baseline.append(counts)
            ablation.append({
                **counts, "precision": caught / reviews, "recall": caught / 202,
                "average_precision": 0.2,
                "log_loss": "" if model == "amount_ranking" else 0.01,
                "brier_score": "" if model == "amount_ranking" else 0.002,
            })
    write_rows(base / "logistic_baseline_review_counts.csv", baseline)
    write_rows(base / "logistic_ablation_summary.csv", ablation)
    return tmp_path


def test_published_counts_reconcile_without_full_dataset(tmp_path):
    report = audit.verify_aggregates(make_tables(tmp_path))
    assert report["baseline_rows"] == 12
    assert report["ablation_rows"] == 30
    assert report["total_positive_labels"] == 1_411


def test_detects_changed_fraud_count(tmp_path):
    root = make_tables(tmp_path)
    path = root / "results/tables/logistic_ablation_summary.csv"
    rows = audit.read_csv(path)
    rows[3]["true_positives"] = "9"
    write_rows(path, rows)
    with pytest.raises(ValueError, match="inconsistent positives"):
        audit.verify_aggregates(root)


def test_detects_inconsistent_precision(tmp_path):
    root = make_tables(tmp_path)
    path = root / "results/tables/logistic_ablation_summary.csv"
    rows = audit.read_csv(path)
    rows[0]["precision"] = "0.99"
    write_rows(path, rows)
    with pytest.raises(ValueError, match="precision mismatch"):
        audit.verify_aggregates(root)


def test_detects_boundary_overlap(tmp_path):
    root = make_tables(tmp_path)
    path = root / "results/tables/amlnet_v2_split_summary.csv"
    rows = audit.read_csv(path)
    rows[0]["latest"] = rows[1]["earliest"]
    write_rows(path, rows)
    with pytest.raises(ValueError, match="overlap"):
        audit.verify_aggregates(root)


def test_detects_changed_baseline_comparison(tmp_path):
    root = make_tables(tmp_path)
    path = root / "results/tables/logistic_baseline_review_counts.csv"
    rows = audit.read_csv(path)
    rows[0]["true_positives"] = "2"
    rows[0]["false_positives"] = "166"
    rows[0]["false_negatives"] = "200"
    rows[0]["true_negatives"] = str(167_788 - 166)
    write_rows(path, rows)
    with pytest.raises(ValueError, match="disagreement"):
        audit.verify_aggregates(root)


@pytest.mark.parametrize("name,log_amount", [
    ("logistic_raw", False),
    ("logistic_log", True),
])
def test_october_6_baseline_manifest_without_removed_features(name, log_amount):
    # Original baseline run.json only recorded log_amount; it predates feature ablation.
    assert audit.saved_feature_recipe(name, {"log_amount": log_amount}, ablation=False) == (log_amount, ())


@pytest.mark.parametrize("name,log_amount,removed", [
    ("logistic_raw_without_type", False, ["type"]),
    ("logistic_log_without_category", True, ["category"]),
    ("logistic_log_without_both", True, ["type", "category"]),
])
def test_october_7_ablation_manifest(name, log_amount, removed):
    assert audit.saved_feature_recipe(name, {
        "log_amount": log_amount, "excluded_features": removed,
    }, ablation=True) == (log_amount, tuple(removed))


def test_ablation_manifest_does_not_silently_assume_missing_features():
    with pytest.raises(ValueError, match="Missing excluded_features"):
        audit.saved_feature_recipe("logistic_log_without_type", {"log_amount": True}, ablation=True)


def test_manifest_rejects_wrong_transformation_or_field_removals():
    with pytest.raises(ValueError, match="amount transformation"):
        audit.saved_feature_recipe("logistic_log", {"log_amount": False}, ablation=False)
    with pytest.raises(ValueError, match="feature removal"):
        audit.saved_feature_recipe("logistic_log_without_type", {
            "log_amount": True, "excluded_features": ["category"],
        }, ablation=True)
