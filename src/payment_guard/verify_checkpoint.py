"""Read-only audit of published PaymentGuard counts and optional saved model run.

The aggregate audit uses Python's standard library. A saved-run audit additionally
uses the environment dependencies already required for baseline training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {"train": (668_820, 931), "validation": (167_990, 202), "test": (253_190, 278)}
BUDGETS = (168, 840, 1_680)
REFERENCE_MODELS = ("constant_train_rate", "amount_ranking", "logistic_raw", "logistic_log")
ALL_MODELS = REFERENCE_MODELS + (
    "logistic_raw_without_type", "logistic_log_without_type",
    "logistic_raw_without_category", "logistic_log_without_category",
    "logistic_raw_without_both", "logistic_log_without_both",
)


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise ValueError(explanation)


def saved_feature_recipe(name: str, record: dict, *, ablation: bool) -> tuple[bool, tuple[str, ...]]:
    """Resolve both Oct 6 legacy baseline and Oct 7 ablation manifest formats.

    The original two-model manifest predates excluded_features. Only full-feature
    records from that baseline are permitted to omit it; we never guess which
    predictors were removed from an ablation fit.
    """
    expected: dict[str, tuple[bool, tuple[str, ...]]] = {
        "logistic_raw": (False, ()),
        "logistic_log": (True, ()),
        "logistic_raw_without_type": (False, ("type",)),
        "logistic_log_without_type": (True, ("type",)),
        "logistic_raw_without_category": (False, ("category",)),
        "logistic_log_without_category": (True, ("category",)),
        "logistic_raw_without_both": (False, ("type", "category")),
        "logistic_log_without_both": (True, ("type", "category")),
    }
    require(name in expected, f"Unexpected fitted model in saved manifest: {name}")
    expected_log, expected_excluded = expected[name]
    actual_log = record.get("log_amount")
    require(type(actual_log) is bool and actual_log == expected_log,
            f"Saved amount transformation conflicts with fitted model name: {name}")
    if "excluded_features" in record:
        raw_excluded = record["excluded_features"]
        require(isinstance(raw_excluded, list) and all(isinstance(x, str) for x in raw_excluded),
                f"Malformed excluded_features in saved model: {name}")
        excluded = tuple(raw_excluded)
        require(excluded == expected_excluded, f"Saved feature removal conflicts with model name: {name}")
    else:
        require(not ablation and not expected_excluded,
                f"Missing excluded_features in feature-ablation model: {name}")
        excluded = ()  # Recorded October 6 baseline: both fits used all 13 source predictors.
    return expected_log, excluded


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing required file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-11)


def verify_counts(rows: list[dict[str, str]], *, label: str, models: tuple[str, ...],
                  enriched: bool) -> dict[tuple[str, int], dict[str, str]]:
    require(len(rows) == len(models) * len(BUDGETS), f"{label}: incorrect number of rows")
    known: dict[tuple[str, int], dict[str, str]] = {}
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        model = row["model"]
        k = int(row["selected"])
        key = (model, k)
        require(model in models and k in BUDGETS and key not in known, f"{label}: unexpected or duplicate {key}")
        tp, fp, fn, tn = (int(row[field]) for field in
                          ("true_positives", "false_positives", "false_negatives", "true_negatives"))
        require(min(tp, fp, fn, tn) >= 0, f"{label}: negative count for {key}")
        require(tp + fp == k and tp + fn == EXPECTED["validation"][1], f"{label}: inconsistent positives for {key}")
        require(fp + tn == EXPECTED["validation"][0] - EXPECTED["validation"][1],
                f"{label}: inconsistent negatives for {key}")
        if enriched:
            require(close(float(row["precision"]), tp / k), f"{label}: precision mismatch for {key}")
            require(close(float(row["recall"]), tp / EXPECTED["validation"][1]),
                    f"{label}: recall mismatch for {key}")
            require(0 <= float(row["average_precision"]) <= 1, f"{label}: invalid average precision {key}")
            if model == "amount_ranking":
                require(not row["log_loss"] and not row["brier_score"],
                        f"{label}: non-probability ranking has probability losses {key}")
            else:
                require(float(row["log_loss"]) >= 0 and float(row["brier_score"]) >= 0,
                        f"{label}: invalid probability loss {key}")
        known[key] = row
        by_model[model].append(row)
    require(set(known) == {(m, k) for m in models for k in BUDGETS},
            f"{label}: a model-budget combination is missing")
    for model, three in by_model.items():
        three.sort(key=lambda item: int(item["selected"]))
        caught = [int(item["true_positives"]) for item in three]
        require(caught == sorted(caught), f"{label}: caught count decreases with budget for {model}")
        if enriched:
            for metric in ("average_precision", "log_loss", "brier_score"):
                values = [item[metric] for item in three]
                require(all((not item and not values[0]) or (item and values[0] and close(float(item), float(values[0])))
                            for item in values[1:]), f"{label}: repeated {metric} differs for {model}")
    return known


def verify_aggregates(root: Path = PROJECT_ROOT) -> dict:
    tables = root / "results" / "tables"
    splits = read_csv(tables / "amlnet_v2_split_summary.csv")
    require(len(splits) == 3 and {r["split"] for r in splits} == set(EXPECTED),
            "Split summary must contain train, validation, and test exactly once")
    times = {}
    for row in splits:
        period = row["split"]
        n, p = EXPECTED[period]
        require((int(row["rows"]), int(row["positive_labels"])) == (n, p),
                f"Unexpected dataset counts in {period}")
        require(close(float(row["positive_rate"]), p / n), f"Incorrect positive rate in {period}")
        times[period] = (row["earliest"], row["latest"])
        require(row["earliest"] <= row["latest"], f"Reversed timestamps in {period}")
    require(times["train"][1] < times["validation"][0] and times["validation"][1] < times["test"][0],
            "Chronological split boundaries overlap")
    baseline = verify_counts(read_csv(tables / "logistic_baseline_review_counts.csv"),
                             label="baseline", models=REFERENCE_MODELS, enriched=False)
    ablation = verify_counts(read_csv(tables / "logistic_ablation_summary.csv"),
                             label="ablation", models=ALL_MODELS, enriched=True)
    for key, base in baseline.items():
        for field in ("true_positives", "false_positives", "false_negatives", "true_negatives"):
            require(base[field] == ablation[key][field], f"Baseline/ablation disagreement for {key}: {field}")
    require(sum(n for n, _ in EXPECTED.values()) == 1_090_000 and
            sum(p for _, p in EXPECTED.values()) == 1_411, "Whole-data count mismatch")
    return {"split_rows": 3, "baseline_rows": len(baseline), "ablation_rows": len(ablation),
            "total_transactions": 1_090_000, "total_positive_labels": 1_411}


def verify_saved_run(root: Path, run_dir: Path, prepared_path: Path) -> dict:
    """Verify saved artifact hashes, score vectors, model outputs and aggregate metrics.

    This does not refit any model and does not score the March-April holdout.
    """
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
    from payment_guard.train_logistic_baseline import design_frame, FEATURES

    require(run_dir.is_dir(), f"Saved model directory not found: {run_dir}")
    manifest_path = run_dir / "run.json"
    require(manifest_path.is_file(), f"Run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("complete") is True, "Saved run manifest is not complete")
    require(prepared_path.is_file(), f"Prepared Parquet not found: {prepared_path}")
    require(digest(prepared_path) == manifest.get("input_sha256"), "Prepared Parquet hash differs from run manifest")
    hashes = manifest.get("artifacts_sha256") or {}
    require(len(hashes) >= 4, "Saved artifact manifest is unexpectedly short")
    for filename, expected in hashes.items():
        path = run_dir / filename
        require(path.is_file() and digest(path) == expected, f"Saved artifact hash mismatch: {path}")

    ablation = bool(manifest.get("feature_ablation"))
    expected_models = ALL_MODELS if ablation else REFERENCE_MODELS
    require(set(manifest.get("models", {})) == set(expected_models) - set(REFERENCE_MODELS[:2]),
            "Saved models do not match the experiment scope")
    metrics = pd.read_csv(run_dir / "validation_metrics.csv")
    saved = pd.read_parquet(run_dir / "validation_predictions.parquet")
    cols = list(dict.fromkeys(FEATURES + ["source_row", "timestamp", "isFraud", "split"]))
    valid = pd.read_parquet(prepared_path, columns=cols, filters=[("split", "==", "validation")])
    valid = valid.sort_values("source_row").reset_index(drop=True)
    require(len(saved) == len(valid) == EXPECTED["validation"][0], "Incorrect validation row count")
    require(saved["source_row"].is_unique and valid["source_row"].is_unique,
            "Duplicated validation source identifiers")
    aligned = saved.sort_values("source_row").reset_index(drop=True)
    require(aligned["source_row"].equals(valid["source_row"]), "Source-row alignment failed")
    require(aligned["timestamp"].equals(valid["timestamp"]), "Timestamp alignment failed")
    require(aligned["isFraud"].equals(valid["isFraud"]), "Label alignment failed")
    y = aligned["isFraud"].to_numpy()
    require(int(y.sum()) == EXPECTED["validation"][1], "Unexpected number of validation positives")
    tie_order = aligned["tie_priority"].to_numpy()
    require(np.array_equal(np.sort(tie_order), np.arange(len(valid))), "Tie priority is not a permutation")

    expected_metric_rows = len(expected_models) * len(BUDGETS)
    require(len(metrics) == expected_metric_rows, "Incorrect saved metric row count")
    require(set(metrics["model"]) == set(expected_models), "Unexpected model names in saved metrics")
    summary = read_csv(root / "results" / "tables" /
                       ("logistic_ablation_summary.csv" if ablation else "logistic_baseline_review_counts.csv"))
    published = {(r["model"], int(r["selected"])): r for r in summary}
    actual = metrics.set_index(["model", "selected"])
    require(actual.index.is_unique, "Saved metrics have duplicate model-budget rows")
    require(set(actual.index) == {(m, k) for m in expected_models for k in BUDGETS},
            "Saved metrics have missing or unexpected model-budget rows")

    for name in expected_models:
        require(name in aligned.columns, f"Saved predictions missing model: {name}")
        scores = aligned[name].to_numpy(dtype=float)
        require(np.isfinite(scores).all(), f"Nonfinite scores for {name}")
        probabilistic = name != "amount_ranking"
        if probabilistic:
            require(np.logical_and(scores >= 0, scores <= 1).all(), f"Invalid probability range: {name}")
        order = np.lexsort((tie_order, -scores))
        whole_metrics = {
            "average_precision": average_precision_score(y, scores),
            "log_loss": log_loss(y, scores, labels=[0, 1]) if probabilistic else None,
            "brier_score": brier_score_loss(y, scores) if probabilistic else None,
        }
        for k in BUDGETS:
            tp = int(y[order[:k]].sum())
            predicted_counts = {"true_positives": tp, "false_positives": k - tp,
                                "false_negatives": int(y.sum()) - tp,
                                "true_negatives": len(y) - int(y.sum()) - (k - tp)}
            reported = actual.loc[(name, k)]
            reference = published[(name, k)]
            for column, count in predicted_counts.items():
                require(int(reported[column]) == count and int(reference[column]) == count,
                        f"Score/metric/publication disagreement: {name}, k={k}, {column}")
            for column, value in whole_metrics.items():
                if value is not None:
                    require(close(float(reported[column]), float(value)),
                            f"Score/metric disagreement: {name}, {column}")
        if name == "amount_ranking":
            require(np.array_equal(scores, valid["amount"].to_numpy(dtype=float)),
                    "Amount-ranking reference disagrees with transaction amounts")
        elif name == "constant_train_rate":
            p = EXPECTED["train"][1] / EXPECTED["train"][0]
            require(np.allclose(scores, p, rtol=0, atol=1e-13), "Constant-reference scores changed")
        else:
            record = manifest["models"][name]
            model_path = run_dir / f"{name}.joblib"
            require(model_path.name in hashes, f"Model hash missing: {name}")
            fitted = joblib.load(model_path)  # only trusted, locally generated project artifacts
            log_amount, excluded = saved_feature_recipe(name, record, ablation=ablation)
            features = design_frame(valid, log_amount=log_amount, excluded=excluded)
            reproduced = fitted.predict_proba(features)[:, 1]
            require(np.allclose(reproduced, scores, rtol=1e-9, atol=1e-12),
                    f"Saved fitted model cannot reproduce predictions: {name}")
    return {"run": str(run_dir), "verified_models": len(expected_models),
            "verified_validation_rows": len(valid), "verified_artifacts": len(hashes),
            "scored_holdout_rows": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--saved-run", type=Path, help="Existing local models/logistic_ablation_v1 or baseline folder")
    parser.add_argument("--prepared", type=Path, help="Existing locally prepared Parquet file")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        checks = verify_aggregates(root)
        print("PASS: Split counts, chronological boundaries, three review budgets, and 42 published rows")
        print(json.dumps(checks, indent=2))
        if args.saved_run is not None:
            run = args.saved_run if args.saved_run.is_absolute() else root / args.saved_run
            prepared = args.prepared or root / "data/processed/amlnet_v2_prepared.parquet"
            if not prepared.is_absolute():
                prepared = root / prepared
            result = verify_saved_run(root, run, prepared)
            print("PASS: Saved-run manifest, hashes, predictions, fitted models and counts")
            print(json.dumps(result, indent=2))
        else:
            print("NOTE: Aggregate audit cannot independently authenticate local model predictions.")
            print("For the full audit, add --saved-run models/logistic_ablation_v1.")
    except (ValueError, KeyError, OSError, AssertionError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")


if __name__ == "__main__":
    main()
