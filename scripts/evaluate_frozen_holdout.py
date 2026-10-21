"""One-time frozen March–April holdout evaluation on synthetic AMLNet v2.

Requires the private pre-score fingerprint. Never fits models or calibrators.
Row-level predictions and fitted objects remain local under ignored models/.
Use --preflight-only to check without scoring; --score-frozen-holdout is explicit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

BUDGETS = (0.001, 0.005, 0.01)
SEED = 20261005
PRIMARY = "xgboost_uncalibrated"
CONTROL = "logistic_uncalibrated"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def freeze_module(root: Path):
    location = root / "scripts/verify_final_holdout_freeze.py"
    loader = importlib.util.spec_from_file_location("payment_guard_frozen_preflight", location)
    if loader is None or loader.loader is None:
        raise RuntimeError("Freeze preflight module not available")
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def verified_pre_score_gate(root: Path) -> tuple[dict, dict, str]:
    """Require the previously generated, immutable private pre-score fingerprint."""
    freeze = freeze_module(root)
    spec_path = root / "config/final_holdout_freeze_v1.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    freeze.validate_spec(spec)
    lock_path = root / "models/final_holdout_preflight_v1.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Private pre-score fingerprint missing; stop before scoring")
    saved = json.loads(lock_path.read_text(encoding="utf-8"))
    if saved.get("status") != "PASS_PRE_SCORE_NO_SCORING" or saved.get("march_april_scored") is not False:
        raise ValueError("Original fingerprint was not a passed no-score check")
    current = freeze.preflight(root, spec)  # hashes bytes only; no test rows or model predictions
    for field in ("freeze_spec_sha256", "prepared_parquet_sha256", "source_git_blobs",
                  "existing_run_manifest_sha256", "selected_model_and_map_sha256"):
        if saved.get(field) != current.get(field):
            raise ValueError(f"Original pre-score fingerprint changed: {field}")
    if (spec["data"]["holdout_split"] != "test" or spec["ranking"]["tie_seed"] != SEED
            or spec["ranking"]["primary_budget_fraction"] != 0.01):
        raise ValueError("Unexpected holdout split or primary endpoint")
    return spec, current, digest(lock_path)


def holdout_frame(source: Path, spec: dict, *, reader=None) -> pd.DataFrame:
    """Read ONLY the frozen test split via Parquet predicate pushdown."""
    reader = pd.read_parquet if reader is None else reader
    required = list(spec["source_features"]) + ["split", "timestamp", "source_row", "isFraud"]
    if len(required) != len(set(required)):
        raise ValueError("Duplicated input columns")
    before = source.stat() if source.is_file() else None
    frame = reader(source, engine="pyarrow", columns=required,
                   filters=[("split", "==", "test")])
    if before is not None:
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Prepared data changed while loading test rows")
    return validate_frame(frame, spec)


def validate_frame(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    if len(frame) != spec["data"]["expected_holdout_rows"]:
        raise ValueError("Holdout row count differs from frozen split")
    required = set(spec["source_features"]) | {"split", "timestamp", "source_row", "isFraud"}
    if not required.issubset(frame.columns):
        raise ValueError("Holdout feature/schema mismatch")
    if not frame["split"].eq("test").all():
        raise ValueError("Unexpected non-test rows in holdout")
    if frame["source_row"].isna().any() or not frame["source_row"].is_unique:
        raise ValueError("Missing or duplicate holdout source row")
    if frame["timestamp"].isna().any() or not pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
        raise ValueError("Missing or invalid timestamp dtype")
    if frame["timestamp"].dt.tz is not None:
        raise ValueError("Unexpected timezone-aware timestamps")
    lo = pd.Timestamp(spec["data"]["holdout_start_inclusive"])
    hi = pd.Timestamp(spec["data"]["holdout_end_exclusive"])
    if not frame["timestamp"].ge(lo).all() or not frame["timestamp"].lt(hi).all():
        raise ValueError("Holdout timestamps outside March–April")
    if set(frame["timestamp"].dt.strftime("%Y-%m").unique()) != {"2026-03", "2026-04"}:
        raise ValueError("Holdout must contain March and April")
    if frame["isFraud"].isna().any() or not frame["isFraud"].isin([0, 1]).all():
        raise ValueError("Invalid binary labels")
    if frame["isFraud"].nunique() != 2:
        raise ValueError("Cannot evaluate without both classes")
    return frame.sort_values(["timestamp", "source_row"], kind="mergesort").reset_index(drop=True)


def check_scores(scores, n: int) -> np.ndarray:
    a = np.asarray(scores, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError("Wrong score shape or nonfinite value")
    if not np.logical_and(a >= 0, a <= 1).all():
        raise ValueError("Probability outside [0,1]")
    return a


def collect_model_scores(root: Path, spec: dict, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Only previously fitted frozen models; never call fit() or change their parameters."""
    from payment_guard.train_logistic_baseline import design_frame, FEATURES
    if list(FEATURES) != spec["source_features"]:
        raise ValueError("Preprocessing feature order no longer matches the freeze")
    n, scores = len(frame), {}
    for name, entry in ((PRIMARY, spec["primary"]), (CONTROL, spec["comparator"])):
        model = joblib.load(root / entry["local_artifact"])  # trusted local, hash-checked
        scores[name] = check_scores(
            model.predict_proba(design_frame(frame, log_amount=True))[:, 1], n)
    for entry in spec["fixed_sensitivity_artifacts"]:
        name = f'{entry["family"]}_{entry["variant"]}'
        excluded = ("category",) if entry["variant"] == "without_category" else ("type", "category")
        model = joblib.load(root / entry["local_artifact"])
        scores[name] = check_scores(
            model.predict_proba(design_frame(frame, log_amount=True, excluded=excluded))[:, 1], n)
    for family in ("xgboost", "logistic"):
        raw = scores[f"{family}_uncalibrated"]
        directory = "xgboost_benchmark_v1" if family == "xgboost" else "calibration_saved_v2"
        for method in ("sigmoid", "isotonic"):
            candidate = next((p for p in spec["secondary_calibration_artifacts"]
                              if p.startswith(f"models/{directory}/")
                              and p.endswith(f"_{method}.joblib")), None)
            if candidate is None:
                raise ValueError(f"Missing frozen {family} {method} map")
            cal = joblib.load(root / candidate)
            scores[f"{family}_{method}"] = check_scores(cal.predict(raw), n)
    if len(scores) != 10:
        raise ValueError("Expected ten frozen model/score series")
    return scores


def selected_indices(scores: np.ndarray, priority: np.ndarray, budget: float) -> np.ndarray:
    n = len(scores)
    if not np.array_equal(np.sort(priority), np.arange(n)):
        raise ValueError("Invalid label-independent tie permutation")
    k = math.ceil(float(budget) * n)
    if not 1 <= k <= n:
        raise ValueError("Invalid review budget")
    return np.lexsort((priority, -scores))[:k]


def report_group(frame: pd.DataFrame, scores: dict[str, np.ndarray], scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame["isFraud"].to_numpy(dtype=int)
    n, positive = len(y), int(y.sum())
    if not 0 < positive < n:
        raise ValueError("Both binary classes required")
    ties = np.random.default_rng(SEED).permutation(n)
    metrics, selections = [], {}
    for name, raw in scores.items():
        p = check_scores(raw, n)
        common = {
            "period": scope, "series": name, "transactions": n, "fraud_labels": positive,
            "average_precision": float(average_precision_score(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y, p))
        }
        for fraction in BUDGETS:
            idx = selected_indices(p, ties, fraction)
            k, tp = len(idx), int(y[idx].sum())
            metrics.append({**common, "review_fraction": fraction, "selected": k,
                            "true_positives": tp, "false_positives": k-tp,
                            "false_negatives": positive-tp,
                            "true_negatives": n-positive-(k-tp),
                            "precision": tp/k, "recall": tp/positive})
            if name in (PRIMARY, CONTROL):
                selections[(name, fraction)] = set(idx.tolist())
    paired = []
    for fraction in BUDGETS:
        first = selections[(PRIMARY, fraction)]
        second = selections[(CONTROL, fraction)]
        shared, x_only, l_only = first & second, first-second, second-first
        both_pos = int(y[list(shared)].sum()) if shared else 0
        x_pos = int(y[list(x_only)].sum()) if x_only else 0
        l_pos = int(y[list(l_only)].sum()) if l_only else 0
        missed = positive-both_pos-x_pos-l_pos
        k = math.ceil(fraction*n)
        if missed < 0 or len(first) != k or len(second) != k:
            raise ValueError("Paired review counts do not reconcile")
        paired.append({
            "period": scope, "review_fraction": fraction, "selected_each_model": k,
            "both_caught": both_pos, "only_xgboost": x_pos, "only_logistic": l_pos,
            "neither_caught": missed, "xgboost_caught": both_pos+x_pos,
            "logistic_caught": both_pos+l_pos, "primary_delta_caught": x_pos-l_pos,
            "shared_reviewed_transactions": len(shared),
            "unique_reviews_per_model": k-len(shared)
        })
    return pd.DataFrame(metrics), pd.DataFrame(paired)


def summarize_and_plot(metrics: pd.DataFrame, overlap: pd.DataFrame, folder: Path) -> None:
    pooled = metrics.loc[metrics.period.eq("Pooled") & metrics.series.isin([PRIMARY, CONTROL])]
    x, names = np.arange(3), ["0.1%", "0.5%", "1.0%"]
    fig, ax = plt.subplots(figsize=(8.4, 5), layout="constrained")
    for key, shift, color, label in (
        (CONTROL, -.18, "#24506c", "Logistic"), (PRIMARY, .18, "#c66c27", "XGBoost")
    ):
        row = pooled.loc[pooled.series.eq(key)].set_index("review_fraction").reindex(BUDGETS)
        vals = row.true_positives.to_numpy(dtype=int)
        bars = ax.bar(x+shift, vals, width=.34, color=color, label=label)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, value+.5, str(value), ha="center", va="bottom")
    ax.set(xticks=x, xticklabels=names, ylabel="Positive-labelled transactions caught",
           title="Frozen March–April combined holdout: review budgets")
    ax.set_ylim(0, max(pooled.true_positives)*1.14)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(folder/"pooled_primary_review_budgets.png", dpi=160)
    plt.close(fig)

    rows = overlap[overlap.period.eq("Pooled")].set_index("review_fraction").reindex(BUDGETS)
    fig, ax = plt.subplots(figsize=(8.4, 5), layout="constrained")
    bottoms = np.zeros(len(rows))
    for name, label, color in (
        ("both_caught", "Both", "#4b786a"),
        ("only_xgboost", "XGBoost only", "#c66c27"),
        ("only_logistic", "Logistic only", "#24506c"),
        ("neither_caught", "Neither", "#a6a6a6"),
    ):
        vals = rows[name].to_numpy(dtype=int)
        ax.bar(x, vals, bottom=bottoms, label=label, color=color)
        for j, count in enumerate(vals):
            if count:
                ax.text(j, bottoms[j]+count/2, str(count), ha="center", va="center")
        bottoms += vals
    ax.set(xticks=x, xticklabels=names, ylabel="Positive-labelled transactions",
           title="Pooled holdout: paired fraud-capture disagreement")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(folder/"pooled_primary_disagreement.png", dpi=160)
    plt.close(fig)


def evaluate_once(root: Path, output: Path) -> None:
    if output.resolve() != (root/"models/final_holdout_v1").resolve():
        raise ValueError("Final evaluation output must be models/final_holdout_v1")
    if output.exists():
        raise FileExistsError("Immutable first holdout output already exists; refusing to rerun")
    spec, gate, lock_sha = verified_pre_score_gate(root)
    if output.exists():
        raise FileExistsError("Immutable first holdout output already exists; refusing to rerun")
    print("PASS: private pre-score lock matched; loading frozen test split", flush=True)
    frame = holdout_frame(root/spec["data"]["prepared_parquet"], spec)
    print(f"Holdout rows validated: {len(frame)}; no fitting performed.", flush=True)
    predictions = collect_model_scores(root, spec, frame)
    total_metrics, total_overlap = report_group(frame, predictions, "Pooled")
    metric_sets, overlap_sets = [total_metrics], [total_overlap]
    for period, month in (("March", "2026-03"), ("April", "2026-04")):
        mask = frame.timestamp.dt.strftime("%Y-%m").eq(month).to_numpy()
        subset = frame.loc[mask].reset_index(drop=True)
        selected = {name:p[mask] for name,p in predictions.items() if name in (PRIMARY, CONTROL)}
        m, o = report_group(subset, selected, period)
        metric_sets.append(m)
        overlap_sets.append(o)
    metrics = pd.concat(metric_sets, ignore_index=True)
    overlap = pd.concat(overlap_sets, ignore_index=True)
    main = overlap.loc[overlap.period.eq("Pooled") & np.isclose(overlap.review_fraction, .01)].iloc[0]
    summary = {
        "scope": "March–April 2026 pooled holdout", "synthetic": True,
        "primary_model": PRIMARY, "comparator": CONTROL,
        "review_fraction": .01, "review_count_each_model": int(main.selected_each_model),
        "positive_labels": int(frame.isFraud.sum()),
        "xgboost_caught": int(main.xgboost_caught),
        "logistic_caught": int(main.logistic_caught),
        "paired_xgboost_minus_logistic": int(main.primary_delta_caught),
        "categorical_feature_assumption": "category and type provisionally treated as decision-time available; unverified in real deployment",
        "interpretation": "Descriptive synthetic final holdout; no retuning or real-world operational claim",
    }
    # Score/aggregate entirely in memory before opening the immutable output folder.
    output.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(output/"metrics.csv", index=False)
    overlap.to_csv(output/"paired_overlap.csv", index=False)
    saved = frame[["source_row", "timestamp", "split", "isFraud"]].copy()
    saved["tie_priority_pooled"] = np.random.default_rng(SEED).permutation(len(frame))
    for name, p in predictions.items():
        saved[name] = p
    saved.to_parquet(output/"private_predictions.parquet", index=False)
    (output/"primary_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    summarize_and_plot(metrics, overlap, output)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True, "first_scored_holdout": True,
        "source_lock_sha256": lock_sha,
        "freeze_spec_sha256": gate["freeze_spec_sha256"],
        "prepared_parquet_sha256": gate["prepared_parquet_sha256"],
        "evaluator_python_sha256": digest(Path(__file__)),
        "manifest_sha256_before_scoring": gate["existing_run_manifest_sha256"],
        "original_model_artifact_sha256": gate["selected_model_and_map_sha256"],
        "source_code_git_blobs": gate["source_git_blobs"],
        "primary_result": summary,
        "outputs_sha256": {p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        "private_outputs_do_not_publish": ["private_predictions.parquet", "run.json"],
    }
    (output/"run.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print("PASS: first immutable March–April holdout run saved locally")
    print("PRIMARY POOLED RESULT:")
    print(json.dumps(summary, indent=2))
    print("Do not publish the private transaction-level Parquet or model artifacts.")


def verify_only(root: Path, output: Path) -> None:
    """Read-only audit of first saved outputs; DOES NOT regenerate model predictions."""
    if not output.is_dir():
        raise FileNotFoundError("No completed holdout output exists")
    spec, gate, lock_sha = verified_pre_score_gate(root)
    record = json.loads((output/"run.json").read_text(encoding="utf-8"))
    if (record.get("complete") is not True or record.get("first_scored_holdout") is not True
            or record.get("source_lock_sha256") != lock_sha
            or record.get("freeze_spec_sha256") != gate["freeze_spec_sha256"]
            or record.get("prepared_parquet_sha256") != gate["prepared_parquet_sha256"]
            or record.get("evaluator_python_sha256") != digest(Path(__file__))):
        raise ValueError("First-holdout provenance changed")
    if (record.get("original_model_artifact_sha256") != gate["selected_model_and_map_sha256"]
            or record.get("manifest_sha256_before_scoring") != gate["existing_run_manifest_sha256"]):
        raise ValueError("Saved model lineage changed")
    for name, expected in record["outputs_sha256"].items():
        if name in (".", "..") or "/" in name or "\\" in name or digest(output/name) != expected:
            raise ValueError(f"Saved output checksum changed: {name}")
    saved = pd.read_parquet(output/"private_predictions.parquet")
    if len(saved) != spec["data"]["expected_holdout_rows"] or not saved.source_row.is_unique:
        raise ValueError("Saved row identity changed")
    if not saved.split.eq("test").all() or saved.timestamp.isna().any():
        raise ValueError("Invalid saved test membership")
    if not np.array_equal(saved.tie_priority_pooled, np.random.default_rng(SEED).permutation(len(saved))):
        raise ValueError("Saved tie priority changed")
    keys = [c for c in saved.columns if c not in
            {"source_row", "timestamp", "split", "isFraud", "tie_priority_pooled"}]
    if set(keys) != {PRIMARY, CONTROL, "xgboost_without_category", "logistic_without_category",
                     "xgboost_without_both", "logistic_without_both",
                     "xgboost_sigmoid", "xgboost_isotonic",
                     "logistic_sigmoid", "logistic_isotonic"}:
        raise ValueError("Saved score series incomplete")
    m, p = report_group(saved, {k:saved[k].to_numpy() for k in keys}, "Pooled")
    all_m, all_p = [m], [p]
    for period, month in (("March", "2026-03"), ("April", "2026-04")):
        mask = saved.timestamp.dt.strftime("%Y-%m").eq(month).to_numpy()
        subset = saved.loc[mask].reset_index(drop=True)
        m, p = report_group(subset, {k:subset[k].to_numpy() for k in (PRIMARY, CONTROL)}, period)
        all_m.append(m)
        all_p.append(p)
    expected_m, expected_p = pd.concat(all_m, ignore_index=True), pd.concat(all_p, ignore_index=True)
    stored_m = pd.read_csv(output/"metrics.csv")
    stored_p = pd.read_csv(output/"paired_overlap.csv")
    def compare(actual, expected):
        if list(actual.columns) != list(expected.columns) or actual.shape != expected.shape:
            raise ValueError("Saved aggregate shape changed")
        for col in actual:
            if pd.api.types.is_numeric_dtype(actual[col]):
                np.testing.assert_allclose(actual[col], expected[col], rtol=1e-9, atol=1e-10)
            elif not actual[col].equals(expected[col]):
                raise ValueError(f"Saved aggregate labels changed: {col}")
    compare(stored_m, expected_m)
    compare(stored_p, expected_p)
    summary = json.loads((output/"primary_summary.json").read_text(encoding="utf-8"))
    if summary != record["primary_result"]:
        raise ValueError("Saved primary summary differs")
    row = expected_p.loc[expected_p.period.eq("Pooled")
                         & np.isclose(expected_p.review_fraction, 0.01)].iloc[0]
    if (int(row.primary_delta_caught) != summary["paired_xgboost_minus_logistic"]
            or int(row.xgboost_caught) != summary["xgboost_caught"]
            or int(row.logistic_caught) != summary["logistic_caught"]):
        raise ValueError("Saved primary metric changed")
    print("PASS: read-only saved hashes, lineage, paired counts and metrics; no model rescored")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root/"models/final_holdout_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight-only", action="store_true",
                         help="Verify frozen private fingerprint; do not read holdout rows")
    actions.add_argument("--score-frozen-holdout", action="store_true",
                         help="Explicit one-time March–April scoring to immutable output")
    actions.add_argument("--verify-only", action="store_true",
                         help="Reconcile first saved outputs without model re-prediction")
    args = parser.parse_args()
    if args.preflight_only:
        verified_pre_score_gate(root)
        print("PASS: frozen fingerprint still matches; no holdout rows read")
    elif args.verify_only:
        verify_only(root, output)
    else:
        evaluate_once(root, output)


if __name__ == "__main__":
    main()
