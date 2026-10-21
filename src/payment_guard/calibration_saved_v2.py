"""Versioned, locally auditable chronological calibration experiment.

Does not alter the original v1 run or examine March-April holdout rows.
Only load locally produced, SHA-256 verified joblib files that you trust.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

from .calibration_temporal import (
    BUDGETS, METHODS, SEED, _check_probabilities, calibrated_scores,
    dated_windows, fit_calibrators, plot_reliability, reliability_by_quantile,
    scoring_report, select_method, sha256,
)
from .train_logistic_baseline import FEATURES, design_frame, fit_model, git_state, select_periods

STAGES = {"A_january": "jan", "B_february": "feb"}
OBJECT_FILES = {
    "A_base": "A_base.joblib", "A_sigmoid": "A_sigmoid.joblib", "A_isotonic": "A_isotonic.joblib",
    "B_base": "B_base.joblib", "B_sigmoid": "B_sigmoid.joblib", "B_isotonic": "B_isotonic.joblib",
}


def load_windows(source: Path):
    """Only training + February validation rows may enter the experiment."""
    if not source.is_file():
        raise FileNotFoundError(f"Missing prepared data: {source}")
    before = source.stat()
    digest = sha256(source)  # Whole-file checksum only, not holdout scoring.
    data = pd.read_parquet(
        source, engine="pyarrow", columns=FEATURES + ["timestamp", "split", "source_row", "isFraud"],
        filters=[("split", "in", ["train", "validation"])],
    )
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Input file changed during read")
    train, february = select_periods(data, enforce_counts=True)
    return dated_windows(train, february, enforce_counts=True), digest


def model_probabilities(model, frame: pd.DataFrame):
    scores = model.predict_proba(design_frame(frame, log_amount=True))[:, 1]
    _check_probabilities(scores)
    return scores


def compare_saved_predictions(recomputed, saved, *, label):
    """Reject even a single misaligned source reference or changed score."""
    if set(recomputed) != set(METHODS) or set(saved) != set(METHODS):
        raise ValueError(f"{label}: invalid probability keys")
    for method in METHODS:
        actual = np.asarray(recomputed[method], dtype=float)
        expected = np.asarray(saved[method], dtype=float)
        _check_probabilities(actual)
        _check_probabilities(expected)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11,
                                   err_msg=f"{label} {method} prediction reconstruction failed")


def compare_previous_run(current_folder: Path, old_folder: Path):
    """Compare predictions, row identity, chosen method, and metric values to v1.

    The original files are read only. This is evidence of repeatability in the
    current software environment, not an independent third-party reproduction.
    """
    old_run = json.loads((old_folder / "run.json").read_text(encoding="utf-8"))
    new_run = json.loads((current_folder / "run.json").read_text(encoding="utf-8"))
    if not old_run.get("complete") or not new_run.get("complete"):
        raise ValueError("Run incomplete")
    if old_run["input_sha256"] != new_run["input_sha256"]:
        raise ValueError("Prepared input fingerprint differs from v1")
    if old_run["selected_method_on_january"] != new_run["selected_method_on_january"]:
        raise ValueError("January method selection changed compared with v1")
    for stage in STAGES:
        a = pd.read_parquet(old_folder / f"{stage}_predictions.parquet")
        b = pd.read_parquet(current_folder / f"{stage}_predictions.parquet")
        for col in ("source_row", "timestamp", "isFraud", "tie_priority"):
            if not a[col].equals(b[col]):
                raise ValueError(f"{stage}: historical row alignment changed in {col}")
        compare_saved_predictions({m: b[m] for m in METHODS}, {m: a[m] for m in METHODS}, label=f"{stage} vs v1")
    a_metrics = pd.read_csv(old_folder / "metrics.csv").sort_values(["stage", "method", "review_fraction"]).reset_index(drop=True)
    b_metrics = pd.read_csv(current_folder / "metrics.csv").sort_values(["stage", "method", "review_fraction"]).reset_index(drop=True)
    if list(a_metrics.columns) != list(b_metrics.columns) or len(a_metrics) != len(b_metrics):
        raise ValueError("Published metric structure changed from v1")
    for col in a_metrics:
        if pd.api.types.is_numeric_dtype(a_metrics[col]) and pd.api.types.is_numeric_dtype(b_metrics[col]):
            np.testing.assert_allclose(a_metrics[col], b_metrics[col], rtol=1e-7, atol=1e-9, equal_nan=True,
                                       err_msg=f"Metric changed: {col}")
        elif not a_metrics[col].equals(b_metrics[col]):
            raise ValueError(f"Metric changed: {col}")
    print("PASS: v2 reproduces the original v1 local predictions and recorded metrics.", flush=True)


def audit_run(folder: Path, windows: dict):
    """Verify artifact hashes before loading trusted local fitted objects."""
    manifest = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("march_april_scored") is not False:
        raise ValueError("Run incomplete or unexpected test-split flag")
    expected = manifest["artifact_sha256"]
    if not set(OBJECT_FILES.values()).issubset(expected):
        raise ValueError("Run manifest lacks required serialized models")
    for name, digest in expected.items():
        path = folder / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Saved artifact hash invalid: {name}")
    if manifest["selected_method_on_january"] not in METHODS:
        raise ValueError("Unrecognized chosen method")
    # Never load arbitrary serialized Python objects supplied by another person.
    objects = {key: joblib.load(folder / name) for key, name in OBJECT_FILES.items()}
    for stage, prefix in (("A_january", "A"), ("B_february", "B")):
        frame = windows[STAGES[stage]]
        saved = pd.read_parquet(folder / f"{stage}_predictions.parquet")
        if len(saved) != len(frame) or not saved.source_row.equals(frame.source_row.reset_index(drop=True)):
            raise ValueError(f"{stage}: misaligned source-row predictions")
        if not saved.timestamp.equals(frame.timestamp.reset_index(drop=True)) or not saved.isFraud.equals(frame.isFraud.reset_index(drop=True)):
            raise ValueError(f"{stage}: misaligned timestamps or labels")
        reconstructed = calibrated_scores(model_probabilities(objects[f"{prefix}_base"], frame), {
            "sigmoid": objects[f"{prefix}_sigmoid"], "isotonic": objects[f"{prefix}_isotonic"],
        })
        compare_saved_predictions(reconstructed, {m: saved[m] for m in METHODS}, label=stage)
        # Recalculate all metrics, rather than merely checking score files.
        t = saved.tie_priority.to_numpy()
        if not np.array_equal(np.sort(t), np.arange(len(t))):
            raise ValueError(f"{stage}: invalid saved tie-priority permutation")
        reconstructed_metrics = scoring_report(saved.isFraud.to_numpy(), reconstructed, t, stage=stage)
        published_metrics = pd.read_csv(folder / "metrics.csv").loc[lambda x: x.stage.eq(stage)]
        if len(reconstructed_metrics) != 3 * len(METHODS) or len(published_metrics) != len(reconstructed_metrics):
            raise ValueError(f"{stage}: incorrect number of saved metric rows")
        by = ["stage", "method", "review_fraction"]
        reconstructed_metrics = reconstructed_metrics.sort_values(by).reset_index(drop=True)
        published_metrics = published_metrics.sort_values(by).reset_index(drop=True)
        if set(reconstructed_metrics.columns) != set(published_metrics.columns):
            raise ValueError(f"{stage}: saved metric columns differ")
        for col in reconstructed_metrics:
            new = reconstructed_metrics[col]
            old = published_metrics[col]
            if pd.api.types.is_numeric_dtype(new) and pd.api.types.is_numeric_dtype(old):
                np.testing.assert_allclose(new.to_numpy(dtype=float), old.to_numpy(dtype=float),
                                           rtol=1e-7, atol=1e-9, equal_nan=True,
                                           err_msg=f"{stage}: metric reconstruction failed: {col}")
            elif not new.equals(old):
                raise ValueError(f"{stage}: saved metric disagrees: {col}")
    metric = pd.read_csv(folder / "metrics.csv")
    if len(metric) != 2 * len(METHODS) * len(BUDGETS):
        raise ValueError("Unexpected number of metric rows")
    if select_method(metric.loc[metric.stage.eq("A_january")]) != manifest["selected_method_on_january"]:
        raise ValueError("Chosen January method inconsistent with saved metrics")
    print("PASS: artifact SHA-256, aligned records, six fitted objects, reconstructed scores and metrics, and January selection.", flush=True)


def run(source: Path, output: Path, previous: Path | None = None):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing versioned run: {output}")
    print("Loading training and February validation only...", flush=True)
    windows, source_sha = load_windows(source)
    print("Stage A: fitting Oct-Nov; December calibration; January selection...", flush=True)
    a = fit_model(windows["octnov"], log_amount=True)
    cal_a = fit_calibrators(model_probabilities(a, windows["dec"]), windows["dec"].isFraud.to_numpy())
    jan = windows["jan"]
    jan_predictions = calibrated_scores(model_probabilities(a, jan), cal_a)
    jan_ties = np.random.default_rng(SEED).permutation(len(jan))
    metric_jan = scoring_report(jan.isFraud.to_numpy(), jan_predictions, jan_ties, stage="A_january")
    selected = select_method(metric_jan)
    print(f"January selection: {selected}", flush=True)
    print("Stage B: fitting Oct-Dec; January calibration; February evaluation...", flush=True)
    b = fit_model(windows["octdec"], log_amount=True)
    cal_b = fit_calibrators(model_probabilities(b, jan), jan.isFraud.to_numpy())
    feb = windows["feb"]
    feb_predictions = calibrated_scores(model_probabilities(b, feb), cal_b)
    feb_ties = np.random.default_rng(SEED).permutation(len(feb))
    metric_feb = scoring_report(feb.isFraud.to_numpy(), feb_predictions, feb_ties, stage="B_february")
    metrics = pd.concat([metric_jan, metric_feb], ignore_index=True)
    bins = pd.concat([
        reliability_by_quantile(jan.isFraud, jan_predictions, jan.source_row, stage="A_january"),
        reliability_by_quantile(feb.isFraud, feb_predictions, feb.source_row, stage="B_february"),
    ], ignore_index=True)
    objects = {"A_base": a, "A_sigmoid": cal_a["sigmoid"], "A_isotonic": cal_a["isotonic"],
               "B_base": b, "B_sigmoid": cal_b["sigmoid"], "B_isotonic": cal_b["isotonic"]}
    output.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    bins.to_csv(output / "reliability_bins.csv", index=False)
    for stage, frame, scores, ties in (("A_january", jan, jan_predictions, jan_ties),
                                       ("B_february", feb, feb_predictions, feb_ties)):
        records = frame[["source_row", "timestamp", "isFraud"]].reset_index(drop=True).copy()
        records["tie_priority"] = ties
        for method in METHODS:
            records[method] = scores[method]
        records.to_parquet(output / f"{stage}_predictions.parquet", index=False)
        plot_reliability(bins, stage=stage, path=output / f"{stage}_reliability.png")
    for key, filename in OBJECT_FILES.items():
        joblib.dump(objects[key], output / filename)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": source_sha,
        "git": git_state(Path(__file__).resolve().parents[2]),
        "selected_method_on_january": selected,
        "selection_rule": "lowest January log loss; Brier tiebreak within 1e-6; fixed simplicity order",
        "stage_a": {"fit": "2025-10 through 2025-11", "calibrate": "2025-12", "evaluate": "2026-01"},
        "stage_b": {"fit": "2025-10 through 2025-12", "calibrate": "2026-01", "evaluate": "2026-02"},
        "counts": {k: {"rows": len(windows[k]), "positives": int(windows[k].isFraud.sum())}
                   for k in ("octnov", "dec", "octdec", "jan", "feb")},
        "saved_fitted_objects": OBJECT_FILES,
        "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow")},
        "python": platform.python_version(),
        "january_selection_not_final_holdout": True,
        "march_april_scored": False,
        "caution": "Locally trusted joblib files. February explored earlier. No external validation or scored holdout.",
        "artifact_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
        "complete": True,
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    audit_run(output, windows)
    if previous is not None:
        compare_previous_run(output, previous)
    print("Completed new versioned run:", output.resolve(), flush=True)
    print("The original v1 run and March-April holdout were not changed.", flush=True)


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/processed/amlnet_v2_prepared.parquet")
    parser.add_argument("--output", type=Path, default=root / "models/calibration_saved_v2")
    parser.add_argument("--verify-only", action="store_true", help="Read-only: verify one previously created v2 run")
    parser.add_argument("--compare-v1", type=Path, help="Also compare against an existing original v1 folder")
    args = parser.parse_args(argv)
    if args.verify_only:
        windows, source_sha = load_windows(args.input)
        metadata = json.loads((args.output / "run.json").read_text(encoding="utf-8"))
        if metadata["input_sha256"] != source_sha:
            raise ValueError("Prepared data hash disagrees with saved v2 input")
        audit_run(args.output, windows)
        if args.compare_v1 is not None:
            compare_previous_run(args.output, args.compare_v1)
    else:
        run(args.input, args.output, args.compare_v1)


if __name__ == "__main__":
    main()
