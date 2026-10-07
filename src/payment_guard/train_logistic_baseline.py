"""Run logistic baselines and optional feature ablations without evaluating test rows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = [
    "type", "category", "metadata.payment_method", "metadata.device_info.os",
    "metadata.device_info.type", "metadata.location.city", "metadata.location.state",
]
NUMERICAL = ["amount", "oldbalanceOrg", "hour", "day_of_week", "day_of_month", "month"]
FEATURES = NUMERICAL + CATEGORICAL
FEATURE_REMOVALS = {
    "full": (), "without_type": ("type",),
    "without_category": ("category",), "without_both": ("type", "category"),
}
CYCLIC = {"hour": 24, "day_of_week": 7, "month": 12}
SCALED = ["amount", "oldbalanceOrg", "day_of_month"]
BUDGETS = (0.001, 0.005, 0.01)
SEED = 20261005
EXPECTED_COUNTS = {"train": (668_820, 931), "validation": (167_990, 202)}


def select_periods(data: pd.DataFrame, *, enforce_counts: bool = True):
    """Validate and return only train/validation, in reproducible source order."""
    required = FEATURES + ["isFraud", "timestamp", "source_row", "split"]
    if not data.columns.is_unique or not set(required).issubset(data.columns):
        raise ValueError("Required prepared columns are missing or duplicated.")
    # Never validate, fit on, or score test labels/features in this experiment.
    selected = data.loc[data["split"].isin(EXPECTED_COUNTS), required].copy()
    if selected.empty or selected.isna().any().any():
        raise ValueError("Training/validation data are empty or contain missing values.")
    times = selected["timestamp"]
    if not pd.api.types.is_datetime64_dtype(times.dtype):
        raise ValueError("Expected timezone-naive datetime timestamps.")
    refs = selected["source_row"]
    if not pd.api.types.is_integer_dtype(refs) or (refs <= 0).any() or not refs.is_unique:
        raise ValueError("Source references must be unique positive integers.")
    numeric = selected[NUMERICAL + ["isFraud"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (selected["amount"] < 0).any():
        raise ValueError("Invalid numerical values.")
    if not selected["isFraud"].isin([0, 1]).all():
        raise ValueError("Labels must be binary.")
    for field, attribute in {"hour": "hour", "day_of_week": "dayofweek",
                             "day_of_month": "day", "month": "month"}.items():
        if not selected[field].eq(getattr(times.dt, attribute)).all():
            raise ValueError(f"Calendar disagrees with timestamp: {field}")
    for field in CATEGORICAL:
        if not selected[field].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"Expected nonempty strings: {field}")
    result = []
    for name, (row_count, positives) in EXPECTED_COUNTS.items():
        frame = selected.loc[selected["split"].eq(name)].sort_values("source_row").reset_index(drop=True)
        if frame.empty or frame["isFraud"].nunique() != 2:
            raise ValueError(f"Both label classes are required in {name}.")
        t = frame["timestamp"]
        valid = t.lt("2026-02-01") if name == "train" else (t.ge("2026-02-01") & t.lt("2026-03-01"))
        if not valid.all():
            raise ValueError(f"Chronological boundary disagreement in {name}.")
        if enforce_counts and (len(frame), int(frame["isFraud"].sum())) != (row_count, positives):
            raise ValueError(f"Audited counts disagree in {name}.")
        result.append(frame)
    return tuple(result)


def retained_features(excluded=()):
    """Restrict ablations to the two fields specified in the experiment plan."""
    excluded = tuple(excluded)
    if len(set(excluded)) != len(excluded) or not set(excluded) <= {"type", "category"}:
        raise ValueError("Only unique type/category removals are supported.")
    return [field for field in FEATURES if field not in excluded]


def design_frame(data: pd.DataFrame, *, log_amount: bool, excluded=()) -> pd.DataFrame:
    """Fixed, label-free feature construction; no fitted state or hidden columns."""
    result = data[retained_features(excluded)].copy()
    if log_amount:
        result["amount"] = np.log1p(result["amount"])
    for name, period in CYCLIC.items():
        angle = 2 * np.pi * result.pop(name).astype(float) / period
        result[name + "_sin"] = np.sin(angle)
        result[name + "_cos"] = np.cos(angle)
    return result


def make_pipeline(*, max_iter: int = 2000, excluded=()) -> Pipeline:
    retained = retained_features(excluded)
    categorical = [field for field in CATEGORICAL if field in retained]
    cyclic_columns = [name + suffix for name in CYCLIC for suffix in ("_sin", "_cos")]
    preprocessing = ColumnTransformer([
        ("scaled", StandardScaler(), SCALED),
        ("cyclic", "passthrough", cyclic_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical),
    ], remainder="drop", sparse_threshold=1.0)
    # L2 is the LogisticRegression default across supported sklearn versions.
    # Omitting the deprecated penalty argument avoids version-specific warnings.
    model = LogisticRegression(C=1.0, solver="lbfgs", fit_intercept=True,
                               class_weight=None, max_iter=max_iter, tol=1e-6)
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def fit_model(train: pd.DataFrame, *, log_amount: bool, max_iter: int = 2000, excluded=()):
    pipeline = make_pipeline(max_iter=max_iter, excluded=excluded)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            pipeline.fit(design_frame(train, log_amount=log_amount, excluded=excluded), train["isFraud"])
        except ConvergenceWarning as error:
            raise RuntimeError("Logistic fit did not converge. No completed run was saved; inspect before retrying.") from error
    if not np.isfinite(pipeline.named_steps["model"].coef_).all():
        raise RuntimeError("Nonfinite model coefficients.")
    return pipeline


def evaluate_scores(labels, scores, tie_priority, *, probabilities: bool):
    """Rank by score with a shared label-independent permutation for exact ties."""
    y = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    ties = np.asarray(tie_priority)
    n = len(y)
    if y.ndim != 1 or scores.shape != y.shape or ties.shape != y.shape or n == 0:
        raise ValueError("Evaluation arrays must be aligned, nonempty vectors.")
    if not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Expected finite scores and both binary label classes.")
    if not np.array_equal(np.sort(ties), np.arange(n)):
        raise ValueError("Tie priority must be a permutation of row positions.")
    if probabilities and ((scores < 0).any() or (scores > 1).any()):
        raise ValueError("Probabilities must be between zero and one.")
    order = np.lexsort((ties, -scores))
    positives = int(y.sum())
    summary = {
        "average_precision": float(average_precision_score(y, scores)),
        "log_loss": float(log_loss(y, scores, labels=[0, 1])) if probabilities else None,
        "brier_score": float(brier_score_loss(y, scores)) if probabilities else None,
    }
    rows = []
    for fraction in BUDGETS:
        k = math.ceil(fraction * n)
        tp = int(y[order[:k]].sum())
        cutoff = scores[order[k - 1]]
        selected_ties = int(np.count_nonzero(scores[order[:k]] == cutoff))
        total_ties = int(np.count_nonzero(scores == cutoff))
        rows.append({
            **summary, "review_fraction": fraction, "selected": k,
            "actual_review_fraction": k / n, "true_positives": tp,
            "false_positives": k - tp, "false_negatives": positives - tp,
            "true_negatives": n - positives - (k - tp),
            "precision": tp / k, "recall": tp / positives,
            "cutoff_score": float(cutoff), "cutoff_tie_count": total_ties,
            "selected_from_cutoff_tie": selected_ties,
            "tie_crosses_cutoff": total_ties > selected_ties,
        })
    return rows


def run_experiment(train: pd.DataFrame, validation: pd.DataFrame, *, max_iter: int = 2000,
                   feature_ablation: bool = False):
    """Two full-feature fits, optionally six removals, plus two references."""
    # select_periods supplies a stable source-row order before this permutation.
    ties = np.random.default_rng(SEED).permutation(len(validation))
    y = validation["isFraud"].to_numpy()
    predictions = validation[["source_row", "timestamp", "isFraud"]].copy()
    predictions["tie_priority"] = ties
    metrics, models, records = [], {}, {}
    references = {
        "constant_train_rate": (np.full(len(validation), train["isFraud"].mean()), True),
        "amount_ranking": (validation["amount"].to_numpy(dtype=float), False),
    }
    for name, (scores, probability) in references.items():
        predictions[name] = scores
        metrics.extend({"model": name, **row} for row in evaluate_scores(y, scores, ties, probabilities=probability))
    groups = FEATURE_REMOVALS if feature_ablation else {"full": ()}
    variants = [(f"logistic_{transform}" + ("" if group == "full" else f"_{group}"),
                 log_amount, group, excluded)
                for group, excluded in groups.items()
                for transform, log_amount in (("raw", False), ("log", True))]
    for name, log_amount, group, excluded in variants:
        categorical = [field for field in CATEGORICAL if field not in excluded]
        print(f"Fitting {name} on {len(train):,} training rows...", flush=True)
        started = perf_counter()
        fitted = fit_model(train, log_amount=log_amount, max_iter=max_iter, excluded=excluded)
        fit_seconds = perf_counter() - started
        started = perf_counter()
        scores = fitted.predict_proba(design_frame(validation, log_amount=log_amount, excluded=excluded))[:, 1]
        score_seconds = perf_counter() - started
        metrics.extend({"model": name, **row} for row in evaluate_scores(y, scores, ties, probabilities=True))
        predictions[name] = scores
        preprocess = fitted.named_steps["preprocess"]
        encoder = preprocess.named_transformers_["categorical"]
        scaler = preprocess.named_transformers_["scaled"]
        unknown = {field: int((~validation[field].isin(categories)).sum())
                   for field, categories in zip(categorical, encoder.categories_)}
        model = fitted.named_steps["model"]
        records[name] = {
            "feature_group": group, "excluded_features": list(excluded),
            "source_features": retained_features(excluded),
            "log_amount": log_amount, "feature_names": preprocess.get_feature_names_out().tolist(),
            "unknown_validation_categories": unknown,
            "training_categories": {field: values.tolist() for field, values in zip(categorical, encoder.categories_)},
            "scaler_columns": SCALED, "training_means": scaler.mean_.tolist(),
            "training_scales": scaler.scale_.tolist(), "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]), "iterations": model.n_iter_.tolist(),
            "converged": True, "fit_seconds": fit_seconds, "score_seconds": score_seconds,
            "settings": {"solver": "lbfgs", "regularization": "L2", "C": 1.0,
                         "tol": 1e-6, "max_iter": max_iter, "class_weight": None},
        }
        models[name] = fitted
    return pd.DataFrame(metrics), predictions, records, models


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_state(root: Path):
    def command(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    revision, status = command("rev-parse", "HEAD"), command("status", "--porcelain")
    return {"revision": revision, "working_tree_dirty": bool(status) if status is not None else None}


def save_run(output: Path, metrics, predictions, records, models, metadata):
    """Reserve a new run folder; a completion manifest is written last."""
    import joblib

    output.mkdir(parents=True, exist_ok=False)
    # Interrupted/failed directories are retained for inspection, never reused.
    metrics.to_csv(output / "validation_metrics.csv", index=False)
    predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    for name, model in models.items():
        joblib.dump(model, output / f"{name}.joblib")
    paths = sorted(path for path in output.iterdir() if path.is_file())
    report = {**metadata, "models": records,
              "artifacts_sha256": {path.name: sha256(path) for path in paths}, "complete": True}
    (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/processed/amlnet_v2_prepared.parquet")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--feature-ablation", action="store_true",
                        help="Refit two full-feature controls and six type/category removal variants.")
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()
    if args.output is None:
        args.output = root / "models" / ("logistic_ablation_v1" if args.feature_ablation else "logistic_baseline_v1")
    if args.max_iter < 1:
        parser.error("--max-iter must be positive")
    if args.output.exists():
        parser.error("Output already exists. Choose a new --output directory; existing runs are not overwritten.")
    started = perf_counter()
    print("Reading training and validation rows only...", flush=True)
    before = args.input.stat()
    fingerprint = sha256(args.input)
    data = pd.read_parquet(args.input, engine="pyarrow", columns=FEATURES + ["isFraud", "timestamp", "source_row", "split"],
                           filters=[("split", "in", ["train", "validation"])])
    after = args.input.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Input changed while being read.")
    train, validation = select_periods(data)
    del data
    print(f"Train: {len(train):,} rows; validation: {len(validation):,} rows. Test rows excluded.", flush=True)
    metrics, predictions, records, models = run_experiment(
        train, validation, max_iter=args.max_iter, feature_ablation=args.feature_ablation)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "input_sha256": fingerprint,
        "input_bytes": before.st_size, "git": git_state(root), "features": FEATURES,
        "seed": SEED, "budgets": BUDGETS, "primary_review_fraction": 0.01,
        "counts": {name: {"rows": len(frame), "positives": int(frame.isFraud.sum())}
                   for name, frame in (("train", train), ("validation", validation))},
        "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "joblib")},
        "python": platform.python_version(), "platform": platform.platform(),
        "elapsed_before_save_seconds": perf_counter() - started,
        "scope": ("Eight logistic fits: two full-feature controls and six feature removals; no test evaluation."
                  if args.feature_ablation else "First two full-feature logistic fits; feature removals pending; no test evaluation."),
        "feature_ablation": args.feature_ablation,
        "ranking_note": "Retrospective batch top-k, not a real-time decision threshold.",
    }
    save_run(args.output, metrics, predictions, records, models, metadata)
    print("\nValidation at the predeclared 1% review budget:")
    print(metrics.loc[metrics.review_fraction.eq(0.01),
                      ["model", "selected", "true_positives", "false_positives", "precision", "recall"]].to_string(index=False))
    print(f"\nSaved completed run: {args.output.resolve()}")


if __name__ == "__main__":
    main()
