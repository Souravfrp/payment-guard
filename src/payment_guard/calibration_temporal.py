"""Forward-only probability calibration; reads training and February validation only.

Designed to compare an uncalibrated logistic model with a monotone sigmoid
mapping and isotonic regression, not to optimize top-k detection on February.
This command never loads March-April observations, existing model artifacts,
or pre-existing calibration output directories.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

METHODS = ("uncalibrated", "sigmoid", "isotonic")
EXPECTED_MONTHS = {
    "2025-10": (120879, 226),
    "2025-11": (171789, 226),
    "2025-12": (190055, 198),
    "2026-01": (186097, 281),
}
SEED = 20261005
BUDGETS = (0.001, 0.005, 0.01)


def dated_windows(train: pd.DataFrame, validation: pd.DataFrame, *, enforce_counts: bool = True):
    """Return chronological, nonoverlapping frames; forbid any out-of-scope month."""
    if train.empty or validation.empty:
        raise ValueError("Empty training or validation set")
    if not train["split"].eq("train").all() or not validation["split"].eq("validation").all():
        raise ValueError("Unexpected split label")
    if not train["source_row"].is_unique or not validation["source_row"].is_unique:
        raise ValueError("Duplicate source references")
    if not set(train["source_row"]).isdisjoint(set(validation["source_row"])):
        raise ValueError("Training and validation source rows overlap")
    train_month = train.timestamp.dt.strftime("%Y-%m")
    if set(train_month.unique()) != set(EXPECTED_MONTHS):
        raise ValueError(f"Unexpected/missing training months: {sorted(train_month.unique())}")
    if not validation.timestamp.ge("2026-02-01").all() or not validation.timestamp.lt("2026-03-01").all():
        raise ValueError("February validation boundary violated")
    months = {}
    for month, expected in EXPECTED_MONTHS.items():
        frame = train.loc[train_month.eq(month)].sort_values(["timestamp", "source_row"]).copy()
        counts = len(frame), int(frame.isFraud.sum())
        if enforce_counts and counts != expected:
            raise ValueError(f"Training month {month} changed: {counts} != {expected}")
        months[month] = frame
    octnov = pd.concat([months["2025-10"], months["2025-11"]]).sort_values(["timestamp", "source_row"])
    octdec = pd.concat([octnov, months["2025-12"]]).sort_values(["timestamp", "source_row"])
    dec, jan = months["2025-12"], months["2026-01"]
    feb = validation.sort_values(["timestamp", "source_row"]).copy()
    for fitting, calibration, evaluation in ((octnov, dec, jan), (octdec, jan, feb)):
        if not (fitting.timestamp.max() < calibration.timestamp.min() < calibration.timestamp.max()
                < evaluation.timestamp.min()):
            raise ValueError("Fitting/calibration/evaluation are not strictly chronological")
        if len(set(fitting.source_row) & set(calibration.source_row)) or len(set(calibration.source_row) & set(evaluation.source_row)):
            raise ValueError("Stage source references overlap")
    return {"octnov": octnov, "dec": dec, "jan": jan, "octdec": octdec, "feb": feb}


class PositiveSigmoid:
    """Sigmoid map of logit scores with a nonnegative slope (order preserving)."""
    def __init__(self):
        self.slope = None
        self.intercept = None
        self.success = False

    def fit(self, scores, labels):
        p = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        _check_probabilities(p)
        if len(p) != len(y) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("Sigmoid calibration requires both classes and aligned scores")
        x = logit(np.clip(p, 1e-12, 1 - 1e-12))
        def objective(params):
            a, b = params
            z = a * x + b
            losses = np.logaddexp(0, z) - y * z
            resid = expit(z) - y
            return (float(np.mean(losses)), np.array([np.mean(resid * x), np.mean(resid)]))
        fitted = minimize(objective, x0=np.array([1.0, 0.0]), jac=True,
                          method="L-BFGS-B", bounds=((0.0, 20.0), (-30.0, 30.0)),
                          options={"maxiter": 500, "ftol": 1e-12})
        if not fitted.success or not np.isfinite(fitted.x).all():
            raise RuntimeError(f"Sigmoid calibration did not converge: {fitted.message}")
        self.slope, self.intercept = map(float, fitted.x)
        self.success = True
        return self

    def predict(self, scores):
        if not self.success:
            raise ValueError("Sigmoid calibrator was not fitted")
        p = np.asarray(scores, dtype=float)
        _check_probabilities(p)
        return expit(self.slope * logit(np.clip(p, 1e-12, 1 - 1e-12)) + self.intercept)


def _check_probabilities(p):
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Expected finite one-dimensional probabilities in [0,1]")


def fit_calibrators(calibration_scores, calibration_labels):
    scores = np.asarray(calibration_scores, dtype=float)
    y = np.asarray(calibration_labels, dtype=int)
    _check_probabilities(scores)
    if len(y) != len(scores) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Calibration data must include both classes")
    sigmoid = PositiveSigmoid().fit(scores, y)
    isotonic = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip")
    isotonic.fit(scores, y)
    return {"sigmoid": sigmoid, "isotonic": isotonic}


def calibrated_scores(base_scores, calibrators):
    scores = np.asarray(base_scores, dtype=float)
    _check_probabilities(scores)
    all_scores = {"uncalibrated": scores,
                  "sigmoid": calibrators["sigmoid"].predict(scores),
                  "isotonic": np.asarray(calibrators["isotonic"].predict(scores), dtype=float)}
    for p in all_scores.values():
        _check_probabilities(p)
    return all_scores


def scoring_report(labels, predictions, tie_priority, *, stage):
    """Probability metrics and retrospective fixed-budget review diagnostics."""
    from .train_logistic_baseline import evaluate_scores
    rows = []
    for method in METHODS:
        for row in evaluate_scores(labels, predictions[method], tie_priority, probabilities=True):
            rows.append({"stage": stage, "method": method, **row})
    return pd.DataFrame(rows)


def select_method(january_metrics: pd.DataFrame):
    """Choose by January log loss, then Brier within 1e-6, then simplicity."""
    at_primary = january_metrics.loc[np.isclose(january_metrics.review_fraction, 0.01)].set_index("method")
    if set(at_primary.index) != set(METHODS) or at_primary.index.has_duplicates:
        raise ValueError("Missing January method metrics")
    if not np.isfinite(at_primary[["log_loss", "brier_score"]].to_numpy()).all():
        raise ValueError("Nonfinite method selection metrics")
    best_loss = float(at_primary.log_loss.min())
    candidates = [name for name in METHODS if float(at_primary.loc[name, "log_loss"]) <= best_loss + 1e-6]
    return min(candidates, key=lambda name: (float(at_primary.loc[name, "brier_score"]), METHODS.index(name)))


def reliability_by_quantile(labels, predictions, source_rows, *, stage, nominal_bins=10):
    """Tie-respecting probability quantiles; bin counts may be fewer than ten."""
    y = np.asarray(labels, dtype=int)
    refs = np.asarray(source_rows)
    rows = []
    for method in METHODS:
        p = np.asarray(predictions[method], dtype=float)
        _check_probabilities(p)
        if len(p) != len(y) or len(refs) != len(y):
            raise ValueError("Reliability columns are not aligned")
        edges = np.unique(np.quantile(p, np.linspace(0, 1, nominal_bins + 1)))
        buckets = np.searchsorted(edges[1:-1], p, side="right")
        for b in np.unique(buckets):
            idx = buckets == b
            rows.append({"stage": stage, "method": method, "bin": int(b) + 1,
                         "rows": int(idx.sum()), "positives": int(y[idx].sum()),
                         "mean_probability": float(np.mean(p[idx])),
                         "observed_rate": float(np.mean(y[idx])),
                         "min_probability": float(np.min(p[idx])),
                         "max_probability": float(np.max(p[idx])),
                         "min_source_row": int(np.min(refs[idx])),
                         "max_source_row": int(np.max(refs[idx]))})
    return pd.DataFrame(rows)


def plot_reliability(bins, *, stage, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    for method in METHODS:
        frame = bins.loc[(bins.stage == stage) & (bins.method == method)].sort_values("bin")
        ax.plot(np.arange(1, len(frame) + 1), 100 * frame.observed_rate, "o-", label=f"Observed, {method}")
        ax.plot(np.arange(1, len(frame) + 1), 100 * frame.mean_probability, "--", label=f"Predicted, {method}")
    ax.set_xlabel("Score quantile bin (low to high); boundaries differ by method")
    ax.set_ylabel("Percentage of fraud-labelled transactions")
    ax.set_title(f"{stage}: observed and predicted rates by score quantile")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main(argv=None):
    from .train_logistic_baseline import FEATURES, design_frame, fit_model, select_periods
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--input", type=Path, default=root / "data/processed/amlnet_v2_prepared.parquet")
    parser.add_argument("--output", type=Path, default=root / "models/calibration_temporal_v1")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new --output. No file will be overwritten.")
    if not args.input.is_file():
        parser.error(f"Prepared data missing: {args.input}")
    print("Loading only training and February validation Parquet rows...", flush=True)
    input_sha = sha256(args.input)  # Byte-level fingerprint; does not score holdout.
    before = args.input.stat()
    data = pd.read_parquet(args.input, engine="pyarrow", columns=FEATURES + ["timestamp", "split", "source_row", "isFraud"],
                           filters=[("split", "in", ["train", "validation"])])
    after = args.input.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Prepared input changed during read")
    train, validation = select_periods(data, enforce_counts=True)
    windows = dated_windows(train, validation, enforce_counts=True)
    del data
    print("Stage A: fitting Oct-Nov, calibrating December, assessing January...", flush=True)
    fitted_a = fit_model(windows["octnov"], log_amount=True)
    def predict(fitted, frame):
        p = fitted.predict_proba(design_frame(frame, log_amount=True))[:, 1]
        _check_probabilities(p)
        return p
    dec_probs = predict(fitted_a, windows["dec"])
    jan_probs = predict(fitted_a, windows["jan"])
    calibrators_a = fit_calibrators(dec_probs, windows["dec"].isFraud.to_numpy())
    jan_predictions = calibrated_scores(jan_probs, calibrators_a)
    january = windows["jan"]
    tie_jan = np.random.default_rng(SEED).permutation(len(january))
    jan_metrics = scoring_report(january.isFraud.to_numpy(), jan_predictions, tie_jan, stage="A_january")
    choice = select_method(jan_metrics)
    print(f"Method selected on January log loss: {choice}", flush=True)
    print("Stage B: refitting Oct-Dec, calibrating January, assessing February...", flush=True)
    fitted_b = fit_model(windows["octdec"], log_amount=True)
    jan_probs_b = predict(fitted_b, january)
    feb = windows["feb"]
    feb_probs = predict(fitted_b, feb)
    calibrators_b = fit_calibrators(jan_probs_b, january.isFraud.to_numpy())
    feb_predictions = calibrated_scores(feb_probs, calibrators_b)
    tie_feb = np.random.default_rng(SEED).permutation(len(feb))
    feb_metrics = scoring_report(feb.isFraud.to_numpy(), feb_predictions, tie_feb, stage="B_february")
    metrics = pd.concat([jan_metrics, feb_metrics], ignore_index=True)
    bins = pd.concat([
        reliability_by_quantile(january.isFraud.to_numpy(), jan_predictions, january.source_row, stage="A_january"),
        reliability_by_quantile(feb.isFraud.to_numpy(), feb_predictions, feb.source_row, stage="B_february"),
    ], ignore_index=True)
    # No local output directory is created until all fitting, checks and evaluation succeed.
    args.output.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    bins.to_csv(args.output / "reliability_bins.csv", index=False)
    for stage, frame, predictions, ties in (("A_january", january, jan_predictions, tie_jan),
                                            ("B_february", feb, feb_predictions, tie_feb)):
        pred = frame[["source_row", "timestamp", "isFraud"]].copy()
        pred["tie_priority"] = ties
        for key in METHODS:
            pred[key] = predictions[key]
        pred.to_parquet(args.output / f"{stage}_predictions.parquet", index=False)
        plot_reliability(bins, stage=stage, path=args.output / f"{stage}_reliability.png")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": input_sha, "selected_method_on_january": choice,
        "method_selection_rule": "min Jan log loss, Brier within 1e-6, fixed simplicity order",
        "selection_scope": "January selected calibration method; February reported but did not choose method",
        "models": {"base": "logistic log amount, full 13 features", "fixed_C": 1.0,
                   "sigmoid": "nonnegative slope applied to base log odds", "isotonic": "monotone isotonic mapping"},
        "stage_a": {"fit": ["2025-10", "2025-11"], "calibrate": "2025-12", "evaluate": "2026-01"},
        "stage_b": {"fit": ["2025-10", "2025-11", "2025-12"], "calibrate": "2026-01", "evaluate": "2026-02"},
        "counts": {k: {"rows": len(windows[k]), "positives": int(windows[k].isFraud.sum())}
                   for k in ("octnov", "dec", "octdec", "jan", "feb")},
        "january_selection_not_final_holdout": True, "march_april_scored": False,
        "caution": "February was explored earlier. The final holdout has not been scored. Local row-level outputs not for Git.",
        "packages": {key: version(key) for key in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "matplotlib")},
        "python": platform.python_version(),
        "artifact_sha256": {path.name: sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()},
        "complete": True,
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\nJanuary exploratory method selection (lowest log loss):", choice)
    print(jan_metrics.loc[jan_metrics.review_fraction.eq(.01),
                          ["method", "log_loss", "brier_score", "average_precision", "true_positives", "recall"]].to_string(index=False))
    print("\nFebruary historical comparison (NOT method selection):")
    print(feb_metrics.loc[feb_metrics.review_fraction.eq(.01),
                          ["method", "log_loss", "brier_score", "average_precision", "true_positives", "recall"]].to_string(index=False))
    print("\nCompleted local output:", args.output.resolve())
    return args.output


if __name__ == "__main__":
    main()
