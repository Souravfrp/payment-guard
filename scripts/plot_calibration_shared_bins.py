"""Audit original calibration files and plot method-comparable reliability bins.

Read-only access to models/calibration_temporal_v1. No model fitting, no holdout
read, no rewriting previous predictions/figures/metrics. The only output is a
new, distinct local directory (ignored by the project's gitignore).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ("uncalibrated", "sigmoid", "isotonic")
STAGES = {
    "A_january": ("2026-01-01", "2026-02-01", 186_097, 281),
    "B_february": ("2026-02-01", "2026-03-01", 167_990, 202),
}
REQUIRED = {"source_row", "timestamp", "isFraud", "tie_priority", *METHODS}


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_saved_artifacts(source: Path) -> dict:
    """Check the original v1 manifest and all recorded output digests."""
    manifest = json.loads((source / "run.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("march_april_scored") is not False:
        raise ValueError("The calibration source must be complete and holdout-unscored")
    if manifest.get("selected_method_on_january") != "isotonic":
        raise ValueError("Unexpected preselected calibration method")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing saved artifact hashes")
    for filename, digest in hashes.items():
        if Path(filename).name != filename or not isinstance(digest, str):
            raise ValueError("Invalid manifest artifact name or digest")
        artifact = source / filename
        if not artifact.is_file() or file_sha256(artifact) != digest:
            raise ValueError(f"Missing or changed saved artifact: {filename}")
    for stage in STAGES:
        if f"{stage}_predictions.parquet" not in hashes:
            raise ValueError("Stage predictions are absent from verified manifest")
    return manifest


def validate_predictions(frame: pd.DataFrame, stage: str, *, full_counts: bool = True):
    """Check source alignment and forbid rows outside the appropriate month."""
    if stage not in STAGES:
        raise ValueError("Unknown stage")
    if not frame.columns.is_unique or not REQUIRED.issubset(frame.columns):
        raise ValueError("Required prediction columns are missing or duplicated")
    start, end, expected_n, expected_p = STAGES[stage]
    n = len(frame)
    if n == 0 or (full_counts and n != expected_n):
        raise ValueError("Prediction row count mismatch")
    if not pd.api.types.is_datetime64_any_dtype(frame.timestamp):
        raise ValueError("Expected datetime timestamps")
    if not frame.timestamp.ge(start).all() or not frame.timestamp.lt(end).all():
        raise ValueError("Unexpected timestamp: only January and February allowed")
    refs = frame.source_row.to_numpy()
    y = frame.isFraud.to_numpy()
    ties = frame.tie_priority.to_numpy()
    if not pd.api.types.is_integer_dtype(frame.source_row) or not frame.source_row.is_unique or (refs <= 0).any():
        raise ValueError("Invalid or duplicate source references")
    if not np.isin(y, (0, 1)).all() or (full_counts and int(y.sum()) != expected_p):
        raise ValueError("Positive-label count mismatch")
    if not pd.api.types.is_integer_dtype(frame.tie_priority) or not np.array_equal(np.sort(ties), np.arange(n)):
        raise ValueError("Tie priorities must be a permutation")
    for method in METHODS:
        scores = frame[method].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError(f"Invalid probability values: {method}")


def shared_bin_table(frame: pd.DataFrame, *, stage: str, bins: int = 10) -> pd.DataFrame:
    """Create equal-count rank bins using the SAME uncalibrated ordering.

    At a tied boundary, the original label-independent priority determines
    membership. In contrast to method-specific quantiles, this can split an
    uncalibrated probability tie, which is disclosed in the output table.
    """
    validate_predictions(frame, stage, full_counts=False)
    n = len(frame)
    if not (2 <= bins <= n):
        raise ValueError("Invalid number of bins")
    base = frame.uncalibrated.to_numpy(dtype=float)
    tie = frame.tie_priority.to_numpy(dtype=int)
    order = np.lexsort((tie, -base))  # highest base score first, then saved tie priority
    groups = np.array_split(order, bins)
    labels = frame.isFraud.to_numpy(dtype=int)
    result = []
    for i, indices in enumerate(groups):
        rank_bin = bins - i  # bin 1 is lowest, bin 10 is highest
        row = {"stage": stage, "bin": rank_bin, "rows": int(len(indices)),
               "positives": int(labels[indices].sum()),
               "observed_rate": float(labels[indices].mean()),
               "boundary_split_base_tie": bool(
                   i < bins - 1 and base[indices[-1]] == base[groups[i + 1][0]]
               )}
        for method in METHODS:
            row[f"mean_probability_{method}"] = float(frame[method].to_numpy(dtype=float)[indices].mean())
        result.append(row)
    table = pd.DataFrame(result).sort_values("bin").reset_index(drop=True)
    if table.rows.sum() != n or table.positives.sum() != int(labels.sum()):
        raise AssertionError("Shared bins failed to preserve transaction counts")
    if table.rows.max() - table.rows.min() > 1:
        raise AssertionError("Shared bin sizes unexpectedly imbalanced")
    return table


def plot_shared_bins(table: pd.DataFrame, stage: str, destination: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10.6, 6.3))
    x = table["bin"].to_numpy()
    ax.plot(x, 100 * table.observed_rate, "ko-", linewidth=2, label="Observed (same cases for all methods)")
    for method in METHODS:
        ax.plot(x, 100 * table[f"mean_probability_{method}"], "--o", label=f"Predicted — {method}")
    ax.set(xlabel="Shared uncalibrated-score rank decile (low to high)",
           ylabel="Fraud-labelled share / mean prediction (%)",
           title=f"{stage}: shared transaction groups, not method-specific quantiles")
    ax.set_xticks(np.arange(1, len(table) + 1))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    top = table.iloc[-1]
    ax.text(0.98, 0.84, f"Highest decile: {int(top.positives):,} positives / {int(top.rows):,} rows",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.85))
    fig.tight_layout()
    fig.savefig(destination, dpi=175)
    plt.close(fig)


def plot_predicted_vs_observed(table: pd.DataFrame, stage: str, destination: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    maxima = [100 * table.observed_rate.max()]
    for method in METHODS:
        x = 100 * table[f"mean_probability_{method}"].to_numpy()
        y = 100 * table.observed_rate.to_numpy()
        ax.scatter(x, y, s=28, alpha=0.8, label=method)
        # Index the highest uncalibrated-score decile, shared by all methods.
        ax.annotate("10", (x[-1], y[-1]), xytext=(6, 4),
                    textcoords="offset points", fontsize=8)
        maxima.append(float(np.max(x)))
    hi = max(maxima) * 1.15
    ax.plot([0, hi], [0, hi], linestyle=":", color="0.4", label="Ideal agreement")
    ax.set(xlim=(0, hi), ylim=(0, hi),
           xlabel="Mean predicted probability in shared rank decile (%)",
           ylabel="Observed fraud-labelled share in same decile (%)",
           title=f"{stage}: predicted vs observed on identical deciles")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=175)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--source", type=Path, default=root / "models/calibration_temporal_v1")
    parser.add_argument("--output", type=Path, default=root / "models/calibration_visual_refresh_v1")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists. Existing outputs will not be replaced")
    manifest = verify_saved_artifacts(args.source)
    tables = []
    for stage in STAGES:
        path = args.source / f"{stage}_predictions.parquet"
        frame = pd.read_parquet(path, columns=sorted(REQUIRED))
        validate_predictions(frame, stage)
        t = shared_bin_table(frame, stage=stage)
        tables.append(t)
        print(f"{stage}: {len(frame):,} rows, {int(frame.isFraud.sum())} positives; "
              f"shared top-decile {int(t.iloc[-1].positives)} / {int(t.iloc[-1].rows)}", flush=True)
    all_bins = pd.concat(tables, ignore_index=True)
    # Original saved-run artifacts are read only. New output is untracked by design.
    args.output.mkdir(parents=True, exist_ok=False)
    all_bins.to_csv(args.output / "shared_reliability_bins.csv", index=False)
    for stage, table in zip(STAGES, tables):
        plot_shared_bins(table, stage, args.output / f"{stage}_shared_deciles.png")
        plot_predicted_vs_observed(table, stage, args.output / f"{stage}_predicted_vs_observed.png")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_complete": manifest["complete"],
        "source_selected_method": manifest["selected_method_on_january"],
        "source_manifest_sha256": file_sha256(args.source / "run.json"),
        "data_scope": "Only saved January and February predictions; March-April never read",
        "bin_definition": "10 equal-count bins by uncalibrated score, ties ordered with saved label-independent priority; same transactions for all methods",
        "method_selection_unchanged": True,
        "artifacts_sha256": {p.name: file_sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: four new figures and shared-bin CSV saved in {args.output}")
    print("The original calibration_v1 outputs and selection rule were not modified.")
    return args.output


if __name__ == "__main__":
    main()
