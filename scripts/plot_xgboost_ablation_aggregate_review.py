"""Reconcile and plot published *aggregate* January/February feature ablations.

No project source data, holdout, predictions or fitted models are read.
This reproduces two descriptive figures locally while verifying immutable
SHA-256 digests for the public source tables/figures.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE_FOLDER = Path("results/tables/xgboost_benchmark_v1")
FIG_FOLDER = Path("results/figures/xgboost_benchmark_v1")
TABLE_DIGESTS = {
    "feature_ablation_metrics.csv": "1846caf207013443792aaa610d5f6abbdc4be61e5966f9d8811d1e5a5993726e",
    "feature_ablation_changes_vs_full.csv": "a3855579a2fba83a3d3ba723c8b189355624166d066c710c2cfd9de25b145b42",
}
FIG_DIGESTS = {
    "feature_ablation_at_one_percent.png": "14992473649399fac842732b7d5cbcf99238e27d8d779307458009e5e16fb4d3",
    "feature_ablation_average_precision.png": "8cda02a333bd8f5ae543f1ff61504815dc37cdd689a812d5983154c4e6f9f58f",
}
MONTHS = {"January": (186_097, 281), "February": (167_990, 202)}
VARIANTS = {
    "full": (13, ""),
    "without_type": (12, "type"),
    "without_category": (12, "category"),
    "without_both": (11, "type,category"),
}
FAMILIES = ("logistic", "xgboost")
BUDGETS = (.001, .005, .01)
COLORS = {"logistic": "#315D7E", "xgboost": "#C77825"}
LABELS = {"full": "All 13 features", "without_type": "Without type", "without_category": "Without category", "without_both": "Without both"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_published_inputs(root=ROOT) -> None:
    root = Path(root)
    for folder, digests in ((TABLE_FOLDER, TABLE_DIGESTS), (FIG_FOLDER, FIG_DIGESTS)):
        for name, expected in digests.items():
            path = root / folder / name
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Published source/figure missing or SHA-256 mismatch: {path}")


def audit_metrics(m: pd.DataFrame, changes: pd.DataFrame) -> None:
    """Fail closed on missing rows, mismatched budgets, counts, or delta arithmetic."""
    key = ["month", "family", "variant", "review_fraction"]
    expected = {(month, family, var, budget)
                for month in MONTHS for family in FAMILIES
                for var in VARIANTS for budget in BUDGETS}
    if len(m) != 48 or len(changes) != 48:
        raise ValueError("Exactly 48 records expected in each CSV")
    if m[key].isna().any().any() or changes[key].isna().any().any():
        raise ValueError("Missing identity or budget fields")
    for name, table in (("metrics", m), ("changes", changes)):
        keys = {tuple(x) for x in table[key].itertuples(index=False, name=None)}
        if keys != expected or table.duplicated(key).any():
            raise ValueError(f"{name}: missing, extra or duplicated month/family/variant/budget rows")
    if not m.equals(changes[m.columns]):
        raise ValueError("Base metric fields differ between the two exported CSVs")
    numeric = ["average_precision", "log_loss", "brier_score", "precision", "recall", "cutoff_score", "actual_review_fraction"]
    if not np.isfinite(m[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite metrics")
    for _, row in m.iterrows():
        n, positives = MONTHS[row.month]
        fcount, excluded = VARIANTS[row.variant]
        actual_excluded = "" if pd.isna(row.excluded) else str(row.excluded)
        if (int(row.transactions) != n or int(row.fraud_labels) != positives
                or int(row.n_source_features) != fcount or actual_excluded != excluded):
            raise ValueError("Unexpected cohort, positives, excluded features or source-feature count")
        budget_n = math.ceil(n * row.review_fraction)
        tp, fp, fn, tn = (int(row[x]) for x in
                          ("true_positives", "false_positives", "false_negatives", "true_negatives"))
        if min(tp, fp, fn, tn) < 0 or int(row.selected) != budget_n or tp + fp != budget_n or tp + fn != positives or fp + tn != n - positives:
            raise ValueError("Review budget or confusion matrix disagrees")
        for name, true in (("precision", tp / budget_n), ("recall", tp / positives),
                           ("actual_review_fraction", budget_n/n)):
            if not np.isclose(row[name], true, atol=1e-12, rtol=0):
                raise ValueError(f"Arithmetic error in {name}")
        if not (0 <= row.average_precision <= 1 and 0 <= row.cutoff_score <= 1
                and row.log_loss >= 0 and row.brier_score >= 0):
            raise ValueError("Metric outside allowed bounds")
        tie_total = int(row.cutoff_tie_count)
        tie_selected = int(row.selected_from_cutoff_tie)
        if tie_total < 1 or not 1 <= tie_selected <= min(tie_total, budget_n):
            raise ValueError("Invalid cutoff tie counts")
        tie_crosses = str(row.tie_crosses_cutoff).lower() in {"true", "1"}
        if tie_crosses != (tie_total > tie_selected):
            raise ValueError("Cutoff tie flag disagreement")
    for (month, family, variant), part in m.groupby(["month", "family", "variant"]):
        if len(part) != 3:
            raise ValueError("Each family/variant/month must have 3 budgets")
        for col in ("average_precision", "log_loss", "brier_score"):
            if part[col].nunique() != 1:
                raise ValueError(f"Metric {col} varied with review fraction")
        ordered = part.sort_values("review_fraction")
        if not ordered.true_positives.is_monotonic_increasing:
            raise ValueError("Caught counts decreased when review capacity increased")
    keys_full = m.query('variant == "full"').set_index(["month", "family", "review_fraction"])
    for _, row in changes.iterrows():
        full = keys_full.loc[(row.month, row.family, row.review_fraction)]
        for metric in ("true_positives", "average_precision", "log_loss", "brier_score"):
            recorded_baseline = row[f"full_{metric}"]
            recorded_delta = row[f"delta_{metric}_vs_full"]
            if not np.isclose(recorded_baseline, full[metric], atol=1e-12, rtol=0):
                raise ValueError(f"Incorrect full-model reference for {metric}")
            if not np.isclose(recorded_delta, row[metric]-full[metric], atol=1e-12, rtol=0):
                raise ValueError(f"Incorrect ablation delta for {metric}")


def plot_comparison(metrics: pd.DataFrame, output: Path, metric="true_positives") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.3, 5.5), layout="constrained")
    variants = list(VARIANTS)
    x = np.arange(len(variants))
    for ax, month in zip(axes, MONTHS):
        n, positives = MONTHS[month]
        part = metrics[(metrics.month == month) & np.isclose(metrics.review_fraction, .01)]
        for shift, fam in ((-.18, "logistic"), (.18, "xgboost")):
            indexed = part[part.family == fam].set_index("variant")
            values = indexed.loc[variants, metric].to_numpy(dtype=float)
            bars = ax.bar(x + shift, values, width=.35, label="Logistic" if fam == "logistic" else "XGBoost", color=COLORS[fam])
            ax.bar_label(bars, fmt="%.3f" if metric == "average_precision" else "%.0f", padding=3, fontsize=9)
        ax.set_xticks(x, [LABELS[v] for v in variants], rotation=15, ha="right")
        ax.set_title(f"{month} · {positives} positive labels")
        ax.set_ylim(0, 1.04 if metric == "average_precision" else positives * 1.15)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.set_ylabel("Average precision (full ranked list)" if metric == "average_precision" else "Positive labels caught (1% reviews)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(.5,1.06), frameon=False)
    fig.suptitle("Matched, fixed-setting feature removals — logistic vs XGBoost", weight="bold", y=1.13, fontsize=15)
    fig.text(.5,-.02,"Historical development periods · Each removed-feature variant refitted · Synthetic AMLNet v2",ha="center",fontsize=9)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="Audit public inputs without writing outputs")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root")
    parser.add_argument("--output", type=Path, default=None, help="Ignored local output directory")
    args = parser.parse_args(argv)
    root = args.root
    verify_published_inputs(root)
    m = pd.read_csv(root/TABLE_FOLDER/"feature_ablation_metrics.csv")
    d = pd.read_csv(root/TABLE_FOLDER/"feature_ablation_changes_vs_full.csv")
    audit_metrics(m, d)
    if args.check_only:
        print("PASS: 48-row aggregate arithmetic, matched design, delta calculations and published SHA-256 digests; no files rewritten")
        return
    output = args.output or root / "models/xgboost_ablation_aggregate_review_v1"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing local outputs: {output}")
    output.mkdir(parents=True)
    plot_comparison(m, output/"feature_ablation_at_one_percent.png", "true_positives")
    plot_comparison(m, output/"feature_ablation_average_precision.png", "average_precision")
    print(f"PASS: audited 48 metric and delta rows; generated two charts in {output.resolve()}")
    print("Only aggregate data read. No fitted models, row-level scores, or March-April observations loaded.")


if __name__ == "__main__":
    main()
