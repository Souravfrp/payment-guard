"""Plot review-budget trade-offs from aggregate validation counts; no training."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

STYLES = {
    "constant_train_rate": ("Constant reference", "#7b8494", "X"),
    "amount_ranking": ("Amount ranking", "#c67912", "s"),
    "logistic_raw": ("Logistic · raw amount", "#3274bc", "^"),
    "logistic_log": ("Logistic · log amount", "#007f73", "o"),
}


def validate_counts(data):
    required = ["model", "selected", "true_positives", "false_positives",
                "false_negatives", "true_negatives"]
    if not set(required).issubset(data.columns) or len(data) != 12:
        raise ValueError("Expected four approaches at three review budgets.")
    numbers = data[required[1:]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or (numbers < 0).any() or (numbers != np.floor(numbers)).any():
        raise ValueError("Counts must be finite nonnegative integers.")
    if set(data.model) != set(STYLES) or data.duplicated(["model", "selected"]).any():
        raise ValueError("Unexpected or duplicated model/budget rows.")
    for _, group in data.groupby("model"):
        if sorted(group.selected.tolist()) != [168, 840, 1680]:
            raise ValueError("Unexpected review budgets.")
    tp, fp, fn, tn = [data[x] for x in required[2:]]
    if not ((tp + fp == data.selected) & (tp + fn == 202) & (tp + fp + fn + tn == 167990)).all():
        raise ValueError("Counts disagree with the February validation checkpoint.")
    for field, expected in {"precision": tp / data.selected, "recall": tp / 202}.items():
        if field in data and not np.allclose(data[field], expected, rtol=1e-6, atol=1e-8):
            raise ValueError(f"Reported {field} disagrees with counts.")


def plot_results(data, output):
    validate_counts(data)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "payment-guard-baseline-v1"}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))
        fig.subplots_adjust(left=0.075, right=0.965, bottom=0.25, top=0.73, wspace=0.22)
        fig.suptitle("PaymentGuard | What do more reviews buy?", x=0.075, ha="left", y=0.97,
                     fontsize=20, fontweight="bold", color="#182c40")
        fig.text(0.075, 0.895, "February 2026 validation · 167,990 synthetic transactions · 202 fraud labels",
                 color="#526171", fontsize=11)
        handles = []
        for name, (label, color, marker) in STYLES.items():
            group = data.loc[data.model.eq(name)].sort_values("selected")
            x, tp = group.selected, group.true_positives
            line, = axes[0].plot(x, tp, label=label, color=color, marker=marker,
                                 markersize=7, linewidth=2, linestyle="--", alpha=0.95)
            axes[1].plot(x, 100 * tp / x, color=color, marker=marker,
                         markersize=7, linewidth=2, linestyle="--", alpha=0.95)
            handles.append(line)
            if name == "logistic_log":
                for reviews, caught in zip(x, tp):
                    axes[0].annotate(str(caught), (reviews, caught), xytext=(0, 9),
                                     textcoords="offset points", ha="center", color=color, fontsize=10)
        axes[0].set_title("Fraud caught", loc="left", fontsize=13, fontweight="bold", pad=12)
        axes[0].set_ylabel("Caught fraud cases (out of 202)")
        axes[0].set_ylim(-5, 223)
        axes[0].set_yticks([0, 50, 100, 150, 202])
        axes[0].axhline(202, color="#aeb8c2", lw=0.8, zorder=0)
        axes[1].set_title("How many flags were correct?", loc="left", fontsize=13, fontweight="bold", pad=12)
        axes[1].set_ylabel("Precision")
        axes[1].set_ylim(-3, 106)
        axes[1].yaxis.set_major_formatter(PercentFormatter(100))
        for ax in axes:
            ax.set_xlim(70, 1800)
            ax.set_xticks([168, 840, 1680], ["168\n0.1%", "840\n0.5%", "1,680\n1%"])
            ax.set_xlabel("Transactions selected for review", labelpad=8)
            ax.grid(axis="y", color="#e0e5eb", linewidth=0.7)
            ax.set_axisbelow(True)
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.067, 0.855),
                   ncol=4, frameon=False, fontsize=10)
        log_counts = data.loc[data.model.eq("logistic_log")].set_index("selected")
        extra_tp = int(log_counts.loc[1680, "true_positives"] - log_counts.loc[840, "true_positives"])
        extra_fp = int(log_counts.loc[1680, "false_positives"] - log_counts.loc[840, "false_positives"])
        fig.text(0.075, 0.105, f"Log model: 840 → 1,680 reviews added {extra_tp} fraud detections and {extra_fp} false alarms.",
                 fontsize=11, fontweight="bold", color="#182c40")
        fig.text(0.075, 0.045, "Only three budgets were evaluated; dashed lines connect observations, not measured intermediate results.\n"
                 "Retrospective batch ranking, not a live decision policy. Constant reference uses seeded tie-breaking.",
                 fontsize=9, color="#526171")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, facecolor="white", metadata={"Date": None} if output.suffix == ".svg" else None)
        plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "results/tables/logistic_baseline_review_counts.csv")
    parser.add_argument("--output", type=Path, default=root / "results/figures/logistic_baseline_review_tradeoff.svg")
    args = parser.parse_args()
    plot_results(pd.read_csv(args.input), args.output)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
