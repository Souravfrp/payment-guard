"""Visualize chronological splits using aggregate preparation evidence."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[2]

    summary_path = (
        root / "results/tables/amlnet_v2_split_summary.csv"
    )
    figure_path = (
        root / "results/figures/amlnet_v2_chronological_splits.png"
    )

    summary = pd.read_csv(summary_path).set_index("split")
    periods = ["train", "validation", "test"]
    summary = summary.loc[periods]

    if not (
        summary["rows"].gt(0).all()
        and summary["positive_labels"].between(
            0,
            summary["rows"],
        ).all()
    ):
        raise ValueError("Invalid split counts.")

    starts = pd.to_datetime(summary["earliest"])
    ends = pd.to_datetime(summary["latest"])

    for earlier, later in zip(periods, periods[1:]):
        if not ends[earlier] < starts[later]:
            raise ValueError("Split periods overlap.")

    colors = ["#2563eb", "#d97706", "#059669"]
    labels = ["Training", "Validation", "Test"]
    rates = (
        100 * summary["positive_labels"] / summary["rows"]
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        layout="constrained",
    )

    timeline, rate_axis = axes

    for index, split in enumerate(periods):
        left = mdates.date2num(starts[split])
        right = mdates.date2num(ends[split])

        timeline.barh(
            index,
            right - left,
            left=left,
            height=0.55,
            color=colors[index],
        )

    timeline.set_yticks(range(3), labels)
    timeline.invert_yaxis()
    timeline.xaxis_date()
    timeline.xaxis.set_major_locator(
        mdates.MonthLocator()
    )
    timeline.xaxis.set_major_formatter(
        mdates.DateFormatter("%b %Y")
    )

    for boundary in ("2026-02-01", "2026-03-01"):
        timeline.axvline(
            pd.Timestamp(boundary),
            color="#475569",
            linestyle="--",
            linewidth=1,
        )

    timeline.set_title("Observed transaction periods")
    timeline.set_xlabel("Dataset timestamp; timezone unspecified")
    timeline.grid(axis="x", alpha=0.2)

    rate_axis.bar(labels, rates, color=colors)

    for index, split in enumerate(periods):
        rate_axis.text(
            index,
            rates[split] + rates.max() * 0.04,
            f"{rates[split]:.4f}%"
            f"\n{int(summary.loc[split, 'positive_labels']):,}"
            f" / {int(summary.loc[split, 'rows']):,}",
            ha="center",
            va="bottom",
        )

    rate_axis.set_ylim(0, rates.max() * 1.5)
    rate_axis.set_ylabel("Positive-label rate (%)")
    rate_axis.set_title(
        "Synthetic positive labels divided by transactions"
    )
    rate_axis.grid(axis="y", alpha=0.2)
    rate_axis.set_axisbelow(True)

    fig.suptitle(
        "AMLNet v2: chronological preparation splits",
        fontsize=14,
    )

    fig.supxlabel(
        "Descriptive evidence only. Test period follows full-period "
        "exploratory auditing.",
        fontsize=9,
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
