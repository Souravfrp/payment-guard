"""Measure and plot temporal AMLNet Version 2.0 risk patterns."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from audit_amlnet_v2_schema_and_leakage import (
    extract_timestamps,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_ROOT
    / "data/raw/amlnet_v2/AMLNet_v2_transactions.csv"
)
TABLE_PATH = (
    PROJECT_ROOT
    / "results/tables/amlnet_v2_temporal_risk_profile.csv"
)
FIGURE_PATH = (
    PROJECT_ROOT
    / "results/figures/amlnet_v2_temporal_risk_profile.png"
)

CHUNK_SIZE_ROWS = 50_000

SELECTED_GROUPS = {
    "type": (
        "PAYMENT",
        "TRANSFER",
    ),
    "category": (
        "Property Investment",
        "Cryptocurrency",
        "Shell Company",
        "Other",
        "Recreation",
    ),
}

PARTIAL_MONTHS = {
    "2025-10",
    "2026-04",
}


def scan_dataset() -> dict:
    """Collect monthly totals and fraud counts in one chunked scan."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    totals = Counter()
    frauds = Counter()
    months = set()
    row_count = 0
    timestamp_failures = 0

    reader = pd.read_csv(
        DATA_PATH,
        usecols=[
            "metadata",
            "isFraud",
            "type",
            "category",
        ],
        chunksize=CHUNK_SIZE_ROWS,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):
        timestamps = extract_timestamps(
            chunk["metadata"]
            .fillna("")
            .astype(str)
        )

        labels = pd.to_numeric(
            chunk["isFraud"],
            errors="raise",
        ).astype(int)

        timestamp_failures += int(
            timestamps.isna().sum()
        )

        frame = pd.DataFrame(
            {
                "month": timestamps.dt.strftime("%Y-%m"),
                "label": labels,
                "type": chunk["type"].astype(str),
                "category": chunk["category"].astype(str),
            }
        )

        months.update(
            frame["month"].dropna()
        )

        overall = frame.groupby(
            "month"
        )["label"].agg(
            [
                "size",
                "sum",
            ]
        )

        for month, values in overall.iterrows():
            key = (
                month,
                "overall",
                "All transactions",
            )

            totals[key] += int(values["size"])
            frauds[key] += int(values["sum"])

        for field, selected_values in SELECTED_GROUPS.items():
            selected = frame[
                frame[field].isin(selected_values)
            ]

            grouped = selected.groupby(
                [
                    "month",
                    field,
                ]
            )["label"].agg(
                [
                    "size",
                    "sum",
                ]
            )

            for (month, value), values in grouped.iterrows():
                key = (
                    month,
                    field,
                    value,
                )

                totals[key] += int(values["size"])
                frauds[key] += int(values["sum"])

        row_count += len(chunk)

        print(
            f"Processed chunk {chunk_number}: "
            f"{row_count:,} cumulative rows"
        )

    return {
        "months": sorted(months),
        "totals": totals,
        "frauds": frauds,
        "row_count": row_count,
        "timestamp_failures": timestamp_failures,
    }


def build_profile(
    results: dict,
) -> pd.DataFrame:
    """Create one tidy row for every month and selected group."""
    groups = [
        (
            "overall",
            "All transactions",
        )
    ]

    groups.extend(
        (field, value)
        for field, values in SELECTED_GROUPS.items()
        for value in values
    )

    rows = []

    for month in results["months"]:
        for field, value in groups:
            key = (
                month,
                field,
                value,
            )

            total = results["totals"][key]
            fraud = results["frauds"][key]

            rows.append(
                {
                    "month": month,
                    "period_status": (
                        "partial"
                        if month in PARTIAL_MONTHS
                        else "complete"
                    ),
                    "field": field,
                    "value": value,
                    "transaction_rows": total,
                    "fraud_rows": fraud,
                    "fraud_rate_pct": (
                        100 * fraud / total
                        if total
                        else float("nan")
                    ),
                }
            )

    return pd.DataFrame(rows)


def series(
    profile: pd.DataFrame,
    months: list[str],
    field: str,
    value: str,
    column: str,
) -> list[float]:
    """Return one metric in chronological month order."""
    selected = profile[
        (profile["field"] == field)
        & (profile["value"] == value)
    ]

    lookup = dict(
        zip(
            selected["month"],
            selected[column],
        )
    )

    return [
        lookup[month]
        for month in months
    ]


def create_figure(
    profile: pd.DataFrame,
) -> None:
    """Create a labelled four-panel temporal-risk figure."""
    months = sorted(
        profile["month"].unique()
    )

    labels = [
        (
            f"{month}*"
            if month in PARTIAL_MONTHS
            else month
        )
        for month in months
    ]

    x = list(
        range(len(months))
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(
            15,
            10,
        ),
        constrained_layout=True,
    )

    (
        overall_axis,
        mixed_axis,
        shell_axis,
        support_axis,
    ) = axes.flatten()

    overall_rates = series(
        profile,
        months,
        "overall",
        "All transactions",
        "fraud_rate_pct",
    )

    overall_axis.plot(
        x,
        overall_rates,
        marker="o",
        linewidth=2.2,
        label="Overall",
    )

    for position, rate in zip(
        x,
        overall_rates,
    ):
        overall_axis.annotate(
            f"{rate:.3f}%",
            (
                position,
                rate,
            ),
            xytext=(
                0,
                8,
            ),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    overall_axis.set_title(
        "A. Overall monthly fraud rate"
    )
    overall_axis.set_ylim(
        bottom=0
    )

    mixed_groups = (
        (
            "type",
            "TRANSFER",
            "#0072B2",
        ),
        (
            "category",
            "Other",
            "#E69F00",
        ),
        (
            "category",
            "Recreation",
            "#009E73",
        ),
    )

    for field, value, color in mixed_groups:
        mixed_axis.plot(
            x,
            series(
                profile,
                months,
                field,
                value,
                "fraud_rate_pct",
            ),
            marker="o",
            linewidth=2,
            color=color,
            label=value,
        )

    mixed_axis.set_title(
        "B. Rates in large mixed-risk groups"
    )

    shell_rates = series(
        profile,
        months,
        "category",
        "Shell Company",
        "fraud_rate_pct",
    )

    shell_axis.plot(
        x,
        shell_rates,
        marker="o",
        linewidth=2.2,
        color="#D55E00",
        label="Shell Company",
    )

    for position, rate in zip(
        x,
        shell_rates,
    ):
        shell_axis.annotate(
            f"{rate:.1f}%",
            (
                position,
                rate,
            ),
            xytext=(
                0,
                8,
            ),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    shell_axis.set_title(
        "C. Shell Company rate on its own scale"
    )
    shell_axis.set_ylim(
        bottom=0
    )

    deterministic_groups = (
        (
            "type",
            "PAYMENT",
            "#CC79A7",
        ),
        (
            "category",
            "Property Investment",
            "#8C564B",
        ),
        (
            "category",
            "Cryptocurrency",
            "#56B4E9",
        ),
    )

    width = 0.24

    for offset, (
        field,
        value,
        color,
    ) in zip(
        (
            -width,
            0.0,
            width,
        ),
        deterministic_groups,
    ):
        support_axis.bar(
            [
                position + offset
                for position in x
            ],
            series(
                profile,
                months,
                field,
                value,
                "transaction_rows",
            ),
            width=width,
            color=color,
            label=value,
        )

    support_axis.set_title(
        "D. Support for observed 100%-fraud groups"
    )
    support_axis.set_ylabel(
        "Transactions (count)"
    )

    for axis in (
        overall_axis,
        mixed_axis,
        shell_axis,
    ):
        axis.set_ylabel(
            "Fraud rate (%)"
        )

    for axis in axes.flatten():
        axis.set_xticks(x)
        axis.set_xticklabels(
            labels,
            rotation=35,
            ha="right",
        )
        axis.set_xlabel(
            "Calendar month"
        )
        axis.grid(
            axis="y",
            alpha=0.25,
        )
        axis.spines[
            [
                "top",
                "right",
            ]
        ].set_visible(False)
        axis.legend(
            fontsize=8
        )

    figure.suptitle(
        "AMLNet Version 2.0 temporal risk profile",
        fontsize=16,
        fontweight="bold",
    )

    figure.text(
        0.5,
        -0.015,
        (
            "* October 2025 begins on 13 October; "
            "April 2026 ends on 27 April. "
            "Rates are observed, not annualized. "
            "Source: AMLNet Version 2.0 "
            "synthetic transactions."
        ),
        ha="center",
        fontsize=9,
    )

    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    """Run the temporal profile and save its table and figure."""
    results = scan_dataset()

    if results["timestamp_failures"]:
        raise ValueError(
            "Timestamp parse failures: "
            f"{results['timestamp_failures']:,}"
        )

    profile = build_profile(
        results
    )

    TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile.to_csv(
        TABLE_PATH,
        index=False,
    )

    create_figure(
        profile
    )

    print(
        "\nAMLNet Version 2.0 temporal risk profile"
    )
    print(
        f"Rows scanned: "
        f"{results['row_count']:,}"
    )
    print(
        "Timestamp parse failures: 0"
    )
    print(
        "Saved table: "
        f"{TABLE_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Saved figure: "
        f"{FIGURE_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Profile result: COMPLETED"
    )


if __name__ == "__main__":
    main()
