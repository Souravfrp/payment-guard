"""Redraw verified February missed-fraud diagnostics from existing CSV outputs.

Reads only results/tables/missed_fraud_{matched_pairs,contribution_differences}.csv.
Never loads models, refits, scores holdout, or overwrites existing output files.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

EXPECTED_ROWS = {830467, 830408, 719998, 722735, 722740, 816607}
FULL_NAME = "missed_fraud_feature_differences_diverging_full.png"
DETAIL_NAME = "missed_fraud_feature_differences_diverging_detail.png"
SCORE_NAME = "missed_fraud_score_comparison_v2.png"
DETAIL_LIMIT = 2.0  # For colours only, never clips source data or numerical annotations.

GROUP_LABELS = {
    "amount": "Amount (log-transformed)",
    "oldbalanceOrg": "Origin balance",
    "day_of_month": "Day of month",
    "hour_cycle": "Hour (cyclic)",
    "day_of_week_cycle": "Day of week (cyclic)",
    "month_cycle": "Month (cyclic)",
    "type": "Transaction type",
    "category": "Category",
    "metadata.payment_method": "Payment method",
    "metadata.device_info.os": "Device OS",
    "metadata.device_info.type": "Device type",
    "metadata.location.city": "Location: city",
    "metadata.location.state": "Location: state",
}


def load_checked_tables(root: Path):
    """Validate correspondence and log-odds identities before drawing figures."""
    folder = root / "results" / "tables"
    pairs = pd.read_csv(folder / "missed_fraud_matched_pairs.csv")
    diffs = pd.read_csv(folder / "missed_fraud_contribution_differences.csv")
    required_pairs = {
        "missed_source_row", "caught_source_row", "missed_log_odds", "caught_log_odds",
        "delta_log_odds_missed_minus_caught", "cutoff_log_odds",
    }
    if not required_pairs.issubset(pairs.columns):
        raise ValueError("Missing expected columns in matched pairs table")
    if "missed_source_row" not in diffs.columns:
        raise ValueError("Missing missed_source_row in contribution differences table")
    if len(pairs) != 6 or len(diffs) != 6:
        raise ValueError("Expected exactly six February missed cases")
    for frame in (pairs, diffs):
        ids = pd.to_numeric(frame["missed_source_row"], errors="raise")
        if ids.isna().any() or not np.array_equal(ids.to_numpy(), ids.astype("int64").to_numpy()):
            raise ValueError("Source row identifiers must be integers")
        if set(ids.astype(int)) != EXPECTED_ROWS:
            raise ValueError("Unexpected missed-case identifiers")
        if not ids.is_unique or not frame.columns.is_unique:
            raise ValueError("Duplicate source row identifiers or columns")
        frame["missed_source_row"] = ids.astype("int64")
    expected_groups = {"contrib__" + label for label in GROUP_LABELS}
    actual_groups = set(diffs.columns) - {"missed_source_row"}
    if actual_groups != expected_groups:
        raise ValueError("Unexpected/missing contribution groups: " + repr(sorted(actual_groups ^ expected_groups)))
    # Keep original frozen ranking order; align the contribution rows by source ID.
    pairs = pairs.reset_index(drop=True)
    diffs = diffs.set_index("missed_source_row").loc[pairs.missed_source_row].reset_index()
    if not pairs.missed_source_row.equals(diffs.missed_source_row):
        raise ValueError("Row alignment disagreement")
    metrics = ["missed_log_odds", "caught_log_odds", "delta_log_odds_missed_minus_caught", "cutoff_log_odds"]
    for col in metrics:
        pairs[col] = pd.to_numeric(pairs[col], errors="raise")
    for col in expected_groups:
        diffs[col] = pd.to_numeric(diffs[col], errors="raise")
    if not np.isfinite(pairs[metrics].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite comparison scores")
    if not np.isfinite(diffs[list(expected_groups)].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite contribution differences")
    gaps = pairs.missed_log_odds - pairs.caught_log_odds
    np.testing.assert_allclose(gaps, pairs.delta_log_odds_missed_minus_caught, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(diffs[list(expected_groups)].sum(axis=1), gaps, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(pairs.cutoff_log_odds, pairs.cutoff_log_odds.iloc[0], atol=1e-10, rtol=0)
    if not (pairs.missed_log_odds < pairs.cutoff_log_odds).all() or not (
        pairs.caught_log_odds >= pairs.cutoff_log_odds
    ).all():
        raise ValueError("Missed/caught scores disagree with review cutoff")
    return pairs, diffs


def short_group_name(column: str) -> str:
    return GROUP_LABELS[column.removeprefix("contrib__")]


def draw_heatmap(diffs: pd.DataFrame, *, colour_limit: float, detail: bool, destination: Path):
    names = ["contrib__" + group for group in GROUP_LABELS]
    values = diffs[names].to_numpy(dtype=float).T
    if colour_limit <= 0 or not np.isfinite(colour_limit):
        raise ValueError("Colour limit must be finite and positive")
    fig, ax = plt.subplots(figsize=(13.4, 7.8))
    fig.subplots_adjust(left=.255, right=.89, top=.83, bottom=.25)
    norm = TwoSlopeNorm(vmin=-colour_limit, vcenter=0, vmax=colour_limit)
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
    ax.set_xticks(np.arange(values.shape[1]), diffs.missed_source_row.astype(str).tolist())
    ax.set_yticks(np.arange(values.shape[0]), [short_group_name(name) for name in names])
    ax.set_xlabel("Missed transaction source row", labelpad=12)
    fig.suptitle("Feature contributions | Missed minus matched caught fraud", x=.255, ha="left", y=.965, fontsize=15)
    fig.text(.255, .905, "February validation | Grouped fitted log-odds differences, not causal effects", fontsize=10)
    if detail:
        note = ("Detail view: colours saturate outside ±2.0 log-odds. "
                "Asterisks mark saturated cells; numbers show true values.")
    else:
        note = "Full range: symmetric colour limits preserve every observed magnitude."
    fig.text(.255, .105, note, fontsize=9, color="#475569")
    fig.text(.255, .070, "Blue lowers the missed case's log-odds; red raises them. Pale cells are near zero.",
             fontsize=9, color="#475569")
    ax.set_xticks(np.arange(-.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#dddddd", linestyle="-", linewidth=.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    # Numerical annotation ensures that saturated values are never hidden or misreported.
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            value = values[r, c]
            if abs(value) < .15:
                continue
            clipped = bool(detail and abs(value) > colour_limit)
            label = f"{value:+.1f}{'*' if clipped else ''}"
            ink = "white" if abs(value) > .65 * colour_limit else "#1e293b"
            ax.text(c, r, label, ha="center", va="center", fontsize=8.3, color=ink)
    has_clipping = bool(detail and (np.abs(values) > colour_limit).any())
    cb = fig.colorbar(im, ax=ax, shrink=.94, pad=.03, extend="both" if has_clipping else "neither")
    cb.set_label("Difference in fitted log-odds contribution", fontsize=10)
    fig.savefig(destination, dpi=175, facecolor="white")
    plt.close(fig)


def draw_score_comparison(pairs: pd.DataFrame, destination: Path):
    x = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(11.8, 6.7))
    fig.subplots_adjust(left=.12, right=.97, top=.78, bottom=.23)
    ax.scatter(x, pairs.missed_log_odds, marker="x", s=96, linewidths=2.4,
               color="#2166ac", label="Missed labelled fraud", zorder=3)
    ax.scatter(x, pairs.caught_log_odds, marker="o", s=80,
               color="#d95f02", label="Matched caught fraud (reused)", zorder=3)
    ax.axhline(pairs.cutoff_log_odds.iloc[0], linestyle="--", linewidth=1.7,
               color="#64748b", label="1,680-review cutoff")
    ax.set_xticks(x, pairs.missed_source_row.astype(str).tolist())
    ax.set_xlabel("Missed transaction source row")
    ax.set_ylabel("Saved logistic log-odds (not calibrated risk)")
    distinct = int(pairs.caught_source_row.nunique())
    fig.suptitle("February validation | Missed vs matched caught fraud", x=.12, ha="left", y=.965, fontsize=15)
    fig.text(.12, .875, f"{distinct} distinct caught comparators reused across {len(pairs)} pairs; each orange point is a matched reference.",
             fontsize=10, color="#475569")
    fig.text(.12, .075, "Comparisons are descriptive, not causal. The review boundary is determined by capacity, not a 50% threshold.",
             fontsize=9, color="#475569")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=.22)
    fig.savefig(destination, dpi=175, facecolor="white")
    plt.close(fig)


def generate(root: Path):
    root = Path(root).resolve()
    pairs, diffs = load_checked_tables(root)
    output = root / "results" / "figures"
    destinations = [output / SCORE_NAME, output / FULL_NAME, output / DETAIL_NAME]
    present = [str(p) for p in destinations if p.exists()]
    if present:
        raise FileExistsError("Refusing to replace existing figures: " + ", ".join(present))
    output.mkdir(parents=True, exist_ok=True)
    limit = float(np.abs(diffs.drop(columns="missed_source_row").to_numpy(dtype=float)).max())
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("All contribution differences are zero or invalid")
    staged_published = []
    with tempfile.TemporaryDirectory(prefix=".missed-plots-v2-", dir=root / "results") as temp:
        stage = Path(temp)
        draw_score_comparison(pairs, stage / SCORE_NAME)
        draw_heatmap(diffs, colour_limit=limit, detail=False, destination=stage / FULL_NAME)
        draw_heatmap(diffs, colour_limit=DETAIL_LIMIT, detail=True, destination=stage / DETAIL_NAME)
        try:
            for dest in destinations:
                os.link(stage / dest.name, dest)  # atomic create-if-absent; never overwrites
                staged_published.append(dest)
        except Exception:
            for dest in staged_published:
                dest.unlink()
            raise
    print("PASS: six case IDs, February matching, log-odds sums and cutoff consistency")
    print(f"Full-scale magnitude: ±{limit:.6g}; detail colour limit: ±{DETAIL_LIMIT:g}")
    print(f"Distinct matched caught examples: {pairs.caught_source_row.nunique()}")
    print("Created three new, non-overwriting visualization files:")
    for dest in destinations:
        print(" -", dest.relative_to(root))
    print("No model loading, retraining, original-figure changes or holdout scoring.")
    return destinations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    options = parser.parse_args()
    generate(options.root)


if __name__ == "__main__":
    main()
