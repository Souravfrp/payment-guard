"""Audit January/February logistic vs XGBoost positive-label selection overlap.

Read only saved January/February predictions. Never loads or scores March-April.
No row-level outputs are published or written. Existing models are not loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BUDGETS = (0.001, 0.005, 0.01)
STAGES = (
    ("January", "A_january_predictions.parquet", "logistic", "xgboost", "2026-01", 186097, 281),
    ("February", "B_february_predictions.parquet", "logistic_uncalibrated", "xgboost_uncalibrated", "2026-02", 167990, 202),
)
IDENTITY = ("source_row", "timestamp", "isFraud", "tie_priority")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_manifest(folder: Path) -> dict:
    path = folder / "run.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("complete") is not True or meta.get("march_april_scored") is not False:
        raise ValueError(f"Incomplete run or holdout flag disagrees: {folder}")
    if not isinstance(meta.get("artifact_sha256"), dict):
        raise ValueError(f"No saved artifact hashes in {path}")
    return meta


def load_verified_predictions(xgb_folder: Path, logistic_folder: Path):
    xmeta = checked_manifest(xgb_folder)
    lmeta = checked_manifest(logistic_folder)
    if xmeta.get("input_sha256") != lmeta.get("input_sha256"):
        raise ValueError("Prepared input identity differs between runs")
    if xmeta.get("existing_logistic_manifest_sha256") != digest(logistic_folder / "run.json"):
        raise ValueError("The logistic reference manifest changed since the XGBoost run")
    from payment_guard.train_logistic_baseline import SEED

    seen_rows = set()
    result = {}
    for month, filename, logistic_col, xgb_col, expected_month, n_expected, p_expected in STAGES:
        src = xgb_folder / filename
        if xmeta["artifact_sha256"].get(filename) != digest(src):
            raise ValueError(f"XGBoost predictions failed saved hash: {filename}")
        frame = pd.read_parquet(src)
        required = set(IDENTITY) | {logistic_col, xgb_col}
        if not required.issubset(frame):
            raise ValueError(f"Missing saved prediction columns: {required - set(frame)}")
        if len(frame) != n_expected or int(frame.isFraud.sum()) != p_expected:
            raise ValueError(f"{month} population or label counts differ from verified baseline")
        ts = pd.to_datetime(frame["timestamp"], errors="raise")
        if not ts.dt.strftime("%Y-%m").eq(expected_month).all():
            raise ValueError(f"Out-of-period observation in {month}")
        if frame.source_row.isna().any() or not frame.source_row.is_unique:
            raise ValueError(f"Duplicate or missing row identifiers in {month}")
        row_ids = set(frame.source_row.tolist())
        if seen_rows & row_ids:
            raise ValueError("January and February transaction identifiers overlap")
        seen_rows |= row_ids
        y = frame.isFraud.to_numpy()
        if not np.isin(y, (0, 1)).all():
            raise ValueError("Fraud labels are not binary")
        ties = frame.tie_priority.to_numpy()
        expected_ties = np.random.default_rng(SEED).permutation(len(frame))
        if not np.array_equal(ties, expected_ties):
            raise ValueError(f"Tie priority differs from pre-existing scoring rule: {month}")
        for col in (logistic_col, xgb_col):
            scores = frame[col].to_numpy(dtype=float)
            if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
                raise ValueError(f"Invalid score probabilities in {month}: {col}")

        # Independently verify row identity and the frozen logistic scores.
        old = logistic_folder / filename
        if lmeta["artifact_sha256"].get(filename) != digest(old):
            raise ValueError(f"Frozen logistic predictions failed saved hash: {filename}")
        previous = pd.read_parquet(old)
        for col in IDENTITY:
            if not frame[col].reset_index(drop=True).equals(previous[col].reset_index(drop=True)):
                raise ValueError(f"{month} logistic/XGBoost row alignment differs: {col}")
        comparison_col = "uncalibrated"
        np.testing.assert_allclose(frame[logistic_col], previous[comparison_col], rtol=1e-8, atol=1e-10,
                                   err_msg=f"{month}: logistic raw scores differ from previous saved baseline")
        result[month] = (frame, logistic_col, xgb_col)
    return result


def topk(scores, tie_priority, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    ties = np.asarray(tie_priority)
    n = len(scores)
    if scores.shape != (n,) or ties.shape != (n,) or not np.isfinite(scores).all():
        raise ValueError("Invalid input shape or scores")
    if not np.array_equal(np.sort(ties), np.arange(n)):
        raise ValueError("Tie priority is not a complete permutation")
    if not 1 <= k <= n:
        raise ValueError("Review budget outside population")
    mask = np.zeros(n, dtype=bool)
    mask[np.lexsort((ties, -scores))[:k]] = True
    return mask


def paired_row(month: str, frac: float, frame: pd.DataFrame, logistic_col: str, xgb_col: str) -> dict:
    y = frame.isFraud.to_numpy(dtype=int)
    n = len(y)
    if n == 0 or not np.isin(y, (0, 1)).all():
        raise ValueError("Invalid binary labels")
    k = math.ceil(frac * n)
    ties = frame.tie_priority.to_numpy()
    l = topk(frame[logistic_col].to_numpy(), ties, k)
    x = topk(frame[xgb_col].to_numpy(), ties, k)
    positive = y == 1
    both = int(np.count_nonzero(positive & l & x))
    only_l = int(np.count_nonzero(positive & l & ~x))
    only_x = int(np.count_nonzero(positive & ~l & x))
    neither = int(np.count_nonzero(positive & ~l & ~x))
    p = int(positive.sum())
    assert both + only_l + only_x + neither == p
    assert both + only_l == int(np.count_nonzero(positive & l))
    assert both + only_x == int(np.count_nonzero(positive & x))
    overlap = int(np.count_nonzero(l & x))
    return {
        "month": month, "review_fraction": frac, "review_budget_percent": frac * 100,
        "transactions": n, "fraud_labels": p, "selected_each_model": k,
        "both_caught": both, "only_logistic": only_l, "only_xgboost": only_x,
        "neither_caught": neither, "logistic_caught": both + only_l,
        "xgboost_caught": both + only_x, "xgboost_minus_logistic": only_x - only_l,
        "both_selected_all_labels": overlap,
        "selected_only_logistic_all_labels": k - overlap,
        "selected_only_xgboost_all_labels": k - overlap,
        "logistic_false_positives": k - both - only_l,
        "xgboost_false_positives": k - both - only_x,
    }


def make_summary(months: dict) -> pd.DataFrame:
    rows = [paired_row(month, frac, *months[month]) for month in ("January", "February")
            for frac in BUDGETS]
    table = pd.DataFrame(rows)
    jan = table.query("month == 'January' and review_fraction == 0.01").iloc[0]
    feb = table.query("month == 'February' and review_fraction == 0.01").iloc[0]
    if (int(jan.logistic_caught), int(jan.xgboost_caught)) != (271, 271):
        raise ValueError("January 1% captured positives conflict with saved aggregate results")
    if (int(feb.logistic_caught), int(feb.xgboost_caught)) != (197, 198):
        raise ValueError("February 1% captured positives conflict with saved aggregate results")
    return table


def make_chart(table: pd.DataFrame, output: Path) -> None:
    # One figure: full positive-label allocation and a detail for model-only catches.
    colors = {"both_caught":"#3D6685", "only_logistic":"#DE9F38",
              "only_xgboost":"#8A5894", "neither_caught":"#BFC5CB"}
    names = {"both_caught":"Both models", "only_logistic":"Only logistic",
             "only_xgboost":"Only XGBoost", "neither_caught":"Neither"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.3), height_ratios=[2.3, 1.0],
                             gridspec_kw={"hspace": 0.44, "wspace": 0.23})
    for i, month in enumerate(("January", "February")):
        data = table.loc[table.month.eq(month)].sort_values("review_fraction")
        x = np.arange(3)
        ax = axes[0, i]
        bottom = np.zeros(3, dtype=float)
        for key in ("both_caught", "only_logistic", "only_xgboost", "neither_caught"):
            vals = data[key].to_numpy(dtype=int)
            bars = ax.bar(x, vals, bottom=bottom, color=colors[key], width=0.66,
                          label=names[key], edgecolor="white", linewidth=0.6)
            for j, val in enumerate(vals):
                if val >= 7:
                    ax.text(x[j], bottom[j] + val / 2, str(val), ha="center", va="center",
                            fontsize=9, color="white" if key == "both_caught" else "#1B2530", weight="bold")
            bottom += vals
        npos = int(data.fraud_labels.iloc[0]); ax.set_ylim(0, npos * 1.11)
        ax.set_xticks(x, [f"{v:g}%\n({int(k):,} reviews)" for v, k in
                           zip(data.review_budget_percent, data.selected_each_model)])
        ax.set_title(f"{month} · {npos} positive labels", fontsize=12, weight="bold")
        ax.set_ylabel("Number of positive labels" if i == 0 else "")
        ax.spines[["top", "right"]].set_visible(False)
        for j, (_, row) in enumerate(data.iterrows()):
            ax.text(x[j], npos * 1.025, f"LR {int(row.logistic_caught)} | XGB {int(row.xgboost_caught)}",
                    ha="center", va="bottom", fontsize=8)
        axd = axes[1, i]
        width = 0.32
        for shift, key in [(-width / 2, "only_logistic"), (width / 2, "only_xgboost")]:
            vals = data[key].to_numpy(dtype=int)
            axd.bar(x + shift, vals, width, color=colors[key], label=names[key])
            for j, v in enumerate(vals):
                axd.text(x[j] + shift, v + .10, str(v), ha="center", va="bottom", fontsize=9)
        ymax = max(3, int(max(data.only_logistic.max(), data.only_xgboost.max())) + 2)
        axd.set_ylim(0, ymax)
        axd.set_yticks(range(0, ymax + 1, max(1, ymax // 4)))
        axd.set_ylabel("Caught only by one model" if i == 0 else "")
        axd.set_xticks(x, [f"{v:g}%" for v in data.review_budget_percent])
        axd.set_xlabel("Review budget (% of transactions)")
        axd.spines[["top", "right"]].set_visible(False)
        axd.grid(axis="y", alpha=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, .948), frameon=False)
    fig.suptitle("Which fraud-labelled transactions were captured?", fontsize=16, weight="bold", y=.997)
    fig.text(.5, .009,
             "Historical development months · Same saved rows, fixed label-independent tie priority, raw models only · Synthetic AMLNet v2",
             ha="center", fontsize=9, color="#45515c")
    fig.subplots_adjust(top=.855, bottom=.08)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xgb-dir", type=Path, default=Path("models/xgboost_benchmark_v1"))
    parser.add_argument("--logistic-dir", type=Path, default=Path("models/calibration_saved_v2"))
    parser.add_argument("--output", type=Path, default=Path("models/xgboost_disagreement_v1"))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite versioned analysis: {args.output}")
    months = load_verified_predictions(args.xgb_dir, args.logistic_dir)
    table = make_summary(months)
    args.output.mkdir(parents=True, exist_ok=False)
    csv_path = args.output / "january_february_paired_disagreements.csv"
    png_path = args.output / "january_february_paired_disagreements.png"
    table.to_csv(csv_path, index=False)
    make_chart(table, png_path)
    print("PASS: saved hash, identity, periods, tie order and captured-positive checks")
    print(table[["month", "review_budget_percent", "selected_each_model", "both_caught", "only_logistic",
                 "only_xgboost", "neither_caught", "logistic_caught", "xgboost_caught"]].to_string(index=False))
    print(f"PASS: wrote aggregate-only CSV and combined figure in {args.output.resolve()}")
    print("No individual transaction rows saved or exposed. March-April remained unscored.")


if __name__ == "__main__":
    main()
