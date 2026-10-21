"""Publish aggregate-only calibration results from verified local runs.

This command does NOT fit models, load serialized objects, inspect March–April,
modify prior run folders, stage Git, or push GitHub. It writes only new paths
under docs/, results/tables/ and results/figures/, refusing existing targets.

Usage from repository root:
  python scripts/publish_calibration_checkpoint.py --check-only
  python scripts/publish_calibration_checkpoint.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

METHODS = ("uncalibrated", "sigmoid", "isotonic")
STAGES = {"A_january": (186_097, 281, 18_610, 277),
          "B_february": (167_990, 202, 16_799, 199)}
FIGURES = {
    "A_january_shared_deciles.png": "calibration_january_shared_deciles.png",
    "B_february_shared_deciles.png": "calibration_february_shared_deciles.png",
    "A_january_predicted_vs_observed.png": "calibration_january_predicted_vs_observed.png",
    "B_february_predicted_vs_observed.png": "calibration_february_predicted_vs_observed.png",
}
TABLE_SUMMARY = "results/tables/calibration_temporal_1pct_summary.csv"
TABLE_BINS = "results/tables/calibration_shared_reliability_bins.csv"
DOC = "docs/calibration_first_results.md"
OBJECTS = {"A_base.joblib", "A_sigmoid.joblib", "A_isotonic.joblib",
           "B_base.joblib", "B_sigmoid.joblib", "B_isotonic.joblib"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_manifest(folder: Path, *, require_objects: bool = False) -> dict:
    manifest_path = folder / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("march_april_scored") is not False:
        raise ValueError(f"Incomplete or holdout-scored run: {folder}")
    if manifest.get("selected_method_on_january") != "isotonic":
        raise ValueError("January-selected method is not the originally declared isotonic")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Run has no recorded artifacts")
    if require_objects and not OBJECTS.issubset(hashes):
        raise ValueError("Saved v2 run lacks the six fitted object hashes")
    for name, expected in sorted(hashes.items()):
        if name != Path(name).name or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Invalid artifact name or SHA-256 value")
        path = folder / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Artifact fingerprint mismatch: {path}")
    return manifest


def checked_metrics(v2: Path, v1: Path, v2_manifest: dict, v1_manifest: dict) -> pd.DataFrame:
    if v2_manifest.get("input_sha256") != v1_manifest.get("input_sha256"):
        raise ValueError("Original and saved v2 runs used different prepared input fingerprints")
    cur = pd.read_csv(v2 / "metrics.csv")
    prev = pd.read_csv(v1 / "metrics.csv")
    sort_by = ["stage", "method", "review_fraction"]
    required = set(sort_by) | {"log_loss", "brier_score", "average_precision", "selected",
                               "true_positives", "false_positives", "false_negatives", "precision", "recall"}
    if len(cur) != 18 or len(prev) != 18 or not required.issubset(cur.columns):
        raise ValueError("Expected 18 metric rows and the declared metric columns")
    if set(cur.columns) != set(prev.columns):
        raise ValueError("v1 and v2 metric columns disagree")
    cur = cur.sort_values(sort_by).reset_index(drop=True)
    prev = prev.sort_values(sort_by).reset_index(drop=True)
    if cur.duplicated(sort_by).any() or set(cur.stage) != set(STAGES) or set(cur.method) != set(METHODS):
        raise ValueError("Invalid metric rows or duplicated method/budget combinations")
    for stage in STAGES:
        for method in METHODS:
            subs = cur.loc[(cur.stage == stage) & (cur.method == method)]
            if not np.allclose(sorted(subs.review_fraction.to_numpy()), [0.001, 0.005, 0.01], atol=1e-12):
                raise ValueError("Unexpected or missing review budgets")
    for col in cur.columns:
        if pd.api.types.is_numeric_dtype(cur[col]) and pd.api.types.is_numeric_dtype(prev[col]):
            np.testing.assert_allclose(cur[col].to_numpy(dtype=float), prev[col].to_numpy(dtype=float),
                                       atol=1e-9, rtol=1e-7, equal_nan=True,
                                       err_msg=f"v1-v2 changed metric: {col}")
        elif not cur[col].equals(prev[col]):
            raise ValueError(f"v1-v2 changed metric: {col}")
    summary = cur.loc[np.isclose(cur.review_fraction, .01)].copy()
    for stage, (_, positives, _, _) in STAGES.items():
        rows = summary.loc[summary.stage == stage]
        if len(rows) != 3:
            raise ValueError("Missing stage/method rows at 1%")
        if not (rows.true_positives + rows.false_negatives).eq(positives).all():
            raise ValueError("Positive-label totals do not reconcile")
        if not (rows.selected == (1861 if stage == "A_january" else 1680)).all():
            raise ValueError("Incorrect declared 1% review workload")
        if not (rows.true_positives + rows.false_positives).eq(rows.selected).all():
            raise ValueError("Review confusion counts do not reconcile")
    jan = summary.loc[summary.stage == "A_january"].set_index("method")
    best = min(METHODS, key=lambda method: (float(jan.loc[method, "log_loss"]), METHODS.index(method)))
    if best != "isotonic":
        raise ValueError("January log-loss choice no longer isotonic")
    return summary[["stage", "method", "review_fraction", "selected", "true_positives",
                    "false_positives", "false_negatives", "precision", "recall", "average_precision",
                    "log_loss", "brier_score"]].reset_index(drop=True)


def checked_bins(folder: Path, v1: Path) -> pd.DataFrame:
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if report.get("source_selected_method") != "isotonic" or not report.get("method_selection_unchanged"):
        raise ValueError("Visualization report has changed selection or method")
    if sha256(v1 / "run.json") != report.get("source_manifest_sha256"):
        raise ValueError("Figures do not derive from the verified original v1 run")
    hashes = report.get("artifacts_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Missing visualization artifact hashes")
    for name in ["shared_reliability_bins.csv", *FIGURES]:
        if name not in hashes or sha256(folder / name) != hashes[name]:
            raise ValueError(f"Changed or missing visualization source: {name}")
    bins = pd.read_csv(folder / "shared_reliability_bins.csv")
    required = {"stage", "bin", "rows", "positives", "observed_rate", "boundary_split_base_tie"} | {
        f"mean_probability_{m}" for m in METHODS}
    if len(bins) != 20 or not required.issubset(bins.columns):
        raise ValueError("Expected 20 aggregate shared-bin rows")
    if not bins.columns.is_unique or bins.duplicated(["stage", "bin"]).any():
        raise ValueError("Duplicate bin schema or identifiers")
    if "source_row" in " ".join(bins.columns).lower():
        raise ValueError("Source record identifiers must never be published")
    for stage, (n, p, top_n, top_p) in STAGES.items():
        group = bins.loc[bins.stage == stage].sort_values("bin")
        if len(group) != 10 or group.bin.tolist() != list(range(1, 11)):
            raise ValueError("Shared rank-decile bins are not exactly 1..10")
        if int(group.rows.sum()) != n or int(group.positives.sum()) != p:
            raise ValueError("Shared-bin transaction/positive totals disagree")
        last = group.iloc[-1]
        if int(last.rows) != top_n or int(last.positives) != top_p:
            raise ValueError("Highest-decile counts have changed")
        np.testing.assert_allclose(group.observed_rate, group.positives / group.rows, atol=1e-11)
        for method in METHODS:
            if not group[f"mean_probability_{method}"].between(0, 1).all():
                raise ValueError("Invalid group mean probability")
    for name in FIGURES:
        with (folder / name).open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"Expected a valid PNG header: {name}")
    return bins.sort_values(["stage", "bin"]).reset_index(drop=True)


def render_report(summary: pd.DataFrame, bins: pd.DataFrame) -> str:
    def stat(stage: str, method: str, metric: str):
        return float(summary.loc[(summary.stage == stage) & (summary.method == method), metric].iloc[0])
    def metric_rows(stage: str) -> str:
        result = []
        for method in METHODS:
            tp = round(stat(stage, method, "true_positives"))
            fp = round(stat(stage, method, "false_positives"))
            result.append(f"| {method} | {stat(stage, method, 'log_loss'):.6f} | "
                          f"{stat(stage, method, 'brier_score'):.6f} | "
                          f"{stat(stage, method, 'average_precision'):.6f} | "
                          f"{tp} | {fp} | {100 * stat(stage, method, 'recall'):.2f}% |")
        return "\n".join(result)
    jan_top = bins.loc[(bins.stage == "A_january") & (bins.bin == 10)].iloc[0]
    feb_top = bins.loc[(bins.stage == "B_february") & (bins.bin == 10)].iloc[0]
    doc = f"""# What I found in the chronological calibration experiment

## Research question and the boundary I kept

I wanted to know whether the original log-amount logistic scores could be interpreted more reliably as probabilities, separately from their ability to rank transactions for review. I kept the original baseline and ablation runs unchanged. All fitted models and calibrators in this experiment used only October 2025 through January 2026 for fitting or calibration, and only January and February for reported evaluation. I did not score the reserved March–April holdout. February had already been explored in earlier work, so I do not call it an untouched final test.

My input is the same synthetic AMLNet v2 preparation. Row-level predictions, prepared data, six fitted joblib objects, and the unmodified v1/v2 run folders stay local and outside Git. These public figures and tables contain **aggregates only**. A GitHub reader cannot reproduce the saved-run audit from a clone alone without obtaining and preparing the dataset.

## Chronological design and model choices

In stage A, I fitted the 13-feature logistic log-amount pipeline on October–November (292,668 rows; 452 positives), fitted calibration mappings on December (190,055 rows; 198 positives), and selected a method using **January** (186,097 rows; 281 positives). The predeclared primary choice was January log loss, with Brier score used within a 1e-6 near-tie and a fixed simplicity order. The selected method was **isotonic regression**.

In stage B, I refitted the same base-model family on October–December (482,723 rows; 650 positives), fitted each calibration mapping on January, and reported their February performance (167,990 rows; 202 positives). I did not change the selection based on February. All three February calibration choices share the same stage-B base model; my original earlier full-feature logistic baseline used a *different training window*, so a comparison with that baseline does not isolate a calibration effect.

For a base score p = sigmoid(z), I compared: unchanged p, a monotone sigmoid mapping q = sigmoid(a logit(p) + b) with a >= 0, and isotonic regression with a nondecreasing fitted mapping f(p). The sigmoid calibrator minimizes calibration-period negative log likelihood subject to its slope constraint; isotonic minimizes squared error under its monotonicity constraint. Neither adds new information to the original feature representation. Monotonic transforms generally preserve ranking, but flat maps and clipping can create ties. The saved seeded label-independent tie order controls retrospective review selection.

## Actual numerical results

Lower values are better for log loss and Brier score. Recall describes top-1% review capture, not probability accuracy.

**Stage A — January method selection (1% reviews):**

| Method | Log loss | Brier | Average precision | Caught positives | False alerts | Recall |
|---|---:|---:|---:|---:|---:|---:|
{metric_rows('A_january')}

Isotonic's January log-loss improvement over the uncalibrated model was **{stat('A_january','uncalibrated','log_loss')-stat('A_january','isotonic','log_loss'):.6f}** in absolute units. This is a small observed difference, not evidence of statistical significance or robust superiority across months.

**Stage B — February historical comparison (1% reviews; not used for selection):**

| Method | Log loss | Brier | Average precision | Caught positives | False alerts | Recall |
|---|---:|---:|---:|---:|---:|---:|
{metric_rows('B_february')}

Sigmoid had the lowest February log loss and Brier score, whereas January had selected isotonic. I will keep that method-selection result instead of changing it after inspecting February. This contrast suggests potential calibration sensitivity across historical windows; two evaluation months do not establish formal temporal drift. Isotonic also created score ties that lowered average precision relative to the two strictly ordered variants.

The original October–January baseline caught 196 of 202 February positive labels at 1,680 reviews. The new stage-B base model caught {round(stat('B_february','uncalibrated','true_positives'))} of 202. I cannot attribute the one-case difference to calibration because the original baseline and stage-B model were fitted using different months.

## Corrected visual comparison

The original method-specific probability-quantile graphs had unequal bin counts: isotonic's ties produced four displayed groups in January and two in February, versus ten for the other methods. Those bin labels did not identify the same transactions. For publication, I fixed ten equal-count **uncalibrated-score rank deciles** and calculated each method's mean prediction on **exactly the same rows**. The saved label-independent priority handles rank ties at boundaries. Within a month, the black observed curve is now identical across methods.

- January's highest shared decile contained **{int(jan_top.positives)} positives / {int(jan_top.rows):,} rows**, observed rate **{100*jan_top.observed_rate:.4f}%**.
- February's highest shared decile contained **{int(feb_top.positives)} positives / {int(feb_top.rows):,} rows**, observed rate **{100*feb_top.observed_rate:.4f}%**.

![January shared-decile observed versus predicted rates](../results/figures/calibration_january_shared_deciles.png)

![February shared-decile observed versus predicted rates](../results/figures/calibration_february_shared_deciles.png)

I also plotted mean predicted rate against observed rate, with the ideal-agreement line. These are **aggregate decile summaries**, not evidence that individual scores are calibrated:

![January predicted versus observed on the same deciles](../results/figures/calibration_january_predicted_vs_observed.png)

![February predicted versus observed on the same deciles](../results/figures/calibration_february_predicted_vs_observed.png)

The high-risk decile dominates each linear-scale graph. Most lower bins contain zero or very few positive labels, so fine-grained conclusions about very small predicted probabilities are unsupported. In particular, the observed decile shares cannot be treated as the probability for every transaction inside that decile. The January and February deciles are constructed separately from each month's original scores, not identical transactions across months.

## Verification and mathematical limits

For binary labels y_i and probabilities p_i, Brier score is mean[(p_i-y_i)^2]; log loss is the negative mean of [y_i log(p_i)+(1-y_i)log(1-p_i)]. A lower value in one observed month describes that sample, not a guarantee of future calibration. Average precision and fixed-budget precision/recall concern ranking; they answer a different question. For fixed k, precision = TP/k and recall = TP/P, so all three methods can tie at the declared 1% budget even if their probability losses differ.

I reran the experiment in a new versioned v2 folder that saved **six fitted objects** (two base pipelines, two sigmoid mappings, two isotonic mappings). The local v2 verifier checked artifact SHA-256 digests before loading my own trusted joblib files, verified row identity and chronology, reconstructed saved probabilities and reported metrics, and confirmed that v2 reproduced v1 predictions and results. The full local test suite passed **116 tests** at this checkpoint. This is a user-observed local audit, not an independent external rerun or GitHub CI certification.

For reproduction, see [my chronological protocol](calibration_temporal_protocol.md), [my saved-run verification notes](calibration_saved_run_verification.md) and [the corrected-bin explanation](calibration_visualization_revision.md). Aggregate values are in [the 1% results table](../results/tables/calibration_temporal_1pct_summary.csv) and [the shared-bin table](../results/tables/calibration_shared_reliability_bins.csv). The fitted objects, full prepared dataset, and row-level predictions remain excluded from the repository.

## Next unanswered questions

I still need forward-only stability checks beyond the two observed evaluations, uncertainty quantification appropriate for rare labels, a capacity-aware review rule that can operate without ranking an entire future month, and a frozen final assessment. Synthetic data, unusually strong category patterns, assumed decision-time feature availability, and already-explored February results limit any claim of real payment fraud performance or operational benefit. The reserved March–April holdout must stay unscored until those choices are fixed.
"""
    return doc


def make_outputs(root: Path, *, write: bool):
    v1 = root / "models/calibration_temporal_v1"
    v2 = root / "models/calibration_saved_v2"
    viz = root / "models/calibration_visual_refresh_v1"
    first = verified_manifest(v1)
    second = verified_manifest(v2, require_objects=True)
    summary = checked_metrics(v2, v1, second, first)
    verified_manifest_artifacts = checked_bins(viz, v1)
    targets = [TABLE_SUMMARY, TABLE_BINS, DOC, *[f"results/figures/{name}" for name in FIGURES.values()]]
    if any((root / dest).exists() for dest in targets):
        existing = [dest for dest in targets if (root / dest).exists()]
        raise FileExistsError("Refusing to replace existing publication output(s): " + ", ".join(existing))
    print("PASS: v1/v2 artifact hashes, matching metric rows, January choice, shared-bin counts and figure hashes")
    print("January: 186,097 rows / 281 positives; highest shared decile 277 / 18,610")
    print("February: 167,990 rows / 202 positives; highest shared decile 199 / 16,799")
    print("Selected on January: isotonic. February results were not used to reselect.")
    if not write:
        print("CHECK ONLY: no repository files written")
        return targets
    with tempfile.TemporaryDirectory(prefix="pg-calibration-publish-") as temp:
        temp_root = Path(temp)
        (temp_root / TABLE_SUMMARY).parent.mkdir(parents=True)
        summary.to_csv(temp_root / TABLE_SUMMARY, index=False, float_format="%.15g")
        (temp_root / TABLE_BINS).parent.mkdir(parents=True, exist_ok=True)
        verified_manifest_artifacts.to_csv(temp_root / TABLE_BINS, index=False, float_format="%.15g")
        (temp_root / DOC).parent.mkdir(parents=True)
        (temp_root / DOC).write_text(render_report(summary, verified_manifest_artifacts), encoding="utf-8")
        for source, target in FIGURES.items():
            dest = temp_root / "results/figures" / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(viz / source, dest)
        # Precompute full set before touching the research repository.
        for dest in targets:
            if not (temp_root / dest).is_file() or (root / dest).exists():
                raise RuntimeError(f"Publication preflight failed: {dest}")
        installed = []
        try:
            for dest in targets:
                path = root / dest
                path.parent.mkdir(parents=True, exist_ok=True)
                # 'x' exclusive mode prevents accidental overwrites if files appeared.
                with (temp_root / dest).open("rb") as source, path.open("xb") as output:
                    shutil.copyfileobj(source, output)
                installed.append(path)
        except Exception:
            for path in installed:
                path.unlink(missing_ok=True)
            raise
    print("PASS: published 1 aggregate results document, 2 aggregate CSV tables and 4 plots")
    print("Source model artifacts, row-level data, v1/v2 runs and final holdout were not modified")
    for name in targets:
        print("  " + name)
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    make_outputs(args.root.resolve(), write=not args.check_only)


if __name__ == "__main__":
    main()
