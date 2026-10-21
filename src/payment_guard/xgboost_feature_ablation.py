"""Chronological XGBoost / logistic feature-family ablations on development rows only.

The pre-existing full-model scores are immutable local reference predictions.
January and February are historical development assessments. The March/April
holdout is not loaded, fitted, scored or used to choose a model here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .calibration_saved_v2 import load_windows
from .calibration_temporal import SEED, sha256, _check_probabilities
from .train_logistic_baseline import (FEATURES, FEATURE_REMOVALS, BUDGETS,
    design_frame, evaluate_scores, fit_model, git_state, make_pipeline)
from .xgboost_chronological import (CANDIDATES, COMMON, audit_saved,
    locked_candidate_choice)

# Exactly the three predefined feature families and an unchanged full reference.
VARIANTS = ("full", "without_type", "without_category", "without_both")
EXCLUDED = {name: tuple(FEATURE_REMOVALS[name]) for name in VARIANTS}
FAMILIES = ("logistic", "xgboost")
STAGES = {
    "January": ("octnov", "jan", "A_january_predictions.parquet"),
    "February": ("octdec", "feb", "B_february_predictions.parquet"),
}
COUNT_GUARDS = {"January": (186097, 281), "February": (167990, 202)}
LOCKED_CHOICE = "depth3_240"


def assert_protocol() -> None:
    """Lock feature definitions and the configuration chosen on December AP."""
    expected = {
        "full": (), "without_type": ("type",),
        "without_category": ("category",),
        "without_both": ("type", "category"),
    }
    if EXCLUDED != expected or VARIANTS != tuple(expected):
        raise ValueError("Feature-family protocol changed")
    if any(item not in FEATURES for drop in EXCLUDED.values() for item in drop):
        raise ValueError("A requested feature does not exist")
    cfg = next((c for c in CANDIDATES if c["name"] == LOCKED_CHOICE), None)
    if cfg != {"name": "depth3_240", "max_depth": 3, "n_estimators": 240}:
        raise ValueError("Locked December configuration changed")
    if tuple(BUDGETS) != (0.001, 0.005, 0.01) or SEED != 20261005:
        raise ValueError("Original budget/tie protocol changed")


def fit_variant(training: pd.DataFrame, family: str, variant: str) -> Pipeline:
    """Fit a single matched feature-subset baseline, without any label tuning."""
    assert_protocol()
    if family not in FAMILIES or variant not in VARIANTS or variant == "full":
        raise ValueError("Only declared non-full ablation fits are permitted")
    if training.isFraud.nunique() != 2:
        raise ValueError("Both fraud classes required for fitting")
    excluded = EXCLUDED[variant]
    data = design_frame(training, log_amount=True, excluded=excluded)
    if set(data.columns) & set(excluded):
        raise ValueError("Excluded features remain in the transformed design")
    if family == "logistic":
        # Preserve baseline's convergence-warning and finite-coefficient checks.
        return fit_model(training, log_amount=True, excluded=excluded)
    cfg = next(c for c in CANDIDATES if c["name"] == LOCKED_CHOICE)
    model = Pipeline([
        ("preprocess", make_pipeline(excluded=excluded).named_steps["preprocess"]),
        ("model", XGBClassifier(**COMMON,
                                max_depth=cfg["max_depth"],
                                n_estimators=cfg["n_estimators"])),
    ])
    model.fit(data, training.isFraud.to_numpy())
    if family == "xgboost" and model.named_steps["model"].get_booster().num_boosted_rounds() != 240:
        raise RuntimeError("Ablation model did not retain 240 rounds")
    return model


def predict_variant(model: Pipeline, frame: pd.DataFrame, variant: str) -> np.ndarray:
    if variant not in VARIANTS:
        raise ValueError("Unknown feature-ablation variant")
    p = model.predict_proba(design_frame(frame, log_amount=True, excluded=EXCLUDED[variant]))[:, 1]
    _check_probabilities(p)
    return np.asarray(p, dtype=float)


def checked_reference(xgb_dir: Path, logistic_dir: Path, windows: dict,
                      source_sha: str) -> dict[str, pd.DataFrame]:
    """Reconstruct both saved model families and validate aligned full-score rows."""
    assert_protocol()
    # Includes logistic manifest, all saved artefact hashes and score reconstruction.
    audit_saved(xgb_dir, windows, source_sha, logistic_dir)
    meta = json.loads((xgb_dir / "run.json").read_text(encoding="utf-8"))
    if meta.get("chosen_configuration") != LOCKED_CHOICE:
        raise ValueError("Saved December-selected configuration disagrees")
    if locked_candidate_choice(pd.read_csv(xgb_dir / "december_candidates.csv")) != LOCKED_CHOICE:
        raise ValueError("December candidate-selection rule disagrees")
    output = {}
    seen = set()
    for month, (_, frame_key, filename) in STAGES.items():
        saved = pd.read_parquet(xgb_dir / filename)
        current = windows[frame_key]
        if (len(saved), int(saved.isFraud.sum())) != COUNT_GUARDS[month]:
            raise ValueError(f"{month} expected row/positive counts changed")
        for col in ("source_row", "timestamp", "isFraud"):
            if not saved[col].reset_index(drop=True).equals(current[col].reset_index(drop=True)):
                raise ValueError(f"{month} saved row identity differs: {col}")
        if not saved.source_row.is_unique or set(saved.source_row) & seen:
            raise ValueError("Duplicate/overlapping historical source rows")
        seen.update(saved.source_row)
        expected_ties = np.random.default_rng(SEED).permutation(len(saved))
        if not np.array_equal(saved.tie_priority.to_numpy(), expected_ties):
            raise ValueError(f"{month} tie priority differs from benchmark")
        for family in FAMILIES:
            name = family if month == "January" else f"{family}_uncalibrated"
            _check_probabilities(saved[name].to_numpy(dtype=float))
        output[month] = saved
    return output


def score_rows(month: str, frame: pd.DataFrame,
               predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Use the same rankings, top-k ceiling, AP, log loss and Brier as baseline."""
    if month not in STAGES:
        raise ValueError("Unregistered evaluation month")
    if len(frame) == 0 or frame.isFraud.nunique() != 2:
        raise ValueError("Expected nonempty binary-label evaluation frame")
    expected = {f"{family}_{variant}" for family in FAMILIES for variant in VARIANTS}
    if set(predictions) != expected:
        raise ValueError("Prediction variants must be complete")
    n, positives = len(frame), int(frame.isFraud.sum())
    tie = frame.tie_priority.to_numpy()
    if not np.array_equal(tie, np.random.default_rng(SEED).permutation(n)):
        raise ValueError("Pre-registered tie permutation differs")
    rows = []
    for family in FAMILIES:
        for variant in VARIANTS:
            name = f"{family}_{variant}"
            p = np.asarray(predictions[name], dtype=float)
            _check_probabilities(p)
            for record in evaluate_scores(frame.isFraud.to_numpy(), p, tie, probabilities=True):
                if (record["selected"] != math.ceil(record["review_fraction"] * n)
                        or record["true_positives"] + record["false_negatives"] != positives
                        or record["true_positives"] + record["false_positives"] != record["selected"]):
                    raise ValueError("Review-budget confusion counts do not reconcile")
                rows.append({"month": month, "family": family, "variant": variant,
                             "excluded": ",".join(EXCLUDED[variant]),
                             "n_source_features": len(FEATURES) - len(EXCLUDED[variant]),
                             "transactions": n, "fraud_labels": positives, **record})
    return pd.DataFrame(rows)


def compare_to_full(table: pd.DataFrame) -> pd.DataFrame:
    """Explicit within-month, within-family, within-budget comparisons only."""
    keys = ["month", "family", "review_fraction"]
    if len(table) != 2 * 2 * len(VARIANTS) * len(BUDGETS):
        raise ValueError("Not all 48 registered metric rows were recorded")
    if table.duplicated(keys + ["variant"]).any():
        raise ValueError("Duplicated metric rows")
    full = table[table.variant.eq("full")][keys + ["true_positives", "average_precision",
                                                  "log_loss", "brier_score"]]
    full = full.rename(columns={c: f"full_{c}" for c in full.columns if c not in keys})
    result = table.merge(full, on=keys, how="left", validate="many_to_one")
    for metric in ("true_positives", "average_precision", "log_loss", "brier_score"):
        result[f"delta_{metric}_vs_full"] = result[metric] - result[f"full_{metric}"]
    if result["full_true_positives"].isna().any():
        raise ValueError("Missing full reference rows")
    if not result[result.variant.eq("full")]["delta_true_positives_vs_full"].eq(0).all():
        raise ValueError("Full reference must have zero change")
    return result


def plot_summary(table: pd.DataFrame, dest: Path) -> None:
    """Show family-matched ablations by historical month without hiding values."""
    one = table.loc[np.isclose(table.review_fraction, 0.01)]
    labels = {"full": "All 13 features", "without_type": "Without type",
              "without_category": "Without category", "without_both": "Without both"}
    colors = {"logistic": "#365F7D", "xgboost": "#C27A2B"}
    fig, ax = plt.subplots(1, 2, figsize=(14.2, 5.5), layout="constrained")
    for axis, month in zip(ax, ("January", "February")):
        frame = one.loc[one.month.eq(month)]
        x = np.arange(len(VARIANTS))
        for family, shift in (("logistic", -0.18), ("xgboost", 0.18)):
            d = frame.loc[frame.family.eq(family)].set_index("variant").reindex(VARIANTS)
            if d.true_positives.isna().any():
                raise ValueError("Cannot plot incomplete family/variant combination")
            vals = d.true_positives.to_numpy(dtype=int)
            bars = axis.bar(x + shift, vals, width=0.34, color=colors[family], label=family.title())
            for bar, val in zip(bars, vals):
                axis.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                          f"{val}", ha="center", va="bottom", fontsize=9)
        pos = COUNT_GUARDS[month][1]
        axis.set_title(f"{month}: {pos} positive labels", fontsize=12, weight="bold")
        axis.set_ylim(0, pos * 1.12)
        axis.set_xticks(x, [labels[k] for k in VARIANTS], rotation=15, ha="right")
        axis.set_ylabel("Positive labels caught at 1% review budget")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
    h, labels_found = ax[0].get_legend_handles_labels()
    fig.legend(h, labels_found, loc="upper center", bbox_to_anchor=(.5, 1.055), ncol=2, frameon=False)
    fig.suptitle("Matched feature-family removal: logistic vs XGBoost", weight="bold", fontsize=15)
    fig.text(.5, -.045,
             "Historical development months | Full-model predictions frozen | Separate fit per month | Synthetic AMLNet v2",
             ha="center", fontsize=9, color="#485562")
    fig.savefig(dest, dpi=165, bbox_inches="tight")
    plt.close(fig)


def _artifact_filenames() -> tuple[str, ...]:
    return tuple(f"{month.lower()}_{family}_{variant}.joblib"
                 for month in STAGES for family in FAMILIES
                 for variant in VARIANTS if variant != "full")


def _comparison_metrics(table: pd.DataFrame) -> list[str]:
    return ["month", "family", "variant", "review_fraction", "selected", "true_positives",
            "false_positives", "false_negatives", "true_negatives", "precision", "recall",
            "average_precision", "log_loss", "brier_score"]


def assert_same_metrics(recorded: pd.DataFrame, reproduced: pd.DataFrame) -> None:
    keys = ["month", "family", "variant", "review_fraction"]
    wanted = _comparison_metrics(recorded)
    a = recorded.sort_values(keys).reset_index(drop=True)
    b = reproduced.sort_values(keys).reset_index(drop=True)
    if len(a) != 48 or list(a[keys].itertuples(index=False, name=None)) != list(b[keys].itertuples(index=False, name=None)):
        raise ValueError("Saved metrics identities or row count differ")
    for col in wanted:
        if col in keys:
            continue
        np.testing.assert_allclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float),
                                   rtol=1e-6, atol=1e-9, equal_nan=True,
                                   err_msg=f"Saved ablation metric differs: {col}")


def verify_run(output: Path, reference_dir: Path, logistic_dir: Path,
               windows: dict, source_sha: str) -> None:
    """Read-only hash, score-reconstruction and derived-metric audit."""
    meta = json.loads((output / "run.json").read_text(encoding="utf-8"))
    if (meta.get("complete") is not True or meta.get("march_april_scored") is not False
            or meta.get("input_sha256") != source_sha
            or meta.get("locked_configuration") != LOCKED_CHOICE):
        raise ValueError("Ablation manifest identity/holdout flags disagree")
    if sha256(reference_dir / "run.json") != meta.get("reference_manifest_sha256"):
        raise ValueError("Full-model manifest changed")
    if sha256(logistic_dir / "run.json") != meta.get("logistic_manifest_sha256"):
        raise ValueError("Logistic saved reference changed")
    hashes = meta.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(meta["artifact_names"]):
        raise ValueError("Ablation hash manifest incomplete")
    for fname, expected in hashes.items():
        if sha256(output / fname) != expected:
            raise ValueError(f"Ablation output hash differs: {fname}")
    reference = checked_reference(reference_dir, logistic_dir, windows, source_sha)
    all_metrics = []
    for month, (_, eval_key, _) in STAGES.items():
        fr = reference[month]
        saved = pd.read_parquet(output / f"{month.lower()}_predictions.parquet")
        if not all(saved[c].reset_index(drop=True).equals(fr[c].reset_index(drop=True))
                   for c in ("source_row", "timestamp", "isFraud", "tie_priority")):
            raise ValueError(f"Ablation {month} row identity changed")
        if len(saved) != len(windows[eval_key]):
            raise ValueError(f"Ablation {month} population changed")
        scores = {}
        for family in FAMILIES:
            for variant in VARIANTS:
                name = f"{family}_{variant}"
                actual = saved[name].to_numpy(dtype=float)
                if variant == "full":
                    ref_col = family if month == "January" else f"{family}_uncalibrated"
                    expected = fr[ref_col].to_numpy(dtype=float)
                    np.testing.assert_array_equal(actual, expected)
                else:
                    model_file = output / f"{month.lower()}_{family}_{variant}.joblib"
                    # Locally trusted saved models only; do not load unknown pickle data.
                    reconstructed = predict_variant(joblib.load(model_file), windows[eval_key], variant)
                    np.testing.assert_allclose(actual, reconstructed, rtol=1e-6, atol=1e-7,
                                               err_msg=f"Saved model differs: {month}/{name}")
                scores[name] = actual
        all_metrics.append(score_rows(month, saved, scores))
    recomputed = pd.concat(all_metrics, ignore_index=True)
    recorded = pd.read_csv(output / "metrics.csv")
    assert_same_metrics(recorded, recomputed)
    delta = compare_to_full(recomputed)
    saved_delta = pd.read_csv(output / "changes_vs_full.csv")
    assert_same_metrics(saved_delta, delta)
    keys = ["month", "family", "variant", "review_fraction"]
    a = saved_delta.sort_values(keys).reset_index(drop=True)
    b = delta.sort_values(keys).reset_index(drop=True)
    for col in ("delta_true_positives_vs_full", "delta_average_precision_vs_full",
                "delta_log_loss_vs_full", "delta_brier_score_vs_full"):
        np.testing.assert_allclose(a[col], b[col], rtol=1e-6, atol=1e-9)
    print("PASS: local model reconstruction, saved rows, fixed full references, 48 metrics, hashes and excluded holdout")


def run(source: Path, output: Path, reference_dir: Path, logistic_dir: Path) -> None:
    assert_protocol()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite versioned ablation: {output}")
    windows, source_sha = load_windows(source)  # only train + February validation
    references = checked_reference(reference_dir, logistic_dir, windows, source_sha)
    frames = {}
    staged_models = {}
    metrics = []
    for month, (fit_key, eval_key, _) in STAGES.items():
        print(f"{month}: comparing full models with six fixed-feature ablation refits", flush=True)
        frame = references[month]
        pred = {}
        for family in FAMILIES:
            base_col = family if month == "January" else f"{family}_uncalibrated"
            pred[f"{family}_full"] = frame[base_col].to_numpy(dtype=float)
            for variant in VARIANTS:
                if variant == "full":
                    continue
                print(f"  Fitting {family} / {variant}", flush=True)
                model = fit_variant(windows[fit_key], family, variant)
                staged_models[f"{month.lower()}_{family}_{variant}.joblib"] = model
                pred[f"{family}_{variant}"] = predict_variant(model, windows[eval_key], variant)
        saved = frame[["source_row", "timestamp", "isFraud", "tie_priority"]].copy()
        for name, p in pred.items():
            saved[name] = p
        frames[month] = saved
        metrics.append(score_rows(month, saved, pred))
    table = pd.concat(metrics, ignore_index=True)
    changes = compare_to_full(table)
    output.mkdir(parents=True, exist_ok=False)
    for month, frame in frames.items():
        frame.to_parquet(output / f"{month.lower()}_predictions.parquet", index=False)
    for name, model in staged_models.items():
        joblib.dump(model, output / name)
    table.to_csv(output / "metrics.csv", index=False)
    changes.to_csv(output / "changes_vs_full.csv", index=False)
    plot_summary(table, output / "ablation_at_one_percent.png")
    artifacts = sorted(p.name for p in output.iterdir() if p.is_file())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True, "march_april_scored": False,
        "input_sha256": source_sha,
        "reference_manifest_sha256": sha256(reference_dir / "run.json"),
        "logistic_manifest_sha256": sha256(logistic_dir / "run.json"),
        "locked_configuration": LOCKED_CHOICE,
        "excluded_variants": {k: list(v) for k, v in EXCLUDED.items()},
        "protocol": "Oct-Nov fits -> January; Oct-Dec fits -> February; raw rankings; no retuning",
        "feature_count": len(FEATURES), "seed": SEED, "budgets": list(BUDGETS),
        "counts": {m: {"transactions": len(frames[m]),
                       "positives": int(frames[m].isFraud.sum())} for m in STAGES},
        "package_versions": {name: version(name) for name in
                             ("numpy", "pandas", "scikit-learn", "xgboost", "joblib", "pyarrow")},
        "python": platform.python_version(),
        "git": git_state(Path(__file__).resolve().parents[2]),
        "artifact_names": artifacts,
        "artifact_sha256": {name: sha256(output / name) for name in artifacts},
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    verify_run(output, reference_dir, logistic_dir, windows, source_sha)
    view = table.loc[np.isclose(table.review_fraction, .01)]
    print(view[["month", "family", "variant", "true_positives", "false_positives",
                "average_precision", "log_loss", "brier_score"]].to_string(index=False))
    print("PASS: saved locally to", output.resolve())
    print("March-April remains unscored; no row-level outputs are for Git publication.")


def main(argv=None) -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/processed/amlnet_v2_prepared.parquet")
    parser.add_argument("--output", type=Path, default=root / "models/xgboost_ablation_v1")
    parser.add_argument("--xgb-dir", type=Path, default=root / "models/xgboost_benchmark_v1")
    parser.add_argument("--logistic-v2", type=Path, default=root / "models/calibration_saved_v2")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        windows, source_sha = load_windows(args.input)
        verify_run(args.output, args.xgb_dir, args.logistic_v2, windows, source_sha)
    else:
        run(args.input, args.output, args.xgb_dir, args.logistic_v2)


if __name__ == "__main__":
    main()
