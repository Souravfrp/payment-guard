"""Compare a bounded XGBoost search with the saved logistic study. No test rows.

This run is deliberately *development evidence*, not a fresh February test.
The final March-April 2026 holdout is not read, fitted, calibrated or scored.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .calibration_temporal import (METHODS, SEED, _check_probabilities,
    calibrated_scores, fit_calibrators, scoring_report, sha256)
from .calibration_saved_v2 import load_windows
from .train_logistic_baseline import (FEATURES, design_frame, fit_model,
    git_state, make_pipeline)

# Lock these configurations before reading December labels. All use unit weights.
CANDIDATES = (
    {"name": "depth2_120", "max_depth": 2, "n_estimators": 120},
    {"name": "depth2_240", "max_depth": 2, "n_estimators": 240},
    {"name": "depth3_240", "max_depth": 3, "n_estimators": 240},
)
COMMON = {"objective": "binary:logistic", "eval_metric": "logloss",
          "tree_method": "hist", "device": "cpu", "n_jobs": 4,
          "random_state": SEED, "learning_rate": 0.05,
          "subsample": 1.0, "colsample_bytree": 1.0,
          "min_child_weight": 5.0, "reg_lambda": 1.0,
          "reg_alpha": 0.0, "gamma": 0.0,
          "scale_pos_weight": 1.0, "base_score": 0.5, "importance_type": "gain"}


def build_booster(config: dict) -> Pipeline:
    """Reuse the same logistic feature transform and fitted-preprocessing structure."""
    if config not in CANDIDATES:
        raise ValueError("An unregistered XGBoost configuration was requested")
    preprocess = make_pipeline().named_steps["preprocess"]
    model = XGBClassifier(**COMMON, max_depth=config["max_depth"],
                          n_estimators=config["n_estimators"])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def fit_booster(training: pd.DataFrame, config: dict) -> Pipeline:
    if training["isFraud"].nunique() != 2:
        raise ValueError("Both classes required for XGBoost training")
    model = build_booster(config)
    model.fit(design_frame(training, log_amount=True), training["isFraud"].to_numpy())
    if model.named_steps["model"].get_booster().num_boosted_rounds() != config["n_estimators"]:
        raise RuntimeError("Fitted tree count differs from locked configuration")
    return model


def probs(pipeline, frame: pd.DataFrame):
    p = pipeline.predict_proba(design_frame(frame, log_amount=True))[:, 1]
    _check_probabilities(p)
    return p


def locked_candidate_choice(table: pd.DataFrame) -> str:
    """December AP desc, then log loss asc within AP near-ties, then fixed order."""
    if list(table["candidate"]) != [x["name"] for x in CANDIDATES]:
        raise ValueError("Candidate table is incomplete or not in declared order")
    if not np.isfinite(table[["average_precision", "log_loss"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite December model selection value")
    top_ap = float(table.average_precision.max())
    near = table.loc[table.average_precision >= top_ap - 1e-6]
    best_loss = float(near.log_loss.min())
    return next(cfg["name"] for cfg in CANDIDATES if
                cfg["name"] in set(near.loc[near.log_loss <= best_loss + 1e-10, "candidate"]))


def candidate_sweep(octnov, dec):
    """Tuning sees December only; January/February outcomes cannot affect choice."""
    records = []
    for cfg in CANDIDATES:
        print(f"December development fit: {cfg['name']}", flush=True)
        model = fit_booster(octnov, cfg)
        scores = probs(model, dec)
        records.append({"candidate": cfg["name"], "max_depth": cfg["max_depth"],
                        "n_estimators": cfg["n_estimators"],
                        "average_precision": float(average_precision_score(dec.isFraud, scores)),
                        "log_loss": float(log_loss(dec.isFraud, scores, labels=[0, 1]))})
    table = pd.DataFrame(records)
    return locked_candidate_choice(table), table


def aligned_logistic_reference(saved_dir: Path, feb: pd.DataFrame, source_sha: str, ties):
    """Treat verified existing logistic v2 scores as a locked reference, not a refit.

    Read only a *locally trusted* saved-run manifest/prediction file after hashes.
    No logistic joblib objects are loaded by this function.
    """
    manifest = json.loads((saved_dir / "run.json").read_text(encoding="utf-8"))
    if (manifest.get("complete") is not True or manifest.get("march_april_scored") is not False
            or manifest.get("input_sha256") != source_sha):
        raise ValueError("Existing logistic run provenance disagrees")
    prediction_name = "B_february_predictions.parquet"
    if sha256(saved_dir / prediction_name) != manifest["artifact_sha256"][prediction_name]:
        raise ValueError("Existing logistic February prediction file failed hash verification")
    saved = pd.read_parquet(saved_dir / prediction_name)
    for col in ("source_row", "timestamp", "isFraud"):
        if not saved[col].reset_index(drop=True).equals(feb[col].reset_index(drop=True)):
            raise ValueError(f"Existing logistic row alignment failed: {col}")
    if not np.array_equal(np.asarray(saved.tie_priority), np.asarray(ties)):
        raise ValueError("Tie-breaking priority differs from locked logistic run")
    prediction = {m: np.asarray(saved[m], dtype=float) for m in METHODS}
    for scores in prediction.values():
        _check_probabilities(scores)
    return prediction


def ranking_deciles(labels, source_rows, tie_priority, reference_scores, predictions):
    """Use a single logistic-reference cohort for all six probability series."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(reference_scores, dtype=float)
    refs = np.asarray(source_rows)
    ties = np.asarray(tie_priority)
    if len(y) != len(p) or len(refs) != len(y) or len(ties) != len(y):
        raise ValueError("Rank cohorts are misaligned")
    _check_probabilities(p)
    if not np.array_equal(np.sort(ties), np.arange(len(y))):
        raise ValueError("Invalid tie priority")
    for values in predictions.values():
        _check_probabilities(np.asarray(values, dtype=float))
        if len(values) != len(y):
            raise ValueError("Prediction length mismatch")
    order = np.lexsort((ties, -p))
    decile = np.empty(len(y), dtype=int)
    for i, members in enumerate(np.array_split(order, 10)):
        decile[members] = 10 - i  # 10 = highest-scored cohort
    rows = []
    for d in range(1, 11):
        mask = decile == d
        for label, score in predictions.items():
            rows.append({"reference_decile": d, "series": label,
                         "rows": int(mask.sum()), "positives": int(y[mask].sum()),
                         "observed_rate": float(y[mask].mean()),
                         "mean_probability": float(np.mean(np.asarray(score)[mask]))})
    return pd.DataFrame(rows)


def feature_importance_table(pipeline) -> pd.DataFrame:
    """Aggregate one-hot/sine/cosine gain to the original 13 feature groups.

    Gain measures split usefulness in *this fitted ensemble*, not causal impact.
    """
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    gains = pipeline.named_steps["model"].feature_importances_
    if len(names) != len(gains) or not np.isfinite(gains).all():
        raise ValueError("XGBoost importance vector does not match transformed features")
    categories = [x for x in FEATURES if x.startswith("metadata.") or x in ("type", "category")]
    records = {name: 0.0 for name in FEATURES}
    for transformed, gain in zip(names, gains):
        if transformed.startswith("scaled__"):
            source = transformed.removeprefix("scaled__")
        elif transformed.startswith("cyclic__"):
            full = transformed.removeprefix("cyclic__")
            source = full.removesuffix("_sin").removesuffix("_cos")
        elif transformed.startswith("categorical__"):
            full = transformed.removeprefix("categorical__")
            matches = [name for name in categories if full.startswith(name + "_")]
            if len(matches) != 1:
                raise ValueError(f"Unmapped categorical feature: {transformed}")
            source = matches[0]
        else:
            raise ValueError(f"Unexpected transformed feature: {transformed}")
        if source not in records:
            raise ValueError(f"Unknown feature group: {source}")
        records[source] += float(gain)
    return pd.DataFrame([{"feature": name, "normalized_gain": value}
                         for name, value in records.items()]).sort_values(
                             ["normalized_gain", "feature"], ascending=[False, True]).reset_index(drop=True)


def plot_importance(table: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6.8))
    positions = np.arange(len(table))
    ax.barh(positions, table.normalized_gain, color="#206582", edgecolor="#132B3B")
    ax.set_yticks(positions, table.feature)
    ax.invert_yaxis()
    ax.set_xlabel("Sum of normalized XGBoost split gain")
    ax.set_title("Stage B: original 13 feature groups (descriptive, not causal)")
    ax.grid(axis="x", alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _assert_decile_report(table: pd.DataFrame, n: int, positives: int):
    for label, part in table.groupby("series"):
        if len(part) != 10 or part.rows.sum() != n or part.positives.sum() != positives:
            raise ValueError(f"Shared cohort totals invalid for {label}")


def plot_reports(january: pd.DataFrame, february: pd.DataFrame, bins: pd.DataFrame, output: Path):
    """High-contrast accessible line/marker palette, labels and explicit units."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = {"logistic": "#15334F", "xgboost": "#C45B15",
         "logistic_uncalibrated": "#15334F", "xgboost_uncalibrated": "#C45B15",
         "logistic_sigmoid": "#256E8A", "xgboost_sigmoid": "#8A407E",
         "logistic_isotonic": "#32845C", "xgboost_isotonic": "#996B15"}
    marker = {k: ("o" if "logistic" in k else "s") for k in c}
    def format_axes(ax):
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    # Month-by-month ranking diagnostics: model-family comparison with no calibrated Jan scores.
    j = january.loc[np.isclose(january.review_fraction, .01)].set_index("series")
    f = february.loc[np.isclose(february.review_fraction, .01)].set_index("series")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, metric, title in ((axes[0], "average_precision", "January: average precision"),
                              (axes[1], "true_positives", "January: positives caught at 1%")):
        ids = ["logistic", "xgboost"]
        vals = [float(j.loc[k, metric]) for k in ids]
        ax.bar(ids, vals, color=[c[k] for k in ids], edgecolor="#212121", width=.58)
        ax.set_title(title)
        ax.set_ylabel("AP (unitless)" if metric == "average_precision" else "Positive-label count")
        for idx, v in enumerate(vals):
            ax.text(idx, v, f"{v:.4f}" if metric == "average_precision" else str(int(v)),
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.18 if max(vals) else 1)
        format_axes(ax)
    fig.suptitle("Same Oct–Nov fitting rows and 13 features; January is explored historical data")
    fig.tight_layout()
    fig.savefig(output / "january_family_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))
    methods = ["uncalibrated", "sigmoid", "isotonic"]
    for ax, metric, title in zip(axes,
        ("average_precision", "true_positives", "log_loss"),
        ("Average precision", "Positives caught at 1%", "Log loss (lower better)")):
        x = np.arange(3)
        for shift, family in ((-.19, "logistic"), (.19, "xgboost")):
            labels = [f"{family}_{m}" for m in methods]
            vals = [float(f.loc[name, metric]) for name in labels]
            ax.bar(x + shift, vals, .37, label=family.title(), color=c[family],
                   edgecolor="#303030", linewidth=.7, hatch="" if family=="logistic" else "//")
        ax.set_xticks(x, ["Raw", "Sigmoid", "Isotonic"])
        ax.set_title(title)
        ax.set_ylabel("AP" if metric=="average_precision" else "Count" if metric=="true_positives" else "Mean negative log likelihood")
        format_axes(ax)
        if metric=="true_positives":
            ax.legend(fontsize=9)
    fig.suptitle("February historical comparison — identical Oct–Dec fitting and Jan calibration periods")
    fig.tight_layout()
    fig.savefig(output / "february_family_metrics.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6.8))
    labels = list(c)[2:]
    for name in labels:
        part = bins.loc[bins.series.eq(name)].sort_values("reference_decile")
        ax.plot(part.reference_decile, 100*part.mean_probability, label=name.replace("_", " / "),
                color=c[name], marker=marker[name], linestyle="-" if "logistic" in name else "--", linewidth=1.7)
    observed = bins.loc[bins.series.eq("logistic_uncalibrated")].sort_values("reference_decile")
    ax.plot(observed.reference_decile, 100*observed.observed_rate, label="Observed labels (same rows)",
            color="#171717", marker="D", linewidth=2.6)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Decile of frozen uncalibrated logistic score (1 low, 10 high)")
    ax.set_ylabel("Mean predicted / observed positive rate (%)")
    ax.set_title("February: same logistic-defined transactions in each bin for every method")
    ax.grid(alpha=.2)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "february_shared_cohort_reliability.png", dpi=160)
    plt.close(fig)

    # Full precision-recall curves are saved by plot_pr_curves.


def plot_pr_curves(feb: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(9.3, 6.2))
    for name, color, ls in (("logistic_uncalibrated", "#15334F", "-"),
                            ("xgboost_uncalibrated", "#C45B15", "--")):
        precision, recall, _ = precision_recall_curve(feb.isFraud, feb[name])
        ap = average_precision_score(feb.isFraud, feb[name])
        ax.step(recall, precision, where="post", color=color, linestyle=ls,
                label=f"{name.split('_')[0].title()} (AP={ap:.4f})", linewidth=2)
    ax.set(xlabel="Recall", ylabel="Precision", title="February historical precision–recall curves (unadjusted scores)",
           xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def model_report(labels, predictions, tie_priority, stage):
    chunks = []
    for series, p in predictions.items():
        from .train_logistic_baseline import evaluate_scores
        chunks.extend({"stage": stage, "series": series, **row}
                      for row in evaluate_scores(labels, p, tie_priority, probabilities=True))
    return pd.DataFrame(chunks)


def same_metric_rows(expected, actual):
    a = expected.sort_values(["stage", "series", "review_fraction"]).reset_index(drop=True)
    b = actual.sort_values(["stage", "series", "review_fraction"]).reset_index(drop=True)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        raise ValueError("Metric report shape changed")
    for col in a:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            np.testing.assert_allclose(a[col], b[col], rtol=1e-7, atol=1e-9, equal_nan=True,
                                       err_msg=f"Metric verification failed: {col}")
        elif not a[col].equals(b[col]):
            raise ValueError(f"Metric verification failed: {col}")


def audit_saved(output: Path, windows, source_sha: str, logistic_saved: Path | None = None):
    """Reconstruct XGBoost predictions with verified local artifacts."""
    manifest = json.loads((output / "run.json").read_text())
    if manifest.get("complete") is not True or manifest.get("march_april_scored") is not False:
        raise ValueError("Run incomplete/holdout flag disagrees")
    if manifest["input_sha256"] != source_sha:
        raise ValueError("Prepared dataset SHA-256 differs")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256(output / name) != digest:
            raise ValueError(f"Saved artifact failed SHA-256: {name}")
    if manifest["chosen_configuration"] not in [k["name"] for k in CANDIDATES]:
        raise ValueError("Unexpected chosen hyperparameters")
    chosen = locked_candidate_choice(pd.read_csv(output / "december_candidates.csv"))
    if chosen != manifest["chosen_configuration"]:
        raise ValueError("December choice not reproducible")
    if logistic_saved is not None:
        if sha256(logistic_saved / "run.json") != manifest["existing_logistic_manifest_sha256"]:
            raise ValueError("Frozen logistic reference manifest changed")
        feb = windows["feb"]
        fixed_ties = np.random.default_rng(SEED).permutation(len(feb))
        old = aligned_logistic_reference(logistic_saved, feb, source_sha, fixed_ties)
        preserved = pd.read_parquet(output / "B_february_predictions.parquet")
        for method in METHODS:
            np.testing.assert_allclose(old[method], preserved[f"logistic_{method}"],
                                       rtol=0, atol=0,
                                       err_msg=f"Frozen logistic {method} predictions changed")
    # Only unpickle locally generated files that the user controls and trusts.
    a = joblib.load(output / "A_xgboost.joblib")
    b = joblib.load(output / "B_xgboost.joblib")
    calibrators = {m: joblib.load(output / f"B_xgboost_{m}.joblib")
                   for m in ("sigmoid", "isotonic")}
    for stage, frame in (("A_january", windows["jan"]), ("B_february", windows["feb"])):
        saved = pd.read_parquet(output / f"{stage}_predictions.parquet")
        for col in ("source_row", "timestamp", "isFraud"):
            if not saved[col].reset_index(drop=True).equals(frame[col].reset_index(drop=True)):
                raise ValueError(f"Saved row identity failed: {stage}/{col}")
        if stage=="A_january":
            reconstructed = probs(a, frame)
            np.testing.assert_allclose(reconstructed, saved.xgboost, rtol=1e-6, atol=1e-7)
            expected = model_report(frame.isFraud, {n: saved[n].to_numpy() for n in
                ("logistic", "xgboost")}, saved.tie_priority, stage)
        else:
            reconstructed = calibrated_scores(probs(b, frame), calibrators)
            for method in METHODS:
                np.testing.assert_allclose(reconstructed[method], saved[f"xgboost_{method}"], rtol=1e-6, atol=1e-7)
            expected = model_report(frame.isFraud, {n: saved[n].to_numpy() for n in
                (f"{family}_{method}" for family in ("logistic", "xgboost") for method in METHODS)},
                saved.tie_priority, stage)
        recorded = pd.read_csv(output / "metrics.csv").loc[lambda d:d.stage.eq(stage)]
        same_metric_rows(recorded, expected)
    print("PASS: saved model reconstruction, row IDs, hashes, selection, metrics and excluded holdout", flush=True)


def run(source: Path, output: Path, logistic_saved: Path):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite versioned results: {output}")
    windows, source_sha = load_windows(source)  # hard-filters to train + February only
    jan, feb = windows["jan"], windows["feb"]
    # Fail before any expensive fitting if the older frozen reference is unavailable or invalid.
    ties_feb = np.random.default_rng(SEED).permutation(len(feb))
    b_logit_preds = aligned_logistic_reference(logistic_saved, feb, source_sha, ties_feb)
    chosen, choices = candidate_sweep(windows["octnov"], windows["dec"])
    config = next(c for c in CANDIDATES if c["name"] == chosen)
    print("Chosen by December AP:", chosen, flush=True)
    # Two comparable fitted model families, not multiple recalibration substitutes.
    a_xgb = fit_booster(windows["octnov"], config)
    a_logit = fit_model(windows["octnov"], log_amount=True)
    ties_jan = np.random.default_rng(SEED).permutation(len(jan))
    a_preds = {"xgboost": probs(a_xgb, jan), "logistic": probs(a_logit, jan)}
    jan_metrics = model_report(jan.isFraud.to_numpy(), a_preds, ties_jan, "A_january")
    # Match the previous chronological benchmark: Oct-Dec fitting, Jan calibration, Feb scoring.
    b_xgb = fit_booster(windows["octdec"], config)
    cal = fit_calibrators(probs(b_xgb, jan), jan.isFraud.to_numpy())
    b_xgb_preds = calibrated_scores(probs(b_xgb, feb), cal)
    b_preds = {f"{family}_{method}": scores for family, mapping in
               (("logistic", b_logit_preds), ("xgboost", b_xgb_preds))
               for method, scores in mapping.items()}
    feb_metrics = model_report(feb.isFraud.to_numpy(), b_preds, ties_feb, "B_february")
    bins = ranking_deciles(feb.isFraud, feb.source_row, ties_feb,
                            b_logit_preds["uncalibrated"], b_preds)
    _assert_decile_report(bins, len(feb), int(feb.isFraud.sum()))
    output.mkdir(parents=True, exist_ok=False)
    choices.to_csv(output / "december_candidates.csv", index=False)
    pd.concat([jan_metrics, feb_metrics], ignore_index=True).to_csv(output / "metrics.csv", index=False)
    bins.to_csv(output / "february_shared_cohort_bins.csv", index=False)
    importance = feature_importance_table(b_xgb)
    importance.to_csv(output / "stage_b_feature_importance.csv", index=False)
    for stage, frame, tie, pred in (("A_january", jan, ties_jan, a_preds),
                                    ("B_february", feb, ties_feb, b_preds)):
        saved = frame[["source_row", "timestamp", "isFraud"]].reset_index(drop=True).copy()
        saved["tie_priority"] = tie
        for name, scores in pred.items():
            saved[name] = scores
        saved.to_parquet(output / f"{stage}_predictions.parquet", index=False)
    joblib.dump(a_xgb, output / "A_xgboost.joblib")
    joblib.dump(b_xgb, output / "B_xgboost.joblib")
    for method in ("sigmoid", "isotonic"):
        joblib.dump(cal[method], output / f"B_xgboost_{method}.joblib")
    # JSON model exports allow inspection of split structure without pickle execution.
    a_xgb.named_steps["model"].save_model(str(output / "A_xgboost.json"))
    b_xgb.named_steps["model"].save_model(str(output / "B_xgboost.json"))
    plot_reports(jan_metrics, feb_metrics, bins, output)
    plot_importance(importance, output / "stage_b_feature_importance.png")
    plot_pr_curves(pd.read_parquet(output / "B_february_predictions.parquet"),
                   output / "february_precision_recall.png")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "input_sha256": source_sha,
                "existing_logistic_manifest_sha256": sha256(logistic_saved / "run.json"),
                "chosen_configuration": chosen, "locked_candidates": list(CANDIDATES),
                "common_xgboost_settings": COMMON,
                "december_rule": "AP maximization; 1e-6 AP near-tie log-loss min; fixed candidate order",
                "stage_a": "Oct-Nov fit, December selection, January historical assessment",
                "stage_b": "Oct-Dec fit, January XGBoost calibration, February comparison",
                "january_is_not_fresh_test": True, "february_is_not_fresh_test": True,
                "march_april_scored": False,
                "counts": {name: {"rows": len(frame), "positives": int(frame.isFraud.sum())}
                           for name, frame in windows.items()},
                "features": FEATURES, "seed": SEED,
                "packages": {name: version(name) for name in
                             ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib", "pyarrow", "matplotlib")},
                "git": git_state(Path(__file__).resolve().parents[2]),
                "python": platform.python_version(),
                "artifact_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
                "complete": True}
    (output / "run.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    audit_saved(output, windows, source_sha, logistic_saved)
    print("PASS: outputs generated in", output.resolve())
    print("January and February are historical; March-April is still unscored.")
    print(feb_metrics.loc[np.isclose(feb_metrics.review_fraction, .01),
        ["series","selected","true_positives","false_positives","average_precision","log_loss","brier_score"]].to_string(index=False))


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/processed/amlnet_v2_prepared.parquet")
    parser.add_argument("--output", type=Path, default=root / "models/xgboost_benchmark_v1")
    parser.add_argument("--logistic-v2", type=Path, default=root / "models/calibration_saved_v2")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        windows, digest = load_windows(args.input)
        audit_saved(args.output, windows, digest, args.logistic_v2)
    else:
        run(args.input, args.output, args.logistic_v2)


if __name__ == "__main__":
    main()
