"""Analyze six missed February fraud labels using saved logistic probabilities.

Read-only access to fitted models and validation data. Outputs are new CSV/PNG
files and do not modify an existing run, dataset, or any holdout predictions.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from payment_guard.train_logistic_baseline import CATEGORICAL, CYCLIC, FEATURES, SCALED, design_frame

MODEL = "logistic_log"
EXPECTED_N = 167_990
EXPECTED_P = 202
EXPECTED_CAUGHT = 196
EXPECTED_MISSED = 6
BUDGET = 1_680
EXPECTED_MISSED_AMOUNT = 5_052.04
EXPECTED_SOURCE_ROWS = {830467, 830408, 719998, 722735, 722740, 816607}


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def grouped_contributions(pipeline, frame: pd.DataFrame):
    """Return intercept, log odds and per-source-field log-odds contributions.

    Groups encoded categories back to the corresponding original source field.
    Effects describe the fitted encoding rather than causal importance.
    """
    pre = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    require(np.array_equal(estimator.classes_, [0, 1]), "Unexpected class order")
    inputs = design_frame(frame, log_amount=True)
    transformed = pre.transform(inputs)
    values = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    weights = estimator.coef_[0]
    require(values.shape[1] == len(weights), "Coefficient/feature length mismatch")
    detailed = values * weights[None, :]
    slices = pre.output_indices_
    grouped = {}
    claimed = np.zeros(len(weights), dtype=bool)

    def add_group(name, idx):
        loc = np.asarray(list(idx), dtype=int)
        require(loc.size > 0 and not claimed[loc].any(), f"Overlapping or empty group: {name}")
        claimed[loc] = True
        grouped[name] = detailed[:, loc].sum(axis=1)

    scale_slice = slices["scaled"]
    require(scale_slice.stop - scale_slice.start == len(SCALED), "Unexpected scaled field count")
    for offset, feature in enumerate(SCALED):
        add_group(feature, [scale_slice.start + offset])

    cyclic_slice = slices["cyclic"]
    require(cyclic_slice.stop - cyclic_slice.start == 2 * len(CYCLIC), "Unexpected cycle field count")
    for offset, feature in enumerate(CYCLIC):
        add_group(feature + "_cycle", [cyclic_slice.start + 2 * offset, cyclic_slice.start + 2 * offset + 1])

    categorical_slice = slices["categorical"]
    encoder = pre.named_transformers_["categorical"]
    require(len(encoder.categories_) == len(CATEGORICAL), "Unexpected category field count")
    index = categorical_slice.start
    for field, cats in zip(CATEGORICAL, encoder.categories_):
        count = len(cats)
        add_group(field, range(index, index + count))
        index += count
    require(index == categorical_slice.stop and claimed.all(), "Unassigned transformed features")

    contributions = pd.DataFrame(grouped, index=frame.index)
    intercept = float(estimator.intercept_[0])
    log_odds = intercept + contributions.sum(axis=1).to_numpy()
    reconstructed = expit(log_odds)
    expected = pipeline.predict_proba(inputs)[:, 1]
    np.testing.assert_allclose(reconstructed, expected, rtol=1e-9, atol=1e-12)
    return intercept, contributions, log_odds, reconstructed


def prepare_ranked_validation(data: pd.DataFrame, predictions: pd.DataFrame, *, expected_n=EXPECTED_N,
                              expected_p=EXPECTED_P, budget=BUDGET, expected_caught=EXPECTED_CAUGHT):
    """Join by source row, validate labels and time boundary, reproduce top-k."""
    require(data.source_row.is_unique and predictions.source_row.is_unique, "Source rows not unique")
    require(data.split.eq("validation").all(), "Nonvalidation rows in prepared input")
    require(data.timestamp.ge(pd.Timestamp("2026-02-01")).all(), "Pre-February records found")
    require(data.timestamp.lt(pd.Timestamp("2026-03-01")).all(), "March/holdout records found")
    joined = data.merge(predictions, on="source_row", how="outer", validate="one_to_one",
                        indicator=True, suffixes=("", "_saved"))
    require(joined._merge.eq("both").all(), "Validation prediction/data row disagreement")
    require(joined.isFraud.eq(joined.isFraud_saved).all(), "Prediction/data label disagreement")
    require(pd.to_datetime(joined.timestamp).eq(pd.to_datetime(joined.timestamp_saved)).all(), "Timestamp disagreement")
    require(len(joined) == expected_n and int(joined.isFraud.sum()) == expected_p, "Validation totals disagree")
    require(joined[MODEL].notna().all() and joined[MODEL].between(0, 1).all(), "Invalid probabilities")
    require(np.array_equal(np.sort(joined.tie_priority.to_numpy()), np.arange(expected_n)), "Invalid tie priority")
    ranked = joined.sort_values([MODEL, "tie_priority"], ascending=[False, True], kind="stable").reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["reviewed"] = ranked["rank"].le(budget)
    caught = ranked.isFraud.eq(1) & ranked.reviewed
    missed = ranked.isFraud.eq(1) & ~ranked.reviewed
    require(int(caught.sum()) == expected_caught, "Caught count does not match frozen run")
    require(int(missed.sum()) == expected_p - expected_caught, "Missed count does not reconcile")
    return ranked, ranked.loc[missed].copy(), ranked.loc[caught].copy()


def select_matched_caught(missed, caught, pipeline):
    """Match each miss to a caught positive of identical type/category.

    Match by Euclidean distance in training-standardized (log amount, balance).
    Matching is descriptive; matched records may be reused and are not causal controls.
    """
    scaler = pipeline.named_steps["preprocess"].named_transformers_["scaled"]
    require(list(SCALED[:2]) == ["amount", "oldbalanceOrg"], "Unexpected scaler column order")
    mean = np.asarray(scaler.mean_[:2], dtype=float)
    scale = np.asarray(scaler.scale_[:2], dtype=float)
    require(np.isfinite(mean).all() and (scale > 0).all(), "Invalid fitted scaling statistics")

    def numeric_distance(frame):
        return (np.column_stack([np.log1p(frame.amount.to_numpy(dtype=float)),
                                 frame.oldbalanceOrg.to_numpy(dtype=float)]) - mean) / scale

    caught = caught.sort_values("source_row")
    matched = []
    for miss in missed.itertuples():
        candidates = caught.loc[(caught.type == miss.type) & (caught.category == miss.category)]
        require(not candidates.empty, f"No same-type/category caught comparator for source_row {miss.source_row}")
        target = (np.asarray([np.log1p(miss.amount), miss.oldbalanceOrg]) - mean) / scale
        distances = np.linalg.norm(numeric_distance(candidates) - target, axis=1)
        ix = int(np.argmin(distances))
        item = candidates.iloc[ix].copy()
        item["distance"] = float(distances[ix])
        item["missed_source_row"] = int(miss.source_row)
        matched.append(item)
    return pd.DataFrame(matched).reset_index(drop=True)


def build_analysis(ranked, missed, caught, pipeline, *, budget=BUDGET):
    matched = select_matched_caught(missed, caught, pipeline)
    chosen = pd.concat([missed.assign(case_role="missed"),
                        matched.assign(case_role="matched_caught")], ignore_index=True)
    intercept, contributions, logits, probs = grouped_contributions(pipeline, chosen)
    np.testing.assert_allclose(probs, chosen[MODEL].to_numpy(dtype=float), rtol=1e-9, atol=1e-12)
    details = chosen[["source_row", "rank", "case_role", "amount", "oldbalanceOrg", "type",
                      "category", MODEL]].reset_index(drop=True).copy()
    details["log_odds"] = logits
    details["intercept"] = intercept
    details = pd.concat([details, contributions.reset_index(drop=True).add_prefix("contrib__")], axis=1)
    miss_details = details.iloc[:len(missed)].reset_index(drop=True)
    caught_details = details.iloc[len(missed):].reset_index(drop=True)
    require(len(miss_details) == len(caught_details), "Pairing alignment failure")
    groups = ["contrib__" + name for name in contributions.columns]
    cutoff_probability = float(ranked.iloc[budget - 1][MODEL])
    cutoff_log_odds = float(logit(np.clip(cutoff_probability, 1e-15, 1 - 1e-15)))
    pairs = pd.DataFrame({
        "missed_source_row": miss_details.source_row.astype("int64"),
        "caught_source_row": caught_details.source_row.astype("int64"),
        "type": miss_details.type,
        "category": miss_details.category,
        "missed_rank": miss_details["rank"].astype("int64"),
        "caught_rank": caught_details["rank"].astype("int64"),
        "missed_amount": miss_details.amount,
        "caught_amount": caught_details.amount,
        "missed_score": miss_details[MODEL],
        "caught_score": caught_details[MODEL],
        "missed_log_odds": miss_details.log_odds,
        "caught_log_odds": caught_details.log_odds,
        "delta_log_odds_missed_minus_caught": miss_details.log_odds - caught_details.log_odds,
        "cutoff_probability": cutoff_probability,
        "cutoff_log_odds": cutoff_log_odds,
        "match_distance_standardized_money": matched.distance.reset_index(drop=True),
    })
    delta = miss_details[groups].subtract(caught_details[groups])
    delta.insert(0, "missed_source_row", miss_details.source_row.to_numpy())
    np.testing.assert_allclose(delta[groups].sum(axis=1), pairs.delta_log_odds_missed_minus_caught,
                               rtol=1e-9, atol=1e-10)
    return pairs, details, delta, cutoff_log_odds


def draw_figures(pairs, delta, cutoff_log_odds, fig_dir):
    labels = [str(x) for x in pairs.missed_source_row]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(x, pairs.missed_log_odds, s=70, marker="x", label="Missed fraud")
    ax.scatter(x, pairs.caught_log_odds, s=70, marker="o", label="Matched caught fraud")
    ax.axhline(cutoff_log_odds, linestyle="--", label="1,680-review cutoff")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Missed transaction source row")
    ax.set_ylabel("Fitted logistic log-odds (not calibrated risk)")
    ax.set_title("February validation | Missed and matched caught fraud")
    ax.legend()
    ax.grid(axis="y", alpha=.22)
    fig.tight_layout()
    fig.savefig(fig_dir / "missed_fraud_score_comparison.png", dpi=170)
    plt.close(fig)

    groups = [c for c in delta.columns if c != "missed_source_row"]
    array = delta[groups].to_numpy(dtype=float).T
    fig, ax = plt.subplots(figsize=(12, 7.5))
    heatmap = ax.imshow(array, aspect="auto")
    ax.set_xticks(x, labels)
    ax.set_yticks(np.arange(len(groups)), groups)
    ax.set_xlabel("Missed transaction source row")
    ax.set_title("Log-odds contribution difference: missed minus matched caught")
    fig.colorbar(heatmap, ax=ax, label="Difference in fitted log-odds contribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "missed_fraud_feature_differences.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    folder = root / "models/logistic_ablation_v1"
    data_path = root / "data/processed/amlnet_v2_prepared.parquet"
    summary_path = root / "results/tables/logistic_ablation_summary.csv"
    pred_path = folder / "validation_predictions.parquet"
    model_path = folder / (MODEL + ".joblib")
    manifest_path = folder / "run.json"

    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("complete") is True and manifest.get("feature_ablation") is True, "Unexpected saved run")
    require(MODEL in manifest["models"], "Missing model manifest entry")
    public = pd.read_csv(summary_path)
    row = public[(public.model == MODEL) & (public.selected == BUDGET)]
    require(len(row) == 1 and int(row.iloc[0].true_positives) == EXPECTED_CAUGHT, "Public summary disagreement")

    data = pd.read_parquet(data_path, columns=FEATURES + ["source_row", "isFraud", "timestamp", "split"],
                           filters=[("split", "==", "validation")])
    predictions = pd.read_parquet(pred_path, columns=["source_row", "isFraud", "timestamp", "tie_priority", MODEL])
    ranked, missed, caught = prepare_ranked_validation(data, predictions)
    require(set(missed.source_row.astype(int)) == EXPECTED_SOURCE_ROWS, "Unexpected missed case identifiers")
    require(abs(missed.amount.sum() - EXPECTED_MISSED_AMOUNT) < .005, "Missed transaction amounts disagree")

    # These are trusted local joblib artifacts created by this project; never load untrusted joblib files.
    pipeline = joblib.load(model_path)
    pairs, contributions, delta, cutoff_log_odds = build_analysis(ranked, missed, caught, pipeline)
    table_dir = root / "results/tables"
    fig_dir = root / "results/figures"
    outputs = [table_dir / "missed_fraud_matched_pairs.csv",
               table_dir / "missed_fraud_group_contributions.csv",
               table_dir / "missed_fraud_contribution_differences.csv",
               fig_dir / "missed_fraud_score_comparison.png",
               fig_dir / "missed_fraud_feature_differences.png"]
    if any(p.exists() for p in outputs):
        raise FileExistsError("One or more diagnostic outputs already exist; no files were replaced: " +
                              ", ".join(str(p) for p in outputs if p.exists()))
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Stage complete outputs first, then publish without overwriting existing files.
    # Use hard links and roll back newly published paths on any failure.
    with tempfile.TemporaryDirectory(prefix=".missed-cases-", dir=root / "results") as temporary:
        stage = Path(temporary)
        pairs.to_csv(stage / outputs[0].name, index=False)
        contributions.to_csv(stage / outputs[1].name, index=False)
        delta.to_csv(stage / outputs[2].name, index=False)
        draw_figures(pairs, delta, cutoff_log_odds, stage)
        published = []
        try:
            for destination in outputs:
                os.link(stage / destination.name, destination)
                published.append(destination)
        except Exception:
            for destination in published:
                destination.unlink()
            raise
    print("PASS: February-only join, counts, saved scores, pair matches and logistic reconstruction")
    print(f"Reviewed {BUDGET:,} of {EXPECTED_N:,} transactions; {EXPECTED_CAUGHT} caught and {EXPECTED_MISSED} missed positives")
    print("\nMissed vs same-type/category caught comparisons:")
    print(pairs[["missed_source_row", "caught_source_row", "missed_rank", "caught_rank",
                 "missed_score", "caught_score", "delta_log_odds_missed_minus_caught"]]
          .to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    print("\nCreated five new diagnostic files:")
    for path in outputs:
        print(" -", path.relative_to(root))
    print("No model fitting and no March-April holdout scoring performed.")
    print("CAUTION: Matched comparisons and log-odds contributions are descriptive, not causal.")


if __name__ == "__main__":
    main()
