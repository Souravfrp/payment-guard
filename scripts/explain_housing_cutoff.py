"""Explain two validation Housing transfers around the review cutoff."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from payment_guard.train_logistic_baseline import FEATURES, design_frame


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "models/logistic_ablation_v1"
    name = "logistic_log_without_category"

    data = pd.read_parquet(
        root / "data/processed/amlnet_v2_prepared.parquet",
        columns=FEATURES + ["source_row", "isFraud", "timestamp"],
        filters=[("split", "==", "validation")],
    )
    predictions = pd.read_parquet(
        folder / "validation_predictions.parquet",
        columns=["source_row", "isFraud", "timestamp",
                 "tie_priority", name],
    )

    joined = data.merge(
        predictions,
        on="source_row",
        how="outer",
        validate="one_to_one",
        suffixes=("", "_saved"),
        indicator=True,
    )
    assert joined["_merge"].eq("both").all()
    assert joined["isFraud"].eq(joined["isFraud_saved"]).all()
    assert joined["timestamp"].eq(joined["timestamp_saved"]).all()
    assert len(joined) == 167990
    assert np.isfinite(joined[name]).all()
    assert joined[name].between(0, 1).all()
    assert np.array_equal(
        np.sort(joined["tie_priority"].to_numpy()),
        np.arange(len(joined)),
    )

    joined = joined.sort_values(
        [name, "tie_priority"], ascending=[False, True]
    ).reset_index(drop=True)
    joined["rank"] = np.arange(1, len(joined) + 1)
    joined["reviewed"] = joined["rank"].le(1680)

    positive = joined["isFraud"].eq(1)
    assert (joined["reviewed"] & positive).sum() == 130
    assert (joined["reviewed"] & ~positive).sum() == 1550

    housing = joined.loc[
        joined["category"].eq("Housing")
        & joined["type"].eq("TRANSFER")
    ]
    assert housing["reviewed"].sum() == 1519

    examples = pd.concat([
        housing.loc[housing["reviewed"]].tail(1),
        housing.loc[~housing["reviewed"]].head(1),
    ]).copy()
    assert len(examples) == 2

    # Load only the locally trained, trusted model artifact.
    pipeline = joblib.load(folder / f"{name}.joblib")
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    assert np.array_equal(model.classes_, [0, 1])

    inputs = design_frame(
        examples, log_amount=True, excluded=("category",)
    )
    transformed = preprocessor.transform(inputs)
    values = (
        transformed.toarray()
        if hasattr(transformed, "toarray")
        else np.asarray(transformed)
    )
    names = preprocessor.get_feature_names_out()
    contributions = values * model.coef_[0]
    log_odds = model.intercept_[0] + contributions.sum(axis=1)
    reconstructed = expit(log_odds)

    np.testing.assert_allclose(
        reconstructed, pipeline.predict_proba(inputs)[:, 1],
        rtol=1e-9, atol=1e-12,
    )
    np.testing.assert_allclose(
        reconstructed, examples[name].to_numpy(),
        rtol=1e-9, atol=1e-12,
    )
    print("Checks passed: ranking and example probabilities reproduced.")

    for i, (_, row) in enumerate(examples.iterrows()):
        print("\n" + "=" * 65)
        print("SELECTED" if row["reviewed"] else "NOT SELECTED")
        for field in ["source_row", "rank", "amount",
                      "oldbalanceOrg", "isFraud"]:
            print(f"{field}: {row[field]}")
        print("Intercept:", model.intercept_[0])
        print("Total log-odds:", log_odds[i])
        print("Saved probability:", row[name])
        print("Reconstructed probability:", reconstructed[i])

        table = pd.DataFrame({
            "feature": names,
            "transformed_value": values[i],
            "coefficient": model.coef_[0],
            "contribution": contributions[i],
        })
        table = table.loc[table["contribution"].abs() > 1e-12]
        order = table["contribution"].abs().sort_values(
            ascending=False
        ).index
        print(table.loc[order].to_string(
            index=False, float_format="%.6f"
        ))


if __name__ == "__main__":
    main()
