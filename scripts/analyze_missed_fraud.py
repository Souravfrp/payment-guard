from pathlib import Path
import pandas as pd

root = Path(".")
data = pd.read_parquet(root / "data/processed/amlnet_v2_prepared.parquet",
    columns=["source_row", "amount", "oldbalanceOrg", "type", "category",
             "isFraud", "timestamp", "split"])
pred = pd.read_parquet(root / "models/logistic_ablation_v1/validation_predictions.parquet",
    columns=["source_row", "isFraud", "timestamp", "tie_priority", "logistic_log"])
assert data["source_row"].is_unique and pred["source_row"].is_unique
df = pred.merge(data, on="source_row", how="left", validate="one_to_one",
                suffixes=("_pred", "_data"), indicator=True)
assert df["_merge"].eq("both").all() and df["split"].eq("validation").all()
assert len(df) == 167990
assert df["isFraud_pred"].eq(df["isFraud_data"]).all()
assert pd.to_datetime(df["timestamp_pred"]).eq(
    pd.to_datetime(df["timestamp_data"])).all()
assert df["tie_priority"].is_unique and df["logistic_log"].notna().all()
ranked = df.sort_values(["logistic_log", "tie_priority"],
                        ascending=[False, True], kind="stable").copy()
ranked["rank"] = range(1, len(ranked)+1)
missed = ranked.loc[(ranked["isFraud_pred"] == 1) & (ranked["rank"] > 1680)]
print("Missed positives:", len(missed))
print("Summed missed transaction amount:", round(missed["amount"].sum(), 2))
print(missed[["source_row", "rank", "amount", "oldbalanceOrg",
              "type", "category", "logistic_log"]].sort_values("rank").to_string(index=False))
assert len(missed) == 6
assert round(missed["amount"].sum(), 2) == 5052.04
print("\nSaved-run checks passed. Amount is not observed loss.")
