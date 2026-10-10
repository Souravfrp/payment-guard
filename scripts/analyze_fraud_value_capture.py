from pathlib import Path
import pandas as pd

root = Path(".")
data = pd.read_parquet(
    root / "data/processed/amlnet_v2_prepared.parquet",
    columns=["source_row", "amount", "isFraud", "split"],
)
pred = pd.read_parquet(root / "models/logistic_ablation_v1/validation_predictions.parquet")
assert data["source_row"].is_unique and pred["source_row"].is_unique
df = pred.merge(data, on="source_row", how="left", validate="one_to_one",
                suffixes=("_pred", "_data"), indicator=True)
assert df["_merge"].eq("both").all()
assert df["split"].eq("validation").all()
assert len(df) == 167990
assert df["isFraud_pred"].eq(df["isFraud_data"]).all()
assert df["amount"].notna().all() and df["amount"].ge(0).all()
assert df["tie_priority"].notna().all() and df["tie_priority"].is_unique
fraud = df["isFraud_pred"].eq(1)
assert int(fraud.sum()) == 202
total_value = df.loc[fraud, "amount"].sum()
assert total_value > 0

models = ["amount_ranking", "logistic_raw", "logistic_log",
          "logistic_raw_without_type", "logistic_log_without_type",
          "logistic_raw_without_category", "logistic_log_without_category",
          "logistic_raw_without_both", "logistic_log_without_both"]
rows = []
for model in models:
    assert df[model].notna().all(), f"Missing scores: {model}"
    ranked = df.sort_values([model, "tie_priority"],
                            ascending=[False, True], kind="stable")
    for k in (168, 840, 1680):
        chosen = ranked.head(k)
        positives = chosen["isFraud_pred"].eq(1)
        caught = int(positives.sum())
        captured_value = chosen.loc[positives, "amount"].sum()
        rows.append({"model": model, "reviews": k, "caught_count": caught,
                     "count_recall_pct": round(100*caught/202, 2),
                     "caught_value": round(captured_value, 2),
                     "value_capture_pct": round(100*captured_value/total_value, 2)})
results = pd.DataFrame(rows)
reference = pd.read_csv(root / "results/tables/logistic_ablation_summary.csv")
check = results.merge(reference, left_on=["model", "reviews"],
                      right_on=["model", "selected"], validate="one_to_one")
assert len(check) == len(results)
assert check["caught_count"].eq(check["true_positives"]).all()
print("Total fraud-labelled transaction amount:", round(total_value, 2))
print(results.to_string(index=False))
print("\nAll caught counts matched committed aggregate results.")
print("Value capture is not observed or prevented financial loss.")
