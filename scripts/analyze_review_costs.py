from pathlib import Path
import pandas as pd

df = pd.read_csv(Path("results/tables/logistic_ablation_summary.csv"))
required = {"model", "selected", "true_positives", "false_positives", "false_negatives"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
assert (df["selected"] == df["true_positives"] + df["false_positives"]).all()
assert (df["true_positives"] + df["false_negatives"] == 202).all()

review_cost_inr = 100
missed_label_cost_inr = 50_000
result = df.copy()
result["review_cost_inr"] = result["selected"] * review_cost_inr
result["missed_label_cost_inr"] = result["false_negatives"] * missed_label_cost_inr
result["hypothetical_total_cost_inr"] = (
    result["review_cost_inr"] + result["missed_label_cost_inr"]
)
result = result.sort_values(
    ["hypothetical_total_cost_inr", "selected", "model"], kind="stable"
)
print(result[["model", "selected", "true_positives", "false_positives",
              "false_negatives", "hypothetical_total_cost_inr"]].to_string(index=False))
print("\nCosts are hypothetical INR assumptions, not observed bank losses.")
