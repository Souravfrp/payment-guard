from pathlib import Path
import pandas as pd

df = pd.read_csv(Path("results/tables/logistic_ablation_summary.csv"))
required = {"model", "selected", "true_positives", "false_negatives"}
assert required.issubset(df.columns)
assert df[list(required - {"model"})].notna().all().all()
assert (df["selected"] >= 0).all()
assert (df["false_negatives"] >= 0).all()

review_cost_inr = 100  # Hypothetical scenario, not observed costs.
ratios = [1, 10, 24, 50, 100, 200, 420, 500, 1000]
rows = []
for ratio in ratios:
    comparison = df.copy()
    comparison["total_cost"] = review_cost_inr * (
        comparison["selected"] + ratio * comparison["false_negatives"]
    )
    best = comparison.sort_values(
        ["total_cost", "selected", "model"], kind="stable"
    ).iloc[0]
    rows.append({
        "missed_to_review_cost_ratio": ratio,
        "best_model_within_compared_options": best["model"],
        "reviews": int(best["selected"]),
        "fraud_caught": int(best["true_positives"]),
        "hypothetical_total_cost_inr": int(best["total_cost"]),
    })
print(pd.DataFrame(rows).to_string(index=False))
print("\nScenario analysis only; compares existing budgets and rankings.")
