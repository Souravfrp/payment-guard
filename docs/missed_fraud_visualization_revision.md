# Why I revised the missed-fraud visualizations

The initial contribution heatmap used a sequential colour scale. I could see the very large negative contribution for source row 816607, but its magnitude compressed most of the other values into nearly indistinguishable colours. That made it difficult to see feature-level differences for the other five cases.

I now plot the same fitted contributions on a blue–white–red **zero-centred diverging scale**. The contribution is always computed as missed minus matched caught. Blue means that a source-feature group *reduces the missed case's fitted log-odds relative to its comparator*. Red means the group raises its fitted log-odds. This is not an assertion that blue means a safe transaction or that red means a genuinely risky transaction.

For complete magnitude information, I keep a full-range symmetric scale covering the largest absolute observed contribution. I also make a second view with the colour scale fixed at ±2 log-odds. Cells beyond ±2 are saturated visually and marked with an asterisk; where numerical annotations are shown, they display the original value, not the saturated value. Neither heatmap changes the CSV values, model parameters, or any fitted probability.

I also revised the score chart caption to disclose that six comparisons reuse only two different caught-positive reference records. These are **six descriptive pairs, not six statistically independent matched controls**.

The score differences satisfy the fitted-model identity

\[
z_m-z_c=\sum_g\{C_g(m)-C_g(c)\}.
\]

This explains the model in its transformed feature space. It does not establish causal effects or calibrated real-world fraud probabilities. The study uses February validation outputs only. The final March–April holdout remains untouched.

## Reproduce

From the repository root, after running the existing missed-fraud analysis once:

```bash
PYTHONPATH=src python -m pytest -q tests/test_plot_missed_fraud_v2.py
PYTHONPATH=src python scripts/plot_missed_fraud_v2.py
```

The plotting script reads only the already-created `missed_fraud_matched_pairs.csv` and `missed_fraud_contribution_differences.csv`. It produces three **new** PNGs and refuses to overwrite existing files:

- `results/figures/missed_fraud_score_comparison_v2.png`
- `results/figures/missed_fraud_feature_differences_diverging_full.png`
- `results/figures/missed_fraud_feature_differences_diverging_detail.png`

No refitting, local model loading, holdout scoring, or manual file edits are required.
