# Checking that my published results can be reproduced

## Why I added this verification step

Before changing the fraud model, I want to establish that my saved results tell a consistent story. I do not want a later result to appear better simply because a CSV, score ordering, or model file silently changed. The first audit does **not** train or select a model. It checks the evidence already published, then optionally compares that evidence with the files produced by my earlier local experiments.

My earlier baseline and ablation evaluations were run on the February 2026 validation period. The March–April holdout was reserved for later evaluation, and I am not using it to choose or calibrate a model here. The prepared file's SHA-256 is checked as a whole when a saved run is audited, but the verification routine filters Parquet reads to validation rows. Hashing the file is not the same as scoring or examining its test labels.

## First checkpoint: published aggregates (no dataset required)

From the repository root, in my `payment-guard-verify` Conda environment:

```bash
PYTHONPATH=src python -m payment_guard.verify_checkpoint
```

The script verifies:

1. Split totals: 668,820 training rows, 167,990 validation rows, and 253,190 reserved test rows. Positives must be 931, 202 and 278, respectively.
2. Chronological split boundaries and reported fraud prevalence in each split.
3. Four original baseline approaches at three review budgets (12 rows), plus ten approaches in the ablation summary at those same budgets (30 rows).
4. Each confusion matrix and the precision/recall arithmetic. The 12 repeated reference rows must agree across the two published CSVs.
5. Nondecreasing caught positives as the review budget increases for the same ranking. Ranking and probability metrics must be constant across the three cutoffs of a fixed score vector.

A passing aggregate audit establishes internal consistency of published tables. It does **not** prove that the table was produced by the fitted model. The distinction is important.

## Second checkpoint: local saved run (prepared data required)

After the aggregate step passes, and only if my original local Parquet and model files are present:

```bash
PYTHONPATH=src python -m payment_guard.verify_checkpoint \
  --saved-run models/logistic_ablation_v1
```

This additional check reads `run.json`, verifies file SHA-256 hashes, checks the saved scores against the February validation labels, recomputes budget-level confusion counts and summary metrics, and loads each locally saved logistic model to reproduce its stored validation probabilities. It compares the counts with the published aggregate table. It never refits a model, rewrites results, or scores the final holdout period.

If I also want to verify the original two-model experiment separately:

```bash
PYTHONPATH=src python -m payment_guard.verify_checkpoint \
  --saved-run models/logistic_baseline_v1
```

These commands load `.joblib` files only from my own trusted local experiment. I must not load arbitrary model files obtained from unfamiliar sources.

If either local folder is missing, I should first locate my actual saved run; I should **not** claim a failed or skipped hash check as a passing full-data reproduction. If dependencies have changed since the recorded run, I should inspect the versions before deciding whether a rerun is necessary.

## Mathematical viewpoint

For each review budget \(k\), the fixed ranking produces true positives \(TP_k\), false positives \(FP_k\), false negatives \(FN_k\), and true negatives \(TN_k\). I require

\[
TP_k+FP_k=k,\quad TP_k+FN_k=P,\quad FP_k+TN_k=N-P,
\]

where the February validation sample has \(N=167{,}990\) and \(P=202\). I also verify

\[
\mathrm{Precision}@k=\frac{TP_k}{k},\qquad
\mathrm{Recall}@k=\frac{TP_k}{P}.
\]

These identities detect arithmetic and publication errors, not modelling leakage. Recomputing scores from saved fitted pipelines is a separate, stronger verification. The SHA-256 digest catches accidental byte changes, but it does not by itself establish that the original experiment was scientifically appropriate.

## Algorithm and computational cost

The aggregate audit takes time linear in the small number of published rows. The saved-run audit reads and hashes the prepared Parquet and model files, then sorts \(N\) validation scores for each approach. Ranking is \(O(N\log N)\) per model; checking the confusion counts after sorting is \(O(N)\). With \(M\) rankings, the ranking component is \(O(MN\log N)\); fitted-model scoring adds its own model-specific computation. This is why I verify my existing results before trying new models: it is much cheaper than a new research sweep, while reducing the risk of building on an inconsistent result.

## Why this is not a new model comparison

I chose logistic regression for my initial experiment because I could examine feature contributions and perform controlled ablations. A tree-based model might capture nonlinear interactions more naturally, while a calibrated or time-aware decision rule might better fit an operational use case; neither is justified merely by being more complex. Before selecting a new method I want to check model validity, calibration, uncertainty and behavior over time. For each future candidate I will record what weakness it is intended to address, what assumptions it adds, and which comparable evaluation could disprove the claimed improvement.

## Evidence and limitations

The current GitHub results are aggregate snapshots; row-level Parquet predictions and fitted models are intentionally kept locally. The new verification command can audit those artifacts on my Mac, but someone seeing the code on GitHub should not infer that the private artifacts have been independently reproduced. I will record my actual terminal PASS/FAIL output and package versions only after running the commands. No values for the reserved March–April holdout are reported by this audit.

## Verification correction — original baseline manifest

My first local saved-run verification passed for the October 7 feature-ablation
experiment, but it failed for the October 6 baseline with `KeyError: 'excluded_features'`.
I traced this to a change in the *manifest format*, not to evidence of incorrect
model predictions. The original baseline source code recorded `log_amount` for
each full-feature fit but did not yet record `excluded_features`; that field was
added for the later feature-removal experiment.

I updated the verifier so that it treats a missing `excluded_features` as an
empty removal set **only for the two original full-feature baseline models**.
It refuses to assume missing fields for an ablation model, and it checks that
the saved amount transformation and removed features agree with the model name.
The updated tool also checks multi-index uniqueness explicitly instead of using
pandas' deprecated `verify_integrity` argument to `set_index`. The saved
experiment files, datasets, model weights and original predictions are unchanged.

I will record a successful baseline audit only after the command verifies the
saved file digests, reproduced probabilities and February metrics on my Mac.
This compatibility fix addresses verification code, not model performance or
scientific validity.
