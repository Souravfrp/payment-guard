# Making the calibration figures comparable

## What I found in the first full run

My original January plot contained ten uncalibrated and sigmoid score groups but only four isotonic groups. The February plot had ten groups for the first two methods but only **two** plotted isotonic groups. The first chart displayed an isotonic group at x=2 alongside a very different uncalibrated group at x=2. The horizontal numbers were not shared groups. I should not interpret them as if they contained the same records.

For the saved February isotonic results, the lower group contained **150,249 rows and three positives**; the upper contained **17,741 rows and 199 positives**. Their observed positive fractions were about 0.0020% and 1.1217%, respectively. This reflects the grouping produced by the existing quantile/tie procedure. Importantly, *two displayed bins do not mean only two distinct isotonic prediction values*: each bin had a reported minimum and maximum probability.

## My plotting correction

I do not need to refit any model to correct this display. I read only the original, verified January and February saved prediction Parquet files. For each month separately, I create **ten equal-count groups from the uncalibrated scores**, using the original saved label-independent tie priority. The identical row membership is then used to calculate the mean probabilities from all three methods. There is exactly **one observed positive fraction per group**, since the transaction labels are identical across methods.

The corrected rank-bin chart places the observed fraction and all three predicted fractions at the *same* x coordinate for the same transactions. I also plot mean predicted versus observed positive fraction for each method on those shared groups. These are aggregate calibration diagnostics, not transaction-level explanations. A ten-decile summary can hide very large score variation within the highest decile; detailed high-score investigations and event counts remain necessary.

Equal-count shared groups may split an **uncalibrated** score tie at a bin boundary. When this occurs, the saved seeded priority (not labels) determines membership; the aggregate CSV records whether each boundary split a tie. This differs intentionally from my first tie-preserving, method-specific quantile grouping. It does not alter the original ranking or selection results.

## What stays frozen

- Stage A chose **isotonic** using the predeclared January log-loss rule. This decision does not change because February favoured sigmoid on log loss and Brier score.
- Both stages, original metrics and fitted predictions remain unchanged.
- The revised chart script does not load the prepared dataset, fit models, load saved fitted objects or score March–April.
- The saved v1 report is verified against its recorded SHA-256 values before new figures are generated.
- The new figures and aggregate shared-bin table remain in a separate, locally ignored `models/calibration_visual_refresh_v1` directory until inspected. Nothing is pushed or committed by the plotting script.

## Commands

From the project root, with `payment-guard-verify` active:

```bash
PYTHONPATH=src python -m pytest -q tests/test_plot_calibration_shared_bins.py
PYTHONPATH=src python -m pytest -q
python scripts/plot_calibration_shared_bins.py
ls -lh models/calibration_visual_refresh_v1
```

The script refuses to overwrite any existing output directory. It saves four PNGs, a `shared_reliability_bins.csv` and a SHA-256 report, while leaving the initial calibration run untouched.

## Interpretation boundary

My January-selected method remains isotonic, but the difference in log loss on January was small. On February, sigmoid had the lower reported log loss and Brier score. At the 1% review budget all three approaches selected 197 of 202 positive February labels, so this experiment does not show a calibration-induced improvement in top-k fraud capture. Also, I retrained the base model on a different historical fitting period than the earlier published baseline; the one-positive difference from that earlier ranking is not attributable to calibration.

I will check the refreshed plots, positive counts and high-score groups before publication. Saving fitted base models and calibrators for a full prediction reconstruction requires a separate, new versioned run rather than retroactively modifying the completed v1 results.
