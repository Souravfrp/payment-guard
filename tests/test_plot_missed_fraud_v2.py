"""Synthetic CSV-only tests. No model fitting, local ML artifacts or holdout rows."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import plot_missed_fraud_v2 as charts


def sample_data(root: Path):
    tab = root / "results" / "tables"
    tab.mkdir(parents=True)
    ids = sorted(charts.EXPECTED_ROWS)
    caught = [746221, 746221, 655244, 655244, 655244, 746221]
    delta = []
    pairs = []
    for k, row in enumerate(ids):
        fields = {"contrib__" + group: 0. for group in charts.GROUP_LABELS}
        fields["contrib__amount"] = -(k + 1) * 2.4
        fields["contrib__hour_cycle"] = .2 * (k + 1)
        diff = sum(fields.values())
        delta.append({"missed_source_row": row, **fields})
        pairs.append({"missed_source_row": row, "caught_source_row": caught[k],
                      "missed_log_odds": -5. - k * 2.2, "caught_log_odds": -5. - k * 2.2 - diff,
                      "delta_log_odds_missed_minus_caught": diff, "cutoff_log_odds": -1.0})
    # Ensure caught comparators always rank above the cutoff in the synthetic fixture.
    for entry in pairs:
        entry["caught_log_odds"] = 0.0
        entry["missed_log_odds"] = entry["caught_log_odds"] + entry["delta_log_odds_missed_minus_caught"]
    pd.DataFrame(pairs).to_csv(tab / "missed_fraud_matched_pairs.csv", index=False)
    pd.DataFrame(delta).to_csv(tab / "missed_fraud_contribution_differences.csv", index=False)
    return tab


def test_all_three_images_generated_without_input_modification(tmp_path):
    table_folder = sample_data(tmp_path)
    inputs = {f: f.read_bytes() for f in table_folder.iterdir()}
    paths = charts.generate(tmp_path)
    assert len(paths) == 3
    assert all(path.exists() and path.stat().st_size > 10_000 for path in paths)
    assert all(f.read_bytes() == contents for f, contents in inputs.items())
    with pytest.raises(FileExistsError, match="Refusing"):
        charts.generate(tmp_path)


def test_detects_difference_disagreement(tmp_path):
    folder = sample_data(tmp_path)
    path = folder / "missed_fraud_contribution_differences.csv"
    wrong = pd.read_csv(path)
    wrong.loc[0, "contrib__amount"] += 1
    wrong.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        charts.generate(tmp_path)
    assert not (tmp_path / "results" / "figures").exists()


def test_detects_case_mismatch(tmp_path):
    folder = sample_data(tmp_path)
    path = folder / "missed_fraud_matched_pairs.csv"
    wrong = pd.read_csv(path)
    wrong.loc[0, "missed_source_row"] = 12345
    wrong.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unexpected missed-case"):
        charts.generate(tmp_path)


def test_diverging_map_is_zero_centered():
    from matplotlib.colors import TwoSlopeNorm
    normalizer = TwoSlopeNorm(vmin=-12.5, vcenter=0, vmax=12.5)
    assert normalizer(0) == pytest.approx(.5)
    assert normalizer(-12.5) == pytest.approx(0)
    assert normalizer(12.5) == pytest.approx(1)
