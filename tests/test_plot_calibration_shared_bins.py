"""Fixture tests for method-comparable calibration visualizations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scripts.plot_calibration_shared_bins import (
    METHODS, shared_bin_table, validate_predictions,
)


def example():
    n = 20
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-11"] * n),
        "source_row": np.arange(1, n+1),
        "isFraud": [0] * 18 + [1, 1],
        "tie_priority": np.arange(n-1, -1, -1),
        "uncalibrated": np.linspace(.001, .99, n),
        "sigmoid": np.linspace(.005, .95, n),
        "isotonic": [0.0] * 10 + [.02] * 8 + [0.8] * 2,
    })
    return frame


def test_bins_same_transactions_and_positive_totals():
    f = example()
    table = shared_bin_table(f, stage="A_january")
    assert len(table) == 10
    assert (table.rows == 2).all()
    assert table.positives.sum() == 2
    assert table.loc[table.bin.eq(10), "positives"].iloc[0] == 2
    assert np.isclose(table.observed_rate.iloc[-1], 1.0)
    assert np.isclose(table.mean_probability_isotonic.iloc[-1], .8)
    assert set(table.columns).issuperset({f"mean_probability_{m}" for m in METHODS})


def test_reject_wrong_period_and_duplicate_row_id():
    f = example()
    f.loc[1, "timestamp"] = pd.Timestamp("2026-03-10")
    with pytest.raises(ValueError, match="timestamp"):
        validate_predictions(f, "A_january", full_counts=False)
    f = example()
    f.loc[1, "source_row"] = 1
    with pytest.raises(ValueError, match="duplicate"):
        validate_predictions(f, "A_january", full_counts=False)


def test_reject_invalid_tie_priority_and_scores():
    f = example()
    f.loc[0, "tie_priority"] = f.loc[1, "tie_priority"]
    with pytest.raises(ValueError, match="Tie priorities"):
        shared_bin_table(f, stage="A_january")
    f = example()
    f.loc[0, "isotonic"] = 1.02
    with pytest.raises(ValueError, match="Invalid probability"):
        shared_bin_table(f, stage="A_january")


def test_tied_base_scores_are_not_sorted_using_labels():
    f = example()
    f["uncalibrated"] = 0.05
    table = shared_bin_table(f, stage="A_january", bins=10)
    assert table.positives.sum() == 2
    assert table.boundary_split_base_tie.iloc[1:].all()
    assert not bool(table.boundary_split_base_tie.iloc[0])
    # The split is disclosed: equal base scores are assigned by tie priority,
    # not by observed fraud labels.


def test_reject_empty_or_overspecified_bins():
    f = example()
    with pytest.raises(ValueError, match="number of bins"):
        shared_bin_table(f, stage="A_january", bins=25)
