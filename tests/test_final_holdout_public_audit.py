"""Tests of published aggregate-only final holdout audit; never load private records."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.util.spec_from_file_location(
    "public_holdout_aggregate_audit", ROOT / "scripts/audit_final_holdout_public.py")
assert LOADER is not None and LOADER.loader is not None
auditor = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(auditor)


def test_public_aggregate_results_reconcile(capsys):
    auditor.audit(ROOT / "results/tables/final_holdout_v1")
    assert "42 metric rows, 9 paired rows" in capsys.readouterr().out


def test_public_hash_guard_refuses_modified_aggregate(tmp_path):
    original = ROOT / "results/tables/final_holdout_v1"
    for name in auditor.EXPECTED:
        (tmp_path / name).write_bytes((original / name).read_bytes())
    (tmp_path / "paired_overlap.csv").write_text("modified aggregate")
    with pytest.raises(ValueError, match="hash differs"):
        auditor.audit(tmp_path)
