"""Offline checks of the pre-score freeze gate; uses no AMLNet rows."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.util.spec_from_file_location(
    "freeze_checker_for_tests", ROOT / "scripts/verify_final_holdout_freeze.py")
assert LOADER is not None and LOADER.loader is not None
m = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(m)


def public_spec():
    return json.loads((ROOT / "config/final_holdout_freeze_v1.json").read_text())


def put(root, rel, value=b"fixture"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(value)
    return p


def shasum(value):
    return hashlib.sha256(value).hexdigest()


def test_public_freeze_spec_is_locked():
    spec = public_spec()
    m.validate_spec(spec)
    assert spec["primary"]["score"] == "uncalibrated"
    assert spec["data"]["holdout_split"] == "test"
    assert spec["ranking"]["primary_budget_fraction"] == 0.01


def test_policy_drift_is_rejected():
    original = public_spec()
    for mutate in (
        lambda s: s["primary"].update(family="logistic"),
        lambda s: s["ranking"].update(primary_budget_fraction=0.02),
        lambda s: s["data"].update(march_april_scored=True),
        lambda s: s["fixed_sensitivity_artifacts"].pop(),
    ):
        changed = copy.deepcopy(original)
        mutate(changed)
        with pytest.raises(ValueError):
            m.validate_spec(changed)


def test_manifest_bytes_and_flags_are_checked(tmp_path):
    source_hash = "a" * 64
    folder = tmp_path / "reference"
    put(tmp_path, "reference/fixture.joblib", b"unchanged")
    record = {"complete": True, "march_april_scored": False,
              "input_sha256": source_hash,
              "artifact_sha256": {"fixture.joblib": shasum(b"unchanged")}}
    put(tmp_path, "reference/run.json", json.dumps(record).encode())
    assert m.verify_manifest(folder, source_hash)["complete"] is True
    put(tmp_path, "reference/fixture.joblib", b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        m.verify_manifest(folder, source_hash)
    put(tmp_path, "reference/fixture.joblib", b"unchanged")
    record["march_april_scored"] = True
    put(tmp_path, "reference/run.json", json.dumps(record).encode())
    with pytest.raises(ValueError, match="holdout-scored"):
        m.verify_manifest(folder, source_hash)


def test_unsafe_or_missing_paths_fail(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        m.checked_path(tmp_path, "../outside")
    with pytest.raises(FileNotFoundError):
        m.checked_path(tmp_path, "models/nonexistent.joblib")


def test_full_synthetic_preflight_and_artifact_tampering(tmp_path, monkeypatch):
    spec = public_spec()
    put(tmp_path, "config/final_holdout_freeze_v1.json", json.dumps(spec).encode())
    put(tmp_path, spec["data"]["prepared_parquet"], b"opaque fixture bytes not Parquet")
    source_sha = shasum(b"opaque fixture bytes not Parquet")
    for rel in spec["pinned_source_git_blobs"]:
        put(tmp_path, rel, b"synthetic code fixture")
    original_run_paths = spec["artifact_verification"]["local_run_manifests"]
    artifact_paths = ([spec["primary"]["local_artifact"], spec["comparator"]["local_artifact"]]
                      + [x["local_artifact"] for x in spec["fixed_sensitivity_artifacts"]]
                      + spec["secondary_calibration_artifacts"])
    by_dir = {}
    for rel in artifact_paths:
        p = put(tmp_path, rel, b"local model fixture:" + rel.encode())
        by_dir.setdefault(p.parent, {})[p.name] = m.digest(p)

    def create_record(rel, extra=None):
        record = {"complete": True, "march_april_scored": False,
                  "input_sha256": source_sha,
                  "artifact_sha256": by_dir[(tmp_path / rel).parent]}
        record.update(extra or {})
        p = put(tmp_path, rel, json.dumps(record).encode())
        return m.digest(p)

    logistic_path, xgb_path, ablation_path = original_run_paths
    logistic_digest = create_record(logistic_path)
    xgb_digest = create_record(xgb_path, {
        "chosen_configuration": "depth3_240",
        "features": spec["source_features"],
        "existing_logistic_manifest_sha256": logistic_digest})
    create_record(ablation_path, {
        "locked_configuration": "depth3_240",
        "reference_manifest_sha256": xgb_digest,
        "logistic_manifest_sha256": logistic_digest})

    def fake_git(args, **kwargs):
        relative = str(Path(args[-1]).relative_to(tmp_path))
        return subprocess.CompletedProcess(args, 0,
            stdout=spec["pinned_source_git_blobs"][relative] + "\n", stderr="")
    monkeypatch.setattr(m.subprocess, "run", fake_git)
    result = m.preflight(tmp_path, spec)
    assert result["status"] == "PASS_PRE_SCORE_NO_SCORING"
    assert len(result["selected_model_and_map_sha256"]) == 10
    assert result["prepared_parquet_sha256"] == source_sha
    assert result["march_april_scored"] is False
    put(tmp_path, spec["primary"]["local_artifact"], b"modified model")
    with pytest.raises(ValueError, match="checksum"):
        m.preflight(tmp_path, spec)
