"""Small synthetic, no-dataset tests for aggregate-only publication."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/publish_calibration_checkpoint.py"
spec = importlib.util.spec_from_file_location("publish_calibration_checkpoint", MODULE)
publication = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = publication
spec.loader.exec_module(publication)


def write_manifest(folder: Path, *, files: dict[str, bytes], input_hash="inputdigest", **extra):
    folder.mkdir(parents=True, exist_ok=True)
    for name, contents in files.items():
        (folder / name).write_bytes(contents)
    manifest = {"complete": True, "march_april_scored": False,
                "selected_method_on_january": "isotonic",
                "input_sha256": input_hash,
                "artifact_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
                **extra}
    (folder / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def fixture_metrics():
    rows = []
    for stage, (n, positives, _, _) in publication.STAGES.items():
        for method in publication.METHODS:
            for fraction in (0.001, 0.005, 0.01):
                selected = int(np.ceil(n*fraction))
                tp = min(positives, 271 if stage == "A_january" else 197) if fraction == .01 else 1
                loss = ({"uncalibrated": .001470, "sigmoid": .001486, "isotonic": .001459} if stage == "A_january" else
                        {"uncalibrated": .001086, "sigmoid": .000993, "isotonic": .001054})[method]
                rows.append({"stage": stage, "method": method, "review_fraction": fraction,
                             "selected": selected, "true_positives": tp,
                             "false_positives": selected - tp, "false_negatives": positives - tp,
                             "recall": tp/positives, "precision": tp/selected,
                             "average_precision": .92, "log_loss": loss,
                             "brier_score": loss/5})
    return pd.DataFrame(rows)


def setup_fixture(root: Path):
    v1 = root / "models/calibration_temporal_v1"
    v2 = root / "models/calibration_saved_v2"
    viz = root / "models/calibration_visual_refresh_v1"
    data = fixture_metrics().to_csv(index=False).encode()
    write_manifest(v1, files={"metrics.csv": data})
    v2files = {"metrics.csv": data} | {name: b"test-only-not-a-real-joblib" for name in publication.OBJECTS}
    write_manifest(v2, files=v2files)
    viz.mkdir(parents=True)
    rows = []
    for stage, (n, positives, top_n, top_p) in publication.STAGES.items():
        counts = [len(a) for a in np.array_split(np.arange(n), 10)][::-1]
        p = [0]*8 + [positives-top_p, top_p]
        for i, (count, pos) in enumerate(zip(counts, p), 1):
            row = {"stage": stage, "bin": i, "rows": count, "positives": pos,
                   "observed_rate": pos/count, "boundary_split_base_tie": False}
            row.update({f"mean_probability_{method}": pos/count*.9 for method in publication.METHODS})
            rows.append(row)
    bins = pd.DataFrame(rows)
    # Explicitly match the top-decile sizes of the output from numpy.array_split.
    assert all(int(bins[(bins.stage==stage)&(bins.bin==10)].rows.iloc[0]) == top_n
               for stage, (_, _, top_n, _) in publication.STAGES.items())
    csv = bins.to_csv(index=False).encode()
    source_files = {"shared_reliability_bins.csv": csv} | {name: b"\x89PNG\r\n\x1a\nFAKE_TEST_ONLY" for name in publication.FIGURES}
    for name, contents in source_files.items():
        (viz / name).write_bytes(contents)
    (viz / "report.json").write_text(json.dumps({
        "source_selected_method": "isotonic", "method_selection_unchanged": True,
        "source_manifest_sha256": publication.sha256(v1 / "run.json"),
        "artifacts_sha256": {name: hashlib.sha256(contents).hexdigest() for name, contents in source_files.items()},
    }), encoding="utf-8")
    return v1, v2, viz


def test_check_only_does_not_write(tmp_path):
    setup_fixture(tmp_path)
    targets = publication.make_outputs(tmp_path, write=False)
    assert len(targets) == 7
    assert all(not (tmp_path / target).exists() for target in targets)


def test_aggregate_publication_only(tmp_path):
    setup_fixture(tmp_path)
    targets = publication.make_outputs(tmp_path, write=True)
    assert len(targets) == 7
    assert all((tmp_path / target).is_file() for target in targets)
    assert "source_row" not in (tmp_path / publication.TABLE_BINS).read_text()
    assert "source_row" not in (tmp_path / publication.TABLE_SUMMARY).read_text()
    doc = (tmp_path / publication.DOC).read_text()
    assert "**116 tests**" in doc and "**isotonic regression**" in doc
    assert "277 positives / 18,610 rows" in doc
    assert "199 positives / 16,799 rows" in doc
    assert "March–April" in doc
    assert not (tmp_path / "results/tables/calibration_full_predictions.csv").exists()


def test_no_overwrite(tmp_path):
    setup_fixture(tmp_path)
    existing = tmp_path / publication.DOC
    existing.parent.mkdir(parents=True)
    existing.write_text("DO NOT REPLACE")
    with pytest.raises(FileExistsError):
        publication.make_outputs(tmp_path, write=True)
    assert existing.read_text() == "DO NOT REPLACE"
    assert not (tmp_path / publication.TABLE_SUMMARY).exists()


def test_detects_corrupt_model_artifact(tmp_path):
    _, v2, _ = setup_fixture(tmp_path)
    (v2 / "A_base.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        publication.make_outputs(tmp_path, write=False)


def test_rejects_wrong_method(tmp_path):
    v1, _, _ = setup_fixture(tmp_path)
    data = json.loads((v1 / "run.json").read_text())
    data["selected_method_on_january"] = "sigmoid"
    (v1 / "run.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="originally declared isotonic"):
        publication.make_outputs(tmp_path, write=False)


def test_detects_modified_v2_metrics(tmp_path):
    _, v2, _ = setup_fixture(tmp_path)
    metrics = pd.read_csv(v2 / "metrics.csv")
    metrics.loc[(metrics.stage=="B_february")&(metrics.method=="sigmoid"), "log_loss"] += .01
    modified = metrics.to_csv(index=False).encode()
    (v2 / "metrics.csv").write_bytes(modified)
    manifest_path = v2 / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"]["metrics.csv"] = hashlib.sha256(modified).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="v1-v2 changed metric"):
        publication.make_outputs(tmp_path, write=False)


def test_detects_changed_figure(tmp_path):
    _, _, viz = setup_fixture(tmp_path)
    (viz / "B_february_shared_deciles.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="visualization source"):
        publication.make_outputs(tmp_path, write=False)


def test_rejects_holdout_scored(tmp_path):
    _, v2, _ = setup_fixture(tmp_path)
    manifest_path = v2 / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["march_april_scored"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="holdout-scored"):
        publication.make_outputs(tmp_path, write=False)
