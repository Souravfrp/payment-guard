"""Read-only, no-scoring preflight for the frozen final holdout protocol.

Only reads Git-tracked code, JSON manifests and artifact bytes to compute hashes.
Never loads a fitted model, Parquet rows or any individual fraud labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("Empty/invalid local path")
    parent = root.resolve()
    path = (parent / relative).resolve()
    if not path.is_relative_to(parent):
        raise ValueError("Path outside the project root")
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen input: {path}")
    return path


def verify_manifest(directory: Path, expected_source_sha: str) -> dict:
    path = directory / "run.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("complete") is not True or metadata.get("march_april_scored") is not False:
        raise ValueError(f"Incomplete/holdout-scored run: {directory}")
    if metadata.get("input_sha256") != expected_source_sha:
        raise ValueError(f"Prepared dataset hash changed: {directory}")
    artifacts = metadata.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"Missing artifact hashes: {directory}")
    for name, fingerprint in artifacts.items():
        if (not isinstance(name, str) or name in (".", "..")
                or "/" in name or "\\" in name):
            raise ValueError("Unsafe artifact name")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError(f"Invalid artifact hash: {name}")
        if digest(directory / name) != fingerprint:
            raise ValueError(f"Artifact checksum mismatch: {directory / name}")
    return metadata


def validate_spec(spec: dict) -> None:
    if spec.get("schema_version") != 1 or spec.get("state") != "PROTOCOL_FROZEN__MODEL_HASH_PREFLIGHT_PENDING":
        raise ValueError("Unknown or already-modified freeze contract")
    if len(spec.get("source_features", [])) != 13:
        raise ValueError("Expected exactly 13 frozen source features")
    if spec["data"].get("holdout_split") != "test" or spec["data"].get("expected_holdout_rows") != 253190:
        raise ValueError("Holdout boundaries changed")
    if (spec["primary"].get("family") != "xgboost"
            or spec["primary"].get("score") != "uncalibrated"
            or spec["comparator"].get("family") != "logistic"
            or spec["comparator"].get("score") != "uncalibrated"):
        raise ValueError("Primary family or score policy changed")
    rank = spec["ranking"]
    if (rank.get("primary_budget_fraction") != 0.01
            or rank.get("secondary_budget_fractions") != [0.001, 0.005]
            or rank.get("tie_seed") != 20261005):
        raise ValueError("Review budget or tie rule changed")
    if (len(spec.get("fixed_sensitivity_artifacts", [])) != 4
            or len(spec.get("secondary_calibration_artifacts", [])) != 4
            or len(spec.get("pinned_source_git_blobs", {})) != 5):
        raise ValueError("Incomplete predeclared models or code pins")
    if spec["data"].get("march_april_scored") is not False:
        raise ValueError("Freeze no longer describes an unscored holdout")


def preflight(root: Path, spec: dict) -> dict:
    """Require original saved model artifacts and source code; do not score."""
    validate_spec(spec)
    pinned = {}
    for rel, expected in spec["pinned_source_git_blobs"].items():
        path = checked_path(root, rel)
        actual = subprocess.run(
            ["git", "hash-object", str(path)],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip()
        if actual != expected:
            raise ValueError(f"Frozen source changed: {rel}")
        pinned[rel] = actual

    source = checked_path(root, spec["data"]["prepared_parquet"])
    source_sha = digest(source)  # Reads bytes for integrity; parses no holdout records.

    refs = spec["artifact_verification"]["local_run_manifests"]
    if len(refs) != 3:
        raise ValueError("Expected logistic, XGBoost and ablation run manifests")
    manifest_shas = {}
    loaded = {}
    for rel in refs:
        path = checked_path(root, rel)
        manifest_shas[rel] = digest(path)
        loaded[rel] = verify_manifest(path.parent, source_sha)

    logistic_path, xgb_path, ablation_path = refs
    logistic, xgb, ablation = (loaded[logistic_path], loaded[xgb_path], loaded[ablation_path])
    if xgb.get("chosen_configuration") != "depth3_240":
        raise ValueError("XGBoost selection is not the frozen December choice")
    if xgb.get("features") != spec["source_features"]:
        raise ValueError("Frozen XGBoost feature order changed")
    if xgb.get("existing_logistic_manifest_sha256") != manifest_shas[logistic_path]:
        raise ValueError("XGBoost/logistic manifest reference changed")
    if (ablation.get("locked_configuration") != "depth3_240"
            or ablation.get("reference_manifest_sha256") != manifest_shas[xgb_path]
            or ablation.get("logistic_manifest_sha256") != manifest_shas[logistic_path]):
        raise ValueError("Ablation full-model/manifest reference changed")
    required = (
        [spec["primary"]["local_artifact"], spec["comparator"]["local_artifact"]]
        + [x["local_artifact"] for x in spec["fixed_sensitivity_artifacts"]]
        + spec["secondary_calibration_artifacts"]
    )
    model_hashes = {}
    for rel in required:
        file = checked_path(root, rel)
        manifest = next((m for p, m in loaded.items() if file.parent == (root / p).parent.resolve()), None)
        if manifest is None or manifest["artifact_sha256"].get(file.name) != digest(file):
            raise ValueError(f"Model/map is missing from a verified original manifest: {rel}")
        model_hashes[rel] = digest(file)

    return {
        "status": "PASS_PRE_SCORE_NO_SCORING",
        "preflight_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_spec_sha256": digest(checked_path(root, "config/final_holdout_freeze_v1.json")),
        "prepared_parquet_sha256": source_sha,
        "source_git_blobs": pinned,
        "existing_run_manifest_sha256": manifest_shas,
        "selected_model_and_map_sha256": model_hashes,
        "march_april_scored": False,
        "note": "Checksums only. No model was loaded and no transaction was predicted.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validate-spec-only", action="store_true",
                        help="Validate the public frozen JSON without any local model access")
    parser.add_argument("--write-local-lock", action="store_true",
                        help="Create a new ignored models/final_holdout_preflight_v1.json file, never overwrite")
    args = parser.parse_args()
    root = args.root.resolve()
    spec = json.loads(checked_path(root, "config/final_holdout_freeze_v1.json").read_text(encoding="utf-8"))
    validate_spec(spec)
    if args.validate_spec_only:
        print("PASS: frozen final holdout specification is internally consistent; no model/holdout read")
        return
    result = preflight(root, spec)
    if args.write_local_lock:
        output = root / "models/final_holdout_preflight_v1.json"
        with output.open("x", encoding="utf-8") as target:
            json.dump(result, target, indent=2)
            target.write("\n")
        print("Created private pre-score model fingerprint:", output)
    print("PASS: frozen code, saved manifests and original model hashes match; no test predictions computed")
    print("Freeze-spec SHA-256:", result["freeze_spec_sha256"])
    print("Prepared-data SHA-256:", result["prepared_parquet_sha256"])
    for path, fingerprint in result["existing_run_manifest_sha256"].items():
        print(f"Manifest {path}: {fingerprint}")


if __name__ == "__main__":
    main()
