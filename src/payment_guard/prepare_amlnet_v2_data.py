"""Prepare AMLNet data using explicitly permitted transaction fields."""

from __future__ import annotations

import ast
from datetime import datetime


METADATA_FIELDS = {
    "metadata.payment_method": ("payment_method",),
    "metadata.device_info.os": ("device_info", "os"),
    "metadata.device_info.type": ("device_info", "type"),
    "metadata.location.city": ("location", "city"),
    "metadata.location.state": ("location", "state"),
}


def dictionary_nodes(node: ast.AST) -> dict[str, ast.AST]:
    """Read dictionary syntax without executing its contents."""
    if not isinstance(node, ast.Dict):
        raise ValueError("Expected a metadata dictionary.")

    result = {}

    for key, value in zip(node.keys, node.values):
        if not (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
        ):
            raise ValueError("Expected a string dictionary key.")

        if key.value in result:
            raise ValueError(f"Duplicate metadata key: {key.value}")

        result[key.value] = value

    return result


def extract_metadata(text: str) -> dict[str, object]:
    """Extract permitted strings and a validated naive timestamp."""
    if not isinstance(text, str):
        raise ValueError("Metadata must be a string.")

    root = ast.parse(text, mode="eval").body
    top_nodes = dictionary_nodes(root)
    result = {}

    for field, path in METADATA_FIELDS.items():
        if path[0] not in top_nodes:
            raise ValueError(f"Missing required field: {field}")

        node = top_nodes[path[0]]

        if len(path) == 2:
            children = dictionary_nodes(node)

            if path[1] not in children:
                raise ValueError(f"Missing required field: {field}")

            node = children[path[1]]

        value = ast.literal_eval(node)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Expected a nonempty string for {field}")

        result[field] = value

    node = top_nodes.get("timestamp")

    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "datetime"
        and node.func.attr == "datetime"
        and not node.keywords
        and 6 <= len(node.args) <= 7
    ):
        raise ValueError("Unexpected timestamp expression.")

    parts = []

    for argument in node.args:
        if not (
            isinstance(argument, ast.Constant)
            and type(argument.value) is int
        ):
            raise ValueError("Expected integer timestamp components.")

        parts.append(argument.value)

    result["timestamp"] = datetime(*parts)
    return result


TOP_LEVEL_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "type",
    "category",
]


def prepare_chunk(raw, source_offset=0):
    """Validate and extract one chunk without sorting or fitting transforms."""
    import numpy as np
    import pandas as pd

    if raw.empty:
        raise ValueError("Cannot prepare an empty chunk.")

    if type(source_offset) is not int or source_offset < 0:
        raise ValueError("Source offset must be a nonnegative integer.")

    raw = raw.reset_index(drop=True)
    output = raw[TOP_LEVEL_FEATURES + ["isFraud"]].copy()

    numeric_fields = [
        "amount",
        "oldbalanceOrg",
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "isFraud",
    ]

    for field in numeric_fields:
        values = pd.to_numeric(output[field], errors="raise")

        if not np.isfinite(
            values.to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Missing or nonfinite numbers: {field}")

        output[field] = values

    if (output["amount"] < 0).any():
        raise ValueError("Negative transaction amount.")

    if not output["isFraud"].isin([0, 1]).all():
        raise ValueError("Outcome must be zero or one.")

    output["isFraud"] = output["isFraud"].astype("int8")

    for field in ("type", "category"):
        values = output[field]

        if values.isna().any():
            raise ValueError(f"Missing values: {field}")

        if not values.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError(f"Expected nonempty strings: {field}")

    records = []

    for index, text in enumerate(raw["metadata"]):
        source_row = source_offset + index + 1

        try:
            records.append(extract_metadata(text))
        except (
            SyntaxError,
            TypeError,
            ValueError,
            RecursionError,
            OverflowError,
        ) as error:
            raise ValueError(
                f"Metadata failed at source row {source_row}"
            ) from error

    output = pd.concat(
        [output, pd.DataFrame(records)],
        axis=1,
    )

    calendar_attributes = {
        "hour": "hour",
        "day_of_week": "dayofweek",
        "day_of_month": "day",
        "month": "month",
    }

    for field, attribute in calendar_attributes.items():
        expected = getattr(output["timestamp"].dt, attribute)

        if not output[field].eq(expected).all():
            raise ValueError(f"Timestamp/calendar disagreement: {field}")

        output[field] = output[field].astype("int8")

    output.insert(
        0,
        "source_row",
        np.arange(
            source_offset + 1,
            source_offset + len(output) + 1,
            dtype=np.int64,
        ),
    )

    return output


VALIDATION_START = "2026-02-01"
TEST_START = "2026-03-01"


def validate_feature_policy(policy_path):
    """Require the implemented feature set to match the audited policy."""
    import pandas as pd

    policy = pd.read_csv(policy_path)

    required_columns = {"field", "resolved_model_status"}

    if not required_columns.issubset(policy.columns):
        raise ValueError("Feature policy is missing required columns.")

    if policy["field"].isna().any():
        raise ValueError("Feature policy contains missing field names.")

    if policy["field"].duplicated().any():
        raise ValueError("Feature policy contains duplicate fields.")

    statuses = {
        "include",
        "include_with_assumption",
        "include_with_ablation",
    }

    approved = set(
        policy.loc[
            policy["resolved_model_status"].isin(statuses),
            "field",
        ]
    )

    implemented = set(TOP_LEVEL_FEATURES) | set(METADATA_FIELDS)

    if approved != implemented:
        raise ValueError(
            "Feature-policy mismatch. "
            f"Approved but absent: {sorted(approved - implemented)}. "
            f"Implemented but unapproved: {sorted(implemented - approved)}."
        )

    roles = policy.set_index("field")["resolved_model_status"]

    if roles.get("isFraud") != "target":
        raise ValueError("isFraud must be the target.")

    if roles.get("metadata.timestamp") != "split_only":
        raise ValueError("Timestamp must have the split_only role.")

    return sorted(approved)


def sort_and_split(prepared):
    """Order records reproducibly and assign fixed calendar splits."""
    import numpy as np
    import pandas as pd

    expected_columns = (
        set(TOP_LEVEL_FEATURES)
        | set(METADATA_FIELDS)
        | {"source_row", "timestamp", "isFraud"}
    )

    if set(prepared.columns) != expected_columns:
        raise ValueError("Unexpected prepared-table columns.")

    if prepared.empty:
        raise ValueError("Cannot split an empty table.")

    if prepared["timestamp"].isna().any():
        raise ValueError("Missing timestamps.")

    if (
        prepared["source_row"].isna().any()
        or prepared["source_row"].duplicated().any()
    ):
        raise ValueError("Source references must be present and unique.")

    ordered = prepared.sort_values(
        ["timestamp", "source_row"],
    ).reset_index(drop=True)

    validation_start = pd.Timestamp(VALIDATION_START)
    test_start = pd.Timestamp(TEST_START)

    ordered["split"] = np.select(
        [
            ordered["timestamp"] < validation_start,
            ordered["timestamp"] < test_start,
        ],
        ["train", "validation"],
        default="test",
    )

    if not ordered["timestamp"].is_monotonic_increasing:
        raise ValueError("Chronological sorting failed.")

    return ordered


def publish_output_pair(temporary_table, table_path, temporary_report, report_path):
    """Publish without overwriting; roll back the table on report failure.

    Staged files must be on the destination filesystem. Hard links provide
    exclusive creation, unlike replace(), which can overwrite another run.
    This handles raised errors, not process termination or power loss.
    Readers must require both files before treating a run as complete.
    """
    import os

    os.link(temporary_table, table_path)
    try:
        os.link(temporary_report, report_path)
    except BaseException:
        table_path.unlink()
        raise


def main():
    """Build the full prepared table and record verification evidence."""
    import hashlib
    import json
    import resource
    import sys
    from tempfile import TemporaryDirectory
    from time import perf_counter
    from pathlib import Path

    import pandas as pd

    root = Path(__file__).resolve().parents[2]
    raw_path = root / "data/raw/amlnet_v2/AMLNet_v2_transactions.csv"
    policy_path = (
        root
        / "results/tables/amlnet_v2_initial_model_feature_policy.csv"
    )
    output_dir = root / "data/processed"
    table_path = output_dir / "amlnet_v2_prepared.parquet"
    report_path = output_dir / "amlnet_v2_preparation_report.json"

    if table_path.exists() or report_path.exists():
        raise FileExistsError(
            "Preparation outputs already exist; refusing to overwrite."
        )

    started = perf_counter()
    features = validate_feature_policy(policy_path)

    digest = hashlib.sha256()

    with raw_path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    expected_checksum = (
        "09328eb063d525a23a1ce9be663615fae4"
        "afdfa082f2b622c462b7d0fe0eab2d"
    )

    if digest.hexdigest() != expected_checksum:
        raise ValueError("Raw-file checksum differs from the audited source.")

    chunks = []
    row_count = 0
    extraction_started = perf_counter()

    reader = pd.read_csv(
        raw_path,
        usecols=TOP_LEVEL_FEATURES + ["isFraud", "metadata"],
        chunksize=50_000,
    )

    for number, raw in enumerate(reader, start=1):
        chunk = prepare_chunk(raw, source_offset=row_count)
        row_count += len(chunk)
        chunks.append(chunk)

        print(
            f"Chunk {number}: {row_count:,} rows validated",
            flush=True,
        )

    extraction_seconds = perf_counter() - extraction_started
    prepared = pd.concat(chunks, ignore_index=True)
    del chunks

    positives = int(prepared["isFraud"].sum())

    if row_count != 1_090_000 or positives != 1_411:
        raise ValueError(
            f"Audit reconciliation failed: {row_count} rows, "
            f"{positives} positives."
        )

    sorting_started = perf_counter()
    prepared = sort_and_split(prepared)
    sorting_seconds = perf_counter() - sorting_started

    summary = prepared.groupby("split").agg(
        rows=("isFraud", "size"),
        positive_labels=("isFraud", "sum"),
        earliest=("timestamp", "min"),
        latest=("timestamp", "max"),
    )

    periods = ["train", "validation", "test"]

    if set(summary.index) != set(periods):
        raise ValueError("All three splits must contain transactions.")

    for earlier, later in zip(periods, periods[1:]):
        if not (
            summary.loc[earlier, "latest"]
            < summary.loc[later, "earliest"]
        ):
            raise ValueError("Split periods overlap.")

    duplicate_columns = features + ["timestamp", "isFraud"]
    repeated_rows = int(
        prepared.duplicated(
            subset=duplicate_columns,
            keep=False,
        ).sum()
    )

    memory_bytes = int(
        prepared.memory_usage(index=True, deep=True).sum()
    )

    split_report = {}

    for split in periods:
        row = summary.loc[split]
        count = int(row["rows"])
        positive = int(row["positive_labels"])

        split_report[split] = {
            "rows": count,
            "positive_labels": positive,
            "positive_rate": positive / count,
            "earliest": row["earliest"].isoformat(),
            "latest": row["latest"].isoformat(),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    writing_started = perf_counter()

    with TemporaryDirectory(prefix=".amlnet-preparation-", dir=output_dir) as staging:
        temporary_table = Path(staging) / table_path.name
        temporary_report = Path(staging) / report_path.name
        prepared.to_parquet(
            temporary_table,
            engine="pyarrow",
            index=False,
        )

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = (
            peak_rss if sys.platform == "darwin" else peak_rss * 1024
        )

        report = {
            "raw_sha256": digest.hexdigest(),
            "rows": row_count,
            "positive_labels": positives,
            "predictors": features,
            "validation_start": VALIDATION_START,
            "test_start": TEST_START,
            "splits": split_report,
            "rows_in_repeated_prepared_groups": repeated_rows,
            "duplicate_interpretation": (
                "Equality of selected predictors, timestamp and outcome; "
                "not proof of duplicate business transactions. Rows retained."
            ),
            "prepared_memory_bytes": memory_bytes,
            "process_peak_rss_bytes": int(peak_bytes),
            "extraction_seconds": extraction_seconds,
            "sorting_seconds": sorting_seconds,
            "parquet_write_seconds": perf_counter() - writing_started,
            "elapsed_before_publication_seconds": perf_counter() - started,
            "test_status": (
                "Chronological holdout after full-period exploratory auditing."
            ),
        }

        temporary_report.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        publish_output_pair(
            temporary_table, table_path, temporary_report, report_path
        )

    print(summary.loc[periods].to_string())
    print(f"Rows in repeated prepared groups: {repeated_rows:,}")
    print(f"Prepared memory: {memory_bytes / 1024**2:.2f} MiB")
    print(f"Process peak RSS: {peak_bytes / 1024**2:.2f} MiB")
    print(f"Total elapsed: {perf_counter() - started:.2f} seconds")
    print(f"Table: {table_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
