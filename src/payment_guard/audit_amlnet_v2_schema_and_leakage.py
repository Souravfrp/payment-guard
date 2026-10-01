"""Audit AMLNet Version 2.0 fields for chronology and leakage risk.

This script complements validate_amlnet_v2_data.py. It does not repeat the
general schema and quality checks. Instead, it examines timestamp consistency,
metadata structure, decision-time availability, and initial feature eligibility.
"""

from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "amlnet_v2"
    / "AMLNet_v2_transactions.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "amlnet_v2_leakage_audit.csv"
)

CHUNK_SIZE_ROWS = 50_000
AUDIT_INPUT_COLUMNS = [
    "metadata",
    "isFraud",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
]

TARGET_MARKER_PATHS = {
    "integration_info": "metadata.integration_info",
    "layering": "metadata.layering",
    "structuring": "metadata.structuring",
}

TARGET_REVEALING_SIBLINGS = {
    "metadata.sophistication": "integration_info",
    "metadata.layering_sophistication": "layering",
}

TIMESTAMP_PATTERN = re.compile(
    r"'timestamp':\s*datetime\.datetime\(\s*"
    r"(?P<year>\d{4}),\s*"
    r"(?P<month>\d{1,2}),\s*"
    r"(?P<day>\d{1,2}),\s*"
    r"(?P<hour>\d{1,2}),\s*"
    r"(?P<minute>\d{1,2}),\s*"
    r"(?P<second>\d{1,2})"
    r"(?:,\s*(?P<microsecond>\d{1,6}))?\s*\)"
)

CALENDAR_FIELDS = {
    "hour": "hour",
    "day_of_week": "dayofweek",
    "day_of_month": "day",
    "month": "month",
}

TOP_LEVEL_POLICIES = {
    "step": (
        "time/index field",
        "no",
        "exclude",
        "The released CSV contains only zero, so it cannot identify or order transactions.",
    ),
    "type": (
        "transaction attribute",
        "yes",
        "include",
        "The transaction type is treated as known when the payment request arrives.",
    ),
    "amount": (
        "transaction attribute",
        "yes",
        "include",
        "The requested amount is known before an approve, review, or block decision.",
    ),
    "category": (
        "transaction attribute",
        "yes",
        "include",
        "The transaction category is treated as decision-time information.",
    ),
    "nameOrig": (
        "account identifier",
        "yes",
        "derive_with_history",
        "Use only for aggregates built from earlier transactions; do not use the raw identifier directly.",
    ),
    "nameDest": (
        "account identifier",
        "yes",
        "derive_with_history",
        "Use only for aggregates built from earlier transactions; do not use the raw identifier directly.",
    ),
    "oldbalanceOrg": (
        "pre-transaction balance",
        "yes",
        "include",
        "The dataset defines this as the origin balance before the transaction; retain that provenance in the documentation.",
    ),
    "newbalanceOrig": (
        "post-transaction balance",
        "no",
        "exclude",
        "The dataset defines this as the origin balance after the transaction, so it is unavailable for a pre-authorization decision.",
    ),
    "isFraud": (
        "outcome label",
        "no",
        "target",
        "This is the selected project target and must never be used as a predictor.",
    ),
    "isMoneyLaundering": (
        "duplicate outcome label",
        "no",
        "exclude",
        "Validation found it identical to isFraud in every row.",
    ),
    "laundering_typology": (
        "outcome description",
        "no",
        "exclude",
        "The typology describes the labelled event and may reveal the target.",
    ),
    "metadata": (
        "nested-data container",
        "not_applicable",
        "exclude",
        "The raw text is not a model feature; separately audited nested fields may be used.",
    ),
    "fraud_probability": (
        "supplied risk score",
        "uncertain",
        "exclude",
        "Its construction and decision-time availability are not established, and it is closely related to the target.",
    ),
    "hour": (
        "calendar attribute",
        "yes_if_consistent",
        "include_if_consistent",
        "Include only if it agrees with the safely extracted timestamp.",
    ),
    "day_of_week": (
        "calendar attribute",
        "yes_if_consistent",
        "include_if_consistent",
        "Include only if it agrees with the safely extracted timestamp.",
    ),
    "day_of_month": (
        "calendar attribute",
        "yes_if_consistent",
        "include_if_consistent",
        "Include only if it agrees with the safely extracted timestamp.",
    ),
    "month": (
        "calendar attribute",
        "yes_if_consistent",
        "include_if_consistent",
        "Include only if it agrees with the safely extracted timestamp.",
    ),
}


def extract_timestamps(metadata: pd.Series) -> pd.Series:
    """Extract timestamps with a regular expression without executing metadata."""
    parts = metadata.str.extract(TIMESTAMP_PATTERN)

    for column in parts.columns:
        parts[column] = pd.to_numeric(parts[column], errors="coerce")

    timestamps = pd.to_datetime(
        parts[["year", "month", "day", "hour", "minute", "second"]],
        errors="coerce",
    )
    microseconds = parts["microsecond"].fillna(0)
    return timestamps + pd.to_timedelta(microseconds, unit="us")


def collect_dict_paths(node: ast.AST, prefix: str, paths: set[str]) -> None:
    """Collect dictionary-key paths from a parsed syntax tree."""
    if isinstance(node, ast.Dict):
        for key_node, value_node in zip(node.keys, node.values):
            if not (
                isinstance(key_node, ast.Constant)
                and isinstance(key_node.value, str)
            ):
                continue

            path = f"{prefix}.{key_node.value}"
            paths.add(path)
            collect_dict_paths(value_node, path, paths)

    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for item in node.elts:
            collect_dict_paths(item, prefix, paths)


def extract_metadata_paths(raw_metadata: str) -> set[str]:
    """Inspect metadata structure with AST parsing, without evaluating it."""
    expression = ast.parse(raw_metadata, mode="eval")
    paths: set[str] = set()
    collect_dict_paths(expression.body, "metadata", paths)
    return paths


def audit_dataset() -> dict:
    """Scan the CSV once and collect temporal and metadata evidence."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    columns = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
    missing_audit_columns = sorted(set(AUDIT_INPUT_COLUMNS) - set(columns))

    if missing_audit_columns:
        raise ValueError(
            "Required audit columns are missing: "
            + ", ".join(missing_audit_columns)
        )

    row_count = 0
    timestamp_parse_failures = 0
    chronological_decreases = 0
    calendar_mismatches = Counter()
    metadata_parse_failures = 0
    metadata_path_counts = Counter()
    metadata_path_label_counts = defaultdict(Counter)
    target_marker_label_counts = {
        name: Counter() for name in TARGET_MARKER_PATHS
    }
    combined_target_marker_counts = Counter()
    target_marker_overlap_count = 0
    target_label_counts = Counter()
    observed_dates = set()
    earliest_timestamp = None
    latest_timestamp = None
    previous_timestamp = None

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            usecols=AUDIT_INPUT_COLUMNS,
            chunksize=CHUNK_SIZE_ROWS,
        ),
        start=1,
    ):
        row_count += len(chunk)
        metadata = chunk["metadata"].fillna("").astype(str)
        numeric_labels = pd.to_numeric(chunk["isFraud"], errors="raise")
        unexpected_labels = sorted(
            value for value in numeric_labels.unique() if value not in {0, 1}
        )
        if unexpected_labels:
            raise ValueError(
                "Unexpected isFraud values: "
                + ", ".join(map(str, unexpected_labels))
            )
        labels = numeric_labels.astype(int)
        target_label_counts.update(labels.tolist())
        timestamps = extract_timestamps(metadata)
        valid_timestamps = timestamps.dropna()
        timestamp_parse_failures += int(timestamps.isna().sum())

        if not valid_timestamps.empty:
            chunk_earliest = valid_timestamps.min()
            chunk_latest = valid_timestamps.max()

            if earliest_timestamp is None or chunk_earliest < earliest_timestamp:
                earliest_timestamp = chunk_earliest

            if latest_timestamp is None or chunk_latest > latest_timestamp:
                latest_timestamp = chunk_latest

            if (
                previous_timestamp is not None
                and valid_timestamps.iloc[0] < previous_timestamp
            ):
                chronological_decreases += 1

            chronological_decreases += int(
                valid_timestamps.diff().lt(pd.Timedelta(0)).sum()
            )
            previous_timestamp = valid_timestamps.iloc[-1]
            observed_dates.update(valid_timestamps.dt.date.unique())

        for csv_field, timestamp_attribute in CALENDAR_FIELDS.items():
            csv_values = pd.to_numeric(chunk[csv_field], errors="coerce")
            timestamp_values = getattr(timestamps.dt, timestamp_attribute)
            comparable = timestamps.notna() & csv_values.notna()
            calendar_mismatches[csv_field] += int(
                (csv_values[comparable] != timestamp_values[comparable]).sum()
            )

        for raw_metadata, label in zip(metadata, labels):
            try:
                paths = extract_metadata_paths(raw_metadata)
            except (SyntaxError, TypeError, ValueError, RecursionError):
                metadata_parse_failures += 1
                continue

            metadata_path_counts.update(paths)
            path_label_name = (
                "fraud_present" if label == 1 else "nonfraud_present"
            )
            for path in paths:
                metadata_path_label_counts[path][path_label_name] += 1

            marker_presence = {
                name: marker_path in paths
                for name, marker_path in TARGET_MARKER_PATHS.items()
            }

            for name, present in marker_presence.items():
                if present:
                    label_name = (
                        "fraud_present"
                        if label == 1
                        else "nonfraud_present"
                    )
                    target_marker_label_counts[name][label_name] += 1

            markers_in_row = sum(marker_presence.values())
            if markers_in_row:
                label_name = (
                    "fraud_present"
                    if label == 1
                    else "nonfraud_present"
                )
                combined_target_marker_counts[label_name] += 1
            elif label == 1:
                combined_target_marker_counts["fraud_absent"] += 1

            if markers_in_row > 1:
                target_marker_overlap_count += 1

        print(
            f"Processed chunk {chunk_number}: "
            f"{row_count:,} cumulative rows"
        )

    return {
        "columns": columns,
        "row_count": row_count,
        "timestamp_parse_failures": timestamp_parse_failures,
        "chronological_decreases": chronological_decreases,
        "calendar_mismatches": calendar_mismatches,
        "metadata_parse_failures": metadata_parse_failures,
        "metadata_path_counts": metadata_path_counts,
        "metadata_path_label_counts": metadata_path_label_counts,
        "target_marker_label_counts": target_marker_label_counts,
        "combined_target_marker_counts": combined_target_marker_counts,
        "target_marker_overlap_count": target_marker_overlap_count,
        "target_label_counts": target_label_counts,
        "observed_date_count": len(observed_dates),
        "earliest_timestamp": earliest_timestamp,
        "latest_timestamp": latest_timestamp,
    }


def target_marker_group_for_field(field: str) -> str | None:
    """Return the target-marker group associated with one metadata field."""
    sibling_group = TARGET_REVEALING_SIBLINGS.get(field)
    if sibling_group is not None:
        return sibling_group

    for group, root_path in TARGET_MARKER_PATHS.items():
        if field == root_path or field.startswith(f"{root_path}."):
            return group

    return None


def direct_target_leakage_confirmed(results: dict) -> bool:
    """Test whether the three markers exactly and exclusively reveal fraud."""
    group_counts = results["target_marker_label_counts"]
    combined = results["combined_target_marker_counts"]
    fraud_total = results["target_label_counts"][1]

    return (
        fraud_total > 0
        and all(
            group_counts[name]["nonfraud_present"] == 0
            and group_counts[name]["fraud_present"] > 0
            for name in TARGET_MARKER_PATHS
        )
        and combined["nonfraud_present"] == 0
        and combined["fraud_present"] == fraud_total
        and combined["fraud_absent"] == 0
        and results["target_marker_overlap_count"] == 0
    )


def field_target_leakage_confirmed(field: str, results: dict) -> bool:
    """Confirm that one annotated field occurs only in fraud-labelled rows."""
    if (
        target_marker_group_for_field(field) is None
        or not direct_target_leakage_confirmed(results)
    ):
        return False

    counts = results["metadata_path_label_counts"][field]
    return (
        counts["fraud_present"] > 0
        and counts["nonfraud_present"] == 0
    )


def nested_field_policy(
    field: str,
    target_leakage_confirmed: bool,
) -> tuple[str, str, str, str]:
    """Assign a conservative initial policy to one nested metadata field."""
    marker_group = target_marker_group_for_field(field)
    if marker_group is not None and target_leakage_confirmed:
        return (
            "target-revealing annotation",
            "no",
            "exclude",
            f"The {marker_group} metadata group occurs only in fraud-labelled rows and directly reveals the supplied target.",
        )

    if field == "metadata.timestamp":
        return (
            "transaction timestamp",
            "yes",
            "split_only",
            "Use for chronological ordering and time-derived features, not as an unrestricted raw predictor.",
        )

    if field.startswith("metadata.risk_indicators"):
        return (
            "supplied risk indicator",
            "uncertain",
            "exclude",
            "Its construction may incorporate target-related or future information.",
        )

    if field.startswith("metadata.network_metrics"):
        return (
            "network-derived attribute",
            "uncertain",
            "exclude",
            "Use only after proving that it can be calculated from information available before each decision.",
        )

    if field.endswith(".ip_address"):
        return (
            "device identifier",
            "yes",
            "derive_with_history",
            "Do not use the raw high-cardinality identifier; later derive leakage-safe historical behaviour.",
        )

    if field in {
        "metadata.location",
        "metadata.device_info",
        "metadata.merchant_info",
    }:
        return (
            "nested-data container",
            "not_applicable",
            "exclude",
            "The container itself is not a model feature; assess its child fields separately.",
        )

    if field.startswith("metadata.location"):
        return (
            "location attribute",
            "uncertain",
            "needs_review",
            "Confirm that the field is captured before authorization and choose an appropriate encoding.",
        )

    if field.startswith("metadata.device_info"):
        return (
            "device attribute",
            "uncertain",
            "needs_review",
            "Confirm decision-time availability before including this field.",
        )

    if field == "metadata.payment_method":
        return (
            "payment attribute",
            "yes",
            "include",
            "The selected payment method is treated as known when the transaction arrives.",
        )

    if field.startswith("metadata.merchant_info"):
        return (
            "merchant attribute",
            "uncertain",
            "needs_review",
            "Its meaning, missingness, cardinality, and decision-time availability must be established.",
        )

    return (
        "unclassified metadata attribute",
        "uncertain",
        "needs_review",
        "The audit discovered this field, but there is not yet enough evidence to include it.",
    )


def build_audit_table(results: dict) -> pd.DataFrame:
    """Combine observed evidence with explicit initial modelling policies."""
    rows = []
    row_count = results["row_count"]
    all_calendar_fields_consistent = (
        results["timestamp_parse_failures"] == 0
        and not any(results["calendar_mismatches"].values())
    )
    for field in results["columns"]:
        role, availability, status, reason = TOP_LEVEL_POLICIES.get(
            field,
            (
                "unclassified top-level field",
                "uncertain",
                "needs_review",
                "This unexpected field requires manual investigation.",
            ),
        )

        evidence = "Observed as a top-level CSV column."

        if field in CALENDAR_FIELDS:
            mismatch_count = results["calendar_mismatches"][field]
            evidence = (
                f"Timestamp comparison found {mismatch_count:,} "
                "mismatched rows."
            )
            if all_calendar_fields_consistent:
                availability = "yes"
                status = "include"
            else:
                availability = "uncertain"
                status = "needs_review"

        rows.append(
            {
                "field": field,
                "location": "top_level",
                "structural_presence_rows": row_count,
                "project_role": role,
                "decision_time_status": availability,
                "initial_model_status": status,
                "reason": reason,
                "audit_evidence": evidence,
            }
        )

    for field, presence_count in sorted(
        results["metadata_path_counts"].items()
    ):
        role, availability, status, reason = nested_field_policy(
            field,
            field_target_leakage_confirmed(field, results),
        )
        evidence = f"Detected structurally in {presence_count:,} rows."

        marker_group = target_marker_group_for_field(field)
        if field_target_leakage_confirmed(field, results):
            group_counts = results["target_marker_label_counts"][marker_group]
            field_counts = results["metadata_path_label_counts"][field]
            evidence = (
                f"This field was structurally present in "
                f"{field_counts['fraud_present']:,} fraud rows and "
                f"{field_counts['nonfraud_present']:,} non-fraud rows; "
                f"the {marker_group} marker covered "
                f"{group_counts['fraud_present']:,} fraud rows and "
                f"{group_counts['nonfraud_present']:,} non-fraud rows."
            )

        if field == "metadata.timestamp":
            evidence = (
                f"Parsed {row_count - results['timestamp_parse_failures']:,} "
                f"of {row_count:,} rows; range "
                f"{results['earliest_timestamp']} to "
                f"{results['latest_timestamp']}; "
                f"chronological decreases: "
                f"{results['chronological_decreases']:,}."
            )

        rows.append(
            {
                "field": field,
                "location": "metadata",
                "structural_presence_rows": presence_count,
                "project_role": role,
                "decision_time_status": availability,
                "initial_model_status": status,
                "reason": reason,
                "audit_evidence": evidence,
            }
        )

    return pd.DataFrame(rows)


def print_results(results: dict, audit_table: pd.DataFrame) -> None:
    """Print the evidence needed for the next project decisions."""
    print("\nAMLNet Version 2.0 schema and leakage audit")
    print(f"Rows scanned: {results['row_count']:,}")
    print(
        "Timestamp parse failures: "
        f"{results['timestamp_parse_failures']:,}"
    )
    print(f"Earliest timestamp: {results['earliest_timestamp']}")
    print(f"Latest timestamp: {results['latest_timestamp']}")
    print(f"Observed calendar dates: {results['observed_date_count']:,}")
    print(
        "Chronological decreases in CSV row order: "
        f"{results['chronological_decreases']:,}"
    )

    print("\nCalendar-field mismatches:")
    for field in CALENDAR_FIELDS:
        print(f"  {field}: {results['calendar_mismatches'][field]:,}")

    print(
        "\nMetadata structural parse failures: "
        f"{results['metadata_parse_failures']:,}"
    )
    print(
        "Discovered metadata paths: "
        f"{len(results['metadata_path_counts']):,}"
    )

    print("\nTarget-marker presence by fraud label:")
    for name in TARGET_MARKER_PATHS:
        counts = results["target_marker_label_counts"][name]
        print(
            f"  {name}: "
            f"non-fraud={counts['nonfraud_present']:,}, "
            f"fraud={counts['fraud_present']:,}"
        )

    combined = results["combined_target_marker_counts"]
    print("Target-marker combined result:")
    print(
        "  Non-fraud rows with any marker: "
        f"{combined['nonfraud_present']:,}"
    )
    print(
        "  Fraud rows with any marker: "
        f"{combined['fraud_present']:,}"
    )
    print(
        "  Fraud rows without a marker: "
        f"{combined['fraud_absent']:,}"
    )
    print(
        "  Rows containing multiple markers: "
        f"{results['target_marker_overlap_count']:,}"
    )
    print(
        "  Direct target leakage confirmed: "
        f"{direct_target_leakage_confirmed(results)}"
    )

    status_counts = audit_table["initial_model_status"].value_counts()
    print("\nInitial field-policy counts:")
    for status, count in status_counts.items():
        print(f"  {status}: {count:,}")

    print(f"\nSaved audit table: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")

    issues = (
        results["timestamp_parse_failures"]
        + results["chronological_decreases"]
        + results["metadata_parse_failures"]
        + sum(results["calendar_mismatches"].values())
    )

    if issues:
        print("Audit result: COMPLETED WITH ISSUES TO RESOLVE")
    else:
        print("Audit result: COMPLETED")


def main() -> None:
    """Run the audit and save its reproducible field-level output."""
    results = audit_dataset()
    audit_table = build_audit_table(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_table.to_csv(OUTPUT_PATH, index=False)
    print_results(results, audit_table)


if __name__ == "__main__":
    main()
