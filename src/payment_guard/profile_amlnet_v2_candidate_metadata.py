"""Profile unresolved AMLNet metadata fields before feature selection."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/raw/amlnet_v2/AMLNet_v2_transactions.csv"
OUTPUT_PATH = (
    PROJECT_ROOT / "results/tables/amlnet_v2_candidate_metadata_profile.csv"
)
CHUNK_SIZE_ROWS = 50_000
MINIMUM_GROUP_ROWS = 100

FIELD_LOCATIONS = {
    "metadata.device_info.os": ("device_info", "os"),
    "metadata.device_info.type": ("device_info", "type"),
    "metadata.location.city": ("location", "city"),
    "metadata.location.country": ("location", "country"),
    "metadata.location.postcode": ("location", "postcode"),
    "metadata.location.state": ("location", "state"),
    "metadata.merchant_info.avg_transaction": (
        "merchant_info",
        "avg_transaction",
    ),
    "metadata.merchant_info.category": ("merchant_info", "category"),
    "metadata.merchant_info.merchant_id": ("merchant_info", "merchant_id"),
    "metadata.merchant_info.risk_level": ("merchant_info", "risk_level"),
}
NUMERIC_FIELD = "metadata.merchant_info.avg_transaction"


def extract_values(raw_metadata: str) -> tuple[dict[str, object], set[str]]:
    """Extract candidate subfields with AST parsing, never eval()."""
    expression = ast.parse(raw_metadata, mode="eval")
    if not isinstance(expression.body, ast.Dict):
        raise ValueError("Metadata root is not a dictionary.")

    top_nodes = {}
    for key_node, value_node in zip(expression.body.keys, expression.body.values):
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            top_nodes[key_node.value] = value_node

    containers = {}
    for container_name in {item[0] for item in FIELD_LOCATIONS.values()}:
        node = top_nodes.get(container_name)
        value = ast.literal_eval(node) if node is not None else None
        containers[container_name] = value if isinstance(value, dict) else None

    values = {}
    present = set()
    for field, (container_name, child_name) in FIELD_LOCATIONS.items():
        container = containers[container_name]
        if container is not None and child_name in container:
            present.add(field)
            values[field] = container[child_name]
    return values, present


def normalise(value: object) -> str | None:
    """Convert one scalar value to a stable Counter key."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def scan_dataset() -> dict:
    """Scan the raw CSV once and collect field-level evidence."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    stats = {
        field: {
            "structural": 0,
            "non_null": 0,
            "values": Counter(),
            "fraud": Counter(),
        }
        for field in FIELD_LOCATIONS
    }
    row_count = 0
    parse_failures = 0
    label_counts = Counter()
    merchant_by_type = Counter()
    merchant_by_label = Counter()
    category_comparisons = 0
    category_matches = 0

    reader = pd.read_csv(
        DATA_PATH,
        usecols=["metadata", "isFraud", "type", "category"],
        chunksize=CHUNK_SIZE_ROWS,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        row_count += len(chunk)
        labels = pd.to_numeric(chunk["isFraud"], errors="raise").astype(int)

        for raw, label, transaction_type, category in zip(
            chunk["metadata"].fillna("").astype(str),
            labels,
            chunk["type"].astype(str),
            chunk["category"].astype(str),
        ):
            label_counts[label] += 1
            try:
                values, present = extract_values(raw)
            except (SyntaxError, TypeError, ValueError, RecursionError):
                parse_failures += 1
                continue

            for field in present:
                stats[field]["structural"] += 1
                value = normalise(values.get(field))
                if value is None:
                    continue
                stats[field]["non_null"] += 1
                stats[field]["values"][value] += 1
                if label == 1:
                    stats[field]["fraud"][value] += 1

            merchant_present = any(
                field.startswith("metadata.merchant_info.") for field in present
            )
            if merchant_present:
                merchant_by_type[transaction_type] += 1
                merchant_by_label[label] += 1

            merchant_category = normalise(
                values.get("metadata.merchant_info.category")
            )
            if merchant_category is not None:
                category_comparisons += 1
                category_matches += int(merchant_category == category)

        print(f"Processed chunk {chunk_number}: {row_count:,} cumulative rows")

    return {
        "row_count": row_count,
        "parse_failures": parse_failures,
        "stats": stats,
        "label_counts": label_counts,
        "merchant_by_type": merchant_by_type,
        "merchant_by_label": merchant_by_label,
        "category_comparisons": category_comparisons,
        "category_matches": category_matches,
    }


def cardinality(unique_values: int) -> str:
    """Classify cardinality with visible, fixed thresholds."""
    if unique_values <= 1:
        return "constant"
    if unique_values <= 20:
        return "low"
    if unique_values <= 100:
        return "moderate"
    return "high"


def group_rate_range(values: Counter, fraud: Counter) -> tuple[float, float]:
    """Measure categorical fraud-rate variation, ignoring groups below 100 rows."""
    rates = [
        fraud[value] / count
        for value, count in values.items()
        if count >= MINIMUM_GROUP_ROWS
    ]
    return (min(rates), max(rates)) if rates else (float("nan"), float("nan"))


def numeric_summary(values: Counter, fraud: Counter) -> dict[str, float]:
    """Calculate exact weighted summaries without expanding repeated values."""
    if not values:
        return {
            key: float("nan")
            for key in (
                "minimum",
                "median",
                "mean",
                "maximum",
                "nonfraud_mean",
                "fraud_mean",
            )
        }

    ordered = sorted((float(value), count) for value, count in values.items())
    total = sum(count for _, count in ordered)
    total_sum = sum(value * count for value, count in ordered)
    median_positions = ((total - 1) // 2, total // 2)
    median_values = []
    cumulative = 0
    for value, count in ordered:
        previous = cumulative
        cumulative += count
        median_values.extend(
            value
            for position in median_positions
            if previous <= position < cumulative
        )
        if len(median_values) == 2:
            break

    fraud_rows = sum(fraud.values())
    fraud_sum = sum(float(value) * count for value, count in fraud.items())
    nonfraud_rows = total - fraud_rows
    return {
        "minimum": ordered[0][0],
        "median": sum(median_values) / len(median_values),
        "mean": total_sum / total,
        "maximum": ordered[-1][0],
        "nonfraud_mean": (
            (total_sum - fraud_sum) / nonfraud_rows
            if nonfraud_rows
            else float("nan")
        ),
        "fraud_mean": (
            fraud_sum / fraud_rows
            if fraud_rows
            else float("nan")
        ),
    }


def recommendation(
    field: str,
    unique_values: int,
    merchant_category_is_duplicate: bool,
) -> tuple[str, str]:
    """Attach a conservative, explainable provisional field decision."""
    if field.startswith("metadata.device_info."):
        return (
            "include_with_decision_time_assumption",
            "Device context is plausible at authorization time; document this assumption.",
        )
    if field == "metadata.location.country" and unique_values == 1:
        return (
            "exclude_constant",
            "A constant field cannot distinguish risk.",
        )
    if field in {
        "metadata.location.city",
        "metadata.location.state",
    }:
        return (
            "include_with_decision_time_assumption",
            "Fit categorical encoding on training data only.",
        )
    if field == "metadata.location.postcode":
        return (
            "encode_or_aggregate_with_care",
            "High-cardinality location can encourage memorization.",
        )
    if field == "metadata.merchant_info.category":
        if merchant_category_is_duplicate:
            return (
                "exclude_duplicate",
                "It duplicates the top-level category.",
            )
        return (
            "needs_review",
            "It differs from the top-level category.",
        )
    if field == "metadata.merchant_info.merchant_id":
        return (
            "derive_with_history",
            "Use only for aggregates from earlier transactions, not as a raw feature.",
        )
    if field == NUMERIC_FIELD:
        return (
            "exclude_until_construction_verified",
            "The averaging window and temporal cutoff are undocumented.",
        )
    if field == "metadata.merchant_info.risk_level":
        return (
            "exclude_precomputed_risk",
            "A supplied risk class may encode generator or target information.",
        )
    return (
        "needs_review",
        "No field-specific decision has been established.",
    )


def build_profile(results: dict) -> pd.DataFrame:
    """Create one compact evidence row for each candidate field."""
    rows = []
    total_rows = results["row_count"]
    comparisons = results["category_comparisons"]
    category_is_duplicate = (
        comparisons > 0
        and results["category_matches"] == comparisons
    )

    for field, field_stats in results["stats"].items():
        values = field_stats["values"]
        fraud = field_stats["fraud"]
        non_null = field_stats["non_null"]
        unique_values = len(values)
        top_value, top_rows = (
            values.most_common(1)[0]
            if values
            else (None, 0)
        )
        min_rate, max_rate = (
            (float("nan"), float("nan"))
            if field == NUMERIC_FIELD
            else group_rate_range(values, fraud)
        )
        number_summary = (
            numeric_summary(values, fraud)
            if field == NUMERIC_FIELD
            else numeric_summary(Counter(), Counter())
        )
        decision, reason = recommendation(
            field,
            unique_values,
            category_is_duplicate,
        )
        fraud_rows = sum(fraud.values())
        small_perfect_fraud_group_count = 0
        small_perfect_fraud_rows = 0

        for value, count in values.items():
            if (
                count < MINIMUM_GROUP_ROWS
                and fraud[value] == count
            ):
                small_perfect_fraud_group_count += 1
                small_perfect_fraud_rows += count

        rows.append(
            {
                "field": field,
                "data_kind": (
                    "numeric"
                    if field == NUMERIC_FIELD
                    else "categorical"
                ),
                "structural_presence_rows": field_stats["structural"],
                "non_null_rows": non_null,
                "coverage_pct": 100 * non_null / total_rows,
                "unique_values": unique_values,
                "cardinality": cardinality(unique_values),
                "most_common_value": top_value,
                "most_common_rows": top_rows,
                "most_common_share_pct": (
                    100 * top_rows / non_null
                    if non_null
                    else None
                ),
                "fraud_rows_among_non_null": fraud_rows,
                "fraud_rate_pct": (
                    100 * fraud_rows / non_null
                    if non_null
                    else None
                ),
                "minimum_group_fraud_rate_pct": 100 * min_rate,
                "maximum_group_fraud_rate_pct": 100 * max_rate,
                "group_rate_minimum_rows": MINIMUM_GROUP_ROWS,
                "small_perfect_fraud_group_count": (
                    small_perfect_fraud_group_count
                ),
                "small_perfect_fraud_rows": (
                    small_perfect_fraud_rows
                ),
                "numeric_minimum": number_summary["minimum"],
                "numeric_median": number_summary["median"],
                "numeric_mean": number_summary["mean"],
                "numeric_maximum": number_summary["maximum"],
                "nonfraud_numeric_mean": (
                    number_summary["nonfraud_mean"]
                ),
                "fraud_numeric_mean": number_summary["fraud_mean"],
                "provisional_recommendation": decision,
                "reason": reason,
            }
        )

    return pd.DataFrame(rows)


def print_results(results: dict, profile: pd.DataFrame) -> None:
    """Print a compact interpretation and the saved-output location."""
    print("\nAMLNet Version 2.0 candidate metadata profile")
    print(f"Rows scanned: {results['row_count']:,}")
    print(f"Metadata parse failures: {results['parse_failures']:,}")

    print("\nMerchant-information presence by transaction type:")

    for transaction_type, count in sorted(
        results["merchant_by_type"].items()
    ):
        print(f"  {transaction_type}: {count:,}")

    print("\nMerchant-information presence by fraud label:")

    for label in (0, 1):
        present = results["merchant_by_label"][label]
        total = results["label_counts"][label]
        rate = present / total if total else float("nan")

        print(
            f"  {label}: "
            f"{present:,} of {total:,} "
            f"({rate:.6%})"
        )

    comparisons = results["category_comparisons"]
    matches = results["category_matches"]

    print("\nMerchant category versus top-level category:")
    print(f"  Comparable rows: {comparisons:,}")
    print(f"  Exact matches: {matches:,}")
    print(f"  Disagreements: {comparisons - matches:,}")

    columns = [
        "field",
        "coverage_pct",
        "unique_values",
        "cardinality",
        "fraud_rate_pct",
        "provisional_recommendation",
    ]

    print("\nCandidate-field summary:")
    print(profile[columns].to_string(index=False))
    print(
        f"\nSaved profile: "
        f"{OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )

    status = (
        "COMPLETED"
        if not results["parse_failures"]
        else "COMPLETED WITH PARSE FAILURES"
    )
    print(f"Profile result: {status}")


def main() -> None:
    """Run the profiler and save its field-level summary."""
    results = scan_dataset()
    profile = build_profile(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(OUTPUT_PATH, index=False)
    print_results(results, profile)


if __name__ == "__main__":
    main()
