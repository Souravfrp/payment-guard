"""Validate the structure and basic quality of AMLNet Version 2.0."""

from collections import Counter
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

CHUNK_SIZE_ROWS = 50_000
EXPECTED_ROW_COUNT = 1_090_000

EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "category",
    "nameOrig",
    "nameDest",
    "oldbalanceOrg",
    "newbalanceOrig",
    "isFraud",
    "isMoneyLaundering",
    "laundering_typology",
    "metadata",
    "fraud_probability",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
]

REQUIRED_NON_NULL_COLUMNS = [
    column
    for column in EXPECTED_COLUMNS
    if column != "fraud_probability"
]

NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "isFraud",
    "isMoneyLaundering",
    "fraud_probability",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
]

RANGE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "fraud_probability",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
]

INTEGER_COLUMNS = [
    "step",
    "isFraud",
    "isMoneyLaundering",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
]

METADATA_KEYS = [
    "timestamp",
    "location",
    "device_info",
    "payment_method",
    "merchant_info",
    "risk_indicators",
    "network_metrics",
]

TIME_LIMITS = {
    "hour": (0, 23),
    "day_of_week": (0, 6),
    "day_of_month": (1, 31),
    "month": (1, 12),
}


def update_numeric_summaries(
    chunk: pd.DataFrame,
    minimums: dict,
    maximums: dict,
    numeric_conversion_failures: Counter,
) -> dict[str, pd.Series]:
    """Convert numeric columns and update their validation summaries."""
    numeric_data = {}

    for column in NUMERIC_COLUMNS:
        original = chunk[column]
        converted = pd.to_numeric(original, errors="coerce")
        numeric_data[column] = converted

        failed_conversion = original.notna() & converted.isna()
        numeric_conversion_failures[column] += int(
            failed_conversion.sum()
        )

        valid_values = converted.dropna()

        if valid_values.empty:
            continue

        chunk_minimum = valid_values.min()
        chunk_maximum = valid_values.max()

        if (
            column not in minimums
            or chunk_minimum < minimums[column]
        ):
            minimums[column] = chunk_minimum

        if (
            column not in maximums
            or chunk_maximum > maximums[column]
        ):
            maximums[column] = chunk_maximum

    return numeric_data


def validate_dataset() -> dict:
    """Scan the complete CSV in chunks and collect validation results."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    actual_columns = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()

    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "CSV columns do not match the expected AMLNet Version 2.0 "
            "schema."
        )

    row_count = 0
    missing_counts = Counter()
    fraud_counts = Counter()
    laundering_counts = Counter()
    transaction_types = set()
    categories = set()
    minimums = {}
    maximums = {}
    numeric_conversion_failures = Counter()
    non_integer_counts = Counter()
    invalid_time_counts = Counter()
    missing_metadata_markers = Counter()
    negative_amount_count = 0
    label_disagreement_count = 0

    chunks = pd.read_csv(
        DATA_PATH,
        chunksize=CHUNK_SIZE_ROWS,
    )

    for chunk_number, chunk in enumerate(chunks, start=1):
        row_count += len(chunk)
        missing_counts.update(chunk.isna().sum().to_dict())

        numeric_data = update_numeric_summaries(
            chunk=chunk,
            minimums=minimums,
            maximums=maximums,
            numeric_conversion_failures=numeric_conversion_failures,
        )

        fraud_labels = numeric_data["isFraud"]
        laundering_labels = numeric_data["isMoneyLaundering"]

        fraud_counts.update(fraud_labels.dropna().tolist())
        laundering_counts.update(
            laundering_labels.dropna().tolist()
        )

        comparable_labels = (
            fraud_labels.notna() & laundering_labels.notna()
        )
        label_disagreement_count += int(
            (
                fraud_labels[comparable_labels]
                != laundering_labels[comparable_labels]
            ).sum()
        )

        transaction_types.update(
            chunk["type"].dropna().astype(str)
        )
        categories.update(
            chunk["category"].dropna().astype(str)
        )

        amount = numeric_data["amount"]
        negative_amount_count += int((amount < 0).sum())

        for column in INTEGER_COLUMNS:
            values = numeric_data[column]
            non_integer = values.notna() & (values % 1 != 0)
            non_integer_counts[column] += int(non_integer.sum())

        for column, (lower, upper) in TIME_LIMITS.items():
            values = numeric_data[column]
            invalid = values.notna() & ~values.between(lower, upper)
            invalid_time_counts[column] += int(invalid.sum())

        metadata = chunk["metadata"].fillna("").astype(str)

        for key in METADATA_KEYS:
            marker = f"'{key}':"
            missing_marker = ~metadata.str.contains(
                marker,
                regex=False,
            )
            missing_metadata_markers[key] += int(
                missing_marker.sum()
            )

        print(
            f"Processed chunk {chunk_number}: "
            f"{row_count:,} cumulative rows"
        )

    return {
        "row_count": row_count,
        "missing_counts": missing_counts,
        "fraud_counts": fraud_counts,
        "laundering_counts": laundering_counts,
        "transaction_types": transaction_types,
        "categories": categories,
        "minimums": minimums,
        "maximums": maximums,
        "numeric_conversion_failures": (
            numeric_conversion_failures
        ),
        "non_integer_counts": non_integer_counts,
        "invalid_time_counts": invalid_time_counts,
        "missing_metadata_markers": (
            missing_metadata_markers
        ),
        "negative_amount_count": negative_amount_count,
        "label_disagreement_count": label_disagreement_count,
    }


def collect_failures(results: dict) -> list[str]:
    """Convert invalid validation results into clear failure messages."""
    failures = []

    if results["row_count"] != EXPECTED_ROW_COUNT:
        failures.append(
            f"Expected {EXPECTED_ROW_COUNT:,} rows but found "
            f"{results['row_count']:,}."
        )

    fraud_values = set(results["fraud_counts"])
    laundering_values = set(results["laundering_counts"])

    if not fraud_values.issubset({0, 1}):
        failures.append("Unexpected values exist in isFraud.")

    if not laundering_values.issubset({0, 1}):
        failures.append(
            "Unexpected values exist in isMoneyLaundering."
        )

    for column in REQUIRED_NON_NULL_COLUMNS:
        count = results["missing_counts"][column]

        if count:
            failures.append(
                f"{count:,} missing values exist in required column "
                f"'{column}'."
            )

    for column in NUMERIC_COLUMNS:
        count = results["numeric_conversion_failures"][column]

        if count:
            failures.append(
                f"{count:,} non-numeric values exist in numeric column "
                f"'{column}'."
            )

    for column in INTEGER_COLUMNS:
        count = results["non_integer_counts"][column]

        if count:
            failures.append(
                f"{count:,} non-integer values exist in integer column "
                f"'{column}'."
            )

    if results["negative_amount_count"]:
        failures.append(
            f"{results['negative_amount_count']:,} negative amounts "
            "were found."
        )

    for column, count in results["invalid_time_counts"].items():
        if count:
            failures.append(
                f"{count:,} values lie outside the valid "
                f"{column} range."
            )

    for key, count in results["missing_metadata_markers"].items():
        if count:
            failures.append(
                f"{count:,} rows lack metadata marker '{key}'."
            )

    return failures


def collect_warnings(results: dict) -> list[str]:
    """Record limitations that do not invalidate the raw file."""
    warnings = []

    if results["row_count"] > 0 and (
        results["minimums"].get("step")
        == results["maximums"].get("step")
    ):
        warnings.append(
            "The step column is constant and cannot be used as a "
            "transaction identifier or time variable."
        )

    fraud_label_count = sum(results["fraud_counts"].values())
    laundering_label_count = sum(
        results["laundering_counts"].values()
    )

    labels_are_complete = (
        fraud_label_count == results["row_count"]
        and laundering_label_count == results["row_count"]
    )

    if (
        results["row_count"] > 0
        and labels_are_complete
        and results["label_disagreement_count"] == 0
    ):
        warnings.append(
            "isFraud and isMoneyLaundering are identical in every row."
        )

    missing_probability_count = results["missing_counts"][
        "fraud_probability"
    ]

    warnings.append(
        "fraud_probability is a supplied risk score with "
        f"{missing_probability_count:,} missing values. It must remain "
        "excluded from predictive features unless its construction and "
        "decision-time availability are established."
    )

    return warnings


def print_results(results: dict) -> bool:
    """Print results and return whether all required checks passed."""
    print("\nAMLNet Version 2.0 validation")
    print(f"Rows: {results['row_count']:,}")
    print(f"Expected rows: {EXPECTED_ROW_COUNT:,}")
    print(f"Columns: {len(EXPECTED_COLUMNS)}")

    print("\nMissing values:")
    missing_found = False

    for column in EXPECTED_COLUMNS:
        count = results["missing_counts"][column]

        if count:
            print(f"  {column}: {count:,}")
            missing_found = True

    if not missing_found:
        print("  None")

    print("\nNumeric conversion failures:")
    conversion_failure_found = False

    for column in NUMERIC_COLUMNS:
        count = results["numeric_conversion_failures"][column]

        if count:
            print(f"  {column}: {count:,}")
            conversion_failure_found = True

    if not conversion_failure_found:
        print("  None")

    print("\nFraud-label counts:")
    for value, count in sorted(results["fraud_counts"].items()):
        print(f"  {value}: {count:,}")

    if results["row_count"]:
        positive_fraud_count = results["fraud_counts"].get(1, 0)
        fraud_rate = positive_fraud_count / results["row_count"]
        print(f"Fraud rate: {fraud_rate:.6%}")
    else:
        print("Fraud rate: undefined because the dataset has no rows")

    print("\nMoney-laundering-label counts:")
    for value, count in sorted(
        results["laundering_counts"].items()
    ):
        print(f"  {value}: {count:,}")

    print(
        "Label disagreements: "
        f"{results['label_disagreement_count']:,}"
    )

    print("\nTransaction types:")
    print("  " + ", ".join(sorted(results["transaction_types"])))

    print("\nCategories:")
    print("  " + ", ".join(sorted(results["categories"])))

    print("\nNumeric ranges:")
    for column in RANGE_COLUMNS:
        print(
            f"  {column}: "
            f"{results['minimums'].get(column)} to "
            f"{results['maximums'].get(column)}"
        )

    print("\nMissing metadata key markers:")
    for key in METADATA_KEYS:
        count = results["missing_metadata_markers"][key]
        print(f"  {key}: {count:,}")

    failures = collect_failures(results)
    warnings = collect_warnings(results)

    if failures:
        print("\nValidation result: FAILED")

        for failure in failures:
            print(f"  - {failure}")

        if warnings:
            print("\nAdditional warnings:")

            for warning in warnings:
                print(f"  - {warning}")

        return False

    if warnings:
        print("\nValidation result: PASSED WITH WARNINGS")

        for warning in warnings:
            print(f"  - {warning}")

        return True

    print("\nValidation result: PASSED")
    return True


def main() -> None:
    """Run the complete validation and expose failure to automation."""
    results = validate_dataset()
    validation_passed = print_results(results)

    if not validation_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
