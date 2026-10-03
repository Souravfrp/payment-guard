"""Checks for permitted metadata extraction and invalid inputs."""

from datetime import datetime

import pytest

from payment_guard.prepare_amlnet_v2_data import (
    extract_metadata,
)


VALID_METADATA = (
    "{"
    "'payment_method': 'card', "
    "'device_info': {'os': 'Linux', 'type': 'mobile'}, "
    "'location': {'city': 'Mumbai', 'state': 'Maharashtra'}, "
    "'timestamp': datetime.datetime(2025, 10, 13, 17, 8, 30, 636449)"
    "}"
)


def test_extracts_only_permitted_fields():
    result = extract_metadata(VALID_METADATA)

    assert result == {
        "metadata.payment_method": "card",
        "metadata.device_info.os": "Linux",
        "metadata.device_info.type": "mobile",
        "metadata.location.city": "Mumbai",
        "metadata.location.state": "Maharashtra",
        "timestamp": datetime(
            2025, 10, 13, 17, 8, 30, 636449
        ),
    }


def test_accepts_timestamp_without_microseconds():
    text = VALID_METADATA.replace(
        ", 636449)",
        ")",
    )

    result = extract_metadata(text)

    assert result["timestamp"].microsecond == 0


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        VALID_METADATA.replace(
            "'payment_method': 'card', ",
            "",
        ),
        VALID_METADATA.replace(
            "'payment_method': 'card'",
            "'payment_method': 'card', 'payment_method': 'cash'",
        ),
        VALID_METADATA.replace(
            "'os': 'Linux'",
            "'os': 'Linux', 'os': 'Other'",
        ),
        VALID_METADATA.replace(
            "'payment_method': 'card'",
            "'payment_method': '   '",
        ),
        VALID_METADATA.replace(
            "'payment_method': 'card'",
            "'payment_method': 123",
        ),
        VALID_METADATA.replace(
            "2025, 10, 13",
            "2025, 2, 30",
        ),
        VALID_METADATA.replace(
            "2025, 10, 13",
            "True, 10, 13",
        ),
        VALID_METADATA.replace(
            "datetime.datetime(",
            "other.datetime(",
        ),
    ],
)
def test_rejects_invalid_required_metadata(text):
    with pytest.raises(ValueError):
        extract_metadata(text)


def test_does_not_execute_a_required_field_expression(tmp_path):
    marker = tmp_path / "unexpected_execution.txt"

    expression = (
        "__import__('pathlib').Path("
        f"{str(marker)!r}"
        ").write_text('executed')"
    )

    text = VALID_METADATA.replace(
        "'payment_method': 'card'",
        f"'payment_method': {expression}",
    )

    with pytest.raises(ValueError):
        extract_metadata(text)

    assert not marker.exists()


def make_raw_chunk():
    import pandas as pd

    return pd.DataFrame({
        "amount": [100.0, 200.0],
        "oldbalanceOrg": [1000.0, 1500.0],
        "hour": [17, 17],
        "day_of_week": [0, 0],
        "day_of_month": [13, 13],
        "month": [10, 10],
        "type": ["TRANSFER", "TRANSFER"],
        "category": ["example", "example"],
        "isFraud": [0, 1],
        "metadata": [VALID_METADATA, VALID_METADATA],
    })


def test_chunk_preserves_labels_and_source_references():
    from payment_guard.prepare_amlnet_v2_data import prepare_chunk

    raw = make_raw_chunk()
    raw.index = [50, 60]

    result = prepare_chunk(raw, source_offset=100)

    assert result["source_row"].tolist() == [101, 102]
    assert result["isFraud"].tolist() == [0, 1]
    assert len(result) == 2
    assert "metadata" not in result.columns
    assert "fraud_probability" not in result.columns


@pytest.mark.parametrize(
    "field, value",
    [
        ("amount", -1.0),
        ("amount", float("inf")),
        ("oldbalanceOrg", float("nan")),
        ("isFraud", 2),
        ("isFraud", 0.5),
        ("hour", 16),
        ("hour", 17.5),
        ("day_of_week", 1),
        ("day_of_month", 14),
        ("month", 11),
        ("type", ""),
        ("category", "   "),
    ],
)
def test_chunk_rejects_invalid_values(field, value):
    from payment_guard.prepare_amlnet_v2_data import prepare_chunk

    raw = make_raw_chunk()
    raw[field] = raw[field].astype(object)
    raw.loc[0, field] = value

    with pytest.raises(ValueError):
        prepare_chunk(raw)


def test_chunk_reports_global_source_row_on_parse_failure():
    from payment_guard.prepare_amlnet_v2_data import prepare_chunk

    raw = make_raw_chunk()
    raw.loc[1, "metadata"] = "[]"

    with pytest.raises(ValueError, match="source row 102"):
        prepare_chunk(raw, source_offset=100)


def test_chunk_rejects_empty_input():
    from payment_guard.prepare_amlnet_v2_data import prepare_chunk

    raw = make_raw_chunk().iloc[:0]

    with pytest.raises(ValueError, match="empty"):
        prepare_chunk(raw)


def test_sorting_boundaries_and_equal_timestamps():
    import pandas as pd

    from payment_guard.prepare_amlnet_v2_data import (
        prepare_chunk,
        sort_and_split,
    )

    base = prepare_chunk(make_raw_chunk()).iloc[[0]]

    example = pd.concat(
        [base.copy() for _ in range(4)],
        ignore_index=True,
    )

    example["source_row"] = [4, 3, 2, 1]
    example["timestamp"] = pd.to_datetime([
        "2026-03-01",
        "2026-02-01",
        "2026-02-01",
        "2026-01-31",
    ])

    result = sort_and_split(example)

    assert result["source_row"].tolist() == [1, 2, 3, 4]
    assert result["split"].tolist() == [
        "train",
        "validation",
        "validation",
        "test",
    ]

    assert result.groupby("timestamp")["split"].nunique().max() == 1


def test_sorting_rejects_duplicate_source_references():
    from payment_guard.prepare_amlnet_v2_data import (
        prepare_chunk,
        sort_and_split,
    )

    prepared = prepare_chunk(make_raw_chunk())
    prepared["source_row"] = [1, 1]

    with pytest.raises(ValueError, match="unique"):
        sort_and_split(prepared)


def test_sorting_rejects_unexpected_predictor():
    from payment_guard.prepare_amlnet_v2_data import (
        prepare_chunk,
        sort_and_split,
    )

    prepared = prepare_chunk(make_raw_chunk())
    prepared["fraud_probability"] = [0.1, 0.9]

    with pytest.raises(ValueError, match="columns"):
        sort_and_split(prepared)


def make_test_policy():
    import pandas as pd

    from payment_guard.prepare_amlnet_v2_data import (
        METADATA_FIELDS,
        TOP_LEVEL_FEATURES,
    )

    fields = TOP_LEVEL_FEATURES + list(METADATA_FIELDS)

    return pd.DataFrame({
        "field": fields + ["isFraud", "metadata.timestamp"],
        "resolved_model_status": (
            ["include"] * len(fields)
            + ["target", "split_only"]
        ),
    })


def test_policy_rejects_a_revoked_predictor(tmp_path):
    from payment_guard.prepare_amlnet_v2_data import (
        validate_feature_policy,
    )

    policy = make_test_policy()
    policy.loc[
        policy["field"] == "amount",
        "resolved_model_status",
    ] = "exclude"

    path = tmp_path / "policy.csv"
    policy.to_csv(path, index=False)

    with pytest.raises(ValueError, match="mismatch"):
        validate_feature_policy(path)


def test_policy_rejects_target_as_predictor(tmp_path):
    from payment_guard.prepare_amlnet_v2_data import (
        validate_feature_policy,
    )

    policy = make_test_policy()
    policy.loc[
        policy["field"] == "isFraud",
        "resolved_model_status",
    ] = "include"

    path = tmp_path / "policy.csv"
    policy.to_csv(path, index=False)

    with pytest.raises(ValueError, match="mismatch"):
        validate_feature_policy(path)
