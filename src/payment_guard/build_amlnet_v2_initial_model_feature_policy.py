"""Build the initial AMLNet Version 2.0 model feature policy."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIT_PATH = (
    PROJECT_ROOT
    / "results/tables/amlnet_v2_leakage_audit.csv"
)

CANDIDATE_PATH = (
    PROJECT_ROOT
    / "results/tables/"
    "amlnet_v2_candidate_metadata_profile.csv"
)

TEMPORAL_PATH = (
    PROJECT_ROOT
    / "results/tables/"
    "amlnet_v2_temporal_risk_profile.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results/tables/"
    "amlnet_v2_initial_model_feature_policy.csv"
)


CANDIDATE_RESOLUTIONS = {
    "metadata.device_info.os": (
        "assumed_yes",
        "include_with_assumption",
        (
            "Treat device operating system as available "
            "before authorization, but document this "
            "dataset assumption."
        ),
        "Fit categorical encoding on training data only.",
    ),
    "metadata.device_info.type": (
        "assumed_yes",
        "include_with_assumption",
        (
            "Treat device type as available before "
            "authorization, but document this dataset "
            "assumption."
        ),
        "Fit categorical encoding on training data only.",
    ),
    "metadata.location.city": (
        "assumed_yes",
        "include_with_assumption",
        (
            "Treat city as captured before authorization, "
            "subject to a documented availability assumption."
        ),
        (
            "Fit categorical encoding on training data only "
            "and monitor temporal stability."
        ),
    ),
    "metadata.location.country": (
        "assumed_yes",
        "exclude",
        (
            "Exclude country because it is constant in the "
            "released dataset and cannot distinguish risk."
        ),
        "None for the initial model.",
    ),
    "metadata.location.postcode": (
        "assumed_yes",
        "exclude_initial",
        (
            "Exclude raw postcode from the initial baseline "
            "because its high cardinality can encourage "
            "memorization."
        ),
        (
            "Later compare a training-only frequency encoding "
            "or historical aggregation."
        ),
    ),
    "metadata.location.state": (
        "assumed_yes",
        "include_with_assumption",
        (
            "Treat state as captured before authorization, "
            "subject to a documented availability assumption."
        ),
        (
            "Fit categorical encoding on training data only "
            "and monitor temporal stability."
        ),
    ),
    "metadata.merchant_info.avg_transaction": (
        "uncertain",
        "exclude",
        (
            "Exclude the supplied merchant average because "
            "its averaging window is undocumented and rare "
            "values perfectly separate 58 fraud-labelled rows."
        ),
        (
            "Reconsider only if a past-only construction "
            "can be reproduced."
        ),
    ),
    "metadata.merchant_info.category": (
        "uncertain",
        "exclude",
        (
            "Exclude the merchant category because it "
            "duplicates the top-level category and rare "
            "values perfectly separate 58 fraud-labelled rows."
        ),
        (
            "Use the top-level category only, with an "
            "ablation comparison."
        ),
    ),
    "metadata.merchant_info.merchant_id": (
        "uncertain",
        "derive_with_history",
        (
            "Do not use the raw merchant identifier; "
            "it can encourage memorization."
        ),
        (
            "Use only aggregates calculated from transactions "
            "earlier than each decision."
        ),
    ),
    "metadata.merchant_info.risk_level": (
        "uncertain",
        "exclude",
        (
            "Exclude the supplied risk class because it is "
            "precomputed and rare values perfectly separate "
            "58 fraud-labelled rows."
        ),
        "None for the initial model.",
    ),
}


def require_inputs() -> None:
    """Stop clearly if a required evidence table is absent."""
    for path in (
        AUDIT_PATH,
        CANDIDATE_PATH,
        TEMPORAL_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )


def aggregate_temporal(
    temporal: pd.DataFrame,
    field: str,
    value: str,
) -> tuple[int, int, float]:
    """Aggregate one selected group over all observed months."""
    selected = temporal[
        (temporal["field"] == field)
        & (temporal["value"] == value)
    ]

    rows = int(
        selected["transaction_rows"].sum()
    )

    fraud = int(
        selected["fraud_rows"].sum()
    )

    rate = (
        100 * fraud / rows
        if rows
        else float("nan")
    )

    return rows, fraud, rate


def temporal_evidence(
    temporal: pd.DataFrame,
    field: str,
    values: tuple[str, ...],
) -> str:
    """Describe support and fraud rates for selected groups."""
    parts = []

    for value in values:
        rows, fraud, rate = aggregate_temporal(
            temporal,
            field,
            value,
        )

        parts.append(
            f"{value}: "
            f"{fraud:,}/{rows:,} fraud-labelled "
            f"({rate:.6f}%)"
        )

    return "; ".join(parts) + "."


def candidate_evidence(
    row: pd.Series,
) -> str:
    """Summarize saved evidence for one candidate field."""
    evidence = [
        f"coverage={row['coverage_pct']:.6f}%",
        f"unique_values={int(row['unique_values']):,}",
        f"fraud_rate={row['fraud_rate_pct']:.6f}%",
        (
            "small_perfect_fraud_groups="
            f"{int(row['small_perfect_fraud_group_count']):,}"
        ),
        (
            "small_perfect_fraud_rows="
            f"{int(row['small_perfect_fraud_rows']):,}"
        ),
        (
            "profile_recommendation="
            f"{row['provisional_recommendation']}"
        ),
    ]

    return "; ".join(evidence) + "."


def default_required_check(
    status: str,
) -> str:
    """Attach a safeguard to unchanged audit decisions."""
    if status == "include":
        return (
            "Fit transformations on training data only."
        )

    if status == "derive_with_history":
        return (
            "Use information strictly earlier than "
            "each decision."
        )

    if status == "split_only":
        return (
            "Use for chronological ordering and "
            "time-derived features only."
        )

    return "None for the initial model."


def build_policy(
    audit: pd.DataFrame,
    candidates: pd.DataFrame,
    temporal: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve audited fields using later profiling evidence."""
    if audit["field"].duplicated().any():
        raise ValueError(
            "Duplicate fields exist in the leakage audit."
        )

    if candidates["field"].duplicated().any():
        raise ValueError(
            "Duplicate fields exist in the candidate profile."
        )

    candidate_fields = set(
        candidates["field"]
    )

    expected_candidates = set(
        CANDIDATE_RESOLUTIONS
    )

    if candidate_fields != expected_candidates:
        raise ValueError(
            "Candidate-profile fields do not match "
            "the policy map."
        )

    candidate_lookup = candidates.set_index(
        "field"
    )

    rows = []

    for audit_row in audit.itertuples(
        index=False
    ):
        field = audit_row.field
        decision_status = (
            audit_row.decision_time_status
        )
        resolved_status = (
            audit_row.initial_model_status
        )
        resolved_reason = audit_row.reason
        evidence = audit_row.audit_evidence
        required_check = default_required_check(
            resolved_status
        )

        if field == "type":
            resolved_status = (
                "include_with_ablation"
            )
            resolved_reason = (
                "Retain transaction type as decision-time "
                "information, but measure how much performance "
                "depends on synthetic group structure."
            )
            evidence = temporal_evidence(
                temporal,
                "type",
                (
                    "PAYMENT",
                    "TRANSFER",
                ),
            )
            required_check = (
                "Compare identical chronological baselines "
                "with and without type."
            )

        elif field == "category":
            resolved_status = (
                "include_with_ablation"
            )
            resolved_reason = (
                "Retain the top-level category as decision-time "
                "information, while testing sensitivity to "
                "rare deterministic categories."
            )
            evidence = temporal_evidence(
                temporal,
                "category",
                (
                    "Property Investment",
                    "Cryptocurrency",
                    "Shell Company",
                    "Other",
                    "Recreation",
                ),
            )
            required_check = (
                "Fit encoding on training data only and "
                "compare the same model with category removed."
            )

        elif field in CANDIDATE_RESOLUTIONS:
            (
                decision_status,
                resolved_status,
                resolved_reason,
                required_check,
            ) = CANDIDATE_RESOLUTIONS[field]

            evidence = candidate_evidence(
                candidate_lookup.loc[field]
            )

        rows.append(
            {
                "field": field,
                "location": audit_row.location,
                "structural_presence_rows": (
                    audit_row.structural_presence_rows
                ),
                "project_role": audit_row.project_role,
                "audit_decision_time_status": (
                    audit_row.decision_time_status
                ),
                "audit_model_status": (
                    audit_row.initial_model_status
                ),
                "resolved_decision_time_status": (
                    decision_status
                ),
                "resolved_model_status": (
                    resolved_status
                ),
                "resolved_reason": resolved_reason,
                "supporting_evidence": evidence,
                "required_model_check": required_check,
            }
        )

    policy = pd.DataFrame(
        rows
    )

    if len(policy) != len(audit):
        raise ValueError(
            "Policy row count does not match the audit."
        )

    if policy["field"].duplicated().any():
        raise ValueError(
            "Duplicate fields exist in the resolved policy."
        )

    unresolved = policy[
        "resolved_model_status"
    ].eq(
        "needs_review"
    ).sum()

    if unresolved:
        raise ValueError(
            f"Unresolved policy rows remain: {unresolved}"
        )

    return policy


def main() -> None:
    """Build, validate, save and summarize the policy."""
    require_inputs()

    audit = pd.read_csv(
        AUDIT_PATH
    )

    candidates = pd.read_csv(
        CANDIDATE_PATH
    )

    temporal = pd.read_csv(
        TEMPORAL_PATH
    )

    policy = build_policy(
        audit,
        candidates,
        temporal,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "AMLNet Version 2.0 initial model feature policy"
    )
    print(
        f"Fields resolved: {len(policy):,}"
    )
    print(
        "Resolved status counts:"
    )

    for status, count in (
        policy["resolved_model_status"]
        .value_counts()
        .items()
    ):
        print(
            f"  {status}: {count:,}"
        )

    print(
        "Saved policy: "
        f"{OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Policy result: COMPLETED"
    )


if __name__ == "__main__":
    main()
