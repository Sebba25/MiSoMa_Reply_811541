"""Builds the remediation plan from the validation bundle.

Walks the schema, completeness, consistency, anomaly, cross-column and
duplicate reports and emits a flat ``list[RemediationAction]`` covering:
rename, placeholder-null replacement, dtype cast, cleaner generation, and
duplicate-column drop. Each action has an auto-apply policy bit that the
``application`` stage uses to decide whether to run it unattended.
"""

from __future__ import annotations

from pathlib import Path

from cache import load_remediation_plan, load_validation_results, save_remediation_plan
from models import OrchestrationStepResult, RemediationAction, RemediationPlan, SchemaColumnEntry
from validation import build_validation_results
from tools import normalized_schema_name


def _action_id(prefix: str, *parts: object) -> str:
    normalized_parts = [normalized_schema_name(str(part)) for part in parts if str(part).strip()]
    suffix = "__".join(part for part in normalized_parts if part)
    return f"{prefix}__{suffix}" if suffix else prefix


def _schema_column_map(validation_results: OrchestrationStepResult) -> dict[str, SchemaColumnEntry]:
    return {column.name: column for column in validation_results.schema_validation.columns}


def _target_canonical_name(column: SchemaColumnEntry) -> str:
    return column.rename_suggestion or column.name


def _planned_rename_map(columns: list[SchemaColumnEntry]) -> dict[str, str]:
    existing = set(column.name for column in columns)
    rename_map: dict[str, str] = {}
    for column in columns:
        if column.naming_valid or not column.rename_suggestion:
            continue

        target = column.rename_suggestion
        if target in existing and target != column.name:
            suffix = 2
            while f"{target}_{suffix}" in existing:
                suffix += 1
            target = f"{target}_{suffix}"
        rename_map[column.name] = target
        existing.add(target)
    return rename_map


def _column_keep_sort_key(column: SchemaColumnEntry) -> tuple[int, int, int, str]:
    target_name = _target_canonical_name(column)
    return (
        0 if column.naming_valid else 1,
        -column.non_null_rows,
        0 if normalized_schema_name(target_name) == target_name else 1,
        column.name,
    )


def _choose_keep_drop_columns(
    left_name: str,
    right_name: str,
    schema_map: dict[str, SchemaColumnEntry],
) -> tuple[str, str]:
    left = schema_map[left_name]
    right = schema_map[right_name]
    keep = sorted([left, right], key=_column_keep_sort_key)[0]
    drop = right if keep.name == left.name else left
    return keep.name, drop.name


def _build_summary(actions: list[RemediationAction]) -> str:
    auto_apply = sum(1 for action in actions if action.auto_apply)
    manual = sum(1 for action in actions if not action.auto_apply)
    return (
        f"Planned {len(actions)} remediation actions: {auto_apply} auto-apply and {manual} review/report actions."
    )


def build_remediation_plan(validation_results: OrchestrationStepResult) -> RemediationPlan:
    schema_map = _schema_column_map(validation_results)
    rename_map = _planned_rename_map(validation_results.schema_validation.columns)
    actions: list[RemediationAction] = []

    for column in validation_results.schema_validation.columns:
        if not column.naming_valid and column.rename_suggestion:
            actions.append(
                RemediationAction(
                    action_id=_action_id("rename_column", column.name, rename_map.get(column.name, column.rename_suggestion)),
                    action_type="rename_column",
                    object_type="column",
                    target={"column_name": column.name, "new_name": rename_map.get(column.name, column.rename_suggestion)},
                    source_check="schema_validation",
                    confidence="high",
                    risk_level="low",
                    auto_apply=True,
                    status="planned",
                    reason=column.naming_reason or "Column name violates the naming policy.",
                    preview_stats={"non_null_rows": column.non_null_rows},
                )
            )

        if column.pandas_dtype in {"datetime64[ns]", "Int64", "Float64", "boolean", "string"}:
            final_column_name = rename_map.get(column.name, column.name)
            actions.append(
                RemediationAction(
                    action_id=_action_id("cast_dtype", final_column_name, column.pandas_dtype),
                    action_type="cast_dtype",
                    object_type="column",
                    target={"column_name": final_column_name, "target_dtype": column.pandas_dtype},
                    source_check="schema_validation",
                    confidence="high",
                    risk_level="low",
                    auto_apply=True,
                    status="planned",
                    reason=f"Cast the column to inferred dtype {column.pandas_dtype}.",
                    preview_stats={"non_null_rows": column.non_null_rows},
                )
            )

    for column_finding in validation_results.completeness_analysis.per_column:
        if column_finding.missing_like_count <= 0 or not column_finding.missing_like_examples:
            continue
        actions.append(
            RemediationAction(
                action_id=_action_id("replace_placeholders_with_null", column_finding.column_name),
                action_type="replace_placeholders_with_null",
                object_type="column",
                target={"column_name": column_finding.column_name},
                source_check="completeness_analysis",
                confidence="high",
                risk_level="low",
                auto_apply=True,
                status="planned",
                reason="Replace detected placeholder tokens with null values.",
                preview_stats={
                    "missing_like_count": column_finding.missing_like_count,
                    "missing_like_examples": column_finding.missing_like_examples[:8],
                },
            )
        )

    for finding in validation_results.consistency_validation.format_consistency_findings:
        actions.append(
            RemediationAction(
                action_id=_action_id("generate_cleaner", finding.column_name),
                action_type="generate_cleaner",
                object_type="column",
                target={"column_name": finding.column_name, "expected_pattern": finding.expected_pattern},
                source_check="consistency_validation",
                confidence="high",
                risk_level="medium",
                auto_apply=True,
                status="planned",
                reason=f"Generate and apply a cleaner for column {finding.column_name!r} targeting pattern {finding.expected_pattern!r}.",
                preview_stats={
                    "inconsistent_rows": finding.inconsistent_rows,
                    "example_inconsistent_values": finding.example_inconsistent_values[:8],
                },
            )
        )

    if validation_results.cross_column_validation is not None:
        for finding in validation_results.cross_column_validation.findings:
            if finding.check_type == "exact_duplicate_columns" and len(finding.columns) == 2:
                left_name, right_name = finding.columns
                if left_name not in schema_map or right_name not in schema_map:
                    continue
                keep_name, drop_name = _choose_keep_drop_columns(left_name, right_name, schema_map)
                actions.append(
                    RemediationAction(
                        action_id=_action_id("drop_exact_duplicate_column", drop_name, keep_name),
                        action_type="drop_exact_duplicate_column",
                        object_type="column_pair",
                        target={"keep_column": keep_name, "drop_column": drop_name},
                        source_check="cross_column_validation",
                        confidence="high",
                        risk_level="low",
                        auto_apply=True,
                        status="planned",
                        reason=(
                            f"Drop exact duplicate column {drop_name!r} and keep {keep_name!r} based on naming validity, "
                            "non-null coverage, and canonical naming preference."
                        ),
                        preview_stats={
                            "affected_rows": finding.affected_rows,
                            "similarity_pct": finding.similarity_pct,
                        },
                    )
                )
                continue

            if finding.check_type in {"near_duplicate_columns", "duplicate_semantic_conflict", "year_month_period_mismatch", "date_order_violation"}:
                actions.append(
                    RemediationAction(
                        action_id=_action_id(finding.check_type, *finding.columns),
                        action_type="manual_review",
                        object_type="column_pair" if len(finding.columns) >= 2 else "column",
                        target={"columns": finding.columns},
                        source_check="cross_column_validation",
                        confidence="medium",
                        risk_level="medium" if finding.check_type == "near_duplicate_columns" else "high",
                        auto_apply=False,
                        status="proposed_not_applied",
                        reason=finding.evidence,
                        preview_stats={
                            "affected_rows": finding.affected_rows,
                            "similarity_pct": finding.similarity_pct,
                            "example_row_indices": finding.example_row_indices[:8],
                        },
                    )
                )

    if validation_results.anomaly_detection is not None:
        for finding in validation_results.anomaly_detection.findings:
            action_type = "manual_review" if finding.severity == "high" else "report_only"
            actions.append(
                RemediationAction(
                    action_id=_action_id(finding.anomaly_type, finding.column_name),
                    action_type=action_type,
                    object_type="column",
                    target={"column_name": finding.column_name},
                    source_check="anomaly_detection",
                    confidence="medium" if action_type == "manual_review" else "low",
                    risk_level=finding.severity,
                    auto_apply=False,
                    status="proposed_not_applied",
                    reason=finding.evidence,
                    preview_stats={
                        "affected_rows": finding.affected_rows,
                        "example_values": finding.example_values[:8],
                    },
                )
            )

    if validation_results.duplicate_detection is not None:
        for index, group in enumerate(validation_results.duplicate_detection.groups, start=1):
            if group.duplicate_type == "exact_row":
                action_type = "drop_rows_candidate"
                confidence = "medium"
                risk_level = "medium"
            else:
                action_type = "manual_review"
                confidence = "low"
                risk_level = "high"
            actions.append(
                RemediationAction(
                    action_id=_action_id(group.duplicate_type, index),
                    action_type=action_type,
                    object_type="row_group",
                    target={"row_indices": group.row_indices[:20], "key_columns": group.key_columns},
                    source_check="duplicate_detection",
                    confidence=confidence,
                    risk_level=risk_level,
                    auto_apply=False,
                    status="proposed_not_applied",
                    reason=group.evidence,
                    preview_stats={
                        "group_size": len(group.row_indices),
                        "key_columns": group.key_columns,
                    },
                )
            )

    actions.sort(key=lambda action: (0 if action.auto_apply else 1, action.action_type, action.action_id))
    return RemediationPlan(
        dataset_name=validation_results.schema_validation.dataset_name,
        actions=actions,
        summary=_build_summary(actions),
    )


def _resolve_validation_results(
    path: Path,
    validation_results: OrchestrationStepResult | None,
    reuse_saved_validation: bool,
) -> OrchestrationStepResult:
    if validation_results is not None:
        return validation_results
    if reuse_saved_validation:
        try:
            return load_validation_results(path)
        except FileNotFoundError:
            pass
    return build_validation_results(path)


def run_remediation_planning(
    path: Path,
    validation_results: OrchestrationStepResult | None = None,
    reuse_saved_validation: bool = False,
    reuse_saved_remediation: bool = False,
) -> RemediationPlan:
    if reuse_saved_remediation:
        try:
            return load_remediation_plan(path)
        except FileNotFoundError:
            pass

    resolved_validation = _resolve_validation_results(path, validation_results, reuse_saved_validation)
    plan = build_remediation_plan(resolved_validation)
    save_remediation_plan(path, plan)
    return plan
