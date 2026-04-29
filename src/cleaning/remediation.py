"""Builds the remediation plan from the validation bundle.

Walks the schema, completeness, consistency, anomaly, cross-column and
duplicate reports and emits a flat ``list[RemediationAction]`` covering:
rename, placeholder-null replacement, dtype cast, cleaner generation, and
duplicate-column drop. Each action has an auto-apply policy bit that the
``application`` stage uses to decide whether to run it unattended.
"""

from __future__ import annotations

from pathlib import Path

from src.core.cache import load_remediation_plan, load_validation_results, save_remediation_plan
from src.core.models import OrchestrationStepResult, RemediationAction, RemediationPlan, SchemaColumnEntry
from src.validation import build_validation_results
from src.tools import normalized_schema_name

#It generates a stable action ID for each remediation action
def _action_id(prefix: str, *parts: object) -> str:
    '''Generates a stable action ID by normalizing and concatenating the prefix and parts.'''
    #normalizes each part to ensure consistent formatting and joins them with double underscores
    normalized_parts = [normalized_schema_name(str(part)) for part in parts if str(part).strip()]
    suffix = "__".join(part for part in normalized_parts if part)
    #returns something like rename_column__old_name__new_name, or just the prefix if no suffix exists.
    return f"{prefix}__{suffix}" if suffix else prefix

def _schema_column_map(validation_results: OrchestrationStepResult) -> dict[str, SchemaColumnEntry]:
    '''It converts the schema column list into a dictionary like: {column_name: column_object} for fast lookup'''
    return {column.name: column for column in validation_results.schema_validation.columns}

def _target_canonical_name(column: SchemaColumnEntry) -> str:
    '''Determines the canonical target name for a column, preferring the rename suggestion if it exists.'''
    return column.rename_suggestion or column.name
#This is used to consistently refer to the intended final name of the column throughout the remediation planning process

#It determines the planned renames for columns that violate naming policies 
def _planned_rename_map(columns: list[SchemaColumnEntry]) -> dict[str, str]:
    '''Determines the planned rename mapping for columns with naming violations, ensuring no conflicts with existing column names.'''
    #collects existing column names to ensure that any rename suggestions do not create conflicts with existing names
    existing = set(column.name for column in columns)
    #creates an empty dictionary for planned renames
    rename_map: dict[str, str] = {}
    #loops through each schema column, skipping columns that are already valid or have no rename suggestion
    for column in columns:
        if column.naming_valid or not column.rename_suggestion:
            continue
        #starts with the rename suggestion as the target name, but checks for conflicts with existing names
        target = column.rename_suggestion
        #if target name already exists and belongs to a different column add a numeric suffix to make it unique
        if target in existing and target != column.name:
            suffix = 2
            while f"{target}_{suffix}" in existing:
                suffix += 1
            #stores the mapping from old column name to the unique target name in the rename_map 
            target = f"{target}_{suffix}"
        rename_map[column.name] = target
        #and adds the new target name to the existing set to prevent future conflicts
        existing.add(target)
    #returns the final rename map
    return rename_map


def _column_keep_sort_key(column: SchemaColumnEntry) -> tuple[int, int, int, str]:
    '''Determines the sort key for deciding which column to keep when dropping exact duplicates, based on naming validity, non-null coverage, and canonical naming preference'''
    #gets the column's canonical target name
    target_name = _target_canonical_name(column)
    #returns a tuple used for sorting: prefer naming-valid columns, prefer more non-null rows, prefer columns whose canonical name matches the normalized target name, 
    # and then fall back to alphabetical column name.
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
    '''Given two column names that are exact duplicates, determines which one to keep and which one to drop based on their schema properties'''
    #gets the two column objects from the schema map
    left = schema_map[left_name]
    right = schema_map[right_name]
    #sorts the two columns using the _column_keep_sort_key to determine which one is more suitable to keep 
    keep = sorted([left, right], key=_column_keep_sort_key)[0]
    #the other one becomes the drop candidate
    drop = right if keep.name == left.name else left
    #returns the names of the column to keep and the column to drop
    return keep.name, drop.name


def _build_summary(actions: list[RemediationAction]) -> str:
    '''Builds a summary string for the remediation plan, counting auto-apply and manual actions'''
    #counts auto-applicable actions
    auto_apply = sum(1 for action in actions if action.auto_apply)
    #counts manual/review actions
    manual = sum(1 for action in actions if not action.auto_apply)
    #returns a summary sentence describing the plan
    return (
        f"Planned {len(actions)} remediation actions: {auto_apply} auto-apply and {manual} review/report actions."
    )


def build_remediation_plan(validation_results: OrchestrationStepResult) -> RemediationPlan:
    '''Builds a remediation plan based on the validation results, generating actions for schema issues, completeness problems, consistency violations, anomalies, cross-column issues, and duplicates'''
    #builds a quick lookup map for schema columns
    schema_map = _schema_column_map(validation_results)
    #computes the rename map
    rename_map = _planned_rename_map(validation_results.schema_validation.columns)
    #starts an empty list of actions
    actions: list[RemediationAction] = []
    #Loops through schema columns and adds rename actions when needed
    for column in validation_results.schema_validation.columns:
        #only runs if a column name is invalid and has a rename suggestion (with conflict resolution)
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
        #Still inside the schema-column loop, adds dtype-casting actions, only for upported inferred pandas dtypes
        if column.pandas_dtype in {"datetime64[ns]", "Int64", "Float64", "boolean", "string"}:
            #if the column will be renamed, use the final renamed column name
            final_column_name = rename_map.get(column.name, column.name)
            #creates a cast_dtype remediation action with high confidence, low risk, and auto-apply enabled.
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
    #Next, it loops through the completeness analysis findings to identify columns with placeholder-like values and adds actions to replace them with nulls
    for column_finding in validation_results.completeness_analysis.per_column:
        #skips columns with no placeholder-missing values
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
    #Then, it processes the consistency validation findings to identify columns with consistent pattern violations and adds actions to generate cleaners for them
    for finding in validation_results.consistency_validation.format_consistency_findings:
        actions.append(
            #For each inconsistent-format column, creates a generate_cleaner action.
            #This means the system should generate and apply a cleaning rule to match the expected format pattern
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
    #Finally, it processes ross-column validation findings. 
    # It runs if cross-column validation exists and loops through each finding.
    if validation_results.cross_column_validation is not None:
        for finding in validation_results.cross_column_validation.findings:
            #Special handling for exact duplicate columns. 
            #checks that the finding is exactly two identical columns and extracts the two names and skips if either is missing from the schema map.
            if finding.check_type == "exact_duplicate_columns" and len(finding.columns) == 2:
                left_name, right_name = finding.columns
                if left_name not in schema_map or right_name not in schema_map:
                    continue
                #chooses which column to keep and which to drop
                keep_name, drop_name = _choose_keep_drop_columns(left_name, right_name, schema_map)
                actions.append(
                    #creates an auto-applied drop_exact_duplicate_column action with reasoning and preview stats.
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
            #Handles more ambiguous cross-column issues, he code creates a manual_review action instead of auto-fixing it
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
    #Handles anomaly detection findings. It only runs if anomaly detection exists and loops through each finding. 
    if validation_results.anomaly_detection is not None:
        for finding in validation_results.anomaly_detection.findings:
            # It creates a manual_review action for high-severity anomalies and a report_only action for lower-severity ones, with appropriate reasoning and preview stats.
            action_type = "manual_review" if finding.severity == "high" else "report_only"
            actions.append(
                #builds the corresponding remediation action.Nothing here is auto-applied
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
    #Handles duplicate row groups. It only runs if duplicate detection exists  
    if validation_results.duplicate_detection is not None:
        #loops through duplicate groups and numbers them starting from 1.
        for index, group in enumerate(validation_results.duplicate_detection.groups, start=1):
            # Exact duplicate rows can be removed deterministically by keeping the
            # first occurrence and dropping the remaining row indices in the group.
            if group.duplicate_type == "exact_row":
                keep_row_index = group.row_indices[0] if group.row_indices else None
                drop_row_indices = group.row_indices[1:] if len(group.row_indices) > 1 else []
                actions.append(
                    RemediationAction(
                        action_id=_action_id("drop_exact_duplicate_rows", index),
                        action_type="drop_exact_duplicate_rows",
                        object_type="row_group",
                        target={
                            "keep_row_index": keep_row_index,
                            "drop_row_indices": drop_row_indices,
                            "key_columns": group.key_columns,
                        },
                        source_check="duplicate_detection",
                        confidence="high",
                        risk_level="medium",
                        auto_apply=True,
                        status="planned",
                        reason=(
                            f"{group.evidence} Keep the first occurrence at row {keep_row_index} "
                            f"and drop {len(drop_row_indices)} later duplicate row(s)."
                        ),
                        preview_stats={
                            "group_size": len(group.row_indices),
                            "drop_count": len(drop_row_indices),
                            "key_columns": group.key_columns,
                        },
                    )
                )
                continue
            #other duplicate types become manual_review with low confidence and high risk
            actions.append(
                #creates the action, storing up to 20 row indices and key columns in the target
                RemediationAction(
                    action_id=_action_id(group.duplicate_type, index),
                    action_type="manual_review",
                    object_type="row_group",
                    target={"row_indices": group.row_indices[:20], "key_columns": group.key_columns},
                    source_check="duplicate_detection",
                    confidence="low",
                    risk_level="high",
                    auto_apply=False,
                    status="proposed_not_applied",
                    reason=group.evidence,
                    preview_stats={
                        "group_size": len(group.row_indices),
                        "key_columns": group.key_columns,
                    },
                )
            )
    #Finalizes the plan
    #sorts actions so auto-apply actions come first, then by action type and ID for consistency
    actions.sort(key=lambda action: (0 if action.auto_apply else 1, action.action_type, action.action_id))
    #returns a RemediationPlan with dataset name, all actions, and a generated summary
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
    '''A helper for deciding where validation results come from. It prioritizes provided results, then saved results if reuse is enabled, and finally builds new results if needed.'''
    #if results were directly passed in, return them immediately without checking for saved results or rebuilding
    if validation_results is not None:
        return validation_results
    #if reuse of saved validation is allowed, try to load cached validation results; ignore FileNotFoundError and proceed to build validation results from scratch if cached results are not found
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
    '''The main entry point for remediation planning. It resolves validation results, builds the remediation plan, and handles caching based on the provided flags.'''
    # reuse of saved remediation is allowed, try loading the cached plan; ignore FileNotFoundError and proceed to resolve validation results and build a new remediation plan if cached plan is not found
    if reuse_saved_remediation:
        try:
            return load_remediation_plan(path)
        except FileNotFoundError:
            pass
    #resolve validation results using the helper above
    resolved_validation = _resolve_validation_results(path, validation_results, reuse_saved_validation)
    #build the remediation plan based on the resolved validation results
    plan = build_remediation_plan(resolved_validation)
    #save the plan to cache/storage for future reuse and return it
    save_remediation_plan(path, plan)
    return plan
