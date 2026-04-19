from __future__ import annotations

import pandas as pd

from pathlib import Path

from agents import (
    completeness_analysis_agent,
    dtype_inference_agent,
    format_consistency_agent,
    schema_summary_agent,
)
from cache import (
    load_completeness,
    load_consistency,
    load_schema_handoff,
    save_completeness,
    save_consistency,
    save_schema_handoff,
    save_validation_results,
    load_validation_results,
)
from models import (
    ColumnConsistencyReport,
    CompletenessAnalysisReport,
    ConsistencyValidationReport,
    DatasetDtypeInference,
    FormatConsistencyFinding,
    OrchestrationStepResult,
    SchemaColumnEntry,
    SchemaHandoff,
    SchemaIssue,
)
from tools.tools import (
    attach_profile_text,
    attach_text_document,
    build_column_format_facts,
    build_completeness_profile,
    build_dataset_profile,
    build_dtype_inference_text,
    is_valid_schema_name,
    load_dataset_frame,
    naming_rule_reason,
    normalized_schema_name,
    run_agent_with_backoff,
    SchemaDuplicateGroup,
    suggest_schema_name,
)


def build_schema_issues(
    columns: list[SchemaColumnEntry],
    duplicate_groups: list[SchemaDuplicateGroup],
) -> list[SchemaIssue]:
    issues: list[SchemaIssue] = []

    for col in columns:
        if not col.naming_valid:
            issues.append(
                SchemaIssue(
                    column_name=col.name,
                    issue_type="naming_standard",
                    severity="high",
                    evidence=col.naming_reason or "Column name violates the lowercase snake_case naming rule.",
                    fix_confidence="high",
                    suggested_fix=col.rename_suggestion or "",
                    suggested_strategy=f"Safe local rename to '{col.rename_suggestion}'.",
                )
            )

    for group in duplicate_groups:
        for column_name in group.columns:
            peer_columns = [peer for peer in group.columns if peer != column_name]
            peer_text = ", ".join(peer_columns)
            issues.append(
                SchemaIssue(
                    column_name=column_name,
                    issue_type="duplicate_column_semantics",
                    severity="medium",
                    evidence=(
                        f"Normalized schema name matches the peer column group '{group.canonical_name}', "
                        f"suggesting overlap with: {peer_text}."
                    ),
                    fix_confidence="medium",
                    suggested_fix="",
                    suggested_strategy=(
                        f"Compare values, null patterns, and business usage with: {peer_text} before any merge or drop."
                    ),
                )
            )

    return issues


def run_dtype_inference(path: Path) -> DatasetDtypeInference:
    df = load_dataset_frame(path)
    text = build_dtype_inference_text(df)
    prompt = [
        "Infer the correct pandas dtype for each column based on the attached CSV sample.",
        attach_text_document(text),
    ]
    result = run_agent_with_backoff(dtype_inference_agent, prompt)
    return result.output


def run_schema_validation(path: Path, reuse_cache: bool = False) -> SchemaHandoff:
    if reuse_cache:
        return load_schema_handoff(path)

    df = load_dataset_frame(path)

    # LLM call 1: infer dtype, role, pattern, rationale per column
    dtype_inference = run_dtype_inference(path)
    dtype_map = {col.column_name: col for col in dtype_inference.columns}
    dtype_overrides = {name: col.pandas_dtype for name, col in dtype_map.items()}

    # Statistical profile (non-null counts, parse pcts, sample values)
    profile = build_dataset_profile(df, path.stem, dtype_overrides=dtype_overrides)

    # Duplicate detection: group columns that normalize to the same canonical name
    duplicate_groups_by_name: dict[str, list[str]] = {}
    for col_name in df.columns:
        canonical = normalized_schema_name(col_name)
        duplicate_groups_by_name.setdefault(canonical, []).append(col_name)
    duplicate_groups = [
        SchemaDuplicateGroup(canonical_name=cn, columns=cols)
        for cn, cols in duplicate_groups_by_name.items()
        if len(cols) > 1
    ]

    # Merge everything into one entry per column
    columns: list[SchemaColumnEntry] = []
    for col_profile in profile.columns_profiles:
        name = col_profile.column_name
        dtype_col = dtype_map.get(name)
        naming_valid = is_valid_schema_name(name)
        columns.append(SchemaColumnEntry(
            name=name,
            pandas_dtype=col_profile.pandas_dtype,
            numeric_role=dtype_col.numeric_role if dtype_col else None,
            string_role=dtype_col.string_role if dtype_col else None,
            detected_pattern=dtype_col.detected_pattern if dtype_col else None,
            rationale=dtype_col.rationale if dtype_col else "",
            non_null_rows=col_profile.non_null_rows,
            distinct_non_null_values=col_profile.distinct_non_null_values,
            numeric_parse_pct=col_profile.numeric_parse_pct,
            datetime_parse_pct=col_profile.datetime_parse_pct,
            empty_like_pct=col_profile.empty_like_pct,
            sample_values=col_profile.sample_values,
            naming_valid=naming_valid,
            rename_suggestion=suggest_schema_name(name) if not naming_valid else None,
            naming_reason=naming_rule_reason(name) if not naming_valid else None,
        ))

    issues = build_schema_issues(columns, duplicate_groups)

    # LLM call 2: summary agent receives the full handoff (minus summary)
    handoff = SchemaHandoff(
        dataset_name=path.stem,
        total_rows=len(df),
        total_columns=len(df.columns),
        columns=columns,
        issues=issues,
        duplicate_groups=duplicate_groups,
    )
    result = run_agent_with_backoff(schema_summary_agent, [
        (
            f"Summarize the provided schema analysis for dataset {path.stem}. "
            "Do not infer new findings. Summarize the provided findings for a later cleaner or validator."
        ),
        attach_profile_text(handoff),
    ])
    handoff = handoff.model_copy(update={"summary": result.output.summary})

    save_schema_handoff(path, handoff)
    return handoff


def run_completeness_analysis(path: Path, reuse_cache: bool = False) -> CompletenessAnalysisReport:
    if reuse_cache:
        return load_completeness(path)
    df = load_dataset_frame(path)
    profile = build_completeness_profile(df, path.stem)
    prompt = [
        (
            f"Analyze the attached completeness profile for dataset {path.stem}. "
            "Use Python in code execution to inspect the profile document. "
            "This is step 2 of the orchestration only: Completeness Analysis. "
            "Use the provided metrics to summarize per-column completeness, detect missing-like and placeholder values, "
            "identify actual placeholder tokens present in the dataset, and flag sparse columns that may be candidates for removal or investigation."
        ),
        attach_profile_text(profile),
    ]
    result = run_agent_with_backoff(completeness_analysis_agent, prompt)
    report = result.output
    save_completeness(path, report)
    return report


def _build_suggested_strategy(
    expected_pattern: str,
    dominant_shape: str | None,
    inconsistent_examples,
    dominant_example_values: list[str] | None = None,
) -> str:
    """Build a cleaner-agent strategy string that enumerates each outlier shape group
    with representative examples, so the agent handles every case explicitly."""
    from collections import defaultdict
    groups: dict[str, list[str]] = defaultdict(list)
    for ex in inconsistent_examples:
        if ex.shape != dominant_shape:
            groups[ex.shape].append(ex.value)

    if not groups:
        return (
            f"Normalize all values to '{expected_pattern}'. "
            "Map to null when the value cannot be converted."
        )

    lines = [
        f"Target format: '{expected_pattern}'. "
        f"Dominant valid shape: '{dominant_shape}' — values matching this shape are already valid, preserve them. "
    ]

    if dominant_example_values:
        dominant_sample = ", ".join(repr(v) for v in dominant_example_values[:5])
        lines.append(
            f"Examples of already-valid values (the OUTPUT must look exactly like these): {dominant_sample}.\n"
        )

    lines.append(
        "Handle every outlier shape group below by inferring the transformation from the examples. "
        "For each group, verify your transformation produces output that matches the already-valid examples above — "
        "same length, same character structure, same field order (e.g. YYYY before MM, not MM before YYYY). "
        "Use partial matches, prefix stripping, abbreviation expansion, or abbreviation mapping as needed. "
        "Map to null ONLY when a value is completely unrecognisable — never null a value that contains "
        "recoverable information:\n"
    )

    for shape, values in sorted(groups.items(), key=lambda x: -len(x[1])):
        examples = ", ".join(repr(v) for v in values[:5])
        lines.append(f"  shape '{shape}': e.g. {examples}")

    lines.append(
        "\nEVERY value in example_inconsistent_values must be explicitly handled — "
        "do not leave any outlier value unchanged unless it already matches the target format. "
        "Prefer a best-effort conversion over null whenever the value contains recoverable information."
    )
    return "\n".join(lines)


def run_column_format_check(
    df: pd.DataFrame,
    column_name: str,
    dataset_name: str,
    context_label: str,
    schema_entry: SchemaColumnEntry | None = None,
) -> ColumnConsistencyReport:
    # Skip columns that are never machine-format candidates by role
    if schema_entry is not None and schema_entry.string_role in ("name", "free_text"):
        return ColumnConsistencyReport(
            finding=None,
            summary=(
                f"Column '{column_name}' skipped: string_role='{schema_entry.string_role}' "
                f"is not a machine-format candidate."
            ),
        )

    format_facts = build_column_format_facts(df, column_name)
    if not format_facts.machine_format_candidate or format_facts.inconsistent_rows <= 0:
        return ColumnConsistencyReport(
            finding=None,
            summary=(
                f"No actionable machine-format inconsistency detected for column '{column_name}' "
                f"in {context_label}."
            ),
        )

    # Fast path: schema already identified the dominant pattern — no LLM needed.
    # The profiler has already counted inconsistent rows and collected outlier examples;
    # we just need to package them into a finding.
    if schema_entry is not None and schema_entry.detected_pattern:
        inconsistent_examples = [ex.value for ex in format_facts.inconsistent_examples]
        return ColumnConsistencyReport(
            finding=FormatConsistencyFinding(
                column_name=column_name,
                expected_pattern=schema_entry.detected_pattern,
                inconsistent_rows=format_facts.inconsistent_rows,
                example_inconsistent_values=inconsistent_examples,
                evidence=(
                    f"Schema handoff identified dominant pattern '{schema_entry.detected_pattern}'. "
                    f"Profiler found {format_facts.inconsistent_rows} rows deviating from the "
                    f"dominant shape '{format_facts.dominant_shape}' "
                    f"({format_facts.dominant_shape_pct:.1f}% of non-null rows match the dominant shape)."
                ),
                suggested_strategy=_build_suggested_strategy(
                    schema_entry.detected_pattern,
                    format_facts.dominant_shape,
                    format_facts.inconsistent_examples,
                    format_facts.dominant_example_values,
                ),
            ),
            summary=(
                f"Column '{column_name}': {format_facts.inconsistent_rows} inconsistent rows detected "
                f"against schema pattern '{schema_entry.detected_pattern}' (no LLM call)."
            ),
        )

    # Slow path: no schema pattern available — let the agent infer it.
    schema_context = ""
    if schema_entry is not None:
        schema_context = (
            f" Target dtype: {schema_entry.pandas_dtype}."
            f" Semantic role: {schema_entry.numeric_role or schema_entry.string_role or 'unknown'}."
        )
    prompt = [
        (
            f"Analyze the attached ColumnFormatFacts for dataset '{dataset_name}', column '{column_name}'."
            f"{schema_context}"
            f" Total rows: {len(df)}."
            f" Dominant shape: '{format_facts.dominant_shape}' ({format_facts.dominant_shape_pct:.1f}% of non-null rows)."
            f" Inconsistent rows: {format_facts.inconsistent_rows}."
            " Determine whether a format inconsistency exists that a cleaning function could fix."
            " If yes, write a precise suggested_strategy listing every outlier shape with concrete transformation steps."
            " Return finding=null if the column is consistent or format consistency is not applicable."
        ),
        attach_profile_text(format_facts),
    ]
    result = run_agent_with_backoff(format_consistency_agent, prompt)
    output = result.output
    if output.finding is not None and output.finding.inconsistent_rows <= 0:
        return ColumnConsistencyReport(
            finding=None,
            summary=output.summary,
        )
    return output


def run_format_consistency_validation(path: Path, reuse_cache: bool = False, read_as_str: bool = False) -> ConsistencyValidationReport:
    if reuse_cache:
        return load_consistency(path)
    df = load_dataset_frame(path, dtype=str if read_as_str else None)
    format_findings: list[FormatConsistencyFinding] = []

    # Load schema handoff for richer column context; fall back gracefully if unavailable
    schema_map: dict[str, SchemaColumnEntry] = {}
    try:
        handoff = load_schema_handoff(path)
        schema_map = {col.name: col for col in handoff.columns}
    except Exception:
        pass

    for column_name in df.columns:
        schema_entry = schema_map.get(column_name)
        result = run_column_format_check(df, column_name, path.stem, "original validation", schema_entry=schema_entry)
        if result.finding is not None:
            format_findings.append(result.finding)

    report = ConsistencyValidationReport(
        dataset_name=path.stem,
        total_rows=len(df),
        format_consistency_findings=format_findings,
        summary=(
            f"Analyzed all {len(df.columns)} columns individually for format consistency and "
            f"detected {len(format_findings)} format issues."
        ),
    )
    save_consistency(path, report)
    return report


def run_consistency_validation(path: Path, reuse_cache: bool = False, read_as_str: bool = False) -> ConsistencyValidationReport:
    return run_format_consistency_validation(path, reuse_cache=reuse_cache, read_as_str=read_as_str)


def build_validation_results(
    path: Path,
    reuse_schema: bool = False,
    reuse_completeness: bool = False,
    reuse_consistency: bool = False,
) -> OrchestrationStepResult:
    validation_results = OrchestrationStepResult(
        schema_validation=run_schema_validation(path, reuse_cache=reuse_schema),
        completeness_analysis=run_completeness_analysis(path, reuse_cache=reuse_completeness),
        consistency_validation=run_consistency_validation(path, reuse_cache=reuse_consistency),
    )
    save_validation_results(path, validation_results)
    return validation_results
