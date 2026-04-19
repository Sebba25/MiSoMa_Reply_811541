"""Validation-stage orchestration.

Drives the validation half of the pipeline: dtype inference, schema
validation, completeness analysis, per-column format consistency, anomaly
detection, cross-column checks and duplicate detection. Each stage function
caches its artifact via ``cache.py`` and — where an LLM summary is needed —
funnels through ``_summarize_validation_report`` to keep the
agent-call boilerplate in one place.

The public entry point is ``build_validation_results`` (run every stage),
but each ``run_<stage>`` helper is also importable on its own for the CLI
and the Streamlit UI.
"""

from __future__ import annotations

from collections import Counter
import sys
import pandas as pd

from pathlib import Path

from agents import (
    anomaly_summary_agent,
    completeness_analysis_agent,
    cross_column_summary_agent,
    duplicate_summary_agent,
    dtype_inference_agent,
    format_consistency_agent,
    schema_summary_agent,
)
from cache import (
    load_anomaly,
    load_completeness,
    load_consistency,
    load_cross_column,
    load_duplicates,
    load_schema_handoff,
    save_anomaly,
    save_completeness,
    save_consistency,
    save_cross_column,
    save_duplicates,
    save_schema_handoff,
    save_validation_results,
)
from models import (
    AnomalyDetectionReport,
    AnomalyFinding,
    ColumnConsistencyReport,
    CompletenessAnalysisReport,
    ConsistencyValidationReport,
    CrossColumnFinding,
    CrossColumnValidationReport,
    DatasetDtypeInference,
    DuplicateDetectionReport,
    DuplicateRecordGroup,
    FormatConsistencyFinding,
    OrchestrationStepResult,
    SchemaColumnEntry,
    SchemaHandoff,
    SchemaIssue,
)
from tools.tools import (
    PLACEHOLDER_TOKENS,
    FormatOutlierExample,
    attach_profile_text,
    attach_text_document,
    build_column_format_facts,
    build_completeness_profile,
    build_dataset_profile,
    build_dtype_inference_text,
    detect_date_order_violations,
    detect_duplicate_like_columns,
    detect_duplicate_semantic_conflicts,
    detect_exact_duplicate_groups,
    detect_near_duplicate_groups,
    detect_numeric_outlier_candidates,
    detect_rare_category_candidates,
    detect_year_month_period_mismatches,
    infer_duplicate_key_columns,
    is_valid_schema_name,
    load_dataset_frame,
    matches_numeric_schema_pattern,
    naming_rule_reason,
    numeric_pattern_allows_variable_width,
    normalized_schema_name,
    run_agent_with_backoff,
    SchemaDuplicateGroup,
    suggest_schema_name,
    value_shape,
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
    print(f"[orchestrator][schema][dtype-inference] dataset='{path.stem}'", file=sys.stderr, flush=True)
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
    print(f"[orchestrator][schema][summary] dataset='{path.stem}'", file=sys.stderr, flush=True)
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
    print(f"[orchestrator][completeness] dataset='{path.stem}'", file=sys.stderr, flush=True)
    result = run_agent_with_backoff(completeness_analysis_agent, prompt)
    report = result.output
    save_completeness(path, report)
    return report


def _summarize_validation_report(agent, prompt_text: str, report, fallback_summary: str) -> str:
    try:
        result = run_agent_with_backoff(agent, [prompt_text, attach_profile_text(report)])
        summary = getattr(result.output, "summary", "").strip()
        return summary or fallback_summary
    except Exception:
        return fallback_summary


def _duplicate_semantic_suppressed_columns(columns: list[SchemaColumnEntry], duplicate_groups) -> set[str]:
    by_name = {column.name: column for column in columns}
    suppressed: set[str] = set()
    for group in duplicate_groups:
        present = [name for name in group.columns if name in by_name]
        if len(present) < 2:
            continue

        def _sort_key(name: str) -> tuple[int, int, str]:
            column = by_name[name]
            return (
                0 if column.naming_valid else 1,
                0 if normalized_schema_name(name) == name else 1,
                name,
            )

        preferred = sorted(present, key=_sort_key)[0]
        suppressed.update(name for name in present if name != preferred)
    return suppressed


def run_anomaly_detection(path: Path, reuse_cache: bool = False) -> AnomalyDetectionReport:
    if reuse_cache:
        return load_anomaly(path)

    df = load_dataset_frame(path)
    try:
        handoff = load_schema_handoff(path)
        schema_columns = handoff.columns
        suppressed_columns = _duplicate_semantic_suppressed_columns(handoff.columns, handoff.duplicate_groups)
    except FileNotFoundError:
        schema_columns = []
        suppressed_columns = set()

    findings = [
        AnomalyFinding(**finding)
        for finding in (
            detect_numeric_outlier_candidates(df, schema_columns)
            + detect_rare_category_candidates(df, schema_columns)
        )
        if finding["column_name"] not in suppressed_columns
    ]
    findings.sort(key=lambda finding: (-finding.affected_rows, finding.column_name, finding.anomaly_type))
    fallback_summary = (
        f"Detected {len(findings)} anomaly findings across numeric outliers and rare categorical values."
        if findings
        else "No anomaly findings were detected by the current heuristic checks."
    )
    report = AnomalyDetectionReport(
        dataset_name=path.stem,
        total_rows=len(df),
        total_columns=len(df.columns),
        findings=findings,
        summary=fallback_summary,
    )
    print(f"[orchestrator][anomaly][summary] dataset='{path.stem}'", file=sys.stderr, flush=True)
    report = report.model_copy(
        update={
            "summary": _summarize_validation_report(
                anomaly_summary_agent,
                (
                    f"Summarize the provided anomaly-detection findings for dataset {path.stem}. "
                    "Do not infer new findings or alter the provided findings."
                ),
                report,
                fallback_summary,
            )
        }
    )
    save_anomaly(path, report)
    return report


def run_cross_column_validation(path: Path, reuse_cache: bool = False) -> CrossColumnValidationReport:
    if reuse_cache:
        return load_cross_column(path)

    df = load_dataset_frame(path)
    try:
        handoff = load_schema_handoff(path)
        schema_columns = handoff.columns
        duplicate_groups = handoff.duplicate_groups
    except FileNotFoundError:
        schema_columns = []
        duplicate_groups = []

    findings = [
        CrossColumnFinding(**finding)
        for finding in (
            detect_duplicate_like_columns(df, schema_columns)
            + detect_duplicate_semantic_conflicts(df, duplicate_groups)
            + detect_year_month_period_mismatches(df, schema_columns)
            + detect_date_order_violations(df, schema_columns)
        )
    ]
    findings.sort(key=lambda finding: (-finding.affected_rows, ",".join(finding.columns), finding.check_type))
    fallback_summary = (
        f"Detected {len(findings)} cross-column consistency findings."
        if findings
        else "No cross-column consistency findings were detected by the current rule set."
    )
    report = CrossColumnValidationReport(
        dataset_name=path.stem,
        total_rows=len(df),
        findings=findings,
        summary=fallback_summary,
    )
    print(f"[orchestrator][cross-column][summary] dataset='{path.stem}'", file=sys.stderr, flush=True)
    report = report.model_copy(
        update={
            "summary": _summarize_validation_report(
                cross_column_summary_agent,
                (
                    f"Summarize the provided cross-column validation findings for dataset {path.stem}. "
                    "Do not infer new findings or alter the provided findings."
                ),
                report,
                fallback_summary,
            )
        }
    )
    save_cross_column(path, report)
    return report


def run_duplicate_detection(path: Path, reuse_cache: bool = False) -> DuplicateDetectionReport:
    if reuse_cache:
        return load_duplicates(path)

    df = load_dataset_frame(path)
    try:
        handoff = load_schema_handoff(path)
        schema_columns = handoff.columns
    except FileNotFoundError:
        schema_columns = []

    key_columns = infer_duplicate_key_columns(schema_columns, df)
    groups = [
        DuplicateRecordGroup(**group)
        for group in (
            detect_exact_duplicate_groups(df)
            + detect_near_duplicate_groups(df, key_columns)
        )
    ]
    fallback_summary = (
        f"Detected {len(groups)} duplicate-record groups."
        if groups
        else "No duplicate-record groups were detected by the current exact and near-duplicate checks."
    )
    report = DuplicateDetectionReport(
        dataset_name=path.stem,
        total_rows=len(df),
        groups=groups,
        summary=fallback_summary,
    )
    print(f"[orchestrator][duplicates][summary] dataset='{path.stem}'", file=sys.stderr, flush=True)
    report = report.model_copy(
        update={
            "summary": _summarize_validation_report(
                duplicate_summary_agent,
                (
                    f"Summarize the provided duplicate-detection findings for dataset {path.stem}. "
                    "Do not infer new findings or alter the provided findings."
                ),
                report,
                fallback_summary,
            )
        }
    )
    save_duplicates(path, report)
    return report


def _build_suggested_strategy(
    expected_pattern: str,
    dominant_shape: str | None,
    inconsistent_examples,
    dominant_example_values: list[str] | None = None,
    allow_variable_numeric_width: bool = False,
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

    if allow_variable_numeric_width:
        lines.append(
            "Handle every outlier shape group below by converting values into a valid numeric representation for the "
            "target pattern. Do not force every output to have the same string length as the examples above — values "
            "with different widths can still be valid if they parse correctly and preserve the intended meaning. Use "
            "partial matches, prefix stripping, abbreviation expansion, or mapping as needed. Map to null ONLY when a "
            "value is completely unrecognisable — never null a value that contains recoverable information:\n"
        )
        if expected_pattern.strip().lower() == "month number (1-12)":
            lines.append(
                "Pure numeric values that are outside the valid month range 1-12 (for example 0, 13, 99, or negative numbers) "
                "are invalid and should map to null rather than being coerced to a different month.\n"
            )
    else:
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


def _profile_schema_guided_inconsistencies(
    df: pd.DataFrame,
    column_name: str,
    schema_entry: SchemaColumnEntry,
) -> tuple[int, list[FormatOutlierExample]] | None:
    if schema_entry.pandas_dtype not in {"Int64", "Float64"}:
        return None

    rendered = df[column_name].dropna().astype(str).str.strip()
    rendered = rendered[rendered != ""]
    rendered = rendered[~rendered.str.lower().isin(PLACEHOLDER_TOKENS)]

    invalid_values = [
        value
        for value in rendered
        if not (
            matches_numeric_schema_pattern(
                value,
                pandas_dtype=schema_entry.pandas_dtype,
                numeric_role=schema_entry.numeric_role,
                detected_pattern=schema_entry.detected_pattern,
            )
            or False
        )
    ]
    if not invalid_values:
        return 0, []

    counts = Counter(invalid_values)
    examples = [
        FormatOutlierExample(
            value=value[:80],
            shape=value_shape(value),
            count=count,
        )
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:60]
    ]
    return len(invalid_values), examples


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
        inconsistent_rows = format_facts.inconsistent_rows
        inconsistent_example_profiles = format_facts.inconsistent_examples
        used_schema_override = False

        guided_profile = _profile_schema_guided_inconsistencies(df, column_name, schema_entry)
        if guided_profile is not None:
            inconsistent_rows, inconsistent_example_profiles = guided_profile
            used_schema_override = True

        if inconsistent_rows <= 0:
            return ColumnConsistencyReport(
                finding=None,
                summary=(
                    f"No actionable machine-format inconsistency detected for column '{column_name}' "
                    f"against schema pattern '{schema_entry.detected_pattern}' in {context_label}."
                ),
            )

        inconsistent_examples = [ex.value for ex in inconsistent_example_profiles]
        evidence = (
            f"Schema handoff identified dominant pattern '{schema_entry.detected_pattern}'. "
            f"Profiler found {inconsistent_rows} rows deviating from the "
            f"dominant shape '{format_facts.dominant_shape}' "
            f"({format_facts.dominant_shape_pct:.1f}% of non-null rows match the dominant shape)."
        )
        if used_schema_override:
            evidence = (
                f"Schema handoff identified target dtype '{schema_entry.pandas_dtype}' and pattern "
                f"'{schema_entry.detected_pattern}'. Schema-guided validation found "
                f"{inconsistent_rows} rows that do not match the target numeric representation."
            )

        return ColumnConsistencyReport(
            finding=FormatConsistencyFinding(
                column_name=column_name,
                expected_pattern=schema_entry.detected_pattern,
                inconsistent_rows=inconsistent_rows,
                example_inconsistent_values=inconsistent_examples,
                evidence=evidence,
                suggested_strategy=_build_suggested_strategy(
                    schema_entry.detected_pattern,
                    format_facts.dominant_shape,
                    inconsistent_example_profiles,
                    format_facts.dominant_example_values,
                    allow_variable_numeric_width=numeric_pattern_allows_variable_width(
                        pandas_dtype=schema_entry.pandas_dtype,
                        numeric_role=schema_entry.numeric_role,
                        detected_pattern=schema_entry.detected_pattern,
                    ),
                ),
            ),
            summary=(
                f"Column '{column_name}': {inconsistent_rows} inconsistent rows detected "
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
    print(
        f"[orchestrator][consistency][format-agent] dataset='{dataset_name}' column='{column_name}'",
        file=sys.stderr,
        flush=True,
    )
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


def build_validation_results(
    path: Path,
    reuse_schema: bool = False,
    reuse_completeness: bool = False,
    reuse_consistency: bool = False,
) -> OrchestrationStepResult:
    schema_validation = run_schema_validation(path, reuse_cache=reuse_schema)
    completeness_analysis = run_completeness_analysis(path, reuse_cache=reuse_completeness)
    consistency_validation = run_format_consistency_validation(path, reuse_cache=reuse_consistency)
    print(f"[orchestrator][anomaly] dataset='{path.stem}'", file=sys.stderr, flush=True)
    anomaly_detection = run_anomaly_detection(path)
    print(f"[orchestrator][cross-column] dataset='{path.stem}'", file=sys.stderr, flush=True)
    cross_column_validation = run_cross_column_validation(path)
    print(f"[orchestrator][duplicates] dataset='{path.stem}'", file=sys.stderr, flush=True)
    duplicate_detection = run_duplicate_detection(path)
    validation_results = OrchestrationStepResult(
        schema_validation=schema_validation,
        completeness_analysis=completeness_analysis,
        consistency_validation=consistency_validation,
        anomaly_detection=anomaly_detection,
        cross_column_validation=cross_column_validation,
        duplicate_detection=duplicate_detection,
    )
    save_validation_results(path, validation_results)
    return validation_results
