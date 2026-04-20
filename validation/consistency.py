"""Per-column format-consistency stage.

Has both a fast path (schema-guided, no LLM call — packages the profiler's
outlier count/examples into a ``FormatConsistencyFinding`` directly) and a
slow path where ``format_consistency_agent`` infers the dominant shape
from the attached ``ColumnFormatFacts`` when no schema pattern is available.

``_build_suggested_strategy`` is the cleaner-agent strategy string: it
enumerates every outlier shape with concrete examples so the downstream
generator handles each group explicitly.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from agents import format_consistency_agent
from cache import load_consistency, load_schema_handoff, save_consistency
from models import (
    ColumnConsistencyReport,
    ConsistencyValidationReport,
    FormatConsistencyFinding,
    SchemaColumnEntry,
)
from tools import (
    PLACEHOLDER_TOKENS,
    FormatOutlierExample,
    attach_profile_text,
    build_column_format_facts,
    load_dataset_frame,
    matches_numeric_schema_pattern,
    numeric_pattern_allows_variable_width,
    run_agent_with_backoff,
    value_shape,
)


def _build_suggested_strategy(
    expected_pattern: str,
    dominant_shape: str | None,
    inconsistent_examples,
    dominant_example_values: list[str] | None = None,
    allow_variable_numeric_width: bool = False,
) -> str:
    """Build a cleaner-agent strategy string that enumerates each outlier shape
    group with representative examples, so the agent handles every case explicitly."""
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


def run_format_consistency_validation(
    path: Path,
    reuse_cache: bool = False,
    read_as_str: bool = False,
) -> ConsistencyValidationReport:
    if reuse_cache:
        return load_consistency(path)
    df = load_dataset_frame(path, dtype=str if read_as_str else None)
    format_findings: list[FormatConsistencyFinding] = []

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
