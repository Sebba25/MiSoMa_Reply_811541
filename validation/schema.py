"""Dtype inference + schema validation.

Two LLM calls per run:

1. ``dtype_inference_agent`` — pandas dtype, semantic role, and detected
   pattern per column.
2. ``schema_summary_agent`` — a human-readable summary over the already-
   built ``SchemaHandoff`` (does not re-derive findings).

Around those calls we build the statistical profile, detect duplicate
columns via canonical-name collisions, and emit ``SchemaIssue``s for both
naming violations and duplicate-column peers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from agents import dtype_inference_agent, schema_summary_agent
from cache import load_schema_handoff, save_schema_handoff
from models import (
    DatasetDtypeInference,
    SchemaColumnEntry,
    SchemaHandoff,
    SchemaIssue,
)
from tools import (
    SchemaDuplicateGroup,
    attach_profile_text,
    attach_text_document,
    build_dataset_profile,
    build_dtype_inference_text,
    is_valid_schema_name,
    load_dataset_frame,
    naming_rule_reason,
    normalized_schema_name,
    run_agent_with_backoff,
    suggest_schema_name,
)


def _pattern_is_ambiguous(pattern: str | None) -> bool:
    if not pattern:
        return False
    normalized = pattern.strip().lower()
    if not normalized:
        return False
    return any(
        token in normalized
        for token in (" / ", "/", " and ", " or ", "mixed ", "multiple ", "various ", "several ")
    )


def _parse_numeric_samples(sample_values: list[str]) -> tuple[list[int], bool]:
    ints: list[int] = []
    saw_decimal = False
    for raw in sample_values:
        value = str(raw).strip()
        if not value:
            continue
        if re.fullmatch(r"-?\d+", value):
            ints.append(int(value))
            continue
        if re.fullmatch(r"-?\d+\.\d+", value):
            number = float(value)
            if number.is_integer():
                ints.append(int(number))
            else:
                saw_decimal = True
            continue
        if re.fullmatch(r"-?\d+,\d+", value):
            number = float(value.replace(",", "."))
            if number.is_integer():
                ints.append(int(number))
            else:
                saw_decimal = True
    return ints, saw_decimal


def _infer_numeric_pattern(column_name: str, sample_values: list[str], numeric_role: str | None) -> str:
    normalized_name = normalized_schema_name(column_name)
    tokens = set(normalized_name.split("_"))
    ints, saw_decimal = _parse_numeric_samples(sample_values)
    rendered_samples = [str(value).strip() for value in sample_values if str(value).strip()]

    if (
        ints
        and all(1 <= value <= 12 for value in ints[:10])
        and ({"mese", "month"} & tokens)
    ):
        return "month number (1-12)"

    if (
        ints
        and all(1900 <= value <= 2100 for value in ints[:10])
        and ({"anno", "year"} & tokens)
    ):
        return "4-digit year"

    if rendered_samples and all(re.fullmatch(r"\d{6}", value) for value in rendered_samples[:10]):
        return "YYYYMM"

    if saw_decimal:
        return "decimal numeric string"
    if numeric_role in {"code", "indicator"}:
        return "integer code"
    return "integer count"


def _normalize_numeric_role_for_pattern(
    pandas_dtype: str,
    numeric_role: str | None,
    detected_pattern: str | None,
) -> str | None:
    if pandas_dtype not in {"Int64", "Float64"}:
        return None
    if pandas_dtype == "Float64":
        return "measure"

    normalized_pattern = (detected_pattern or "").strip().lower()
    if normalized_pattern == "integer count":
        return "measure"
    if numeric_role is not None:
        return numeric_role
    return "code"


def _normalize_dtype_inference_choice(column_name: str, dtype_col, col_profile):
    pandas_dtype = dtype_col.pandas_dtype if dtype_col else col_profile.pandas_dtype
    numeric_role = dtype_col.numeric_role if dtype_col else None
    string_role = dtype_col.string_role if dtype_col else None
    detected_pattern = dtype_col.detected_pattern if dtype_col else None
    rationale = dtype_col.rationale if dtype_col else ""

    numeric_parse_pct = col_profile.numeric_parse_pct
    datetime_parse_pct = col_profile.datetime_parse_pct

    if numeric_parse_pct >= 80 and datetime_parse_pct < 20:
        _, saw_decimal = _parse_numeric_samples(col_profile.sample_values)
        forced_dtype = "Float64" if saw_decimal else "Int64"
        forced_numeric_role = numeric_role if forced_dtype == "Float64" and numeric_role == "measure" else "code"
        forced_pattern = detected_pattern
        if forced_dtype == "Int64" and (_pattern_is_ambiguous(detected_pattern) or not detected_pattern):
            forced_pattern = _infer_numeric_pattern(column_name, col_profile.sample_values, forced_numeric_role)
        elif forced_dtype == "Float64" and (_pattern_is_ambiguous(detected_pattern) or not detected_pattern):
            forced_pattern = "decimal numeric string"
        forced_numeric_role = _normalize_numeric_role_for_pattern(forced_dtype, forced_numeric_role, forced_pattern)

        if pandas_dtype != forced_dtype or string_role is not None or _pattern_is_ambiguous(detected_pattern):
            rationale = (
                f"Forced to {forced_dtype} because numeric_parse_pct={numeric_parse_pct:.1f}% "
                f"and datetime_parse_pct={datetime_parse_pct:.1f}%. "
                "Above the hard numeric threshold, minority textual or malformed values are treated as corruption. "
                f"Canonical target pattern: {forced_pattern}."
            )

        return forced_dtype, forced_numeric_role, None, forced_pattern, rationale

    if pandas_dtype in {"Int64", "Float64"}:
        string_role = None
        if numeric_role is None:
            numeric_role = "measure" if pandas_dtype == "Float64" else "code"
        if _pattern_is_ambiguous(detected_pattern) or not detected_pattern:
            detected_pattern = (
                "decimal numeric string"
                if pandas_dtype == "Float64"
                else _infer_numeric_pattern(column_name, col_profile.sample_values, numeric_role)
            )
            rationale = (
                rationale + " " if rationale else ""
            ) + f"Normalized detected_pattern to the single dominant target format '{detected_pattern}'."
        numeric_role = _normalize_numeric_role_for_pattern(pandas_dtype, numeric_role, detected_pattern)

    return pandas_dtype, numeric_role, string_role, detected_pattern, rationale


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

    dtype_inference = run_dtype_inference(path)
    dtype_map = {col.column_name: col for col in dtype_inference.columns}
    dtype_overrides = {name: col.pandas_dtype for name, col in dtype_map.items()}

    profile = build_dataset_profile(df, path.stem, dtype_overrides=dtype_overrides)

    duplicate_groups_by_name: dict[str, list[str]] = {}
    for col_name in df.columns:
        canonical = normalized_schema_name(col_name)
        duplicate_groups_by_name.setdefault(canonical, []).append(col_name)
    duplicate_groups = [
        SchemaDuplicateGroup(canonical_name=cn, columns=cols)
        for cn, cols in duplicate_groups_by_name.items()
        if len(cols) > 1
    ]

    columns: list[SchemaColumnEntry] = []
    for col_profile in profile.columns_profiles:
        name = col_profile.column_name
        dtype_col = dtype_map.get(name)
        naming_valid = is_valid_schema_name(name)
        pandas_dtype, numeric_role, string_role, detected_pattern, rationale = _normalize_dtype_inference_choice(
            name,
            dtype_col,
            col_profile,
        )
        columns.append(SchemaColumnEntry(
            name=name,
            pandas_dtype=pandas_dtype,
            numeric_role=numeric_role,
            string_role=string_role,
            detected_pattern=detected_pattern,
            rationale=rationale,
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
