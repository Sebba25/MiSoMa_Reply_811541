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
