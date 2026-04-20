"""Applies remediation actions + generated cleaners to produce the cleaned CSV.

Executes the remediation plan in order: column renames from schema,
placeholder-to-null replacements from completeness, per-column cleaner
programs, then the explicit dtype casts. Emits a ``CleaningReport`` with
one ``ColumnCleanerExecutionReport`` per applied cleaner.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

from cache import load_completeness, load_remediation_plan, load_schema_handoff, save_remediation_plan
from models import (
    CleaningReport,
    ColumnCleanerExecutionReport,
    ColumnCleanerProgram,
    GeneratedCleanerArtifact,
    RemediationAction,
    RemediationPlan,
)
from tools import gzip_text_to_base64, load_dataset_frame

from .paths import cleaned_dataset_path, cleaning_cache_dir, load_cleaner_manifest, save_cleaner_manifest
from .runtime import apply_cleaner_to_series


def _apply_column_renames(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    try:
        handoff = load_schema_handoff(path)
    except FileNotFoundError:
        print("[apply] schema cache not found - skipping column renames.", file=sys.stderr)
        return df, {}

    existing = set(df.columns)
    rename_map = {}
    for column in handoff.columns:
        if column.naming_valid or not column.rename_suggestion or column.name not in existing:
            continue

        target = column.rename_suggestion
        if target in existing and target != column.name:
            suffix = 2
            while f"{target}_{suffix}" in existing:
                suffix += 1
            target = f"{target}_{suffix}"
            print(f"  '{column.name}' -> '{target}' (suffixed - base name already exists)", file=sys.stderr)

        rename_map[column.name] = target
        existing.add(target)

    if rename_map:
        df = df.rename(columns=rename_map)
        for old, new in rename_map.items():
            print(f"  '{old}' -> '{new}'", file=sys.stderr)
    else:
        print("  no renames needed.", file=sys.stderr)

    return df, rename_map


def _apply_placeholder_nulls(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, int, dict[str, int]]:
    try:
        completeness = load_completeness(path)
    except FileNotFoundError:
        print("[apply] completeness cache not found - skipping placeholder replacement.", file=sys.stderr)
        return df, 0, {}

    total_replaced = 0
    replaced_by_column: dict[str, int] = {}
    for column_finding in completeness.per_column:
        column_name = column_finding.column_name
        if column_name not in df.columns or not column_finding.missing_like_examples:
            continue

        placeholders = {
            example.strip().lower()
            for example in column_finding.missing_like_examples
            if example.strip()
        }
        if not placeholders:
            continue

        mask = df[column_name].astype(str).str.strip().str.lower().isin(placeholders)
        count = int(mask.sum())
        if count > 0:
            df.loc[mask, column_name] = pd.NA
            total_replaced += count
            replaced_by_column[column_name] = count
            print(f"  '{column_name}': {count} placeholder values -> null", file=sys.stderr)

    return df, total_replaced, replaced_by_column


def _coerce_boolean(value: object) -> object:
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "yes", "si", "s\u00ec"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    return pd.NA


def _apply_dtype_casts(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    try:
        handoff = load_schema_handoff(path)
    except FileNotFoundError:
        print("  schema cache not found - skipping dtype casts.", file=sys.stderr)
        return df, {}

    rename_map = {
        column.name: column.rename_suggestion
        for column in handoff.columns
        if not column.naming_valid and column.rename_suggestion
    }
    schema_by_current: dict[str, Any] = {}
    for column in handoff.columns:
        current_name = rename_map.get(column.name, column.name)
        if current_name in df.columns:
            schema_by_current[current_name] = column

    cast_ok, cast_fail = 0, 0
    cast_results: dict[str, str] = {}
    for column_name, schema_column in schema_by_current.items():
        target = schema_column.pandas_dtype
        try:
            if target == "datetime64[ns]":
                df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
            elif target == "Int64":
                df[column_name] = pd.to_numeric(df[column_name], errors="coerce").astype("Int64")
            elif target == "Float64":
                numeric = pd.to_numeric(df[column_name], errors="coerce")
                non_null = numeric.dropna()
                if not non_null.empty and (non_null == non_null.round()).all():
                    df[column_name] = numeric.astype("Int64")
                    print(f"  '{column_name}' -> Int64 (auto-upgraded from Float64)", file=sys.stderr)
                else:
                    df[column_name] = numeric.astype("Float64")
                    print(f"  '{column_name}' -> Float64", file=sys.stderr)
                cast_ok += 1
                cast_results[column_name] = "applied"
                continue
            elif target == "boolean":
                df[column_name] = df[column_name].map(_coerce_boolean).astype("boolean")
            elif target == "string":
                df[column_name] = df[column_name].where(df[column_name].notna(), pd.NA).astype("string")
            else:
                cast_results[column_name] = "not_needed"
                continue

            print(f"  '{column_name}' -> {target}", file=sys.stderr)
            cast_ok += 1
            cast_results[column_name] = "applied"
        except Exception as error:
            print(f"  '{column_name}' cast to {target} SKIPPED: {error}", file=sys.stderr)
            cast_fail += 1
            cast_results[column_name] = "failed"

    print(f"  {cast_ok} columns cast successfully, {cast_fail} skipped.", file=sys.stderr)
    return df, cast_results


def _load_artifact_program(artifact: GeneratedCleanerArtifact) -> ColumnCleanerProgram | None:
    code_path = Path(artifact.code_path)
    if not code_path.exists():
        return None

    return ColumnCleanerProgram(
        column_name=artifact.column_name,
        function_name=artifact.function_name,
        python_code=code_path.read_text(encoding="utf-8"),
        example_transformations=[],
        verification_summary=artifact.summary,
    )


def _clone_remediation_plan(remediation_plan: RemediationPlan | None) -> RemediationPlan | None:
    if remediation_plan is None:
        return None
    return remediation_plan.model_copy(deep=True)


def _apply_exact_duplicate_column_drops(
    df: pd.DataFrame,
    actions: list[RemediationAction],
) -> tuple[pd.DataFrame, list[str]]:
    unresolved_risks: list[str] = []
    for action in actions:
        if action.action_type != "drop_exact_duplicate_column" or not action.auto_apply:
            continue

        keep_column = str(action.target.get("keep_column", ""))
        drop_column = str(action.target.get("drop_column", ""))
        if not keep_column or not drop_column or keep_column == drop_column:
            action.status = "not_needed"
            continue
        if drop_column not in df.columns:
            action.status = "not_needed"
            continue
        if keep_column not in df.columns:
            action.status = "failed"
            unresolved_risks.append(
                f"Could not drop duplicate column {drop_column!r} because keep column {keep_column!r} is missing."
            )
            continue

        df = df.drop(columns=[drop_column])
        action.status = "applied"
        print(f"  dropped exact duplicate column '{drop_column}' (kept '{keep_column}')", file=sys.stderr)

    return df, unresolved_risks


def run_cleaner_application_with_plan(
    path: Path,
    remediation_plan: RemediationPlan | None = None,
    on_event=None,
) -> tuple[CleaningReport, list[ColumnCleanerExecutionReport], RemediationPlan | None]:
    def _emit(message: str) -> None:
        if on_event is None:
            return
        try:
            on_event(message)
        except Exception:
            pass

    if remediation_plan is None:
        try:
            remediation_plan = load_remediation_plan(path)
        except FileNotFoundError:
            remediation_plan = None
    remediation_plan = _clone_remediation_plan(remediation_plan)
    actions = remediation_plan.actions if remediation_plan is not None else []

    try:
        artifacts = load_cleaner_manifest(path)
    except FileNotFoundError:
        artifacts = []
    df = load_dataset_frame(path)
    rows_before = len(df)
    columns_before = len(df.columns)
    execution_reports: list[ColumnCleanerExecutionReport] = []
    applied_artifacts: list[GeneratedCleanerArtifact] = []
    unresolved_risks: list[str] = []

    print(f"\n[apply] step 1 - format cleaners ({len(artifacts)} columns)", file=sys.stderr)
    _emit(f"step 1 — running {len(artifacts)} format cleaner{'s' if len(artifacts) != 1 else ''}")
    if not artifacts:
        print("  no cleaner manifest found or no generated cleaners recorded.", file=sys.stderr)
    for idx, artifact in enumerate(artifacts, start=1):
        program = _load_artifact_program(artifact)
        if program is None:
            unresolved_risks.append(f"{artifact.column_name}: cleaner file missing at {artifact.code_path}")
            continue
        if artifact.column_name not in df.columns:
            unresolved_risks.append(f"{artifact.column_name}: source column missing from dataset")
            continue

        cleaned_series, report = apply_cleaner_to_series(df[artifact.column_name], program)
        execution_reports.append(report)

        if report.execution_ok and cleaned_series is not None:
            df[artifact.column_name] = cleaned_series
            print(f"  '{artifact.column_name}' OK - {report.changed_rows} rows changed", file=sys.stderr)
            _emit(f"  [{idx}/{len(artifacts)}] ✓ '{artifact.column_name}' — {report.changed_rows} rows changed")
        else:
            print(f"  '{artifact.column_name}' FAILED - {'; '.join(report.unresolved_risks)}", file=sys.stderr)
            _emit(f"  [{idx}/{len(artifacts)}] ✗ '{artifact.column_name}' failed")
            unresolved_risks.extend(f"{artifact.column_name}: {risk}" for risk in report.unresolved_risks)

        applied_artifacts.append(
            GeneratedCleanerArtifact(
                column_name=artifact.column_name,
                function_name=artifact.function_name,
                code_path=artifact.code_path,
                changed_rows=report.changed_rows,
                summary=report.summary,
                example_transformations=list(artifact.example_transformations),
            )
        )

    execution_by_column = {report.column_name: report for report in execution_reports}
    for action in actions:
        if action.action_type != "generate_cleaner":
            continue
        report = execution_by_column.get(str(action.target.get("column_name", "")))
        if report is None:
            action.status = "failed"
        elif report.execution_ok:
            action.status = "applied"
        else:
            action.status = "failed"

    failed = [report for report in execution_reports if not report.execution_ok]
    print(
        f"\n  execution summary: {len(execution_reports) - len(failed)}/{len(execution_reports)} succeeded"
        + (f", {len(failed)} FAILED: {[report.column_name for report in failed]}" if failed else ""),
        file=sys.stderr,
    )

    print("\n[apply] step 2 - placeholder -> null (from completeness cache)", file=sys.stderr)
    _emit("step 2 — replacing placeholder tokens with null")
    df, total_replaced, placeholder_by_column = _apply_placeholder_nulls(df, path)
    print(f"  total placeholder replacements: {total_replaced}", file=sys.stderr)
    _emit(f"  {total_replaced} placeholder values replaced across {len(placeholder_by_column)} columns")
    for action in actions:
        if action.action_type != "replace_placeholders_with_null":
            continue
        column_name = str(action.target.get("column_name", ""))
        action.status = "applied" if placeholder_by_column.get(column_name, 0) > 0 else "not_needed"

    print("\n[apply] step 3 - exact duplicate column drops (from remediation plan)", file=sys.stderr)
    _emit("step 3 — dropping exact duplicate columns")
    if actions:
        df, drop_risks = _apply_exact_duplicate_column_drops(df, actions)
        unresolved_risks.extend(drop_risks)
    else:
        print("  no remediation plan loaded - skipping exact duplicate column drops.", file=sys.stderr)

    print("\n[apply] step 4 - column renames (from schema cache)", file=sys.stderr)
    _emit("step 4 — applying schema renames")
    df, rename_map = _apply_column_renames(df, path)
    if rename_map:
        _emit(f"  renamed {len(rename_map)} columns")
    for action in actions:
        if action.action_type != "rename_column":
            continue
        column_name = str(action.target.get("column_name", ""))
        new_name = str(action.target.get("new_name", ""))
        if rename_map.get(column_name) == new_name:
            action.status = "applied"
        elif column_name not in df.columns:
            action.status = "not_needed"
        else:
            action.status = "not_needed"

    print("\n[apply] step 5 - dtype casting (from schema cache)", file=sys.stderr)
    _emit("step 5 — casting dtypes")
    df, cast_results = _apply_dtype_casts(df, path)
    _applied_casts = sum(1 for s in cast_results.values() if s == "applied")
    if _applied_casts:
        _emit(f"  {_applied_casts} dtype cast{'s' if _applied_casts != 1 else ''} applied")
    for action in actions:
        if action.action_type != "cast_dtype":
            continue
        column_name = str(action.target.get("column_name", ""))
        status = cast_results.get(column_name, "not_needed")
        action.status = status if status in {"applied", "failed"} else "not_needed"

    output_dir = cleaning_cache_dir(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = cleaned_dataset_path(path)
    df.to_csv(cleaned_path, index=False)
    print(f"\n[apply] cleaned dataset saved -> {cleaned_path}", file=sys.stderr)

    save_cleaner_manifest(path, applied_artifacts)
    if remediation_plan is not None:
        save_remediation_plan(path, remediation_plan)

    cleaning_report = CleaningReport(
        dataset_name=path.stem,
        rows_before=rows_before,
        rows_after=len(df),
        columns_before=columns_before,
        columns_after=len(df.columns),
        generated_cleaners=applied_artifacts,
        unresolved_risks=unresolved_risks,
        cleaned_csv_gzip_base64=gzip_text_to_base64(df.to_csv(index=False)),
        summary=(
            f"Applied {len(applied_artifacts)} format cleaners, replaced {total_replaced} placeholder values, "
            f"renamed {len(rename_map)} columns, and cast dtypes. Cleaned dataset saved to `{cleaned_path.as_posix()}`."
        ),
    )
    return cleaning_report, execution_reports, remediation_plan


def run_cleaner_application(path: Path, on_event=None) -> CleaningReport:
    cleaning_report, _, _ = run_cleaner_application_with_plan(path, on_event=on_event)
    return cleaning_report
