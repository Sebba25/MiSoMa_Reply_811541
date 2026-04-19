from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

from cache import load_completeness, load_schema_handoff
from models import CleaningReport, ColumnCleanerExecutionReport, ColumnCleanerProgram, GeneratedCleanerArtifact
from tools.tools import gzip_text_to_base64, load_dataset_frame

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


def _apply_placeholder_nulls(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, int]:
    try:
        completeness = load_completeness(path)
    except FileNotFoundError:
        print("[apply] completeness cache not found - skipping placeholder replacement.", file=sys.stderr)
        return df, 0

    total_replaced = 0
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
            print(f"  '{column_name}': {count} placeholder values -> null", file=sys.stderr)

    return df, total_replaced


def _coerce_boolean(value: object) -> object:
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "yes", "si", "s\u00ec"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    return pd.NA


def _apply_dtype_casts(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    try:
        handoff = load_schema_handoff(path)
    except FileNotFoundError:
        print("  schema cache not found - skipping dtype casts.", file=sys.stderr)
        return df

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
                continue
            elif target == "boolean":
                df[column_name] = df[column_name].map(_coerce_boolean).astype("boolean")
            elif target == "string":
                df[column_name] = df[column_name].where(df[column_name].notna(), pd.NA).astype("string")
            else:
                continue

            print(f"  '{column_name}' -> {target}", file=sys.stderr)
            cast_ok += 1
        except Exception as error:
            print(f"  '{column_name}' cast to {target} SKIPPED: {error}", file=sys.stderr)
            cast_fail += 1

    print(f"  {cast_ok} columns cast successfully, {cast_fail} skipped.", file=sys.stderr)
    return df


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


def run_cleaner_application(path: Path) -> CleaningReport:
    artifacts = load_cleaner_manifest(path)
    df = load_dataset_frame(path)
    rows_before = len(df)
    columns_before = len(df.columns)
    execution_reports: list[ColumnCleanerExecutionReport] = []
    applied_artifacts: list[GeneratedCleanerArtifact] = []
    unresolved_risks: list[str] = []

    print(f"\n[apply] step 1 - format cleaners ({len(artifacts)} columns)", file=sys.stderr)
    for artifact in artifacts:
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
        else:
            print(f"  '{artifact.column_name}' FAILED - {'; '.join(report.unresolved_risks)}", file=sys.stderr)
            unresolved_risks.extend(f"{artifact.column_name}: {risk}" for risk in report.unresolved_risks)

        applied_artifacts.append(
            GeneratedCleanerArtifact(
                column_name=artifact.column_name,
                function_name=artifact.function_name,
                code_path=artifact.code_path,
                changed_rows=report.changed_rows,
                summary=report.summary,
            )
        )

    failed = [report for report in execution_reports if not report.execution_ok]
    print(
        f"\n  execution summary: {len(execution_reports) - len(failed)}/{len(execution_reports)} succeeded"
        + (f", {len(failed)} FAILED: {[report.column_name for report in failed]}" if failed else ""),
        file=sys.stderr,
    )

    print("\n[apply] step 2 - placeholder -> null (from completeness cache)", file=sys.stderr)
    df, total_replaced = _apply_placeholder_nulls(df, path)
    print(f"  total placeholder replacements: {total_replaced}", file=sys.stderr)

    print("\n[apply] step 3 - column renames (from schema cache)", file=sys.stderr)
    df, rename_map = _apply_column_renames(df, path)

    print("\n[apply] step 4 - dtype casting (from schema cache)", file=sys.stderr)
    df = _apply_dtype_casts(df, path)

    output_dir = cleaning_cache_dir(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = cleaned_dataset_path(path)
    df.to_csv(cleaned_path, index=False)
    print(f"\n[apply] cleaned dataset saved -> {cleaned_path}", file=sys.stderr)

    save_cleaner_manifest(path, applied_artifacts)

    return CleaningReport(
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
            f"renamed {len(rename_map)} columns, and cast dtypes. Cleaned dataset saved to {cleaned_path}."
        ),
    )

