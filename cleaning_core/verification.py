"""Compares format-consistency findings before and after cleaning.

Re-runs ``run_format_consistency_validation`` on the cleaned CSV, aligns the
findings through the schema rename map, and emits a per-column diff
(``resolved`` / ``improved`` / ``unchanged`` / ``regressed`` / ``new``) for
the verification agent and the CLI to surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from cache import load_consistency, load_schema_handoff
from models import ConsistencyVerificationReport, FindingDiff
from pipeline import run_format_consistency_validation
from tools.tools import load_dataset_frame

from .paths import cleaned_dataset_path


def _schema_rename_map(path: Path) -> dict[str, str]:
    try:
        handoff = load_schema_handoff(path)
    except FileNotFoundError:
        return {}

    return {
        column.name: column.rename_suggestion
        for column in handoff.columns
        if not column.naming_valid and column.rename_suggestion
    }


def _numeric_original_names(cleaned_df: pd.DataFrame, reverse_rename: dict[str, str]) -> set[str]:
    numeric_columns = {
        column
        for column in cleaned_df.columns
        if pd.api.types.is_integer_dtype(cleaned_df[column]) or pd.api.types.is_float_dtype(cleaned_df[column])
    }
    return {reverse_rename.get(column, column) for column in numeric_columns}


def _print_diff_table(diffs: list[FindingDiff], original_count: int, remaining_count: int) -> None:
    print(f"\n[verify] diff - {original_count} original findings -> {remaining_count} remaining\n", file=sys.stderr)
    column_width = max((len(diff.column_name) for diff in diffs), default=10)
    header = f"  {'COLUMN':<{column_width}}  {'STATUS':<10}  {'BEFORE':>8}  {'AFTER':>8}  {'REDUCTION':>10}"
    print(header, file=sys.stderr)
    print(f"  {'-' * column_width}  {'-' * 10}  {'-' * 8}  {'-' * 8}  {'-' * 10}", file=sys.stderr)

    for diff in sorted(diffs, key=lambda item: item.status):
        print(
            f"  {diff.column_name:<{column_width}}  {diff.status:<10}  {diff.before_inconsistent_rows:>8}  "
            f"{diff.after_inconsistent_rows:>8}  {diff.reduction_pct:>9.1f}%",
            file=sys.stderr,
        )
        if diff.status in ("improved", "unchanged", "regressed", "new") and diff.remaining_examples:
            sample = ", ".join(repr(value) for value in diff.remaining_examples[:5])
            print(f"  {'':>{column_width}}  {'':10}  remaining examples: {sample}", file=sys.stderr)


def _diff_summary(diffs: list[FindingDiff]) -> str:
    summary_parts = []
    resolved = [diff for diff in diffs if diff.status == "resolved"]
    improved = [diff for diff in diffs if diff.status == "improved"]
    unchanged = [diff for diff in diffs if diff.status == "unchanged"]
    regressed = [diff for diff in diffs if diff.status == "regressed"]
    new = [diff for diff in diffs if diff.status == "new"]

    if resolved:
        summary_parts.append(f"{len(resolved)} resolved ({', '.join(diff.column_name for diff in resolved)})")
    if improved:
        summary_parts.append(f"{len(improved)} improved")
    if unchanged:
        summary_parts.append(f"{len(unchanged)} unchanged")
    if regressed:
        summary_parts.append(f"{len(regressed)} regressed")
    if new:
        summary_parts.append(f"{len(new)} new findings introduced")

    return "; ".join(summary_parts) if summary_parts else "No changes detected."


def run_verify(path: Path) -> ConsistencyVerificationReport:
    cleaned_path = cleaned_dataset_path(path)
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {cleaned_path}. Run --stage apply first."
        )

    original = load_consistency(path)
    original_map = {finding.column_name: finding for finding in original.format_consistency_findings}

    rename_map = _schema_rename_map(path)
    reverse_rename = {new: old for old, new in rename_map.items()}

    cleaned_df = load_dataset_frame(cleaned_path)
    numeric_original_names = _numeric_original_names(cleaned_df, reverse_rename)

    print(f"\n[verify] running consistency on cleaned dataset: {cleaned_path}", file=sys.stderr)
    after = run_format_consistency_validation(cleaned_path, reuse_cache=False, read_as_str=True)
    after_map = {
        reverse_rename.get(finding.column_name, finding.column_name): finding
        for finding in after.format_consistency_findings
        if reverse_rename.get(finding.column_name, finding.column_name) not in numeric_original_names
    }

    diffs: list[FindingDiff] = []
    for column_name, before_finding in original_map.items():
        after_finding = after_map.get(column_name)
        before_rows = before_finding.inconsistent_rows
        after_rows = after_finding.inconsistent_rows if after_finding else 0
        reduction_pct = round((before_rows - after_rows) / before_rows * 100, 1) if before_rows > 0 else 0.0

        if after_finding is None or after_rows == 0:
            status = "resolved"
        elif after_rows < before_rows:
            status = "improved"
        elif after_rows == before_rows:
            status = "unchanged"
        else:
            status = "regressed"

        diffs.append(
            FindingDiff(
                column_name=column_name,
                status=status,
                before_inconsistent_rows=before_rows,
                after_inconsistent_rows=after_rows,
                reduction_pct=reduction_pct,
                remaining_examples=after_finding.example_inconsistent_values if after_finding else [],
            )
        )

    for column_name, after_finding in after_map.items():
        if column_name in original_map:
            continue
        diffs.append(
            FindingDiff(
                column_name=column_name,
                status="new",
                before_inconsistent_rows=0,
                after_inconsistent_rows=after_finding.inconsistent_rows,
                reduction_pct=-100.0,
                remaining_examples=after_finding.example_inconsistent_values,
            )
        )

    _print_diff_table(diffs, len(original_map), len(after_map))

    return ConsistencyVerificationReport(
        dataset_name=path.stem,
        original_finding_count=len(original_map),
        remaining_finding_count=len(after_map),
        diffs=diffs,
        summary=_diff_summary(diffs),
    )

