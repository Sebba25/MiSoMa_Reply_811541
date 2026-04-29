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

from src.core.cache import load_consistency, load_schema_handoff
from src.core.models import ConsistencyVerificationReport, FindingDiff
from src.validation import run_format_consistency_validation
from src.tools import load_dataset_frame

from .paths import cleaned_dataset_path


def _schema_rename_map(path: Path) -> dict[str, str]:
    '''Builds a map from original column names to their schema-suggested renamed versions.'''
    try:
        # Load the schema handoff, which may contain rename suggestions for invalid column names
        handoff = load_schema_handoff(path)
    except FileNotFoundError:
        # If no schema cache exists, there are no renames to align during verification
        return {}

    # Keep only columns that were invalid and actually received a rename suggestion
    return {
        column.name: column.rename_suggestion
        for column in handoff.columns
        if not column.naming_valid and column.rename_suggestion
    }
#It helps match original column names with cleaned column names during verification

def _numeric_original_names(cleaned_df: pd.DataFrame, reverse_rename: dict[str, str]) -> set[str]:
    '''Finds numeric columns in the cleaned dataset and maps them back to their original names.'''
    # Collect the current cleaned columns whose dtype is integer or float
    numeric_columns = {
        column
        for column in cleaned_df.columns
        #Keep the column only if its dtype is integer or float
        if pd.api.types.is_integer_dtype(cleaned_df[column]) or pd.api.types.is_float_dtype(cleaned_df[column])
    }
    # Convert cleaned column names back to original names so they align with pre-cleaning findings
    return {reverse_rename.get(column, column) for column in numeric_columns}
    #If it was not renamed, keep the same name
#After cleaning, some columns may become numeric, and those columns should be excluded from format-consistency comparison

def _print_diff_table(diffs: list[FindingDiff], original_count: int, remaining_count: int) -> None:
    '''Prints a readable before/after comparison table for consistency findings.'''
    # Print a high-level summary line showing how many findings existed before and after cleaning
    print(f"\n[verify] diff - {original_count} original findings -> {remaining_count} remaining\n", file=sys.stderr)
    # Size the table so the column-name field is wide enough for the longest column label
    column_width = max((len(diff.column_name) for diff in diffs), default=10)
    #This builds the header line of the table. The formatting makes each label align into a fixed-width column
    header = f"  {'COLUMN':<{column_width}}  {'STATUS':<10}  {'BEFORE':>8}  {'AFTER':>8}  {'REDUCTION':>10}"
    print(header, file=sys.stderr)
    #Print a separator line made of dashes
    print(f"  {'-' * column_width}  {'-' * 10}  {'-' * 8}  {'-' * 8}  {'-' * 10}", file=sys.stderr)

    # Print one row per diff, sorted by status so similar outcomes are grouped together
    for diff in sorted(diffs, key=lambda item: item.status):
        #diff.before_inconsistent_rows is the number of bad rows before cleaning
        #diff.after_inconsistent_rows is the number of bad rows after cleaning
        #diff.reduction_pct is the percentage reduction
        print(
            f"  {diff.column_name:<{column_width}}  {diff.status:<10}  {diff.before_inconsistent_rows:>8}  "
            f"{diff.after_inconsistent_rows:>8}  {diff.reduction_pct:>9.1f}%",
            file=sys.stderr,
        )
        # For findings that still exist, also show a few remaining problematic example values
        if diff.status in ("improved", "unchanged", "regressed", "new") and diff.remaining_examples:
            #Build a short sample string using up to 5 remaining example values
            sample = ", ".join(repr(value) for value in diff.remaining_examples[:5])
            #Print another line under that row showing the remaining problematic examples
            print(f"  {'':>{column_width}}  {'':10}  remaining examples: {sample}", file=sys.stderr)


def _diff_summary(diffs: list[FindingDiff]) -> str:
    '''Builds a short human-readable summary of how findings changed after cleaning.'''
    # Prepare a list of summary fragments that will be joined into one sentence
    summary_parts = []
    # Split the diffs into groups by status so they can be summarized clearly
    resolved = [diff for diff in diffs if diff.status == "resolved"]
    improved = [diff for diff in diffs if diff.status == "improved"]
    unchanged = [diff for diff in diffs if diff.status == "unchanged"]
    regressed = [diff for diff in diffs if diff.status == "regressed"]
    new = [diff for diff in diffs if diff.status == "new"]

    if resolved:
        # For resolved findings, include the renamed target when a column was renamed
        resolved_names = [
            #If the column was renamed, show the rename as old→new. Otherwise show only the original column name.
            f"{diff.column_name}→{diff.renamed_to}" if diff.renamed_to else diff.column_name
            for diff in resolved
        ]
        #Add a summary fragment describing how many findings were resolved and in which columns
        summary_parts.append(f"{len(resolved)} resolved ({', '.join(resolved_names)})")
    #Add a fragment saying how many improved/stayed unchanged/regressed and are new
    if improved:
        summary_parts.append(f"{len(improved)} improved")
    if unchanged:
        summary_parts.append(f"{len(unchanged)} unchanged")
    if regressed:
        summary_parts.append(f"{len(regressed)} regressed")
    if new:
        summary_parts.append(f"{len(new)} new findings introduced")

    # Return the combined summary, or a fallback message if no changes were recorded
    return "; ".join(summary_parts) if summary_parts else "No changes detected."
#It creates a compact human-readable summary for the final verification report

def run_verify(path: Path, on_event=None, max_workers: int = 1) -> ConsistencyVerificationReport:
    '''Re-runs consistency validation on the cleaned CSV and compares the results with the original findings.'''
    # Verification must use at least one worker
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")

    def _emit(message: str) -> None:
        '''Sends progress updates to the optional callback without breaking verification.'''
        if on_event is None:
            return
        try:
            on_event(message)
        except Exception:
            pass #ignore the error so verification continues normally

    # The cleaned dataset must already exist before verification can compare before vs after
    cleaned_path = cleaned_dataset_path(path)
    if not cleaned_path.exists():
        #If the cleaned CSV does not exist, raise an error telling the user to run the apply stage first
        raise FileNotFoundError(
            f"Cleaned dataset not found at {cleaned_path}. Run --stage apply first."
        )

    # Load the original consistency findings and index them by original column name
    original = load_consistency(path)
    original_map = {finding.column_name: finding for finding in original.format_consistency_findings}

    # Build rename maps so cleaned columns can be aligned with original column names
    rename_map = _schema_rename_map(path)
    #Build the reverse mapping: cleaned/renamed name -> original name. This is useful because after cleaning a column may have a different name than before.
    reverse_rename = {new: old for old, new in rename_map.items()}

    # Load the cleaned dataset and identify columns that became numeric after cleaning, then map them back to original names
    cleaned_df = load_dataset_frame(cleaned_path)
    numeric_original_names = _numeric_original_names(cleaned_df, reverse_rename)

    print(f"\n[verify] running consistency on cleaned dataset: {cleaned_path}", file=sys.stderr)
    _emit(f"re-running consistency on {len(original_map)} original finding columns")
    # Re-run consistency validation on the cleaned CSV without using cached results
    after = run_format_consistency_validation(
        cleaned_path,
        reuse_cache=False,
        read_as_str=True, #means read values as strings for consistency analysis
        max_workers=max_workers, #controls parallelism
        schema_path=path,
    )
    # Map the new findings back to original column names and ignore columns that are now numeric,
    # because format-consistency findings on numeric columns are no longer meaningful here
    after_map = {
        reverse_rename.get(finding.column_name, finding.column_name): finding
        for finding in after.format_consistency_findings
        if reverse_rename.get(finding.column_name, finding.column_name) not in numeric_original_names
    }
    #if a column became numeric after cleaning, do not include its format-consistency finding here.

    # Build one diff entry per original finding to show whether cleaning resolved or changed it
    #Start an empty list to store the comparison results
    diffs: list[FindingDiff] = []
    for column_name, before_finding in original_map.items():
        after_finding = after_map.get(column_name)
        before_rows = before_finding.inconsistent_rows
        after_rows = after_finding.inconsistent_rows if after_finding else 0
        # Compute the percentage reduction in inconsistent rows for this column
        reduction_pct = round((before_rows - after_rows) / before_rows * 100, 1) if before_rows > 0 else 0.0

        # Classify the outcome by comparing the number of inconsistent rows before and after cleaning
        if after_finding is None or after_rows == 0:
            status = "resolved"
        elif after_rows < before_rows:
            status = "improved"
        elif after_rows == before_rows:
            status = "unchanged"
        else:
            status = "regressed"

        # Save the diff result, including remaining examples and any schema rename
        diffs.append(
            FindingDiff(
                column_name=column_name,
                status=status,
                before_inconsistent_rows=before_rows,
                after_inconsistent_rows=after_rows,
                reduction_pct=reduction_pct,
                #If the finding still exists after cleaning, store its remaining inconsistent examples. Otherwise use an empty list
                remaining_examples=after_finding.example_inconsistent_values if after_finding else [],
                #Store the new name if this column was renamed
                renamed_to=rename_map.get(column_name),
            )
        )
        # Emit a short progress line using the renamed display name when available
        display_name = rename_map.get(column_name, column_name)
        _emit(f"  {status}: '{display_name}' ({before_rows}→{after_rows} rows)")

    # Add any findings that appear only after cleaning as brand-new issues
    for column_name, after_finding in after_map.items():
        #If the column already existed in the original findings, skip it, because it was already handled abov
        if column_name in original_map:
            continue
        diffs.append(
            FindingDiff(
                column_name=column_name,
                status="new", #Mark it as a new finding introduced after cleaning
                before_inconsistent_rows=0,
                #Store how many inconsistent rows it has after cleaning
                after_inconsistent_rows=after_finding.inconsistent_rows,
                #Use -100.0 to show this is not an improvement but a new issue
                reduction_pct=-100.0,
                #Store the problematic examples
                remaining_examples=after_finding.example_inconsistent_values,
            )
        )

    # Print a detailed table to stderr for CLI visibility.
    _print_diff_table(diffs, len(original_map), len(after_map))

    # Return the final verification report with the diffs and a short summary sentence
    return ConsistencyVerificationReport(
        dataset_name=path.stem,
        original_finding_count=len(original_map),
        remaining_finding_count=len(after_map),
        diffs=diffs, #Store the full list of per-column diff results
        summary=_diff_summary(diffs), 
    )
