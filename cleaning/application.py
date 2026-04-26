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
    '''Applies schema-based column renames and returns the updated dataframe and rename map.'''
    try:
        # Load the schema handoff, which may contain rename suggestions for invalid column names
        handoff = load_schema_handoff(path)
    except FileNotFoundError:
        # If the schema cache is missing, this step cannot rename anything
        print("[apply] schema cache not found - skipping column renames.", file=sys.stderr)
        #Return the original dataframe and an empty rename map
        return df, {}

    # Track current column names so new names do not collide with existing ones
    existing = set(df.columns)
    #Create an empty dictionary to store planned renames
    rename_map = {}
    #Loop through each column definition from the schema handoff
    for column in handoff.columns:
        # Skip columns that are already valid, have no suggestion, or do not exist in the dataframe
        if column.naming_valid or not column.rename_suggestion or column.name not in existing:
            continue #Move to the next schema column

        # Start from the suggested target name from schema validation
        target = column.rename_suggestion
        if target in existing and target != column.name:
            # If the target name already exists, add a numeric suffix until a unique name is found
            suffix = 2
            while f"{target}_{suffix}" in existing:
                suffix += 1
            target = f"{target}_{suffix}"
            #Log that a suffixed rename was needed because the base name already existed
            print(f"  '{column.name}' -> '{target}' (suffixed - base name already exists)", file=sys.stderr)
        #Save the rename mapping from old name to new name
        rename_map[column.name] = target
        #Add the new target name to the set so later renames do not collide with it
        existing.add(target)
    #Check whether at least one rename was collected
    if rename_map:
        # Apply all collected renames in one dataframe operation
        df = df.rename(columns=rename_map)
        for old, new in rename_map.items():
            print(f"  '{old}' -> '{new}'", file=sys.stderr) #Print a log line for the rename
    #f no rename was needed
    else:
        # Report that no rename action was needed
        print("  no renames needed.", file=sys.stderr)

    return df, rename_map
#It applies the naming corrections suggested by schema validation

def _apply_placeholder_nulls(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, int, dict[str, int]]:
    '''Replaces placeholder-like missing tokens with real null values using completeness findings.'''
    try:
        # Load completeness analysis, which stores placeholder examples such as "N/A" or "unknown"
        completeness = load_completeness(path)
    except FileNotFoundError:
        # Without completeness cache, this normalization step must be skipped
        print("[apply] completeness cache not found - skipping placeholder replacement.", file=sys.stderr)
        return df, 0, {}

    # Keep both a total replacement count and a per-column breakdown
    total_replaced = 0
    #Create a dictionary to store replacement counts by column
    replaced_by_column: dict[str, int] = {}
    #Loop through completeness findings for each column
    for column_finding in completeness.per_column:
        column_name = column_finding.column_name #Get the current column name
        # Skip columns that do not exist anymore or have no placeholder examples to use
        if column_name not in df.columns or not column_finding.missing_like_examples:
            continue

        # Normalize the placeholder examples so matching is case-insensitive and whitespace-insensitive
        placeholders = {
            example.strip().lower()
            for example in column_finding.missing_like_examples
            if example.strip() #Keep only non-empty examples
        }
        #If the placeholder set is empty, skip this column and go to the next
        if not placeholders:
            continue

        # Find cells whose normalized text matches one of the detected placeholder tokens
        mask = df[column_name].astype(str).str.strip().str.lower().isin(placeholders)
        #Count how many cells match
        count = int(mask.sum())
        #If at least one placeholder was found
        if count > 0:
            # Replace those placeholder values with pandas missing values
            df.loc[mask, column_name] = pd.NA
            total_replaced += count
            replaced_by_column[column_name] = count
            #Log how many placeholder values were replaced in that column
            print(f"  '{column_name}': {count} placeholder values -> null", file=sys.stderr)
    #Return the updated dataframe and replacement statistics
    return df, total_replaced, replaced_by_column
#It converts fake missing values like "N/A" or "unknown" into real null values

def _coerce_boolean(value: object) -> object:
    '''Converts common text-like boolean values into True, False, or missing.'''
    # Normalize the value first so comparisons are consistent
    normalized = str(value).strip().lower()
    #Check whether the normalized value represents True
    if normalized in ("true", "1", "yes", "si", "s\u00ec"):
        return True
    #Check whether the normalized value represents False
    if normalized in ("false", "0", "no"):
        return False
    #If the value is not recognized as true or false, return a missing value
    return pd.NA
#It helps convert text-like boolean values into proper boolean data

def _apply_dtype_casts(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict[str, str]]: 
    #it returns the updated dataframe and a dictionary describing the cast result for each column
    '''Casts dataframe columns to the schema-inferred pandas dtypes and reports the result per column.'''
    try:
        # Load schema metadata that contains the target pandas dtype for each column
        handoff = load_schema_handoff(path)
    except FileNotFoundError:
        # If schema information is unavailable, dtype casting cannot be performed safely, print a skip message
        print("  schema cache not found - skipping dtype casts.", file=sys.stderr)
        return df, {}

    # Rebuild the rename map so schema entries can be matched to the current dataframe column names
    rename_map = {
        #Map each original name to its suggested name
        column.name: column.rename_suggestion
        for column in handoff.columns
        #Keep only renamed columns
        if not column.naming_valid and column.rename_suggestion
    }
    #Create a dictionary that will map current dataframe column names to schema entries
    schema_by_current: dict[str, Any] = {}
    for column in handoff.columns:
        # Use the renamed column name if a rename was planned earlier
        current_name = rename_map.get(column.name, column.name)
        #Check whether the current column exists in the dataframe
        if current_name in df.columns:
            #Save the schema entry under the current dataframe column name
            schema_by_current[current_name] = column

    # Track successful and failed casts and keep a per-column status dictionary
   #Start counters for successful and failed casts
    cast_ok, cast_fail = 0, 0
    #Create a dictionary to store the cast status per column
    cast_results: dict[str, str] = {}
    #Loop through the dataframe columns that have schema information
    for column_name, schema_column in schema_by_current.items():
        target = schema_column.pandas_dtype
        try:
            if target == "datetime64[ns]":
                # Convert values to datetime; invalid values become missing
                df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
            elif target == "Int64":
                # Convert values to nullable integer dtype through numeric parsing
                df[column_name] = pd.to_numeric(df[column_name], errors="coerce").astype("Int64")
            elif target == "Float64":
                # Parse numeric values first, then check whether they are actually all whole numbers
                numeric = pd.to_numeric(df[column_name], errors="coerce")
                non_null = numeric.dropna()
                #Check whether all non-missing values are whole numbers
                if not non_null.empty and (non_null == non_null.round()).all():
                    # If every valid value is integer-like, upgrade to Int64 for cleaner typing
                    df[column_name] = numeric.astype("Int64")
                    #Log that auto-upgrade
                    print(f"  '{column_name}' -> Int64 (auto-upgraded from Float64)", file=sys.stderr)
                else:
                    # Otherwise keep the intended Float64 target
                    df[column_name] = numeric.astype("Float64")
                    print(f"  '{column_name}' -> Float64", file=sys.stderr)
                cast_ok += 1
                #Mark this cast as applied
                cast_results[column_name] = "applied"
                #Skip the rest of the loop body because this branch already finished its work
                continue
            elif target == "boolean":
                # Normalize values through the boolean coercion helper and use pandas nullable boolean type
                df[column_name] = df[column_name].map(_coerce_boolean).astype("boolean")
            elif target == "string":
                # Keep missing values as pd.NA and cast the rest to pandas string dtype
                df[column_name] = df[column_name].where(df[column_name].notna(), pd.NA).astype("string")
            else:
                # Skip dtypes that are not handled in this application step
                cast_results[column_name] = "not_needed"
                continue
            #Log the successful cast for the handled dtypes other than Float64 special case
            print(f"  '{column_name}' -> {target}", file=sys.stderr)
            cast_ok += 1
            cast_results[column_name] = "applied"
        except Exception as error:
            # Record the failure but keep the overall cleaning process running
            print(f"  '{column_name}' cast to {target} SKIPPED: {error}", file=sys.stderr)
            cast_fail += 1
            cast_results[column_name] = "failed"
    #Print a summary of cast success/failure counts
    print(f"  {cast_ok} columns cast successfully, {cast_fail} skipped.", file=sys.stderr)
    #Return the updated dataframe and cast results
    return df, cast_results


def _apply_string_lowercasing(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, int, dict[str, int]]:
    #It returns the updated dataframe, the total number of changed values and a dictionary of changed counts by column
    '''Lowercases values in schema-declared string columns and reports how many changed.'''
    try:
        # Load schema information so only intended string columns are normalized
        handoff = load_schema_handoff(path)
    except FileNotFoundError:
        # If schema metadata is missing, this lowercasing step is skipped
        print("  schema cache not found - skipping string lowercasing.", file=sys.stderr)
        return df, 0, {}

    # Rebuild the rename map so the function works with the dataframe's current column names
    rename_map = {
        #Map original names to renamed versions
        column.name: column.rename_suggestion
        for column in handoff.columns
        #Keep only renamed columns
        if not column.naming_valid and column.rename_suggestion
    }
    #Start building the list of string columns
    string_columns = [
        #Use the current column name after rename if needed
        rename_map.get(column.name, column.name)
        for column in handoff.columns
        #Keep only columns whose target dtype is string
        if column.pandas_dtype == "string"
    ]

    # Track total changed values and a per-column count for reporting
    total_changed = 0
    changed_by_column: dict[str, int] = {}
    for column_name in string_columns:
        #Skip columns that are not in the dataframe
        if column_name not in df.columns:
            continue
        #Get the current column series
        series = df[column_name]
        # Create a lowercased version of the series 
        lowered = series.str.lower()
        # Count only values that were non-null and actually changed after lowercasing
        changed_mask = series.notna() & lowered.notna() & series.ne(lowered)
        changed_count = int(changed_mask.sum())
        #if nothing changes skip the column
        if changed_count <= 0:
            continue

        # Replace the column with the normalized lowercased version
        df[column_name] = lowered
        total_changed += changed_count
        #Store the count for this column
        changed_by_column[column_name] = changed_count
        #Log how many values were lowercased
        print(f"  '{column_name}': {changed_count} string values lowercased", file=sys.stderr)
    #Return the updated dataframe and stats
    return df, total_changed, changed_by_column


def _load_artifact_program(artifact: GeneratedCleanerArtifact) -> ColumnCleanerProgram | None:
    '''Loads a saved cleaner file from disk and rebuilds a ColumnCleanerProgram object.'''
    # Convert the saved code path into a Path object
    code_path = Path(artifact.code_path)
    if not code_path.exists():
        # If the code file is missing, the cleaner cannot be executed
        return None

    # Rebuild the cleaner program object using the saved code and artifact metadata
    return ColumnCleanerProgram(
        column_name=artifact.column_name,
        function_name=artifact.function_name,
        python_code=code_path.read_text(encoding="utf-8"), #Read the Python code from disk
        example_transformations=[],
        verification_summary=artifact.summary, #Reuse the artifact summary as verification summary
    )


def _clone_remediation_plan(remediation_plan: RemediationPlan | None) -> RemediationPlan | None:
    '''Creates a deep copy of the remediation plan so action statuses can be updated safely.'''
    if remediation_plan is None:
        return None
    return remediation_plan.model_copy(deep=True)
#It prevents accidental modification of the original plan object while applying actions


def _apply_exact_duplicate_column_drops(
    df: pd.DataFrame,
    actions: list[RemediationAction], #List of remediation actions
) -> tuple[pd.DataFrame, list[str]]: #returns the updated datframe and a list of unresolved risk messages
    '''Drops exact duplicate columns for auto-apply remediation actions and returns any unresolved risks.'''
    unresolved_risks: list[str] = [] #Start an empty list of risks
    for action in actions:
        #Skip actions that are not exact-duplicate-column drops or are not auto-applicable
        if action.action_type != "drop_exact_duplicate_column" or not action.auto_apply:
            continue

        # Read the keep/drop column names from the remediation action target payload
        keep_column = str(action.target.get("keep_column", ""))
        drop_column = str(action.target.get("drop_column", ""))
        if not keep_column or not drop_column or keep_column == drop_column:
            # Malformed or pointless actions are marked as not needed
            action.status = "not_needed"
            continue
        if drop_column not in df.columns:
            # If the duplicate column is already absent, nothing needs to be done
            action.status = "not_needed"
            continue
        if keep_column not in df.columns:
            # If the keep column is missing, do not drop anything and record a risk
            action.status = "failed"
            unresolved_risks.append(
                f"Could not drop duplicate column {drop_column!r} because keep column {keep_column!r} is missing."
            )
            continue

        # Drop the duplicate column and mark the remediation action as applied
        df = df.drop(columns=[drop_column])
        action.status = "applied"
        print(f"  dropped exact duplicate column '{drop_column}' (kept '{keep_column}')", file=sys.stderr)
    #Return the updated dataframe and any unresolved risks
    return df, unresolved_risks


def run_cleaner_application_with_plan(
    path: Path,
    remediation_plan: RemediationPlan | None = None,
    on_event=None,
) -> tuple[CleaningReport, list[ColumnCleanerExecutionReport], RemediationPlan | None]:
    '''Applies generated cleaners and remediation steps, saves the cleaned CSV, and returns the final report.'''
    #it returns the cleaning report, a list of execution reports and the updated remediation plan
    def _emit(message: str) -> None:
        '''Sends progress updates to the optional callback without interrupting the main workflow.'''
        #If no callback exists, do nothing
        if on_event is None:
            return
        #Try calling the callback and send the message
        try:
            on_event(message)
        #If callback fails, ignore it
        except Exception:
            pass

    # Load the remediation plan from cache if the caller did not provide one
    if remediation_plan is None:
        remediation_plan = load_remediation_plan(path)
    # Clone the plan so action statuses can be updated without mutating a shared object
    remediation_plan = _clone_remediation_plan(remediation_plan)
    #Get the plan’s action list, or use an empty list if there is no plan
    actions = remediation_plan.actions if remediation_plan is not None else []

    # Load the generated cleaner manifest if it exists
    artifacts = load_cleaner_manifest(path)
    # Load the raw dataset and record the initial shape for the final report
    df = load_dataset_frame(path)
    #the rows and columns before cleaning are recorded
    rows_before = len(df)
    columns_before = len(df.columns)
    #Start a list for cleaner execution reports and applied artificats and unresolved risks
    execution_reports: list[ColumnCleanerExecutionReport] = []
    applied_artifacts: list[GeneratedCleanerArtifact] = []
    unresolved_risks: list[str] = []

    #Log step 1
    print(f"\n[apply] step 1 - format cleaners ({len(artifacts)} columns)", file=sys.stderr)
    #Send a short progress update
    _emit(f"step 1 — running {len(artifacts)} format cleaner{'s' if len(artifacts) != 1 else ''}")
    #If no generated cleaners were found, Log that step 1 has nothing to do
    if not artifacts:
        print("  no cleaner manifest found or no generated cleaners recorded.", file=sys.stderr)
    #Loop through each generated cleaner artifact
    for idx, artifact in enumerate(artifacts, start=1):
        # Load the cleaner code saved for this artifact
        #Try to reconstruct the cleaner program from disk
        program = _load_artifact_program(artifact)
        #If the code file is missing,
        if program is None:
            #Add a risk message
            unresolved_risks.append(f"{artifact.column_name}: cleaner file missing at {artifact.code_path}")
            continue #and skip the artifact
        #If the source column is missing from the dataset,
        if artifact.column_name not in df.columns:
            #add a risk message and skip the artifact
            unresolved_risks.append(f"{artifact.column_name}: source column missing from dataset")
            continue

        # Execute the cleaner against the dataframe column and collect an execution report
        cleaned_series, report = apply_cleaner_to_series(df[artifact.column_name], program)
        execution_reports.append(report)
        
        #If the cleaner ran successfully and returned a cleaned series
        if report.execution_ok and cleaned_series is not None:
            # Replace the original column with the cleaned output series when execution succeeds
            df[artifact.column_name] = cleaned_series
            print(f"  '{artifact.column_name}' OK - {report.changed_rows} rows changed", file=sys.stderr)
            _emit(f"  [{idx}/{len(artifacts)}] ✓ '{artifact.column_name}' — {report.changed_rows} rows changed")
        else:
            # Keep the original column unchanged and store the reported risks
            print(f"  '{artifact.column_name}' FAILED - {'; '.join(report.unresolved_risks)}", file=sys.stderr)
            _emit(f"  [{idx}/{len(artifacts)}] ✗ '{artifact.column_name}' failed")
            #Add all cleaner failure risks to the global unresolved risks list
            unresolved_risks.extend(f"{artifact.column_name}: {risk}" for risk in report.unresolved_risks)

        # Create a new artifact that records the execution results of this cleaner, including changed row count and summary
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

    # Map execution reports by column name so remediation actions can be updated later
    execution_by_column = {report.column_name: report for report in execution_reports}
    for action in actions:
        #Skip non-cleaner actions
        if action.action_type != "generate_cleaner":
            continue
        #Look up the execution report for the target column
        report = execution_by_column.get(str(action.target.get("column_name", "")))
        if report is None:
            action.status = "failed"
        elif report.execution_ok:
            action.status = "applied"
        else:
            action.status = "failed"

    # Print a short summary of cleaner execution success and failure counts
    failed = [report for report in execution_reports if not report.execution_ok]
    #Print a summary saying how many cleaner executions succeeded and which ones failed
    print(
        f"\n  execution summary: {len(execution_reports) - len(failed)}/{len(execution_reports)} succeeded"
        + (f", {len(failed)} FAILED: {[report.column_name for report in failed]}" if failed else ""),
        file=sys.stderr,
    )

    #Start step 2 logging
    print("\n[apply] step 2 - placeholder -> null (from completeness cache)", file=sys.stderr)
    _emit("step 2 — replacing placeholder tokens with null")
    # Replace placeholder-like tokens such as "n/a" with proper missing values
    df, total_replaced, placeholder_by_column = _apply_placeholder_nulls(df, path)
    print(f"  total placeholder replacements: {total_replaced}", file=sys.stderr)
    _emit(f"  {total_replaced} placeholder values replaced across {len(placeholder_by_column)} columns")
    #Loop through remediation actions again
    for action in actions:
        if action.action_type != "replace_placeholders_with_null":
            continue # Skip unrelated actions
        column_name = str(action.target.get("column_name", ""))
        #mark the action as applied if at least one placeholder was replaced otheriwse not needed
        action.status = "applied" if placeholder_by_column.get(column_name, 0) > 0 else "not_needed"

    #Start step 3 logging
    print("\n[apply] step 3 - exact duplicate column drops (from remediation plan)", file=sys.stderr)
    _emit("step 3 — dropping exact duplicate columns")
    #If there is a remediation plan with actions,Apply the duplicate-column drops
    if actions:
        # Apply auto-approved duplicate column drops from the remediation plan
        df, drop_risks = _apply_exact_duplicate_column_drops(df, actions)
        unresolved_risks.extend(drop_risks) #Add any new risk messages
    else: #if there is no plan log the skip
        print("  no remediation plan loaded - skipping exact duplicate column drops.", file=sys.stderr)

    #Start step 4 logging
    print("\n[apply] step 4 - column renames (from schema cache)", file=sys.stderr)
    _emit("step 4 — applying schema renames")
    # Rename columns according to schema-based naming suggestions
    df, rename_map = _apply_column_renames(df, path)
    if rename_map:
        _emit(f"  renamed {len(rename_map)} columns")
    #Loop through actions
    for action in actions:
        #skip unrelated actions
        if action.action_type != "rename_column":
            continue
        #Read old column name
        column_name = str(action.target.get("column_name", ""))
        #Read expected new name
        new_name = str(action.target.get("new_name", ""))
        #If the actual rename map matches the action, mark the action as applied
        if rename_map.get(column_name) == new_name:
            action.status = "applied"
        #If the old column is no longer in the dataframe,mark the action as not needed
        elif column_name not in df.columns:
            action.status = "not_needed"
        #otherwise, mark as not needed
        else:
            action.status = "not_needed"

    #Start step 5 logging
    print("\n[apply] step 5 - dtype casting (from schema cache)", file=sys.stderr)
    _emit("step 5 — casting dtypes")
    # Cast columns into their target dtypes using schema metadata
    df, cast_results = _apply_dtype_casts(df, path)
    #Count how many casts were actually applied
    _applied_casts = sum(1 for s in cast_results.values() if s == "applied")
    if _applied_casts: #if at least one was applied, send update
        _emit(f"  {_applied_casts} dtype cast{'s' if _applied_casts != 1 else ''} applied")
    for action in actions:
        if action.action_type != "cast_dtype": #Skip unrelated actions
            continue
        column_name = str(action.target.get("column_name", ""))
        #Look up the cast result, defaulting to not needed if the column was not in the dataframe or had no schema info
        status = cast_results.get(column_name, "not_needed")
        #Set the action status accordingly
        action.status = status if status in {"applied", "failed"} else "not_needed"

    #Start step 6 logging
    print("\n[apply] step 6 - lowercase string columns", file=sys.stderr)
    _emit("step 6 — lowercasing string columns")
    # Lowercase string columns as a final text-normalization pass
    df, lowercased_total, lowercased_by_column = _apply_string_lowercasing(df, path)
    if lowercased_total:
        #Send a short lowercasing summary if it was applied
        _emit(
            f"  lowercased {lowercased_total} string value{'s' if lowercased_total != 1 else ''} "
            f"across {len(lowercased_by_column)} column{'s' if len(lowercased_by_column) != 1 else ''}"
        )

    # Save the final cleaned dataframe to the cleaning cache directory
    output_dir = cleaning_cache_dir(path) #Get the cleaning cache directory
    output_dir.mkdir(parents=True, exist_ok=True) #Ensure the output directory exists
    cleaned_path = cleaned_dataset_path(path) #Build the output path of the cleaned CSV
    df.to_csv(cleaned_path, index=False) #Save the cleaned dataframe as CSV
    #Log where the cleaned dataset was saved
    print(f"\n[apply] cleaned dataset saved -> {cleaned_path}", file=sys.stderr)

    save_cleaner_manifest(path, applied_artifacts) #Save the updated cleaner artifact manifest
    if remediation_plan is not None:
        #Save the updated remediation plan with new action statuses
        save_remediation_plan(path, remediation_plan)

    # Build the final cleaning report with dataset shape changes, unresolved risks, and a compressed CSV snapshot
    cleaning_report = CleaningReport( #Create the final cleaning report object
        dataset_name=path.stem,
        rows_before=rows_before,
        rows_after=len(df),
        columns_before=columns_before,
        columns_after=len(df.columns),
        generated_cleaners=applied_artifacts,
        unresolved_risks=unresolved_risks,
        #Store the cleaned CSV content compressed and encoded as base64
        cleaned_csv_gzip_base64=gzip_text_to_base64(df.to_csv(index=False)),
        summary=(
            f"Applied {len(applied_artifacts)} format cleaners, replaced {total_replaced} placeholder values, "
            f"renamed {len(rename_map)} columns, cast dtypes, and lowercased {lowercased_total} string values. "
            f"Cleaned dataset saved to `{cleaned_path.as_posix()}`."
        ),
    )
    return cleaning_report, execution_reports, remediation_plan


def run_cleaner_application(path: Path, on_event=None) -> CleaningReport:
    '''Runs the full application workflow and returns only the final cleaning report.'''
    # This is a small convenience wrapper around the more detailed application function
    cleaning_report, _, _ = run_cleaner_application_with_plan(path, on_event=on_event)
    #Call the full application function, but ignore execution reports and remediation plan
    #it returns only the final cleaning report
    return cleaning_report
#It gives a simpler public entry point when only the final cleaning report is needed