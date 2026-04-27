"""schema.py (validation pipeline): dtype inference and schema validation.

This module exposes two public functions. run_dtype_inference builds a column-by-column
statistical profile and infers the target pandas dtype, semantic role, and dominant value
pattern for each column. run_schema_validation builds on that inference to detect naming
convention violations, identify duplicate columns via canonical name comparison, collect
all findings into SchemaIssue entries, and produce the final SchemaHandoff object that is
cached and passed to every downstream validation and cleaning stage.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

from core.agents import dtype_inference_agent, schema_summary_agent
from core.cache import load_schema_handoff, save_schema_handoff
from core.models import DatasetDtypeInference, SchemaColumnEntry, SchemaHandoff, SchemaIssue
from tools import (SchemaDuplicateGroup,
    attach_profile_text, attach_text_document,
    build_dataset_profile,build_dtype_inference_text,
    is_valid_schema_name, suggest_schema_name,
    load_dataset_frame,
    naming_rule_reason,
    normalized_schema_name,
    run_agent_with_backoff,
)

def _pattern_is_ambiguous(pattern: str | None) -> bool:
    """Return True if the pattern contains common ambiguous tokens suggesting it's not specific enough."""
    if not pattern:
        return False
    # Normalize the pattern to check for empty strings
    normalized = pattern.strip().lower()
    if not normalized:
        return False
    # Check for common ambiguous tokens that suggest the pattern is not specific enough
    ambiguous_tokens = {" / ", "/", " and ", " or ", "mixed ", "multiple ", "various ", "several "}
    return any(token in normalized for token in ambiguous_tokens)

def _parse_numeric_samples(sample_values: list[str]) -> tuple[list[int], bool]:
    """Parse raw sample strings into integers and flag whether any true decimals were found.

    Returns the list of successfully parsed integer values and a boolean that is True if at
    least one value had a non-zero fractional part. Used downstream to decide between Int64
    and Float64 and to infer the canonical numeric pattern.
    """
    ints = []
    saw_decimal = False
    for raw in sample_values:
        # Cast to string and strip whitespace to handle mixed-type columns safely
        value = str(raw).strip()
        if not value:
            continue
        # Replace the comma if the entire value is digits, a comma, then digits
        if re.fullmatch(r"-?\d+,\d+", value):
            value = value.replace(",", ".")
        try:
            number = float(value)
            if number.is_integer():
                ints.append(int(number))
            else:
                saw_decimal = True
        except ValueError:
            continue
    return ints, saw_decimal


def _infer_year_width_pattern(column_name: str, sample_values: list[str]) -> str | None:
    """Infer a fixed-width year pattern for anno/year columns from predominant samples.

    Year-like columns often contain a mixture of canonical values (for example
    ``2023``) and abbreviated variants (for example ``23``). For these columns
    we want the schema contract to follow the predominant width instead of
    falling back to a generic ``integer code`` label that would accept both.
    """
    # Only anno/year-like columns are eligible for this stricter year-width inference.
    tokens = set(normalized_schema_name(column_name).split("_"))
    if not ({"anno", "year"} & tokens):
        return None

    # Count how often 2-digit and 4-digit year tokens appear across the samples.
    width_counts: dict[int, int] = {}
    for raw in sample_values:
        # Normalize each sample to a stripped string before looking for year-like digit groups.
        value = str(raw).strip()
        if not value:
            continue

        # Extract standalone 2-digit or 4-digit integer tokens such as "23", "2024",
        # or the numeric part of strings like "anno 2023".
        for match in re.finditer(r"(?<!\d)(\d{2}|\d{4})(?!\d)", value):
            digits = match.group(1)
            width = len(digits)
            # Accept plausible 4-digit Gregorian years or generic 2-digit abbreviations.
            # The goal is not to validate the final cleaned output here, only to infer
            # whether the dominant contract in the raw data is year-width 2 or 4.
            if width == 4 and 1900 <= int(digits) <= 2100:
                width_counts[4] = width_counts.get(4, 0) + 1
            elif width == 2:
                width_counts[2] = width_counts.get(2, 0) + 1

    if not width_counts:
        return None

    # Prefer the most frequent width; on ties, keep the stricter 4-digit contract
    # so columns that are mostly canonical years do not get downgraded accidentally.
    predominant_width = max(width_counts.items(), key=lambda item: (item[1], item[0]))[0]
    return "4-digit year" if predominant_width == 4 else "2-digit year"


def _infer_numeric_pattern(column_name: str, sample_values: list[str], numeric_role: str | None) -> str:
    """Infer a single canonical pattern name for a numeric column when the agent's output is absent or ambiguous.

    Uses the column name tokens and the parsed sample values to pick the most specific pattern,
    falling back to generic labels when no specific pattern is recognisable.
    """
    # Normalise the column name and split into tokens to check for semantic keywords
    normalized_name = normalized_schema_name(column_name)
    tokens = set(normalized_name.split("_"))

    # Parse the sample values into integers and check for decimals
    ints, saw_decimal = _parse_numeric_samples(sample_values)
    
    # Keep only non-empty string representations of the sample values for shape checks
    rendered_samples = [str(value).strip() for value in sample_values if str(value).strip()]

    # If values are in the 1-12 range and the column name contains a month keyword, treat as month number
    if ints and all(1 <= value <= 12 for value in ints[:10]) and ({"mese", "month"} & tokens):
        return "month number (1-12)"

    # If values are in the 1900-2100 range and the column name contains a year keyword, treat as year
    if ints and all(1900 <= value <= 2100 for value in ints[:10]) and ({"anno", "year"} & tokens):
        return "4-digit year"

    # For anno/year columns with mixed values such as 2023 plus 23, follow the
    # predominant year width instead of accepting a generic integer code.
    year_width_pattern = _infer_year_width_pattern(column_name, rendered_samples)
    if year_width_pattern is not None:
        return year_width_pattern

    # Check for the compact YYYYMM period key format
    if rendered_samples and all(
        re.fullmatch(r"\d{6}", value) # Check the value is exactly 6 digits
        and 1900 <= int(value[:4]) <= 2100  # Check the year part is in a plausible range
        and 1 <= int(value[4:]) <= 12  # Check the month part is valid
        for value in rendered_samples[:10]):
        return "YYYYMM"

    # Fall back based on whether decimals were seen or the semantic role indicates a code
    if saw_decimal:
        return "decimal numeric string"
    if numeric_role in {"code", "indicator"}:
        return "integer code"
    return "integer count"

def _normalize_numeric_role_for_pattern(pandas_dtype: str, numeric_role: str | None, detected_pattern: str | None) -> str | None:
    """Ensure numeric_role is consistent with the resolved dtype and pattern.

    Returns None for non-numeric dtypes, always returns 'measure' for Float64, and for Int64
    decides between the agent-provided role and the 'code' default based on the pattern.
    """
    # Roles only apply to numeric dtypes
    if pandas_dtype not in {"Int64", "Float64"}:
        return None
    # Float64 columns always represent a measurable quantity
    if pandas_dtype == "Float64":
        return "measure"

    # Normalise the pattern for a safe string comparison
    normalized_pattern = (detected_pattern or "").strip().lower()
    # "integer count" implies a real quantity even if the dtype is Int64, so promote to measure
    if normalized_pattern == "integer count":
        return "measure"
    # Keep the agent's role if it provided one, otherwise default to code
    if numeric_role is not None:
        return numeric_role
    return "code"

def _normalize_dtype_inference_choice(column_name: str, dtype_col, col_profile):
    """Resolve the final dtype, role, and pattern for a column, overriding the agent when needed.

    If the column passes the hard numeric threshold (>=80% numeric parse rate and <20% datetime),
    statistical evidence overrides the LLM's choice entirely. Otherwise the agent's output is
    accepted but cleaned up to remove ambiguous patterns and fill in missing roles.
    """
    # Extract the agent's output if available, otherwise fall back to the statistical profile
    pandas_dtype = dtype_col.pandas_dtype if dtype_col else col_profile.pandas_dtype
    numeric_role = dtype_col.numeric_role if dtype_col else None
    string_role = dtype_col.string_role if dtype_col else None
    detected_pattern = dtype_col.detected_pattern if dtype_col else None
    rationale = dtype_col.rationale if dtype_col else ""

    # Extract the relevant stats from the column profile
    numeric_parse_pct = col_profile.numeric_parse_pct
    datetime_parse_pct = col_profile.datetime_parse_pct

    # If 80% of values parse as numeric and datetime evidence is weak, force a numeric dtype regardless of what the agent returned
    if numeric_parse_pct >= 80 and datetime_parse_pct < 20:
        _, saw_decimal = _parse_numeric_samples(col_profile.sample_values)
        # If decimal values were seen, the column must be Float64
        forced_dtype = "Float64" if saw_decimal else "Int64"
        # Preserve 'measure' role for Float64 only if the agent already identified it, otherwise default to 'code'
        forced_numeric_role = numeric_role if forced_dtype == "Float64" and numeric_role == "measure" else "code"
        forced_pattern = detected_pattern
        # If the pattern is absent or ambiguous, infer a canonical one from the sample values
        if forced_dtype == "Int64" and (_pattern_is_ambiguous(detected_pattern) or not detected_pattern):
            forced_pattern = _infer_numeric_pattern(column_name, col_profile.sample_values, forced_numeric_role)
        elif forced_dtype == "Float64" and (_pattern_is_ambiguous(detected_pattern) or not detected_pattern):
            forced_pattern = "decimal numeric string"
        forced_numeric_role = _normalize_numeric_role_for_pattern(forced_dtype, forced_numeric_role, forced_pattern)

        # Only update the rationale when the override actually changed something
        if pandas_dtype != forced_dtype or string_role is not None or _pattern_is_ambiguous(detected_pattern):
            rationale = (
                f"Forced to {forced_dtype} because numeric_parse_pct={numeric_parse_pct:.1f}% "
                f"and datetime_parse_pct={datetime_parse_pct:.1f}%. "
                "Above the hard numeric threshold, minority textual or malformed values are treated as corruption. "
                f"Canonical target pattern: {forced_pattern}."
            )

        return forced_dtype, forced_numeric_role, None, forced_pattern, rationale

    # For numeric dtypes below the hard threshold, clean up missing or ambiguous roles and patterns
    if pandas_dtype in {"Int64", "Float64"}:
        # string_role is not valid for numeric dtypes
        string_role = None
        if numeric_role is None:
            numeric_role = "measure" if pandas_dtype == "Float64" else "code"
        # Replace ambiguous or missing patterns with a single canonical target format
        if _pattern_is_ambiguous(detected_pattern) or not detected_pattern:
                if pandas_dtype == "Float64":
                    detected_pattern = "decimal numeric string"
                else:
                    detected_pattern = _infer_numeric_pattern(column_name, col_profile.sample_values, numeric_role)
                rationale = (rationale + " " if rationale else "") + f"Normalized detected_pattern to the single dominant target format '{detected_pattern}'."
        numeric_role = _normalize_numeric_role_for_pattern(pandas_dtype, numeric_role, detected_pattern)

    return pandas_dtype, numeric_role, string_role, detected_pattern, rationale


def build_schema_issues(columns: list[SchemaColumnEntry], duplicate_groups: list[SchemaDuplicateGroup]) -> list[SchemaIssue]:
    """Build the list of SchemaIssue entries from naming violations and duplicate column groups."""
    issues: list[SchemaIssue] = []

    # First pass: flag columns whose name violates the lowercase snake_case convention
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

    # Second pass: for each duplicate group, emit one issue per member pointing to its peers
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
    """Run the dtype inference agent on the given dataset path and return the inferred dtypes for each column."""
    df = load_dataset_frame(path)
    # Build the text document for the agent: include column names and samples for each column, formatted clearly.
    text = build_dtype_inference_text(df)
    # Attach the text as a document and run the agent with backoff retries. The agent will return a DatasetDtypeInference object.
    prompt = ["Infer the correct pandas dtype for each column based on the attached CSV sample.",attach_text_document(text)]
    print(f"[orchestrator][schema][dtype-inference] dataset='{path.stem}'", file=sys.stderr, flush=True)
    # Run the agent with backoff retries and return the output field, which contains the inferred dtypes.
    result = run_agent_with_backoff(dtype_inference_agent, prompt)
    return result.output

def run_schema_validation(path: Path, reuse_cache: bool = False) -> SchemaHandoff:
    """Run the schema validation agent on the given dataset path and return a SchemaHandoff object containing the analysis results."""
    if reuse_cache:
        # If reuse_cache is True, load the existing handoff from disk instead of re-running the analysis. This allows us to skip expensive agent calls when we just want to inspect or use the results.
        return load_schema_handoff(path)

    df = load_dataset_frame(path)

    # Run dtype inference first to get the agent's best guess at each column's dtype.
    dtype_inference = run_dtype_inference(path)
    # Build a mapping from column name to inferred dtype column for easy lookup.
    dtype_map = {col.column_name: col for col in dtype_inference.columns}
    dtype_overrides = {name: col.pandas_dtype for name, col in dtype_map.items()}

    # Build the dataset profile, which includes statistics and samples for each column. Pass the dtype overrides to ensure the profile uses the agent's inferred dtypes.
    profile = build_dataset_profile(df, path.stem, dtype_overrides=dtype_overrides)

    # Store columns with the same normalized name together to detect potential duplicates.
    duplicate_groups_by_name: dict[str, list[str]] = {}
    for col_name in df.columns:
        canonical = normalized_schema_name(col_name)
        duplicate_groups_by_name.setdefault(canonical, []).append(col_name)
    
    # Only keep groups with more than one column, as singletons are not duplicates.
    duplicate_groups = [SchemaDuplicateGroup(canonical_name=cn, columns=cols) for cn, cols in duplicate_groups_by_name.items() if len(cols) > 1]

    # Build the list of SchemaColumnEntry objects that contain the analysis for each column
    columns: list[SchemaColumnEntry] = []

    # Iterate over the column profiles from the dataset profile 
    for col_profile in profile.columns_profiles:
        name = col_profile.column_name
        dtype_col = dtype_map.get(name)
        # Determine the final pandas dtype and roles for this column, based on the agent's inference and the column profile.
        dtype_inference_choice = _normalize_dtype_inference_choice(name, dtype_col, col_profile)
        columns.append(SchemaColumnEntry(
            name = col_profile.column_name,
            pandas_dtype = dtype_inference_choice[0], # The final pandas dtype to assign to this column, based on the agent's inference and the profile statistics.
            numeric_role = dtype_inference_choice[1], # The numeric role assigned to this column based on the agent's inference and the profile statistics.
            string_role = dtype_inference_choice[2], # The string role assigned to this column based on the agent's inference and the profile statistics.
            detected_pattern = dtype_inference_choice[3], # The pattern detected for this column based on the agent's inference and the profile statistics.
            rationale = dtype_inference_choice[4], # The rationale for the dtype inference choice.
            non_null_rows = col_profile.non_null_rows,
            distinct_non_null_values = col_profile.distinct_non_null_values,
            numeric_parse_pct=col_profile.numeric_parse_pct,
            datetime_parse_pct=col_profile.datetime_parse_pct,
            empty_like_pct=col_profile.empty_like_pct,
            sample_values=col_profile.sample_values,
            naming_valid = is_valid_schema_name(name), # Check if the column name is valid according to the naming rules.
            naming_reason = naming_rule_reason(name) if not is_valid_schema_name(name) else None, # If the name is not valid, provide a reason why it violates the naming rules
            rename_suggestion = suggest_schema_name(name) if not is_valid_schema_name(name) else None,# If the name is not valid, suggest a valid name based on the original.
        ))

    # Build the list of schema issues based on the column analysis and duplicate groups. 
    issues = build_schema_issues(columns, duplicate_groups)

    # Create the SchemaHandoff object that contains the dataset name, column analysis, and detected issues.
    handoff = SchemaHandoff(
        dataset_name=path.stem,
        total_rows=len(df),
        total_columns=len(df.columns),
        columns=columns,
        issues=issues,
        duplicate_groups=duplicate_groups,
    )
    print(f"[orchestrator][schema][summary] dataset='{path.stem}'", file=sys.stderr, flush=True)

    # Run the schema summary agent to generate a concise summary of the schema analysis, which will be included in the handoff for downstream use.
    prompt = f"Summarize the provided schema analysis for dataset {path.stem}. Do not infer new findings."                                                                                                                                                  
    result = run_agent_with_backoff(schema_summary_agent, [prompt, attach_profile_text(handoff)])
    
    # Update the handoff with the generated summary and save it to disk for downstream stages to consume.
    handoff = handoff.model_copy(update={"summary": result.output.summary})
    save_schema_handoff(path, handoff)
    return handoff
