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

from agents import dtype_inference_agent, schema_summary_agent
from cache import load_schema_handoff, save_schema_handoff
from models import DatasetDtypeInference, SchemaColumnEntry, SchemaHandoff, SchemaIssue
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
    """Load the dataset, build a column profile text, and ask the agent to infer the target dtype per column."""
    df = load_dataset_frame(path)
    # Build a human-readable profile text to attach to the agent prompt
    text = build_dtype_inference_text(df)
    prompt = ["Infer the correct pandas dtype for each column based on the attached CSV sample.", attach_text_document(text)]
    print(f"[orchestrator][schema][dtype-inference] dataset='{path.stem}'", file=sys.stderr, flush=True)
    result = run_agent_with_backoff(dtype_inference_agent, prompt)
    return result.output

def run_schema_validation(path: Path, reuse_cache: bool = False) -> SchemaHandoff:
    """Run the full schema validation stage and return a SchemaHandoff ready for downstream use.

    Orchestrates dtype inference, statistical profiling, duplicate column detection, naming
    validation, and a final summary pass. The result is cached to disk.
    """
    if reuse_cache:
        return load_schema_handoff(path)

    df = load_dataset_frame(path)

    # Run dtype inference first so the profile can use the inferred dtypes as overrides
    dtype_inference = run_dtype_inference(path)
    dtype_map = {col.column_name: col for col in dtype_inference.columns}
    dtype_overrides = {name: col.pandas_dtype for name, col in dtype_map.items()}

    profile = build_dataset_profile(df, path.stem, dtype_overrides=dtype_overrides)

    # Detect duplicate columns by comparing their canonical (normalised) names
    duplicate_groups_by_name: dict[str, list[str]] = {}
    for col_name in df.columns:
        canonical = normalized_schema_name(col_name)
        duplicate_groups_by_name.setdefault(canonical, []).append(col_name)
    # Keep only groups with more than one column — single entries are not duplicates
    duplicate_groups = [
        SchemaDuplicateGroup(canonical_name=cn, columns=cols)
        for cn, cols in duplicate_groups_by_name.items()
        if len(cols) > 1
    ]

    # Build one SchemaColumnEntry per column, merging agent inference with the statistical profile
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

    # Build the list of SchemaIssue entries from naming violations and duplicate column groups
    issues = build_schema_issues(columns, duplicate_groups)

    handoff = SchemaHandoff(
        dataset_name=path.stem,
        total_rows=len(df),
        total_columns=len(df.columns),
        columns=columns,
        issues=issues,
        duplicate_groups=duplicate_groups,
    )
    # Ask the summary agent to write a concise handoff text over the already-built findings
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
