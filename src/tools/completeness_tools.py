"""Completeness profiling for the completeness-analysis agent.

Computes column-level missingness statistics, detects placeholder tokens
such as ``n/a`` or ``-``, and assembles the dataset-level profile that is
handed to the completeness analysis agent."""

from __future__ import annotations
from pydantic import BaseModel, Field
from src.tools.common_tools import PLACEHOLDER_TOKENS


# --- Data Models ---

class CompletenessColumnProfile(BaseModel):
    """Statistical completeness snapshot for one column.

    Captures how many rows contain a real value versus a null-like or
    placeholder token, along with a few representative placeholder examples
    for downstream explanation by the agent.
    """
    column_name: str
    pandas_dtype: str
    total_rows: int = Field(ge=0)
    non_null_rows: int = Field(ge=0)
    completeness_pct: float = Field(ge=0, le=100)
    missing_like_count: int = Field(ge=0)
    missing_like_pct: float = Field(ge=0, le=100)
    placeholder_examples: list[str] = Field(default_factory=list)  # distinct raw tokens such as "-", "N/A", "unknown"
    distinct_non_null_values: int = Field(ge=0)


class CompletenessProfile(BaseModel):
    """Full completeness profile for a dataset: one per-column snapshot plus rollups."""
    dataset_name: str
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    overall_completeness_pct: float = Field(ge=0, le=100)
    placeholder_values_detected: list[str] = Field(default_factory=list)  # all distinct placeholder spellings seen anywhere in the dataset
    columns: list[CompletenessColumnProfile] = Field(default_factory=list)


# Completeness Helpers
def compute_missing_like_mask(series) -> object:
    """Return a boolean mask for values that should count as missing-like.
    Treats true nulls, empty strings, and normalized placeholder tokens as
    equivalent missingness signals for completeness scoring.
    """
    # Normalize nulls and strings into one lowercase representation so placeholder matching is deterministic.
    rendered = series.fillna("").astype(str).str.strip().str.lower()
    return rendered.isin(PLACEHOLDER_TOKENS)

def sample_placeholder_examples(series) -> list[str]:
    """Collect a few distinct raw placeholder spellings from a column.
    Preserves the original rendered token (for example ``"N/A"`` instead of
    ``"n/a"``) so downstream summaries can show concrete evidence from the data.
    """
    examples: list[str] = []
    rendered = series.fillna("").astype(str).str.strip()
    lowered = rendered.str.lower()
    for original, normalized in zip(rendered, lowered, strict=False):
        # Keep only values that normalize to one of the configured placeholder tokens.
        if normalized not in PLACEHOLDER_TOKENS:
            continue
        # De-duplicate while preserving first-seen order so examples stay readable.
        if original not in examples:
            examples.append(original)
    return examples

def detect_placeholder_values(df) -> list[str]:
    """Return all distinct placeholder spellings found anywhere in the dataset."""
    detected: set[str] = set()
    for column_name in df.columns:
        rendered = df[column_name].fillna("").astype(str).str.strip()
        lowered = rendered.str.lower()
        for original, normalized in zip(rendered, lowered, strict=False):
            # Ignore empty strings here; they count as missing-like but are not useful as explicit placeholder examples.
            if normalized in PLACEHOLDER_TOKENS and original != "":
                detected.add(original)
    return sorted(detected, key=str.lower)


# Profile Builder
def build_completeness_profile(df, dataset_name: str) -> CompletenessProfile:
    """Build the dataset-level completeness profile from a raw DataFrame.

    Computes one CompletenessColumnProfile per column, then rolls those values
    up into an overall completeness percentage and a dataset-wide placeholder list.
    """
    column_profiles: list[CompletenessColumnProfile] = []
    total_cells = len(df) * len(df.columns)
    total_present_cells = 0

    for column_name in df.columns:
        series = df[column_name]
        # Missing-like means true nulls plus configured placeholders such as "-", "n/a", and similar tokens.
        missing_like_mask = compute_missing_like_mask(series)
        missing_like_count = int(missing_like_mask.sum())
        total_rows = len(series)
        # "Present" cells exclude any missing-like token because downstream cleaning should treat them as absent values.
        non_missing_like_rows = total_rows - missing_like_count
        total_present_cells += non_missing_like_rows

        completeness_pct = 0.0 if total_rows == 0 else float((non_missing_like_rows / total_rows) * 100)
        missing_like_pct = 0.0 if total_rows == 0 else float((missing_like_count / total_rows) * 100)

        column_profiles.append(
            CompletenessColumnProfile(
                column_name=column_name,
                pandas_dtype=str(series.dtype),  # raw pandas dtype before any cleaning or schema normalization
                total_rows=total_rows,
                non_null_rows=non_missing_like_rows,
                completeness_pct=completeness_pct,
                missing_like_count=missing_like_count,
                missing_like_pct=missing_like_pct,
                placeholder_examples=sample_placeholder_examples(series),
                distinct_non_null_values=int(series[~missing_like_mask].nunique(dropna=True)),  # cardinality of genuinely present values only
            )
        )

    overall_completeness_pct = 0.0 if total_cells == 0 else float((total_present_cells / total_cells) * 100)
    return CompletenessProfile(
        dataset_name=dataset_name,
        total_rows=len(df),
        total_columns=len(df.columns),
        overall_completeness_pct=overall_completeness_pct,
        placeholder_values_detected=detect_placeholder_values(df),
        columns=column_profiles,
    )