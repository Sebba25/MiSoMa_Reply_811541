from __future__ import annotations

from pydantic import BaseModel, Field

from tools.common_tools import PLACEHOLDER_TOKENS


# Completeness Data Models

class CompletenessColumnProfile(BaseModel):
    column_name: str
    pandas_dtype: str
    total_rows: int = Field(ge=0)
    non_null_rows: int = Field(ge=0)
    completeness_pct: float = Field(ge=0, le=100)
    missing_like_count: int = Field(ge=0)
    missing_like_pct: float = Field(ge=0, le=100)
    placeholder_examples: list[str] = Field(default_factory=list)
    distinct_non_null_values: int = Field(ge=0)


class CompletenessProfile(BaseModel):
    dataset_name: str
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    overall_completeness_pct: float = Field(ge=0, le=100)
    placeholder_values_detected: list[str] = Field(default_factory=list)
    columns: list[CompletenessColumnProfile] = Field(default_factory=list)


# Completeness Helpers

def compute_missing_like_mask(series) -> object:
    rendered = series.fillna("").astype(str).str.strip().str.lower()
    return rendered.isin(PLACEHOLDER_TOKENS)


def sample_placeholder_examples(series, limit: int = 5) -> list[str]:
    examples: list[str] = []
    rendered = series.fillna("").astype(str).str.strip()
    lowered = rendered.str.lower()
    for original, normalized in zip(rendered, lowered, strict=False):
        if normalized not in PLACEHOLDER_TOKENS:
            continue
        if original not in examples:
            examples.append(original)
        if len(examples) >= limit:
            break
    return examples


def detect_placeholder_values(df) -> list[str]:
    detected: set[str] = set()
    for column_name in df.columns:
        rendered = df[column_name].fillna("").astype(str).str.strip()
        lowered = rendered.str.lower()
        for original, normalized in zip(rendered, lowered, strict=False):
            if normalized in PLACEHOLDER_TOKENS and original != "":
                detected.add(original)
    return sorted(detected, key=str.lower)


# Completeness Profile Builder

def build_completeness_profile(df, dataset_name: str) -> CompletenessProfile:
    column_profiles: list[CompletenessColumnProfile] = []
    total_cells = len(df) * len(df.columns)
    total_present_cells = 0

    for column_name in df.columns:
        series = df[column_name]
        missing_like_mask = compute_missing_like_mask(series)
        missing_like_count = int(missing_like_mask.sum())
        total_rows = len(series)
        non_missing_like_rows = total_rows - missing_like_count
        total_present_cells += non_missing_like_rows

        completeness_pct = 0.0 if total_rows == 0 else float((non_missing_like_rows / total_rows) * 100)
        missing_like_pct = 0.0 if total_rows == 0 else float((missing_like_count / total_rows) * 100)

        column_profiles.append(
            CompletenessColumnProfile(
                column_name=column_name,
                pandas_dtype=str(series.dtype),
                total_rows=total_rows,
                non_null_rows=non_missing_like_rows,
                completeness_pct=completeness_pct,
                missing_like_count=missing_like_count,
                missing_like_pct=missing_like_pct,
                placeholder_examples=sample_placeholder_examples(series),
                distinct_non_null_values=int(series[~missing_like_mask].nunique(dropna=True)),
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
