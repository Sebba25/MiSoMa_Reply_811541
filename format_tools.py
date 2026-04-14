from __future__ import annotations

from collections import Counter
import re
from pydantic import BaseModel, Field

from common_tools import (
    PLACEHOLDER_TOKENS,
    compute_datetime_parse_pct,
    compute_empty_like_pct,
    compute_numeric_parse_pct,
    sample_non_null_values,
    value_shape,
)
from schema_tools import normalized_schema_name


# Format Data Models

class ValueShapeProfile(BaseModel):
    shape: str
    count: int = Field(ge=0)
    pct: float = Field(ge=0, le=100)
    sample_values: list[str] = Field(default_factory=list)


class ColumnFormatProfile(BaseModel):
    column_name: str
    pandas_dtype: str
    total_rows: int = Field(ge=0)
    non_null_rows: int = Field(ge=0)
    distinct_non_null_values: int = Field(ge=0)
    numeric_parse_pct: float = Field(ge=0, le=100)
    datetime_parse_pct: float = Field(ge=0, le=100)
    empty_like_pct: float = Field(ge=0, le=100)
    sample_values: list[str] = Field(default_factory=list)
    top_value_shapes: list[ValueShapeProfile] = Field(default_factory=list)


class FormatOutlierExample(BaseModel):
    value: str
    shape: str
    count: int = Field(ge=0)


class ColumnFormatFacts(BaseModel):
    column_name: str
    pandas_dtype: str
    total_rows: int = Field(ge=0)
    non_null_rows: int = Field(ge=0)
    distinct_non_null_values: int = Field(ge=0)
    numeric_parse_pct: float = Field(ge=0, le=100)
    datetime_parse_pct: float = Field(ge=0, le=100)
    empty_like_pct: float = Field(ge=0, le=100)
    semantic_hint: str
    machine_format_candidate: bool = False
    dominant_shape: str | None = None
    dominant_shape_pct: float = Field(ge=0, le=100)
    dominant_example_values: list[str] = Field(default_factory=list)
    inconsistent_rows: int = Field(ge=0)
    inconsistent_examples: list[FormatOutlierExample] = Field(default_factory=list)
    top_value_shapes: list[ValueShapeProfile] = Field(default_factory=list)


# Format Helpers

def infer_format_semantic_hint(column_name: str) -> str:
    tokens = set(normalized_schema_name(column_name).split("_"))
    if {"id", "cod", "code"} & tokens:
        return "code_or_identifier"
    if {"date", "time", "timestamp", "datetime", "period", "month", "year", "rata", "aggregation", "mese", "anno"} & tokens:
        return "temporal_period"
    if {"spesa", "amount", "importo", "price", "prezzo", "cost", "costo", "total", "totale", "attivazioni", "cessazioni"} & tokens:
        return "numeric_amount_or_measure"
    if {"descrizione", "description", "name", "note", "qualifica"} & tokens:
        return "descriptive_text"
    if {"area", "regione", "provincia", "geografica", "sede"} & tokens:
        return "categorical_location"
    return "unknown"


def is_plain_numeric_value(value: str) -> bool:
    return bool(re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", value))


def compute_top_value_shapes(series, limit: int = 5, sample_limit: int = 250) -> list[ValueShapeProfile]:
    rendered = series.dropna().astype(str).str.strip()
    rendered = rendered[rendered != ""]
    if rendered.empty:
        return []

    sample = rendered.head(sample_limit)
    shape_counts: dict[str, int] = {}
    shape_examples: dict[str, list[str]] = {}
    for value in sample:
        shape = value_shape(value)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        if shape not in shape_examples:
            shape_examples[shape] = []
        if value not in shape_examples[shape] and len(shape_examples[shape]) < 3:
            shape_examples[shape].append(value[:80])

    total = len(sample)
    ranked_shapes = sorted(shape_counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        ValueShapeProfile(
            shape=shape,
            count=count,
            pct=float((count / total) * 100),
            sample_values=shape_examples.get(shape, []),
        )
        for shape, count in ranked_shapes[:limit]
    ]


def build_column_format_facts(df, column_name: str) -> ColumnFormatFacts:
    profile = build_column_format_profile(df, column_name)
    series = df[column_name]
    rendered = series.dropna().astype(str).str.strip()
    rendered = rendered[rendered != ""]
    normalized = rendered.str.lower()
    rendered = rendered[~normalized.isin(PLACEHOLDER_TOKENS)]

    semantic_hint = infer_format_semantic_hint(column_name)
    machine_format_candidate = False

    if rendered.empty:
        return ColumnFormatFacts(
            column_name=profile.column_name,
            pandas_dtype=profile.pandas_dtype,
            total_rows=profile.total_rows,
            non_null_rows=profile.non_null_rows,
            distinct_non_null_values=profile.distinct_non_null_values,
            numeric_parse_pct=profile.numeric_parse_pct,
            datetime_parse_pct=profile.datetime_parse_pct,
            empty_like_pct=profile.empty_like_pct,
            semantic_hint=semantic_hint,
            machine_format_candidate=machine_format_candidate,
            top_value_shapes=profile.top_value_shapes,
        )

    if semantic_hint == "numeric_amount_or_measure":
        pure_numeric_values = rendered[rendered.apply(is_plain_numeric_value)]
        contaminated_values = rendered[~rendered.apply(is_plain_numeric_value)]

        dominant_example_values = []
        for value in pure_numeric_values:
            if value not in dominant_example_values:
                dominant_example_values.append(value[:80])
            if len(dominant_example_values) >= 5:
                break

        inconsistent_examples: list[FormatOutlierExample] = []
        seen_shapes: set[str] = set()
        if not contaminated_values.empty:
            value_counts = Counter(contaminated_values)
            for value, count in value_counts.most_common():
                shape = value_shape(value)
                if shape in seen_shapes:
                    continue
                inconsistent_examples.append(
                    FormatOutlierExample(
                        value=value[:80],
                        shape=shape,
                        count=count,
                    )
                )
                seen_shapes.add(shape)
                if len(inconsistent_examples) >= 10:
                    break

        inconsistent_rows = len(contaminated_values)
        machine_format_candidate = inconsistent_rows > 0 and profile.numeric_parse_pct >= 80

        return ColumnFormatFacts(
            column_name=profile.column_name,
            pandas_dtype=profile.pandas_dtype,
            total_rows=profile.total_rows,
            non_null_rows=profile.non_null_rows,
            distinct_non_null_values=profile.distinct_non_null_values,
            numeric_parse_pct=profile.numeric_parse_pct,
            datetime_parse_pct=profile.datetime_parse_pct,
            empty_like_pct=profile.empty_like_pct,
            semantic_hint=semantic_hint,
            machine_format_candidate=machine_format_candidate,
            dominant_shape="numeric",
            dominant_shape_pct=0.0 if rendered.empty else float((len(pure_numeric_values) / len(rendered)) * 100),
            dominant_example_values=dominant_example_values,
            inconsistent_rows=inconsistent_rows,
            inconsistent_examples=inconsistent_examples,
            top_value_shapes=profile.top_value_shapes,
        )

    shape_counts = Counter(value_shape(value) for value in rendered)
    dominant_shape, dominant_count = shape_counts.most_common(1)[0]
    dominant_shape_pct = float((dominant_count / len(rendered)) * 100)
    distinct_shape_count = len(shape_counts)

    dominant_example_values: list[str] = []
    for value in rendered:
        if value_shape(value) == dominant_shape and value not in dominant_example_values:
            dominant_example_values.append(value[:80])
        if len(dominant_example_values) >= 5:
            break

    inconsistent_rows = len(rendered) - dominant_count
    inconsistent_examples: list[FormatOutlierExample] = []
    seen_values: set[str] = set()
    seen_shapes: set[str] = set()
    if inconsistent_rows > 0:
        value_counts = Counter(rendered)
        for value, count in value_counts.most_common():
            shape = value_shape(value)
            if shape == dominant_shape or value in seen_values or shape in seen_shapes:
                continue
            inconsistent_examples.append(
                FormatOutlierExample(
                    value=value[:80],
                    shape=shape,
                    count=count,
                )
            )
            seen_values.add(value)
            seen_shapes.add(shape)
            if len(inconsistent_examples) >= 10:
                break

    if semantic_hint == "temporal_period":
        machine_format_candidate = (
            dominant_shape_pct >= 70
            and inconsistent_rows > 0
            and (profile.datetime_parse_pct >= 20 or profile.numeric_parse_pct >= 70)
        )
    elif semantic_hint == "numeric_amount_or_measure":
        machine_format_candidate = (
            dominant_shape_pct >= 70
            and inconsistent_rows > 0
            and profile.numeric_parse_pct >= 80
        )
    elif semantic_hint == "code_or_identifier":
        machine_format_candidate = (
            dominant_shape_pct >= 85
            and inconsistent_rows > 0
            and distinct_shape_count <= 3
        )

    return ColumnFormatFacts(
        column_name=profile.column_name,
        pandas_dtype=profile.pandas_dtype,
        total_rows=profile.total_rows,
        non_null_rows=profile.non_null_rows,
        distinct_non_null_values=profile.distinct_non_null_values,
        numeric_parse_pct=profile.numeric_parse_pct,
        datetime_parse_pct=profile.datetime_parse_pct,
        empty_like_pct=profile.empty_like_pct,
        semantic_hint=semantic_hint,
        machine_format_candidate=machine_format_candidate,
        dominant_shape=dominant_shape,
        dominant_shape_pct=dominant_shape_pct,
        dominant_example_values=dominant_example_values,
        inconsistent_rows=inconsistent_rows,
        inconsistent_examples=inconsistent_examples,
        top_value_shapes=profile.top_value_shapes,
    )


# Format Profile Builder

def build_column_format_profile(df, column_name: str) -> ColumnFormatProfile:
    series = df[column_name]
    return ColumnFormatProfile(
        column_name=column_name,
        pandas_dtype=str(series.dtype),
        total_rows=len(series),
        non_null_rows=int(series.notna().sum()),
        distinct_non_null_values=int(series.nunique(dropna=True)),
        numeric_parse_pct=compute_numeric_parse_pct(series),
        datetime_parse_pct=compute_datetime_parse_pct(series),
        empty_like_pct=compute_empty_like_pct(series),
        sample_values=sample_non_null_values(series),
        top_value_shapes=compute_top_value_shapes(series),
    )
