"""Per-column format profiling for the format-consistency and cleaning stages.

Builds value-shape summaries, identifies dominant versus outlier formats, and
assembles the ``ColumnFormatFacts`` bundle consumed by the format-consistency
agent and the downstream cleaner generator."""

from __future__ import annotations

from collections import Counter
import re
from pydantic import BaseModel, Field

from src.tools.common_tools import (
    PLACEHOLDER_TOKENS,
    compute_datetime_parse_pct,
    compute_empty_like_pct,
    compute_numeric_parse_pct,
    sample_non_null_values,
    value_shape,
)
from src.tools.schema_tools import normalized_schema_name


# --- Data Models ---

class ValueShapeProfile(BaseModel):
    """Compact summary of one structural value shape seen in a column."""
    shape: str
    count: int = Field(ge=0)
    pct: float = Field(ge=0, le=100)
    sample_values: list[str] = Field(default_factory=list)  # representative raw values matching this shape


class ColumnFormatProfile(BaseModel):
    """Column-level structural profile computed directly from the raw DataFrame.

    This is the deterministic profiling layer: parse rates, sample values,
    and the most common structural shapes before any higher-level heuristics
    decide whether a format inconsistency is actionable.
    """
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
    """One concrete outlier value grouped by its structural shape."""
    value: str
    shape: str
    count: int = Field(ge=0)


class ColumnFormatFacts(BaseModel):
    """Final format-consistency facts handed to the validation and cleaning stages.

    Combines deterministic profiling with heuristic judgments about whether
    the column behaves like a machine-format candidate and what its dominant
    canonical format appears to be.
    """
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


def select_outlier_examples(
    values,
    *,
    exclude_shape: str | None = None,
    max_shapes: int = 6,
    max_per_shape: int = 3,
    max_total: int = 15,
) -> list[FormatOutlierExample]:
    """Select representative outlier examples, grouped and ranked by value shape.

    Used to keep downstream prompts compact while still showing the cleaner
    generator or format agent the main families of inconsistent values.
    """
    if len(values) == 0:
        return []

    value_counts = Counter(values)
    shape_totals: dict[str, int] = {}
    grouped_values: dict[str, list[tuple[str, int]]] = {}

    for value, count in value_counts.items():
        # Collapse individual values into structural shapes such as YYYY-MM or DIGITx6.
        shape = value_shape(value)
        if exclude_shape is not None and shape == exclude_shape:
            continue
        shape_totals[shape] = shape_totals.get(shape, 0) + count
        grouped_values.setdefault(shape, []).append((value, count))

    # Rank shapes by frequency first so the most important inconsistency families are shown first.
    ranked_shapes = sorted(shape_totals.items(), key=lambda item: (-item[1], item[0]))
    selected_shapes = [shape for shape, _ in ranked_shapes[:max_shapes]]

    examples: list[FormatOutlierExample] = []
    for shape in selected_shapes:
        # Within each shape family, show the most frequent concrete values.
        ranked_values = sorted(grouped_values[shape], key=lambda item: (-item[1], item[0]))
        for value, count in ranked_values[:max_per_shape]:
            examples.append(
                FormatOutlierExample(
                    value=value[:80],  # trim long raw values so prompts stay readable
                    shape=shape,
                    count=count,
                )
            )
            if len(examples) >= max_total:
                return examples

    return examples

def infer_format_semantic_hint(column_name: str) -> str:
    """Infer a coarse semantic hint from the normalized column name.

    The hint steers later heuristics toward the right notion of "consistent":
    dates care about one temporal layout, codes care about stable width/shape,
    and descriptive text usually should not trigger machine-format validation.
    """
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
    """Return True if the rendered value is a bare numeric literal.

    Allows optional sign plus either dot or comma decimals, but rejects extra
    text, currency symbols, units, or formatting noise.
    """
    return bool(re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", value))


def compute_top_value_shapes(series, limit: int = 5, sample_limit: int = 250) -> list[ValueShapeProfile]:
    """Profile the most common structural shapes in a column from a bounded sample.

    Sampling keeps the profiling fast while still giving downstream stages a
    representative view of the main layouts present in the data.
    """
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
        # Keep a few distinct examples per shape so the profile remains concrete and human-readable.
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
    """Build the format-consistency facts for one column.

    Starts from the raw structural profile, removes placeholder tokens, then
    applies heuristics tailored to temporal, numeric, and identifier columns
    to decide whether the column is a machine-format candidate with actionable
    inconsistencies.
    """
    profile = build_column_format_profile(df, column_name)
    series = df[column_name]
    rendered = series.dropna().astype(str).str.strip()
    rendered = rendered[rendered != ""]
    normalized = rendered.str.lower()
    # Placeholder tokens count as missingness, not as evidence of a format family.
    rendered = rendered[~normalized.isin(PLACEHOLDER_TOKENS)]

    semantic_hint = infer_format_semantic_hint(column_name)
    machine_format_candidate = False

    # If nothing remains after filtering null-like values, return a facts object with no dominant pattern.
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
        # Separate plain numeric values from contaminated ones such as currency/unit suffixed strings.
        pure_numeric_values = rendered[rendered.apply(is_plain_numeric_value)]
        contaminated_values = rendered[~rendered.apply(is_plain_numeric_value)]

        dominant_example_values = []
        for value in pure_numeric_values:
            if value not in dominant_example_values:
                dominant_example_values.append(value[:80])
            if len(dominant_example_values) >= 5:
                break

        inconsistent_examples = select_outlier_examples(
            contaminated_values,
            max_shapes=10,
            max_per_shape=10,
            max_total=60,
        )

        inconsistent_rows = len(contaminated_values)
        # Only treat the column as an actionable machine-format issue when the global parse signal is strongly numeric.
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

    # For non-measure columns, define the dominant format as the most common structural shape.
    shape_counts = Counter(value_shape(value) for value in rendered)
    dominant_shape, dominant_count = shape_counts.most_common(1)[0]
    dominant_shape_pct = float((dominant_count / len(rendered)) * 100)
    distinct_shape_count = len(shape_counts)

    dominant_example_values: list[str] = []
    for value in rendered:
        # Keep a few concrete examples of the dominant shape for prompt grounding.
        if value_shape(value) == dominant_shape and value not in dominant_example_values:
            dominant_example_values.append(value[:80])
        if len(dominant_example_values) >= 5:
            break

    inconsistent_rows = len(rendered) - dominant_count
    inconsistent_examples: list[FormatOutlierExample] = []
    if inconsistent_rows > 0:
        # Show only non-dominant shapes as outliers; the dominant family is already represented separately.
        inconsistent_examples = select_outlier_examples(
            rendered,
            exclude_shape=dominant_shape,
            max_shapes=10,
            max_per_shape=10,
            max_total=60,
        )

    # Heuristic thresholds vary by semantic type because "good enough" consistency differs for dates, codes, and measures.
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


# Profile Builder

def build_column_format_profile(df, column_name: str) -> ColumnFormatProfile:
    """Build the deterministic structural profile for one column.

    This lower-level profile does not decide whether a column is problematic;
    it only measures parse rates, samples, and shape distributions so later
    heuristics can make that judgment.
    """
    series = df[column_name]
    return ColumnFormatProfile(
        column_name=column_name,
        pandas_dtype=str(series.dtype),  # raw pandas dtype before any schema-guided reinterpretation
        total_rows=len(series),
        non_null_rows=int(series.notna().sum()),  # true pandas non-null count before placeholder filtering
        distinct_non_null_values=int(series.nunique(dropna=True)),  # cardinality over non-null raw values
        numeric_parse_pct=compute_numeric_parse_pct(series),  # global numeric signal for the column
        datetime_parse_pct=compute_datetime_parse_pct(series),  # global datetime signal for the column
        empty_like_pct=compute_empty_like_pct(series),  # share of rows that are empty or null-like
        sample_values=sample_non_null_values(series),  # small representative sample used in downstream prompts
        top_value_shapes=compute_top_value_shapes(series),  # most common structural layouts seen in the rendered values
    )
