"""Builds ``ColumnCleaningRequest`` from a consistency finding + format facts.

Merges schema, completeness, and format-profile signals into a single bundle
fed to the generator agent. Includes datetime-specific augmentation: when the
target dtype is ``datetime64[ns]`` we derive a canonical expected pattern
from dominant examples and prepend a datetime output contract to the
suggested strategy.
"""

from __future__ import annotations

import re
from typing import Any

from models import ColumnCleaningRequest


def _build_datetime_expected_pattern(format_facts: Any, fallback_pattern: str) -> str:
    dominant_examples = getattr(format_facts, "dominant_example_values", None) or []
    example = next((value for value in dominant_examples if isinstance(value, str) and value.strip()), None)
    if example is None:
        return fallback_pattern

    stripped = example.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}", stripped):
        return "ISO timestamp YYYY-MM-DDTHH:MM:SS.ffffff"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", stripped):
        return "timestamp YYYY-MM-DD HH:MM:SS"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
        return "date YYYY-MM-DD"
    return f"datetime format like {stripped!r}"


def _augment_datetime_strategy(format_facts: Any, suggested_strategy: str) -> str:
    dominant_examples = getattr(format_facts, "dominant_example_values", None) or []
    example = next((value for value in dominant_examples if isinstance(value, str) and value.strip()), None)
    if example is None:
        return suggested_strategy

    guidance = (
        "Datetime output contract:\n"
        f"- Preserve already-valid dominant timestamps unchanged, for example {example!r}.\n"
        "- The cleaned output must use that same canonical datetime layout, including the same date order, separator style, "
        "time component, and fractional-second precision.\n"
        "- For date-only inputs, emit midnight in that same canonical layout.\n"
        "- For slash, dot, or hyphen separated numeric dates, infer component order from token widths: if the first token has "
        "4 digits, treat it as year-first; otherwise treat it as day-first unless stronger evidence says otherwise.\n"
        "- Do not just replace separators blindly. Reorder components explicitly before formatting the final timestamp.\n"
        "- If any earlier heuristic suggests a separator-only rewrite that leaves the date order ambiguous or wrong, ignore it and "
        "follow the canonical output format above."
    )
    return guidance + "\n\nExisting shape notes:\n" + suggested_strategy


def build_column_cleaning_request(
    dataset_name: str,
    column_name: str,
    finding: Any,
    format_facts: Any,
    schema_entry: Any | None = None,
) -> ColumnCleaningRequest:
    example_inconsistent_values = list(dict.fromkeys(finding.example_inconsistent_values))
    if not example_inconsistent_values:
        example_inconsistent_values = list(
            dict.fromkeys(example.value for example in format_facts.inconsistent_examples)
        )

    target_dtype = schema_entry.pandas_dtype if schema_entry else None
    target_role = schema_entry.numeric_role or schema_entry.string_role if schema_entry else None
    expected_pattern = finding.expected_pattern
    suggested_strategy = finding.suggested_strategy

    if target_dtype == "datetime64[ns]":
        expected_pattern = _build_datetime_expected_pattern(format_facts, expected_pattern)
        suggested_strategy = _augment_datetime_strategy(format_facts, suggested_strategy)

    return ColumnCleaningRequest(
        dataset_name=dataset_name,
        column_name=column_name,
        expected_pattern=expected_pattern,
        semantic_hint=format_facts.semantic_hint,
        target_dtype=target_dtype,
        target_role=target_role,
        dominant_shape=format_facts.dominant_shape,
        dominant_example_values=format_facts.dominant_example_values,
        example_inconsistent_values=example_inconsistent_values,
        suggested_strategy=suggested_strategy,
    )

