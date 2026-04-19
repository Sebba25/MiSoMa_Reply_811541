from __future__ import annotations

from typing import Any

from models import ColumnCleaningRequest


def build_column_cleaning_request(
    dataset_name: str,
    column_name: str,
    finding: Any,
    format_facts: Any,
    schema_entry: Any | None = None,
) -> ColumnCleaningRequest:
    raw_examples = [example.value for example in format_facts.inconsistent_examples]
    if not raw_examples:
        raw_examples = finding.example_inconsistent_values

    target_dtype = schema_entry.pandas_dtype if schema_entry else None
    target_role = schema_entry.numeric_role or schema_entry.string_role if schema_entry else None

    return ColumnCleaningRequest(
        dataset_name=dataset_name,
        column_name=column_name,
        expected_pattern=finding.expected_pattern,
        semantic_hint=format_facts.semantic_hint,
        target_dtype=target_dtype,
        target_role=target_role,
        dominant_shape=format_facts.dominant_shape,
        dominant_example_values=format_facts.dominant_example_values,
        example_inconsistent_values=raw_examples,
        suggested_strategy=finding.suggested_strategy,
    )

