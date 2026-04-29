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

from src.core.models import ColumnCleaningRequest

#This defines a helper function that returns the expected datetime pattern as a string
def _build_datetime_expected_pattern(format_facts: Any, fallback_pattern: str) -> str:
    '''This function looks at example datetime values and tries to describe the expected datetime format in a clear human-readable way. It adds extra instructions to the cleaning strategy so datetime values are cleaned into one consistent final format'''
    #This tries to read dominant_example_values from format_facts.
    #If that attribute does not exist or is empty, it uses an empty list instead.
    dominant_examples = getattr(format_facts, "dominant_example_values", None) or []
    #This iterates through the dominant examples to find the first one that is a non-empty string, and uses that as the example for inferring the datetime format. 
    # If no such example is found, it returns the fallback pattern it received as input.
    example = next((value for value in dominant_examples if isinstance(value, str) and value.strip()), None)
    if example is None:
        return fallback_pattern
    #This removes extra spaces at the beginning and end of the example string
    stripped = example.strip()
    #This checks if the example matches a full ISO timestamp format like 2024-01-15T10:30:45.123456
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}", stripped):
        #If the example matches that format, the function returns a clear text description of it
        return "ISO timestamp YYYY-MM-DDTHH:MM:SS.ffffff"
    #This checks if the example matches a timestamp like 2024-01-15 10:30:45
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", stripped):
        return "timestamp YYYY-MM-DD HH:MM:SS"
    #This checks if the example is only a date like 2024-01-15
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
        return "date YYYY-MM-DD"
    #If the example does not match the known patterns above, the function still returns a generic description based on the example itself.
    return f"datetime format like {stripped!r}"

#This defines a helper function that improves the cleaning strategy for datetime columns
def _augment_datetime_strategy(format_facts: Any, suggested_strategy: str) -> str:
    '''This function adds specific guidance to the cleaning strategy for datetime columns, based on the dominant example values. It emphasizes the importance of preserving valid timestamps and following a consistent output format.'''
    #This gets the list of dominant example values from format_facts, or an empty list if it is missing
    dominant_examples = getattr(format_facts, "dominant_example_values", None) or []
    #This looks for the first non-empty string example in the dominant examples, which will be used as a reference for the expected datetime format. 
    # If no such example is found, it returns the original suggested strategy without augmentation.
    example = next((value for value in dominant_examples if isinstance(value, str) and value.strip()), None)
    if example is None:
        return suggested_strategy
    #This creates a multi-line string containing extra instructions for datetime cleaning
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
    #This returns the new datetime guidance plus the old strategy text, so both are included together
    return guidance + "\n\nExisting shape notes:\n" + suggested_strategy


def _augment_yyyymm_strategy(example_inconsistent_values: list[str], suggested_strategy: str) -> tuple[str, bool]:
    """Add an explicit missing-month rule for YYYYMM period keys when only the year is visible.

    Returns the augmented strategy plus a flag the validator can later enforce.
    """
    year_only_examples: list[str] = []
    for value in example_inconsistent_values:
        stripped = str(value).strip()
        if re.fullmatch(r"\d{4}", stripped) or re.fullmatch(r"[A-Za-z]+\s+\d{4}", stripped):
            year_only_examples.append(stripped)
    if not year_only_examples:
        return suggested_strategy, False

    example_text = ", ".join(repr(value) for value in year_only_examples[:5])
    guidance = (
        "YYYYMM missing-month rule:\n"
        "- If the input contains a recoverable 4-digit year but no month, default the month to '01' instead of returning None.\n"
        f"- Apply this to year-only shapes such as {example_text}.\n"
        "- Example: 'Rata 2024' -> '202401' and 'Rata 2023' -> '202301'.\n"
        "- Treat a visible year as recoverable information, not as an unrecoverable value."
    )
    return guidance + "\n\nExisting shape notes:\n" + suggested_strategy, True


def build_column_cleaning_request(
    dataset_name: str, 
    column_name: str,
    finding: Any, #This parameter contains the inconsistency finding for the column
    format_facts: Any, #This parameter contains information about the observed format of the column values
    schema_entry: Any | None = None, #This optional parameter contains schema information for the column. If it is not provided, the default is None
) -> ColumnCleaningRequest:
    '''It builds a ColumnCleaningRequest object by combining dataset information, inconsistency findings, format facts, and optional schema details'''
    #It takes the inconsistent example values from finding, removes duplicates while keeping the original order, and converts the result to a list
    example_inconsistent_values = list(dict.fromkeys(finding.example_inconsistent_values))
    #This checks whether the list is empty
    if not example_inconsistent_values:
        #If the list is empty, the code starts building a fallback list of inconsistent example values 
        # by looking at the examples in format_facts.inconsistent_examples, extracting their value attribute, removing duplicates while preserving order, and converting the result to a list
        example_inconsistent_values = list(
            dict.fromkeys(example.value for example in format_facts.inconsistent_examples)
        )
    #If schema_entry exists, this gets its pandas dtype. Otherwise, it stores None
    target_dtype = schema_entry.pandas_dtype if schema_entry else None
    #If schema_entry exists, this gets its numeric_role if it exists; if not (empty or false), it gets its string_role; if schema_entry does not exist, it stores None
    target_role = schema_entry.numeric_role or schema_entry.string_role if schema_entry else None
    #Starts with the expected pattern from the inconsistency finding
    expected_pattern = finding.expected_pattern
    #starts with the suggested cleaning strategy from the finding
    suggested_strategy = finding.suggested_strategy
    # This stays false unless the YYYYMM year-only fallback rule is explicitly activated.
    enforce_year_only_yyyymm_january = False
    #This checks if the target column type is a datetime column based on the schema information.
    # If it is, it applies datetime-specific logic, this replaces the normal expected pattern with a better datetime-specific one
    if target_dtype == "datetime64[ns]":
        expected_pattern = _build_datetime_expected_pattern(format_facts, expected_pattern)
        suggested_strategy = _augment_datetime_strategy(format_facts, suggested_strategy)
    elif expected_pattern.strip().lower().startswith("yyyymm"):
        suggested_strategy, enforce_year_only_yyyymm_january = _augment_yyyymm_strategy(
            example_inconsistent_values,
            suggested_strategy,
        )

    #starts creating and returning the final ColumnCleaningRequest object
    return ColumnCleaningRequest(
        dataset_name=dataset_name, 
        column_name=column_name,
        expected_pattern=expected_pattern,
        semantic_hint=format_facts.semantic_hint, #Adds the semantic hint from the format facts
        target_dtype=target_dtype,
        target_role=target_role,
        dominant_shape=format_facts.dominant_shape, #Adds the dominant structural shape of the values
        dominant_example_values=format_facts.dominant_example_values,
        example_inconsistent_values=example_inconsistent_values,
        enforce_year_only_yyyymm_january=enforce_year_only_yyyymm_january,
        suggested_strategy=suggested_strategy,
    )
