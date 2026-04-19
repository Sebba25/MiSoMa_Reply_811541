from __future__ import annotations

from collections import Counter

import pandas as pd

from models import CleanerValidationIssue, ColumnCleanerProgram, ColumnCleaningRequest, ExampleTransformation
from tools.tools import value_shape

from .runtime import load_cleaner_callable


def dominant_output_shape(request: ColumnCleaningRequest) -> str | None:
    shapes = [value_shape(value) for value in request.dominant_example_values if isinstance(value, str) and value]
    if not shapes:
        return None
    return Counter(shapes).most_common(1)[0][0]


def is_parseable_output(value: str, target_dtype: str | None) -> bool:
    if target_dtype == "datetime64[ns]":
        parsed = pd.to_datetime(pd.Series([value]), errors="coerce", format="mixed")
        return bool(parsed.notna().iloc[0])
    if target_dtype in {"Int64", "Float64"}:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce")
        return bool(parsed.notna().iloc[0])
    if target_dtype == "boolean":
        return value.strip().lower() in {"true", "false", "1", "0", "yes", "no", "si", "s\u00ec"}
    return True


def requires_fixed_output_shape(request: ColumnCleaningRequest) -> bool:
    if request.target_dtype in {"Int64", "Float64"} and request.target_role == "measure":
        return False
    return True


def build_validation_issue(
    *,
    category: str,
    severity: str,
    message: str,
    expected_behavior: str,
    input_value: str | None = None,
    actual_output: str | None = None,
) -> CleanerValidationIssue:
    return CleanerValidationIssue(
        category=category,
        severity=severity,
        message=message,
        input_value=input_value,
        actual_output=actual_output,
        expected_behavior=expected_behavior,
    )


def detect_shadowed_delimiter_branches(program: ColumnCleanerProgram) -> list[CleanerValidationIssue]:
    issues: list[CleanerValidationIssue] = []
    lines = program.python_code.splitlines()
    delimiter_specs = {
        "/": ("s.count('/')", "split('/')", "re.match(", "re.fullmatch("),
        "-": ("s.count('-')", "split('-')", "re.match(", "re.fullmatch("),
        ".": ("s.count('.')", "split('.')", "re.match(", "re.fullmatch("),
    }

    for delimiter, specific_markers in delimiter_specs.items():
        generic_branches: list[tuple[int, int]] = []
        for index, line in enumerate(lines, start=1):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip(" "))
            if stripped.startswith(("if ", "elif ")) and (
                f"'{delimiter}' in s" in stripped or f'"{delimiter}" in s' in stripped
            ):
                generic_branches.append((index, indent))

        for generic_line, generic_indent in generic_branches:
            for later_index, later_line in enumerate(lines[generic_line:], start=generic_line + 1):
                stripped = later_line.strip()
                indent = len(later_line) - len(later_line.lstrip(" "))
                if indent != generic_indent or not stripped.startswith(("if ", "elif ")):
                    continue
                if f"'{delimiter}' in s" in stripped or f'"{delimiter}" in s' in stripped:
                    continue
                if delimiter not in stripped:
                    continue
                if any(marker in stripped for marker in specific_markers):
                    issues.append(
                        build_validation_issue(
                            category="shadowed_specific_branch",
                            severity="high",
                            message=(
                                f"Generic delimiter branch on line {generic_line} for {delimiter!r} appears before "
                                f"a more specific branch on line {later_index}, which can make the later branch unreachable."
                            ),
                            expected_behavior=(
                                "place the more specific pattern branch before the generic delimiter-based branch, "
                                "or make the branches mutually exclusive."
                            ),
                        )
                    )
                    break
            if issues:
                break

    return issues


def validate_generated_cleaner_program(
    request: ColumnCleaningRequest,
    program: ColumnCleanerProgram,
) -> list[CleanerValidationIssue]:
    issues: list[CleanerValidationIssue] = []

    if program.column_name != request.column_name:
        issues.append(
            build_validation_issue(
                category="program_mismatch",
                severity="high",
                message=f"Program column_name was {program.column_name!r}, expected {request.column_name!r}.",
                expected_behavior=f"column_name must equal {request.column_name!r}.",
            )
        )

    try:
        cleaner = load_cleaner_callable(program)
    except Exception as error:
        return [
            build_validation_issue(
                category="runtime_exception",
                severity="high",
                message=f"Generated cleaner code could not be loaded: {error}",
                expected_behavior="generated python_code must load into a callable cleaner without exceptions.",
            )
        ]

    issues.extend(detect_shadowed_delimiter_branches(program))

    target_shape = dominant_output_shape(request)
    require_fixed_shape = requires_fixed_output_shape(request)

    for value in request.dominant_example_values:
        try:
            cleaned = cleaner(value)
        except Exception as error:
            issues.append(
                build_validation_issue(
                    category="runtime_exception",
                    severity="high",
                    message=f"Dominant example {value!r} raised {type(error).__name__}: {error}.",
                    input_value=value,
                    expected_behavior="already-valid dominant examples must execute successfully and be preserved exactly.",
                )
            )
            continue

        cleaned_str = None if cleaned is None else str(cleaned)
        if cleaned != value:
            issues.append(
                build_validation_issue(
                    category="dominant_value_modified",
                    severity="high",
                    message=f"Dominant example {value!r} changed to {cleaned_str!r}; already-valid values must be preserved exactly.",
                    input_value=value,
                    actual_output=cleaned_str,
                    expected_behavior="return the dominant example unchanged.",
                )
            )

    for value in request.example_inconsistent_values:
        try:
            cleaned = cleaner(value)
        except Exception as error:
            issues.append(
                build_validation_issue(
                    category="runtime_exception",
                    severity="high",
                    message=f"Inconsistent example {value!r} raised {type(error).__name__}: {error}.",
                    input_value=value,
                    expected_behavior="recoverable outliers must execute successfully and be transformed.",
                )
            )
            continue

        if cleaned is None:
            issues.append(
                build_validation_issue(
                    category="outlier_returned_none",
                    severity="high",
                    message=f"Inconsistent example {value!r} returned None; recoverable outliers must be converted.",
                    input_value=value,
                    expected_behavior="transform the outlier into a valid cleaned value instead of None.",
                )
            )
            continue

        cleaned_str = str(cleaned)
        if cleaned == value:
            issues.append(
                build_validation_issue(
                    category="outlier_unchanged",
                    severity="high",
                    message=f"Inconsistent example {value!r} was returned unchanged.",
                    input_value=value,
                    actual_output=cleaned_str,
                    expected_behavior="the outlier should be normalized, not passed through unchanged.",
                )
            )

        if require_fixed_shape and target_shape is not None and value_shape(cleaned_str) != target_shape:
            issues.append(
                build_validation_issue(
                    category="wrong_output_shape",
                    severity="medium",
                    message=(
                        f"Inconsistent example {value!r} cleaned to {cleaned_str!r} with shape "
                        f"{value_shape(cleaned_str)!r}, expected dominant output shape {target_shape!r}."
                    ),
                    input_value=value,
                    actual_output=cleaned_str,
                    expected_behavior=f"produce output matching the dominant structural shape {target_shape!r}.",
                )
            )

        if not is_parseable_output(cleaned_str, request.target_dtype):
            issues.append(
                build_validation_issue(
                    category="not_parseable_as_target_dtype",
                    severity="high",
                    message=f"Inconsistent example {value!r} cleaned to {cleaned_str!r}, which is not parseable as {request.target_dtype}.",
                    input_value=value,
                    actual_output=cleaned_str,
                    expected_behavior=f"produce a value parseable as {request.target_dtype}.",
                )
            )

    return issues


def rebuild_verified_program(
    request: ColumnCleaningRequest,
    program: ColumnCleanerProgram,
) -> ColumnCleanerProgram:
    cleaner = load_cleaner_callable(program)
    transformations: list[ExampleTransformation] = []

    for value in request.dominant_example_values:
        cleaned = cleaner(value)
        transformations.append(
            ExampleTransformation(
                original_value=value,
                cleaned_value=None if cleaned is None else str(cleaned),
                rationale="Preserved already-valid example during host verification.",
            )
        )

    for value in request.example_inconsistent_values:
        cleaned = cleaner(value)
        transformations.append(
            ExampleTransformation(
                original_value=value,
                cleaned_value=None if cleaned is None else str(cleaned),
                rationale="Converted during host-side verification of the accepted cleaner.",
            )
        )

    verification_summary = (
        f"Host validation passed: preserved {len(request.dominant_example_values)} dominant examples "
        f"and converted {len(request.example_inconsistent_values)} inconsistent examples."
    )

    return program.model_copy(
        update={
            "example_transformations": transformations,
            "verification_summary": verification_summary,
        }
    )


def format_validation_issue(issue: CleanerValidationIssue) -> str:
    parts = [issue.message]
    if issue.input_value is not None:
        parts.append(f"input={issue.input_value!r}")
    if issue.actual_output is not None:
        parts.append(f"actual={issue.actual_output!r}")
    parts.append(f"expected={issue.expected_behavior}")
    return " | ".join(parts)


def format_validation_examples(issues: list[CleanerValidationIssue], limit: int = 5) -> str:
    lines: list[str] = []
    for issue in issues[:limit]:
        line = f"- category={issue.category}"
        if issue.input_value is not None:
            line += f", input={issue.input_value!r}"
        if issue.actual_output is not None:
            line += f", actual={issue.actual_output!r}"
        line += f", expected={issue.expected_behavior}"
        lines.append(line)
    return "\n".join(lines)


def validation_issue_fingerprint(issues: list[CleanerValidationIssue], limit: int = 10) -> tuple[str, ...]:
    return tuple(
        f"{issue.category}|{issue.input_value}|{issue.actual_output}|{issue.expected_behavior}"
        for issue in issues[:limit]
    )

