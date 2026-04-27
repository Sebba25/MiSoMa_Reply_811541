"""Host-side validator for generated cleaner programs.

Runs each ``ColumnCleanerProgram`` against the dominant and inconsistent
example values from its ``ColumnCleaningRequest`` and returns a list of
``CleanerValidationIssue``s covering the known failure categories:

    * ``non_self_contained_function`` / ``runtime_exception``
    * ``shadowed_specific_branch``  (static pattern check)
    * ``dominant_value_modified``   (must be identity on valid inputs)
    * ``outlier_unchanged`` / ``wrong_output_shape``
    * ``not_parseable_as_target_dtype`` / ``not_matching_target_pattern``

The generator/critic loop reads these issues to decide whether to retry, ask
the critic, or escalate temperature.
"""

from __future__ import annotations

from collections import Counter
import re

import pandas as pd

from core.models import CleanerValidationIssue, ColumnCleanerProgram, ColumnCleaningRequest, ExampleTransformation
from tools import matches_numeric_schema_pattern, value_shape

from .runtime import load_cleaner_callable


def dominant_output_shape(request: ColumnCleaningRequest) -> str | None:
    '''Returns the most common visible shape among the dominant example values, or None if no valid examples exist.'''
    # Build a list of shapes only from non-empty string values among the dominant examples, since those are the only ones relevant for shape validation.
    shapes = [value_shape(value) for value in request.dominant_example_values if isinstance(value, str) and value]
    # If no valid examples exist, there is no dominant shape to return
    if not shapes:
        return None
    # Return the shape that appears most often
    return Counter(shapes).most_common(1)[0][0]



def is_parseable_output(value: str, target_dtype: str | None) -> bool:
    '''Checks whether a cleaned value can be interpreted as the target data type'''
    # For datetime targets, let pandas try to parse the value as a date/time
    if target_dtype == "datetime64[ns]":
        parsed = pd.to_datetime(pd.Series([value]), errors="coerce", format="mixed")
        return bool(parsed.notna().iloc[0])
    # For numeric targets, let pandas try to convert the value to a number
    if target_dtype in {"Int64", "Float64"}:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce")
        # If conversion failed, the output is not valid for the target dtype 
        if parsed.isna().iloc[0]:
            return False
        # Integer outputs must not contain a decimal part, even if they can be parsed as floats; 
        # check that the parsed number is an integer when the target dtype is Int64.
        if target_dtype == "Int64":
            return float(parsed.iloc[0]).is_integer()
        return True
    # For booleans, accept only a known set of true/false tokens.
    if target_dtype == "boolean":
        return value.strip().lower() in {"true", "false", "1", "0", "yes", "no", "si", "s\u00ec"}
    # Other target types are accepted without extra parsing checks here 
    # because they are either too flexible (e.g. string) or will be checked by pandas during the actual cleaning process (e.g. categorical with limited categories).
    return True


def requires_fixed_output_shape(request: ColumnCleaningRequest) -> bool:
    '''Decides whether cleaned outputs must follow one fixed visible shape based on the target dtype and pattern requirements.'''
    # Numeric strings may vary in shape and still be valid numbers, so shape validation is not required for numeric targets.
    if request.target_dtype in {"Int64", "Float64"}:
        return False
    # For all other target types, the validator expects a consistent shape
    return True


def matches_request_target_pattern(value: str, request: ColumnCleaningRequest) -> bool | None:
    '''Checks whether a numeric output matches the numeric pattern requested by the schema, or returns None if no pattern check is applicable.'''
    # This pattern check is used only for numeric target dtypes
    if request.target_dtype not in {"Int64", "Float64"}:
        return None
    # Delegate the detailed pattern matching to the shared helper
    return matches_numeric_schema_pattern(
        value,
        pandas_dtype=request.target_dtype,
        numeric_role=request.target_role,
        detected_pattern=request.expected_pattern,
    )

def _datetime_format_regex_from_example(example: str) -> re.Pattern[str] | None:
    '''Converts a datetime example string into a regex pattern that enforces the same visible layout of digits and separators.'''
    # This checks if the input string is empty or missing, if the example is empty, the function cannot build a useful regex, so it returns None
    if not example:
        return None

    # This creates an empty list called parts, the list will store the different pieces of the regex that will be built step by step
    parts: list[str] = []
    #It is used to track the current position while scanning the example string
    cursor = 0
    # It looks through the example string and finds every group of one or more digits
    for match in re.finditer(r"\d+", example):
        # Extract the start and end indices of the current digit group inside the string
        start, end = match.span()
        # This checks whether there is some non-digit text between the current cursor position and the start of the digit group
        if start > cursor:
            #If there is separator text, this adds it to the regex, re.escape(...) is used so special characters are treated literally
            parts.append(re.escape(example[cursor:start]))
        # Replace the digits with a regex that expects the same number of digits
        parts.append(rf"\d{{{end - start}}}")
        #This moves the cursor to the end of the current digit group, so the next iteration starts from there
        cursor = end
    # After the loop is finished, this checks whether there is any remaining text at the end of the string after the last digit group
    if cursor < len(example):
        #If there is trailing non-digit text, it adds that final separator part to the regex, again escaped safely
        parts.append(re.escape(example[cursor:]))
    # If nothing useful was collected, stop here
    if not parts:
        #the function returns None because it could not build a useful pattern
        return None
    # Compile a full regex that matches the same datetime layout
    return re.compile("^" + "".join(parts) + "$")

def dominant_datetime_example(request: ColumnCleaningRequest) -> str | None:
    '''Returns the first usable dominant datetime example for the request, or None if no such example exists'''
    # If the target type is not datetime, there is no datetime example to return
    if request.target_dtype != "datetime64[ns]":
        return None
    # Return the first non-empty datetime example
    for value in request.dominant_example_values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def matches_dominant_datetime_format(value: str, request: ColumnCleaningRequest) -> bool | None:
    '''defines a function that checks whether a cleaned datetime string has the same format layout as the dominant datetime example for the request, or returns None if no such check can be performed.'''
    # Get the reference datetime example
    example = dominant_datetime_example(request)
    #This checks whether no reference example exists or if the example is empty, in which case the function cannot perform a format check and returns None
    if example is None:
        return None
    # Convert that example into a regex pattern
    pattern = _datetime_format_regex_from_example(example)
    if pattern is None:
        return None
    # This checks whether the whole cleaned value matches the regex. It returns True if the format shape matches, otherwise False
    return bool(pattern.fullmatch(value))


def _suggest_corrected_datetime(raw_input: str, canonical_example: str) -> str | None:
    """It tries to build a corrected datetime string using the canonical example’s format as a template. This is used to provide helpful suggestions when the cleaned output has the right digits but the wrong format."""
    try:
        # Try to parse the raw input as a datetime value
        #dayfirst=True tells pandas to prefer day-month-year when ambiguous. format="mixed" allows multiple possible format
        parsed = pd.to_datetime(raw_input, dayfirst=True, format="mixed")
    except Exception:
        # If parsing fails, no corrected suggestion can be produced
        return None

    # This checks whether the canonical example looks like a timestamp style where date-only values should become midnight
    midnight_tail = _midnight_tail_like(canonical_example)
    if midnight_tail is not None:
        return parsed.strftime("%Y-%m-%d") + midnight_tail

    # Otherwise, tries to capture everything that comes after the YYYY-MM-DD part in the canonical example and append it to the parsed date, 
    # so that the suggestion keeps the same extra formatting or timezone information as the canonical example.
    tail_match = re.match(r"^\d{4}-\d{2}-\d{2}(.*)", canonical_example)
    #If the regex matched, this gets the trailing part. If not, it uses an empty string
    tail = tail_match.group(1) if tail_match else ""
    #This returns the parsed date formatted as YYYY-MM-DD, plus the same trailing time part as the canonical example
    return parsed.strftime("%Y-%m-%d") + tail


def _midnight_tail_like(canonical_example: str) -> str | None:
    """Build a midnight time suffix with the same separator and fractional precision as a canonical timestamp."""
    # Match canonical timestamps such as YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD HH:MM:SS.ffffff
    match = re.match(r"^\d{4}-\d{2}-\d{2}([T ])\d{2}:\d{2}:\d{2}(\.\d+)?$", canonical_example)
    # If the example is not a full timestamp, no midnight suffix can be created
    if not match:
        return None
    # Extract the separator and optional fractional part from the canonical example
    separator, fractional = match.groups()
    #This checks whether fractional seconds exist
    if fractional:
        # Keep the same fractional precision, but replace the digits with zeros
        fractional = "." + ("0" * (len(fractional) - 1))
    #This handles the case with no fractional seconds, the fractional part becomes an empty string
    else:
        fractional = ""
    # Return a midnight time part with the same formatting style.
    return f"{separator}00:00:00{fractional}"


def _diagnose_datetime_component_order(
    raw_input: str, #original input string
    actual_output: str, #cleaner’s produced output
    canonical_example: str, #expected canonical example
) -> str:
    """Return a one-sentence diagnosis when when date parts are in the wrong order."""
    # Extract numeric groups from the input, output, and canonical example
    input_groups = re.findall(r"\d+", raw_input)  #extracts all digit groups from the input
    actual_groups = re.findall(r"\d+", actual_output)[:3] #extracts the first three digit groups from the cleaner’s output
    canonical_groups = re.findall(r"\d+", canonical_example)[:3] #xtracts the first three digit groups from the canonical example
    # If there is not enough information, no diagnosis can be made
    if not (input_groups and canonical_groups):
        return ""
    # Try to identify the year and check whether it was moved incorrectly into the wrong position. 
    # The year is the most distinctive component and the most likely to be misplaced, so diagnosing it can provide useful hints for correction.
    canonical_year = next((g for g in canonical_groups if len(g) == 4), None) #looks for the first 4-digit group in the canonical example. That is assumed to be the year
    #takes the first numeric group from the output, if one exists
    actual_year = actual_groups[0] if actual_groups else None
    #checks whether both years exist and the output seems to place a different first group than expected
    if canonical_year and actual_year and canonical_year != actual_year:
        # This case means the original year may have been merged into the wrong number
        if canonical_year in input_groups and actual_year not in input_groups:
            #eturns a detailed diagnosis saying the 4-digit year was recombined incorrectly and that the code should identify the year explicitly instead of relying on position
            return (
                f" The 4-digit year {canonical_year!r} from the input was recombined incorrectly into {actual_year!r}."
                f" Split on the delimiter and identify the 4-digit component explicitly (e.g. int(part) > 999) rather than relying on index position."
            )
        # This case means the year exists but was placed in the wrong position
        if canonical_year in input_groups:
            #returns a diagnosis saying the year should appear first in YYYY-MM-DD order and the code should reorder components correctly
            return (
                f" The 4-digit year {canonical_year!r} must appear first in the output (YYYY-MM-DD);"
                f" your output put it as {actual_year!r}. Reorder components: identify which split part is the year, which is the month, which is the day."
            )
    return ""


def build_validation_issue(
    *, #all following parameters must be passed by keyword to ensure clarity when building issues
    category: str,
    severity: str,
    message: str,
    expected_behavior: str,
    input_value: str | None = None,
    actual_output: str | None = None,
) -> CleanerValidationIssue:
    '''Build one standardized validation issue object. defines a helper function that creates a CleanerValidationIssue object'''
    return CleanerValidationIssue(
        category=category,
        severity=severity,
        message=message,
        input_value=input_value,
        actual_output=actual_output,
        expected_behavior=expected_behavior,
    )


def build_runtime_exception_issue(
    *, #All parameters must be passed by keyword for clarity when building issues
    stage_label: str, #where the error happened, for example on a dominant example or inconsistent example
    input_value: str | None, #value being processed when the error happened
    error: Exception, #exception object
) -> CleanerValidationIssue:
    '''defines a helper function that turns a runtime exception into a validation issue, with special handling for NameError to detect non-self-contained functions.'''
    # NameError usually means the generated function depends on missing external names
    if isinstance(error, NameError):
        #If it is a NameError, build a special issue that explains the likely cause 
        # and emphasizes the requirement for self-contained functions without external dependencies or scratchpad variables.
        return build_validation_issue(
            category="non_self_contained_function",
            severity="high",
            message=(
                f"{stage_label} {input_value!r} raised NameError: {error}. "
                "The generated function is not self-contained and referenced an undefined outer-scope name."
            ),
            input_value=input_value,
            expected_behavior=(
                "the final function must be fully self-contained, define all helper data it uses internally, "
                "and run in a fresh module without scratchpad variables such as dominant or inconsistent."
            ),
        )

    # If the error is not a NameError, build a generic runtime exception issue 
    # So, any other exception is treated as a general runtime failure.
    return build_validation_issue(
        category="runtime_exception",
        severity="high",
        #message says which stage failed, on which input, and which exception was raised
        message=f"{stage_label} {input_value!r} raised {type(error).__name__}: {error}.", 
        input_value=input_value,
        #explains what the function should have done
        expected_behavior="the cleaner must execute successfully for every validation example.",
    )

def detect_shadowed_delimiter_branches(program: ColumnCleanerProgram) -> list[CleanerValidationIssue]:
    '''It statically inspects generated code to find branch-ordering problems.'''
    #starts an empty list of issues
    issues: list[CleanerValidationIssue] = []
    # Split the generated Python code into separate lines.
    lines = program.python_code.splitlines()
    # Define the markers used to detect generic and more specific delimiter logic
    #starts a dictionary of delimiters to inspect
    delimiter_specs = {
        "/": ("s.count('/')", "split('/')", "re.match(", "re.fullmatch("), #defines markers for slash-based logic
        "-": ("s.count('-')", "split('-')", "re.match(", "re.fullmatch("), #defines markers for dash-based logic
        ".": ("s.count('.')", "split('.')", "re.match(", "re.fullmatch("), #defines markers for dot-based logic
    }

    # Check each delimiter family one by one
    for delimiter, specific_markers in delimiter_specs.items():
        #creates a list to store generic delimiter branches found in the code, each item is a tuple of (line number, indentation level)
        generic_branches: list[tuple[int, int]] = []
        # Find generic branches such as `if '/' in s`. It loops through every line of code, starting line numbers from 1
        for index, line in enumerate(lines, start=1):
            stripped = line.strip() #removes leading and trailing whitespace from the line, so it can be analyzed more easily
            indent = len(line) - len(line.lstrip(" ")) #it calculates the indentation level by counting leading spaces
            #checks whether the line begins with if or elif
            if stripped.startswith(("if ", "elif ")) and (
                #checks whether the line contains a generic delimiter check such as if '-' in s
                f"'{delimiter}' in s" in stripped or f'"{delimiter}" in s' in stripped
            ):
                #If the line is a generic delimiter branch, save its line number and indentation level in the generic_branches list for later analysis
                generic_branches.append((index, indent))

        # loop through every generic branch found and check whether there is a more specific branch later in the code that could be shadowed by it
        for generic_line, generic_indent in generic_branches:
            #Look at later lines after the generic branch
            for later_index, later_line in enumerate(lines[generic_line:], start=generic_line + 1):
                stripped = later_line.strip()
                indent = len(later_line) - len(later_line.lstrip(" ")) #Compute its indentation level.
                #If the later line is not at the same indentation level or is not another conditional branch, skip it
                if indent != generic_indent or not stripped.startswith(("if ", "elif ")):
                    continue
                #If the later line is also generic, skip it because it does not create a shadowing problem
                if f"'{delimiter}' in s" in stripped or f'"{delimiter}" in s' in stripped:
                    continue
                #If the later line does not even mention the delimiter, skip it
                if delimiter not in stripped:
                    continue
                #checks whether the later line looks like a more specific branch for that delimiter
                if any(marker in stripped for marker in specific_markers):
                    #If such a specific branch appears later, add an issue 
                    # because the generic branch would shadow it and make it unreachable for inputs that match both patterns. 
                    # The issue message explains the problem and advises to reorder the branches or make them mutually exclusive.
                    issues.append(
                        build_validation_issue(
                            category="shadowed_specific_branch",
                            severity="high",
                            #explains that a generic delimiter branch appears before a more specific one and may make the later one unreachable
                            message=(
                                f"Generic delimiter branch on line {generic_line} for {delimiter!r} appears before "
                                f"a more specific branch on line {later_index}, which can make the later branch unreachable."
                            ),
                            #explains that the more specific branch should come first, or the branches should be mutually exclusive
                            expected_behavior=(
                                "place the more specific pattern branch before the generic delimiter-based branch, "
                                "or make the branches mutually exclusive."
                            ),
                        )
                    )
                    #Once one conflicting later branch is found, stop scanning later lines for that generic branch
                    break
            # Stop early if an issue has already been found.
            if issues:
                break

    return issues

def validate_generated_cleaner_program(
    request: ColumnCleaningRequest, #cleaning request that contains dominant examples, inconsistent examples, target dtype, and expected pattern
    program: ColumnCleanerProgram, #generated cleaner program to validate
) -> list[CleanerValidationIssue]:
    '''Validates a generated cleaner by running it on valid and invalid example values'''
    issues: list[CleanerValidationIssue] = [] #Starts with an empty list of issues

    # First check that the cleaner was generated for the correct column
    if program.column_name != request.column_name:
        issues.append(
            build_validation_issue(
                category="program_mismatch",
                severity="high",
                message=f"Program column_name was {program.column_name!r}, expected {request.column_name!r}.",
                expected_behavior=f"column_name must equal {request.column_name!r}.",
            )
        )

    # It tries to load the cleaner code into a callable Python function
    try:
        cleaner = load_cleaner_callable(program)
    except Exception as error: #If loading fails, catch the error
        # Report NameError separately because it suggests missing outer-scope dependencies
        if isinstance(error, NameError):
            return [
                build_validation_issue(
                    category="non_self_contained_function",
                    severity="high",
                    message=(
                        f"Generated cleaner code could not be loaded because it referenced an undefined name: {error}. "
                        "The function is not self-contained."
                    ),
                    expected_behavior=(
                        "generated python_code must load into a callable cleaner without relying on outer-scope "
                        "variables, scratchpad names, or test-block state."
                    ),
                )
            ]
        # Report any other load failure as a runtime exception
        return [
            build_validation_issue(
                category="runtime_exception",
                severity="high",
                message=f"Generated cleaner code could not be loaded: {error}",
                expected_behavior="generated python_code must load into a callable cleaner without exceptions.",
            )
        ]

    # Add static code issues found before running the cleaner
    issues.extend(detect_shadowed_delimiter_branches(program))

    # Precompute the expected output shape and canonical datetime example
    target_shape = dominant_output_shape(request) #Compute the most common dominant output shape
    require_fixed_shape = requires_fixed_output_shape(request) #Decide whether fixed output shape should be enforced
    target_datetime_example = dominant_datetime_example(request) #Get the canonical dominant datetime example, if one exists

    # Dominant examples are already valid, so the cleaner must preserve them exactly
    for value in request.dominant_example_values:
        try:
            cleaned = cleaner(value) #Apply the generated cleaner to the dominant example value
        except Exception as error:
            #Add a runtime exception issue for that dominant example if the cleaner raised an error when processing it, since dominant examples should be handled without errors
            issues.append(build_runtime_exception_issue(stage_label="Dominant example", input_value=value, error=error))
            continue
        #Convert the cleaned result to string unless it is None, so it can be compared and reported in issues
        cleaned_str = None if cleaned is None else str(cleaned)
        # Flag any dominant example that was changed by the cleaner
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

    # Now loop through all inconsistent examples. Inconsistent examples should be normalized into the target format.
    for value in request.example_inconsistent_values:
        try:
            cleaned = cleaner(value) #run the cleaner
        except Exception as error:
            #Add a runtime issue
            issues.append(build_runtime_exception_issue(stage_label="Inconsistent example", input_value=value, error=error))
            continue
        #if cleaner returns None,skip further checks for that example
        if cleaned is None:
            continue

        cleaned_str = str(cleaned) #Convert the cleaned output to string
        # Flag any inconsistent example that was returned unchanged
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

        # Only do output-shape validation if fixed shape is required.
        if require_fixed_shape:
            # Datetime outputs must match the canonical datetime layout
            if request.target_dtype == "datetime64[ns]" and target_datetime_example is not None:
                if matches_dominant_datetime_format(cleaned_str, request) is False:
                    # Build extra help text when the datetime component order looks wrong.
                    suggested = _suggest_corrected_datetime(value, target_datetime_example) #Try to build a suggested corrected datetime
                    diagnosis = _diagnose_datetime_component_order(value, cleaned_str, target_datetime_example) #Try to generate a diagnosis of component-order mistakes
                    suggestion_note = f" Correct output for this input: {suggested!r}." if suggested else "" #If a suggested corrected value exists, create extra message text
                    #Add a wrong_output_shape issue because the cleaned datetime does not match the expected canonical format, and include the diagnosis and suggestion in the message to help guide correction
                    issues.append(
                        build_validation_issue(
                            category="wrong_output_shape",
                            severity="medium",
                            message=(
                                f"Inconsistent example {value!r} cleaned to {cleaned_str!r}, which does not match "
                                f"the canonical datetime format used by dominant examples such as {target_datetime_example!r}."
                                f"{diagnosis}{suggestion_note}"
                            ),
                            input_value=value,
                            actual_output=cleaned_str,
                            expected_behavior=(
                                "produce output matching the canonical datetime layout shown by dominant examples, "
                                f"for example {target_datetime_example!r}."
                                + (f" Expected for this input: {suggested!r}." if suggested else "")
                            ),
                        )
                    )
            # For non-datetime values, compare the cleaned shape with the dominant shape
            elif target_shape is not None and value_shape(cleaned_str) != target_shape:
                #If the shape is wrong, add a wrong_output_shape issue that explains the inconsistency and reports the expected dominant shape, so the developer can adjust the code to produce consistent output shapes.
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

        # Check that the cleaned value can actually be parsed as the requested dtype
        if not is_parseable_output(cleaned_str, request.target_dtype):
            #If not, add a parsing issue
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
            continue

        # If the request defines a numeric pattern, verify the cleaned value also matches it
        pattern_match = matches_request_target_pattern(cleaned_str, request)
        #If it explicitly fails
        if pattern_match is False:
            #add a pattern mismatch issue because the cleaned value does not conform to the expected numeric pattern defined in the request, 
            #which may indicate that the cleaner is not correctly normalizing numeric formats as required by the schema.
            issues.append(
                build_validation_issue(
                    category="not_matching_target_pattern",
                    severity="high",
                    message=(
                        f"Inconsistent example {value!r} cleaned to {cleaned_str!r}, which does not match "
                        f"the target pattern {request.expected_pattern!r}."
                    ),
                    input_value=value,
                    actual_output=cleaned_str,
                    expected_behavior=f"produce a value matching the target pattern {request.expected_pattern!r}.",
                )
            )

    return issues


def rebuild_verified_program(
    request: ColumnCleaningRequest, #request containing the example values
    program: ColumnCleanerProgram, #accepted cleaner program
) -> ColumnCleanerProgram:
    '''Rebuilds an accepted cleaner program with verified example transformations attached. This allows the system to keep track of how the cleaner behaves on validation examples and use that information for monitoring, debugging, or future improvement of the cleaner.'''
    # Load the cleaner so it can be replayed on the validation examples
    cleaner = load_cleaner_callable(program)
    #Creates an empty list to store before/after example transformations
    transformations: list[ExampleTransformation] = []

    # Save how the cleaner behaves on already-valid dominant examples
    #Loop through dominant examples
    for value in request.dominant_example_values:
        #Run the cleaner on the dominant value
        cleaned = cleaner(value)
        #Add an ExampleTransformation that captures how the cleaner processed the dominant example during verification, so we have a record of whether it preserved the value as expected or not. This is useful for monitoring and debugging the cleaner’s behavior on these important reference examples.
        transformations.append(
            ExampleTransformation( #Create the transformation object
                original_value=value,
                cleaned_value=None if cleaned is None else str(cleaned),
                rationale="Preserved already-valid example during host verification.",
            )
        )

    # Save how the cleaner transforms inconsistent examples
    #Loop through inconsistent examples
    for value in request.example_inconsistent_values:
        cleaned = cleaner(value)
        #Add another transformation that captures how the cleaner transformed the inconsistent example during verification, so we have a record of the cleaner’s actual behavior on these examples.
        transformations.append( 
            ExampleTransformation(
                original_value=value,
                cleaned_value=None if cleaned is None else str(cleaned),
                rationale="Converted during host-side verification of the accepted cleaner.",
            )
        )

    # Build a summary string saying how many dominant examples were preserved and how many inconsistent examples were converted
    verification_summary = (
        f"Host validation passed: preserved {len(request.dominant_example_values)} dominant examples "
        f"and converted {len(request.example_inconsistent_values)} inconsistent examples."
    )

    # Return a copy of the original program with updated fields
    return program.model_copy(
        update={
            "example_transformations": transformations,
            "verification_summary": verification_summary,
            "residual_risks": [],
        }
    )

def format_validation_issue(issue: CleanerValidationIssue) -> str:
    '''Formats one validation issue as a compact readable string that includes the main message and relevant details such as the input, actual output, and expected behavior. This is used to create clear and informative issue reports for developers when a cleaner fails validation.'''
    # Start with the main issue message.
    parts = [issue.message]
    # Add the original input if available
    if issue.input_value is not None:
        parts.append(f"input={issue.input_value!r}")
    # Add the produced output if available
    if issue.actual_output is not None:
        parts.append(f"actual={issue.actual_output!r}")
    # Always include the expected behavior for context
    parts.append(f"expected={issue.expected_behavior}")
    # Join the pieces into one readable line
    return " | ".join(parts)

def format_validation_examples(issues: list[CleanerValidationIssue], limit: int = 5) -> str:
    '''Formats a few validation issues as short example lines that show the category of the issue, the input value, the actual output, and the expected behavior. This is used to provide concrete examples of validation failures in a compact format.'''
    lines: list[str] = []
    # Only include up to the requested number of issue examples
    for issue in issues[:limit]:
        line = f"- category={issue.category}"
        # Add the input value if present
        if issue.input_value is not None:
            line += f", input={issue.input_value!r}"
        # Add the actual output if present
        if issue.actual_output is not None:
            line += f", actual={issue.actual_output!r}"
        # Add the expected behavior for reference
        line += f", expected={issue.expected_behavior}"
        lines.append(line)
    # Return the formatted lines as a single text block
    return "\n".join(lines)


def validation_issue_fingerprint(issues: list[CleanerValidationIssue], limit: int = 10) -> tuple[str, ...]:
    '''Builds a stable text fingerprint for the first few validation issues, so that repeated validation failures with the same underlying issues can be recognized and grouped together. The fingerprint captures the category of each issue and the key details of the failure, while ignoring variable factors such as error messages or stack traces.'''
    # Convert each issue into a compact key so repeated failure sets can be compared
    return tuple(
        #For each issue, build one compact text key using category, input, actual output, and expected behavior
        f"{issue.category}|{issue.input_value}|{issue.actual_output}|{issue.expected_behavior}"
        for issue in issues[:limit]
    )
