"""Runtime for generated cleaner programs.

``load_cleaner_callable`` ``exec``s the generated Python code in a locked-down
namespace (``re``, ``datetime``, optional ``dateutil`` / ``dateparser``) and
returns the callable. ``apply_cleaner_to_series`` iterates a column, collects
sample updates, and returns a per-column execution report.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from models import CellUpdate, ColumnCleanerExecutionReport, ColumnCleanerProgram


def normalize_scalar(value: Any) -> str | None:
    '''Converts one scalar value into a normalized string form or None for missing values.'''
    # Treat pandas missing values as None so comparisons are consistent
    if pd.isna(value):
        return None
    # Convert every non-missing scalar to string for uniform comparison and storage
    return str(value)
#It gives a consistent way to compare original and cleaned values, especially when missing values are involved

def load_cleaner_callable(program: ColumnCleanerProgram):
    '''Executes generated cleaner code in a restricted namespace and returns the named cleaner function.'''
    # Import only the modules that generated cleaners are allowed to use
    import datetime as _datetime
    import re as _re

    #Start a block that tries to import an optional dependency
    try:
        # dateutil is optional; if unavailable, expose None instead of failing here
        from dateutil.parser import parse as _dateutil_parse
    except ImportError:
        _dateutil_parse = None
    try:
        # dateparser is also optional for generated date-cleaning functions.
        import dateparser as _dateparser
    except ImportError:
        _dateparser = None

    # Build the execution namespace with only the approved helper modules and names
    #This starts building the namespace dictionary that will be used when executing the generated code
    namespace: dict[str, Any] = {
        "re": _re,
        "datetime": _datetime,
        "parse": _dateutil_parse,
        "dateutil": __import__("dateutil") if _dateutil_parse else None,
        "dateparser": _dateparser,
    }
    # Execute the generated Python code so it defines its cleaning function inside this namespace
    #program.python_code contains the text of the generated cleaner code
    #after exec, the function defined in that code should exist inside namespace
    exec(program.python_code, namespace)
    # Look up the function by the expected name recorded in the program metadata
    #The expected function name is stored in program.function_name
    cleaner = namespace.get(program.function_name)
    #This checks whether the loaded object is actually a callable function
    if not callable(cleaner):
        # Fail clearly if the generated code did not actually define the expected callable
        raise ValueError(
            f"Generated cleaner for '{program.column_name}' did not define callable '{program.function_name}'."
        )
    #If everything worked, return the loaded cleaner function
    return cleaner


def apply_cleaner_to_series(
    series: pd.Series, #This is the dataframe column to clean
    program: ColumnCleanerProgram, #This is the generated cleaner program containing function name and Python code
) -> tuple[pd.Series | None, ColumnCleanerExecutionReport]:
    '''Applies one generated cleaner to a whole dataframe column and returns the cleaned series plus an execution report.'''
    #teh function returns the cleaned series, or None if execution failed and an execution report describing what happened 
    try:
        # First load the generated cleaner function from its saved Python code
        cleaner = load_cleaner_callable(program)
    except Exception as error:
        # If the cleaner cannot even be loaded, return a failed execution report immediately
        return None, ColumnCleanerExecutionReport(
            column_name=program.column_name,
            function_name=program.function_name,
            execution_ok=False,
            changed_rows=0, #No rows were changed because execution never started
            unresolved_risks=[f"Failed to load cleaner code: {error}"], #Store the failure reason
            summary="Cleaner code failed to load.", #Store a short summary
        )

    # Work on an object-typed copy so mixed values can be replaced safely
    #This makes it easier to store mixed values and replacements safely
    cleaned = series.astype("object").copy()
    # Keep a small sample of changed cells for reporting
    sample_updates: list[CellUpdate] = []
    #Start a counter for how many rows changed
    changed_rows = 0

    #Now start the main per-row execution block
    try:
        # Run the cleaner on every value in the series
        for row_index, original_value in series.items():
            #Call the generated cleaner function on the original value
            cleaned_value = cleaner(original_value)
            # Normalize the original value into either None or a string
            norm_original = normalize_scalar(original_value)
            norm_cleaned = normalize_scalar(cleaned_value)
            if norm_original != norm_cleaned:
                # Count every row whose normalized value changed
                changed_rows += 1
                if len(sample_updates) < 10:
                    # Save up to 10 example cell updates for the execution report
                    sample_updates.append(
                        CellUpdate(
                            row_index=int(row_index),
                            old_value=norm_original,
                            new_value=norm_cleaned,
                        )
                    )
            # Write the cleaned value back, using pandas NA for missing outputs
            #If the cleaned value is None, store pd.NA. Otherwise store the cleaned normalized string
            cleaned.at[row_index] = pd.NA if norm_cleaned is None else norm_cleaned
    except Exception as error:
        # If execution fails partway through the column, return a failed report with partial progress info
        return None, ColumnCleanerExecutionReport(
            column_name=program.column_name,
            function_name=program.function_name,
            execution_ok=False,
            changed_rows=changed_rows, #Store how many rows had already changed before the failure happened
            sample_updates=sample_updates, #Store the sample updates collected before the failure
            unresolved_risks=[f"Cleaner failed during execution: {error}"],
            summary="Cleaner failed during full column execution.",
        )

    # If the full loop succeeds, return the cleaned series and a successful execution report
    return cleaned, ColumnCleanerExecutionReport(
        column_name=program.column_name,
        function_name=program.function_name,
        execution_ok=True,
        changed_rows=changed_rows,
        sample_updates=sample_updates,
        unresolved_risks=[],
        summary=(
            f"Applied successfully: {changed_rows} rows changed."
            if changed_rows > 0
            else "Applied successfully: no rows changed."
        ),
    )
#It is the real runtime executor that applies generated cleaning logic to a whole dataframe column and records what happened