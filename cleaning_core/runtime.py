from __future__ import annotations

from typing import Any

import pandas as pd

from models import CellUpdate, ColumnCleanerExecutionReport, ColumnCleanerProgram


def normalize_scalar(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def load_cleaner_callable(program: ColumnCleanerProgram):
    import datetime as _datetime
    import re as _re

    try:
        from dateutil.parser import parse as _dateutil_parse
    except ImportError:
        _dateutil_parse = None
    try:
        import dateparser as _dateparser
    except ImportError:
        _dateparser = None

    namespace: dict[str, Any] = {
        "re": _re,
        "datetime": _datetime,
        "parse": _dateutil_parse,
        "dateutil": __import__("dateutil") if _dateutil_parse else None,
        "dateparser": _dateparser,
    }
    exec(program.python_code, namespace)
    cleaner = namespace.get(program.function_name)
    if not callable(cleaner):
        raise ValueError(
            f"Generated cleaner for '{program.column_name}' did not define callable '{program.function_name}'."
        )
    return cleaner


def apply_cleaner_to_series(
    series: pd.Series,
    program: ColumnCleanerProgram,
) -> tuple[pd.Series | None, ColumnCleanerExecutionReport]:
    try:
        cleaner = load_cleaner_callable(program)
    except Exception as error:
        return None, ColumnCleanerExecutionReport(
            column_name=program.column_name,
            function_name=program.function_name,
            execution_ok=False,
            changed_rows=0,
            unresolved_risks=[f"Failed to load cleaner code: {error}"],
            summary="Cleaner code failed to load.",
        )

    cleaned = series.astype("object").copy()
    sample_updates: list[CellUpdate] = []
    changed_rows = 0

    try:
        for row_index, original_value in series.items():
            cleaned_value = cleaner(original_value)
            norm_original = normalize_scalar(original_value)
            norm_cleaned = normalize_scalar(cleaned_value)
            if norm_original != norm_cleaned:
                changed_rows += 1
                if len(sample_updates) < 10:
                    sample_updates.append(
                        CellUpdate(
                            row_index=int(row_index),
                            old_value=norm_original,
                            new_value=norm_cleaned,
                        )
                    )
            cleaned.at[row_index] = pd.NA if norm_cleaned is None else norm_cleaned
    except Exception as error:
        return None, ColumnCleanerExecutionReport(
            column_name=program.column_name,
            function_name=program.function_name,
            execution_ok=False,
            changed_rows=changed_rows,
            sample_updates=sample_updates,
            unresolved_risks=[f"Cleaner failed during execution: {error}"],
            summary="Cleaner failed during full column execution.",
        )

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

