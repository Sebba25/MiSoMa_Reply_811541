"""Duplicate-record detection stage.

Heuristic detectors find exact-duplicate row groups and near-duplicate row
groups (keyed on inferred identifier columns). The ``duplicate_summary_agent``
only writes the summary over the already-built report.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agents import duplicate_summary_agent
from cache import load_duplicates, load_schema_handoff, save_duplicates
from models import DuplicateDetectionReport, DuplicateRecordGroup
from tools import (
    detect_exact_duplicate_groups,
    detect_near_duplicate_groups,
    infer_duplicate_key_columns,
    load_dataset_frame,
)

from validation._summary import summarize_validation_report


def run_duplicate_detection(path: Path, reuse_cache: bool = False) -> DuplicateDetectionReport:
    if reuse_cache:
        return load_duplicates(path)

    df = load_dataset_frame(path)
    try:
        handoff = load_schema_handoff(path)
        schema_columns = handoff.columns
    except FileNotFoundError:
        schema_columns = []

    key_columns = infer_duplicate_key_columns(schema_columns, df)
    groups = [
        DuplicateRecordGroup(**group)
        for group in (
            detect_exact_duplicate_groups(df)
            + detect_near_duplicate_groups(df, key_columns)
        )
    ]
    fallback_summary = (
        f"Detected {len(groups)} duplicate-record groups."
        if groups
        else "No duplicate-record groups were detected by the current exact and near-duplicate checks."
    )
    report = DuplicateDetectionReport(
        dataset_name=path.stem,
        total_rows=len(df),
        groups=groups,
        summary=fallback_summary,
    )
    print(f"[orchestrator][duplicates][summary] dataset='{path.stem}'", file=sys.stderr, flush=True)
    report = report.model_copy(
        update={
            "summary": summarize_validation_report(
                duplicate_summary_agent,
                (
                    f"Summarize the provided duplicate-detection findings for dataset {path.stem}. "
                    "Do not infer new findings or alter the provided findings."
                ),
                report,
                fallback_summary,
            )
        }
    )
    save_duplicates(path, report)
    return report
