"""duplicates.py (validation pipeline): exact and near-duplicate record detection.

This module exposes one public function, run_duplicate_detection, which identifies exact
duplicate rows and near-duplicate row groups keyed on inferred identifier columns. All
findings are built locally from heuristics; the agent only writes the human-readable summary.
"""

from __future__ import annotations
import sys
from pathlib import Path

from core.agents import duplicate_summary_agent
from core.cache import load_duplicates, load_schema_handoff, save_duplicates
from core.models import DuplicateDetectionReport, DuplicateRecordGroup
from tools import (
    detect_exact_duplicate_groups,
    detect_near_duplicate_groups,
    infer_duplicate_key_columns,
    load_dataset_frame,
)

from validation._summary import summarize_validation_report


def run_duplicate_detection(path: Path, reuse_cache: bool = False) -> DuplicateDetectionReport:
    """Run exact and near-duplicate detection and return a report with an agent-written summary.

    Key columns for near-duplicate grouping are inferred from the schema if available;
    the pipeline falls back to empty schema context if the schema stage has not run yet.
    """
    if reuse_cache:
        return load_duplicates(path)

    df = load_dataset_frame(path)
    # Load schema to help infer which columns are likely identifier keys for near-duplicate detection
    try:
        handoff = load_schema_handoff(path)
        schema_columns = handoff.columns
    except FileNotFoundError:
        schema_columns = []

    # Infer the columns most likely to serve as a logical row key
    key_columns = infer_duplicate_key_columns(schema_columns, df)
    # Run both detectors and merge their groups into a single list
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
    # Ask the summary agent to write a human-readable narrative over the already-built findings
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
