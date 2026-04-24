"""cross_column.py (validation pipeline): cross-column consistency checks.

This module exposes one public function, run_cross_column_validation, which runs four
heuristic detectors to find duplicate-like columns, duplicate-semantic conflicts,
year/month/period mismatches, and date-order violations. All findings are built locally
without LLM involvement; the agent only writes the human-readable summary at the end.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agents import cross_column_summary_agent
from cache import load_cross_column, load_schema_handoff, save_cross_column
from models import CrossColumnFinding, CrossColumnValidationReport
from tools import (
    detect_date_order_violations,
    detect_duplicate_like_columns,
    detect_duplicate_semantic_conflicts,
    detect_year_month_period_mismatches,
    load_dataset_frame,
)

from validation._summary import summarize_validation_report


def run_cross_column_validation(path: Path, reuse_cache: bool = False) -> CrossColumnValidationReport:
    """Run all cross-column heuristic checks and return a report with an agent-written summary.

    Schema information is loaded if available to guide duplicate-semantic conflict detection;
    the pipeline continues with empty schema context if the schema stage has not run yet.
    """
    if reuse_cache:
        return load_cross_column(path)

    df = load_dataset_frame(path)
    # Load schema to provide dtype and duplicate-group context to the detectors
    try:
        handoff = load_schema_handoff(path)
        schema_columns = handoff.columns
        duplicate_groups = handoff.duplicate_groups
    except FileNotFoundError:
        schema_columns = []
        duplicate_groups = []

    # Run all four detectors and merge their findings into a single list
    findings = [
        CrossColumnFinding(**finding)
        for finding in (
            detect_duplicate_like_columns(df, schema_columns)
            + detect_duplicate_semantic_conflicts(df, duplicate_groups)
            + detect_year_month_period_mismatches(df, schema_columns)
            + detect_date_order_violations(df, schema_columns)
        )
    ]
    # Sort by volume descending so the most impactful findings appear first
    findings.sort(key=lambda finding: (-finding.affected_rows, ",".join(finding.columns), finding.check_type))
    fallback_summary = (
        f"Detected {len(findings)} cross-column consistency findings."
        if findings
        else "No cross-column consistency findings were detected by the current rule set."
    )
    report = CrossColumnValidationReport(
        dataset_name=path.stem,
        total_rows=len(df),
        findings=findings,
        summary=fallback_summary,
    )
    # Ask the summary agent to write a human-readable narrative over the already-built findings
    print(f"[orchestrator][cross-column][summary] dataset='{path.stem}'", file=sys.stderr, flush=True)
    report = report.model_copy(
        update={
            "summary": summarize_validation_report(
                cross_column_summary_agent,
                (
                    f"Summarize the provided cross-column validation findings for dataset {path.stem}. "
                    "Do not infer new findings or alter the provided findings."
                ),
                report,
                fallback_summary,
            )
        }
    )
    save_cross_column(path, report)
    return report
