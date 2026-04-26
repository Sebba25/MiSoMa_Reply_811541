"""anomaly.py (validation pipeline): numeric outlier and rare-category detection.

This module exposes one public function, run_anomaly_detection, which runs heuristic
detectors locally to find numeric outliers and rare categorical values. Columns that were
flagged as duplicate-semantic peers in the schema stage are suppressed for the non-preferred
alias to avoid reporting the same anomaly twice. The agent only writes the human-readable
summary over the already-built findings.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agents import anomaly_summary_agent
from cache import load_anomaly, load_schema_handoff, save_anomaly
from models import AnomalyDetectionReport, AnomalyFinding, SchemaColumnEntry
from tools import (
    detect_numeric_outlier_candidates,
    detect_rare_category_candidates,
    load_dataset_frame,
    normalized_schema_name,
)

from validation._summary import summarize_validation_report


def _duplicate_semantic_suppressed_columns(columns: list[SchemaColumnEntry], duplicate_groups) -> set[str]:
    """Return the set of column names that should be suppressed in anomaly reporting.

    For each group of duplicate-semantic columns, only the preferred alias is kept.
    The preferred column is the one with a valid name and the smallest normalised form,
    so anomalies are reported once rather than once per alias.
    """
    by_name = {column.name: column for column in columns}
    suppressed: set[str] = set()

    # Sort key: prefer columns with valid names, then those whose name is already normalised, then alphabetically
    def _sort_key(name: str) -> tuple[int, int, str]:
        column = by_name[name]
        return (
            0 if column.naming_valid else 1,
            0 if normalized_schema_name(name) == name else 1,
            name,
        )

    for group in duplicate_groups:
        # Only consider columns that are actually present in the loaded dataset
        present = [name for name in group.columns if name in by_name]
        if len(present) < 2:
            continue
        # The first column after sorting is the preferred one; all others are suppressed
        preferred = sorted(present, key=_sort_key)[0]
        suppressed.update(name for name in present if name != preferred)
    return suppressed


def run_anomaly_detection(path: Path, reuse_cache: bool = False) -> AnomalyDetectionReport:
    """Run heuristic anomaly detection and return a structured report with an agent-written summary.

    Findings are built entirely from local heuristics — the agent only narrates the result.
    Schema information is loaded if available to suppress duplicate-semantic column aliases.
    """
    if reuse_cache:
        return load_anomaly(path)

    df = load_dataset_frame(path)
    # Load the schema handoff to get the list of columns and duplicate groups for suppression
    handoff = load_schema_handoff(path)
    schema_columns = handoff.columns
    # Get the set of column names that should be suppressed in anomaly reporting due to being duplicate-semantic aliases of a preferred column
    suppressed_columns = _duplicate_semantic_suppressed_columns(handoff.columns, handoff.duplicate_groups)

    # Run both detectors and filter out suppressed duplicate-semantic aliases
    findings = [
        AnomalyFinding(**finding)
        for finding in (
            detect_numeric_outlier_candidates(df, schema_columns) # Detect numeric outliers based on IQR heuristics
            + detect_rare_category_candidates(df, schema_columns) # Detect rare categorical values based on low-cardinality heuristics
        )
        if finding["column_name"] not in suppressed_columns
    ]
    # Sort by volume descending so the most impactful findings appear first
    findings.sort(key=lambda finding: (-finding.affected_rows, finding.column_name, finding.anomaly_type))
    fallback_summary = (
        f"Detected {len(findings)} anomaly findings across numeric outliers and rare categorical values."
        if findings
        else "No anomaly findings were detected by the current heuristic checks."
    )
    # Build the initial report with findings and a fallback summary, which will be replaced by the agent's summary after narrative generation.
    report = AnomalyDetectionReport(dataset_name=path.stem,total_rows=len(df),total_columns=len(df.columns),findings=findings,summary=fallback_summary)
    # Ask the summary agent to write a human-readable narrative over the already-built findings
    print(f"[orchestrator][anomaly][summary] dataset='{path.stem}'", file=sys.stderr, flush=True)
    report = report.model_copy(
        update={
            "summary": summarize_validation_report(
                anomaly_summary_agent,
                (
                    f"Summarize the provided anomaly-detection findings for dataset {path.stem}. "
                    "Do not infer new findings or alter the provided findings."
                ),
                report,
                fallback_summary,
            )
        }
    )
    # Save the final report
    save_anomaly(path, report)
    return report
