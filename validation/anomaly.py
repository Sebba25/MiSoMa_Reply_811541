"""Anomaly detection stage.

Heuristic detectors (numeric outliers + rare categoricals) build the
findings locally; ``anomaly_summary_agent`` only writes the human-readable
summary over the already-built report. Columns flagged as duplicate-semantic
peers in the schema stage are suppressed for the non-preferred alias to
avoid reporting the same anomaly twice.
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
    by_name = {column.name: column for column in columns}
    suppressed: set[str] = set()
    for group in duplicate_groups:
        present = [name for name in group.columns if name in by_name]
        if len(present) < 2:
            continue

        def _sort_key(name: str) -> tuple[int, int, str]:
            column = by_name[name]
            return (
                0 if column.naming_valid else 1,
                0 if normalized_schema_name(name) == name else 1,
                name,
            )

        preferred = sorted(present, key=_sort_key)[0]
        suppressed.update(name for name in present if name != preferred)
    return suppressed


def run_anomaly_detection(path: Path, reuse_cache: bool = False) -> AnomalyDetectionReport:
    if reuse_cache:
        return load_anomaly(path)

    df = load_dataset_frame(path)
    try:
        handoff = load_schema_handoff(path)
        schema_columns = handoff.columns
        suppressed_columns = _duplicate_semantic_suppressed_columns(handoff.columns, handoff.duplicate_groups)
    except FileNotFoundError:
        schema_columns = []
        suppressed_columns = set()

    findings = [
        AnomalyFinding(**finding)
        for finding in (
            detect_numeric_outlier_candidates(df, schema_columns)
            + detect_rare_category_candidates(df, schema_columns)
        )
        if finding["column_name"] not in suppressed_columns
    ]
    findings.sort(key=lambda finding: (-finding.affected_rows, finding.column_name, finding.anomaly_type))
    fallback_summary = (
        f"Detected {len(findings)} anomaly findings across numeric outliers and rare categorical values."
        if findings
        else "No anomaly findings were detected by the current heuristic checks."
    )
    report = AnomalyDetectionReport(
        dataset_name=path.stem,
        total_rows=len(df),
        total_columns=len(df.columns),
        findings=findings,
        summary=fallback_summary,
    )
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
    save_anomaly(path, report)
    return report
