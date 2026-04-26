"""Validation-stage orchestration.

Drives the validation half of the pipeline: dtype inference, schema,
completeness, per-column format consistency, anomaly detection,
cross-column checks, and duplicate detection. Each stage function caches
its artifact via ``cache.py``.

Public entry point is ``build_validation_results`` (runs every stage);
individual ``run_<stage>`` helpers are also importable for the CLI and
the Streamlit UI.
"""

from __future__ import annotations

from validation.anomaly import run_anomaly_detection
from validation.bundle import build_validation_results
from validation.completeness import run_completeness_analysis
from validation.consistency import run_column_format_check, run_format_consistency_validation
from validation.cross_column import run_cross_column_validation
from validation.duplicates import run_duplicate_detection
from validation.schema import build_schema_issues, run_dtype_inference, run_schema_validation

__all__ = [
    "build_schema_issues",
    "build_validation_results",
    "run_anomaly_detection",
    "run_column_format_check",
    "run_completeness_analysis",
    "run_cross_column_validation",
    "run_dtype_inference",
    "run_duplicate_detection",
    "run_format_consistency_validation",
    "run_schema_validation",
]
