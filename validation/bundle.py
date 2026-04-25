"""bundle.py (validation pipeline): end-to-end validation orchestrator.

This module exposes one public function, build_validation_results, which runs all six
validation stages in order — schema, completeness, consistency, anomaly, cross-column, and
duplicates — and persists the combined result as an OrchestrationStepResult. Each stage can
optionally reuse its cached artifact to avoid redundant LLM calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cache import save_validation_results
from models import OrchestrationStepResult

from validation.anomaly import run_anomaly_detection
from validation.completeness import run_completeness_analysis
from validation.consistency import run_format_consistency_validation
from validation.cross_column import run_cross_column_validation
from validation.duplicates import run_duplicate_detection
from validation.schema import run_schema_validation


def build_validation_results(
    path: Path,
    reuse_schema: bool = False,
    reuse_completeness: bool = False,
    reuse_consistency: bool = False,
) -> OrchestrationStepResult:
    """Run all validation stages in order and return the combined result.

    Schema, completeness, and consistency support cache reuse via their respective flags.
    Anomaly, cross-column, and duplicate detection always run fresh as they are fast and
    depend on the schema cache produced earlier in the same run.
    """
    # Schema must run first — its cache is consumed by anomaly, cross-column, and duplicate stages
    schema_validation = run_schema_validation(path, reuse_cache=reuse_schema)
    completeness_analysis = run_completeness_analysis(path, reuse_cache=reuse_completeness)
    consistency_validation = run_format_consistency_validation(path, reuse_cache=reuse_consistency)
    # Anomaly, cross-column, and duplicate stages always run fresh
    print(f"[orchestrator][anomaly] dataset='{path.stem}'", file=sys.stderr, flush=True)
    anomaly_detection = run_anomaly_detection(path)
    print(f"[orchestrator][cross-column] dataset='{path.stem}'", file=sys.stderr, flush=True)
    cross_column_validation = run_cross_column_validation(path)
    print(f"[orchestrator][duplicates] dataset='{path.stem}'", file=sys.stderr, flush=True)
    duplicate_detection = run_duplicate_detection(path)
    # Collect all stage results into a single object and persist it
    validation_results = OrchestrationStepResult(
        schema_validation=schema_validation,
        completeness_analysis=completeness_analysis,
        consistency_validation=consistency_validation,
        anomaly_detection=anomaly_detection,
        cross_column_validation=cross_column_validation,
        duplicate_detection=duplicate_detection,
    )
    save_validation_results(path, validation_results)
    return validation_results
