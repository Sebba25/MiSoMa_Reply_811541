from __future__ import annotations

from pathlib import Path

from models import (
    AnomalyDetectionReport,
    CompletenessAnalysisReport,
    ConsistencyValidationReport,
    CrossColumnValidationReport,
    DuplicateDetectionReport,
    OrchestrationStepResult,
    RemediationPlan,
    SchemaHandoff,
)


def validation_cache_dir(path: Path) -> Path:
    return path.parent / ".validation_cache"


def validation_cache_paths(path: Path) -> dict[str, Path]:
    cache_dir = validation_cache_dir(path)
    stem = path.stem
    return {
        "schema_handoff": cache_dir / f"{stem}.schema_handoff.json",
        "completeness": cache_dir / f"{stem}.completeness.json",
        "consistency": cache_dir / f"{stem}.consistency.json",
        "anomaly": cache_dir / f"{stem}.anomaly.json",
        "cross_column": cache_dir / f"{stem}.cross_column.json",
        "duplicates": cache_dir / f"{stem}.duplicates.json",
        "remediation_plan": cache_dir / f"{stem}.remediation_plan.json",
        "bundle": cache_dir / f"{stem}.validation_bundle.json",
    }


def save_schema_handoff(path: Path, handoff: SchemaHandoff) -> None:
    cache_dir = validation_cache_dir(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    validation_cache_paths(path)["schema_handoff"].write_text(
        handoff.model_dump_json(indent=2), encoding="utf-8",
    )


def load_schema_handoff(path: Path) -> SchemaHandoff:
    cache_path = validation_cache_paths(path)["schema_handoff"]
    if not cache_path.exists():
        raise FileNotFoundError(f"Schema handoff cache not found: {cache_path}")
    return SchemaHandoff.model_validate_json(cache_path.read_text(encoding="utf-8"))


def save_completeness(path: Path, result: CompletenessAnalysisReport) -> None:
    cache_dir = validation_cache_dir(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    validation_cache_paths(path)["completeness"].write_text(
        result.model_dump_json(indent=2), encoding="utf-8",
    )


def load_completeness(path: Path) -> CompletenessAnalysisReport:
    cache_path = validation_cache_paths(path)["completeness"]
    if not cache_path.exists():
        raise FileNotFoundError(f"Completeness cache not found: {cache_path}")
    return CompletenessAnalysisReport.model_validate_json(cache_path.read_text(encoding="utf-8"))


def save_consistency(path: Path, result: ConsistencyValidationReport) -> None:
    cache_dir = validation_cache_dir(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    validation_cache_paths(path)["consistency"].write_text(
        result.model_dump_json(indent=2), encoding="utf-8",
    )


def load_consistency(path: Path) -> ConsistencyValidationReport:
    cache_path = validation_cache_paths(path)["consistency"]
    if not cache_path.exists():
        raise FileNotFoundError(f"Consistency cache not found: {cache_path}")
    return ConsistencyValidationReport.model_validate_json(cache_path.read_text(encoding="utf-8"))


def save_anomaly(path: Path, result: AnomalyDetectionReport) -> None:
    cache_dir = validation_cache_dir(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    validation_cache_paths(path)["anomaly"].write_text(
        result.model_dump_json(indent=2), encoding="utf-8",
    )


def load_anomaly(path: Path) -> AnomalyDetectionReport:
    cache_path = validation_cache_paths(path)["anomaly"]
    if not cache_path.exists():
        raise FileNotFoundError(f"Anomaly cache not found: {cache_path}")
    return AnomalyDetectionReport.model_validate_json(cache_path.read_text(encoding="utf-8"))


def save_cross_column(path: Path, result: CrossColumnValidationReport) -> None:
    cache_dir = validation_cache_dir(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    validation_cache_paths(path)["cross_column"].write_text(
        result.model_dump_json(indent=2), encoding="utf-8",
    )


def load_cross_column(path: Path) -> CrossColumnValidationReport:
    cache_path = validation_cache_paths(path)["cross_column"]
    if not cache_path.exists():
        raise FileNotFoundError(f"Cross-column cache not found: {cache_path}")
    return CrossColumnValidationReport.model_validate_json(cache_path.read_text(encoding="utf-8"))


def save_duplicates(path: Path, result: DuplicateDetectionReport) -> None:
    cache_dir = validation_cache_dir(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    validation_cache_paths(path)["duplicates"].write_text(
        result.model_dump_json(indent=2), encoding="utf-8",
    )


def load_duplicates(path: Path) -> DuplicateDetectionReport:
    cache_path = validation_cache_paths(path)["duplicates"]
    if not cache_path.exists():
        raise FileNotFoundError(f"Duplicate-detection cache not found: {cache_path}")
    return DuplicateDetectionReport.model_validate_json(cache_path.read_text(encoding="utf-8"))


def save_remediation_plan(path: Path, result: RemediationPlan) -> None:
    cache_dir = validation_cache_dir(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    validation_cache_paths(path)["remediation_plan"].write_text(
        result.model_dump_json(indent=2), encoding="utf-8",
    )


def load_remediation_plan(path: Path) -> RemediationPlan:
    cache_path = validation_cache_paths(path)["remediation_plan"]
    if not cache_path.exists():
        raise FileNotFoundError(f"Remediation-plan cache not found: {cache_path}")
    return RemediationPlan.model_validate_json(cache_path.read_text(encoding="utf-8"))


def save_validation_results(path: Path, validation_results: OrchestrationStepResult) -> None:
    cache_dir = validation_cache_dir(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = validation_cache_paths(path)
    cache_paths["schema_handoff"].write_text(
        validation_results.schema_validation.model_dump_json(indent=2),
        encoding="utf-8",
    )
    cache_paths["completeness"].write_text(
        validation_results.completeness_analysis.model_dump_json(indent=2),
        encoding="utf-8",
    )
    cache_paths["consistency"].write_text(
        validation_results.consistency_validation.model_dump_json(indent=2),
        encoding="utf-8",
    )
    if validation_results.anomaly_detection is not None:
        cache_paths["anomaly"].write_text(
            validation_results.anomaly_detection.model_dump_json(indent=2),
            encoding="utf-8",
        )
    if validation_results.cross_column_validation is not None:
        cache_paths["cross_column"].write_text(
            validation_results.cross_column_validation.model_dump_json(indent=2),
            encoding="utf-8",
        )
    if validation_results.duplicate_detection is not None:
        cache_paths["duplicates"].write_text(
            validation_results.duplicate_detection.model_dump_json(indent=2),
            encoding="utf-8",
        )
    cache_paths["bundle"].write_text(
        validation_results.model_dump_json(indent=2),
        encoding="utf-8",
    )


def load_validation_results(path: Path) -> OrchestrationStepResult:
    bundle_path = validation_cache_paths(path)["bundle"]
    if not bundle_path.exists():
        raise FileNotFoundError(f"Validation cache not found: {bundle_path}")
    return OrchestrationStepResult.model_validate_json(bundle_path.read_text(encoding="utf-8"))
