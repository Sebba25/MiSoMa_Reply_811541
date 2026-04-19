from __future__ import annotations

from pathlib import Path

from models import (
    CompletenessAnalysisReport,
    ConsistencyValidationReport,
    OrchestrationStepResult,
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
        "bundle": cache_dir / f"{stem}.validation_bundle.json",
    }


def save_schema_handoff(path: Path, handoff: SchemaHandoff) -> None:
    cache_dir = validation_cache_dir(path)
    cache_dir.mkdir(exist_ok=True)
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
    cache_dir.mkdir(exist_ok=True)
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
    cache_dir.mkdir(exist_ok=True)
    validation_cache_paths(path)["consistency"].write_text(
        result.model_dump_json(indent=2), encoding="utf-8",
    )


def load_consistency(path: Path) -> ConsistencyValidationReport:
    cache_path = validation_cache_paths(path)["consistency"]
    if not cache_path.exists():
        raise FileNotFoundError(f"Consistency cache not found: {cache_path}")
    return ConsistencyValidationReport.model_validate_json(cache_path.read_text(encoding="utf-8"))


def save_validation_results(path: Path, validation_results: OrchestrationStepResult) -> None:
    cache_dir = validation_cache_dir(path)
    cache_dir.mkdir(exist_ok=True)
    cache_paths = validation_cache_paths(path)
    cache_paths["completeness"].write_text(
        validation_results.completeness_analysis.model_dump_json(indent=2),
        encoding="utf-8",
    )
    cache_paths["consistency"].write_text(
        validation_results.consistency_validation.model_dump_json(indent=2),
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
