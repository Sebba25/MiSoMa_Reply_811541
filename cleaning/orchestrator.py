"""End-to-end orchestrator for the cleaning pipeline.

Ties the cleaning stages together: resolves the validation bundle, builds
per-column cleaning requests, drives the remediation planner, runs the
generator/critic loop, applies remediations + generated cleaners, verifies the
result, and produces the narrative report.

Public entry point: ``run_cleaning``. The private helpers are consumed by
``app.py`` when it wants to stitch stages together itself.
"""

from __future__ import annotations

from pathlib import Path

from cache import load_schema_handoff, load_validation_results, save_validation_results
from models import (
    CleaningPipelineResult,
    ColumnCleanerProgram,
    ColumnCleaningRequest,
    OrchestrationStepResult,
    RemediationPlan,
)
from validation import build_validation_results
from tools import build_column_format_facts, load_dataset_frame

from .application import run_cleaner_application_with_plan
from .generation import run_cleaner_generation
from .paths import cleaned_dataset_path, load_cleaner_manifest
from .remediation import run_remediation_planning
from .reporting import build_final_report, generate_narrative_report, save_final_report, save_narrative_report
from .request import build_column_cleaning_request
from .verification import run_verify


def _resolve_validation_results(
    path: Path,
    validation_results: OrchestrationStepResult | None,
    reuse_saved_validation: bool,
) -> OrchestrationStepResult:
    if validation_results is not None:
        return validation_results

    if reuse_saved_validation:
        try:
            resolved = load_validation_results(path)
            save_validation_results(path, resolved)
            return resolved
        except FileNotFoundError:
            pass

    return build_validation_results(path)


def _build_cleaning_requests(
    path: Path,
    validation_results: OrchestrationStepResult,
) -> list[ColumnCleaningRequest]:
    df = load_dataset_frame(path)
    schema_map = {}
    try:
        handoff = load_schema_handoff(path)
        schema_map = {column.name: column for column in handoff.columns}
    except FileNotFoundError:
        pass

    requests: list[ColumnCleaningRequest] = []
    for finding in validation_results.consistency_validation.format_consistency_findings:
        format_facts = build_column_format_facts(df, finding.column_name)
        requests.append(
            build_column_cleaning_request(
                path.stem,
                finding.column_name,
                finding,
                format_facts,
                schema_entry=schema_map.get(finding.column_name),
            )
        )
    return requests


def _resolve_remediation_plan(
    path: Path,
    validation_results: OrchestrationStepResult,
    remediation_plan: RemediationPlan | None,
    reuse_saved_validation: bool,
    reuse_saved_remediation: bool,
) -> RemediationPlan:
    if remediation_plan is not None:
        return remediation_plan
    return run_remediation_planning(
        path,
        validation_results=validation_results,
        reuse_saved_validation=reuse_saved_validation,
        reuse_saved_remediation=reuse_saved_remediation,
    )


def _load_generated_programs(path: Path) -> list[ColumnCleanerProgram]:
    programs: list[ColumnCleanerProgram] = []
    try:
        artifacts = load_cleaner_manifest(path)
    except FileNotFoundError:
        return programs

    for artifact in artifacts:
        artifact_path = Path(artifact.code_path)
        if not artifact_path.exists():
            continue
        programs.append(
            ColumnCleanerProgram(
                column_name=artifact.column_name,
                function_name=artifact.function_name,
                python_code=artifact_path.read_text(encoding="utf-8"),
                example_transformations=[],
                verification_summary=artifact.summary,
            )
        )

    return programs


def run_cleaning(
    path: Path,
    validation_results: OrchestrationStepResult | None = None,
    remediation_plan: RemediationPlan | None = None,
    reuse_saved_validation: bool = False,
    reuse_saved_remediation: bool = False,
    cleaner_attempts: int = 10,
) -> CleaningPipelineResult:
    if cleaner_attempts < 1:
        raise ValueError("cleaner_attempts must be at least 1.")

    validation_results = _resolve_validation_results(path, validation_results, reuse_saved_validation)
    remediation_plan = _resolve_remediation_plan(
        path,
        validation_results,
        remediation_plan,
        reuse_saved_validation,
        reuse_saved_remediation,
    )
    cleaning_requests = _build_cleaning_requests(path, validation_results)
    run_cleaner_generation(
        path,
        reuse_consistency=True,
        max_attempts=cleaner_attempts,
    )
    cleaning_report, execution_reports, remediation_plan = run_cleaner_application_with_plan(path, remediation_plan)
    verification_report = run_verify(path)
    final_report = build_final_report(
        validation_results, remediation_plan, cleaning_report, verification_report,
        dataset_path=path,
    )
    save_final_report(path, final_report)
    narrative = generate_narrative_report(final_report)
    save_narrative_report(path, narrative)

    return CleaningPipelineResult(
        dataset_name=path.stem,
        source_path=str(path),
        cleaned_path=str(cleaned_dataset_path(path)),
        validation_results=validation_results,
        remediation_plan=remediation_plan,
        cleaning_requests=cleaning_requests,
        generated_programs=_load_generated_programs(path),
        execution_reports=execution_reports,
        cleaning_report=cleaning_report,
        verification_report=verification_report,
        final_report=final_report,
    )
