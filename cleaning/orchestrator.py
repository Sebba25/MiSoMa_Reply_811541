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
from .reporting import build_final_report, _generate_narrative_report_chunked, save_final_report, save_narrative_report
from .request import build_column_cleaning_request
from .verification import run_verify


def _resolve_validation_results(
    path: Path,
    validation_results: OrchestrationStepResult | None,
    reuse_saved_validation: bool, #tells the function whether it should try reusing cached validation results
) -> OrchestrationStepResult:
    '''Returns validation results from the provided object, cache, or a fresh validation run. It decides where the validation results should come from'''
    # If the caller already provided validation results, use them directly
    if validation_results is not None:
        return validation_results

    # If cache reuse is allowed, try loading the saved validation bundle first.
    if reuse_saved_validation:
        try:
            resolved = load_validation_results(path)
            # Save the loaded result again so downstream stages have a normalized cached copy
            save_validation_results(path, resolved)
            return resolved
        except FileNotFoundError:
            # If no cached validation exists, fall back to rebuilding it
            pass

    # Run validation from scratch when no direct or cached result is available
    return build_validation_results(path)


def _build_cleaning_requests(
    path: Path,
    validation_results: OrchestrationStepResult, #This is the full validation bundle
) -> list[ColumnCleaningRequest]: # returns a list of ColumnCleaningRequest objects
    '''Builds one ColumnCleaningRequest for each format-consistency finding in the validation bundle.'''
    # Load the raw dataset so per-column format facts can be computed
    df = load_dataset_frame(path)
    schema_map = {}
    try:
        # Load schema metadata so dtype and role hints can be attached to each request
        handoff = load_schema_handoff(path)
        #Convert the schema columns into a dictionary
        schema_map = {column.name: column for column in handoff.columns}
    except FileNotFoundError:
        # If schema hints are unavailable, requests will still be built with format facts alone
        pass

    # Build one cleaning request per format-consistency finding
    #Start an empty list of cleaning requests
    requests: list[ColumnCleaningRequest] = []
    #Loop through each format-consistency finding found during validation
    for finding in validation_results.consistency_validation.format_consistency_findings:
        # Compute observed format facts for the current column from the raw dataframe
        format_facts = build_column_format_facts(df, finding.column_name)
        requests.append(
            #Call the helper that assembles the request
            build_column_cleaning_request(
                path.stem,
                finding.column_name,
                finding,
                format_facts,
                #If schema info exists for this column, include it.
                schema_entry=schema_map.get(finding.column_name),
            )
        )
    #Return the full list of per-column cleaning requests
    return requests


def _resolve_remediation_plan(
    path: Path,
    validation_results: OrchestrationStepResult, #The validation bundle that remediation planning may use
    remediation_plan: RemediationPlan | None, #An optional remediation plan already provided by the caller
    reuse_saved_validation: bool, #Whether remediation planning may reuse cached validation
    reuse_saved_remediation: bool, #Whether remediation planning may reuse a cached remediation plan
) -> RemediationPlan:
    '''Returns the provided remediation plan or builds/loads one from the remediation stage.'''
    # If the caller already passed a remediation plan, keep using it
    if remediation_plan is not None:
        return remediation_plan
    # Otherwise run remediation planning, optionally reusing cached validation or remediation results
    return run_remediation_planning(
        path,
        validation_results=validation_results,
        reuse_saved_validation=reuse_saved_validation,
        reuse_saved_remediation=reuse_saved_remediation,
    )
#It centralizes the logic for choosing between a provided plan and a generated/cached one

def _load_generated_programs(path: Path) -> list[ColumnCleanerProgram]:
    '''Loads generated cleaner programs from the saved cleaner manifest and code files.'''
    programs: list[ColumnCleanerProgram] = [] #Start an empty list of generated programs
    try:
        # Read the cleaner manifest, which lists the generated code files for each column
        artifacts = load_cleaner_manifest(path)
    except FileNotFoundError:
        # If no cleaner manifest exists, return an empty list of generated programs
        return programs

    # Rebuild each generated cleaner program from its saved Python file and artifact metadata
    for artifact in artifacts:
        #Convert the artifact’s code path into a Path object
        artifact_path = Path(artifact.code_path)
        if not artifact_path.exists():
            # Skip missing cleaner files instead of failing the entire orchestration result
            continue
        programs.append(
            ColumnCleanerProgram(
                column_name=artifact.column_name,
                function_name=artifact.function_name,
                python_code=artifact_path.read_text(encoding="utf-8"),
                example_transformations=[],
                verification_summary=artifact.summary, #Reuse the artifact summary as the verification summary
            )
        )

    return programs


def run_cleaning(
    path: Path,
    validation_results: OrchestrationStepResult | None = None,
    remediation_plan: RemediationPlan | None = None,
    reuse_saved_validation: bool = False,
    reuse_saved_remediation: bool = False,
    cleaner_attempts: int = 10, #Maximum retry attempts for each cleaner-generation loop
    cleaner_workers: int = 1, #Number of workers for cleaner generation
    verify_workers: int = 1, #Number of workers for verification
) -> CleaningPipelineResult:
    '''Runs the full cleaning pipeline from validation resolution to final report generation.'''
    # Validate basic user-controlled settings before running any stage
    if cleaner_attempts < 1:
        raise ValueError("cleaner_attempts must be at least 1.")
    if cleaner_workers < 1:
        raise ValueError("cleaner_workers must be at least 1.")
    if verify_workers < 1:
        raise ValueError("verify_workers must be at least 1.")

    # Resolve or build the validation bundle used by downstream stages
    validation_results = _resolve_validation_results(path, validation_results, reuse_saved_validation)
    # Resolve or build the remediation plan that decides which actions should be applied
    #use teh provided one or build or load it
    remediation_plan = _resolve_remediation_plan(
        path,
        validation_results,
        remediation_plan,
        reuse_saved_validation,
        reuse_saved_remediation,
    )
    # Build the per-column request objects that describe the cleaning work for the generator stage
    #Build one per-column cleaning request for each consistency finding
    cleaning_requests = _build_cleaning_requests(path, validation_results)
    # Run the cleaner-generation stage to create per-column cleaning programs
    #This creates the per-column cleaning programs and saves them through the generation subsystem
    run_cleaner_generation(
        path,
        reuse_consistency=True, #means consistency findings can be reused
        max_attempts=cleaner_attempts,
        max_workers=cleaner_workers,
    )
    # Apply generated cleaners and remediation actions to produce the cleaned dataset
    cleaning_report, execution_reports, remediation_plan = run_cleaner_application_with_plan(path, remediation_plan)
    # Verify the cleaned dataset by re-running consistency checks and comparing before vs after
    verification_report = run_verify(path, max_workers=verify_workers)
    # Merge all outputs into one final structured report object
    final_report = build_final_report(
        validation_results, remediation_plan, cleaning_report, verification_report,
        dataset_path=path,
    )
    # Save the final JSON report and the human-readable narrative report
    save_final_report(path, final_report)
    narrative = _generate_narrative_report_chunked(final_report)
    save_narrative_report(path, narrative)

    # Return a single result object containing the outputs of every major pipeline stage
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
