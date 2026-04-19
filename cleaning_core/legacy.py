from __future__ import annotations

import sys
from pathlib import Path

from cache import load_validation_results
from models import (
    CleaningPipelineResult,
    CleaningReport,
    ColumnCleanerExecutionReport,
    ColumnCleanerProgram,
    ColumnCleaningRequest,
    GeneratedCleanerArtifact,
    OrchestrationStepResult,
)
from pipeline import build_validation_results
from tools.tools import build_column_format_facts, gzip_text_to_base64, load_dataset_frame

from .generation import run_column_cleaner_program
from .paths import cleaned_dataset_path, cleaning_cache_dir, save_generated_cleaner
from .request import build_column_cleaning_request
from .runtime import apply_cleaner_to_series


def run_cleaning(
    path: Path,
    validation_results: OrchestrationStepResult | None = None,
    reuse_saved_validation: bool = False,
    cleaner_attempts: int = 5,
) -> CleaningPipelineResult:
    if cleaner_attempts < 1:
        raise ValueError("cleaner_attempts must be at least 1.")

    if validation_results is None:
        if reuse_saved_validation:
            try:
                validation_results = load_validation_results(path)
            except FileNotFoundError:
                validation_results = build_validation_results(path)
        else:
            validation_results = build_validation_results(path)

    df = load_dataset_frame(path)
    rows_before = len(df)
    columns_before = len(df.columns)
    cleaning_requests: list[ColumnCleaningRequest] = []
    generated_programs: list[ColumnCleanerProgram] = []
    execution_reports: list[ColumnCleanerExecutionReport] = []
    generated_cleaners: list[GeneratedCleanerArtifact] = []
    unresolved_risks: list[str] = []

    for finding in validation_results.consistency_validation.format_consistency_findings:
        column_name = finding.column_name
        format_facts = build_column_format_facts(df, column_name)
        request = build_column_cleaning_request(path.stem, column_name, finding, format_facts)
        cleaning_requests.append(request)

        print(f"running cleaner generator for '{column_name}'...", file=sys.stderr)
        try:
            program = run_column_cleaner_program(path.stem, request, max_attempts=cleaner_attempts)
        except ValueError as error:
            unresolved_risks.append(f"{column_name}: {error}")
            print(f"  FAILED - {error}", file=sys.stderr)
            continue

        generated_programs.append(program)
        code_path = save_generated_cleaner(path, program)

        print(f"running local cleaner execution for '{column_name}'...", file=sys.stderr)
        cleaned_series, execution_report = apply_cleaner_to_series(df[column_name], program)
        execution_reports.append(execution_report)

        column_risks = [
            f"{column_name}: {risk}"
            for risk in [*program.residual_risks, *execution_report.unresolved_risks]
            if risk
        ]
        unresolved_risks.extend(column_risks)

        if execution_report.execution_ok and cleaned_series is not None:
            df[column_name] = cleaned_series
        else:
            unresolved_risks.append(f"{column_name}: local execution failed, cleaner was not applied.")

        generated_cleaners.append(
            GeneratedCleanerArtifact(
                column_name=column_name,
                function_name=program.function_name,
                code_path=str(code_path),
                changed_rows=execution_report.changed_rows,
                summary=execution_report.summary,
            )
        )

    output_dir = cleaning_cache_dir(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = cleaned_dataset_path(path)
    df.to_csv(cleaned_path, index=False)

    cleaning_report = CleaningReport(
        dataset_name=path.stem,
        rows_before=rows_before,
        rows_after=len(df),
        columns_before=columns_before,
        columns_after=len(df.columns),
        generated_cleaners=generated_cleaners,
        unresolved_risks=unresolved_risks,
        cleaned_csv_gzip_base64=gzip_text_to_base64(df.to_csv(index=False)),
        summary=(
            f"Generated and verified {len(generated_programs)} column cleaners from format findings. "
            f"Saved the cleaned dataset to {cleaned_path}."
        ),
    )

    return CleaningPipelineResult(
        dataset_name=path.stem,
        source_path=str(path),
        cleaned_path=str(cleaned_path),
        validation_results=validation_results,
        cleaning_requests=cleaning_requests,
        generated_programs=generated_programs,
        execution_reports=execution_reports,
        cleaning_report=cleaning_report,
    )
