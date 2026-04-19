from __future__ import annotations

from cleaning_core.application import (
    _apply_column_renames,
    _apply_dtype_casts,
    _apply_placeholder_nulls,
    run_cleaner_application,
)
from cleaning_core.generation import (
    _build_cleaner_generation_prompt,
    _build_validation_issue,
    _detect_shadowed_delimiter_branches,
    _dominant_output_shape,
    _format_validation_examples,
    _format_validation_issue,
    _is_parseable_output,
    _print_example_transformations,
    _rebuild_verified_program,
    _requires_fixed_output_shape,
    _validate_generated_cleaner_program,
    _validation_issue_fingerprint,
    run_cleaner_generation,
    run_cleaner_repair_critic,
    run_column_cleaner_program,
)
from cleaning_core.legacy import run_cleaning
from cleaning_core.paths import (
    cleaned_dataset_path,
    cleaner_manifest_path,
    cleaning_cache_dir,
    final_report_path,
    generated_cleaner_path,
    load_cleaner_manifest,
    save_cleaner_manifest,
    save_generated_cleaner,
)
from cleaning_core.remediation import run_remediation_planning
from cleaning_core.reporting import build_final_report, generate_narrative_report, save_final_report, save_narrative_report
from cleaning_core.request import build_column_cleaning_request
from cleaning_core.runtime import (
    apply_cleaner_to_series as _apply_cleaner_to_series,
    load_cleaner_callable,
    normalize_scalar as _normalize_scalar,
)
from cleaning_core.verification import run_verify


__all__ = [
    "build_column_cleaning_request",
    "cleaned_dataset_path",
    "cleaner_manifest_path",
    "cleaning_cache_dir",
    "final_report_path",
    "generated_cleaner_path",
    "load_cleaner_callable",
    "load_cleaner_manifest",
    "build_final_report",
    "run_cleaner_application",
    "run_cleaner_generation",
    "run_cleaner_repair_critic",
    "run_cleaning",
    "run_remediation_planning",
    "run_column_cleaner_program",
    "run_verify",
    "generate_narrative_report",
    "save_final_report",
    "save_narrative_report",
    "save_cleaner_manifest",
    "save_generated_cleaner",
]
