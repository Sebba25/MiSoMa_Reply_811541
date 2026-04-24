from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from agents import setup_logfire
from cleaning import run_cleaner_application, run_cleaner_generation, run_cleaning, run_remediation_planning, run_verify
from cleaning.reporting import generate_narrative_report, save_narrative_report
from validation import (
    build_validation_results,
    run_completeness_analysis,
    run_dtype_inference,
    run_format_consistency_validation,
    run_schema_validation,
)

VISIBLE_STAGES = (
    "validate",
    "dtype",
    "schema",
    "completeness",
    "consistency",
    "remediate",
    "generate",
    "apply",
    "verify",
    "clean",
    "report",
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the dataset validation and cleaning pipeline.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="Data/spesa.csv",
        help="CSV file to inspect. Defaults to Data/spesa.csv.",
    )
    parser.add_argument(
        "--stage",
        default="validate",
        metavar="{" + ",".join(VISIBLE_STAGES) + "}",
        help=(
            "Top-level stage. Use 'validate' to build the validation bundle, 'remediate' "
            "to build the remediation plan, and 'clean' for the full remediation/generate/apply/verify flow."
        ),
    )
    parser.add_argument(
        "--consistency-agent",
        choices=["all", "format"],
        default="all",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reuse-schema",
        action="store_true",
        help="Load schema handoff from the dataset-adjacent .validation_cache directory instead of re-running it.",
    )
    parser.add_argument(
        "--reuse-completeness",
        action="store_true",
        help="Load completeness analysis from the dataset-adjacent .validation_cache directory instead of re-running it.",
    )
    parser.add_argument(
        "--reuse-consistency",
        action="store_true",
        help="Load consistency validation from the dataset-adjacent .validation_cache directory instead of re-running it.",
    )
    parser.add_argument(
        "--reuse-validation",
        action="store_true",
        help="Reuse the saved validation bundle from the dataset-adjacent .validation_cache directory for clean/pipeline.",
    )
    parser.add_argument(
        "--reuse-remediation",
        action="store_true",
        help="Reuse the saved remediation plan from the dataset-adjacent .validation_cache directory for remediate/clean.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream live agent text/tool events to stderr while the run is in progress.",
    )
    parser.add_argument(
        "--column",
        help="When --stage generate is used, run cleaner generation only for the specified column name.",
    )
    parser.add_argument(
        "--cleaner-attempts",
        type=int,
        default=10,
        help="Maximum generator/critic attempts per column during cleaner generation. Defaults to 10.",
    )
    parser.add_argument(
        "--concurrent-agents",
        action="store_true",
        help=(
            "Run independent per-column agent work concurrently where supported "
            "(cleaner generation and verification consistency checks). Defaults to sequential."
        ),
    )
    parser.add_argument(
        "--agent-workers",
        type=int,
        default=3,
        help="Maximum worker count when --concurrent-agents is enabled. Defaults to 3.",
    )
    return parser


def normalize_stage(stage: str) -> str:
    normalized = stage.strip().lower()
    if normalized not in VISIBLE_STAGES:
        supported = ", ".join(VISIBLE_STAGES)
        raise ValueError(f"Unknown stage {stage!r}. Supported stages: {supported}.")
    return normalized


def print_dtype_inference(dataset_path: Path) -> None:
    inference = run_dtype_inference(dataset_path)
    column_width = max(len(column.column_name) for column in inference.columns)
    dtype_width = max(len(column.pandas_dtype) for column in inference.columns)
    role_width = max(len(column.numeric_role or column.string_role or "") for column in inference.columns)
    pattern_width = max(len(column.detected_pattern or "") for column in inference.columns)
    header = (
        f"{'COLUMN':<{column_width}}  {'DTYPE':<{dtype_width}}  "
        f"{'ROLE':<{role_width}}  {'PATTERN':<{pattern_width}}  RATIONALE"
    )
    print(header)
    print("-" * len(header))
    for column in inference.columns:
        role = column.numeric_role or column.string_role or ""
        pattern = column.detected_pattern or ""
        print(
            f"{column.column_name:<{column_width}}  {column.pandas_dtype:<{dtype_width}}  "
            f"{role:<{role_width}}  {pattern:<{pattern_width}}  {column.rationale}"
        )


def run_validation_bundle(args: argparse.Namespace, dataset_path: Path):
    return build_validation_results(
        dataset_path,
        reuse_schema=args.reuse_schema,
        reuse_completeness=args.reuse_completeness,
        reuse_consistency=args.reuse_consistency,
    )


def run_narrative_report(args: argparse.Namespace, dataset_path: Path):
    from cleaning.paths import final_report_path
    from models import FinalPipelineReport

    report_path = final_report_path(dataset_path)
    if not report_path.exists():
        raise SystemExit(
            f"Final report not found at {report_path}. Run --stage clean first."
        )
    final_report = FinalPipelineReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    narrative = generate_narrative_report(final_report)
    output_path = save_narrative_report(dataset_path, narrative)
    import sys as _sys
    _sys.stdout.buffer.write(output_path.read_bytes())
    _sys.stdout.buffer.write(b"\n")
    return narrative


def run_stage(args: argparse.Namespace, dataset_path: Path):
    if args.stage == "dtype":
        print_dtype_inference(dataset_path)
        raise SystemExit(0)

    agent_workers = args.agent_workers if args.concurrent_agents else 1

    stage_handlers: dict[str, Callable[[argparse.Namespace, Path], Any]] = {
        "validate": run_validation_bundle,
        "schema": lambda parsed_args, path: run_schema_validation(path, reuse_cache=parsed_args.reuse_schema),
        "completeness": lambda parsed_args, path: run_completeness_analysis(path, reuse_cache=parsed_args.reuse_completeness),
        "consistency": lambda parsed_args, path: run_format_consistency_validation(
            path,
            reuse_cache=parsed_args.reuse_consistency,
            max_workers=agent_workers,
        ),
        "remediate": lambda parsed_args, path: run_remediation_planning(
            path,
            reuse_saved_validation=parsed_args.reuse_validation,
            reuse_saved_remediation=parsed_args.reuse_remediation,
        ),
        "clean": lambda parsed_args, path: run_cleaning(
            path,
            reuse_saved_validation=parsed_args.reuse_validation,
            reuse_saved_remediation=parsed_args.reuse_remediation,
            cleaner_attempts=parsed_args.cleaner_attempts,
            cleaner_workers=agent_workers,
            verify_workers=agent_workers,
        ),
        "generate": lambda parsed_args, path: run_cleaner_generation(
            path,
            reuse_consistency=parsed_args.reuse_consistency,
            column_name=parsed_args.column,
            max_attempts=parsed_args.cleaner_attempts,
            max_workers=agent_workers,
        ),
        "apply": lambda parsed_args, path: run_cleaner_application(path),
        "verify": lambda parsed_args, path: run_verify(path, max_workers=agent_workers),
        "report": run_narrative_report,
    }
    return stage_handlers[args.stage](args, dataset_path)


def print_result(result) -> None:
    def _strip_large_payloads(value):
        if isinstance(value, dict):
            stripped = {}
            for key, item in value.items():
                if key == "cleaned_csv_gzip_base64":
                    continue
                stripped[key] = _strip_large_payloads(item)
            return stripped
        if isinstance(value, list):
            return [_strip_large_payloads(item) for item in value]
        return value

    if isinstance(result, list):
        print(json.dumps(_strip_large_payloads([entry.model_dump() for entry in result]), ensure_ascii=False, indent=2))
        return

    dump = _strip_large_payloads(result.model_dump())
    print(json.dumps(dump, ensure_ascii=False, indent=2))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.stage = normalize_stage(args.stage)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if args.verbose:
        os.environ["AGENT_VERBOSE"] = "1"

    if args.agent_workers < 1:
        raise SystemExit("--agent-workers must be at least 1.")

    setup_logfire()

    dataset_path = Path(__file__).parent / args.dataset
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    print_result(run_stage(args, dataset_path))
