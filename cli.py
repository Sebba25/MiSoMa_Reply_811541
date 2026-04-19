from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agents import setup_logfire
from cleaning import run_cleaner_application, run_cleaner_generation, run_cleaning, run_verify
from pipeline import (
    build_validation_results,
    run_completeness_analysis,
    run_consistency_validation,
    run_dtype_inference,
    run_format_consistency_validation,
    run_schema_validation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Pydantic AI dataset validation agents.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="Data/spesa.csv",
        help="CSV file to inspect. Defaults to Data/spesa.csv.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "dtype", "schema", "completeness", "consistency", "clean", "generate", "apply", "verify", "pipeline"],
        default="all",
        help="Which top-level stage to run. Defaults to all.",
    )
    parser.add_argument(
        "--consistency-agent",
        choices=["all", "format"],
        default="all",
        help="When --stage consistency is used, choose which consistency sub-agent to run. Defaults to all.",
    )
    parser.add_argument(
        "--reuse-schema",
        action="store_true",
        help="Load schema handoff from .validation_cache instead of re-running dtype inference and schema agents.",
    )
    parser.add_argument(
        "--reuse-completeness",
        action="store_true",
        help="Load completeness analysis from .validation_cache instead of re-running.",
    )
    parser.add_argument(
        "--reuse-consistency",
        action="store_true",
        help="Load consistency validation from .validation_cache instead of re-running.",
    )
    parser.add_argument(
        "--reuse-validation",
        action="store_true",
        help="Reuse saved validation bundle from .validation_cache when running the cleaning stage.",
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
        default=5,
        help="Maximum generator/critic attempts per column during cleaner generation. Defaults to 5.",
    )
    return parser


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


def run_stage(args: argparse.Namespace, dataset_path: Path):
    if args.stage == "dtype":
        print_dtype_inference(dataset_path)
        raise SystemExit(0)
    if args.stage == "schema":
        return run_schema_validation(dataset_path, reuse_cache=args.reuse_schema)
    if args.stage == "completeness":
        return run_completeness_analysis(dataset_path, reuse_cache=args.reuse_completeness)
    if args.stage == "consistency":
        if args.consistency_agent == "format":
            return run_format_consistency_validation(dataset_path, reuse_cache=args.reuse_consistency)
        return run_consistency_validation(dataset_path, reuse_cache=args.reuse_consistency)
    if args.stage == "clean":
        return run_cleaning(
            dataset_path,
            reuse_saved_validation=args.reuse_validation,
            cleaner_attempts=args.cleaner_attempts,
        )
    if args.stage == "generate":
        return run_cleaner_generation(
            dataset_path,
            reuse_consistency=args.reuse_consistency,
            column_name=args.column,
            max_attempts=args.cleaner_attempts,
        )
    if args.stage == "apply":
        return run_cleaner_application(dataset_path)
    if args.stage == "verify":
        return run_verify(dataset_path)
    if args.stage == "pipeline":
        run_schema_validation(dataset_path)
        run_completeness_analysis(dataset_path)
        run_format_consistency_validation(dataset_path)
        run_cleaner_generation(
            dataset_path,
            reuse_consistency=True,
            max_attempts=args.cleaner_attempts,
        )
        run_cleaner_application(dataset_path)
        return run_verify(dataset_path)

    return build_validation_results(
        dataset_path,
        reuse_schema=args.reuse_schema,
        reuse_completeness=args.reuse_completeness,
        reuse_consistency=args.reuse_consistency,
    )


def print_result(result) -> None:
    if isinstance(result, list):
        print(json.dumps([entry.model_dump() for entry in result], ensure_ascii=False, indent=2))
        return

    dump = result.model_dump()
    dump.pop("cleaned_csv_gzip_base64", None)
    print(json.dumps(dump, ensure_ascii=False, indent=2))


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        os.environ["AGENT_VERBOSE"] = "1"

    setup_logfire()

    dataset_path = Path(__file__).parent / args.dataset
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    print_result(run_stage(args, dataset_path))
