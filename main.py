import argparse
import asyncio
import base64
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, CodeExecutionTool, PromptedOutput
from pydantic_ai.messages import BinaryContent

load_dotenv()

MODEL = "openai-responses:gpt-4o-mini"
PLACEHOLDER_VALUES = ["", "N/A", "NA", "-", "--", "unknown", "UNKNOWN", "NULL", "null", "n.d.", "?", "//"]


class ColumnFinding(BaseModel):
    column_name: str
    expected_type: str = Field(description="Semantic type inferred from the column values.")
    detected_storage_type: str = Field(description="Physical pandas dtype observed before cleaning.")
    completeness_pct: float = Field(ge=0, le=100)
    missing_like_count: int = Field(ge=0)
    missing_like_examples: list[str] = Field(default_factory=list)
    naming_issues: list[str] = Field(default_factory=list)
    data_type_issues: list[str] = Field(default_factory=list)
    format_issues: list[str] = Field(default_factory=list)
    sparse_candidate: bool = False
    recommended_action: str


class DuplicateFinding(BaseModel):
    duplicate_type: str = Field(description="Use exact or near.")
    candidate_key_columns: list[str] = Field(default_factory=list)
    affected_rows: int = Field(ge=0)
    description: str


class CrossColumnFinding(BaseModel):
    columns: list[str] = Field(default_factory=list)
    rule_checked: str
    affected_rows: int = Field(ge=0)
    description: str


class ValidationReport(BaseModel):
    dataset_name: str
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    overall_completeness_pct: float = Field(ge=0, le=100)
    naming_convention_findings: list[str] = Field(default_factory=list)
    column_findings: list[ColumnFinding] = Field(default_factory=list)
    duplicate_findings: list[DuplicateFinding] = Field(default_factory=list)
    cross_column_findings: list[CrossColumnFinding] = Field(default_factory=list)
    global_findings: list[str] = Field(default_factory=list)
    cleaning_priorities: list[str] = Field(default_factory=list)
    summary: str


class CleaningAction(BaseModel):
    action_type: str
    affected_columns: list[str] = Field(default_factory=list)
    affected_rows: int = Field(ge=0)
    description: str


class CleaningReport(BaseModel):
    dataset_name: str
    rows_before: int = Field(ge=0)
    rows_after: int = Field(ge=0)
    columns_before: int = Field(ge=0)
    columns_after: int = Field(ge=0)
    actions: list[CleaningAction] = Field(default_factory=list)
    unresolved_risks: list[str] = Field(default_factory=list)
    cleaned_csv_gzip_base64: str = Field(description="The cleaned CSV encoded as gzip+base64.")
    summary: str


class PipelineResult(BaseModel):
    dataset_name: str
    source_path: str
    cleaned_path: str
    report_path: str
    validation_report: ValidationReport
    cleaning_report: CleaningReport
    cleaned_file_sha256: str


@dataclass
class DatasetDeps:
    source_path: Path
    cleaned_path: Path
    report_path: Path


def gzip_text_to_base64(text: str) -> str:
    return base64.b64encode(gzip.compress(text.encode("utf-8"))).decode("ascii")


def ungzip_base64_to_text(payload: str) -> str:
    return gzip.decompress(base64.b64decode(payload.encode("ascii"))).decode("utf-8")


def dataset_attachment(path: Path) -> BinaryContent:
    return BinaryContent(data=path.read_bytes(), media_type="text/csv")


def report_to_markdown(validation: ValidationReport, cleaning: CleaningReport) -> str:
    lines = [
        f"# Data Quality Report: {validation.dataset_name}",
        "",
        "## Validation Summary",
        f"- Rows: {validation.total_rows}",
        f"- Columns: {validation.total_columns}",
        f"- Overall completeness: {validation.overall_completeness_pct:.2f}%",
        f"- Summary: {validation.summary}",
        "",
        "## Key Findings",
    ]

    if validation.global_findings:
        lines.extend(f"- {finding}" for finding in validation.global_findings)
    else:
        lines.append("- No global findings reported.")

    lines.extend(["", "## Cleaning Summary", f"- Summary: {cleaning.summary}"])

    if cleaning.actions:
        lines.extend(
            f"- {action.action_type}: {action.description} "
            f"(rows={action.affected_rows}, columns={', '.join(action.affected_columns) or 'n/a'})"
            for action in cleaning.actions
        )
    else:
        lines.append("- No cleaning actions were required.")

    lines.extend(["", "## Remaining Risks"])
    if cleaning.unresolved_risks:
        lines.extend(f"- {risk}" for risk in cleaning.unresolved_risks)
    else:
        lines.append("- No remaining risks reported.")

    return "\n".join(lines) + "\n"


validation_agent = Agent(
    MODEL,
    output_type=PromptedOutput(ValidationReport),
    builtin_tools=[CodeExecutionTool()],
    retries=4,
    instructions=(
        "You are a senior data quality analyst. Always use the code execution tool to inspect the attached CSV dataset. "
        "Write Python in the isolated execution environment to load the attached file into pandas and analyze it. "
        "Do not guess from the prompt alone. "
        "Return valid JSON only that matches the ValidationReport schema exactly. Do not use markdown, prose wrappers, or follow-up questions. "
        "Return a structured ValidationReport that covers these checks: "
        "schema validation for column names against naming standards (casing, special characters, reserved-word risk), "
        "data type validation, completeness rate per column and overall, null and placeholder detection, sparse column detection, "
        "format consistency, cross-column logical checks when meaningful, and exact plus near-duplicate detection. "
        "If a cross-column check is not applicable, say so explicitly in the findings rather than inventing one. "
        "Use the provided placeholder examples, but also detect additional null-like tokens if they appear. "
        "Be conservative and evidence-based."
    ),
)


cleaning_agent = Agent(
    MODEL,
    output_type=PromptedOutput(CleaningReport),
    builtin_tools=[CodeExecutionTool()],
    retries=4,
    instructions=(
        "You are a senior data remediation analyst. Always use the code execution tool to transform the attached CSV dataset. "
        "Load it into pandas and fix the issues identified in the validation report. "
        "Do not use eval. Do not describe hypothetical fixes: execute the transformations in the isolated environment and return the cleaned dataset. "
        "Return valid JSON only that matches the CleaningReport schema exactly. Do not use markdown, prose wrappers, or follow-up questions. "
        "Apply corrections that are strongly supported by the data, including standardizing invalid column names, normalizing null-like placeholders, "
        "coercing values to the correct data types when safe, harmonizing inconsistent formats, removing exact duplicates, addressing near duplicates conservatively, "
        "and handling sparse columns with explicit justification. "
        "Do not fabricate business values. If something is ambiguous, preserve the row and record the risk. "
        "Return the cleaned CSV as gzip+base64 in cleaned_csv_gzip_base64 together with a detailed action log."
    ),
)


async def validate_dataset(deps: DatasetDeps) -> ValidationReport:
    prompt = [
        (
            f"Dataset name: {deps.source_path.stem}\n"
            f"Source path: {deps.source_path}\n"
            f"Suggested cleaned output path: {deps.cleaned_path}\n"
            f"Placeholder values to treat as missing candidates: {PLACEHOLDER_VALUES}\n"
            "Use code execution to inspect the attached CSV file and produce the validation report."
        ),
        dataset_attachment(deps.source_path),
    ]
    result = await validation_agent.run(prompt)
    return result.output


async def clean_dataset(deps: DatasetDeps, validation_report: ValidationReport) -> CleaningReport:
    prompt = [
        (
            f"Dataset name: {deps.source_path.stem}\n"
            f"Source path: {deps.source_path}\n"
            f"Required cleaned output path: {deps.cleaned_path}\n"
            f"Report path: {deps.report_path}\n"
            "Use code execution to clean the attached CSV file.\n"
            "Use this validation report as the authoritative list of issues to address:\n"
            f"{validation_report.model_dump_json(indent=2)}"
        ),
        dataset_attachment(deps.source_path),
    ]
    result = await cleaning_agent.run(prompt)
    return result.output


def persist_outputs(deps: DatasetDeps, validation_report: ValidationReport, cleaning_report: CleaningReport) -> str:
    cleaned_csv = ungzip_base64_to_text(cleaning_report.cleaned_csv_gzip_base64)
    deps.cleaned_path.write_text(cleaned_csv, encoding="utf-8")
    deps.report_path.write_text(report_to_markdown(validation_report, cleaning_report), encoding="utf-8")
    return hashlib.sha256(cleaned_csv.encode("utf-8")).hexdigest()


async def run_pipeline(source_path: Path) -> PipelineResult:
    cleaned_path = source_path.with_name(f"{source_path.stem}_cleaned.csv")
    report_path = source_path.with_name(f"{source_path.stem}_quality_report.md")
    deps = DatasetDeps(
        source_path=source_path,
        cleaned_path=cleaned_path,
        report_path=report_path,
    )

    print(f"\n{'=' * 72}")
    print(f"Dataset : {source_path.name}")
    print(f"Cleaned : {cleaned_path.name}")
    print(f"Report  : {report_path.name}")
    print("=" * 72)

    validation_report = await validate_dataset(deps)
    cleaning_report = await clean_dataset(deps, validation_report)
    cleaned_file_sha256 = persist_outputs(deps, validation_report, cleaning_report)

    return PipelineResult(
        dataset_name=source_path.stem,
        source_path=str(source_path),
        cleaned_path=str(cleaned_path),
        report_path=str(report_path),
        validation_report=validation_report,
        cleaning_report=cleaning_report,
        cleaned_file_sha256=cleaned_file_sha256,
    )


def discover_default_datasets(base_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in base_dir.glob("*.csv")
        if not path.stem.endswith("_cleaned")
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Pydantic AI data quality pipeline.")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="CSV files to process. Defaults to all source CSV files in the current directory.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    dataset_paths = [base_dir / item for item in args.datasets] if args.datasets else discover_default_datasets(base_dir)

    if not dataset_paths:
        raise SystemExit("No source CSV files found.")

    results = []
    for dataset_path in dataset_paths:
        results.append(await run_pipeline(dataset_path))

    print("\nPipeline finished.")
    for result in results:
        print(
            json.dumps(
                {
                    "dataset_name": result.dataset_name,
                    "cleaned_path": result.cleaned_path,
                    "report_path": result.report_path,
                    "overall_completeness_pct": result.validation_report.overall_completeness_pct,
                    "actions": len(result.cleaning_report.actions),
                    "cleaned_file_sha256": result.cleaned_file_sha256,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
