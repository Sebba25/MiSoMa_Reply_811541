import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, CodeExecutionTool, PromptedOutput
from pydantic_ai.messages import BinaryContent

load_dotenv()

MODEL = "openai-responses:gpt-4o-mini"


def setup_logfire() -> None:
    logfire.configure(
        send_to_logfire="if-token-present",
        service_name="pydantic-dataset-smoke-test",
        service_version="1.0.0",
        environment=os.getenv("LOGFIRE_ENVIRONMENT", "dev"),
    )
    logfire.instrument_pydantic_ai()
    if os.getenv("LOGFIRE_CAPTURE_HTTPX") == "1":
        logfire.instrument_httpx(capture_all=True)


class SchemaIssue(BaseModel):
    column_name: str
    issue_type: str = Field(
        description="Use values such as naming_standard, reserved_word_risk, inferred_type_mismatch, or duplicate_column_semantics."
    )
    severity: str = Field(description="Use low, medium, or high.")
    evidence: str
    fix_confidence: str = Field(description="Use high, medium, or low.")
    suggested_fix: str
    suggested_strategy: str = Field(
        description="If fix_confidence is not high, provide a cautious remediation strategy instead of a precise correction."
    )


class SchemaValidationReport(BaseModel):
    dataset_name: str
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    column_names: list[str] = Field(default_factory=list)
    invalid_naming_columns: list[str] = Field(default_factory=list)
    data_type_risk_columns: list[str] = Field(default_factory=list)
    duplicate_semantic_columns: list[str] = Field(default_factory=list)
    issues: list[SchemaIssue] = Field(default_factory=list)
    summary: str


class CompletenessColumnFinding(BaseModel):
    column_name: str
    completeness_pct: float = Field(ge=0, le=100)
    missing_like_count: int = Field(ge=0)
    missing_like_examples: list[str] = Field(default_factory=list)
    sparse_candidate: bool = False
    recommended_action: str


class CompletenessAnalysisReport(BaseModel):
    dataset_name: str
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    overall_completeness_pct: float = Field(ge=0, le=100)
    columns_with_missing_values: list[str] = Field(default_factory=list)
    sparse_columns: list[str] = Field(default_factory=list)
    placeholder_values_detected: list[str] = Field(default_factory=list)
    per_column: list[CompletenessColumnFinding] = Field(default_factory=list)
    summary: str


class FormatConsistencyFinding(BaseModel):
    column_name: str
    expected_pattern: str
    inconsistent_rows: int = Field(ge=0)
    evidence: str
    suggested_strategy: str


class CrossColumnFinding(BaseModel):
    columns: list[str] = Field(default_factory=list)
    rule_checked: str
    applicable: bool
    affected_rows: int = Field(ge=0)
    evidence: str
    suggested_strategy: str


class DuplicateFinding(BaseModel):
    duplicate_type: str = Field(description="Use exact or near.")
    key_columns: list[str] = Field(default_factory=list)
    affected_rows: int = Field(ge=0)
    evidence: str
    suggested_strategy: str


class ConsistencyValidationReport(BaseModel):
    dataset_name: str
    total_rows: int = Field(ge=0)
    format_consistency_findings: list[FormatConsistencyFinding] = Field(default_factory=list)
    cross_column_findings: list[CrossColumnFinding] = Field(default_factory=list)
    duplicate_findings: list[DuplicateFinding] = Field(default_factory=list)
    summary: str


class OrchestrationStepResult(BaseModel):
    schema_validation: SchemaValidationReport
    completeness_analysis: CompletenessAnalysisReport
    consistency_validation: ConsistencyValidationReport


schema_validator_agent = Agent(
    MODEL,
    builtin_tools=[CodeExecutionTool()],
    output_type=PromptedOutput(SchemaValidationReport),
    retries=4,
    instructions=(
        "You are the Schema Validator agent from the project orchestration. "
        "Always use the code execution tool to inspect the attached CSV file. "
        "Return valid JSON only that matches the SchemaValidationReport schema exactly. "
        "Do not use markdown or ask follow-up questions. "
        "Execute only the schema validation part of the project scope from Reply_projects.pdf. "
        "Focus on column-name standards, naming convention violations, special characters, casing, reserved-word risk, "
        "duplicate or overlapping column semantics, and obvious inferred data-type mismatches. "
        "Be evidence-based and conservative. "
        "Use this naming policy: valid names are lowercase snake_case identifiers; leading underscore is allowed for technical id columns; "
        "names are invalid only if they contain spaces, hyphens, percent signs, uppercase letters, or start with a digit. "
        "Do not mark already valid names like rata, ente, descrizione, spesa, or tipo_imposta as invalid. "
        "Use duplicate_column_semantics only when two columns clearly represent the same business concept, not just because they are both textual. "
        "Use inferred_type_mismatch only when the column name and the observed values strongly indicate a different semantic type, "
        "for example a date/time column with free text, a numeric code column containing many non-code strings, or a monetary column containing non-numeric tokens. "
        "Do not assume every object dtype should be numeric. IDs such as _id may legitimately be alphanumeric strings. "
        "Inspect the full dataset, not just a head sample, before deciding. "
        "For every issue, assign fix_confidence as high, medium, or low. "
        "Only give a specific corrective action in suggested_fix when fix_confidence is high. "
        "When confidence is medium or low, keep suggested_fix generic and use suggested_strategy to describe a safer approach, "
        "such as profiling values, validating against source-system metadata, or normalizing with manual review."
    ),
)


completeness_analysis_agent = Agent(
    MODEL,
    builtin_tools=[CodeExecutionTool()],
    output_type=PromptedOutput(CompletenessAnalysisReport),
    retries=4,
    instructions=(
        "You are the Completeness Analysis agent from the project orchestration. "
        "Always use the code execution tool to inspect the attached CSV file. "
        "Return valid JSON only that matches the CompletenessAnalysisReport schema exactly. "
        "Do not use markdown or ask follow-up questions. "
        "Execute only the completeness-analysis scope from Reply_projects.pdf. "
        "Compute completeness percentage per column and overall, count null-like and placeholder values, "
        "identify placeholder tokens such as N/A, -, unknown, //, empty strings, and flag sparse columns that are almost entirely empty. "
        "Be evidence-based and conservative."
    ),
)


consistency_validation_agent = Agent(
    MODEL,
    builtin_tools=[CodeExecutionTool()],
    output_type=PromptedOutput(ConsistencyValidationReport),
    retries=4,
    instructions=(
        "You are the Consistency Validation agent from the project orchestration. "
        "Always use the code execution tool to inspect the attached CSV file. "
        "Return valid JSON only that matches the ConsistencyValidationReport schema exactly. "
        "Do not use markdown or ask follow-up questions. "
        "Execute only the consistency-validation scope from Reply_projects.pdf. "
        "Focus on format consistency within columns, cross-column logical checks when they are actually applicable, "
        "and exact plus near-duplicate detection. "
        "Only report cross-column checks that make sense for the current dataset. "
        "If a textbook example like birth_date versus age is not applicable here, say so through applicable=false and explain why. "
        "For duplicate detection, use evidence from the real dataset and suggest a cautious strategy rather than deleting records blindly."
    ),
)

setup_logfire()


def attach_csv(path: Path) -> BinaryContent:
    return BinaryContent(data=path.read_bytes(), media_type="text/csv")


def run_schema_validation(path: Path) -> SchemaValidationReport:
    prompt = [
        (
            f"Analyze the attached CSV dataset named {path.stem}. "
            "Use Python in code execution to load it with pandas and inspect the real file. "
            "This is step 1 of the orchestration only: Schema Validator. "
            "Check column names against naming standards, identify naming convention problems, "
            "flag special characters and reserved-word risks, identify duplicate semantic columns, "
            "and flag columns whose observed values strongly suggest a different data type than the current representation."
        ),
        attach_csv(path),
    ]
    result = schema_validator_agent.run_sync(prompt)
    return result.output


def run_completeness_analysis(path: Path) -> CompletenessAnalysisReport:
    prompt = [
        (
            f"Analyze the attached CSV dataset named {path.stem}. "
            "Use Python in code execution to load it with pandas and inspect the real file. "
            "This is step 2 of the orchestration only: Completeness Analysis. "
            "Compute completeness percentage per column and overall, count null, empty, and placeholder values, "
            "identify the actual placeholder tokens present in the dataset, and flag sparse columns that may be candidates for removal or investigation."
        ),
        attach_csv(path),
    ]
    result = completeness_analysis_agent.run_sync(prompt)
    return result.output


def run_consistency_validation(path: Path) -> ConsistencyValidationReport:
    prompt = [
        (
            f"Analyze the attached CSV dataset named {path.stem}. "
            "Use Python in code execution to load it with pandas and inspect the real file. "
            "This is step 3 of the orchestration only: Consistency Validation. "
            "Check format consistency inside columns, especially code, period, date, timestamp, and monetary fields. "
            "Run cross-column checks only when a real logical relationship exists in this dataset. "
            "Detect exact duplicates and also near-duplicates using sensible business keys when possible."
        ),
        attach_csv(path),
    ]
    result = consistency_validation_agent.run_sync(prompt)
    return result.output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the schema-validator stage on a real CSV dataset.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="spesa.csv",
        help="CSV file to inspect. Defaults to spesa.csv.",
    )
    args = parser.parse_args()

    dataset_path = Path(__file__).parent / args.dataset
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    result = OrchestrationStepResult(
        schema_validation=run_schema_validation(dataset_path),
        completeness_analysis=run_completeness_analysis(dataset_path),
        consistency_validation=run_consistency_validation(dataset_path),
    )
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
