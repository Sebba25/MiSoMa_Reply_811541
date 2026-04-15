import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
import logfire
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, CodeExecutionTool, PromptedOutput
from tools.tools import (
    attach_profile_text,
    attach_text_document,
    build_column_format_facts,
    build_completeness_profile,
    build_dataset_profile,
    build_dtype_inference_text,
    build_schema_local_facts,
    gzip_text_to_base64,
    load_dataset_frame,
    normalized_schema_name,
    run_agent_with_backoff,
    SchemaLocalFacts,
)

load_dotenv()

MODEL = "openai-responses:gpt-4o-mini"


def setup_logfire() -> None:
    logfire.configure(
        data_dir=Path(__file__).parent / ".logfire",
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
    duplicate_semantic_columns: list[str] = Field(default_factory=list)
    issues: list[SchemaIssue] = Field(default_factory=list)
    summary: str


class SchemaSummaryOutput(BaseModel):
    summary: str


VALID_PANDAS_DTYPE = Literal[
    "Int64",
    "Float64",
    "datetime64[ns]",
    "string",
    "boolean",
    "object",
]

NUMERIC_ROLE = Literal[
    "measure",    # a real quantity used arithmetically (price, count, amount)
    "code",       # a numeric identifier that must not be used arithmetically (CAP, commune code)
    "indicator",  # a numeric flag or ordinal category (0/1, 1/2/3 for status)
]

STRING_ROLE = Literal[
    "identifier",   # codes, IDs, fiscal codes — must be preserved exactly
    "categorical",  # bounded low-cardinality values (status, region, gender)
    "name",         # person or place names
    "free_text",    # unstructured narrative text
]


class ColumnDtypeInference(BaseModel):
    column_name: str
    pandas_dtype: VALID_PANDAS_DTYPE
    numeric_role: NUMERIC_ROLE | None = Field(
        default=None,
        description="Only set when pandas_dtype is Int64 or Float64.",
    )
    string_role: STRING_ROLE | None = Field(
        default=None,
        description="Only set when pandas_dtype is string.",
    )
    detected_pattern: str | None = Field(
        default=None,
        description=(
            "Describe the dominant value format when a clear pattern is present. "
            "Examples: 'YYYY-MM', 'DD/MM/YYYY', 'Italian decimal comma (1.234,56)', "
            "'6-digit numeric code', 'ISO 3166-1 alpha-2 country code'. "
            "Leave null when no consistent pattern is detectable."
        ),
    )
    rationale: str


class DatasetDtypeInference(BaseModel):
    columns: list[ColumnDtypeInference]


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
    example_inconsistent_values: list[str] = Field(default_factory=list)
    evidence: str
    suggested_strategy: str


class ConsistencyValidationReport(BaseModel):
    dataset_name: str
    total_rows: int = Field(ge=0)
    format_consistency_findings: list[FormatConsistencyFinding] = Field(default_factory=list)
    summary: str


class ColumnConsistencyReport(BaseModel):
    finding: FormatConsistencyFinding | None = None
    summary: str


class ColumnCleaningRequest(BaseModel):
    dataset_name: str
    column_name: str
    expected_pattern: str
    semantic_hint: str
    dominant_shape: str | None = None
    dominant_example_values: list[str] = Field(default_factory=list)
    example_inconsistent_values: list[str] = Field(default_factory=list)
    suggested_strategy: str


class ExampleTransformation(BaseModel):
    original_value: str
    cleaned_value: str | None = None
    rationale: str


class ColumnCleanerProgram(BaseModel):
    column_name: str
    function_name: str
    python_code: str = Field(
        description="Pure Python code defining exactly one cleaning function with the returned function_name."
    )
    example_transformations: list[ExampleTransformation] = Field(default_factory=list)
    verification_summary: str
    residual_risks: list[str] = Field(default_factory=list)


class CellUpdate(BaseModel):
    row_index: int = Field(ge=0, description="Zero-based row index in the original column.")
    old_value: str | None = Field(
        default=None,
        description="Original value as text for inspection. Use null when the original value is missing.",
    )
    new_value: str | None = Field(
        default=None,
        description="Replacement value as text. Use null when the cleaned value should become missing.",
    )


class ColumnCleanerExecutionReport(BaseModel):
    column_name: str
    function_name: str
    execution_ok: bool = True
    changed_rows: int = Field(ge=0)
    sample_updates: list[CellUpdate] = Field(default_factory=list)
    unresolved_risks: list[str] = Field(default_factory=list)
    summary: str


class GeneratedCleanerArtifact(BaseModel):
    column_name: str
    function_name: str
    code_path: str
    changed_rows: int = Field(ge=0)
    summary: str


class CleaningReport(BaseModel):
    dataset_name: str
    rows_before: int = Field(ge=0)
    rows_after: int = Field(ge=0)
    columns_before: int = Field(ge=0)
    columns_after: int = Field(ge=0)
    generated_cleaners: list[GeneratedCleanerArtifact] = Field(default_factory=list)
    unresolved_risks: list[str] = Field(default_factory=list)
    cleaned_csv_gzip_base64: str = Field(description="The cleaned CSV encoded as gzip+base64.")
    summary: str


class OrchestrationStepResult(BaseModel):
    schema_validation: SchemaValidationReport
    completeness_analysis: CompletenessAnalysisReport
    consistency_validation: ConsistencyValidationReport


class CleaningPipelineResult(BaseModel):
    dataset_name: str
    source_path: str
    cleaned_path: str
    validation_results: OrchestrationStepResult
    cleaning_requests: list[ColumnCleaningRequest] = Field(default_factory=list)
    generated_programs: list[ColumnCleanerProgram] = Field(default_factory=list)
    execution_reports: list[ColumnCleanerExecutionReport] = Field(default_factory=list)
    cleaning_report: CleaningReport


def build_schema_issues(local_facts: SchemaLocalFacts) -> list[SchemaIssue]:
    issues: list[SchemaIssue] = []

    for column_name in local_facts.invalid_naming_columns:
        suggested_fix = local_facts.rename_suggestions.get(column_name, "")
        issues.append(
            SchemaIssue(
                column_name=column_name,
                issue_type="naming_standard",
                severity="high",
                evidence=local_facts.naming_reasons.get(
                    column_name,
                    "Column name violates the lowercase snake_case naming rule.",
                ),
                fix_confidence="high",
                suggested_fix=suggested_fix,
                suggested_strategy=f"Safe local rename to '{suggested_fix}'.",
            )
        )

    for duplicate_group in local_facts.duplicate_groups:
        for column_name in duplicate_group.columns:
            peer_columns = [peer for peer in duplicate_group.columns if peer != column_name]
            peer_text = ", ".join(peer_columns)
            issues.append(
                SchemaIssue(
                    column_name=column_name,
                    issue_type="duplicate_column_semantics",
                    severity="medium",
                    evidence=(
                        f"Normalized schema name matches the peer column group '{duplicate_group.canonical_name}', "
                        f"suggesting overlap with: {peer_text}."
                    ),
                    fix_confidence="medium",
                    suggested_fix="",
                    suggested_strategy=(
                        f"Compare values, null patterns, and business usage with: {peer_text} before any merge or drop."
                    ),
                )
            )

    return issues


schema_summary_agent = Agent(
    MODEL,
    output_type=PromptedOutput(SchemaSummaryOutput),
    retries=4,
    model_settings={"temperature": 0},
    instructions=(
        "You are the Schema Summary agent from the project orchestration. "
        "Inspect the attached local schema facts document. "
        "Return valid JSON only that matches the SchemaSummaryOutput schema exactly. "
        "Do not use markdown or ask follow-up questions. "
        "Execute only the schema-summary scope from Reply_projects.pdf. "
        "Do not infer new facts and do not alter the provided findings. "
        "Your only job is to write a short, precise downstream handoff summary for later validation or cleaning agents. "
        "Use the provided local facts exactly as given. "
        "Mention: how many safe naming fixes were identified, whether any duplicate-semantic groups need review, "
        "and whether any genuine data-type contradictions need manual verification. "
        "If there are no duplicate-semantic groups or no data-type risks, say that clearly. "
        "Keep the summary concrete and grounded in the provided facts, not generic."
    ),
)


dtype_inference_agent = Agent(
    MODEL,
    output_type=PromptedOutput(DatasetDtypeInference),
    retries=4,
    model_settings={"temperature": 0},
    instructions=(
        "You are a data type inference agent working on real-world dirty datasets. "
        "You receive a column-by-column profile: the column name, a sample of its values, "
        "and two statistics — numeric_parse_pct (% of values that parse as a number) "
        "and datetime_parse_pct (% of values that parse as a date/time). "
        "Your task is to infer what the physical pandas dtype WOULD BE if the column were clean. "
        "Treat dirty values (placeholders, formatting noise, unit suffixes, mixed formats) as corruption, "
        "not as evidence of the true type. Ask yourself: 'If this column were clean, what would it contain?'\n\n"
        "Use the column name, the dominant pattern in the samples, and the parse percentages together. "
        "A high numeric_parse_pct (above ~50%) is strong evidence the column should be numeric even if some samples look like strings. "
        "A high datetime_parse_pct (above ~40%) is strong evidence the column should be datetime even if formats vary.\n\n"
        "Choose the dtype from: Int64, Float64, datetime64[ns], string, boolean, object.\n\n"
        "DTYPE RULES (physical, not logical):\n"
        "- Int64: the column stores whole numbers when clean. Use this even if those numbers are codes or indicators.\n"
        "- Float64: the column stores decimal numbers when clean. Use this even if those numbers are codes.\n"
        "- datetime64[ns]: the column stores dates, times, or timestamps when clean, even if formats are inconsistent.\n"
        "- string: the column stores text. Use for names, free text, and codes that contain letters. "
        "Do NOT use string when numeric_parse_pct or datetime_parse_pct is high.\n"
        "- boolean: the column stores true/false or yes/no values when clean.\n"
        "- object: last resort only — when the column genuinely mixes incompatible types with no dominant pattern.\n\n"
        "ROLE RULES:\n"
        "- numeric_role: set only when dtype is Int64 or Float64.\n"
        "  'measure' = a real quantity used arithmetically (price, count, amount).\n"
        "  'code' = a numeric identifier not used arithmetically (postal code, region code, commune code).\n"
        "  'indicator' = a numeric flag or ordinal (0/1 flag, 1/2/3 status encoding).\n"
        "- string_role: set only when dtype is string.\n"
        "  'identifier' = codes or IDs that must be preserved exactly (fiscal code, document number).\n"
        "  'categorical' = bounded low-cardinality labels (status, region name, gender, type).\n"
        "  'name' = person or place names.\n"
        "  'free_text' = unstructured narrative or description.\n\n"
        "PATTERN RULE:\n"
        "- detected_pattern: describe the dominant clean value format when a repeating pattern is visible. "
        "Examples: 'YYYY-MM', 'DD/MM/YYYY', 'Italian decimal comma (1.234,56)', '6-digit numeric code', "
        "'2-letter province code (IT)', 'alphanumeric fiscal code (16 chars)'. "
        "Set to null when no consistent pattern is detectable.\n\n"
        "Return one entry per column in the same order as the input."
    ),
)


completeness_analysis_agent = Agent(
    MODEL,
    builtin_tools=[CodeExecutionTool()],
    output_type=PromptedOutput(CompletenessAnalysisReport),
    retries=4,
    model_settings={"temperature": 0},
    instructions=(
        "You are the Completeness Analysis agent from the project orchestration. "
        "Always use the code execution tool to inspect the attached completeness profile document. "
        "Return valid JSON only that matches the CompletenessAnalysisReport schema exactly. "
        "Do not use markdown or ask follow-up questions. "
        "Execute only the completeness-analysis scope from Reply_projects.pdf. "
        "Use the provided per-column completeness percentages, missing-like counts, missing-like percentages, placeholder examples, "
        "and overall completeness metrics from the attached document. "
        "Identify columns with missing values, placeholder tokens such as N/A, -, unknown, and empty strings, and flag sparse columns that are almost entirely empty. "
        "Be evidence-based and conservative. "
        "Do not invent row-level details that are not present in the profile. "
        "Set recommended_action to a concrete next step based on the evidence, not a generic label. "
        "If completeness_pct is 100 and missing_like_count is 0, recommended_action must be exactly 'No action needed'. "
        "If sparse_candidate is true, recommended_action should clearly say 'Investigate or consider removal due to sparsity'. "
        "If placeholder examples are present, recommend standardizing placeholder tokens and reviewing upstream data entry. "
        "If completeness is high but not perfect, recommend targeted review of missing or placeholder values in that column. "
        "Do not leave recommended_action empty. "
        "The summary should be a short downstream handoff in plain language: mention overall completeness, how many columns have missing values, "
        "which columns are the main sparse or review targets, and whether placeholder normalization should be part of later cleaning."
    ),
)


format_consistency_agent = Agent(
    MODEL,
    output_type=PromptedOutput(ColumnConsistencyReport),
    retries=4,
    model_settings={"temperature": 0},
    instructions=(
        "You are the column-level Format Consistency agent from the project orchestration. "
        "Inspect the attached local format facts document for one column. "
        "Return valid JSON only that matches the ColumnConsistencyReport schema exactly. "
        "Do not use markdown or ask follow-up questions. "
        "Execute only the within-column format-consistency scope from Reply_projects.pdf. "
        "Focus on whether the column mixes incompatible formats for identifiers, codes, periods, dates, timestamps, or monetary values. "
        "Use the local facts as the authoritative evidence: semantic hint, dominant shape, dominant examples, inconsistent row count, and inconsistent example values. "
        "Return finding=null when the column is internally consistent or when format consistency is not meaningfully applicable. "
        "Do not flag descriptive, categorical, label, note, name, or free-text columns just because their values vary in content. "
        "Do not treat missing or placeholder values as format inconsistencies by themselves; that belongs to completeness analysis. "
        "Only report a format issue when machine_format_candidate is true, the dominant shape is clear, and inconsistent_rows is meaningfully greater than zero. "
        "If machine_format_candidate is false, or the dominant shape is weak, return finding=null instead of inventing a rule. "
        "When you do return a finding, include example_inconsistent_values using the provided inconsistent examples only. "
        "Do not invent row-level details or unseen values. "
        "Be conservative and evidence-based."
    ),
)


column_cleaner_generator_agent = Agent(
    MODEL,
    builtin_tools=[CodeExecutionTool()],
    output_type=PromptedOutput(ColumnCleanerProgram),
    retries=4,
    model_settings={"temperature": 0},
    instructions=(
        "You are the column cleaner generator agent from the project orchestration. "
        "Always use the code execution tool to write and test a pure Python cleaning function from the attached request document. "
        "Return valid JSON only that matches the ColumnCleanerProgram schema exactly. "
        "Do not use markdown or ask follow-up questions. "
        "Execute only the code-generation scope for one column. "
        "The attached request document is the full specification: expected pattern, semantic hint, dominant valid examples, inconsistent examples, and the suggested normalization strategy. "
        "Write exactly one deterministic cleaning function for a single cell value. "
        "The function must be pure Python, side-effect free, self-contained, and must not require pandas or external files. "
        "The function must accept raw scalar inputs such as strings, numbers, None, and pandas-style missing values. "
        "If the input is missing, return None. "
        "The function must preserve values that are already consistent. "
        "Only normalize formats that are strongly supported by the provided examples and expected pattern. "
        "Never rename the column, drop rows, or invent business values. "
        "If a value cannot be normalized safely, preserve it unchanged and mention the ambiguity in residual_risks. "
        "Use code execution to run the function on the provided dominant and inconsistent examples before you answer. "
        "example_transformations must show the tested input-output pairs that justify the function. "
        "python_code must define exactly one function whose name matches function_name. "
        "If the function uses any import, helper constant, or helper function, include it inside python_code so the code is fully executable on its own."
    ),
)
setup_logfire()


# Temporary Cached Files
# Remove this section in the final version once the pipeline is stable.

def build_validation_results(path: Path) -> OrchestrationStepResult:
    validation_results = OrchestrationStepResult(
        schema_validation=run_schema_validation(path),
        completeness_analysis=run_completeness_analysis(path),
        consistency_validation=run_consistency_validation(path),
    )
    save_validation_results(path, validation_results)
    return validation_results


def validation_cache_dir(path: Path) -> Path:
    return path.parent / ".validation_cache"


def validation_cache_paths(path: Path) -> dict[str, Path]:
    cache_dir = validation_cache_dir(path)
    stem = path.stem
    return {
        "schema": cache_dir / f"{stem}.schema.json",
        "completeness": cache_dir / f"{stem}.completeness.json",
        "consistency": cache_dir / f"{stem}.consistency.json",
        "bundle": cache_dir / f"{stem}.validation_bundle.json",
    }


def save_validation_results(path: Path, validation_results: OrchestrationStepResult) -> None:
    cache_dir = validation_cache_dir(path)
    cache_dir.mkdir(exist_ok=True)
    cache_paths = validation_cache_paths(path)

    cache_paths["schema"].write_text(
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
    cache_paths["bundle"].write_text(
        validation_results.model_dump_json(indent=2),
        encoding="utf-8",
    )


def load_validation_results(path: Path) -> OrchestrationStepResult:
    bundle_path = validation_cache_paths(path)["bundle"]
    if not bundle_path.exists():
        raise FileNotFoundError(f"Validation cache not found: {bundle_path}")
    return OrchestrationStepResult.model_validate_json(bundle_path.read_text(encoding="utf-8"))


# Validation Stages

def run_dtype_inference(path: Path) -> DatasetDtypeInference:
    df = load_dataset_frame(path)
    text = build_dtype_inference_text(df)
    prompt = [
        "Infer the correct pandas dtype for each column based on the attached CSV sample.",
        attach_text_document(text),
    ]
    result = run_agent_with_backoff(dtype_inference_agent, prompt)
    return result.output


def run_schema_validation(path: Path) -> SchemaValidationReport:
    df = load_dataset_frame(path)
    dtype_inference = run_dtype_inference(path)
    dtype_overrides = {col.column_name: col.pandas_dtype for col in dtype_inference.columns}
    profile = build_dataset_profile(df, path.stem, dtype_overrides=dtype_overrides)
    local_facts = build_schema_local_facts(profile)
    issues = build_schema_issues(local_facts)
    prompt = [
        (
            f"Summarize the provided local schema facts for dataset {path.stem}. "
            "This is the schema-summary handoff only. "
            "Do not infer new findings. Summarize the provided findings for a later cleaner or validator."
        ),
        attach_profile_text(local_facts),
    ]
    result = run_agent_with_backoff(schema_summary_agent, prompt)
    return SchemaValidationReport(
        dataset_name=local_facts.dataset_name,
        total_rows=local_facts.total_rows,
        total_columns=local_facts.total_columns,
        column_names=local_facts.column_names,
        invalid_naming_columns=local_facts.invalid_naming_columns,
        duplicate_semantic_columns=local_facts.duplicate_semantic_columns,
        issues=issues,
        summary=result.output.summary,
    )


def run_completeness_analysis(path: Path) -> CompletenessAnalysisReport:
    df = load_dataset_frame(path)
    profile = build_completeness_profile(df, path.stem)
    prompt = [
        (
            f"Analyze the attached completeness profile for dataset {path.stem}. "
            "Use Python in code execution to inspect the profile document. "
            "This is step 2 of the orchestration only: Completeness Analysis. "
            "Use the provided metrics to summarize per-column completeness, detect missing-like and placeholder values, "
            "identify actual placeholder tokens present in the dataset, and flag sparse columns that may be candidates for removal or investigation."
        ),
        attach_profile_text(profile),
    ]
    result = run_agent_with_backoff(completeness_analysis_agent, prompt)
    return result.output


def run_format_consistency_validation(path: Path) -> ConsistencyValidationReport:
    df = load_dataset_frame(path)
    format_findings: list[FormatConsistencyFinding] = []

    for column_name in df.columns:
        result = run_column_format_check(df, column_name, path.stem, "original validation")
        if result.finding is not None:
            format_findings.append(result.finding)

    return ConsistencyValidationReport(
        dataset_name=path.stem,
        total_rows=len(df),
        format_consistency_findings=format_findings,
        summary=(
            f"Analyzed all {len(df.columns)} columns individually for format consistency and "
            f"detected {len(format_findings)} format issues."
        ),
    )


def run_column_format_check(
    df: pd.DataFrame,
    column_name: str,
    dataset_name: str,
    context_label: str,
) -> ColumnConsistencyReport:
    format_facts = build_column_format_facts(df, column_name)
    if not format_facts.machine_format_candidate or format_facts.inconsistent_rows <= 0:
        return ColumnConsistencyReport(
            finding=None,
            summary=(
                f"No actionable machine-format inconsistency detected for column '{column_name}' "
                f"in {context_label}."
            ),
        )
    prompt = [
        (
            f"Analyze the attached local format facts for dataset {dataset_name}. "
            f"Column name: {column_name}. "
            f"Evaluation context: {context_label}. "
            f"Total rows in the full dataset: {len(df)}. "
            "Check whether the column mixes incompatible formats internally. "
            "Return finding=null if the column is consistent or if format consistency is not meaningfully applicable."
        ),
        attach_profile_text(format_facts),
    ]
    result = run_agent_with_backoff(format_consistency_agent, prompt)
    output = result.output
    if output.finding is not None and output.finding.inconsistent_rows <= 0:
        return ColumnConsistencyReport(
            finding=None,
            summary=output.summary,
        )
    return output


def run_consistency_validation(path: Path) -> ConsistencyValidationReport:
    return run_format_consistency_validation(path)


# Cleaning Stage

def cleaning_cache_dir(path: Path) -> Path:
    return path.parent / ".cleaning_cache" / path.stem


def generated_cleaner_path(path: Path, column_name: str) -> Path:
    return cleaning_cache_dir(path) / "generated_cleaners" / f"{normalized_schema_name(column_name)}.py"


def cleaned_dataset_path(path: Path) -> Path:
    return cleaning_cache_dir(path) / f"{path.stem}.cleaned.csv"


def build_column_cleaning_request(
    dataset_name: str,
    column_name: str,
    finding: FormatConsistencyFinding,
    format_facts,
) -> ColumnCleaningRequest:
    return ColumnCleaningRequest(
        dataset_name=dataset_name,
        column_name=column_name,
        expected_pattern=finding.expected_pattern,
        semantic_hint=format_facts.semantic_hint,
        dominant_shape=format_facts.dominant_shape,
        dominant_example_values=format_facts.dominant_example_values,
        example_inconsistent_values=finding.example_inconsistent_values,
        suggested_strategy=finding.suggested_strategy,
    )


def save_generated_cleaner(path: Path, program: ColumnCleanerProgram) -> Path:
    code_path = generated_cleaner_path(path, program.column_name)
    code_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.write_text(program.python_code.strip() + "\n", encoding="utf-8")
    return code_path


def _normalize_scalar_for_comparison(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def load_cleaner_callable(program: ColumnCleanerProgram):
    namespace: dict[str, Any] = {}
    exec(program.python_code, namespace)
    cleaner = namespace.get(program.function_name)
    if not callable(cleaner):
        raise ValueError(
            f"Generated cleaner for column '{program.column_name}' did not define callable '{program.function_name}'."
        )
    return cleaner


def build_local_execution_report(
    series: pd.Series,
    program: ColumnCleanerProgram,
    request: ColumnCleaningRequest,
) -> tuple[ColumnCleanerExecutionReport, pd.Series | None]:
    try:
        cleaner = load_cleaner_callable(program)
    except Exception as error:
        return (
            ColumnCleanerExecutionReport(
                column_name=program.column_name,
                function_name=program.function_name,
                execution_ok=False,
                changed_rows=0,
                sample_updates=[],
                unresolved_risks=[f"Cleaner code failed to load locally: {error}"],
                summary="Local preflight failed while loading the generated cleaner code.",
            ),
            None,
        )

    sample_values: list[Any] = []
    for value in request.dominant_example_values + request.example_inconsistent_values:
        if value not in sample_values:
            sample_values.append(value)
    for transformation in program.example_transformations:
        if transformation.original_value not in sample_values:
            sample_values.append(transformation.original_value)

    try:
        for value in sample_values:
            cleaner(value)
    except Exception as error:
        return (
            ColumnCleanerExecutionReport(
                column_name=program.column_name,
                function_name=program.function_name,
                execution_ok=False,
                changed_rows=0,
                sample_updates=[],
                unresolved_risks=[f"Cleaner code failed local preflight on sample values: {error}"],
                summary="Local preflight failed while testing the generated cleaner on sample values.",
            ),
            None,
        )

    cleaned = series.astype("object").copy()
    sample_updates: list[CellUpdate] = []
    changed_rows = 0

    try:
        for row_index, original_value in series.items():
            cleaned_value = cleaner(original_value)
            normalized_original = _normalize_scalar_for_comparison(original_value)
            normalized_cleaned = _normalize_scalar_for_comparison(cleaned_value)

            if normalized_original != normalized_cleaned:
                changed_rows += 1
                if len(sample_updates) < 10:
                    sample_updates.append(
                        CellUpdate(
                            row_index=int(row_index),
                            old_value=normalized_original,
                            new_value=normalized_cleaned,
                        )
                    )

            cleaned.at[row_index] = pd.NA if normalized_cleaned is None else normalized_cleaned
    except Exception as error:
        return (
            ColumnCleanerExecutionReport(
                column_name=program.column_name,
                function_name=program.function_name,
                execution_ok=False,
                changed_rows=0,
                sample_updates=sample_updates,
                unresolved_risks=[f"Cleaner code failed during full local execution: {error}"],
                summary="Local execution failed while applying the generated cleaner to the full column.",
            ),
            None,
        )

    summary = (
        "Local preflight succeeded and the generated cleaner was applied to the full column."
        if changed_rows > 0
        else "Local preflight succeeded and the generated cleaner made no changes."
    )
    return (
        ColumnCleanerExecutionReport(
            column_name=program.column_name,
            function_name=program.function_name,
            execution_ok=True,
            changed_rows=changed_rows,
            sample_updates=sample_updates,
            unresolved_risks=[],
            summary=summary,
        ),
        cleaned,
    )


def run_column_cleaner_program(
    dataset_name: str,
    request: ColumnCleaningRequest,
) -> ColumnCleanerProgram:
    prompt = [
        (
            f"Generate and verify a pure Python cleaning function for dataset {dataset_name}, column {request.column_name}. "
            "Use the request document only. "
            "Verify the function against the example values before answering."
        ),
        attach_profile_text(request),
    ]
    return run_agent_with_backoff(column_cleaner_generator_agent, prompt).output

def run_cleaning(
    path: Path,
    validation_results: OrchestrationStepResult | None = None,
    reuse_saved_validation: bool = False,
) -> CleaningPipelineResult:
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

        print(f"running cleaner generator for '{column_name}'...", file=os.sys.stderr)
        program = run_column_cleaner_program(path.stem, request)
        generated_programs.append(program)
        code_path = save_generated_cleaner(path, program)

        print(f"running local cleaner execution for '{column_name}'...", file=os.sys.stderr)
        execution_report, cleaned_series = build_local_execution_report(df[column_name], program, request)
        execution_reports.append(execution_report)

        column_risks = [
            f"{column_name}: {risk}"
            for risk in [*program.residual_risks, *execution_report.unresolved_risks]
            if risk
        ]
        unresolved_risks.extend(column_risks)

        changed_rows = execution_report.changed_rows
        if execution_report.execution_ok and cleaned_series is not None:
            df[column_name] = cleaned_series
        else:
            unresolved_risks.append(
                f"{column_name}: local preflight/execution failed, so the generated cleaner was not applied."
            )

        generated_cleaners.append(
            GeneratedCleanerArtifact(
                column_name=column_name,
                function_name=program.function_name,
                code_path=str(code_path),
                changed_rows=changed_rows,
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
            f"Saved the cleaned dataset to {cleaned_path} and stored per-column cleaner code under "
            f"{output_dir / 'generated_cleaners'}."
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


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Pydantic AI dataset validation agents.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="Data/spesa.csv",
        help="CSV file to inspect. Defaults to Data/spesa.csv.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "dtype", "schema", "completeness", "consistency", "clean"],
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
        "--reuse-validation",
        action="store_true",
        help="Reuse saved validation results from .validation_cache when running the cleaning stage.",
    )
    args = parser.parse_args()

    dataset_path = Path(__file__).parent / args.dataset
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    if args.stage == "dtype":
        inference = run_dtype_inference(dataset_path)
        col_w   = max(len(c.column_name)  for c in inference.columns)
        dtype_w = max(len(c.pandas_dtype) for c in inference.columns)
        role_w  = max(len(c.numeric_role or c.string_role or "") for c in inference.columns)
        pat_w   = max(len(c.detected_pattern or "") for c in inference.columns)
        header  = f"{'COLUMN':<{col_w}}  {'DTYPE':<{dtype_w}}  {'ROLE':<{role_w}}  {'PATTERN':<{pat_w}}  RATIONALE"
        print(header)
        print("-" * len(header))
        for c in inference.columns:
            role    = c.numeric_role or c.string_role or ""
            pattern = c.detected_pattern or ""
            print(f"{c.column_name:<{col_w}}  {c.pandas_dtype:<{dtype_w}}  {role:<{role_w}}  {pattern:<{pat_w}}  {c.rationale}")
        raise SystemExit(0)
    elif args.stage == "schema":
        result = run_schema_validation(dataset_path)
    elif args.stage == "completeness":
        result = run_completeness_analysis(dataset_path)
    elif args.stage == "consistency":
        if args.consistency_agent == "format":
            result = run_format_consistency_validation(dataset_path)
        else:
            result = run_consistency_validation(dataset_path)
    elif args.stage == "clean":
        result = run_cleaning(dataset_path, reuse_saved_validation=args.reuse_validation)
    else:
        result = build_validation_results(dataset_path)

    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
