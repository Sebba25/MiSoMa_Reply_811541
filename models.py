from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from tools.tools import SchemaDuplicateGroup


# --- Type Literals ---

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


# --- Schema Models ---

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


class SchemaColumnEntry(BaseModel):
    """All facts about one column: dtype inference, statistics, and naming check — merged into one place."""
    name: str
    pandas_dtype: str
    numeric_role: NUMERIC_ROLE | None = None
    string_role: STRING_ROLE | None = None
    detected_pattern: str | None = None
    rationale: str
    non_null_rows: int = Field(ge=0)
    distinct_non_null_values: int = Field(ge=0)
    numeric_parse_pct: float = Field(ge=0, le=100)
    datetime_parse_pct: float = Field(ge=0, le=100)
    empty_like_pct: float = Field(ge=0, le=100)
    sample_values: list[str] = Field(default_factory=list)
    naming_valid: bool
    rename_suggestion: str | None = None
    naming_reason: str | None = None


class SchemaHandoff(BaseModel):
    """Complete schema analysis result. Column-centric: all dtype, statistical, and naming
    facts per column are merged. Issues and duplicate groups are kept at the top level
    for easy scanning by downstream fixing agents."""
    dataset_name: str
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    columns: list[SchemaColumnEntry] = Field(default_factory=list)
    issues: list[SchemaIssue] = Field(default_factory=list)
    duplicate_groups: list[SchemaDuplicateGroup] = Field(default_factory=list)
    summary: str = ""


class SchemaSummaryOutput(BaseModel):
    summary: str


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
            "Pick the most common pattern only, no mixed results."
        ),
    )
    rationale: str


class DatasetDtypeInference(BaseModel):
    columns: list[ColumnDtypeInference]


# --- Completeness Models ---

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


# --- Consistency Models ---

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


class FindingDiff(BaseModel):
    column_name: str
    status: Literal["resolved", "improved", "unchanged", "regressed", "new"]
    before_inconsistent_rows: int = Field(ge=0)
    after_inconsistent_rows: int = Field(ge=0)
    reduction_pct: float = Field(ge=-100)
    remaining_examples: list[str] = Field(default_factory=list)


class ConsistencyVerificationReport(BaseModel):
    dataset_name: str
    original_finding_count: int = Field(ge=0)
    remaining_finding_count: int = Field(ge=0)
    diffs: list[FindingDiff] = Field(default_factory=list)
    summary: str


# --- Cleaning Models ---

class ColumnCleaningRequest(BaseModel):
    dataset_name: str
    column_name: str
    expected_pattern: str
    semantic_hint: str
    target_dtype: str | None = None
    target_role: str | None = None
    dominant_shape: str | None = None
    dominant_example_values: list[str] = Field(default_factory=list)
    example_inconsistent_values: list[str] = Field(default_factory=list)
    suggested_strategy: str


class ExampleTransformation(BaseModel):
    original_value: str
    cleaned_value: str | None = None
    rationale: str

    @field_validator("original_value", "cleaned_value", mode="before")
    @classmethod
    def coerce_to_str(cls, v):
        if v is None:
            return None
        return str(v)


class ColumnCleanerProgram(BaseModel):
    column_name: str
    function_name: str
    python_code: str = Field(
        description=(
            "Pure Python source code containing exactly one function definition named function_name. "
            "Must be importable as-is: no test code, no print statements, no variable assignments outside the function, no JSON. "
            "All imports and helper constants must be inside the function body."
        )
    )
    example_transformations: list[ExampleTransformation] = Field(default_factory=list)
    verification_summary: str
    residual_risks: list[str] = Field(default_factory=list)


VALIDATION_FAILURE_CATEGORY = Literal[
    "program_mismatch",
    "verification_report_failed",
    "runtime_exception",
    "shadowed_specific_branch",
    "dominant_value_modified",
    "outlier_unchanged",
    "outlier_returned_none",
    "wrong_output_shape",
    "not_parseable_as_target_dtype",
]


class CleanerValidationIssue(BaseModel):
    category: VALIDATION_FAILURE_CATEGORY
    severity: Literal["high", "medium"]
    message: str
    input_value: str | None = None
    actual_output: str | None = None
    expected_behavior: str


class CleanerRepairContext(BaseModel):
    request: ColumnCleaningRequest
    previous_program: ColumnCleanerProgram
    validation_issues: list[CleanerValidationIssue] = Field(default_factory=list)


class CleanerRepairExample(BaseModel):
    input_value: str
    actual_output: str | None = None
    expected_output: str | None = None
    fix_note: str


class CleanerRepairDiagnosis(BaseModel):
    should_retry: bool = True
    primary_category: VALIDATION_FAILURE_CATEGORY
    root_cause: str
    bug_location: str
    planned_fix: str
    patch_style: Literal["minimal_edit", "targeted_rewrite"]
    priority_issues: list[str] = Field(default_factory=list)
    exact_repairs: list[CleanerRepairExample] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]


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


# --- Orchestration Models ---

class OrchestrationStepResult(BaseModel):
    schema_validation: SchemaHandoff
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
