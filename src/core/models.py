"""models.py: structured output contracts for every pipeline agent.

Pydantic AI derives a JSON schema from each BaseModel defined here and injects it into the
agent's prompt via PromptedOutput(...), so the LLM knows exactly what structure to return.
The same models are then used as typed Python objects throughout the rest of the pipeline —
they carry findings from one stage to the next and are serialised to the validation and
cleaning caches.

Field(description=...) annotations are part of the contract: those descriptions appear
verbatim in the JSON schema the LLM sees, so they act as fine-grained instructions for
each output field.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.tools import SchemaDuplicateGroup


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


class AnomalySummaryOutput(BaseModel):
    summary: str


class CrossColumnSummaryOutput(BaseModel):
    summary: str


class DuplicateSummaryOutput(BaseModel):
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
    renamed_to: str | None = None


class ConsistencyVerificationReport(BaseModel):
    dataset_name: str
    original_finding_count: int = Field(ge=0)
    remaining_finding_count: int = Field(ge=0)
    diffs: list[FindingDiff] = Field(default_factory=list)
    summary: str


# --- Anomaly Models ---

class AnomalyFinding(BaseModel):
    column_name: str
    anomaly_type: Literal["numeric_outlier", "rare_category", "negative_value"]
    severity: Literal["low", "medium", "high"]
    affected_rows: int = Field(ge=0)
    example_values: list[str] = Field(default_factory=list)
    evidence: str
    suggested_action: str


class AnomalyDetectionReport(BaseModel):
    dataset_name: str
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    findings: list[AnomalyFinding] = Field(default_factory=list)
    summary: str = ""


# --- Cross-Column Models ---

class CrossColumnFinding(BaseModel):
    columns: list[str] = Field(default_factory=list)
    check_type: Literal[
        "duplicate_semantic_conflict",
        "exact_duplicate_columns",
        "near_duplicate_columns",
        "year_month_period_mismatch",
        "date_order_violation",
    ]
    severity: Literal["medium", "high"]
    affected_rows: int = Field(ge=0)
    example_row_indices: list[int] = Field(default_factory=list)
    similarity_pct: float | None = Field(default=None, ge=0, le=100)
    evidence: str
    suggested_action: str


class CrossColumnValidationReport(BaseModel):
    dataset_name: str
    total_rows: int = Field(ge=0)
    findings: list[CrossColumnFinding] = Field(default_factory=list)
    summary: str = ""


# --- Duplicate Detection Models ---

class DuplicateRecordGroup(BaseModel):
    duplicate_type: Literal["exact_row", "near_duplicate"]
    row_indices: list[int] = Field(default_factory=list)
    key_columns: list[str] = Field(default_factory=list)
    evidence: str
    suggested_action: str


class DuplicateDetectionReport(BaseModel):
    dataset_name: str
    total_rows: int = Field(ge=0)
    groups: list[DuplicateRecordGroup] = Field(default_factory=list)
    summary: str = ""


# --- Remediation Models ---

REMEDIATION_ACTION_TYPE = Literal[
    "rename_column",
    "replace_placeholders_with_null",
    "generate_cleaner",
    "drop_exact_duplicate_column",
    "drop_exact_duplicate_rows",
    "cast_dtype",
    "manual_review",
    "report_only",
    "drop_rows_candidate",
]

REMEDIATION_OBJECT_TYPE = Literal["column", "column_pair", "row_group", "dataset"]
REMEDIATION_CONFIDENCE = Literal["low", "medium", "high"]
REMEDIATION_RISK_LEVEL = Literal["low", "medium", "high"]
REMEDIATION_STATUS = Literal["planned", "applied", "proposed_not_applied", "failed", "not_needed"]


class RemediationAction(BaseModel):
    action_id: str
    action_type: REMEDIATION_ACTION_TYPE
    object_type: REMEDIATION_OBJECT_TYPE
    target: dict[str, Any] = Field(default_factory=dict)
    source_check: str
    confidence: REMEDIATION_CONFIDENCE
    risk_level: REMEDIATION_RISK_LEVEL
    auto_apply: bool
    status: REMEDIATION_STATUS
    reason: str
    preview_stats: dict[str, Any] = Field(default_factory=dict)


class RemediationPlan(BaseModel):
    dataset_name: str
    actions: list[RemediationAction] = Field(default_factory=list)
    summary: str = ""


class FinalPipelineReport(BaseModel):
    dataset_name: str
    validation_summary: dict[str, int] = Field(default_factory=dict)
    applied_actions: list[RemediationAction] = Field(default_factory=list)
    proposed_not_applied_actions: list[RemediationAction] = Field(default_factory=list)
    failed_actions: list[RemediationAction] = Field(default_factory=list)
    not_needed_actions: list[RemediationAction] = Field(default_factory=list)
    duplicate_row_drop_candidates: list[RemediationAction] = Field(default_factory=list)
    manual_review_queue: list[RemediationAction] = Field(default_factory=list)
    cleaning_summary: str = ""
    verification_summary: str = ""
    verification_diffs: list[FindingDiff] = Field(default_factory=list)
    generated_cleaners: list["GeneratedCleanerArtifact"] = Field(default_factory=list)
    total_rows_cleaned: int = 0
    non_null_counts_cleaned: dict[str, int] = Field(default_factory=dict)
    completeness_details: list[CompletenessColumnFinding] = Field(default_factory=list)
    anomaly_findings: list[AnomalyFinding] = Field(default_factory=list)
    cross_column_findings: list[CrossColumnFinding] = Field(default_factory=list)
    duplicate_groups: list[DuplicateRecordGroup] = Field(default_factory=list)
    unresolved_risks: list[str] = Field(default_factory=list)
    summary: str = ""


class NarrativeReportSection(BaseModel):
    heading: str = Field(description="Section title, e.g. 'Validazione dello Schema'.")
    body: str = Field(
        description=(
            "Markdown-formatted prose for this section. Must be detailed and evidence-rich: "
            "include column names, row counts, percentages, concrete examples, and before/after comparisons. "
            "Use markdown tables, bullet lists, and sub-headings where appropriate. "
            "Minimum 150 words per section."
        )
    )


class NarrativeFrontMatter(BaseModel):
    title: str = Field(description="Report title including the dataset name.")
    executive_summary: str = Field(
        description=(
            "A comprehensive executive summary (8-12 sentences). Cover: dataset dimensions, "
            "overall quality posture, key findings by category, total actions applied vs. proposed, "
            "verification outcome, and residual risk assessment."
        )
    )
    recommendations: list[str] = Field(
        default_factory=list,
        min_length=3,
        description="Prioritized list of actionable next steps for the data steward. At least 3 items.",
    )


class NarrativeReport(BaseModel):
    title: str = Field(description="Report title including the dataset name.")
    executive_summary: str = Field(
        description=(
            "A comprehensive executive summary (8-12 sentences). Cover: dataset dimensions, "
            "overall quality posture, key findings by category, total actions applied vs. proposed, "
            "verification outcome, and residual risk assessment."
        )
    )
    sections: list[NarrativeReportSection] = Field(
        min_length=8,
        description="At least 8 sections covering all aspects of the quality analysis.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        min_length=3,
        description="Prioritized list of actionable next steps for the data steward. At least 3 items.",
    )


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
    enforce_year_only_yyyymm_january: bool = Field(
        default=False,
        description=(
            "When true, recoverable year-only YYYYMM inputs such as 'Rata 2024' must default to January "
            "during validation and may not be mapped to null."
        ),
    )
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
    verification_summary: str = ""
    residual_risks: list[str] = Field(default_factory=list)


VALIDATION_FAILURE_CATEGORY = Literal[
    "program_mismatch",
    "verification_report_failed",
    "non_self_contained_function",
    "runtime_exception",
    "shadowed_specific_branch",
    "dominant_value_modified",
    "outlier_unchanged",
    "outlier_returned_none",
    "unrecoverable_outlier_not_nulled",
    "wrong_output_shape",
    "not_parseable_as_target_dtype",
    "not_matching_target_pattern",
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
    example_transformations: list[ExampleTransformation] = Field(default_factory=list)


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
    anomaly_detection: AnomalyDetectionReport | None = None
    cross_column_validation: CrossColumnValidationReport | None = None
    duplicate_detection: DuplicateDetectionReport | None = None


class CleaningPipelineResult(BaseModel):
    dataset_name: str
    source_path: str
    cleaned_path: str
    validation_results: OrchestrationStepResult
    remediation_plan: RemediationPlan | None = None
    cleaning_requests: list[ColumnCleaningRequest] = Field(default_factory=list)
    generated_programs: list[ColumnCleanerProgram] = Field(default_factory=list)
    execution_reports: list[ColumnCleanerExecutionReport] = Field(default_factory=list)
    cleaning_report: CleaningReport
    verification_report: ConsistencyVerificationReport | None = None
    final_report: FinalPipelineReport | None = None
