"""Final report assembly + narrative generation.

``build_final_report`` merges validation, remediation, cleaning and
verification artifacts into a single ``FinalPipelineReport``;
``_generate_narrative_report_chunked`` hands that report to the narrative agents
to produce the Markdown summary saved alongside the cleaned CSV.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from core.models import (
    CleaningReport,
    ConsistencyVerificationReport,
    FinalPipelineReport,
    NarrativeReport,
    NarrativeReportSection,
    OrchestrationStepResult,
    RemediationPlan,
)

from .paths import cleaned_dataset_path, cleaning_cache_dir, final_report_path


def save_final_report(path: Path, report: FinalPipelineReport) -> Path:
    '''Saves the final structured pipeline report as JSON and returns its output path.'''
    # Compute the standard JSON output path for the final report.
    report_path = final_report_path(path)
    # Ensure the parent directory exists before writing the file.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize the Pydantic model as pretty-printed JSON text.
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report_path


def _compute_cleaned_non_null_counts(dataset_path: Path | None) -> tuple[int, dict[str, int]]:
    """Read the cleaned CSV and return (total_rows, {column: non_null_count}).

    Used to ground the narrative report's '% Non-Null' numbers in real data
    instead of letting the agent guess from the cleaning summary string.
    Returns (0, {}) if the cleaned file cannot be loaded.
    """
    # If no dataset path is available, there is no cleaned file to inspect.
    if dataset_path is None:
        return 0, {}
    # Build the path of the cleaned CSV produced by the application stage.
    cleaned_path = cleaned_dataset_path(dataset_path)
    if not cleaned_path.exists():
        return 0, {}
    try:
        # Import lazily because this helper is only needed while assembling the final report.
        from tools import load_dataset_frame

        # Load the cleaned dataframe and compute one non-null count per column.
        df = load_dataset_frame(cleaned_path)
        total = len(df)
        counts = {str(col): int(df[col].notna().sum()) for col in df.columns}
        return total, counts
    except Exception as error:
        # If the cleaned file cannot be read, log a warning and return safe empty values.
        print(f"[report] warning: could not compute non-null counts: {error}", file=sys.stderr)
        return 0, {}


def build_final_report(
    validation_results: OrchestrationStepResult,
    remediation_plan: RemediationPlan,
    cleaning_report: CleaningReport,
    verification_report: ConsistencyVerificationReport | None,
    dataset_path: Path | None = None,
) -> FinalPipelineReport:
    '''Combines validation, remediation, cleaning, and verification outputs into one final report object.'''
    # Build compact count summaries for each validation area.
    validation_summary = {
        "schema_issues": len(validation_results.schema_validation.issues),
        "completeness_columns_with_missing": len(validation_results.completeness_analysis.columns_with_missing_values),
        "consistency_findings": len(validation_results.consistency_validation.format_consistency_findings),
        "anomaly_findings": len(validation_results.anomaly_detection.findings) if validation_results.anomaly_detection else 0,
        "cross_column_findings": len(validation_results.cross_column_validation.findings) if validation_results.cross_column_validation else 0,
        "duplicate_groups": len(validation_results.duplicate_detection.groups) if validation_results.duplicate_detection else 0,
    }

    # Group remediation actions by status so the final report can summarize what happened.
    applied_actions = [action for action in remediation_plan.actions if action.status == "applied"]
    proposed_not_applied_actions = [action for action in remediation_plan.actions if action.status == "proposed_not_applied"]
    failed_actions = [action for action in remediation_plan.actions if action.status == "failed"]
    not_needed_actions = [action for action in remediation_plan.actions if action.status == "not_needed"]
    duplicate_row_drop_candidates = [
        action
        for action in remediation_plan.actions
        if action.action_type in {"drop_rows_candidate", "drop_exact_duplicate_rows"}
    ]
    manual_review_queue = [
        action
        for action in remediation_plan.actions
        if action.status == "proposed_not_applied" and action.action_type == "manual_review"
    ]

    # Use the verification summary when available, otherwise provide a fallback message.
    verification_summary = verification_report.summary if verification_report is not None else "Verification was not run."
    # Build one overall summary sentence for the final report.
    summary = (
        f"Validation found {sum(validation_summary.values())} section-level findings/signals. "
        f"Applied {len(applied_actions)} remediation actions, left {len(proposed_not_applied_actions)} proposed without auto-apply, "
        f"and recorded {len(failed_actions)} failed actions."
    )

    # Read authoritative non-null counts from the cleaned CSV for later narrative tables.
    total_rows_cleaned, non_null_counts_cleaned = _compute_cleaned_non_null_counts(dataset_path)

    # Normalize optional validation sections into plain lists.
    anomaly_findings = (
        list(validation_results.anomaly_detection.findings)
        if validation_results.anomaly_detection is not None
        else []
    )
    cross_column_findings = (
        list(validation_results.cross_column_validation.findings)
        if validation_results.cross_column_validation is not None
        else []
    )
    duplicate_groups = (
        list(validation_results.duplicate_detection.groups)
        if validation_results.duplicate_detection is not None
        else []
    )
    completeness_details = list(validation_results.completeness_analysis.per_column)

    # Assemble the final structured report model.
    return FinalPipelineReport(
        dataset_name=validation_results.schema_validation.dataset_name,
        validation_summary=validation_summary,
        applied_actions=applied_actions,
        proposed_not_applied_actions=proposed_not_applied_actions,
        failed_actions=failed_actions,
        not_needed_actions=not_needed_actions,
        duplicate_row_drop_candidates=duplicate_row_drop_candidates,
        manual_review_queue=manual_review_queue,
        cleaning_summary=cleaning_report.summary,
        verification_summary=verification_summary,
        verification_diffs=verification_report.diffs if verification_report is not None else [],
        generated_cleaners=cleaning_report.generated_cleaners,
        total_rows_cleaned=total_rows_cleaned,
        non_null_counts_cleaned=non_null_counts_cleaned,
        completeness_details=completeness_details,
        anomaly_findings=anomaly_findings,
        cross_column_findings=cross_column_findings,
        duplicate_groups=duplicate_groups,
        unresolved_risks=cleaning_report.unresolved_risks,
        summary=summary,
    )


def narrative_report_path(path: Path) -> Path:
    '''Builds the output path of the Markdown narrative report.'''
    return cleaning_cache_dir(path) / f"{path.stem}.narrative_report.md"


def save_narrative_report(path: Path, report: NarrativeReport) -> Path:
    '''Saves the narrative report as a Markdown file and returns its output path.'''
    # Compute the narrative report path and make sure the folder exists.
    output_path = narrative_report_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Render the report title, summary, sections, and recommendations as Markdown lines.
    lines: list[str] = [f"# {report.title}", "", report.executive_summary, ""]
    for section in report.sections:
        lines += [f"## {section.heading}", "", section.body, ""]
    if report.recommendations:
        lines += ["## Raccomandazioni", ""]
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    # Write the final Markdown document to disk.
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _format_pct(value: float) -> str:
    '''Formats a numeric percentage with one decimal place.'''
    return f"{value:.1f}%"


def _display_target(target: object) -> str:
    '''Converts an action target into a short readable label for tables and summaries.'''
    # If the target is already a plain value, just stringify it directly.
    if not isinstance(target, dict):
        return str(target)
    # Extract the most common target fields used by remediation actions.
    column_name = target.get("column_name")
    new_name = target.get("new_name")
    target_dtype = target.get("target_dtype")
    expected_pattern = target.get("expected_pattern")
    if column_name and new_name:
        return f"{column_name} -> {new_name}"
    if column_name and target_dtype:
        return f"{column_name} -> {target_dtype}"
    if column_name and expected_pattern:
        return f"{column_name} ({expected_pattern})"
    if column_name:
        return str(column_name)
    return str(target)


def _polish_narrative_body(body: str) -> str:
    '''Applies small cleanup rules to generated narrative text.'''
    # Remove backticks around ordinary inline text.
    body = re.sub(r"`([^`\n]+)`", r"\1", body)
    # Trim overly precise percentages to cleaner one- or two-decimal forms.
    body = re.sub(r"(\d+\.\d{2})\d+%", r"\1%", body)
    body = re.sub(r"(\d+\.\d)0+%", r"\1%", body)
    return body


def _build_frontmatter_brief(final_report: FinalPipelineReport) -> str:
    '''Builds a compact briefing for the narrative frontmatter agent.'''
    return "\n".join(
        [
            f"DATASET: {final_report.dataset_name}",
            f"TOTAL_ROWS_CLEANED: {final_report.total_rows_cleaned}",
            f"VALIDATION_SUMMARY: {final_report.validation_summary}",
            f"APPLIED_ACTIONS: {len(final_report.applied_actions)}",
            f"DEFERRED_ACTIONS: {len(final_report.proposed_not_applied_actions)}",
            f"FAILED_ACTIONS: {len(final_report.failed_actions)}",
            f"NOT_NEEDED_ACTIONS: {len(final_report.not_needed_actions)}",
            f"GENERATED_CLEANERS: {len(final_report.generated_cleaners)}",
            f"ANOMALY_FINDINGS: {len(final_report.anomaly_findings)}",
            f"CROSS_COLUMN_FINDINGS: {len(final_report.cross_column_findings)}",
            f"DUPLICATE_GROUPS: {len(final_report.duplicate_groups)}",
            f"VERIFICATION_SUMMARY: {final_report.verification_summary}",
            f"UNRESOLVED_RISKS: {final_report.unresolved_risks or ['None']}",
            f"OVERALL_SUMMARY: {final_report.summary}",
        ]
    )


def _build_narrative_section_specs(final_report: FinalPipelineReport) -> list[tuple[str, str]]:
    '''Builds one detailed instruction block per narrative section for chunked generation.'''
    # Prepare compact lines for rename and dtype-cast actions.
    rename_lines = [
        f"- rename: {action.target.get('column_name')} -> {action.target.get('new_name')} | reason={action.reason}"
        for action in final_report.applied_actions
        if action.action_type == "rename_column"
    ] or ["- No applied column renames."]
    cast_lines = [
        f"- cast: {action.target.get('column_name')} -> {action.target.get('target_dtype')} | reason={action.reason}"
        for action in final_report.applied_actions
        if action.action_type == "cast_dtype"
    ] or ["- No applied dtype casts."]
    completeness_lines = [
        f"- {detail.column_name}: missing_like_count={detail.missing_like_count}, completeness={_format_pct(detail.completeness_pct)}, examples={detail.missing_like_examples[:5]}"
        for detail in final_report.completeness_details
        if detail.missing_like_count > 0 or detail.missing_like_examples
    ] or ["- No missing-like placeholder findings requiring detail."]
    # Build lookup maps so section prompts can connect cleaner output with verification results.
    cleaner_action_map = {
        action.target.get("column_name"): action
        for action in final_report.applied_actions
        if action.action_type == "generate_cleaner" and isinstance(action.target, dict)
    }
    verification_map = {}
    for diff in final_report.verification_diffs:
        verification_map[diff.column_name] = diff
        if diff.renamed_to:
            verification_map[diff.renamed_to] = diff

    # Build one summary line per generated cleaner using real example transformations only.
    format_lines = []
    for artifact in final_report.generated_cleaners:
        action = cleaner_action_map.get(artifact.column_name)
        diff = verification_map.get(artifact.column_name)
        changed_examples = [
            f"{t.original_value!r} -> {t.cleaned_value!r}"
            for t in artifact.example_transformations
            if t.cleaned_value is not None and t.original_value != t.cleaned_value
        ][:4]
        unchanged_count = sum(
            1
            for t in artifact.example_transformations
            if t.cleaned_value is not None and t.original_value == t.cleaned_value
        )
        format_lines.append(
            f"- column={artifact.column_name}; expected_pattern={action.target.get('expected_pattern') if action else 'not available'}; "
            f"inconsistent_rows_before={diff.before_inconsistent_rows if diff else (action.preview_stats.get('inconsistent_rows') if action else 'not available')}; "
            f"verification_status={diff.status if diff else 'not available'}; renamed_to={diff.renamed_to if diff and diff.renamed_to else 'same name'}; "
            f"changed_examples={changed_examples or ['none available']}; unchanged_examples_not_for_bad_values={unchanged_count}; "
            f"summary={artifact.summary}"
        )
    if not format_lines:
        format_lines = ["- No generated cleaners were recorded."]
    # Build compact prompt lines for the other narrative sections.
    anomaly_lines = [
        f"- {finding.column_name}: type={finding.anomaly_type}, severity={finding.severity}, affected_rows={finding.affected_rows}, examples={finding.example_values[:5]}, evidence={finding.evidence}"
        for finding in final_report.anomaly_findings
    ] or ["- No anomaly findings were recorded."]
    cross_column_lines = [
        f"- columns={finding.columns}, check_type={finding.check_type}, severity={finding.severity}, affected_rows={finding.affected_rows}, similarity_pct={finding.similarity_pct}, evidence={finding.evidence}"
        for finding in final_report.cross_column_findings
    ] or ["- No cross-column findings were recorded."]
    duplicate_lines = [
        f"- duplicate_type={group.duplicate_type}, row_indices={group.row_indices[:8]}, key_columns={group.key_columns}, evidence={group.evidence}"
        for group in final_report.duplicate_groups
    ] or ["- No duplicate row groups were recorded."]
    remediation_lines = [
        f"- action_type={action.action_type}, status={action.status}, target={_display_target(action.target)}, reason={action.reason}"
        for action in (
            final_report.applied_actions
            + final_report.proposed_not_applied_actions
            + final_report.failed_actions
            + final_report.not_needed_actions
        )[:40]
    ] or ["- No remediation actions were recorded."]
    verification_lines = [
        f"- column={diff.column_name}, renamed_to={diff.renamed_to}, status={diff.status}, before={diff.before_inconsistent_rows}, after={diff.after_inconsistent_rows}, remaining={diff.remaining_examples[:5]}"
        for diff in final_report.verification_diffs
    ] or [f"- Verification summary only: {final_report.verification_summary}"]
    risk_lines = [
        f"- unresolved_risk: {risk}" for risk in final_report.unresolved_risks
    ] + [
        f"- manual_review: {action.action_id} | type={action.action_type} | reason={action.reason} | target={action.target}"
        for action in final_report.manual_review_queue[:30]
    ]
    if not risk_lines:
        risk_lines = ["- No unresolved risks or manual-review items were recorded."]

    # Return the ordered list of report sections and their section-specific instructions.
    return [
        (
            "Dataset Overview",
            "\n".join(
                [
                    "FORMAT REQUIREMENTS:",
                    "- Write a strong prose overview section.",
                    "- Do not use a markdown table in this section.",
                    "- Mention dataset size, overall quality posture, applied/deferred/failed counts, and verification summary.",
                    "- Do not use backticks for ordinary column names, values, row counts, or percentages.",
                    "",
                    f"Dataset: {final_report.dataset_name}",
                    f"Total cleaned rows: {final_report.total_rows_cleaned}",
                    f"Validation summary counts: {final_report.validation_summary}",
                    f"Overall summary: {final_report.summary}",
                    f"Verification summary: {final_report.verification_summary}",
                    f"Applied={len(final_report.applied_actions)}, Deferred={len(final_report.proposed_not_applied_actions)}, Failed={len(final_report.failed_actions)}, Not Needed={len(final_report.not_needed_actions)}",
                ]
            ),
        ),
        (
            "Schema Validation",
            "\n".join(
                [
                    "FORMAT REQUIREMENTS:",
                    "- Use two markdown tables in this section.",
                    "- First table heading: Column Renames.",
                    "- Column Renames table columns: Original Name | New Name | Reason.",
                    "- Second table heading: Type Casts.",
                    "- Type Casts table columns: Column | Assigned Type | Non-Null Count.",
                    "- After the tables, add short prose only if needed.",
                    "- Do not use backticks for ordinary column names or values.",
                    "",
                    f"Dataset: {final_report.dataset_name}",
                    "Applied rename and cast actions:",
                    *rename_lines,
                    *cast_lines,
                    f"Cleaned non-null counts: {final_report.non_null_counts_cleaned}",
                ]
            ),
        ),
        (
            "Completeness Analysis",
            "\n".join(
                [
                    "FORMAT REQUIREMENTS:",
                    "- Use a markdown table for placeholder and missing-like findings.",
                    "- Table columns: Column | Missing-Like Count | Completeness % | Example Tokens.",
                    "- Add short prose after the table only if needed.",
                    "- Round percentages to one decimal place.",
                    "- Do not use backticks for ordinary labels or values.",
                    "",
                    "Completeness findings sourced from completeness_details:",
                    *completeness_lines,
                ]
            ),
        ),
        (
            "Format Consistency",
            "\n".join(
                [
                    "FORMAT REQUIREMENTS:",
                    "- Start with one short overview sentence.",
                    "- Then give one subsection per cleaned column using the exact heading format: ### column_name",
                    "- Under each column, use flat markdown bullets for Expected Pattern, Inconsistent Rows, Examples of bad values, Transformation applied, Clean example when available, and Outcome.",
                    "- Do not collapse multiple columns into one paragraph.",
                    "- Treat changed_examples as the only valid source for bad-value and clean-example lines.",
                    "- Never present unchanged preserved values as bad examples.",
                    "- If changed_examples is empty, omit the Clean example bullet instead of inventing one.",
                    "- Do not use backticks for ordinary column names or example values.",
                    "",
                    f"Generated cleaners count: {len(final_report.generated_cleaners)}",
                    f"Verification summary: {final_report.verification_summary}",
                    *format_lines,
                ]
            ),
        ),
        (
            "Anomaly Detection",
            "\n".join(
                [
                    "FORMAT REQUIREMENTS:",
                    "- Separate numeric outliers and rare categories clearly.",
                    "- Use bullet lists or short subsections; a table is optional.",
                    "- If there are no findings, keep the section concise and say so once.",
                    "- Do not repeat the same 'no anomalies' statement in multiple paragraphs.",
                    "",
                    *anomaly_lines,
                ]
            ),
        ),
        (
            "Cross-Column Checks",
            "\n".join(
                [
                    "FORMAT REQUIREMENTS:",
                    "- Use short subsections or bullet lists grouped by check type.",
                    "- Do not invent column pairs beyond the provided lines.",
                    "",
                    *cross_column_lines,
                ]
            ),
        ),
        (
            "Row Duplicate Analysis",
            "\n".join(
                [
                    "FORMAT REQUIREMENTS:",
                    "- Use prose plus a compact bullet list of representative groups.",
                    "- Mention exact vs near-duplicate framing only if present in the provided lines.",
                    "- Do not use backticks for ordinary labels or row-index examples.",
                    "",
                    *duplicate_lines,
                ]
            ),
        ),
        (
            "Remediation Action Summary",
            "\n".join(
                [
                    "FORMAT REQUIREMENTS:",
                    "- Include a markdown table with columns: Action Type | Status | Target | Reason.",
                    "- Start with one short count summary sentence.",
                    "- Use the supplied target display text directly instead of raw dict-like target objects.",
                    "- Do not use backticks for ordinary labels or values.",
                    "",
                    f"Applied={len(final_report.applied_actions)}, Deferred={len(final_report.proposed_not_applied_actions)}, Failed={len(final_report.failed_actions)}, Not Needed={len(final_report.not_needed_actions)}",
                    *remediation_lines,
                ]
            ),
        ),
        (
            "Verification Outcome",
            "\n".join(
                [
                    "FORMAT REQUIREMENTS:",
                    "- Use a markdown table with columns: Column | Renamed To | Status | Before | After.",
                    "- Add a short explanation of what the verification outcome means.",
                    "- Do not use backticks for ordinary labels, column names, or counts.",
                    "",
                    f"Verification summary: {final_report.verification_summary}",
                    *verification_lines,
                ]
            ),
        ),
        (
            "Residual Risks & Manual Review Queue",
            "\n".join(
                [
                    "FORMAT REQUIREMENTS:",
                    "- Use a bullet list of unresolved risks and manual-review items.",
                    "- Keep the section concrete and action-oriented.",
                    "- Do not use backticks for ordinary labels or values.",
                    "",
                    *risk_lines,
                ]
            ),
        ),
    ]


def _generate_narrative_report_chunked(final_report: FinalPipelineReport) -> NarrativeReport:
    '''Generates the narrative report in chunks using separate agents for frontmatter and body sections.'''
    # Import the agents lazily because they are only needed when generating the narrative.
    from core.agents import narrative_frontmatter_agent, narrative_section_agent
    from tools.common_tools import attach_text_document, run_agent_with_backoff

    # Generate the title, executive summary, and recommendations first.
    frontmatter = run_agent_with_backoff(
        narrative_frontmatter_agent,
        [
            f"Write the front matter for dataset '{final_report.dataset_name}'.",
            attach_text_document(_build_frontmatter_brief(final_report)),
        ],
    ).output

    # Generate each narrative section independently from its own briefing block.
    sections: list[NarrativeReportSection] = []
    for heading, briefing in _build_narrative_section_specs(final_report):
        section = run_agent_with_backoff(
            narrative_section_agent,
            [
                (
                    f"Write exactly one report section with heading '{heading}' for dataset "
                    f"'{final_report.dataset_name}'."
                ),
                attach_text_document(briefing),
            ],
        ).output
        # Force the returned heading to match the expected section heading if needed.
        if section.heading != heading:
            section = section.model_copy(update={"heading": heading})
        # Lightly clean the generated body text before saving it.
        section = section.model_copy(update={"body": _polish_narrative_body(section.body)})
        sections.append(section)

    # Assemble the final narrative report from generated frontmatter and sections.
    return NarrativeReport(
        title=frontmatter.title,
        executive_summary=_polish_narrative_body(frontmatter.executive_summary),
        sections=sections,
        recommendations=[_polish_narrative_body(item) for item in frontmatter.recommendations],
    )


