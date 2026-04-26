"""Final report assembly + narrative generation.

``build_final_report`` merges validation, remediation, cleaning and
verification artifacts into a single ``FinalPipelineReport``;
``generate_narrative_report`` hands that report to the narrative agent to
produce the Markdown summary saved alongside the cleaned CSV.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from models import (
    CleaningReport,
    ConsistencyVerificationReport,
    FinalPipelineReport,
    NarrativeReport,
    NarrativeReportSection,
    OrchestrationStepResult,
    RemediationAction,
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
        action for action in remediation_plan.actions if action.action_type == "drop_rows_candidate"
    ]
    manual_review_queue = [
        action
        for action in remediation_plan.actions
        if action.status == "proposed_not_applied"
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


def _md_cell(value: object) -> str:
    '''Normalizes one value so it is safe to insert into a Markdown table cell.'''
    if value is None:
        return "-"
    if isinstance(value, list):
        # Lists become comma-separated text.
        text = ", ".join(str(item) for item in value)
    else:
        text = str(value)
    # Replace characters that would break Markdown table formatting.
    text = text.replace("|", "/").replace("\n", " ").strip()
    return text or "-"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    '''Builds a Markdown table from headers and row values.'''
    # Start with the header row and the Markdown separator row.
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    # Add one sanitized row per record.
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _build_narrative_briefing(report: FinalPipelineReport) -> str:
    """Build a condensed text briefing from the FinalPipelineReport, organized by topic."""
    # Collect briefing lines that summarize the pipeline outputs in a structured text block.
    lines: list[str] = []

    # Start with dataset-level facts and top-level summaries.
    lines.append(f"DATASET: {report.dataset_name}")
    lines.append(f"VALIDATION SUMMARY: {report.validation_summary}")
    lines.append(f"CLEANING SUMMARY: {report.cleaning_summary}")
    if report.total_rows_cleaned:
        lines.append(f"TOTAL ROWS (cleaned dataset, authoritative): {report.total_rows_cleaned}")
    lines.append(f"VERIFICATION SUMMARY: {report.verification_summary}")
    if report.verification_diffs:
        lines.append("VERIFICATION DIFFS (original_name | new_name_if_renamed | status | before→after rows):")
        for diff in report.verification_diffs:
            new_name = diff.renamed_to or diff.column_name
            lines.append(
                f"  {diff.column_name} | {new_name} | {diff.status} | "
                f"{diff.before_inconsistent_rows}→{diff.after_inconsistent_rows} rows"
            )
        lines.append(
            "NOTE: when new_name differs from the original name, the column was renamed during cleaning. "
            "In the Verification Outcome table, report the NEW name (post-rename) so readers see the column "
            "name that actually exists in the cleaned CSV, and show the rename arrow where relevant."
        )
    lines.append(f"UNRESOLVED RISKS: {report.unresolved_risks or 'None'}")
    lines.append(f"OVERALL: {report.summary}")
    lines.append("")

    # Per-column non-null counts from the CLEANED CSV. The agent uses these to
    # compute the '% Non-Null' column in the Type Casts table. Any percentage
    # not derivable from (count / TOTAL ROWS * 100) is a fabrication.
    if report.non_null_counts_cleaned and report.total_rows_cleaned:
        lines.append("=" * 60)
        lines.append(
            f"NON-NULL COUNTS — cleaned CSV (TOTAL ROWS = {report.total_rows_cleaned}) — GROUND TRUTH"
        )
        lines.append("=" * 60)
        lines.append(
            "Use (non_null_count / TOTAL ROWS * 100), rounded to one decimal, for every '% Non-Null' "
            "value in the Type Casts table. Do not estimate or round differently."
        )
        for column_name, non_null in report.non_null_counts_cleaned.items():
            pct = round(non_null / report.total_rows_cleaned * 100, 1) if report.total_rows_cleaned else 0.0
            lines.append(f"  {column_name}: {non_null}/{report.total_rows_cleaned} = {pct}%")
        lines.append("")

    # Completeness per-column truth from the validation cache.
    if report.completeness_details:
        lines.append("=" * 60)
        lines.append("COMPLETENESS DETAILS (GROUND TRUTH) — quote verbatim for Completeness Analysis section")
        lines.append("=" * 60)
        for finding in report.completeness_details:
            if finding.missing_like_count == 0 and not finding.missing_like_examples:
                continue
            examples = ", ".join(repr(v) for v in finding.missing_like_examples[:6]) or "(none listed)"
            lines.append(
                f"  {finding.column_name}: missing_like_count={finding.missing_like_count}, "
                f"completeness={finding.completeness_pct}%, tokens=[{examples}]"
            )
        lines.append("")

    # Anomaly findings ground truth (numeric outliers + rare categories).
    if report.anomaly_findings:
        lines.append("=" * 60)
        lines.append("ANOMALY FINDINGS (GROUND TRUTH) — quote verbatim for Anomaly Detection section")
        lines.append("=" * 60)
        lines.append(
            "Do NOT invent columns, row counts, or example values. If a column is not listed below, "
            "it has no anomaly finding."
        )
        for finding in report.anomaly_findings:
            examples = ", ".join(repr(v) for v in finding.example_values[:6]) or "(no examples)"
            lines.append(
                f"  [{finding.anomaly_type}] '{finding.column_name}' — severity={finding.severity}, "
                f"affected_rows={finding.affected_rows}"
            )
            lines.append(f"      examples: {examples}")
            lines.append(f"      evidence: {finding.evidence}")
            lines.append(f"      suggested_action: {finding.suggested_action}")
        lines.append("")

    # Cross-column findings ground truth (exact/near-duplicate columns, semantic conflicts,
    # date/period checks). Includes similarity_pct when present.
    if report.cross_column_findings:
        lines.append("=" * 60)
        lines.append("CROSS-COLUMN FINDINGS (GROUND TRUTH) — quote verbatim for Cross-Column Checks section")
        lines.append("=" * 60)
        lines.append(
            "Every similarity %, mismatch count, and column pair MUST come from this block. "
            "Do not compute similarity from anywhere else."
        )
        for finding in report.cross_column_findings:
            sim = f", similarity_pct={finding.similarity_pct}" if finding.similarity_pct is not None else ""
            columns = " & ".join(finding.columns)
            lines.append(
                f"  [{finding.check_type}] {columns} — severity={finding.severity}, "
                f"affected_rows={finding.affected_rows}{sim}"
            )
            if finding.example_row_indices:
                sample = finding.example_row_indices[:5]
                lines.append(f"      example_row_indices: {sample}")
            lines.append(f"      evidence: {finding.evidence}")
            lines.append(f"      suggested_action: {finding.suggested_action}")
        lines.append("")

    # Row-duplicate groups ground truth.
    if report.duplicate_groups:
        lines.append("=" * 60)
        lines.append("DUPLICATE ROW GROUPS (GROUND TRUTH) — quote verbatim for Row Duplicate Analysis section")
        lines.append("=" * 60)
        total_dup_rows = sum(len(g.row_indices) for g in report.duplicate_groups)
        lines.append(
            f"Total duplicate groups: {len(report.duplicate_groups)}, "
            f"total affected rows: {total_dup_rows}"
        )
        for idx, group in enumerate(report.duplicate_groups[:8], start=1):
            sample = group.row_indices[:6]
            lines.append(
                f"  group {idx} [{group.duplicate_type}] — {len(group.row_indices)} rows, "
                f"key_columns={group.key_columns}, sample_indices={sample}"
            )
            lines.append(f"      evidence: {group.evidence}")
        if len(report.duplicate_groups) > 8:
            lines.append(f"  ... and {len(report.duplicate_groups) - 8} more groups with similar pattern.")
        lines.append("")

    # Real before→after transformations from each generated cleaner's sandbox run.
    # The narrative agent MUST quote these verbatim; fabricating outputs is forbidden.
    if report.generated_cleaners:
        lines.append("=" * 60)
        lines.append("CLEANER EXAMPLE TRANSFORMATIONS (GROUND TRUTH — QUOTE VERBATIM)")
        lines.append("=" * 60)
        lines.append(
            "For every 'Clean example: X → Y' line in the Format Consistency section, "
            "copy an entry BELOW verbatim. Do NOT invent outputs, drop characters (e.g. '.000' suffix), "
            "or guess cleaned values. If no transformation exists for a column, omit the Clean example line."
        )
        for artifact in report.generated_cleaners:
            if not artifact.example_transformations:
                continue
            lines.append(f"\n--- column: {artifact.column_name} ---")
            transformed = [
                t for t in artifact.example_transformations
                if t.cleaned_value is not None and t.original_value != t.cleaned_value
            ]
            preserved = [
                t for t in artifact.example_transformations
                if t.cleaned_value is not None and t.original_value == t.cleaned_value
            ]
            for t in transformed[:6]:
                lines.append(f"  {t.original_value!r} -> {t.cleaned_value!r}")
            if preserved:
                lines.append(
                    f"  (plus {len(preserved)} already-valid values preserved unchanged; do not list them as 'bad examples')"
                )
        lines.append("")

    # Group applied actions by type
    applied_by_type: dict[str, list[RemediationAction]] = {}
    for action in report.applied_actions:
        applied_by_type.setdefault(action.action_type, []).append(action)

    # Add a grouped summary of all applied actions.
    lines.append("=" * 60)
    lines.append("APPLIED ACTIONS")
    lines.append("=" * 60)

    for action_type, actions in applied_by_type.items():
        lines.append(f"\n--- {action_type} ({len(actions)} actions) ---")
        for action in actions:
            lines.append(f"  ID: {action.action_id}")
            lines.append(f"  Target: {action.target}")
            lines.append(f"  Reason: {action.reason}")
            lines.append(f"  Preview stats: {action.preview_stats}")
            lines.append("")

    # Not-needed actions — genuinely redundant / already-satisfied actions.
    # DO NOT confuse this bucket with DEFERRED (proposed_not_applied) — they are disjoint.
    if report.not_needed_actions:
        lines.append("=" * 60)
        lines.append(f"NOT NEEDED ACTIONS ({len(report.not_needed_actions)}) — redundant or already satisfied")
        lines.append("=" * 60)
        for action in report.not_needed_actions:
            lines.append(f"  {action.action_id} — {action.action_type} — {action.reason}")
        lines.append("")

    # Failed actions
    if report.failed_actions:
        lines.append("=" * 60)
        lines.append(f"FAILED ACTIONS ({len(report.failed_actions)})")
        lines.append("=" * 60)
        for action in report.failed_actions:
            lines.append(f"  {action.action_id} — {action.reason}")
        lines.append("")

    # Proposed-not-applied: group by type and condense duplicate rows
    proposed_by_type: dict[str, list[RemediationAction]] = {}
    for action in report.proposed_not_applied_actions:
        proposed_by_type.setdefault(action.action_type, []).append(action)

    # Add deferred and manual-review actions separately from successful ones.
    lines.append("=" * 60)
    lines.append(
        f"DEFERRED / MANUAL REVIEW ({len(report.proposed_not_applied_actions)} actions) — "
        "proposed but not auto-applied; requires human judgement. "
        "This is distinct from the NOT NEEDED bucket above — do NOT merge the two in the summary."
    )
    lines.append("=" * 60)

    for action_type, actions in proposed_by_type.items():
        lines.append(f"\n--- {action_type} ({len(actions)} actions) ---")
        if action_type == "drop_rows_candidate":
            # Condense: just list group sizes and sample row indices
            total_rows = sum(len(a.target.get("row_indices", [])) for a in actions)
            lines.append(f"  Total candidate rows across {len(actions)} groups: {total_rows}")
            lines.append(f"  Key columns: {actions[0].target.get('key_columns', [])}")
            sample_groups = actions[:3]
            for action in sample_groups:
                indices = action.target.get("row_indices", [])
                lines.append(f"  Example group: rows {indices} ({action.reason})")
            if len(actions) > 3:
                lines.append(f"  ... and {len(actions) - 3} more groups with similar pattern.")
        else:
            for action in actions:
                lines.append(f"  ID: {action.action_id}")
                lines.append(f"  Target: {action.target}")
                lines.append(f"  Reason: {action.reason}")
                if action.preview_stats:
                    lines.append(f"  Preview stats: {action.preview_stats}")
                lines.append("")

    # Manual review queue (non-duplicate items only, to avoid repeating)
    manual_non_dup = [a for a in report.manual_review_queue if a.action_type != "drop_rows_candidate"]
    if manual_non_dup:
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"MANUAL REVIEW QUEUE (non-duplicate items: {len(manual_non_dup)})")
        lines.append("=" * 60)
        for action in manual_non_dup:
            lines.append(f"  {action.action_id}: {action.reason}")
            lines.append(f"    Target: {action.target}")
            if action.preview_stats:
                lines.append(f"    Stats: {action.preview_stats}")

    # Join all collected briefing lines into one long prompt-friendly string.
    return "\n".join(lines)


def _fallback_narrative_report(final_report: FinalPipelineReport) -> NarrativeReport:
    '''Builds a deterministic narrative report when model-based generation is unavailable or fails.'''
    # Prepare schema-related rows for the fallback schema section tables.
    renamed_columns = [
        [action.target.get("column_name", ""), action.target.get("new_name", ""), action.reason]
        for action in final_report.applied_actions
        if action.action_type == "rename_column"
    ]
    dtype_casts = [
        [
            action.target.get("column_name", ""),
            action.target.get("target_dtype", ""),
            final_report.non_null_counts_cleaned.get(action.target.get("column_name", ""), 0),
        ]
        for action in final_report.applied_actions
        if action.action_type == "cast_dtype"
    ]
    placeholder_columns = [
        [
            detail.column_name,
            detail.missing_like_count,
            _format_pct(detail.completeness_pct),
            ", ".join(repr(token) for token in detail.missing_like_examples[:5]) or "-",
        ]
        for detail in final_report.completeness_details
        if detail.missing_like_count > 0
    ]

    # Build quick lookup maps so cleaner summaries can merge action and verification information.
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

    # Build one compact block per generated cleaner for the fallback format-consistency section.
    format_blocks: list[str] = []
    for artifact in final_report.generated_cleaners:
        action = cleaner_action_map.get(artifact.column_name)
        diff = verification_map.get(artifact.column_name)
        changed_examples = [
            f"- {t.original_value!r} -> {t.cleaned_value!r}"
            for t in artifact.example_transformations
            if t.cleaned_value is not None and t.original_value != t.cleaned_value
        ][:4]
        block = [
            f"### {artifact.column_name}",
            f"- Expected Pattern: {action.target.get('expected_pattern') if action else 'Not available'}",
            f"- Inconsistent Rows: {diff.before_inconsistent_rows if diff else (action.preview_stats.get('inconsistent_rows') if action else 'Not available')}",
            f"- Transformation applied: {artifact.summary}",
            f"- Outcome: {diff.status if diff else 'Not available'}",
        ]
        if changed_examples:
            block.append("- Clean examples:")
            block.extend(changed_examples)
        format_blocks.append("\n".join(block))

    # Build deterministic text for the anomaly, cross-column, and duplicate sections.
    anomaly_body = (
        "No anomaly findings were recorded for this dataset. The anomaly stage therefore contributes no "
        "numeric outlier or rare-category items to the remediation queue."
        if not final_report.anomaly_findings
        else "\n".join(
            [
                f"- {finding.column_name}: {finding.anomaly_type}, severity={finding.severity}, "
                f"affected_rows={finding.affected_rows}, examples={finding.example_values[:5]}, evidence={finding.evidence}"
                for finding in final_report.anomaly_findings[:12]
            ]
        )
    )

    cross_column_body = "\n".join(
        [
            f"- {', '.join(finding.columns)}: {finding.check_type}, severity={finding.severity}, "
            f"affected_rows={finding.affected_rows}, similarity_pct={finding.similarity_pct}, evidence={finding.evidence}"
            for finding in final_report.cross_column_findings[:12]
        ]
    ) or "No cross-column findings were recorded."

    duplicate_body = "\n".join(
        [
            f"- {group.duplicate_type}: rows={group.row_indices[:8]}, key_columns={group.key_columns}, evidence={group.evidence}"
            for group in final_report.duplicate_groups[:12]
        ]
    ) or "No duplicate row groups were recorded."

    # Build rows for the remediation and verification summary tables.
    remediation_rows = [
        [action.action_type, action.status, _display_target(action.target), action.reason]
        for action in (
            final_report.applied_actions
            + final_report.proposed_not_applied_actions
            + final_report.failed_actions
            + final_report.not_needed_actions
        )[:20]
    ]
    verification_rows = [
        [
            diff.column_name,
            diff.renamed_to or "-",
            diff.status,
            diff.before_inconsistent_rows,
            diff.after_inconsistent_rows,
        ]
        for diff in final_report.verification_diffs
    ]
    # Build bullet lines for unresolved risks and manual-review items.
    risk_lines = [f"- unresolved risk: {risk}" for risk in final_report.unresolved_risks]
    for action in final_report.manual_review_queue[:20]:
        if action.action_type == "drop_rows_candidate" and isinstance(action.target, dict):
            risk_lines.append(
                f"- manual review: duplicate-row candidate at rows {action.target.get('row_indices', [])} — {action.reason}"
            )
        else:
            risk_lines.append(
                f"- manual review: {action.action_type} on {_display_target(action.target)} — {action.reason}"
            )
    if not risk_lines:
        risk_lines = ["- No unresolved risks or manual-review items were recorded."]

    # Compose reusable body parts for the schema and completeness sections.
    schema_body_parts = [
        "The schema stage standardized naming and dtype assignments so the cleaned dataset follows one consistent contract."
    ]
    if renamed_columns:
        schema_body_parts.extend(
            [
                "### Column Renames",
                _markdown_table(["Original Name", "New Name", "Reason"], renamed_columns[:12]),
            ]
        )
    if dtype_casts:
        schema_body_parts.extend(
            [
                "### Type Casts",
                _markdown_table(["Column", "Assigned Type", "Non-Null Count"], dtype_casts[:12]),
            ]
        )
    completeness_body_parts = [
        "The completeness review highlights missing-like tokens and placeholder-heavy columns that were normalized into nulls where safe."
    ]
    if placeholder_columns:
        completeness_body_parts.append(
            _markdown_table(
                ["Column", "Missing-Like Count", "Completeness %", "Example Tokens"],
                placeholder_columns[:12],
            )
        )
    else:
        completeness_body_parts.append("No placeholder-driven completeness issues were recorded.")

    # Assemble the ordered set of fallback narrative sections.
    sections = [
        NarrativeReportSection(
            heading="Dataset Overview",
            body=(
                f"The dataset {final_report.dataset_name} completed the validation and cleaning pipeline. "
                f"The cleaned output contains {final_report.total_rows_cleaned or 'an unknown number of'} rows. "
                f"Validation summary counts were {final_report.validation_summary}. Applied actions: {len(final_report.applied_actions)}. "
                f"Deferred or manual-review actions: {len(final_report.proposed_not_applied_actions)}. Failed actions: {len(final_report.failed_actions)}. "
                f"Verification summary: {final_report.verification_summary}. Overall summary: {final_report.summary}"
            ),
        ),
        NarrativeReportSection(
            heading="Schema Validation",
            body="\n\n".join(schema_body_parts),
        ),
        NarrativeReportSection(
            heading="Completeness Analysis",
            body="\n\n".join(completeness_body_parts),
        ),
        NarrativeReportSection(
            heading="Format Consistency",
            body="\n\n".join(
                [f"Generated cleaners: {len(final_report.generated_cleaners)}. Verification summary: {final_report.verification_summary}."]
                + (format_blocks if format_blocks else ["No generated cleaners were recorded."])
            ),
        ),
        NarrativeReportSection(
            heading="Anomaly Detection",
            body=anomaly_body,
        ),
        NarrativeReportSection(
            heading="Cross-Column Checks",
            body=cross_column_body,
        ),
        NarrativeReportSection(
            heading="Row Duplicate Analysis",
            body=duplicate_body,
        ),
        NarrativeReportSection(
            heading="Remediation Action Summary",
            body="\n\n".join(
                [
                    f"Applied actions: {len(final_report.applied_actions)}. Deferred actions: {len(final_report.proposed_not_applied_actions)}. "
                    f"Failed actions: {len(final_report.failed_actions)}. Not-needed actions: {len(final_report.not_needed_actions)}.",
                    _markdown_table(["Action Type", "Status", "Target", "Reason"], remediation_rows)
                    if remediation_rows
                    else "No remediation actions were recorded.",
                ]
            ),
        ),
        NarrativeReportSection(
            heading="Verification Outcome",
            body="\n\n".join(
                [
                    "Verification compares the original inconsistency counts with the cleaned-state counts to confirm whether each issue was resolved.",
                    _markdown_table(["Column", "Renamed To", "Status", "Before", "After"], verification_rows)
                    if verification_rows
                    else final_report.verification_summary,
                ]
            ),
        ),
        NarrativeReportSection(
            heading="Residual Risks & Manual Review Queue",
            body="\n".join(risk_lines),
        ),
    ]

    # Return the completed fallback narrative report model.
    return NarrativeReport(
        title=f"Dataset Quality Report — {final_report.dataset_name}",
        executive_summary=(
            f"The narrative fallback was generated because the language-model report did not satisfy the required JSON schema. "
            f"Dataset {final_report.dataset_name} completed the pipeline with {len(final_report.applied_actions)} applied actions, "
            f"{len(final_report.proposed_not_applied_actions)} deferred or manual-review actions, and {len(final_report.failed_actions)} failed actions. "
            f"The cleaned output contains {final_report.total_rows_cleaned} rows. Validation summary counts were {final_report.validation_summary}. "
            f"Consistency verification summary: {final_report.verification_summary}. Outstanding risks: {final_report.unresolved_risks or ['None']}."
        ),
        sections=sections,
        recommendations=[
            "Review deferred manual-review actions before publishing the cleaned dataset.",
            "Resolve duplicate-row and duplicate-semantic conflicts before downstream aggregation or reporting.",
            "Use the cleaned CSV and final JSON report as the authoritative downstream outputs.",
        ],
    )


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
    from agents import narrative_frontmatter_agent, narrative_section_agent
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


def generate_narrative_report(final_report: FinalPipelineReport) -> NarrativeReport:
    '''Generates a narrative report, falling back to a deterministic version if chunked generation fails.'''
    try:
        # Prefer the chunked model-based narrative generation path.
        return _generate_narrative_report_chunked(final_report)
    except Exception as error:
        # If model generation fails, log the warning and return a safe fallback report.
        print(
            f"[report] warning: chunked narrative generation failed, using deterministic fallback: {error}",
            file=sys.stderr,
        )
        return _fallback_narrative_report(final_report)
