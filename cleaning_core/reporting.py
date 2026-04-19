"""Final report assembly + narrative generation.

``build_final_report`` merges validation, remediation, cleaning and
verification artifacts into a single ``FinalPipelineReport``;
``generate_narrative_report`` hands that report to the narrative agent to
produce the Markdown summary saved alongside the cleaned CSV.
"""

from __future__ import annotations

from pathlib import Path

from models import (
    CleaningReport,
    ConsistencyVerificationReport,
    FinalPipelineReport,
    NarrativeReport,
    OrchestrationStepResult,
    RemediationAction,
    RemediationPlan,
)

from .paths import cleaning_cache_dir, final_report_path


def save_final_report(path: Path, report: FinalPipelineReport) -> Path:
    report_path = final_report_path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report_path


def build_final_report(
    validation_results: OrchestrationStepResult,
    remediation_plan: RemediationPlan,
    cleaning_report: CleaningReport,
    verification_report: ConsistencyVerificationReport | None,
) -> FinalPipelineReport:
    validation_summary = {
        "schema_issues": len(validation_results.schema_validation.issues),
        "completeness_columns_with_missing": len(validation_results.completeness_analysis.columns_with_missing_values),
        "consistency_findings": len(validation_results.consistency_validation.format_consistency_findings),
        "anomaly_findings": len(validation_results.anomaly_detection.findings) if validation_results.anomaly_detection else 0,
        "cross_column_findings": len(validation_results.cross_column_validation.findings) if validation_results.cross_column_validation else 0,
        "duplicate_groups": len(validation_results.duplicate_detection.groups) if validation_results.duplicate_detection else 0,
    }

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
        if action.action_type == "manual_review" or action.action_type == "drop_rows_candidate"
    ]

    verification_summary = verification_report.summary if verification_report is not None else "Verification was not run."
    summary = (
        f"Validation found {sum(validation_summary.values())} section-level findings/signals. "
        f"Applied {len(applied_actions)} remediation actions, left {len(proposed_not_applied_actions)} proposed without auto-apply, "
        f"and recorded {len(failed_actions)} failed actions."
    )

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
        unresolved_risks=cleaning_report.unresolved_risks,
        summary=summary,
    )


def narrative_report_path(path: Path) -> Path:
    return cleaning_cache_dir(path) / f"{path.stem}.narrative_report.md"


def save_narrative_report(path: Path, report: NarrativeReport) -> Path:
    output_path = narrative_report_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [f"# {report.title}", "", report.executive_summary, ""]
    for section in report.sections:
        lines += [f"## {section.heading}", "", section.body, ""]
    if report.recommendations:
        lines += ["## Raccomandazioni", ""]
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _build_narrative_briefing(report: FinalPipelineReport) -> str:
    """Build a condensed text briefing from the FinalPipelineReport, organized by topic."""
    lines: list[str] = []

    # Extract total rows from cleaning summary context if available
    total_rows_note = ""
    if report.cleaning_summary:
        import re as _re
        m = _re.search(r"(\d[\d,]+)\s*row", report.cleaning_summary, _re.IGNORECASE)
        if m:
            total_rows_note = f"TOTAL ROWS (for computing % non-null): {m.group(1).replace(',', '')}"

    lines.append(f"DATASET: {report.dataset_name}")
    lines.append(f"VALIDATION SUMMARY: {report.validation_summary}")
    lines.append(f"CLEANING SUMMARY: {report.cleaning_summary}")
    if total_rows_note:
        lines.append(total_rows_note)
    lines.append(f"VERIFICATION SUMMARY: {report.verification_summary}")
    lines.append(f"UNRESOLVED RISKS: {report.unresolved_risks or 'None'}")
    lines.append(f"OVERALL: {report.summary}")
    lines.append("")

    # Group applied actions by type
    applied_by_type: dict[str, list[RemediationAction]] = {}
    for action in report.applied_actions:
        applied_by_type.setdefault(action.action_type, []).append(action)

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

    # Not-needed actions
    if report.not_needed_actions:
        lines.append("=" * 60)
        lines.append(f"NOT NEEDED ACTIONS ({len(report.not_needed_actions)})")
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

    lines.append("=" * 60)
    lines.append(f"PROPOSED NOT APPLIED ({len(report.proposed_not_applied_actions)} actions)")
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

    return "\n".join(lines)


def generate_narrative_report(final_report: FinalPipelineReport) -> NarrativeReport:
    from agents import narrative_report_agent
    from tools.common_tools import attach_text_document, run_agent_with_backoff

    briefing = _build_narrative_briefing(final_report)
    result = run_agent_with_backoff(
        narrative_report_agent,
        [
            f"Generate an exhaustive narrative quality report for dataset '{final_report.dataset_name}'. "
            "The attached briefing contains all findings, actions, and verification results.",
            attach_text_document(briefing),
        ],
    )
    return result.output
