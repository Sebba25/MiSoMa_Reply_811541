"""Final report assembly + narrative generation.

``build_final_report`` merges validation, remediation, cleaning and
verification artifacts into a single ``FinalPipelineReport``;
``generate_narrative_report`` hands that report to the narrative agent to
produce the Markdown summary saved alongside the cleaned CSV.
"""

from __future__ import annotations

import sys
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

from .paths import cleaned_dataset_path, cleaning_cache_dir, final_report_path


def save_final_report(path: Path, report: FinalPipelineReport) -> Path:
    report_path = final_report_path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report_path


def _compute_cleaned_non_null_counts(dataset_path: Path | None) -> tuple[int, dict[str, int]]:
    """Read the cleaned CSV and return (total_rows, {column: non_null_count}).

    Used to ground the narrative report's '% Non-Null' numbers in real data
    instead of letting the agent guess from the cleaning summary string.
    Returns (0, {}) if the cleaned file cannot be loaded.
    """
    if dataset_path is None:
        return 0, {}
    cleaned_path = cleaned_dataset_path(dataset_path)
    if not cleaned_path.exists():
        return 0, {}
    try:
        from tools import load_dataset_frame

        df = load_dataset_frame(cleaned_path)
        total = len(df)
        counts = {str(col): int(df[col].notna().sum()) for col in df.columns}
        return total, counts
    except Exception as error:
        print(f"[report] warning: could not compute non-null counts: {error}", file=sys.stderr)
        return 0, {}


def build_final_report(
    validation_results: OrchestrationStepResult,
    remediation_plan: RemediationPlan,
    cleaning_report: CleaningReport,
    verification_report: ConsistencyVerificationReport | None,
    dataset_path: Path | None = None,
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
        if action.status == "proposed_not_applied"
    ]

    verification_summary = verification_report.summary if verification_report is not None else "Verification was not run."
    summary = (
        f"Validation found {sum(validation_summary.values())} section-level findings/signals. "
        f"Applied {len(applied_actions)} remediation actions, left {len(proposed_not_applied_actions)} proposed without auto-apply, "
        f"and recorded {len(failed_actions)} failed actions."
    )

    total_rows_cleaned, non_null_counts_cleaned = _compute_cleaned_non_null_counts(dataset_path)

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
