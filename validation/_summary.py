"""Shared helper for the summary-agent pattern used by anomaly, cross-column
and duplicate detection. Each stage heuristically builds findings, hands the
report to a summary agent, and falls back to a deterministic summary if the
call fails."""

from __future__ import annotations

from tools import attach_profile_text, run_agent_with_backoff


def summarize_validation_report(agent, prompt_text: str, report, fallback_summary: str) -> str:
    try:
        result = run_agent_with_backoff(agent, [prompt_text, attach_profile_text(report)])
        summary = getattr(result.output, "summary", "").strip()
        return summary or fallback_summary
    except Exception:
        return fallback_summary
