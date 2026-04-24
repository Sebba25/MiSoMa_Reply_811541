"""_summary.py (validation pipeline): shared helper for the summary-agent pattern.

Anomaly, cross-column, and duplicate detection all follow the same pattern: heuristics build
the findings, then a summary agent is asked to write a human-readable summary over the
already-built report. This module centralises that pattern so each stage does not repeat it.
"""

from __future__ import annotations

from tools import attach_profile_text, run_agent_with_backoff


def summarize_validation_report(agent, prompt_text: str, report, fallback_summary: str) -> str:
    """Call the given summary agent and return its summary, falling back to a static string on failure.

    The fallback ensures the pipeline never crashes just because the summary agent failed —
    the findings themselves are already built and cached regardless of this call.
    """
    try:
        result = run_agent_with_backoff(agent, [prompt_text, attach_profile_text(report)])
        # Use the agent's summary only if it returned a non-empty string
        summary = getattr(result.output, "summary", "").strip()
        return summary or fallback_summary
    except Exception:
        # Any agent failure is silently absorbed — the fallback summary is always sufficient
        return fallback_summary
