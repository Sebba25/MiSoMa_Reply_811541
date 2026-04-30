"""completeness.py (validation pipeline): per-column completeness analysis.

This module exposes one public function, run_completeness_analysis, which builds a
statistical completeness profile for the dataset and passes it to the agent. The agent
identifies missing values, placeholder tokens, and sparse columns, returning a structured
CompletenessAnalysisReport that is cached for downstream use.
"""

from __future__ import annotations
import sys
from pathlib import Path
from src.core.agents import completeness_analysis_agent
from src.core.cache import load_completeness, save_completeness
from src.core.models import CompletenessAnalysisReport
from src.tools import attach_profile_text, build_completeness_profile, load_dataset_frame, run_agent_with_backoff



def _backfill_missing_like_examples(report: CompletenessAnalysisReport, profile) -> CompletenessAnalysisReport:
    """Fill missing per-column placeholder examples from the local completeness profile.

    The local profile is computed deterministically from the dataframe before the
    agent runs, so it is the most reliable source for concrete placeholder tokens
    such as ``"N.D."`` or ``"unknown"``. The agent still owns the narrative
    summary and recommendations, but downstream cleaning should not depend on the
    agent remembering to echo these examples back into every column finding.
    """
    # Build a quick lookup from column name to the concrete placeholder examples
    # already computed locally from the raw dataframe.
    placeholder_examples_by_column = {
        column.column_name: column.placeholder_examples
        for column in profile.columns
    }

    # Rebuild the per-column findings so only the missing example lists are patched.
    updated_findings = []
    for finding in report.per_column:
        # Merge agent examples with the exhaustive local profile so all known placeholder spellings are covered.
        local = placeholder_examples_by_column.get(finding.column_name, [])
        merged = list(dict.fromkeys(finding.missing_like_examples + local))
        updated_findings.append(finding.model_copy(update={"missing_like_examples": merged}))

    # Return a fresh report object with the backfilled finding
    return report.model_copy(update={"per_column": updated_findings})


def run_completeness_analysis(path: Path, reuse_cache: bool = False) -> CompletenessAnalysisReport:
    """Build a completeness profile for the dataset and return the agent's structured analysis.

    The profile is computed locally from the raw data, then handed to the agent which
    identifies missing values, placeholder tokens, and sparse columns. The result is cached.
    """
    if reuse_cache:
        return load_completeness(path)
    df = load_dataset_frame(path)
    # Build a per-column completeness profile to attach to the agent prompt
    profile = build_completeness_profile(df, path.stem)
    prompt = [
        (
            f"Analyze the attached completeness profile for dataset {path.stem}. "
            "Use Python in code execution to inspect the profile document. "
            "Use the provided metrics to summarize per-column completeness, detect missing-like and placeholder values, "
            "identify actual placeholder tokens present in the dataset, and flag sparse columns that may be candidates for removal or investigation."
        ),
        attach_profile_text(profile),
    ]
    print(f"[orchestrator][completeness] dataset='{path.stem}'", file=sys.stderr, flush=True)
    result = run_agent_with_backoff(completeness_analysis_agent, prompt)
    # Keep the agent's narrative analysis, but make sure downstream cleaning has
    # concrete per-column placeholder examples even when the model omitted them.
    report = _backfill_missing_like_examples(result.output, profile)
    save_completeness(path, report)
    return report
