"""completeness.py (validation pipeline): per-column completeness analysis.

This module exposes one public function, run_completeness_analysis, which builds a
statistical completeness profile for the dataset and passes it to the agent. The agent
identifies missing values, placeholder tokens, and sparse columns, returning a structured
CompletenessAnalysisReport that is cached for downstream use.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agents import completeness_analysis_agent
from cache import load_completeness, save_completeness
from models import CompletenessAnalysisReport
from tools import (
    attach_profile_text,
    build_completeness_profile,
    load_dataset_frame,
    run_agent_with_backoff,
)


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
    report = result.output
    save_completeness(path, report)
    return report
