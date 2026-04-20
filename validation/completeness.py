"""Completeness analysis stage.

Single LLM call: ``completeness_analysis_agent`` reads the attached
``CompletenessProfile`` document and returns a full
``CompletenessAnalysisReport`` (missing-like %, placeholder tokens
detected per column, sparse-column flags).
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
    if reuse_cache:
        return load_completeness(path)
    df = load_dataset_frame(path)
    profile = build_completeness_profile(df, path.stem)
    prompt = [
        (
            f"Analyze the attached completeness profile for dataset {path.stem}. "
            "Use Python in code execution to inspect the profile document. "
            "This is step 2 of the orchestration only: Completeness Analysis. "
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
