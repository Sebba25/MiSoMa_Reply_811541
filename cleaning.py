"""Public facade for the cleaning pipeline.

Re-exports the five entry points that the CLI calls. Everything else lives in
the ``cleaning_core`` subpackage and should be imported from there directly.
"""

from __future__ import annotations

from cleaning_core.application import run_cleaner_application
from cleaning_core.generation import run_cleaner_generation
from cleaning_core.orchestrator import run_cleaning
from cleaning_core.remediation import run_remediation_planning
from cleaning_core.verification import run_verify

__all__ = [
    "run_cleaner_application",
    "run_cleaner_generation",
    "run_cleaning",
    "run_remediation_planning",
    "run_verify",
]
