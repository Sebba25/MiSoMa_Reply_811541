"""Cleaning pipeline subpackage.

Public facade that re-exports the five entry points the CLI and the
Streamlit app call. Internals are split by concern — each module can also
be imported on its own:

* ``orchestrator``  — end-to-end ``run_cleaning`` driver
* ``generation``    — generator/critic repair loop
* ``validation``    — host-side cleaner program validator (no LLM)
* ``application``   — applies cleaners, renames, and dtype casts to the CSV
* ``remediation``   — builds the remediation plan from validation artifacts
* ``reporting``     — final JSON report + narrative generator
* ``verification``  — before/after consistency diff
* ``request``       — builds ``ColumnCleaningRequest`` bundles
* ``runtime``       — loads and executes generated cleaner functions
* ``paths``         — cache/output path conventions
"""

from __future__ import annotations

from cleaning.application import run_cleaner_application
from cleaning.generation import run_cleaner_generation
from cleaning.orchestrator import run_cleaning
from cleaning.remediation import run_remediation_planning
from cleaning.verification import run_verify

__all__ = [
    "run_cleaner_application",
    "run_cleaner_generation",
    "run_cleaning",
    "run_remediation_planning",
    "run_verify",
]
