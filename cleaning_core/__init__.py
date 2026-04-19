"""Cleaning pipeline subpackage.

Split by concern so each stage can be imported on its own:

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

