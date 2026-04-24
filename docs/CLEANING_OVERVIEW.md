# Cleaning Overview

This document gives a brief explanation of each file inside `cleaning/` and how the modules connect.

## File-by-file map

- `cleaning/__init__.py`  
  Thin package facade. It re-exports the main cleaning entrypoints so the CLI and app can import them from `cleaning` directly.

- `cleaning/orchestrator.py`  
  The top-level driver for the cleaning half. It ties everything together: resolve validation results, build the remediation plan, build cleaning requests, generate cleaners, apply them, verify the result, and assemble the final reports.

- `cleaning/remediation.py`  
  Converts validation findings into a structured `RemediationPlan`. This is where the system decides what actions exist, such as renames, dtype casts, placeholder replacement, cleaner generation, duplicate-column drops, and manual-review items.

- `cleaning/request.py`  
  Builds a `ColumnCleaningRequest` for each dirty column. It combines consistency findings, format facts, and schema information into the exact prompt contract used by the cleaner generator.

- `cleaning/generation.py`  
  The cleaner synthesis engine. It runs the generator/critic retry loop, asks the model to write a Python cleaner for a column, validates it host-side, retries when needed, detects stagnation, and saves accepted cleaners plus the manifest.

- `cleaning/validation.py`  
  The host-side checker for generated cleaners. It tests whether generated code is self-contained, preserves already-valid values, fixes outliers, matches the target shape/pattern, and produces concrete `CleanerValidationIssue`s when it fails.

- `cleaning/runtime.py`  
  Executes accepted cleaner code. It loads the generated Python function into a restricted namespace and applies it row by row to a pandas Series, while collecting execution stats and sample updates.

- `cleaning/application.py`  
  Actually mutates the dataset. It loads generated cleaners and the remediation plan, applies cleaners, replaces placeholders with nulls, drops exact duplicate columns, renames columns, casts dtypes, and writes the cleaned CSV and cleaning report.

- `cleaning/verification.py`  
  Checks whether cleaning really improved the dataset. It re-runs consistency validation on the cleaned CSV and compares before vs. after findings to label each original issue as `resolved`, `improved`, `unchanged`, `regressed`, or `new`.

- `cleaning/reporting.py`  
  Builds the final high-level report object and generates the narrative markdown report. It merges validation, remediation, cleaning, and verification outputs into one `FinalPipelineReport`, then asks the narrative agent to write the human-readable report.

- `cleaning/paths.py`  
  Centralizes all cleaning-side filesystem paths. It defines where cleaner files, manifests, cleaned CSVs, and final reports live under `.cleaning_cache`.

## Compact mental model

```text
orchestrator
  -> remediation
  -> request
  -> generation
  -> validation/runtime
  -> application
  -> verification
  -> reporting
```

## What each module mainly produces

| Module | Main output |
|---|---|
| `orchestrator.py` | `CleaningPipelineResult` |
| `remediation.py` | `RemediationPlan` |
| `request.py` | `ColumnCleaningRequest` |
| `generation.py` | generated cleaner files + `GeneratedCleanerArtifact` |
| `validation.py` | `CleanerValidationIssue` list or accepted cleaner |
| `runtime.py` | `ColumnCleanerExecutionReport` |
| `application.py` | `CleaningReport` + cleaned CSV |
| `verification.py` | `ConsistencyVerificationReport` |
| `reporting.py` | `FinalPipelineReport` + `NarrativeReport` |
| `paths.py` | filesystem path helpers |
