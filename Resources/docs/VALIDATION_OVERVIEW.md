# Validation Overview

This document gives a brief explanation of each file inside `validation/` and how the modules connect.

## File-by-file map

- `validation/__init__.py`  
  Package facade for the validation half. It re-exports the main validation entrypoints so the CLI and app can import them from `validation` directly.

- `validation/_summary.py`  
  Shared helper for summary-style validation stages. It centralizes the pattern "build findings locally, then ask an agent to write a short summary over those findings."

- `validation/bundle.py`  
  The top-level validation driver. It runs all validation stages in order, packages them into one `OrchestrationStepResult`, and saves the combined validation bundle to cache.

- `validation/schema.py`  
  Handles schema understanding. It runs dtype inference, builds per-column schema entries, checks naming rules, detects duplicate semantic column names, creates schema issues, and produces the `SchemaHandoff` used by later stages.

- `validation/completeness.py`  
  Handles missing-data analysis. It builds a completeness profile from the dataframe, then asks the completeness agent to turn that profile into a structured completeness report.

- `validation/consistency.py`  
  Handles per-column format consistency checks. It profiles each column’s dominant format and outlier values, uses a schema-guided fast path when possible, falls back to the format agent when needed, and produces the `ConsistencyValidationReport` that later drives cleaner generation.

- `validation/anomaly.py`  
  Handles anomaly detection. It uses local heuristics to find numeric outliers and rare categories, suppresses duplicate-semantic noise where appropriate, and asks an agent only to summarize the findings.

- `validation/cross_column.py`  
  Handles cross-column checks. It finds duplicate-like columns, semantic conflicts, year-month-period mismatches, and date-order violations, then wraps them into a cross-column validation report.

- `validation/duplicates.py`  
  Handles row-level duplicate analysis. It detects exact duplicate rows and near-duplicate groups, then packages them into a `DuplicateDetectionReport`.

## Compact mental model

```text
schema
  -> completeness
  -> consistency
  -> anomaly
  -> cross-column
  -> duplicates
  -> bundle
```

## Downstream importance

- `schema.py` informs naming, casts, and consistency fast-path logic.
- `completeness.py` informs placeholder-to-null remediation.
- `consistency.py` is the main bridge into `cleaning/`.
- `anomaly.py`, `cross_column.py`, and `duplicates.py` mainly feed remediation planning and final reporting.
- `bundle.py` hands the whole validation half to the cleaning orchestrator.

## What each module mainly produces

| Module | Main output |
|---|---|
| `schema.py` | `SchemaHandoff` |
| `completeness.py` | `CompletenessAnalysisReport` |
| `consistency.py` | `ConsistencyValidationReport` |
| `anomaly.py` | `AnomalyDetectionReport` |
| `cross_column.py` | `CrossColumnValidationReport` |
| `duplicates.py` | `DuplicateDetectionReport` |
| `bundle.py` | `OrchestrationStepResult` |
