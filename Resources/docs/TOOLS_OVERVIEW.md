# Tools Overview

This document gives a brief explanation of each file inside `tools/` and how the modules connect.

## File-by-file map

- `tools/__init__.py`  
  Facade module. It re-exports the helper functions, constants, and profiling models that the rest of the codebase imports from `tools` instead of reaching into each submodule directly.

- `tools/common_tools.py`  
  Shared low-level utilities used across the whole project. This includes CSV loading, text/profile attachment helpers for agent prompts, gzip/base64 encoding, structural value-shape helpers, numeric/date parse-rate helpers, numeric pattern matching, and the central `run_agent_with_backoff()` runtime wrapper used by all agent calls.

- `tools/schema_tools.py`  
  Utilities for schema profiling and naming logic. It builds the dataset profile used by schema validation, renders the dtype-inference text sent to the agent, and defines the canonical naming helpers like `normalized_schema_name()`, `suggest_schema_name()`, and naming-rule checks.

- `tools/completeness_tools.py`  
  Utilities for missingness analysis. It computes missing-like masks, samples placeholder examples, detects placeholder tokens, and builds the `CompletenessProfile` consumed by the completeness stage.

- `tools/format_tools.py`  
  Utilities for per-column format profiling. It computes dominant structural shapes, identifies inconsistent/outlier examples, infers semantic hints from column names, and builds `ColumnFormatFacts`, which is the core evidence object used both by validation consistency checks and cleaner generation.

- `tools/quality_tools.py`  
  Utilities for dataset-wide quality heuristics. It contains the rule-based detectors for numeric outliers, rare categories, duplicate-like columns, semantic conflicts, exact/near duplicate rows, year-month-period mismatches, and date-order violations.

## Compact mental model

```text
common_tools.py         = shared infrastructure
schema_tools.py         = schema profiling + naming
completeness_tools.py   = missingness profiling
format_tools.py         = format profiling
quality_tools.py        = anomaly / cross-column / duplicate heuristics
```

## Where each tool module feeds

- `schema_tools.py` feeds `validation/schema.py`
- `completeness_tools.py` feeds `validation/completeness.py`
- `format_tools.py` feeds `validation/consistency.py` and `cleaning/request.py`
- `quality_tools.py` feeds `validation/anomaly.py`, `validation/cross_column.py`, and `validation/duplicates.py`
- `common_tools.py` supports basically everything
