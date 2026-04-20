# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered dataset validation and cleaning pipeline for Italian public administration (NoiPA) CSV datasets. Uses Pydantic AI agents backed by `openai-responses:gpt-4o-mini` to infer schemas, assess completeness, detect format inconsistencies, and generate per-column Python cleaning functions.

## Commands

```bash
# Run the full validation bundle (schema + completeness + consistency)
python main.py Data/attivazioniCessazioni.csv --stage validate

# Run with cached intermediate results
python main.py Data/attivazioniCessazioni.csv --stage validate --reuse-schema --reuse-completeness --reuse-consistency

# Run the full cleaning pipeline (validate -> generate -> apply -> verify)
python main.py Data/attivazioniCessazioni.csv --stage clean

# Run a single focused stage
python main.py Data/attivazioniCessazioni.csv --stage schema
python main.py Data/attivazioniCessazioni.csv --stage completeness
python main.py Data/attivazioniCessazioni.csv --stage consistency
python main.py Data/attivazioniCessazioni.csv --stage generate --column "aggregation-time"
python main.py Data/attivazioniCessazioni.csv --stage apply
python main.py Data/attivazioniCessazioni.csv --stage verify
python main.py Data/attivazioniCessazioni.csv --stage dtype

# Verbose mode (streams agent events to stderr)
python main.py Data/attivazioniCessazioni.csv --stage clean --verbose
```

Default dataset: `Data/spesa.csv`. There is no test suite.

## Architecture

### Pipeline flow

Two main entrypoints via `--stage`:

1. **`validate`** — runs three analysis agents sequentially, each producing a cached JSON artifact:
   - dtype inference + schema validation → `SchemaHandoff`
   - completeness analysis → `CompletenessAnalysisReport`
   - per-column format consistency → `ConsistencyValidationReport`

2. **`clean`** — runs validation (or loads cached bundle), then for each inconsistent column:
   - Builds a `ColumnCleaningRequest` from schema + consistency findings
   - Generator agent writes a self-contained Python cleaning function
   - Host-side validator checks the function against dominant/outlier values
   - Critic agent diagnoses failures → generator retries (up to `--cleaner-attempts`, default 10)
   - Applies generated cleaners, schema renames, dtype casts, placeholder nulling
   - Verification agent compares pre/post consistency

### Key modules

- **`agents.py`** — All Pydantic AI `Agent` definitions with detailed `instructions` prompts. Model is set via the `MODEL` constant at module top. Logfire instrumentation configured here.
- **`models.py`** — All Pydantic models for structured agent output (schema, completeness, consistency, cleaning request/program/validation/repair).
- **`validation/`** — Validation-stage orchestration, split by stage. Public entry point `build_validation_results()` lives in `validation/bundle.py`; every `run_<stage>` helper is also importable directly from the package root:
  - `schema.py` — dtype inference + schema validation (`SchemaHandoff`)
  - `completeness.py` — completeness analysis (`CompletenessAnalysisReport`)
  - `consistency.py` — per-column format consistency (fast path + format agent)
  - `anomaly.py` — numeric outlier + rare-category detection
  - `cross_column.py` — duplicate-like columns, year/month/date-order checks
  - `duplicates.py` — exact + near duplicate record groups
  - `bundle.py` — `build_validation_results()` orchestrator
  - `_summary.py` — shared helper for the summary-agent-with-fallback pattern
- **`cleaning/`** — Cleaning pipeline subpackage. Public facade exposes `run_cleaning`, `run_cleaner_generation`, `run_cleaner_application`, `run_verify`, `run_remediation_planning` from `cleaning/__init__.py`:
  - `orchestrator.py` — end-to-end `run_cleaning()` driver
  - `generation.py` — generator/critic repair loop
  - `validation.py` — host-side cleaner program validation (no LLM)
  - `application.py` — applies cleaners, renames, dtype casts, placeholder nulling to produce cleaned CSV
  - `remediation.py` — builds the remediation plan from validation artifacts
  - `reporting.py` — final JSON report + narrative generator
  - `verification.py` — before/after consistency diff
  - `request.py` — builds `ColumnCleaningRequest` from schema + consistency findings
  - `runtime.py` — loads and executes generated cleaner `.py` files
  - `paths.py` — all cache/output path conventions
- **`tools/`** — Profiling and utility functions consumed by agents and both pipelines. `tools/__init__.py` is the public facade; import via `from tools import X`:
  - `common_tools.py` — shared helpers: dataset loading, value shape analysis, placeholder detection, agent retry with exponential backoff, attachment helpers
  - `schema_tools.py` — column profiling, dtype inference text builder, naming validation
  - `format_tools.py` — per-column format fact extraction (dominant shape, outlier examples)
  - `completeness_tools.py` — missing/placeholder detection and completeness profiling
  - `quality_tools.py` — anomaly, cross-column, and duplicate-record heuristics
- **`cli.py`** — argparse CLI, stage routing, result printing
- **`cache.py`** — load/save functions for validation cache JSON files

### Data flow conventions

- Validation caches: `Data/.validation_cache/<dataset>.<artifact>.json`
- Cleaning caches: `Data/.cleaning_cache/<dataset>/generated_cleaners/*.py` and `cleaner_manifest.json`
- Cleaned output: `Data/.cleaning_cache/<dataset>/<dataset>.cleaned.csv`
- Agent text attachments use `BinaryContent` with `text/plain` media type
- All agent calls go through `run_agent_with_backoff()` which handles rate-limit retries

### Generator/critic loop

The cleaner generation uses a generate-validate-repair cycle:
1. Generator agent produces a `ColumnCleanerProgram` (single Python function + metadata)
2. Host-side `_validate_generated_cleaner_program()` checks: dominant values unchanged, outliers transformed, output parseable as target dtype, no scope leaks
3. On failure, critic agent produces a `CleanerRepairDiagnosis` (root cause, bug location, repair brief)
4. Diagnosis is fed back to generator as repair context for the next attempt
5. Loop continues until validation passes or `--cleaner-attempts` is exhausted

### Environment

- Python 3.13+
- Dependencies: `pydantic-ai`, `pandas`, `numpy`, `python-dotenv`, `logfire`
- API keys loaded from `.env` via `python-dotenv` (OPENAI_API_KEY required)
- Logfire observability configured in `agents.py:setup_logfire()`
