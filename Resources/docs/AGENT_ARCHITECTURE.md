# Multi-Agent Code Structure

This document maps every source file that participates in the validation and
cleaning pipeline. App-level concerns (`app.py`, Streamlit UI) and caching
(`cache.py`) are intentionally excluded — they are plumbing around the core
pipeline.

All agents are defined with `pydantic-ai`, backed by
`openai-responses:gpt-4o-mini` (constant `MODEL` in `agents.py`), and their
outputs are structured via `PromptedOutput(<pydantic model>)`. Every call
goes through `run_agent_with_backoff()` which handles HTTP 429 retries,
connection-error retries, usage limits, and optional model-setting overrides
(used to escalate temperature under stagnation).

---

## High-level flow

```
                +---------------------+
                |   cli.py / main.py  |
                +----------+----------+
                           |
                           v
            +--------------+-------------+
            |        validation/         |        VALIDATION HALF
            |                            |
            | dtype → schema →           |
            | completeness → consistency |
            | → anomaly → cross-column   |
            | → duplicates               |
            +--------------+-------------+
                           |
                           v
            +--------------+-------------+
            |  cleaning/                 |        CLEANING HALF
            |  orchestrator.run_cleaning |
            |                            |
            |  remediation →             |
            |  generation (gen/critic) → |
            |  application →             |
            |  verification →            |
            |  reporting                 |
            +----------------------------+
```

The validation half is *read-only* on the CSV — it produces JSON artifacts
under `Data/.validation_cache/`. The cleaning half consumes those artifacts,
produces per-column Python cleaner files under
`Data/.cleaning_cache/<dataset>/generated_cleaners/`, and emits a cleaned
CSV plus a narrative Markdown report.

---

## Top-level modules

### `main.py` (8 lines)

Thin entrypoint: `from cli import main; main()`. Exists so `python main.py`
works from the project root.

### `cli.py` (241 lines)

Argparse CLI. Responsibilities:

- Parses `--stage` (values: `validate`, `dtype`, `schema`, `completeness`,
  `consistency`, `remediate`, `generate`, `apply`, `verify`, `clean`,
  `report`) plus `--reuse-*` flags, `--column`, `--cleaner-attempts`,
  `--verbose`.
- Loads `.env`, calls `setup_logfire()`, dispatches to the right
  `run_<stage>` helper from the `validation` or `cleaning` package.
- `print_result` dumps structured output as JSON, stripping the
  base64-gzipped cleaned CSV payload for readability.

### `cleaning/__init__.py` (facade)

Public facade for the cleaning stages. Re-exports the 5 entry points the CLI
calls:

```python
from cleaning.application    import run_cleaner_application
from cleaning.generation     import run_cleaner_generation
from cleaning.orchestrator   import run_cleaning
from cleaning.remediation    import run_remediation_planning
from cleaning.verification   import run_verify
```

Everything else lives under `cleaning/` and can be imported directly.

### `agents.py` (558 lines)

Single source of truth for every pydantic-ai `Agent` instance. Defines **10
agents**, each with its own instructions block, `PromptedOutput` schema,
retry count and `temperature=0` by default. Also sets up `logfire`
observability.

| Agent                              | Used in                          | Output model                  | Purpose                                                                                                                                      |
|------------------------------------|----------------------------------|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `dtype_inference_agent`            | `validation.run_dtype_inference`   | `DatasetDtypeInference`       | Infers pandas dtype + semantic role + detected pattern per column from a CSV sample.                                                         |
| `schema_summary_agent`             | `validation.run_schema_validation` | `SchemaSummaryOutput`         | Writes a human-readable summary over the already-built `SchemaHandoff` (does not re-derive findings).                                        |
| `completeness_analysis_agent`      | `validation.run_completeness_analysis` | `CompletenessAnalysisReport` | Reads the completeness profile, detects missing/placeholder tokens, flags sparse columns.                                                    |
| `format_consistency_agent`         | `validation.run_column_format_check` | `ColumnConsistencyReport`   | *Slow-path only:* when schema has no detected pattern, reads `ColumnFormatFacts` and decides whether an inconsistency exists.                |
| `anomaly_summary_agent`            | `validation.run_anomaly_detection` | `AnomalySummaryOutput`        | Summarises numeric-outlier + rare-category findings produced by heuristics.                                                                  |
| `cross_column_summary_agent`       | `validation.run_cross_column_validation` | `CrossColumnSummaryOutput` | Summarises duplicate-like-columns, semantic-conflict, year-month-period and date-order findings.                                             |
| `duplicate_summary_agent`          | `validation.run_duplicate_detection` | `DuplicateSummaryOutput`    | Summarises exact + near duplicate row groups.                                                                                                |
| `column_cleaner_generator_agent`   | `cleaning.generation`       | `ColumnCleanerProgram`        | Writes a self-contained Python cleaner function for one column, enforced by a single grouped code-execution check (`UsageLimits(tool_calls_limit=1)`). |
| `cleaner_repair_critic_agent`      | `cleaning.generation`       | `CleanerRepairDiagnosis`      | Diagnoses why the last generated cleaner failed host-side validation, prescribes `patch_style` + exact repairs.                              |
| `narrative_report_agent`           | `cleaning.reporting`        | `NarrativeReport`             | Turns the `FinalPipelineReport` into the Markdown narrative saved next to the cleaned CSV.                                                   |

`setup_logfire()` is also defined here and called from both the CLI and the
Streamlit app.

### `models.py` (496 lines)

All Pydantic models used as agent output types or passed between stages.
Grouped by concern:

- **Type literals**: `VALID_PANDAS_DTYPE`, `NUMERIC_ROLE`, `STRING_ROLE`.
- **Schema**: `DatasetDtypeInference`, `SchemaColumnEntry`, `SchemaIssue`,
  `SchemaHandoff`, `SchemaSummaryOutput`.
- **Completeness**: `CompletenessFinding`, `CompletenessAnalysisReport`.
- **Format consistency**: `FormatConsistencyFinding`,
  `ConsistencyValidationReport`, `ColumnConsistencyReport`.
- **Anomaly / cross-column / duplicates**: `AnomalyFinding`,
  `AnomalyDetectionReport`, `AnomalySummaryOutput`, `CrossColumnFinding`,
  `CrossColumnValidationReport`, `CrossColumnSummaryOutput`,
  `DuplicateRecordGroup`, `DuplicateDetectionReport`,
  `DuplicateSummaryOutput`.
- **Remediation**: `RemediationAction`, `RemediationPlan`.
- **Cleaning**: `ColumnCleaningRequest`, `ColumnCleanerProgram`,
  `ExampleTransformation`, `CleanerValidationIssue`,
  `CleanerRepairContext`, `CleanerRepairDiagnosis`,
  `GeneratedCleanerArtifact`, `CleaningReport`,
  `ColumnCleanerExecutionReport`, `CellUpdate`.
- **Verification / final**: `ConsistencyVerificationReport`, `FindingDiff`,
  `FinalPipelineReport`, `NarrativeReport`, `OrchestrationStepResult`,
  `CleaningPipelineResult`.

Nothing in `models.py` makes LLM calls — it is pure schema.

### `validation/` — validation pipeline subpackage

One module per stage; every `run_<stage>` helper (and `build_validation_results`)
is re-exported from `validation/__init__.py`:

| Module                       | Function                                 | Agent call                                | Artifact written                           |
|------------------------------|------------------------------------------|-------------------------------------------|--------------------------------------------|
| `validation/schema.py`       | `run_dtype_inference`                    | dtype_inference_agent                     | (returned in-memory, used by schema stage) |
| `validation/schema.py`       | `run_schema_validation`                  | schema_summary_agent                      | `<dataset>.schema.json`                    |
| `validation/completeness.py` | `run_completeness_analysis`              | completeness_analysis_agent               | `<dataset>.completeness.json`              |
| `validation/consistency.py`  | `run_format_consistency_validation`      | format_consistency_agent (slow path only) | `<dataset>.consistency.json`               |
| `validation/anomaly.py`      | `run_anomaly_detection`                  | anomaly_summary_agent                     | `<dataset>.anomaly.json`                   |
| `validation/cross_column.py` | `run_cross_column_validation`            | cross_column_summary_agent                | `<dataset>.cross_column.json`              |
| `validation/duplicates.py`   | `run_duplicate_detection`                | duplicate_summary_agent                   | `<dataset>.duplicates.json`                |
| `validation/bundle.py`       | `build_validation_results`               | — (runs every stage)                      | `<dataset>.validation_bundle.json`         |

Key helpers:

- `validation/_summary.py::summarize_validation_report(agent, prompt_text, report, fallback)` —
  shared boilerplate for anomaly / cross-column / duplicate summary calls.
  Keeps all three paths identical and fallback-safe.
- `validation/consistency.py::run_column_format_check` — per-column fast-path
  vs slow-path decision. If the schema already carries a `detected_pattern`
  we skip the LLM and build the `FormatConsistencyFinding` directly from
  the profiler output; only when the schema pattern is absent do we invoke
  `format_consistency_agent`.
- `validation/consistency.py::_build_suggested_strategy` — enumerates every
  outlier shape group with representative examples so the downstream cleaner
  generator has a concrete per-shape brief.
- `validation/anomaly.py::_duplicate_semantic_suppressed_columns` — suppresses
  anomaly findings on the losing side of a duplicate-column group so we don't
  double-report.

---

## `cleaning/` — cleaning pipeline subpackage

### `__init__.py` (16 lines)

Package docstring naming every submodule and its role. No runtime re-exports.

### `orchestrator.py` (167 lines)

End-to-end driver. Public entry point:

```python
run_cleaning(
    path,
    validation_results=None,
    remediation_plan=None,
    reuse_saved_validation=False,
    reuse_saved_remediation=False,
    cleaner_attempts=10,
) -> CleaningPipelineResult
```

Sequence:

1. `_resolve_validation_results` — either reuse cached bundle or run
   `validation.build_validation_results`.
2. `_resolve_remediation_plan` — either reuse cached plan or run
   `run_remediation_planning`.
3. `_build_cleaning_requests` — for each format-consistency finding, build a
   `ColumnCleaningRequest` via `request.build_column_cleaning_request`.
4. `run_cleaner_generation` — generator/critic loop per column.
5. `run_cleaner_application_with_plan` — applies remediations + generated
   cleaners, writes cleaned CSV.
6. `run_verify` — diffs original vs. cleaned consistency findings.
7. `build_final_report` + `generate_narrative_report` — serialise and
   narrate.

Streamlit (`app.py`) also imports the private helpers `_build_cleaning_requests`
and `_resolve_remediation_plan` to stitch stages manually with UI progress.

### `request.py` (89 lines)

Builds one `ColumnCleaningRequest` per inconsistent column. Merges schema,
completeness, and format-profile signals into the bundle fed to the
generator agent. Special-case for `datetime64[ns]`:

- `_build_datetime_expected_pattern` promotes the pattern name to
  `ISO timestamp ...` / `date YYYY-MM-DD` / etc. based on the dominant
  example.
- `_augment_datetime_strategy` prepends a hard output contract (preserve
  dominant timestamps unchanged, reorder components explicitly, do not just
  swap separators).

### `generation.py` (423 lines) — the generator/critic loop

The heart of the multi-agent cleaning path.

**Key constants**

- `GENERATOR_USAGE_LIMITS = UsageLimits(tool_calls_limit=1)` — the generator
  is allowed **exactly one** grouped `code_execution` call per attempt. This
  prevents hidden self-repair loops inside a single model call and forces the
  host-side validator to own retries.

**Public entry points**

- `run_cleaner_generation(path, reuse_consistency, column_name, max_attempts)`
  — driver: iterates every inconsistent column (or one column if
  `column_name` is set), calls `run_column_cleaner_program`, saves each
  program to disk, writes the `cleaner_manifest.json`.
- `run_column_cleaner_program(dataset_name, request, max_attempts)` — the
  per-column retry loop.
- `run_cleaner_repair_critic(dataset_name, request, previous_program, issues)`
  — single critic call; returns a `CleanerRepairDiagnosis`.

**Per-column loop (`run_column_cleaner_program`)**

```
for attempt in 1..max_attempts:
    prompt  = _build_cleaner_generation_prompt(request, previous_program, issues, repair_diagnosis, stagnation_detected)
    program = column_cleaner_generator_agent.run(prompt, usage_limits=1, [model_settings?])
    issues  = validate_generated_cleaner_program(request, program)   # host-side, no LLM
    if not issues:
        return rebuild_verified_program(request, program)            # attach example transformations

    if same_code_as_previous or same_issue_fingerprint:
        consecutive_stagnant_attempts += 1        # ← stagnation detector
    else:
        consecutive_stagnant_attempts = 0

    repair_diagnosis = run_cleaner_repair_critic(...)
    if not repair_diagnosis.should_retry:
        raise ValueError(...)
```

**Stagnation handling** (the core novelty)

When `consecutive_stagnant_attempts >= 1`:

1. `_build_stagnation_unblock_brief(request)` is appended to the generator
   prompt. It contains a concrete rewrite skeleton with a structural regex
   guard as Step 1 (mandatory, first logic after the None check) and a
   Step 2 comment mandating mutually exclusive delimiter branches. The goal
   is to get the model out of its deterministic rut by giving it a fresh
   structural template.
2. Temperature is bumped: `min(0.2 + 0.1 * (n - 1), 0.5)`. Ramp
   `0.2 → 0.3 → 0.4 → 0.5` (capped at 0.5 to stay on task). Passed to
   `run_agent_with_backoff` via `model_settings={"temperature": bumped_temp}`.

The ramp / cap is conservative on purpose — we want *just enough* noise to
escape the fixed point without drifting off the task.

### `validation.py` (435 lines) — host-side validator (no LLM)

Executes each generated `ColumnCleanerProgram` against every
`dominant_example_values` and `example_inconsistent_values` entry in its
`ColumnCleaningRequest`, then returns a list of
`CleanerValidationIssue`. Categories it can emit:

| Category                          | Meaning                                                                                                |
|-----------------------------------|--------------------------------------------------------------------------------------------------------|
| `program_mismatch`                | Program column name does not equal request column name.                                                |
| `non_self_contained_function`     | Cleaner references an outer-scope name (NameError on load or during a call).                           |
| `runtime_exception`               | Cleaner raised any other exception.                                                                    |
| `shadowed_specific_branch`        | Static pattern check: generic `if '<sep>' in s:` appears above a more specific branch on same sep.     |
| `dominant_value_modified`         | Cleaner rewrote an already-valid dominant example — identity violation.                                |
| `outlier_unchanged`               | Cleaner returned an inconsistent example unchanged.                                                    |
| `wrong_output_shape`              | Output `value_shape` does not match the dominant output shape (non-numeric) / dominant datetime shape. |
| `not_parseable_as_target_dtype`   | Cleaned value is not parseable as `Int64` / `Float64` / `datetime64[ns]` / `boolean`.                  |
| `not_matching_target_pattern`     | Cleaned value does not match the numeric schema pattern (e.g. `YYYYMM`, `4-digit year`).               |

Supporting helpers:

- `dominant_output_shape` / `dominant_datetime_example` — derive target
  shape from the `dominant_example_values`.
- `_datetime_format_regex_from_example` — builds a structural regex from
  digit runs (`2024-02-01` → `^\d{4}-\d{2}-\d{2}$`). Used both to catch
  wrong datetime shapes and as a guide for the stagnation skeleton.
- `detect_shadowed_delimiter_branches` — scans generated source for the
  specific bug that triggered the ISO-vs-Italian-date deadlock: a broad
  `if '-' in s:` above a more specific regex branch on the same delimiter.
- `format_validation_issue`, `format_validation_examples`,
  `validation_issue_fingerprint` — used by the loop to render repair
  prompts and to detect repeated failure signatures (stagnation).
- `rebuild_verified_program` — on success, re-runs the cleaner to produce
  `example_transformations` and a clean `verification_summary`.

### `runtime.py` (113 lines)

`load_cleaner_callable(program)` executes the generated Python code in a
locked-down namespace that exposes `re`, `datetime`, and (if installed)
`dateutil.parser.parse` and `dateparser`. `apply_cleaner_to_series` iterates
a pandas Series through the callable, collects up to 10 sample updates, and
returns a `ColumnCleanerExecutionReport`.

### `remediation.py` (305 lines)

Walks the validation bundle and emits a flat `list[RemediationAction]`.
Action types:

- `rename_column` — from schema naming violations.
- `replace_placeholders_with_null` — from completeness findings.
- `cast_dtype` — from dtype inference + schema.
- `generate_cleaner` — from format-consistency findings.
- `drop_exact_duplicate_column` — from schema duplicate groups.

Each action carries `auto_apply: bool` (safe to execute unattended) and a
natural-language `evidence` field. `RemediationPlan` bundles them with a
summary and is cached under `.validation_cache/<dataset>.remediation_plan.json`.

### `application.py` (360 lines)

Consumes the `RemediationPlan` + the cleaner manifest and produces the
cleaned CSV. Order of operations:

1. `_apply_column_renames` — rename columns per schema rename suggestions.
2. `_apply_placeholder_nulls` — replace placeholder tokens with `NA` in
   columns named by completeness findings.
3. For each generated cleaner in the manifest, call
   `runtime.apply_cleaner_to_series` and record an
   `ColumnCleanerExecutionReport`.
4. `_apply_dtype_casts` — final dtype coercions per schema.
5. Write cleaned CSV to `Data/.cleaning_cache/<dataset>/<dataset>.cleaned.csv`.
6. Emit `CleaningReport` (with the cleaned CSV base64-gzipped inline for
   downstream transport in the API).

Two public entry points: `run_cleaner_application(path)` and
`run_cleaner_application_with_plan(path, plan)`.

### `verification.py` (160 lines)

Re-runs `pipeline.run_format_consistency_validation` on the cleaned CSV
(read as strings to avoid pandas silently normalising formats), aligns
findings through the schema rename map, and emits a per-column `FindingDiff`
with status `resolved` / `improved` / `unchanged` / `regressed` / `new`.
Returns `ConsistencyVerificationReport`. No LLM call.

### `reporting.py` (223 lines)

- `build_final_report(validation, remediation, cleaning, verification)` —
  pure model merge into `FinalPipelineReport`.
- `generate_narrative_report(final_report)` — single call to
  `narrative_report_agent`, returns a `NarrativeReport` (Markdown body +
  metadata).
- `save_final_report` / `save_narrative_report` / `narrative_report_path` —
  filesystem helpers.

### `paths.py` (62 lines)

Cache + output path conventions. All paths live under
`<dataset.parent>/.cleaning_cache/<dataset.stem>/`:

- `generated_cleaners/<column>.py` — one cleaner function per column.
- `cleaner_manifest.json` — list of `GeneratedCleanerArtifact`.
- `<dataset>.cleaned.csv`.
- `<dataset>.final_report.json`.

Also exposes `normalized_schema_name` (from `tools`) as the canonical name
derivation used to name cleaner files and to match column renames.

---

## `tools/` — profiling & helpers consumed by agents

Nothing in `tools/` calls an LLM directly. The package is split by concern;
the thin `tools/__init__.py` facade re-exports the symbols that cross-cut
into `validation/*`, `models.py`, and `cleaning/*`. Internal helpers stay
private to their module to keep the public API obvious.

### `tools/__init__.py` (facade, ~78 lines)

Public facade. Re-exports: `PLACEHOLDER_TOKENS`, `SchemaDuplicateGroup`,
`ColumnFormatFacts`, `FormatOutlierExample`, attachment helpers
(`attach_profile_text`, `attach_text_document`), `load_dataset_frame`,
`gzip_text_to_base64`, `value_shape`, schema helpers
(`is_valid_schema_name`, `normalized_schema_name`, `suggest_schema_name`,
`naming_rule_reason`, `build_dataset_profile`, `build_dtype_inference_text`),
`build_completeness_profile`, `build_column_format_facts`,
numeric-pattern helpers (`matches_numeric_schema_pattern`,
`numeric_pattern_allows_variable_width`), all cross-column / duplicate /
anomaly detectors, and `run_agent_with_backoff`.

### `tools/common_tools.py` (460 lines)

Cross-cutting helpers:

- Attachment helpers (`attach_text_document`, `attach_profile_text`).
- `load_dataset_frame`: central `pd.read_csv` wrapper.
- Encoding: `gzip_text_to_base64` (the cleaned CSV is gzipped + base64'd
  into the `CleaningReport` for API transport).
- Profiling primitives: `sample_non_null_values`, `value_shape`,
  `compute_numeric_parse_pct`, `compute_datetime_parse_pct`,
  `compute_empty_like_pct`.
- Numeric schema pattern matchers: `matches_numeric_schema_pattern` +
  `numeric_pattern_allows_variable_width` — consulted by both the pipeline
  (to detect schema-guided inconsistencies) and the validator (to check
  whether a cleaned numeric value matches its target pattern).
- **Agent runtime** (the core infrastructure used by every LLM call):
  - `parse_retry_after_seconds` — parses 429 body messages.
  - `build_terminal_event_stream_handler(agent_name)` — renders streaming
    agent events to stderr when `AGENT_VERBOSE=1`.
  - `run_agent_with_backoff(agent, prompt, max_attempts=6, usage_limits, model_settings)`
    — the retrying wrapper. Handles HTTP 429 (with `Retry-After`
    parsing + exponential fallback), `ModelAPIError` connection issues,
    and transparently forwards `usage_limits` and `model_settings` to
    `agent.run_sync`. **Every agent call in the project goes through this
    function.**

### `tools/schema_tools.py` (170 lines)

Dataset + column profiling for the schema + dtype agents.
`build_dtype_inference_text(df)` renders the CSV sample the dtype agent
sees. `is_valid_schema_name`, `normalized_schema_name`, `suggest_schema_name`
and `naming_rule_reason` encode the lowercase-snake-case naming rule.
`build_dataset_profile` aggregates per-column stats into a single document
fed to the schema summary agent.

### `tools/completeness_tools.py` (106 lines)

`build_completeness_profile(df, dataset_name)` returns a `CompletenessProfile`
with per-column null counts, missing-like percentages, dominant placeholder
tokens present, and sample values. Consumed by `completeness_analysis_agent`.

### `tools/format_tools.py` (305 lines)

Per-column format profiling:

- `ValueShapeProfile`, `ColumnFormatProfile`, `ColumnFormatFacts`,
  `FormatOutlierExample` — the Pydantic models.
- `compute_top_value_shapes(series)` — histogram of structural shapes
  (digits → `9`, letters → `A`, everything else verbatim).
- `build_column_format_profile(df, column_name)` — combines parse pcts +
  top shapes + samples.
- `build_column_format_facts(df, column_name)` — the compact bundle fed
  to the format-consistency agent *and* to the cleaner generator. Includes
  `dominant_shape`, `dominant_shape_pct`, `dominant_example_values`, and
  the `inconsistent_examples` list that drives the generator's per-shape
  briefing.

### `tools/quality_tools.py` (502 lines)

Rule-based detectors fed into the anomaly / cross-column / duplicate
summary agents (the agents summarise, they do not re-discover):

- **Numeric outliers**: `detect_numeric_outlier_candidates` (IQR band).
- **Rare categories**: `detect_rare_category_candidates`.
- **Duplicate / near-duplicate columns**:
  `detect_duplicate_like_columns`, `detect_duplicate_semantic_conflicts`.
- **Exact + near duplicate rows**:
  `detect_exact_duplicate_groups`, `detect_near_duplicate_groups`,
  `infer_duplicate_key_columns`.
- **Domain rules**: `detect_year_month_period_mismatches`,
  `detect_date_order_violations`.

---

## Where to look for what

| Task                                                          | Start here                                            |
|---------------------------------------------------------------|-------------------------------------------------------|
| Add a new validation stage                                    | new module in `validation/` + `models.py` + new agent in `agents.py` + cache helpers in `cache.py` |
| Change what the cleaner generator sees                        | `cleaning/request.py`  +  `cleaning/generation.py::_build_cleaner_generation_prompt` |
| Add a new host-side failure category                          | `cleaning/validation.py` (register category, emit issue) |
| Tune generator retry/temperature behaviour                    | `cleaning/generation.py::run_column_cleaner_program` |
| Change the generator / critic instructions                    | `agents.py::column_cleaner_generator_agent` / `cleaner_repair_critic_agent` |
| Add a new remediation action type                             | `cleaning/remediation.py` + `cleaning/application.py` |
| Change the narrative output                                   | `agents.py::narrative_report_agent` + `cleaning/reporting.py` |
| Add a new cross-column / duplicate / anomaly check            | `tools/quality_tools.py` + wire into the matching `validation.run_*` |
| Change agent retry / timeout / temperature override plumbing  | `tools/common_tools.py::run_agent_with_backoff` |
