# Remediation And Reporting Workflow Plan

## Summary
Add a separate remediation-planning stage that sits between `validate` and `clean`, while preserving the current `generate` loop unchanged. The pipeline becomes:

`validate -> remediate -> clean -> verify -> final report`

`validate` continues to produce findings only. `remediate` converts findings into a structured action plan. `clean` executes only safe auto-actions plus the existing generated cleaners. The final report explains what was found, what was applied, what was only proposed, and what still needs review.

Target planning doc to add during implementation: `docs/REMEDIATION_WORKFLOW.md`, using this same section structure.

## Key Changes
### Workflow and CLI
- Add a new visible CLI stage: `remediate`.
- Keep `validate`, `generate`, `apply`, `verify`, and `clean` working as they do now from the user’s perspective.
- Preserve `generate` as an isolated consistency-driven flow:
  - it still reads consistency findings only
  - it still produces cleaner code the same way
  - remediation does not alter generator prompts, retry logic, or validator behavior
- Update `clean` to:
  - resolve validation results
  - resolve or build a remediation plan
  - run cleaner generation for consistency actions
  - run cleaner application plus remediation auto-actions
  - run verification
  - emit a final report artifact

### New Artifacts and Types
- Add `RemediationAction`, `RemediationPlan`, and `FinalPipelineReport` models.
- Add cache artifacts:
  - `Data/.validation_cache/<stem>.remediation_plan.json`
  - `Data/.cleaning_cache/<stem>/<stem>.final_report.json`
- Extend `CleaningPipelineResult` to include:
  - `remediation_plan`
  - `final_report`
- Each remediation action should include:
  - `action_id`
  - `action_type`
  - `object_type`
  - `target`
  - `source_check`
  - `confidence`
  - `risk_level`
  - `auto_apply`
  - `status`
  - `reason`
  - `preview_stats`

### Remediation Planner
Build the remediation plan from the existing validation bundle. Map findings to actions as follows:

- Schema invalid name -> `rename_column`, `auto_apply=true`
- Completeness placeholder findings -> `replace_placeholders_with_null`, `auto_apply=true`
- Consistency findings -> `generate_cleaner`, `auto_apply=true`
- Exact duplicate columns -> `drop_exact_duplicate_column`, `auto_apply=true`
- Near-duplicate columns -> `manual_review`, `auto_apply=false`
- Duplicate semantic conflicts -> `manual_review`, `auto_apply=false`
- Year/month/period mismatches -> `manual_review`, `auto_apply=false`
- Date-order violations -> `manual_review`, `auto_apply=false`
- Numeric/categorical anomalies -> `report_only` or `manual_review`, `auto_apply=false`
- Exact duplicate rows -> `drop_rows_candidate`, `auto_apply=false`
- Near-duplicate rows -> `manual_review`, `auto_apply=false`
- Dtype casts from schema -> `cast_dtype`, `auto_apply=true`

Use a deterministic keep rule for exact duplicate columns:
1. Prefer schema-valid name over invalid name
2. Then prefer higher non-null count
3. Then prefer the column with an explicit rename suggestion target that is cleaner/canonical
4. Then fall back to stable column order

Do not auto-drop any rows in this phase.

### Apply and Report
Refactor apply so it executes remediation actions in a fixed order:
1. Generated cleaners
2. Placeholder-to-null replacements
3. Exact duplicate column drops
4. Column renames
5. Dtype casts

The final report should contain:
- validation findings summary by section
- remediation actions applied
- remediation actions proposed but skipped
- duplicate row drop candidates noted but not executed
- before/after verification summary
- unresolved risks and manual-review queue

The report should explicitly distinguish:
- `applied`
- `proposed_not_applied`
- `failed`
- `not_needed`

## Test Plan
- Validation-only regression:
  - existing `validate` still produces schema, completeness, consistency, anomaly, cross-column, and duplicate outputs
- Generator regression:
  - `generate` on a known column still uses the same request/critic/validator path and produces the same artifact shape
- Remediation planning:
  - exact duplicate columns create `drop_exact_duplicate_column` actions with deterministic keep/drop targets
  - exact duplicate rows create non-auto-applied `drop_rows_candidate` actions
  - near-duplicate columns and semantic conflicts create `manual_review` actions
- Apply execution:
  - exact duplicate columns are dropped only when the planner marked them `auto_apply=true`
  - row counts are unchanged unless cleaners/placeholder logic already change nullability; no rows are dropped
  - rename and dtype-cast behavior still works after duplicate-column removal
- Reporting:
  - final report includes applied actions, skipped actions, and residual risks
  - verification summary still reflects consistency improvements after cleaning
- CLI:
  - `--stage remediate` builds and prints the remediation plan
  - `--stage clean` reuses or builds remediation and emits the final report
  - `--stage generate` remains unaffected

## Assumptions
- Conservative policy is the default:
  - auto-drop exact duplicate columns only
  - never auto-drop rows
- `generate` must remain behaviorally stable; remediation may plan `generate_cleaner` actions but must not redesign the generator/critic loop.
- Report and remediation are separate concerns:
  - remediation decides what to do
  - report explains what was found and what happened
- The first implementation should stay deterministic and rules-based; agent involvement can be added later for remediation summarization, but the planner itself should remain predictable.
