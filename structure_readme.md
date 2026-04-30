# README Planning Structure

This file answers as many README questions as the current repository supports and flags the places where the notebook should be extended before the final README is written.

## Legend

- `ANSWERED`: supported by the checked-in repo, by your confirmed project framing, or by stored notebook outputs.
- `PARTIAL`: usable draft answer, but at least one detail still needs confirmation or implementation.
- `FLAG`: leave unanswered in the final README for now; add notebook instrumentation, final outputs, or explicit confirmation first.

## Section 0 - End-to-End Workflow Process

This section should appear before the question-and-answer part of the README plan. Its purpose is to explain the full workflow in a technically precise way, not just to list the stages. For every stage, the final README should explain:

1. why the stage exists
2. what problem it solves
3. what exact inputs it consumes
4. what logic it applies
5. whether that logic is deterministic, agent-based, or hybrid
6. what artifact it produces
7. why it is implemented in that specific way
8. how it connects to the next stage
9. what failure modes or issue families it is meant to catch
10. how it contributes to the global objective of improving data quality safely

### 0.1 Global pipeline idea

Status: `ANSWERED`

Draft explanation:
- The project is not a single "clean the CSV" prompt. It is a staged pipeline that separates inspection, diagnosis, decision-making, transformation, and verification.
- The reason for this design is that data quality is not one problem but a family of different problems:
  - schema and naming issues
  - missingness and placeholder abuse
  - format inconsistency
  - anomalies
  - cross-column contradictions
  - duplicate rows or duplicate columns
  - unsafe or incorrect cleaning transformations
- A monolithic prompt would blur all of these together, making the behavior harder to debug, harder to trust, and harder to verify.
- The implemented architecture instead follows a strict sequence:
1. load the dataset
2. validate and profile it
3. summarize the evidence into structured artifacts
4. translate findings into remediation actions
5. generate executable cleaners only where needed
6. validate those cleaners before trusting them
7. apply them to the real dataframe
8. re-check whether the targeted issues were actually reduced
9. assemble a factual final report
10. optionally render that factual report as a narrative document

Core design principle:
- Deterministic Python is used for counting, profiling, comparison, grouping, and validation.
- LLM agents are used where interpretation, synthesis, or transformation logic is genuinely useful.
- This division of labor is what makes the system both flexible and auditable.

### 0.2 Main entry surfaces

Status: `ANSWERED`

Draft explanation:
- The workflow can be accessed in three ways:
1. `main.ipynb`: illustrative guided walkthrough
2. CLI entrypoints in `entrypoints/cli.py` and `entrypoints/main.py`
3. `app.py`: interactive Streamlit interface

Why multiple entry surfaces exist:
- The notebook exists to explain the pipeline and inspect intermediate artifacts.
- The CLI exists to run the real pipeline in a reproducible way, stage by stage or end to end.
- The app exists to make the same underlying stages easier to inspect visually.

Important architectural point:
- These are not three different pipelines.
- They are three interfaces over the same core modules.
- The real logic lives in the `validation/`, `cleaning/`, `tools/`, and `core/` packages.

### 0.3 Core architectural layers

Status: `ANSWERED`

Draft explanation:

`Layer 1 - Contracts`
- The Pydantic models in [core/models.py](core/models.py) are the contract layer.
- Every major handoff in the system is represented as a typed object.
- This prevents one stage from passing vague or malformed text to the next stage.
- It also makes the whole workflow inspectable, serializable, and cacheable.

Why this matters:
- In an agentic pipeline, the dangerous part is not only a wrong answer, but an answer that has the wrong structure and silently poisons downstream stages.
- Pydantic is used to constrain that risk.

`Layer 2 - Deterministic evidence builders`
- The `tools/` modules compute the raw evidence:
  - parse rates
  - samples
  - dominant shapes
  - placeholder tokens
  - duplicate similarity
  - anomaly candidates
  - cross-column conflicts
- This layer does not ask the LLM to discover facts from scratch.
- It computes facts locally and then passes them forward.

Why this matters:
- It is cheaper, faster, and more reproducible to let Python count, parse, and compare values than to ask the LLM to do brute-force inspection.

`Layer 3 - Agents`
- Agents are introduced only when the pipeline needs:
  - semantic interpretation
  - structured summarization
  - constrained code generation
  - repair diagnosis
  - narrative reporting

Why this matters:
- The agents are used as narrow specialists, not as a universal controller.

`Layer 4 - Host-side enforcement`
- Even when the LLM generates code, the host system remains the final authority.
- Generated cleaners are re-loaded, re-run, and rejected if they fail deterministic checks.

Why this matters:
- This is the key safety barrier that keeps the system from trusting the model blindly.

### 0.4 Data loading and initial framing

Status: `ANSWERED`

Files involved:
- [tools/common_tools.py](tools/common_tools.py)
- [validation/bundle.py](validation/bundle.py)
- [cleaning/orchestrator.py](cleaning/orchestrator.py)

Why this stage exists:
- Every later stage depends on a clean, centralized way of loading the dataset.
- If CSV reading behavior were inconsistent across stages, later comparisons would become unreliable.

What it does:
- Loads the CSV into a pandas dataframe.
- Keeps the raw table as the authoritative input for the validation half.
- In verification, may re-read the cleaned CSV as strings to avoid pandas silently normalizing format differences away.

Why it is done this way:
- Centralizing CSV loading reduces accidental inconsistency.
- Reading verification inputs as strings is especially important because the goal there is to inspect formatting, not just semantic dtype compatibility.

What it produces:
- An in-memory dataframe that feeds the profiling and validation logic.

How it interacts with the next stage:
- The loaded dataframe becomes the common source for schema, completeness, consistency, anomaly, cross-column, and duplicate detection.

Issues this stage is designed to avoid:
- inconsistent read behavior across stages
- accidental dtype coercion hiding format issues
- stage-specific interpretations of the raw file

Contribution to data-quality goal:
- It creates a stable observation layer before any interpretation begins.

### 0.5 Agent runtime and observability

Status: `ANSWERED`

Files involved:
- [core/agents.py](core/agents.py)
- [tools/common_tools.py](tools/common_tools.py)

Why this stage exists:
- Agent calls are expensive and failure-prone relative to local Python logic.
- The system needs one centralized runtime policy for retries, tracing, and optional model-setting overrides.

What it does:
- Defines all agents in one place.
- Configures Logfire through `setup_logfire()`.
- Routes every agent call through `run_agent_with_backoff(...)` or its async variant.
- Handles:
  - HTTP 429 retries
  - connection-level transient failures
  - optional temperature overrides
  - optional usage limits

Why it is done this way:
- If each stage implemented its own retry behavior, the system would become difficult to reason about and debug.
- A central wrapper makes the agent runtime behavior consistent.

What it produces:
- Reliable, traced agent execution with consistent retry semantics.

How it interacts with the next stage:
- Every LLM-assisted stage depends on this layer.

Issues this stage is designed to catch or mitigate:
- rate limits
- transient connection errors
- invisible prompt/runtime behavior
- duplicated orchestration logic across modules

Contribution to data-quality goal:
- It does not detect data-quality issues directly, but it makes the agentic parts of the pipeline observable, stable, and measurable.

### 0.6 Schema and dtype inference stage

Status: `ANSWERED`

Files involved:
- [validation/schema.py](validation/schema.py)
- [tools/schema_tools.py](tools/schema_tools.py)
- [core/agents.py](core/agents.py)

Why this stage exists:
- Before the pipeline can judge whether a value is wrong, it must understand what a column is supposed to be.
- This stage establishes the baseline semantic and structural interpretation of the dataset.

What it does:
1. builds a bounded textual profile of each column
2. computes local statistics such as:
   - non-null counts
   - distinct counts
   - numeric parse percentage
   - datetime parse percentage
   - sample values
3. sends that profile to the dtype inference agent
4. converts the result into a structured schema handoff
5. applies naming-rule checks
6. proposes rename suggestions
7. detects duplicate-semantic column groups
8. summarizes the schema findings for downstream use

How it does it:
- The deterministic part in [tools/schema_tools.py](tools/schema_tools.py) builds the column profile.
- `build_dtype_inference_text(...)` currently samples up to 5% of dataset rows, capped at `500` unique non-null values per column, for the dtype prompt.
- The `dtype_inference_agent` interprets those statistics and samples to infer:
  - target pandas dtype
  - numeric role or string role
  - dominant pattern
  - rationale
- `run_schema_validation(...)` then merges the agent output with deterministic naming and duplicate-name logic into a `SchemaHandoff`.

What it produces:
- `SchemaHandoff`
- per-column schema entries
- schema issues
- duplicate semantic groups
- summary text

Why it is implemented in this specific way:
- The system wants the cleaned target dtype, not just the raw ingestion dtype.
- Parse percentages provide global context so a few dirty values do not dominate the interpretation.
- Naming checks are deterministic because naming-policy enforcement is a rules problem, not an LLM creativity problem.

How it interacts with the next stage:
- Completeness, consistency, anomaly detection, remediation planning, and verification all depend on the schema handoff.
- Detected patterns from schema also enable a deterministic fast path in format consistency checking.

Issues it is designed to catch:
- invalid column names
- semantically duplicate columns
- dtype misunderstandings
- misleading raw storage formats
- period/date/code confusion

Contribution to data-quality goal:
- It defines what "clean" should mean for each column before later stages start labeling deviations.

### 0.7 Completeness analysis stage

Status: `ANSWERED`

Files involved:
- [validation/completeness.py](validation/completeness.py)
- [tools/completeness_tools.py](tools/completeness_tools.py)
- [core/agents.py](core/agents.py)

Why this stage exists:
- Missingness is not only about `NaN`.
- Real datasets often hide absence behind placeholders such as `-`, `N/A`, `unknown`, or empty strings.
- If the pipeline ignores those tokens, later stages will treat fake values as real data.

What it does:
1. computes per-column completeness statistics
2. identifies missing-like tokens
3. records representative placeholder spellings
4. estimates dataset-wide overall completeness
5. lets the completeness agent interpret that evidence into a structured action-oriented report

How it does it:
- [tools/completeness_tools.py](tools/completeness_tools.py) builds a deterministic `CompletenessProfile`.
- It uses:
  - `compute_missing_like_mask(...)`
  - `sample_placeholder_examples(...)`
  - `detect_placeholder_values(...)`
- The completeness agent then turns that profile into a `CompletenessAnalysisReport`.
- [validation/completeness.py](validation/completeness.py) also backfills placeholder examples from the deterministic profile if the agent omitted them.

What it produces:
- `CompletenessAnalysisReport`
- overall completeness percentage
- columns with missing-like values
- sparse-column candidates
- placeholder values detected
- per-column recommended actions

Why it is implemented in this specific way:
- The system separates measurement from interpretation.
- Python measures the missingness.
- The agent turns those measurements into structured conclusions and downstream-readable recommendations.
- Backfilling placeholder examples protects the pipeline from depending on the model to faithfully echo every concrete token.

How it interacts with the next stage:
- Remediation planning converts these findings into placeholder-to-null actions.
- Cleaning application later uses those actions to replace placeholder tokens with true nulls.

Issues it is designed to catch:
- hidden missingness
- placeholder-token abuse
- nearly empty columns
- misleadingly "non-null" cells that contain no real information

Contribution to data-quality goal:
- It ensures that later stages reason over actual information content rather than over placeholder noise.

### 0.8 Format consistency stage

Status: `ANSWERED`

Files involved:
- [validation/consistency.py](validation/consistency.py)
- [tools/format_tools.py](tools/format_tools.py)
- [core/agents.py](core/agents.py)

Why this stage exists:
- Many columns are not wrong in meaning, but wrong in representation.
- Dates, amounts, period keys, and codes may all be semantically valid while appearing in inconsistent formats inside the same column.
- This is the stage that turns "messy representation" into "cleaning-needed target columns."

What it does:
1. profiles the structural shapes present in each column
2. identifies the dominant canonical shape
3. collects representative outlier families
4. decides whether the inconsistency is actionable
5. produces one `FormatConsistencyFinding` per dirty column

How it does it:
- [tools/format_tools.py](tools/format_tools.py) computes:
  - dominant shape
  - dominant shape percentage
  - semantic hint
  - representative dominant values
  - representative inconsistent values
- Outliers are selected through `select_outlier_examples(...)`, which groups values by structural shape and retains representative examples from the most frequent outlier families, up to `60` total examples.
- If schema already provides a stable pattern, [validation/consistency.py](validation/consistency.py) can take a deterministic fast path.
- If schema does not provide a sufficient pattern, the `format_consistency_agent` is invoked on the slow path to decide whether a real actionable format issue exists.

What it produces:
- `ConsistencyValidationReport`
- one `FormatConsistencyFinding` per actionable dirty column
- summary text

Why it is implemented in this specific way:
- Not every variation is a problem.
- Free text and descriptive fields should not be forced into fake canonical structure.
- The machine-format candidate logic exists so the system only generates cleaners where normalization is meaningful and feasible.
- Grouping outliers by shape is much more useful than sending arbitrary isolated noisy values.

How it interacts with the next stage:
- This is the hinge stage between validation and cleaning.
- Each `FormatConsistencyFinding` becomes:
  - a `generate_cleaner` remediation action
  - then a `ColumnCleaningRequest`
  - then an executable cleaning target

Issues it is designed to catch:
- mixed date formats
- mixed monetary formatting
- mixed `YYYYMM` / `MM/YYYY` / textual period formats
- canonical-format drift within one column

Contribution to data-quality goal:
- It identifies exactly where normalization should happen and defines the target that the cleaning stage will attempt to enforce.

### 0.9 Anomaly detection stage

Status: `ANSWERED`

Files involved:
- [validation/anomaly.py](validation/anomaly.py)
- [tools/quality_tools.py](tools/quality_tools.py)
- [core/agents.py](core/agents.py)

Why this stage exists:
- Some issues are not about formatting or missingness, but about suspicious values that deserve review.
- The system needs to distinguish these from format-cleaning problems.

What it does:
1. detects numeric outlier candidates
2. detects rare categorical labels
3. suppresses duplicate-semantic aliases where appropriate
4. summarizes the findings for downstream review and reporting

How it does it:
- Numeric outliers are detected deterministically with a robust IQR-based rule in [tools/quality_tools.py](tools/quality_tools.py).
- Rare categories are also detected deterministically using bounded-frequency heuristics.
- The anomaly agent does not discover anomalies itself; it only summarizes already computed findings.

What it produces:
- `AnomalyDetectionReport`
- anomaly findings
- summary text

Why it is implemented in this specific way:
- Anomaly detection here is treated conservatively and as a reporting/manual-review problem, not as an auto-cleaning problem.
- That design reduces the risk of the system "fixing" real but extreme values.

How it interacts with the next stage:
- Remediation planning can mark anomaly findings as manual review or report-only actions.
- Final reporting surfaces them as risk signals rather than direct corrections.

Issues it is designed to catch:
- suspiciously extreme numeric values
- rare category spellings
- possible unit errors
- possible spelling variants or category drift

Contribution to data-quality goal:
- It broadens the pipeline beyond formatting and missingness, while remaining conservative about intervention.

### 0.10 Cross-column validation stage

Status: `ANSWERED`

Files involved:
- [validation/cross_column.py](validation/cross_column.py)
- [tools/quality_tools.py](tools/quality_tools.py)
- [core/agents.py](core/agents.py)

Why this stage exists:
- Data quality is not only intra-column.
- A dataset can contain columns that individually look reasonable but contradict each other when compared relationally.

What it does:
1. detects exact duplicate columns
2. detects near-duplicate columns
3. detects duplicate-semantic conflicts
4. detects year/month/period mismatches
5. detects date-order violations
6. summarizes the findings

How it does it:
- All relational checks are deterministic and implemented in [tools/quality_tools.py](tools/quality_tools.py).
- The cross-column summary agent only writes the structured narrative over those findings.

What it produces:
- `CrossColumnValidationReport`
- one `CrossColumnFinding` per relational problem
- summary text

Why it is implemented in this specific way:
- Relational consistency is a rule-comparison problem, so deterministic checks are both cheaper and safer than asking the LLM to infer them from raw tables.
- The agent is reserved for explanation, prioritization, and downstream readability.

How it interacts with the next stage:
- Remediation planning converts:
  - exact duplicate columns into possible auto-drop actions
  - ambiguous relational issues into manual-review actions

Issues it is designed to catch:
- redundant columns
- columns that should agree but do not
- mismatched temporal encodings
- logically impossible date ordering

Contribution to data-quality goal:
- It catches contradictions that are invisible if columns are inspected one by one.

### 0.11 Duplicate-row detection stage

Status: `ANSWERED`

Files involved:
- [validation/duplicates.py](validation/duplicates.py)
- [tools/quality_tools.py](tools/quality_tools.py)
- [core/agents.py](core/agents.py)

Why this stage exists:
- A dataset can be polluted by repeated rows or near-duplicate records even when individual cells look fine.

What it does:
1. detects exact duplicate row groups
2. infers likely business-key columns
3. detects near-duplicate groups that share the inferred key but differ elsewhere
4. summarizes the evidence

How it does it:
- Exact duplicate groups are computed deterministically by normalized full-row signatures.
- Near-duplicate groups are formed by grouping on inferred key columns and inspecting disagreements elsewhere.
- The duplicate summary agent narrates the result but does not decide aggressive deletion.

What it produces:
- `DuplicateDetectionReport`
- exact and near-duplicate record groups
- summary text

Why it is implemented in this specific way:
- Duplicate handling is potentially destructive.
- The system therefore distinguishes exact duplicates from near duplicates and remains conservative about what is auto-applied later.

How it interacts with the next stage:
- Remediation planning can record row-drop candidates or manual-review actions.
- Final reporting can surface duplicate volume even when no rows are auto-removed.

Issues it is designed to catch:
- repeated records
- duplicate records with minor disagreements
- conflicting copies of the same logical record

Contribution to data-quality goal:
- It addresses redundancy and possible record-level inconsistency, not just cell-level noise.

### 0.12 Validation bundling stage

Status: `ANSWERED`

Files involved:
- [validation/bundle.py](validation/bundle.py)

Why this stage exists:
- The cleaning half should not have to re-run or rediscover the entire validation half ad hoc.
- It needs one canonical object representing the complete validation state.

What it does:
- Runs or collects all validation stages:
  - schema
  - completeness
  - consistency
  - anomaly
  - cross-column
  - duplicates
- bundles them into one `OrchestrationStepResult`

What it produces:
- `OrchestrationStepResult`

Why it is implemented in this specific way:
- A single bundle reduces orchestration complexity and makes downstream planning deterministic.

How it interacts with the next stage:
- Remediation planning consumes this bundle as its main input.

Issues it is designed to avoid:
- inconsistent downstream access to partial validation artifacts
- stage-order drift
- fragmented evidence handling

Contribution to data-quality goal:
- It creates a stable transition point between inspection and action.

### 0.13 Remediation-planning stage

Status: `ANSWERED`

Files involved:
- [cleaning/remediation.py](cleaning/remediation.py)
- [cleaning/orchestrator.py](cleaning/orchestrator.py)

Why this stage exists:
- Validation findings alone do not tell the system what it should actually do.
- The system needs a deterministic policy layer that translates findings into actions.

What it does:
- Converts validation findings into a flat `RemediationPlan` made of typed `RemediationAction` objects.

Action families include:
- `rename_column`
- `replace_placeholders_with_null`
- `cast_dtype`
- `generate_cleaner`
- `drop_exact_duplicate_column`
- `manual_review`
- `report_only`
- duplicate-row candidate actions

How it does it:
- It walks the validation bundle and maps each finding type to a remediation policy.
- Safe, high-confidence actions are marked as auto-applicable.
- Ambiguous or risky findings are deferred to manual review.

What it produces:
- `RemediationPlan`
- an ordered list of actions with:
  - action type
  - target
  - source check
  - confidence
  - risk level
  - auto-apply flag
  - status
  - reason
  - preview stats

Why it is implemented in this specific way:
- This stage separates "what the pipeline found" from "what the pipeline is willing to do automatically."
- That separation is crucial for safety and explainability.

How it interacts with the next stage:
- The application stage later executes the safe subset.
- The generation stage uses the `generate_cleaner` actions as its entry points.

Issues it is designed to catch or handle:
- unsafe over-automation
- lack of action traceability
- mixing policy decisions with raw findings

Contribution to data-quality goal:
- It converts evidence into explicit, auditable intervention policy.

### 0.14 Cleaning-request construction stage

Status: `ANSWERED`

Files involved:
- [cleaning/request.py](cleaning/request.py)
- [cleaning/orchestrator.py](cleaning/orchestrator.py)

Why this stage exists:
- A format-consistency finding is not yet a generator-ready object.
- The cleaner generator needs a compact, authoritative contract for one column.

What it does:
- Builds one `ColumnCleaningRequest` for each target column.

Each request includes:
- column name
- expected pattern
- semantic hint
- target dtype / role
- dominant shape
- dominant examples
- inconsistent examples
- suggested strategy

How it does it:
- Merges:
  - the format-consistency finding
  - fresh format facts
  - optional schema metadata
- Adds special augmentation for:
  - datetime columns
  - `YYYYMM` period-key behavior

What it produces:
- `ColumnCleaningRequest`

Why it is implemented in this specific way:
- Cleaner generation works best when the task is column-local, bounded, and explicit.
- This stage compresses all relevant context into a single target contract.

How it interacts with the next stage:
- Each request becomes one generator/critic problem instance.

Issues it is designed to catch or prevent:
- underspecified cleaning tasks
- generator confusion about canonical outputs
- accidental loss of recoverable context

Contribution to data-quality goal:
- It gives the generator just enough grounded context to write a useful cleaner without exposing the whole dataset.

### 0.15 Cleaner generation stage

Status: `ANSWERED`

Files involved:
- [cleaning/generation.py](cleaning/generation.py)
- [core/agents.py](core/agents.py)

Why this stage exists:
- Some inconsistencies cannot be repaired safely with a one-size-fits-all rule.
- The system therefore generates one focused cleaning function per dirty column.

What it does:
1. prompts the generator for one self-contained Python function
2. allows exactly one grouped sandbox test during that attempt
3. validates the generated code host-side
4. if validation fails, asks the critic for a repair diagnosis
5. retries until success or budget exhaustion

How it does it:
- `run_column_cleaner_program(...)` in [cleaning/generation.py](cleaning/generation.py) owns the retry loop.
- The generator agent is restricted by `GENERATOR_USAGE_LIMITS = UsageLimits(tool_calls_limit=1)`.
- The critic does not write code; it diagnoses failure causes and suggests minimal or targeted repairs.
- Stagnation detection prevents the loop from repeating the same wrong solution forever.

What it produces:
- `ColumnCleanerProgram`
- then `GeneratedCleanerArtifact`
- generated cleaner files on disk
- cleaner manifest

Why it is implemented in this specific way:
- The system wants the LLM to propose transformation logic, but not to run an uncontrolled hidden repair loop.
- By limiting sandbox usage and making the outer loop responsible for retries, the pipeline stays observable and auditable.

How it interacts with the next stage:
- Accepted cleaner artifacts are passed to the application stage and applied to real dataframe columns.

Issues it is designed to catch:
- mixed-format columns needing custom transformation logic
- brittle string rewrites
- non-self-contained generated code
- cleaners that damage already-valid values

Contribution to data-quality goal:
- It is the bridge from diagnosis to executable repair.

### 0.16 Host-side cleaner validation stage

Status: `ANSWERED`

Files involved:
- [cleaning/validation.py](cleaning/validation.py)
- [cleaning/runtime.py](cleaning/runtime.py)

Why this stage exists:
- Generated code must never be trusted just because it looks plausible.

What it does:
- Loads each generated cleaner into an isolated runtime.
- Re-tests it against dominant examples and inconsistent examples.
- Emits structured validation issues if anything is wrong.

Checks include:
- self-containment
- runtime exceptions
- dominant-value preservation
- outlier transformation
- output shape
- target dtype parseability
- target pattern matching
- shadowed branch structure

What it produces:
- either:
  - an accepted, rebuilt verified program with example transformations
  - or a list of `CleanerValidationIssue` objects

Why it is implemented in this specific way:
- This is the hard safety barrier of the pipeline.
- The model may suggest code, but deterministic host logic decides whether that code is acceptable.

How it interacts with the next stage:
- Successful cleaners move to application.
- Failed cleaners go back into the generator/critic loop.

Issues it is designed to catch:
- broken code
- unsafe code structure
- already-valid values being modified
- dirty values left unchanged
- output contract violations

Contribution to data-quality goal:
- It prevents the system from introducing new errors while attempting to fix old ones.

### 0.17 Cleaner application stage

Status: `ANSWERED`

Files involved:
- [cleaning/application.py](cleaning/application.py)

Why this stage exists:
- Validation and generation only prepare actions.
- This is the stage that actually mutates the dataset.

What it does:
- Executes generated cleaners and deterministic remediation actions in a fixed order.

Current order:
1. generated cleaners
2. placeholder-to-null replacements
3. duplicate-column drops
4. column renames
5. dtype casts

How it does it:
- Loads the cleaner manifest.
- Applies each cleaner to its target series.
- Tracks changed rows and sample updates.
- Updates remediation statuses.
- Writes the cleaned CSV.

What it produces:
- `CleaningReport`
- per-cleaner execution reports
- cleaned CSV

Why it is implemented in this specific way:
- Order matters.
- For example, cleaners run before renames because cleaner manifests are keyed by original column names.
- Dtype casts run after cleaning because normalization should happen before coercion.

How it interacts with the next stage:
- The cleaned CSV becomes the input to verification.

Issues it is designed to catch or avoid:
- applying actions in the wrong order
- mutating the wrong columns
- losing traceability between generated code and dataframe effects

Contribution to data-quality goal:
- It turns validated plans and validated cleaners into real dataset changes while preserving accountability.

### 0.18 Verification stage

Status: `ANSWERED`

Files involved:
- [cleaning/verification.py](cleaning/verification.py)

Why this stage exists:
- Cleaning is only meaningful if it can be shown to improve the targeted problems.
- The system therefore does not assume success; it measures it.

What it does:
- Re-runs format consistency on the cleaned CSV.
- Aligns renamed columns back to original identities.
- Compares before vs after findings.
- assigns statuses:
  - `resolved`
  - `improved`
  - `unchanged`
  - `regressed`
  - `new`

How it does it:
- Loads the original consistency baseline.
- Re-reads the cleaned CSV as strings.
- Re-runs consistency validation.
- Builds one `FindingDiff` per original issue and additional diffs for new findings.

What it produces:
- `ConsistencyVerificationReport`

Why it is implemented in this specific way:
- The generator stage is mainly about format repair, so verification focuses on consistency findings rather than on a vague "dataset looks better" claim.
- Reading as strings avoids pandas hiding residual formatting problems through automatic coercion.

How it interacts with the next stage:
- The verification report becomes a core input to the final structured report.

Issues it is designed to catch:
- cleaners that did not actually improve the targeted issue
- regressions
- new inconsistencies introduced during cleaning

Contribution to data-quality goal:
- It closes the loop between intervention and measured outcome.

### 0.19 Final structured reporting stage

Status: `ANSWERED`

Files involved:
- [cleaning/reporting.py](cleaning/reporting.py)

Why this stage exists:
- The system needs one canonical factual object summarizing the whole run.
- The narrative layer should be generated from facts, not from free-form recollection.

What it does:
- Merges validation, remediation, cleaning, and verification outputs.
- Builds counts and partitions actions by outcome.

What it produces:
- `FinalPipelineReport`

Why it is implemented in this specific way:
- The final report is the authoritative evidence layer for:
  - README result extraction
  - narrative generation
  - downstream inspection

How it interacts with the next stage:
- The narrative agents consume this object indirectly through compact briefings.

Issues it is designed to avoid:
- mixing facts and prose too early
- inconsistent reporting across interfaces
- manual result assembly errors

Contribution to data-quality goal:
- It makes the full run inspectable as structured evidence.

### 0.20 Narrative reporting stage

Status: `ANSWERED`

Files involved:
- [cleaning/reporting.py](cleaning/reporting.py)
- [core/agents.py](core/agents.py)

Why this stage exists:
- Structured JSON is not the best final communication format for human readers.
- The project needs a readable final report without losing factual grounding.

What it does:
- Builds compact factual briefings from the `FinalPipelineReport`.
- Uses:
  - `narrative_frontmatter_agent`
  - `narrative_section_agent`
- Generates title, summary, recommendations, and section bodies.

What it produces:
- `NarrativeReport`
- saved markdown report

Why it is implemented in this specific way:
- The system avoids a single monolithic narrative prompt.
- Splitting frontmatter and sections makes the final report more controllable and reduces the chance of one giant prompt drifting off the evidence.

How it interacts with the overall pipeline:
- This is the presentation layer, not the truth layer.
- It should never invent new findings.

Issues it is designed to avoid:
- monolithic narrative drift
- unsupported claims in the final report
- low readability of structured output

Contribution to data-quality goal:
- It makes the pipeline outcome understandable to human readers without sacrificing factual grounding.

### 0.21 Why this workflow is a multi-agent system and not just a scripted pipeline

Status: `ANSWERED`

Draft explanation:
- The system is multi-agent because different specialized agents are assigned different cognitive roles:
  - dtype interpretation
  - schema summarization
  - completeness interpretation
  - slow-path consistency judgment
  - anomaly narration
  - cross-column narration
  - duplicate narration
  - cleaner generation
  - repair criticism
  - narrative frontmatter writing
  - narrative section writing
- But it is not an unconstrained "agents talking freely to each other" architecture.
- Their interactions are mediated by typed artifacts and host-controlled orchestration.

Why this matters:
- The architecture keeps the benefits of specialization without losing traceability.

### 0.22 Key workflow takeaway for the final README

Status: `ANSWERED`

Recommended framing:
- The workflow should be presented as a controlled progression from raw evidence to validated intervention.
- The most important architectural message is:
  - the model is used where reasoning helps
  - deterministic Python is used where correctness, measurement, and enforcement matter more
  - every transformation is checked before it is trusted

## Section 1 - Introduction

### 1. What is the exact affiliation/course context for this project?

Status: `ANSWERED`

Draft answer:
- This is a `2026` project for a `Machine Learning` class.
- The project was developed in collaboration with `REPLY`, which provided the task: build a multi-agent system for data quality.
- The repo itself also states that the work belongs to the `Machine Learning` / `Data Science and Management` course context.

Recommended final README wording:
- "This project was developed in 2026 for a Machine Learning course, in collaboration with REPLY, which proposed the task of building a multi-agent system for data quality."

Open gap:
- If you want the title block to be fully formal, we still need the exact official university name as it should appear in the submission.

### 2. Does the project include a Jupyter notebook? If so, what is its filename and location?

Status: `ANSWERED`

Draft answer:
- Yes. The main guided walkthrough is `main.ipynb` in the repository root.
- The notebook is primarily illustrative: it explains the production pipeline step by step and shows the intermediate artifacts in a readable form.
- The full workflow is not notebook-only. The same pipeline can also be run from the CLI or by running `main.py`.
- `main_mattia.ipynb` is an old version and should not be referenced as the canonical notebook in the README.

Recommended final README wording:
- "The repository includes a root-level notebook, `main.ipynb`, used mainly for illustrative and explanatory purposes. The full pipeline can also be executed directly from the CLI or by running `main.py`."

## Section 2 - Methods

### 3. Which model name is accurate for the final submission? Were multiple models tested?

Status: `ANSWERED`

Draft answer:
- The final checked-in production model is `openai-responses:gpt-5.4-nano` in [core/agents.py](core/agents.py).
- Multiple models were tested during development.
- Early exploration included small local-model variants, but they were discarded because the available hardware was not strong enough to support them comfortably.
- The development history you confirmed is:
1. local small-model experiments
2. `gemma3`
3. `gemini 3.1 flash light preview`
4. move to OpenAI
5. initial OpenAI testing with `GPT-4 mini`
6. final production choice: `GPT-5.4 nano`

Interpretation for the README:
- The main reason for the final switch was not only quality, but efficiency.
- With the current prompt-bounded pipeline, a full stage now takes roughly `2-3 minutes` on average, whereas older tested setups could take around `20 minutes`.
- Cost remains negligible because the pipeline deliberately limits what is sent to the model and reserves the LLM for interpretation and synthesis rather than brute-force data processing.

Recommended final README wording:
- "The final submission uses `GPT-5.4 nano`. During development we tested smaller local-model variants, then `gemma3`, `gemini 3.1 flash light preview`, and later OpenAI models including `GPT-4 mini`. The final choice was driven by the combination of strong performance, much shorter stage runtime, and negligible cost under the pipeline's token-efficient design."

Side question - can we compute tokens and cost per run?

Answer: yes.

What the current repo already supports:
- [core/agents.py](core/agents.py) enables `logfire.instrument_pydantic_ai()`.
- The notebook already documents that Logfire captures prompt traces, token usage, and latency.

Practical options:
1. `Token usage`: yes, this can be extracted per agent call or per full run from Logfire traces.
2. `Cost`: yes, if we combine token totals with the model pricing table used for the run.

Best implementation path:
- Add a notebook cell that aggregates token usage from the traced agent calls.
- Add a second cell that multiplies those totals by a manually declared pricing dictionary for the exact model used in that experiment.

Why I am not hard-coding a cost number here:
- Exact pricing is external and can change over time, so the robust solution is to compute cost from recorded usage plus a versioned price table stored in the notebook.

### 4. How were 500 values chosen, and what token count does that correspond to?

Status: `FLAG`

Do not keep the current README claim. It is incorrect for the checked-in code.

What the code actually does today:
- For dtype inference, [tools/schema_tools.py](tools/schema_tools.py) samples up to 5% of dataset rows, capped at `500` unique non-null values per column, in `build_dtype_inference_text(...)`.
- For outlier transmission in format profiling and cleaner generation, [tools/format_tools.py](tools/format_tools.py) uses `select_outlier_examples(...)` with:
  - `max_shapes=10`
  - `max_per_shape=10`
  - `max_total=60`
- Those outliers are selected by grouping values by structural shape, ranking shape families by frequency, and then keeping representative examples from the most common outlier families.
- The generator then also receives a few dominant examples, plus the schema-guided expected pattern and strategy text.

What to say now:
- The real current design is not "500 values per call" for every dataset and column; `500` is only the upper cap.
- The real current design is "bounded representative evidence per call", with:
1. up to 5% of dataset rows, capped at `500`, sampled values for dtype inference
2. up to `60` grouped outlier examples for consistency/cleaning evidence
3. a small number of dominant examples for the target canonical pattern

Why this is a room for improvement:
- The outlier selection rule is deterministic and compact, but still somewhat hand-tuned.
- It would be useful to measure whether different caps or more adaptive sampling improve cleaning quality or reduce cost further.

Recommended notebook addition before answering in the README:
1. add a cell that prints the exact number of values/examples sent to each agent family
2. add a cell that estimates prompt tokens from the real payloads
3. add a short "sampling policy" subsection with the 5%-capped-at-`500` and `60` rules and their file locations

Final README status:
- Leave the old "500 values" sentence out.
- Replace it only after the notebook explicitly reports the real prompt payload sizes and token totals.

### 5. List each agent and its exact responsibility.

Status: `ANSWERED`

Current production agents in [core/agents.py](core/agents.py):

1. `dtype_inference_agent`: infers the cleaned pandas dtype, semantic role, and dominant target pattern for each column from bounded evidence rather than from the entire raw column.
2. `schema_summary_agent`: summarizes schema-level findings such as naming fixes, duplicate-semantic groups, and dtype-risk handoff notes for downstream stages.
3. `completeness_analysis_agent`: turns the deterministic completeness profile into a structured report on missing-like values, placeholders, and sparse columns.
4. `format_consistency_agent`: decides, on the slow path, whether a single column has an actionable dominant format plus inconsistent outliers.
5. `anomaly_summary_agent`: narrates deterministic anomaly findings such as numeric outliers and rare-category signals.
6. `cross_column_summary_agent`: narrates deterministic cross-column findings such as duplicate-like columns, semantic conflicts, and temporal mismatches.
7. `duplicate_summary_agent`: narrates deterministic exact-duplicate and near-duplicate row findings.
8. `column_cleaner_generator_agent`: writes one self-contained Python cleaner for a flagged column and tests it once with the sandbox tool.
9. `cleaner_repair_critic_agent`: diagnoses why a generated cleaner failed host-side validation and prescribes the smallest credible repair before the next attempt.
10. `narrative_frontmatter_agent`: writes the report title, executive summary, and recommendations from the final factual report.
11. `narrative_section_agent`: writes one grounded narrative section at a time from section-specific factual briefings.

Important writing note for the future README:
- Your comment is exactly right: the final README should not stop at one-line roles.
- For each stage/agent, we should explain:
1. why it exists
2. what it does
3. how it does it
4. what it produces
5. why it does it in that specific way
6. how it connects to the next stage
7. which failure modes it is designed to catch
8. how it contributes to the overall goal of improving data quality

That is more a README-writing requirement than a missing-facts problem, so we can address it in the final drafting stage.

### 6. What is the stagnation-breaking mechanism in the generator/critic loop?

Status: `ANSWERED`

Draft answer:
- In [cleaning/generation.py](cleaning/generation.py), stagnation is detected when either:
1. the generator returns the same code as the previous attempt
2. the host-side validation issues have the same fingerprint as the previous attempt

What happens next:
1. the next prompt receives an injected `_build_stagnation_unblock_brief(...)`
2. that brief forces a structural rewrite, especially:
   - a canonical early-exit guard for already-valid values
   - mutually exclusive, shape-specific parsing branches
3. the model temperature is increased conservatively using `_stagnation_temperature(...)`, with the ramp `0.2 -> 0.3 -> 0.4 -> 0.5`

Why I previously wrote that the repo does not prove this is the best strategy:
- The code clearly documents what the chosen strategy is.
- It does not contain an experiment comparing this strategy against alternatives.
- So the README can say "this is the implemented stagnation breaker" but should not say "this is empirically superior" unless we test alternatives.

Answer to your side question: yes, there are other ways around stagnation.

Common alternatives include:
1. changing the prompt template more aggressively after repeated failure
2. forcing a full branch rewrite instead of a patch-style retry
3. mutating the example set or ordering of examples
4. changing the critic style from diagnosis-only to stricter imperative repair instructions
5. using multiple candidate generations and selecting the best under host validation
6. changing the model, not just the temperature

What is good about the current choice:
- It is simple, cheap, and keeps the outer loop deterministic and auditable.
- It introduces just enough variation to escape repetition without giving the model too much freedom.

### 7. Does `CodeExecutionTool` mean the LLM can execute Python code in a sandbox?

Status: `ANSWERED`

Draft answer:
- Yes. In the current code, `CodeExecutionTool()` is enabled for two agents in [core/agents.py](core/agents.py):
1. `completeness_analysis_agent`
2. `column_cleaner_generator_agent`

More precise explanation:

`Completeness analysis`
- The completeness agent receives a deterministic completeness profile.
- It is instructed to use code execution to inspect that attached profile document before returning the final structured report.
- In practice, this means the model can use the sandbox to inspect the profile content rather than relying only on plain-text reading.

`Cleaner generation`
- The cleaner generator receives a `ColumnCleaningRequest`.
- It writes one Python function for one column.
- It is then allowed exactly one grouped sandbox execution per attempt, enforced by `GENERATOR_USAGE_LIMITS = UsageLimits(tool_calls_limit=1)` in [cleaning/generation.py](cleaning/generation.py).
- That single execution is used to test the generated function on:
  - already-valid dominant examples, which must be preserved
  - inconsistent examples, which must be transformed or nulled appropriately
- The model then reports its function plus an honest summary of whether that grouped test passed or still showed failures.

Critical clarification:
- The sandbox execution is not the final authority.
- The real acceptance decision is made afterward by the host-side validator in [cleaning/validation.py](cleaning/validation.py), which re-loads and re-tests the returned program under deterministic checks.
- So the model may self-test, but it does not self-certify.

### 8. What is the reproducible environment and install command?

Status: `ANSWERED`

Draft answer:
- The repo includes `requirements.txt`; there is no `environment.yml`.
- The current workspace and notebook traces indicate Windows development:
1. Windows-style paths appear throughout the notebooks.
2. The local interpreter here is `Python 3.13.2`.
- Minimal install commands from scratch:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Important clarification for the README:
- `main_mattia.ipynb` is an old notebook and should not be treated as the canonical execution path.
- We should avoid pushing a `graphviz` dependency as part of the normal project setup, because it requires an external system installation.
- If images or diagrams are needed for the README or notebook, they should be pre-generated and stored in an `images/` folder, then referenced as static assets.

### 8.b Why was Pydantic / PydanticAI chosen instead of another framework?

Status: `ANSWERED`

Draft answer:
- Pydantic was chosen because the core problem of this project is not only "using agents", but enforcing reliable structured handoffs between many stages.
- In this pipeline, almost every stage produces an artifact that becomes the input to the next one:
  - schema handoff
  - completeness report
  - consistency findings
  - remediation plan
  - cleaning request
  - generated cleaner program
  - verification report
  - final pipeline report
- Because of that, the team needed a framework centered on explicit data contracts rather than on free-form conversational orchestration.

Why Pydantic fits this project particularly well:
1. `Strong typed contracts`
   - [core/models.py](core/models.py) defines all the objects that move through the pipeline.
   - This makes every stage output parseable, inspectable, and serializable.
   - In a multi-agent workflow, this is crucial because a vague or malformed output from one stage can silently corrupt all later stages.

2. `Natural fit with PydanticAI`
   - [core/agents.py](core/agents.py) uses `PromptedOutput(...)` over Pydantic models.
   - This means the model is not just prompted with instructions; it is also constrained by an explicit schema.
   - That is a much better fit for this project than unconstrained text generation.

3. `Traceability and debugging`
   - Because every handoff is a model instance, artifacts can be cached, saved, reloaded, compared, and inspected independently.
   - This is one of the reasons the notebook and CLI can expose intermediate stages so clearly.

4. `Compatibility with deterministic enforcement`
   - The project architecture depends on deterministic host-side validation after every important LLM step.
   - Pydantic objects make it easy to combine agent outputs with Python-side checks without needing fragile text parsing.

5. `Practicality over heavier orchestration`
   - The repository and notes already suggest that more graph-like agent orchestration ideas were considered early on.
   - For this project, however, a large orchestration framework would have added complexity without solving the most important need, which was strict structured communication plus auditable stage-by-stage control.

How to frame the comparison in the README:
- The correct comparison is not "Pydantic is better than every other framework".
- The correct claim is: Pydantic was a better fit for this specific project because the project is artifact-driven, validation-heavy, and built around typed intermediate states.

Recommended final README wording:
- "We chose Pydantic, together with PydanticAI, because the project depends on strict structured handoffs between many stages. Unlike a looser agent framework centered mainly on message passing, this pipeline needs every intermediate artifact to be typed, validated, serializable, and easy to debug. Pydantic provided that contract layer directly, while still integrating naturally with LLM agents through schema-constrained outputs."

What not to overclaim:
- Avoid claiming a broad benchmark-based superiority over frameworks such as LangGraph or similar orchestration stacks unless you explicitly add a comparative implementation.
- The stronger and more defensible statement is that Pydantic matched the needs of this project better: structure, traceability, deterministic validation, and lightweight orchestration.

## Section 3 - Experimental Design

### 9. Were baselines considered?

Status: `FLAG`

Instruction from you:
- Wait for the implementation.

Current note for planning:
- Do not answer this in the final README yet.
- Leave space for a baseline subsection once the implementation exists.

### 10. What metrics were actually measured and recorded?

Status: `FLAG`

Instruction from you:
- Wait for the implementation.

Current note for planning:
- Do not finalize the experimental-metrics subsection yet.
- When implemented, this should be the place where token usage, cost, first-pass acceptance rate, retry counts, and verification summaries are collected into one reproducible table.

### 11. On how many distinct datasets was the pipeline tested, and how large are they?

Status: `FLAG`

Instruction from you:
- Wait for the implementation.

Current note for planning:
- We know the repo contains:
1. `Data/spesa.csv`: `7543 x 18`
2. `Data/attivazioniCessazioni.csv`: `20102 x 19`
- But the final README should wait until the final experimental run policy is settled.

## Section 4 - Results

### 12. Do we have concrete numbers from at least one full pipeline run?

Status: `FLAG`

Instruction from you:
- The final numeric results will be uploaded later.

What the README should do for now:
- explain very carefully how each result is obtained
- explain where each result is taken from
- avoid freezing placeholder numbers as final claims

How each result is produced in the current pipeline:

`Schema results`
- Produced by `run_schema_validation(...)` in [validation/schema.py](validation/schema.py).
- Derived from:
  - dtype inference
  - dataset profiling
  - naming-rule checks
  - duplicate-semantic grouping
- Saved as a `SchemaHandoff`.

`Completeness results`
- Produced by `run_completeness_analysis(...)` in [validation/completeness.py](validation/completeness.py).
- Built from the deterministic completeness profile in [tools/completeness_tools.py](tools/completeness_tools.py), then interpreted by the completeness agent.
- Report fields include overall completeness, missing-like counts, placeholder tokens, and sparse-column flags.

`Format-consistency results`
- Produced by `run_format_consistency_validation(...)` in [validation/consistency.py](validation/consistency.py).
- Built from [tools/format_tools.py](tools/format_tools.py), which profiles dominant shapes, outlier shapes, and representative examples.
- Each `FormatConsistencyFinding` identifies a column, a canonical expected pattern, a count of inconsistent rows, example bad values, and a suggested normalization strategy.

`Anomaly results`
- Produced by `run_anomaly_detection(...)` in [validation/anomaly.py](validation/anomaly.py).
- The actual anomaly detection is deterministic and comes from [tools/quality_tools.py](tools/quality_tools.py):
  - numeric outliers
  - rare categories
- The anomaly agent only summarizes the already computed findings.

`Cross-column results`
- Produced by `run_cross_column_validation(...)` in [validation/cross_column.py](validation/cross_column.py).
- The actual checks are deterministic and come from [tools/quality_tools.py](tools/quality_tools.py):
  - exact duplicate columns
  - near-duplicate columns
  - duplicate-semantic conflicts
  - year/month/period mismatches
  - date-order violations
- The cross-column agent only summarizes them.

`Duplicate-row results`
- Produced by `run_duplicate_detection(...)` in [validation/duplicates.py](validation/duplicates.py).
- The actual grouping logic is deterministic and comes from [tools/quality_tools.py](tools/quality_tools.py):
  - exact duplicate row groups
  - near-duplicate row groups
- The duplicate summary agent only narrates the output.

`Cleaner-generation results`
- Produced by `run_cleaner_generation(...)` in [cleaning/generation.py](cleaning/generation.py).
- Each format-consistency finding becomes one `ColumnCleaningRequest`.
- The generator writes one cleaner per target column, the critic diagnoses failures, and the host validator decides whether a cleaner is accepted.

`Application results`
- Produced by `run_cleaner_application_with_plan(...)` in [cleaning/application.py](cleaning/application.py).
- This is where accepted cleaners and deterministic remediation actions actually modify the dataset.

`Verification results`
- Produced by `run_verify(...)` in [cleaning/verification.py](cleaning/verification.py).
- This stage re-runs consistency on the cleaned CSV and compares "before" vs "after" findings.
- This is where the resolved / improved / unchanged / regressed statuses come from.

`Final report results`
- Produced by `build_final_report(...)` in [cleaning/reporting.py](cleaning/reporting.py).
- This is the canonical factual aggregation layer that merges validation, remediation, cleaning, and verification.

README guidance for now:
- In the current draft phase, focus on explaining where each result comes from and how it is computed.
- Insert final numeric results only after the final report artifacts are available.

### 13. What figure should replace Figure 1, and which script generates it?

Status: `PARTIAL`

Instruction from you:
- Images will eventually live in an image folder and be inserted as static assets.

Implication for the README:
- Do not describe Figure 1 as something that depends on Graphviz or an external runtime dependency.
- Instead, plan Figure 1 as a static pipeline image stored in the future image folder and referenced directly from the README.

Recommended wording for planning:
- "Figure 1 will be a static pipeline overview image stored in the project image folder and referenced from the README."

### 14. Should Table 1 be multi-dataset or single-run, and what columns should it have?

Status: `PARTIAL`

What Table 1 is:
- Table 1 should be the compact factual summary of one full pipeline run.
- It is not the raw logs, and it is not the narrative explanation.
- It is the structured, human-readable extraction of the final pipeline outcomes.

What it represents:
- The table should summarize the most important measurable outputs of the validation, cleaning, and verification stages for the dataset under discussion.
- In other words, it should answer: "What did the pipeline find, what did it act on, and what improved afterward?"

Where it is taken from:
- Its rows/columns should be derived from the structured artifacts produced by:
1. `validation_results`
2. `remediation_plan`
3. `cleaning_report`
4. `verification_report`
5. `final_report`

More concretely:
- The best canonical source is the `FinalPipelineReport` built by `build_final_report(...)` in [cleaning/reporting.py](cleaning/reporting.py), because that object already merges the outputs of all earlier stages into one factual summary layer.

Recommended interpretation for the future README:
- Table 1 is the run-summary table extracted from the final structured report, not a manually assembled anecdotal summary.

Suggested columns once results are finalized:
- Dataset
- Rows x Columns
- Schema Issues
- Columns With Missing-Like Values
- Format Findings
- Anomaly Findings
- Cross-Column Findings
- Duplicate Groups
- Cleaning Requests
- Accepted Cleaners
- First-Pass Accepted Cleaners
- Verification Resolved
- Verification Improved
- Verification Unchanged
- Verification Regressed

## Section 5 - Conclusions

### 15. Were any concrete failure modes observed?

Status: `ANSWERED`

Draft answer:
- Yes, and this section should be expanded in depth in the final README because it is one of the most convincing parts of the project.

Observed concrete failure modes:

1. `Canonical values being damaged by generic branches`
- A major failure mode was that already-valid values, especially ISO-like timestamps, could be accidentally rewritten by overly broad logic such as `if '-' in s:` before the cleaner first checked whether the input was already valid.
- This is why the current generator instructions insist on a canonical early-exit guard before any delimiter-based rewrite logic.

2. `Correct delimiter, wrong component order`
- Another real failure mode was not merely formatting noise, but semantic misassembly.
- Some generated cleaners kept the separators but emitted the date parts in the wrong order.
- Example: an input like `11/01/2024` could become `11-01-2024T...` instead of `2024-01-11T...`.
- This is important because it shows that the core risk was not cosmetic normalization, but incorrect transformation semantics.

3. `Recoverable period values being dropped`
- Period-like inputs such as `Rata 2024` are not fully specified, but they still contain recoverable information.
- Earlier logic risked dropping these to `None`.
- The pipeline now explicitly augments the `YYYYMM` strategy so year-only recoverable period values default to month `01`.
- This is a good example of a failure mode that pushed the design toward more conservative recovery-oriented cleaning.

4. `Non-self-contained generated code`
- A generated cleaner can look correct logically but still fail operationally if it references outer-scope or scratchpad variables.
- That is why [cleaning/validation.py](cleaning/validation.py) explicitly checks for `non_self_contained_function`.
- This failure mode is especially important in a code-generation pipeline because reproducibility depends not just on logic correctness, but also on executable isolation.

5. `Repeated failure loops`
- The generator can get stuck returning essentially the same wrong code or the same wrong failure family.
- This is why the project includes a stagnation detector, a repair critic, and a controlled temperature bump.
- The presence of this logic is itself evidence that repeated local deadlocks were observed as a practical issue during development.

6. `Placeholder-heavy and sparsity-heavy columns`
- Columns such as `note` and `fonte_dato` are examples of fields where the problem is not a few wrong values, but the fact that meaningful information is almost absent.
- This matters because not every data-quality problem should trigger active cleaning; some should trigger review or possible removal.

Why this matters for the conclusion:
- These are much stronger than generic statements like "domain drift" or "denser corruption".
- They show that the project encountered concrete, technically meaningful failure modes and adapted the architecture to handle them.

### 16. Is there a plan to open-source this, or is it course-deliverable only?

Status: `ANSWERED`

Draft answer:
- Open-sourcing is currently not in scope for the project.
- At the same time, it remains a possible future direction that may be considered later.

Recommended final README wording:
- "At present, the project is scoped as a course deliverable rather than as a public open-source release, although a broader release could be considered in future work."

## Recommended Notebook Additions Before Final README

1. `Baselines`
- Wait for the baseline implementation before drafting the experimental-comparison subsection.

2. `Metrics aggregation`
- Add one dataframe collecting validation counts, remediation counts, generation attempts, verification outcomes, token totals, and estimated cost.

3. `Prompt/token instrumentation`
- Add a cell that records the real number of examples sent to each agent family.
- Add a second cell that estimates tokens from the real prompt payloads.

4. `Final artifact persistence`
- Save the validation bundle, final report, and narrative report so the repo contains reproducible outputs, not only notebook traces.

5. `Static images folder`
- Store future figures as pre-generated static assets in an image folder and reference them from the notebook/README without introducing new runtime dependencies.
