"""agents.py: all Pydantic AI Agent instances used by the pipeline.

Each Agent bundles the model identifier, the output type (a model from models.py that
defines the expected JSON structure), retry settings, temperature, and the system prompt
via instructions=(...). Two agents also enable the CodeExecutionTool so the LLM can run
and test its own output before returning it.

All agents share the MODEL constant defined at the top of this file. Every agent call in
the pipeline goes through run_agent_with_backoff() in tools/, which handles rate-limit
retries transparently.
"""

from __future__ import annotations
import os
from pathlib import Path
import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent, CodeExecutionTool, PromptedOutput

# Import structured output schemas
from src.core.models import (
    AnomalySummaryOutput,
    CleanerRepairDiagnosis,
    ColumnConsistencyReport,
    ColumnCleanerProgram,
    CompletenessAnalysisReport,
    CrossColumnSummaryOutput,
    DatasetDtypeInference,
    DuplicateSummaryOutput,
    NarrativeReportSection,
    NarrativeFrontMatter,
    SchemaSummaryOutput,
)

# Load environment variables
load_dotenv()

# Define the model to use for all agents
MODEL = "openai-responses:gpt-5.4-mini"
#MODEL = "openai-responses:gpt-4o-mini"
#MODEL = "openai-responses:gpt-5.4-mini"

MODEL = "openai-responses:gpt-5.4-nano"


def setup_logfire() -> None:
    logfire_token = os.getenv("LOGFIRE_TOKEN")
    send_to_logfire = (
        False
        if os.getenv("LOGFIRE_SEND") == "0"
        else "if-token-present"
    )
    repo_root = Path(__file__).resolve().parents[2]
    logfire.configure(
        data_dir=repo_root / ".logfire",
        service_name="pydantic-dataset-smoke-test",
        service_version="1.0.0",
        environment=os.getenv("LOGFIRE_ENVIRONMENT", "dev"),
        send_to_logfire=send_to_logfire,
        token=logfire_token,
    )
    logfire.instrument_pydantic_ai()
    if os.getenv("LOGFIRE_CAPTURE_HTTPX") == "1":
        logfire.instrument_httpx(capture_all=True)


schema_summary_agent = Agent(
    MODEL,
    name="schema-summary",
    output_type=PromptedOutput(SchemaSummaryOutput),
    retries=4,
    model_settings={"temperature": 0},
    instructions=(
        "You are the Schema Summary agent from the project orchestration. "
        "Inspect the attached local schema facts document. "
        "Return valid JSON only that matches the SchemaSummaryOutput schema exactly. "
        "Do not use markdown or ask follow-up questions. "
        "Execute only the schema-summary scope from Reply_projects.pdf. "
        "Do not infer new facts and do not alter the provided findings. "
        "Your only job is to write a short, precise downstream handoff summary for later validation or cleaning agents. "
        "Use the provided local facts exactly as given. "
        "Mention: how many safe naming fixes were identified, whether any duplicate-semantic groups need review, "
        "and whether any genuine data-type contradictions need manual verification. "
        "If there are no duplicate-semantic groups or no data-type risks, say that clearly. "
        "Keep the summary concrete and grounded in the provided facts, not generic."
    ),
)


dtype_inference_agent = Agent(
    MODEL,
    name="dtype-inference",
    output_type=PromptedOutput(DatasetDtypeInference),
    retries=4,
    model_settings={"temperature": 0},
    instructions=(
        "You are a data type inference agent working on real-world dirty datasets provided by NoiPA. "
        "NoiPA is the digital platform of the Ministero dell'Economia e delle Finanze Italiane that manages salaries, timesheets, "
        "and tax/social security obligations for employees of the Italian Public Administration. "
        "It allows users to view payslips and annual tax certifications online, update personal information, and manage "
        "administrative and HR-related procedures.\n\n"

        "You receive a column-by-column profile containing: the column name, sample values, non-null counts, distinct counts, "
        "numeric_parse_pct, datetime_parse_pct, and related profiling evidence. "
        "Your task is to infer the TARGET CLEANED pandas dtype the column SHOULD HAVE after cleaning. "
        "You are NOT describing the raw dirty storage format. "
        "Treat placeholders, formatting noise, unit suffixes, mixed separators, mixed date formats, and a minority of corrupted values "
        "as corruption, not as evidence of the true type. "
        "Always ask yourself: 'If this column were cleaned correctly, what physical pandas dtype should it have?'\n\n"

        "STRICT DECISION PRIORITY:\n"
        "1. First determine the main dtype family from parse evidence and dominant sample pattern: numeric, datetime, boolean, or text.\n"
        "2. Treat minority dirty values, placeholders, and formatting noise as corruption.\n"
        "3. Only after choosing the dtype family, use the column name and semantic meaning to refine numeric_role, string_role, and detected_pattern.\n"
        "4. Never let the column name override strong parse evidence.\n"
        "5. Infer the cleaned target dtype, not the messy ingestion dtype.\n\n"

        "PARSE EVIDENCE STRENGTH:\n"
        "- numeric_parse_pct >= 80: strong evidence for numeric.\n"
        "- datetime_parse_pct >= 60: strong evidence for datetime.\n"
        "- 60 to 79 numeric_parse_pct: moderate numeric evidence; inspect the dominant sample pattern.\n"
        "- 40 to 59 datetime_parse_pct: moderate datetime evidence; inspect the dominant sample pattern.\n"
        "- Below these thresholds: rely more on dominant pattern and semantic meaning.\n\n"

        "HARD DTYPE GATES:\n"
        "- If numeric_parse_pct >= 80 and datetime_parse_pct < 20, you MUST choose Int64 or Float64. Do not choose string, boolean, or object in that case.\n"
        "- If datetime_parse_pct >= 60, default to datetime64[ns].\n"
        "- If numeric_parse_pct >= 80 and the dominant numeric values are whole numbers, choose Int64.\n"
        "- If numeric_parse_pct >= 80 and the dominant numeric values contain decimals, choose Float64.\n"
        "- Do NOT choose string only because raw values are stored as strings.\n"
        "- Do NOT choose string when a numeric or datetime family clearly dominates.\n"
        "- Use object only as a last resort when no single clean dtype family dominates.\n\n"

        "IMPORTANT SPECIAL RULES:\n"
        "- Numeric strings that resemble compact period or date-like encodings such as YYYYMM, YYYYWW, YYYYQ, or similar numeric period keys "
        "should still be typed as Int64 if the intended clean value is a numeric code/period key rather than a true date column.\n"
        "- Infer datetime64[ns] only when the intended clean meaning is an actual date, time, or timestamp field.\n"
        "- Codes made only of digits can still be Int64 if they are true numeric codes.\n"
        "- Use string instead of Int64 only when the values must be preserved as text exactly for business meaning, especially when letters are intrinsic to the code format.\n"
        "- Mixed formatting alone is not enough reason to use string or object.\n\n"

        "Choose the dtype only from: Int64, Float64, datetime64[ns], string, boolean, object.\n\n"

        "DTYPE RULES (physical, not logical):\n"
        "- Int64: the clean column stores whole numbers. Use this even if the numbers are identifiers, codes, flags, or period keys.\n"
        "- Float64: the clean column stores decimal numbers.\n"
        "- datetime64[ns]: the clean column stores actual dates, times, or timestamps.\n"
        "- string: the clean column stores text such as names, descriptions, textual labels, or alphanumeric identifiers containing letters as part of the real format.\n"
        "- boolean: the clean column stores true/false, yes/no, 0/1-style logical values whose intended clean meaning is binary.\n"
        "- object: use only when the column genuinely mixes incompatible clean value types with no dominant pattern.\n\n"

        "ROLE RULES:\n"
        "- numeric_role: set only when dtype is Int64 or Float64.\n"
        "  'measure' = a real quantity used arithmetically (price, count, amount, duration, salary, quantity).\n"
        "  'code' = a numeric identifier or bounded calendar/classification code not primarily used arithmetically "
        "(postal code, region code, month number, year code, period key).\n"
        "  'indicator' = a numeric flag or ordinal encoding (0/1 flag, ordered category, status encoding).\n"
        "- string_role: set only when dtype is string.\n"
        "  'identifier' = codes or IDs that must be preserved exactly as text.\n"
        "  'categorical' = bounded low-cardinality labels.\n"
        "  'name' = person, organization, or place names.\n"
        "  'free_text' = unstructured narrative, notes, comments, or descriptions.\n"
        "- If dtype is not numeric, numeric_role must be null.\n"
        "- If dtype is not string, string_role must be null.\n\n"

        "PATTERN RULE:\n"
        "- detected_pattern must describe the dominant clean VALUE FORMAT, not a generic statistical interpretation.\n"
        "- detected_pattern must name exactly ONE canonical target format. Never output unions such as 'month label / month number', 'A or B', or 'mixed ...'.\n"
        "- Prefer specific structural patterns over vague labels.\n"
        "- For bounded calendar-like numeric codes, use patterns such as 'month number (1-12)', '4-digit year', or 'YYYYMM'.\n"
        "- Use 'integer count' only for true count variables such as totals, volumes, or frequencies.\n"
        "- If the column is a numeric code with a recognizable domain pattern, prefer that code pattern over 'integer count'.\n"

        "RATIONALE RULE:\n"
        "- The rationale must explain the chosen dtype using the strongest evidence.\n"
        "- Explicitly mention which signal dominated: parse percentages, dominant sample pattern, or semantic meaning.\n"
        "- Explicitly say when minority dirty values were treated as corruption.\n"
        "- Keep the rationale concise, evidence-based, and generic.\n\n"

        "OUTPUT RULES:\n"
        "- Return one entry per column in the same order as the input.\n"
        "- Be conservative and consistent.\n"
        "- Do not invent information not supported by the profile.\n"
        "- If strong parse evidence exists, follow it unless there is clear evidence the clean values belong to another dtype family."
    ),
)


completeness_analysis_agent = Agent(
    MODEL,
    name="completeness-analysis",
    builtin_tools=[CodeExecutionTool()],
    output_type=PromptedOutput(CompletenessAnalysisReport),
    retries=4,
    model_settings={"temperature": 0},
    instructions=(
        "You are the Completeness Analysis agent from the project orchestration. "
        "Always use the code execution tool to inspect the attached completeness profile document. "
        "Return valid JSON only that matches the CompletenessAnalysisReport schema exactly. "
        "Do not use markdown or ask follow-up questions. "
        "Execute only the completeness-analysis scope from Reply_projects.pdf. "
        "Use the provided per-column completeness percentages, missing-like counts, missing-like percentages, placeholder examples, "
        "and overall completeness metrics from the attached document. "
        "Identify columns with missing values, placeholder tokens such as N/A, -, unknown, and empty strings, and flag sparse columns that are almost entirely empty. "
        "Be evidence-based and conservative. "
        "Do not invent row-level details that are not present in the profile. "
        "Set recommended_action to a concrete next step based on the evidence, not a generic label. "
        "If completeness_pct is 100 and missing_like_count is 0, recommended_action must be exactly 'No action needed'. "
        "If sparse_candidate is true, recommended_action should clearly say 'Investigate or consider removal due to sparsity'. "
        "If placeholder examples are present, recommend standardizing placeholder tokens and reviewing upstream data entry. "
        "If completeness is high but not perfect, recommend targeted review of missing or placeholder values in that column. "
        "Do not leave recommended_action empty. "
        "The summary should be a short downstream handoff in plain language: mention overall completeness, how many columns have missing values, "
        "which columns are the main sparse or review targets, and whether placeholder normalization should be part of later cleaning."
    ),
)


format_consistency_agent = Agent(
    MODEL,
    name="format-consistency",
    output_type=PromptedOutput(ColumnConsistencyReport),
    retries=4,
    model_settings={"temperature": 0},
    instructions=(
        "You are the column-level Format Consistency agent. "
        "You receive a ColumnFormatFacts document for one column and must decide whether a format inconsistency exists and, if so, describe it precisely for the downstream cleaning agent.\n\n"

        "DECISION RULES:\n"
        "- Return finding=null if machine_format_candidate is false, dominant_shape_pct is below 70%, or inconsistent_rows is 0.\n"
        "- Return finding=null for descriptive, free-text, name, note, or categorical columns â€” content variation is not a format issue.\n"
        "- Return finding=null if all value variation is explained by missing/placeholder values alone.\n"
        "- Only report a finding when there is a clear dominant format and a measurable set of outliers that a cleaning function could fix.\n\n"

        "WHEN YOU REPORT A FINDING:\n"
        "- expected_pattern: describe ONE canonical dominant target format only (e.g. 'YYYYMM', 'YYYY-MM', 'ISO timestamp YYYY-MM-DDTHH:MM:SS.ffffff', 'two-digit zero-padded month 01-12').\n"
        "- expected_pattern must never describe multiple acceptable formats. Do not use words like 'mixed', 'various', 'multiple', 'and', or 'or'.\n"
        "- Choose the single dominant already-valid pattern shown by dominant_example_values; outlier formats belong in suggested_strategy, not in expected_pattern.\n"
        "- Copy ALL values from inconsistent_examples verbatim into example_inconsistent_values â€” do not filter, deduplicate, or summarize. The cleaner needs the full set.\n"
        "- evidence: cite dominant_shape, dominant_shape_pct, inconsistent_rows, and the target dtype from the prompt context.\n"
        "- suggested_strategy: this is the most important field â€” the downstream cleaner reads it as its normalization contract. "
        "List every outlier shape group with 2-3 concrete examples and the exact transformation needed. "
        "Be specific: 'shape YYYY-MM (e.g. 2023-09): remove dash, concatenate to YYYYMM' is good. "
        "'normalize dates' is not acceptable. "
        "If the target dtype is Int64 or Float64, note that the output must be a numeric string with no unit or symbol.\n\n"

        "OUTPUT: valid JSON matching ColumnConsistencyReport. No markdown, no follow-up questions."
    ),
)


anomaly_summary_agent = Agent(
    MODEL,
    name="anomaly-summary",
    output_type=PromptedOutput(AnomalySummaryOutput),
    retries=4,
    model_settings={"temperature": 0},
    instructions=(
        "You are the Anomaly Detection summary agent from the project orchestration. "
        "Inspect the provided anomaly findings document and write a short, precise downstream summary. "
        "Return valid JSON only that matches the AnomalySummaryOutput schema exactly. "
        "Do not infer new anomalies, do not invent remediation beyond the provided findings, and do not use markdown. "
        "Mention which columns carry the most severe or highest-volume anomalies, distinguish numeric outliers, negative-value findings, and rare-category findings, "
        "and state clearly when no anomalies were found."
    ),
)


cross_column_summary_agent = Agent(
    MODEL,
    name="cross-column-summary",
    output_type=PromptedOutput(CrossColumnSummaryOutput),
    retries=4,
    model_settings={"temperature": 0},
    instructions=(
        "You are the Cross-Column Validation summary agent from the project orchestration. "
        "Inspect the provided cross-column findings document and write a short, concrete summary for downstream review. "
        "Return valid JSON only that matches the CrossColumnSummaryOutput schema exactly. "
        "Do not infer new checks or facts, and do not use markdown. "
        "Highlight the most severe conflicts, especially exact or near-duplicate columns, duplicate-semantic column disagreements, year-month-period mismatches, and date-order violations. "
        "If there are no cross-column findings, say that explicitly."
    ),
)


duplicate_summary_agent = Agent(
    MODEL,
    name="duplicate-summary",
    output_type=PromptedOutput(DuplicateSummaryOutput),
    retries=4,
    model_settings={"temperature": 0},
    instructions=(
        "You are the Duplicate Detection summary agent from the project orchestration. "
        "Inspect the provided duplicate-detection findings document and write a short, concrete summary. "
        "Return valid JSON only that matches the DuplicateSummaryOutput schema exactly. "
        "Do not infer new duplicates, do not use markdown, and do not suggest aggressive deletion without acknowledging uncertainty. "
        "Mention the volume of exact duplicates, whether any near-duplicate groups were found, and which inferred key columns drive the near-duplicate signals. "
        "If there are no duplicate groups, say that explicitly."
    ),
)


column_cleaner_generator_agent = Agent(
    MODEL,
    name="column-cleaner-generator",
    builtin_tools=[CodeExecutionTool()],
    output_type=PromptedOutput(ColumnCleanerProgram),
    retries=4,
    model_settings={"temperature": 0},
    instructions=(
        "You are the Column Cleaner Generator agent. "
        "Given a ColumnCleaningRequest, produce a verified Python cleaning function.\n\n"

        "STEPS:\n"
        "1. Read the request: expected_pattern, dominant_example_values, example_inconsistent_values, suggested_strategy, target_dtype.\n"
        "2. Write the cleaning function.\n"
        "3. Test it once using the mandatory grouped code template below.\n"
        "4. Return JSON output. If the grouped test failed, return the best current function and report the failures honestly.\n\n"

        "EXECUTION DISCIPLINE:\n"
        "- This agent is responsible for one draft-and-test attempt only.\n"
        "- The outer Python orchestration loop plus the critic agent is the ONLY repair loop.\n"
        "- Use code execution exactly once for one grouped test over ALL dominant and inconsistent examples.\n"
        "- Do not patch and re-run inside the same model run, even if the grouped test exposes an obvious bug.\n"
        "- Work in batches, not in one-value-at-a-time loops.\n"
        "- After the grouped test, stop testing and return JSON.\n"
        "- If failures remain, include them in verification_summary and residual_risks; the host-side validator will route them to the critic.\n"
        "- Do not keep checking equivalent values individually once the grouped test already showed the same failure family.\n"
        "- NEVER load request data from uploaded files, request_data variables, or external files. Copy literals into the code block exactly as instructed.\n"
        "- On repair attempts, NEVER read uploaded files to reconstruct context. The request, previous function, validation failures, and critic diagnosis already contain everything needed.\n"
        "- Use code execution to test the function you just wrote, not to inspect attachments or rebuild the prompt context.\n"
        "- FORBIDDEN: repeated micro-diagnoses of equivalent failing values, repeated rewrites of the same function, or a second code-execution call inside a single run.\n\n"

        "FUNCTION CONTRACT:\n"
        "- One pure Python function, fully self-contained (all imports and helpers inside).\n"
        "- The final python_code must run if pasted into a fresh Python file with no surrounding variables. Do not rely on outer-scope names, uploaded files, request_data, or globals defined elsewhere.\n"
        "- The final python_code must NEVER reference scratch variables from the testing block such as dominant, inconsistent, failed, request, request_data, previous_program, or validation_issues unless they are explicitly defined inside the function body.\n"
        "- Variables created in the one-shot code execution block are scratchpad-only and must not appear in the final returned function unless they are redefined inside that function.\n"
        "- Input: any scalar â€” str, int, float, None, NaN. Output: str or None only.\n"
        "- Return None only for missing/empty input or truly unrecoverable values.\n"
        "- Return the value unchanged if it already matches expected_pattern.\n"
        "- Every dominant_example_value is already valid. If your function changes even one dominant example, the function is invalid.\n"
        "- Treat dominant_example_values as evidence of the valid target format, not as an exact allowlist. "
        "For datetime values and fixed-structure string formats, prefer generic pass-through logic for already-valid values instead of checking membership in the exact examples. "
        "For Int64/Float64 targets, do NOT define validity from the width or shape of one dominant example; use expected_pattern and numeric validity instead.\n"
        "- Every value in example_inconsistent_values must be transformed or explicitly nulled â€” never returned as-is.\n"
        "- suggested_strategy is the authoritative contract â€” implement a handler for every shape group it lists, no exceptions.\n"
        "- Prefer recovery over None: strip prefixes, expand abbreviations, extract embedded numbers. "
        "If a value contains any recoverable information, return it transformed â€” not None.\n"
        "- For expected_pattern 'YYYYMM', if the input exposes a recoverable 4-digit year but omits month, default the month to '01' rather than returning None, unless the request explicitly says otherwise.\n"
        "- If a value is invalid but unrecoverable for the target pattern, return None instead of inventing a best-guess correction.\n"

        "OUTPUT FORMAT BY TARGET DTYPE:\n"
        "- datetime64[ns]: string matching the EXACT strftime format seen in dominant_example_values.\n"
        "- Int64 / Float64: numeric string only â€” no units, no symbols. Use zfill/format for zero-padded outputs.\n"
        "- string: clean text matching expected_pattern.\n"
        "Always verify your output against the true target contract before returning.\n"
        "For datetime values and fixed-structure string formats, match the canonical structure shown by dominant_example_values. "
        "For bounded numeric code patterns such as 'month number (1-12)', '4-digit year', or 'YYYYMM', follow the semantic rule in expected_pattern rather than copying the width of one dominant example.\n"
        "For datetime values, do not use brittle length-only guards such as len(s) == N to detect already-valid timestamps. "
        "Use separator structure, parsing, or exact re-rendering against the dominant examples.\n\n"

        "MANDATORY CANONICAL-VALUE EARLY-EXIT GUARD:\n"
        "The FIRST logical step after handling None/empty input MUST be an already-valid guard. "
        "For datetime columns and fixed-structure string formats, this should be a canonical-pattern early-exit that returns "
        "the value unchanged when it already matches the structural layout of a dominant_example_value. "
        "Build that guard by deriving a regex from one dominant example: keep literal separators, replace each digit run with "
        "\\d{N} where N is that run's length. If s.fullmatch(pattern) returns true, return s immediately â€” do not enter any "
        "delimiter-based branch after that point. "
        "For Int64/Float64 targets, do NOT build the already-valid guard from one dominant example or one dominant width. "
        "Use expected_pattern and numeric validity to decide whether a value is already valid. "
        "This guard discipline is NON-NEGOTIABLE for datetime columns where the dominant format contains delimiters that also appear in "
        "outlier formats (for example ISO '2024-03-11T02:01:04.421' vs Italian '11/03/2024' vs '11-03-2024'). Without the "
        "early-exit, a subsequent `if '-' in s:` branch will rewrite already-valid ISO values into gibberish.\n\n"
        "NUMERIC TARGET OVERRIDE:\n"
        "For Int64/Float64 targets, this already-valid rule does NOT mean 'same width as one dominant example = valid'. "
        "Never infer numeric validity from a single sample like '7'. "
        "Instead, implement the numeric rule from expected_pattern directly. "
        "Example: for 'month number (1-12)', accept only integers 1 through 12; preserve 10, 11, and 12 as two-digit outputs when they are the true month values; reject 0 and all out-of-range integers.\n\n"

        "MUTUALLY EXCLUSIVE BRANCHES:\n"
        "Delimiter-based branches must be mutually exclusive and ordered most-specific first. "
        "Never write `if '<sep>' in s:` above another branch that re-inspects the same separator via split() or a regex that "
        "includes that separator. If two branches could both fire, either merge them or gate the generic one on exact structure "
        "(count of separator occurrences AND digit-group shapes). The host validator rejects any program where a generic "
        "`'<sep>' in s` branch precedes a more specific branch for the same separator. "
        "For datetime/date cleaners, prefer shape-first `re.fullmatch(...)` branches for every source layout, or one consolidated "
        "`s.count(sep) == N` branch that handles all layouts for that separator internally. Do not scatter multiple top-level "
        "branches for the same delimiter.\n\n"

        "CODE EXECUTION â€” MANDATORY TEMPLATE (use this exact structure every time):\n"
        "```python\n"
        "# 1. Define test data as literals â€” NEVER use request_data or any external variable\n"
        "dominant = ['...', '...']       # copy exact values from the request\n"
        "inconsistent = ['...', '...']   # copy exact values from the request\n\n"
        "# 2. Define the function â€” all imports and helpers go inside\n"
        "def clean_COLUMN(value):\n"
        "    import re\n"
        "    if value is None or str(value).strip() == '':\n"
        "        return None\n"
        "    s = str(value).strip()\n\n"
        "    # First preserve already-valid values using structural patterns derived from dominant examples.\n"
        "    canonical_examples = ['...']  # copy dominant examples here as literals\n"
        "    def _structural_regex(example):\n"
        "        parts, cursor = [], 0\n"
        "        for match in re.finditer(r'\\d+', example):\n"
        "            start, end = match.span()\n"
        "            if start > cursor:\n"
        "                parts.append(re.escape(example[cursor:start]))\n"
        "            parts.append(r'\\d{' + str(end - start) + '}')\n"
        "            cursor = end\n"
        "        if cursor < len(example):\n"
        "            parts.append(re.escape(example[cursor:]))\n"
        "        return '^' + ''.join(parts) + '$'\n"
        "    canonical_patterns = [_structural_regex(e) for e in canonical_examples if e and e != '...']\n"
        "    if any(re.fullmatch(pattern, s) for pattern in canonical_patterns):\n"
        "        return s\n\n"
        "    # Then handle outlier formats as shape-specific branches. Avoid broad `if '-' in s:` / `if '/' in s:` guards.\n"
        "    m = re.fullmatch(r'([A-Za-z]{3})-(\\d{4})', s)\n"
        "    if m:\n"
        "        mon, year = m.groups()\n"
        "        # map month abbreviation and render target format\n"
        "        pass\n"
        "    m = re.fullmatch(r'(\\d{4})-(\\d{1,2})', s)\n"
        "    if m:\n"
        "        year, month = m.groups()\n"
        "        # render target format\n"
        "        pass\n"
        "    m = re.fullmatch(r'(\\d{1,2})/(\\d{4})', s)\n"
        "    if m:\n"
        "        month, year = m.groups()\n"
        "        # render target format\n"
        "        pass\n"
        "    return None\n\n"
        "# 3. Run and flag failures explicitly. Do not re-run inside this attempt.\n"
        "failed = []\n"
        "for v in dominant + inconsistent:\n"
        "    result = clean_COLUMN(v)\n"
        "    status = 'OK' if result is not None else 'FAIL(None)'\n"
        "    print(f'{status}  {repr(v):40} -> {repr(result)}')\n"
        "    if v in dominant and result != v:\n"
        "        failed.append(v)\n"
        "    elif result is None and v in inconsistent:\n"
        "        failed.append(v)\n"
        "    elif v in inconsistent and result == v:\n"
        "        failed.append(v)\n"
        "if failed:\n"
        "    print(f'\\nFAILED ({len(failed)}): {failed}')\n"
        "    print('Return the current best program; the host validator and critic will handle repair.')\n"
        "```\n\n"
        "ISOLATION: each execution block is a fresh environment â€” nothing from previous runs survives. "
        "You are limited to one code execution call, so include steps 1-3 in that single block.\n\n"

        "OUTPUT RULES:\n"
        "- Return valid JSON matching ColumnCleanerProgram exactly.\n"
        "- python_code must contain ONLY the function definition â€” no test code, no print statements, no variable assignments, no JSON.\n"
        "- verification_summary, example_transformations, and residual_risks are separate top-level fields â€” never embed them inside python_code.\n"
        "- verification_summary must be honest about whether the final grouped test passed or still had failures.\n"
        "- example_transformations must reflect actual code execution results, not hypothetical ones.\n"
        "- cleaned_value must be a string or null â€” never int or float.\n"
        "- No markdown, no follow-up questions."
    ),
)


cleaner_repair_critic_agent = Agent(
    MODEL,
    name="cleaner-repair-critic",
    output_type=PromptedOutput(CleanerRepairDiagnosis),
    retries=4,
    model_settings={"temperature": 0},
    instructions=(
        "You are the Column Cleaner Repair Critic. "
        "You receive a structured CleanerRepairContext containing the cleaning request, the previous generated function, "
        "and authoritative host-side validation issues. "
        "Your job is to diagnose the smallest credible repair before another generator attempt. "
        "Do not write code. Do not restate the whole prompt. Return valid JSON only.\n\n"

        "GOAL:\n"
        "- Explain why the previous cleaner failed.\n"
        "- Point to the logical bug location or branch responsible.\n"
        "- Give a precise repair brief that a generator can follow.\n"
        "- Prefer minimal_edit unless the validation issues clearly show the current approach is fundamentally wrong.\n\n"

        "DECISION RULES:\n"
        "- Treat host-side validation issues as ground truth.\n"
        "- If primary_category is non_self_contained_function, treat it as a code-construction/scoping failure, not a cleaning-rule failure.\n"
        "- If any dominant valid example was modified, prioritize that over outlier handling.\n"
        "- If the issues are localized to one guard, one branch, or one formatting decision, choose patch_style='minimal_edit'.\n"
        "- Use patch_style='targeted_rewrite' only when multiple failure categories show the function structure is wrong.\n"
        "- Set should_retry=false only when another retry is unlikely to help because the evidence is contradictory, missing, or the current request is underspecified.\n"
        "- For numeric measures, do not recommend fixed-width padding unless the request explicitly requires it.\n"
        "- For numeric codes and date/time patterns, structural consistency is important; mention that when relevant.\n\n"

        "COMPOSITE FAILURES â€” DO NOT FIXATE ON A SINGLE CATEGORY:\n"
        "When the issue list contains BOTH a 'shadowed_specific_branch' (structural/order bug) AND a 'dominant_value_modified' "
        "(behavioral bug), they are usually the same root cause: a generic delimiter branch appears before the canonical-value "
        "guard and rewrites valid inputs. In that case:\n"
        "  - root_cause MUST explicitly name BOTH: the missing/misplaced canonical early-exit AND the shadowed delimiter branch.\n"
        "  - planned_fix MUST prescribe TWO concrete structural changes, not just 'check valid format first':\n"
        "      1) insert (or move to the top) a structural regex guard derived from the dominant example that returns s unchanged on match;\n"
        "      2) reorder or merge delimiter branches so no generic `'<sep>' in s` branch precedes a more specific branch inspecting the same separator.\n"
        "  - patch_style should be 'targeted_rewrite' when both categories are present â€” a minimal edit is not sufficient.\n"
        "  - priority_issues must list the structural bug first, the behavioral bug second â€” they share one fix.\n"
        "When the previous critic attempt already gave advice that the generator ignored (you are seeing the same failure pair on a later attempt), escalate the wording: say 'the previous repair brief was not followed' and restate the required rewrite in imperative form.\n\n"

        "COMPONENT-ORDER REWRITES (not delimiter swaps):\n"
        "When the failing output has the correct delimiters but the wrong component order â€” e.g. input '11/01/2024' becoming "
        "'11-01-2024T00:00:00.000' when the expected canonical output is '2024-01-11T00:00:00.000' â€” the bug is that the generator "
        "is swapping the separator character on the raw string instead of parsing the components and reassembling them in the "
        "canonical order. DO NOT say 'change the output format to YYYY-MM-DD' â€” that phrasing is ambiguous and the generator will "
        "re-interpret it as another delimiter swap. Instead:\n"
        "  - root_cause MUST state: 'the branch emits the raw components in source order with a new delimiter instead of reordering them'.\n"
        "  - planned_fix MUST be prescriptive about parsing and reassembly, for example: "
        "'split the value into (day, month, year) for the DD/MM/YYYY branch, then emit f\"{year}-{month:0>2}-{day:0>2}T00:00:00.000\"; "
        "never apply str.replace(\"/\", \"-\") on the whole string'.\n"
        "  - exact_repairs MUST include a line for every distinct source layout (DD/MM/YYYY, YYYY/MM/DD, DD-MM-YY, DD.MM.YYYY, "
        "textual months, etc.) showing input â†’ expected_output and the explicit (year, month, day) assignment the generator must produce.\n"
        "  - patch_style='targeted_rewrite'. A minimal edit is insufficient because the problem is how components are assembled, not which character separates them.\n\n"

        "FIELD RULES:\n"
        "- primary_category: choose the most important validation category to fix first.\n"
        "- For non_self_contained_function, root_cause and bug_location should explicitly mention the undefined name or outer-scope dependency and tell the generator to inline or redefine that data inside the function.\n"
        "- root_cause: one concise diagnosis grounded in the issues and anchored in at least one concrete failing input/output pair when possible.\n"
        "- bug_location: describe the failing logical area as specifically as possible. Name the exact guard, branch, fallback path, or branch ordering mistake responsible, such as "
        "'digit-only early-exit regex derived from a dominant example', 'generic numeric passthrough branch after month parsing', "
        "'currency stripping branch', or 'already-valid timestamp guard before delimiter rewrite'. Do not use vague labels like 'format logic'.\n"
        "- planned_fix: concrete and operational, suitable for the next generator prompt; mention the exact transformation direction that should change when the issue is localized. "
        "When possible, prescribe the exact condition or branch rewrite needed, for example 'replace the one-digit structural early-exit with a semantic range check 1..12' or "
        "'remove the raw numeric passthrough fallback after month normalization'.\n"
        "- priority_issues: list 1-3 short issue summaries, most important first.\n"
        "- exact_repairs: provide 1-3 concrete repair examples. Each one should name the failing input, the wrong output if known, the correct output if it can be inferred, and a short note describing exactly what to change in the named bug_location.\n"
        "- When the correct output is inferable from the dominant examples or expected pattern, fill expected_output explicitly instead of leaving it null.\n"
        "- confidence: high only when the failing pattern is clear and the fix is localized.\n\n"

        "OUTPUT:\n"
        "- Return JSON matching CleanerRepairDiagnosis exactly.\n"
        "- No markdown, no code, no follow-up questions."
    ),
)

narrative_frontmatter_agent = Agent(
    MODEL,
    name="narrative-frontmatter",
    output_type=PromptedOutput(NarrativeFrontMatter),
    retries=4,
    model_settings={"temperature": 0.2},
    instructions=(
        "You write the front matter for the final quality report. "
        "Use only the attached briefing. Return valid JSON matching NarrativeFrontMatter exactly.\n\n"
        "RULES:\n"
        "- title must include the dataset name.\n"
        "- executive_summary must be 8-12 sentences in professional English.\n"
        "- recommendations must contain at least 3 concrete, prioritized actions in English.\n"
        "- Do not use backticks for ordinary labels, column names, percentages, or example values.\n"
        "- Do not invent facts not present in the briefing.\n"
        "- No markdown outside the JSON fields."
    ),
)


narrative_section_agent = Agent(
    MODEL,
    name="narrative-section",
    output_type=PromptedOutput(NarrativeReportSection),
    retries=4,
    model_settings={"temperature": 0.2},
    instructions=(
        "You write exactly one section body for the final dataset quality report. "
        "Use only the attached section briefing. Return valid JSON matching NarrativeReportSection exactly.\n\n"
        "RULES:\n"
        "- heading must exactly match the requested section heading.\n"
        "- body must be markdown-formatted prose in professional English.\n"
        "- body must be at least 150 words and grounded in the provided facts only.\n"
        "- Use tables or bullet lists when they help clarity, but keep everything inside the body field.\n"
        "- Do not use backticks for ordinary column names, labels, values, row counts, or percentages.\n"
        "- Round percentages to one decimal place unless the briefing explicitly requires a different precision.\n"
        "- Do not invent facts, counts, examples, or file paths.\n"
        "- No markdown outside the JSON fields."
    ),
)
