"""Generator / critic repair loop for per-column cleaning functions.

The heart of the multi-agent cleaning pipeline. For each format-consistency
finding we:

1. Build a ``ColumnCleaningRequest`` bundle.
2. Prompt ``column_cleaner_generator_agent`` for a self-contained cleaner.
3. Validate the program host-side (``validate_generated_cleaner_program``).
4. On failure, ask ``cleaner_repair_critic_agent`` for a diagnosis and feed
   it back into the next generator attempt.
5. Detect stagnation (same code or same validation fingerprint) and inject
   a rewrite skeleton + bumped temperature to break deterministic loops.

Public entry points: ``run_cleaner_generation`` (driver) and
``run_column_cleaner_program`` (single-column loop).
"""

from __future__ import annotations

import sys

from agents import cleaner_repair_critic_agent, column_cleaner_generator_agent
from cache import load_schema_handoff
from models import (
    CleanerRepairContext,
    CleanerRepairDiagnosis,
    CleanerValidationIssue,
    ColumnCleanerProgram,
    ColumnCleaningRequest,
    GeneratedCleanerArtifact,
)
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
from pipeline import run_format_consistency_validation
from tools.tools import (
    attach_profile_text,
    attach_text_document,
    build_column_format_facts,
    load_dataset_frame,
    run_agent_with_backoff,
)

from .paths import cleaner_manifest_path, save_cleaner_manifest, save_generated_cleaner
from .request import build_column_cleaning_request
from .validation import (
    format_validation_examples,
    format_validation_issue,
    rebuild_verified_program,
    validate_generated_cleaner_program,
    validation_issue_fingerprint,
)


GENERATOR_USAGE_LIMITS = UsageLimits(tool_calls_limit=1)


def _print_example_transformations(program: ColumnCleanerProgram) -> None:
    print(f"\n  {'ORIGINAL':<35} {'CLEANED':<35} RATIONALE", file=sys.stderr)
    print(f"  {'-' * 35} {'-' * 35} {'-' * 30}", file=sys.stderr)
    for transformation in program.example_transformations:
        original = repr(transformation.original_value)[:33]
        cleaned = repr(transformation.cleaned_value)[:33]
        print(f"  {original:<35} {cleaned:<35} {transformation.rationale}", file=sys.stderr)

    if program.residual_risks:
        print("\n  Residual risks:", file=sys.stderr)
        for risk in program.residual_risks:
            print(f"    - {risk}", file=sys.stderr)


def run_cleaner_repair_critic(
    dataset_name: str,
    request: ColumnCleaningRequest,
    previous_program: ColumnCleanerProgram,
    validation_issues: list[CleanerValidationIssue],
) -> CleanerRepairDiagnosis:
    context = CleanerRepairContext(
        request=request,
        previous_program=previous_program,
        validation_issues=validation_issues,
    )
    prompt = [
        (
            f"Diagnose the failed cleaner for dataset {dataset_name}, column {request.column_name}. "
            "Use the structured repair context only. "
            "Return a precise repair diagnosis for the next generator attempt."
        ),
        attach_profile_text(context),
    ]
    return run_agent_with_backoff(cleaner_repair_critic_agent, prompt).output


def _build_stagnation_unblock_brief(request: ColumnCleaningRequest) -> str:
    """Concrete rewrite skeleton injected when the generator is repeating itself.

    Gives the model a non-negotiable structural template that puts canonical-value
    preservation BEFORE any delimiter-based branching, and enforces mutually
    exclusive branches so the shadowed-specific-branch check can never fire.
    """
    dominants = [v for v in request.dominant_example_values if isinstance(v, str) and v]
    outliers = [v for v in request.example_inconsistent_values if isinstance(v, str) and v]
    dominant_str = ", ".join(repr(v) for v in dominants[:6]) or "(no dominant examples)"
    outlier_str = ", ".join(repr(v) for v in outliers[:6]) or "(no outliers)"

    skeleton = (
        "def clean_column(value):\n"
        "    import re\n"
        "    from datetime import datetime\n"
        "    if value is None:\n"
        "        return None\n"
        "    s = str(value).strip()\n"
        "    if not s:\n"
        "        return None\n"
        "\n"
        "    # STEP 1 — CANONICAL GUARD. Build a structural regex from ONE dominant example\n"
        "    # and early-return if the input already matches. This MUST run before any\n"
        "    # delimiter-based branching so already-valid values are never rewritten.\n"
        "    canonical_examples = [" + ", ".join(repr(v) for v in dominants[:3]) + "]\n"
        "    def _structural_regex(example):\n"
        "        parts, cursor = [], 0\n"
        "        for m in re.finditer(r'\\d+', example):\n"
        "            a, b = m.span()\n"
        "            if a > cursor: parts.append(re.escape(example[cursor:a]))\n"
        "            parts.append(r'\\d{' + str(b - a) + '}')\n"
        "            cursor = b\n"
        "        if cursor < len(example): parts.append(re.escape(example[cursor:]))\n"
        "        return re.compile('^' + ''.join(parts) + '$')\n"
        "    canonical_patterns = [_structural_regex(e) for e in canonical_examples]\n"
        "    if any(p.fullmatch(s) for p in canonical_patterns):\n"
        "        return s\n"
        "\n"
        "    # STEP 2 — MUTUALLY EXCLUSIVE FORMAT BRANCHES. Order them MOST-specific first.\n"
        "    # Never write `if '-' in s:` above a branch that also inspects '-' substructure.\n"
        "    # Each branch must commit or fall through — no overlap.\n"
        "    # (Implement outlier transformations here.)\n"
        "\n"
        "    return None\n"
    )

    return (
        "STAGNATION OVERRIDE — your previous attempts repeated the same bug. "
        "You MUST abandon the previous control flow and rewrite around this skeleton. "
        "The structural canonical-guard (Step 1) is MANDATORY and MUST be the first logic after the None/empty check. "
        "Every dominant example must fall through the guard and return unchanged. "
        "Delimiter branches in Step 2 must be mutually exclusive — do not write a generic `if '<sep>' in s:` above any branch that re-inspects the same separator.\n\n"
        f"Dominant examples (must be preserved EXACTLY): {dominant_str}\n"
        f"Outlier examples (must be transformed): {outlier_str}\n\n"
        f"Target dtype: {request.target_dtype}\n"
        f"Expected pattern: {request.expected_pattern}\n\n"
        "REQUIRED SKELETON (adapt the function name and Step 2 body, but keep Step 1 verbatim):\n"
        "```python\n"
        f"{skeleton}"
        "```\n"
    )


def _build_cleaner_generation_prompt(
    dataset_name: str,
    request: ColumnCleaningRequest,
    previous_program: ColumnCleanerProgram | None = None,
    validation_issues: list[CleanerValidationIssue] | None = None,
    repair_diagnosis: CleanerRepairDiagnosis | None = None,
    attempt_number: int | None = None,
    stagnation_detected: bool = False,
) -> list[object]:
    prompt: list[object] = [
        (
            f"Generate and verify a pure Python cleaning function for dataset {dataset_name}, column {request.column_name}. "
            "Use the request document only. "
            "Your final python_code must be self-contained and executable in a fresh module with no external variables. "
            "Run exactly one grouped code-execution check against ALL provided example values before answering. "
            "Do not repair and re-run inside this model call; the host validator and critic own retries."
        ),
        attach_profile_text(request),
    ]

    if previous_program is not None and validation_issues:
        error_lines = "\n".join(f"- {format_validation_issue(issue)}" for issue in validation_issues[:20])
        concrete_examples = format_validation_examples(validation_issues, limit=8)
        has_self_containment_failure = any(
            issue.category == "non_self_contained_function" for issue in validation_issues
        )
        prompt.extend(
            [
                (
                    f"The previous cleaner attempt FAILED host-side validation"
                    + (f" on repair attempt {attempt_number - 1}" if attempt_number and attempt_number > 1 else "")
                    + ". "
                    "Repair the function so it passes every failing case below. "
                    "Use the prior failures to produce the next best program, then run exactly one grouped code-execution check. "
                    "Patch the previous function directly; do not rebuild the solution from scratch unless the failures prove the entire approach is wrong. "
                    "Do not read uploaded files or reconstruct context from attachments during repair. "
                    "If failures remain after the single grouped check, return the best current program and report those failures honestly. "
                    "The host-side validator will decide whether to call the critic again."
                ),
                attach_text_document(
                    (
                        "Priority note: the previous function was not self-contained. "
                        "Fix undefined-name or outer-scope-reference bugs before changing cleaning logic.\n\n"
                        if has_self_containment_failure
                        else ""
                    )
                    + "Host-side validation failures:\n"
                    f"{error_lines}\n\n"
                    "Concrete failing examples to fix first:\n"
                    f"{concrete_examples}\n\n"
                    "Previous function:\n"
                    f"{previous_program.python_code}"
                ),
            ]
        )
        if repair_diagnosis is not None:
            prompt.extend(
                [
                    (
                        "Use the attached critic diagnosis as the authoritative repair brief. "
                        "Follow its planned_fix, patch_style, and exact_repairs while preserving already-correct logic."
                    ),
                    attach_profile_text(repair_diagnosis),
                ]
            )

        if stagnation_detected:
            prompt.append(attach_text_document(_build_stagnation_unblock_brief(request)))

    return prompt


def run_column_cleaner_program(
    dataset_name: str,
    request: ColumnCleaningRequest,
    max_attempts: int = 10,
) -> ColumnCleanerProgram:
    previous_program: ColumnCleanerProgram | None = None
    validation_issues: list[CleanerValidationIssue] = []
    repair_diagnosis: CleanerRepairDiagnosis | None = None
    last_issue_fingerprint: tuple[str, ...] | None = None
    consecutive_stagnant_attempts = 0

    for attempt in range(1, max_attempts + 1):
        stagnation_detected = consecutive_stagnant_attempts >= 1
        print(
            f"[orchestrator][generator] column='{request.column_name}' attempt={attempt}/{max_attempts}"
            + (f" [STAGNATION OVERRIDE active, temp bumped]" if stagnation_detected else ""),
            file=sys.stderr,
        )
        prompt = _build_cleaner_generation_prompt(
            dataset_name,
            request,
            previous_program=previous_program,
            validation_issues=validation_issues,
            repair_diagnosis=repair_diagnosis,
            attempt_number=attempt,
            stagnation_detected=stagnation_detected,
        )
        try:
            # Escalate temperature to break deterministic repetition when the model keeps
            # emitting the same code. Step up further on longer stagnant streaks.
            override_settings: dict | None = None
            if stagnation_detected:
                bumped_temp = min(0.2 + 0.1 * (consecutive_stagnant_attempts - 1), 0.5)
                override_settings = {"temperature": bumped_temp}
            program = run_agent_with_backoff(
                column_cleaner_generator_agent,
                prompt,
                usage_limits=GENERATOR_USAGE_LIMITS,
                model_settings=override_settings,
            ).output
        except UsageLimitExceeded as error:
            raise ValueError(
                "Generator exceeded the one-code-execution limit. "
                "This prevents hidden self-repair loops; simplify the prompt or raise the explicit limit if needed."
            ) from error
        validation_issues = validate_generated_cleaner_program(request, program)
        if not validation_issues:
            program = rebuild_verified_program(request, program)
            print(
                f"[orchestrator][generator] column='{request.column_name}' accepted on attempt {attempt}",
                file=sys.stderr,
            )
            return program

        issue_fingerprint = validation_issue_fingerprint(validation_issues)
        same_code_as_previous = (
            previous_program is not None
            and program.python_code.strip() == previous_program.python_code.strip()
        )
        repeated_failure = last_issue_fingerprint is not None and issue_fingerprint == last_issue_fingerprint

        leading_issue = validation_issues[0]
        failure_label = (
            "construction-failure"
            if leading_issue.category == "non_self_contained_function"
            else "failed"
        )
        print(
            f"[orchestrator][validator] column='{request.column_name}' {failure_label} attempt {attempt}/{max_attempts}: "
            f"{leading_issue.message}",
            file=sys.stderr,
        )

        if same_code_as_previous or repeated_failure:
            reason = "repeated the same code" if same_code_as_previous else "repeated the same host-side failures"
            print(
                f"[orchestrator][validator] column='{request.column_name}' no-progress warning on attempt {attempt}/{max_attempts}: model {reason}; continuing until retry budget is exhausted unless the critic stops the run",
                file=sys.stderr,
            )
            consecutive_stagnant_attempts += 1
        else:
            consecutive_stagnant_attempts = 0

        previous_program = program
        last_issue_fingerprint = issue_fingerprint

        if attempt < max_attempts:
            print(
                f"[orchestrator][critic] column='{request.column_name}' reviewing {len(validation_issues)} validation issues",
                file=sys.stderr,
            )
            repair_diagnosis = run_cleaner_repair_critic(dataset_name, request, program, validation_issues)
            print(
                f"[orchestrator][critic] column='{request.column_name}' diagnosis: {repair_diagnosis.root_cause}",
                file=sys.stderr,
            )
            for repair in repair_diagnosis.exact_repairs[:3]:
                actual = f", actual={repair.actual_output!r}" if repair.actual_output is not None else ""
                expected = f", expected={repair.expected_output!r}" if repair.expected_output is not None else ""
                print(
                    f"[orchestrator][critic] column='{request.column_name}' repair-example: "
                    f"input={repair.input_value!r}{actual}{expected} | {repair.fix_note}",
                    file=sys.stderr,
                )
            if not repair_diagnosis.should_retry:
                failure_lines = "\n".join(f"- {format_validation_issue(issue)}" for issue in validation_issues[:10])
                raise ValueError(
                    f"Cleaner generation stopped for column '{request.column_name}' because the critic advised against retrying: "
                    f"{repair_diagnosis.root_cause}\n{failure_lines}"
                )

    failure_lines = "\n".join(f"- {format_validation_issue(issue)}" for issue in validation_issues[:10])
    raise ValueError(
        f"Cleaner generation failed local validation for column '{request.column_name}' after {max_attempts} attempts:\n"
        f"{failure_lines}"
    )


def run_cleaner_generation(
    path,
    reuse_consistency: bool = False,
    column_name: str | None = None,
    max_attempts: int = 10,
) -> list[GeneratedCleanerArtifact]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    consistency = run_format_consistency_validation(path, reuse_cache=reuse_consistency)
    df = load_dataset_frame(path)
    artifacts: list[GeneratedCleanerArtifact] = []

    schema_map = {}
    try:
        handoff = load_schema_handoff(path)
        schema_map = {column.name: column for column in handoff.columns}
    except FileNotFoundError:
        pass

    requested_column_name = column_name
    findings = consistency.format_consistency_findings
    if requested_column_name is not None:
        findings = [finding for finding in findings if finding.column_name == requested_column_name]
        if not findings:
            available = ", ".join(finding.column_name for finding in consistency.format_consistency_findings)
            raise ValueError(
                f"Column {requested_column_name!r} not found among format-consistency findings for {path.stem!r}. "
                f"Available columns: {available}"
            )

    failed_columns: list[str] = []
    for finding in findings:
        column_name = finding.column_name
        format_facts = build_column_format_facts(df, column_name)
        schema_entry = schema_map.get(column_name)
        request = build_column_cleaning_request(path.stem, column_name, finding, format_facts, schema_entry)

        print(
            f"\n[generate] '{column_name}' - {len(request.example_inconsistent_values)} outlier examples -> sandbox...",
            file=sys.stderr,
        )
        try:
            program = run_column_cleaner_program(path.stem, request, max_attempts=max_attempts)
        except ValueError as error:
            failed_columns.append(f"{column_name}: {error}")
            print(f"  FAILED - {error}", file=sys.stderr)
            continue
        code_path = save_generated_cleaner(path, program)

        print(f"  saved: {code_path}", file=sys.stderr)
        print(f"  sandbox validation ({len(program.example_transformations)} transformations):", file=sys.stderr)
        _print_example_transformations(program)

        artifacts.append(
            GeneratedCleanerArtifact(
                column_name=column_name,
                function_name=program.function_name,
                code_path=str(code_path),
                changed_rows=0,
                summary=program.verification_summary,
            )
        )

    if failed_columns and requested_column_name is not None:
        raise ValueError(f"Cleaner generation failed for requested column {requested_column_name!r}: {failed_columns[0]}")
    if failed_columns and not artifacts:
        joined_failures = "\n".join(f"- {failure}" for failure in failed_columns)
        raise ValueError(
            "Cleaner generation failed for every format-consistency finding; "
            "the previous cleaner manifest was left unchanged.\n"
            f"{joined_failures}"
        )

    save_cleaner_manifest(path, artifacts)
    print(f"\n[generate] manifest saved -> {cleaner_manifest_path(path)}", file=sys.stderr)
    return artifacts
