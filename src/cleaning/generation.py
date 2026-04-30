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

import asyncio
import os
import sys
from contextlib import contextmanager
from typing import Callable

from src.core.agents import cleaner_repair_critic_agent, column_cleaner_generator_agent
from src.core.cache import load_schema_handoff
from src.core.models import (
    CleanerRepairContext,
    CleanerRepairDiagnosis,
    CleanerValidationIssue,
    ColumnCleanerProgram,
    ColumnCleaningRequest,
    GeneratedCleanerArtifact,
)
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
from src.validation import run_format_consistency_validation
from src.tools import (
    attach_profile_text,
    attach_text_document,
    build_column_format_facts,
    load_dataset_frame,
    run_agent_with_backoff,
    run_agent_with_backoff_async,
)

from .paths import cleaner_manifest_path, cleaning_cache_dir, save_cleaner_manifest, save_generated_cleaner
from .request import build_column_cleaning_request
from .validation import (
    format_validation_examples,
    format_validation_issue,
    rebuild_verified_program,
    validate_generated_cleaner_program,
    validation_issue_fingerprint,
)


GENERATOR_USAGE_LIMITS = UsageLimits(tool_calls_limit=1)

ProgressCallback = Callable[[str], None]


def _emit(on_event: ProgressCallback | None, message: str) -> None:
    '''Sends a progress message to the optional callback if one was provided.'''
    #it recieves on_event: an optional callback function and message: a string to send
    # If there is no callback, there is nothing to send anywhere
    if on_event is None:
        return  #If there is no callback, the function stops immediately
    try:
        # Forward the message to the UI or caller-provided progress handler
        on_event(message)
    except Exception:
        # Progress reporting should never break the main cleaning workflow
        pass


def _stagnation_temperature(consecutive_stagnant_attempts: int) -> float:
    '''Returns a slightly higher model temperature when repeated attempts get stuck.'''
    # Increase the temperature gradually to encourage a different solution,
    # but cap it so the output does not become too random.
    return min(0.2 + 0.1 * (consecutive_stagnant_attempts - 1), 0.5)
    #If the model keeps repeating the same wrong solution, a higher temperature encourages more variation

class _GenerationProgress:
    """Renders generator/critic loop events. Default impl writes to stderr
    and forwards a short message to an optional UI callback (used by Streamlit).
    Subclass and override if a different emission strategy is needed.
    """

    def __init__(self, on_event: ProgressCallback | None = None) -> None:
        '''Stores the optional callback used to forward progress updates.'''
        #This saves the callback inside the object so other methods can use it later
        self._on_event = on_event

    def _forward(self, message: str) -> None:
        '''Forwards a short progress message through the shared emitter helper.'''
        _emit(self._on_event, message)
        #It avoids repeating _emit(self._on_event, ...) everywhere

    def attempt_start(self, column_name: str, attempt: int, total: int, stagnation: bool) -> None:
        '''Reports the start of one generator attempt for a specific column.'''
        # Print a detailed backend log line and also send a shorter UI-friendly message.
        print(
            #This builds the main log message with column name and attempt number
            f"[orchestrator][generator] column='{column_name}' attempt={attempt}/{total}"
            #If stagnation is active, add a note saying special retry mode is on
            + (" [STAGNATION OVERRIDE active, temp bumped]" if stagnation else ""),
            file=sys.stderr, #Send the log to standard error
        )
        # sends a shorter version of the progress message to the optional callback
        self._forward(
            f"  · attempt {attempt}/{total}" #This is the short UI-friendly message
            + (" (stagnation override)" if stagnation else "") #If stagnation is active, mention it here too
        )
    #It reports each new attempt both in backend logs and in the UI

    def generator_accepted(self, column_name: str, attempt: int) -> None:
        '''Reports that the generator produced a valid cleaner on this attempt.'''
        print(
            #This message says the cleaner was accepted
            f"[orchestrator][generator] column='{column_name}' accepted on attempt {attempt}",
            file=sys.stderr, #sends to standard error
        )
        #Send a short success message to the callback
        self._forward(f"  ✓ accepted on attempt {attempt}")

    def validation_failed(
        self,
        column_name: str,
        attempt: int,
        total: int,
        leading_issue: CleanerValidationIssue,
    ) -> None:
        '''Reports that validation failed and summarizes the first important issue.'''
        # Rename one category to make the logs easier to understand at a glance
        label = ( #starts deciding which label to use
            #Use this label if the problem is specifically that the function was built incorrectly
            "construction-failure" 
            #If the main issue is a self-containment problem, use "construction-failure"
            if leading_issue.category == "non_self_contained_function"
            #Otherwise use the more generic label "failed".
            else "failed"
        )
        #Print a detailed backend log
        # This prints which column failed, what kind of failure it was, and which attempt number
        print(
            f"[orchestrator][validator] column='{column_name}' {label} attempt {attempt}/{total}: "
            #This appends the issue message itself
            f"{leading_issue.message}",
            file=sys.stderr,
        )
        #Take only the first sentence of the issue message, limited to 120 characters. This creates a short readable reason
        short_reason = leading_issue.message.split(".")[0][:120]
        #Send a short failure message to the callback
        self._forward(f"  ✗ attempt {attempt} {label}: {short_reason}")
        #It reports the most important validation failure clearly without sending too much text to the UI

    def stagnation_noted(self, column_name: str, attempt: int, total: int, reason: str) -> None:
        '''Reports that the loop is repeating itself without making real progress.It ogs that the generator seems stuck '''
        print(
            #This identifies the attempt
            f"[orchestrator][validator] column='{column_name}' no-progress warning on attempt {attempt}/{total}: "
            #This explains why stagnation was detected and what will happen next
            f"model {reason}; continuing until retry budget is exhausted unless the critic stops the run",
            file=sys.stderr,
        )

    def critic_started(self, column_name: str, n_issues: int) -> None:
        '''Reports that the critic agent is about to review the validation failures.'''
        print(
            #This says how many issues the critic is reviewing
            f"[orchestrator][critic] column='{column_name}' reviewing {n_issues} validation issues",
            file=sys.stderr,
        )
        #Send a short message to the callback
        self._forward(f"  → critic reviewing {n_issues} issues")

    def critic_diagnosis(self, column_name: str, diagnosis: CleanerRepairDiagnosis) -> None:
        '''Logs the critic diagnosis and a few concrete repair examples.'''
        print(
            #This prints the root cause identified by the critic
            f"[orchestrator][critic] column='{column_name}' diagnosis: {diagnosis.root_cause}",
            file=sys.stderr,
        )
        # Show only the first few exact repair examples to keep the logs readable
        #Loop over at most the first 3 repair examples
        for repair in diagnosis.exact_repairs[:3]:
            #Build the actual= part only if an actual output exists
            actual = f", actual={repair.actual_output!r}" if repair.actual_output is not None else ""
            #Build the expected= part only if an expected output exists
            expected = f", expected={repair.expected_output!r}" if repair.expected_output is not None else ""
            print(
                f"[orchestrator][critic] column='{column_name}' repair-example: "
                #Include the input, optional actual/expected values, and the critic’s fix note
                f"input={repair.input_value!r}{actual}{expected} | {repair.fix_note}",
                file=sys.stderr,
            )
    #It gives concrete examples of what the critic wants changed


#This decorator turns the function into a context manager, so it can be used with with.
@contextmanager
def _generation_lock(path):
    '''Prevents two cleaner-generation runs from writing to the same dataset cache at once.'''
    # Build the cache directory and lock-file path for this dataset
    cache_dir = cleaning_cache_dir(path) #This gets the cleaning cache directory for the dataset
    cache_dir.mkdir(parents=True, exist_ok=True) #This creates the directory if it does not already exist
    lock_path = cache_dir / ".generate.lock" #This builds the path of the lock file
    try:
        # Create the lock file only if it does not already exist
        # O_EXCL makes this fail immediately when another process already owns the lock
        #So this ensures only one generation process can hold the lock
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        # Give a clear error so the user knows another generation run is active
        raise RuntimeError(
            #Explain which dataset is locked
            f"Another cleaner generation is already running for {path.stem!r} "
            #Explain what the user can do if the lock is stale
            f"(lock file: {lock_path}). If no other process is running, delete the lock file and retry."
        ) from error #Preserve the original exception as context
    #Once the lock is acquired, start the protected section
    try:
        # Write the current process ID into the lock file for debugging
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        yield #Pause here and let the with block run
    finally: #When the with block finishes, always run cleanup
        try:
            # Remove the lock file when the protected block finishes.
            lock_path.unlink()
        except FileNotFoundError:
            # If it is already gone, there is nothing else to clean up
            pass
    #It prevents two generation runs from writing conflicting files for the same dataset


def _print_example_transformations(program: ColumnCleanerProgram) -> None:
    '''Prints the verified example transformations and any remaining risks to stderr.'''
    # Print a simple table header for the before/after examples, with 3 columns and a separator line
    print(f"\n  {'ORIGINAL':<35} {'CLEANED':<35} RATIONALE", file=sys.stderr)
    print(f"  {'-' * 35} {'-' * 35} {'-' * 30}", file=sys.stderr)
    # Print each example transformation produced by host-side verification
    #Loop through all recorded example transformations
    for transformation in program.example_transformations:
        original = repr(transformation.original_value)[:33] #Take the original value, convert it to a printable representation, and shorten it
        cleaned = repr(transformation.cleaned_value)[:33] #do the same as above but for the cleaned value
        #Print one table row with original value, cleaned value, and rationale
        print(f"  {original:<35} {cleaned:<35} {transformation.rationale}", file=sys.stderr)

    # If the program still declares residual risks, print them after the table
    if program.residual_risks:
        print("\n  Residual risks:", file=sys.stderr)
        for risk in program.residual_risks:
            print(f"    - {risk}", file=sys.stderr)
    #It gives a readable summary of what the cleaner did during verification


def run_cleaner_repair_critic(
    dataset_name: str,
    request: ColumnCleaningRequest,
    previous_program: ColumnCleanerProgram,
    validation_issues: list[CleanerValidationIssue],
) -> CleanerRepairDiagnosis:
    '''This defines the synchronous function that calls the critic agent. It calls the critic agent to diagnose why the previous cleaner failed validation.'''
    # Bundle the request, failed program, and validation issues into one structured object
    context = CleanerRepairContext(
        request=request,
        previous_program=previous_program, #Include the failed cleaner
        validation_issues=validation_issues, #Include the validation failures
    )
    # Build a short instruction plus a structured profile document for the critic
    #This first prompt item is a plain instruction telling the critic what to do
    prompt = [
        (
            f"Diagnose the failed cleaner for dataset {dataset_name}, column {request.column_name}. "
            "Use the structured repair context only. "
            "Return a precise repair diagnosis for the next generator attempt."
        ),
        attach_profile_text(context), #This attaches the structured context as a profile/document
    ]
    # Run the critic synchronously and return its structured diagnosis
    return run_agent_with_backoff(cleaner_repair_critic_agent, prompt).output
#It asks the critic model to explain why the cleaner failed and how to fix it

async def run_cleaner_repair_critic_async(
    dataset_name: str,
    request: ColumnCleaningRequest,
    previous_program: ColumnCleanerProgram,
    validation_issues: list[CleanerValidationIssue],
) -> CleanerRepairDiagnosis:
    '''Async version of the critic call used in parallel generation mode (the async version of the previous function).'''
    # Create the same structured repair context used by the sync version
    context = CleanerRepairContext(
        request=request,
        previous_program=previous_program,
        validation_issues=validation_issues,
    )
    # Build the same critic prompt as the synchronous path
    prompt = [
        (
            f"Diagnose the failed cleaner for dataset {dataset_name}, column {request.column_name}. "
            "Use the structured repair context only. "
            "Return a precise repair diagnosis for the next generator attempt."
        ),
        attach_profile_text(context),
    ]
    # Run the critic asynchronously and return only its parsed output
    result = await run_agent_with_backoff_async(cleaner_repair_critic_agent, prompt)
    return result.output
#It allows critic calls in async / parallel workflows


def _build_stagnation_unblock_brief(request: ColumnCleaningRequest) -> str:
    '''Builds a stronger repair brief (prompt) used when the generator keeps repeating the same mistake.'''
    # Collect a few dominant and outlier examples to show the model what must be preserved
    # and what must be transformed in the next rewrite.
    dominants = [v for v in request.dominant_example_values if isinstance(v, str) and v]
    #Collect valid outlier examples
    outliers = [v for v in request.example_inconsistent_values if isinstance(v, str) and v]
    #Build a readable string with up to 6 dominant examples
    dominant_str = ", ".join(repr(v) for v in dominants[:6]) or "(no dominant examples)"
    #Build a readable string with up to 6 outlier examples
    outlier_str = ", ".join(repr(v) for v in outliers[:6]) or "(no outliers)"

    # Build a concrete code skeleton that forces safer control flow:
    # first preserve canonical values, then handle outlier formats in exclusive branches.
    #The skeleton forces the model to: 
    #return None for missing/empty values
    #preserve already-valid examples first
    #use a structural regex guard
    #avoid overly generic delimiter branches
    #write specific mutually exclusive branches
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
        "    # Prefer one re.fullmatch(...) branch per outlier shape, e.g. DD/MM/YYYY,\n"
        "    # YYYY/MM/DD, DD-MM-YY, textual months, currency-with-symbol, etc.\n"
        "    # Do not write broad top-level guards like `if '-' in s:` or `if '/' in s:`.\n"
        "    # If a generic delimiter fallback is truly needed, make it one consolidated\n"
        "    # s.count(sep) == N branch with all layouts for that separator handled inside.\n"
        "    # (Implement outlier transformations here.)\n"
        "\n"
        "    return None\n"
    )

    # Return a detailed instruction block that explains why the rewrite is required
    # and gives the model an explicit skeleton to follow.
    #This returns a long instruction block that:
    #says stagnation was detected
    #orders the model to abandon the previous control flow
    #explains what must be preserved
    #shows dominant and outlier examples
    #includes target dtype and expected pattern
    #includes the required code skeleton

    return (
        "STAGNATION OVERRIDE — your previous attempts repeated the same bug. "
        "You MUST abandon the previous control flow and rewrite around this skeleton. "
        "The structural canonical-guard (Step 1) is MANDATORY and MUST be the first logic after the None/empty check. "
        "Every dominant example must fall through the guard and return unchanged. "
        "Delimiter branches in Step 2 must be mutually exclusive. Prefer shape-specific `re.fullmatch(...)` branches. "
        "Do not write broad top-level guards like `if '<sep>' in s`; use `s.count(sep) == N` only when all layouts for that separator are handled inside the same branch.\n\n"
        f"Dominant examples (must be preserved EXACTLY): {dominant_str}\n"
        f"Outlier examples (must be transformed): {outlier_str}\n\n"
        f"Target dtype: {request.target_dtype}\n"
        f"Expected pattern: {request.expected_pattern}\n\n"
        "REQUIRED SKELETON (adapt the function name and Step 2 body, but keep Step 1 verbatim):\n"
        "```python\n"
        f"{skeleton}"
        "```\n"
    )
#When the generator keeps repeating the same mistake, this helper gives it a much stricter rewrite template

def _build_cleaner_generation_prompt(
    dataset_name: str,
    request: ColumnCleaningRequest,
    previous_program: ColumnCleanerProgram | None = None,
    validation_issues: list[CleanerValidationIssue] | None = None,
    repair_diagnosis: CleanerRepairDiagnosis | None = None,
    attempt_number: int | None = None,
    stagnation_detected: bool = False,
) -> list[object]:
    '''Builds the generator prompt, optionally enriched with failures, critic advice, and stagnation guidance.'''
    # Start with the base instruction and the structured cleaning request
    prompt: list[object] = [
        (
            #This first item is the main instruction for the generator
            f"Generate and verify a pure Python cleaning function for dataset {dataset_name}, column {request.column_name}. "
            "Use the request document only. "
            "Your final python_code must be self-contained and executable in a fresh module with no external variables. "
            "Run exactly one grouped code-execution check against ALL provided example values before answering. "
            "Do not repair and re-run inside this model call; the host validator and critic own retries."
        ),
        attach_profile_text(request), #This attaches the structured cleaning request
    ]

    # Numeric columns need a special prompt because dominant examples do not define
    # the full acceptance rule as strongly as they do for fixed-format strings.
    if request.target_dtype in {"Int64", "Float64"}:
        prompt.append(
            #This tells the model not to infer validity only from dominant example width
            attach_text_document(
                "Numeric target override:\n"
                "- Do not build the already-valid guard from one dominant example or one dominant width.\n"
                "- For numeric targets, decide validity from expected_pattern and numeric validity only.\n"
                "- Dominant examples illustrate valid outputs, but they do not define the full acceptance rule.\n"
                "- A numeric token that matches the width of a dominant example can still be invalid if it violates expected_pattern."
            )
        )

    # Add an extra rule set for columns that are specifically expected to contain month numbers
    if request.expected_pattern.strip().lower() == "month number (1-12)":
        prompt.append(
            #This explains valid month range, zero-padding normalization, and how to treat month names
            attach_text_document(
                "Bounded month-number contract:\n"
                "- The valid output domain is exactly the integers 1 through 12 rendered as strings.\n"
                "- Do not infer validity from the width of one dominant example. A dominant sample like '7' does not mean every one-digit token is valid.\n"
                "- Preserve the true month semantics: '10' -> '10', '11' -> '11', '12' -> '12'.\n"
                "- Normalize zero-padded valid months by removing the leading zero: '01' -> '1', ..., '09' -> '9'.\n"
                "- Reject out-of-range numeric months such as '0', negative numbers, and values above 12 by returning None.\n"
                "- Month names and abbreviations must map to their true month numbers, not to the last digit of that number."
            )
        )

    # If a previous attempt failed, include the exact failures and the previous code
    # so the next generation attempt can repair it more directly
    if previous_program is not None and validation_issues:
        #Build a readable list of up to 20 validation issue lines
        error_lines = "\n".join(f"- {format_validation_issue(issue)}" for issue in validation_issues[:20])
        #Build a shorter list of concrete example failures
        concrete_examples = format_validation_examples(validation_issues, limit=8)
        #Check whether one of the issues was a self-containment failure
        has_self_containment_failure = any(
            issue.category == "non_self_contained_function" for issue in validation_issues
        )
        #Add more prompt material for repair mode
        prompt.extend(
            [
                (
                    f"The previous cleaner attempt FAILED host-side validation"
                    + (f" on repair attempt {attempt_number - 1}" if attempt_number and attempt_number > 1 else "")
                    + ". "
                    "Repair the function so it passes every failing case below. "
                    "Use the prior failures to produce the next best program, then run exactly one grouped code-execution check. "
                    "Patch the previous function directly for localized errors; when the failure is structural branch ordering, shadowed delimiter logic, or repeated stagnation, rewrite the control flow around shape-specific branches. "
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
        # If the critic already diagnosed the failure, attach that diagnosis as a repair brief
        if repair_diagnosis is not None:
            prompt.extend(
                [
                    (
                        "Use the attached critic diagnosis as the authoritative repair brief. "
                        "Follow its bug_location, planned_fix, patch_style, and exact_repairs while preserving already-correct logic. "
                        "Rewrite the named failing branch or guard first; do not leave the criticized fallback path in place."
                    ),
                    attach_profile_text(repair_diagnosis),
                ]
            )

        # If the loop is stagnating, inject a stronger rewrite skeleton
        if stagnation_detected:
            prompt.append(attach_text_document(_build_stagnation_unblock_brief(request)))

    return prompt
#This is the main prompt-builder for the generator loop. It adapts the prompt depending on failures, critic advice, and stagnation

#This is the core synchronous loop for one column
def run_column_cleaner_program(
    dataset_name: str,
    request: ColumnCleaningRequest,
    column_values,
    max_attempts: int = 10,
    on_event: ProgressCallback | None = None,
) -> ColumnCleanerProgram:
    '''Runs the full generator, validator, and critic retry loop for one column.'''
    progress = _GenerationProgress(on_event)
    previous_program: ColumnCleanerProgram | None = None #At first there is no previous program
    validation_issues: list[CleanerValidationIssue] = [] #Start with no validation issues
    repair_diagnosis: CleanerRepairDiagnosis | None = None #At first there is no critic diagnosis
    last_fingerprint: tuple[str, ...] | None = None #At first there is no fingerprint of previous failures
    consecutive_stagnant = 0 #Start the stagnation counter at 0

    # Try up to max_attempts times to get a cleaner that passes host-side validation
    for attempt in range(1, max_attempts + 1):
        # Once the loop starts repeating, enable the stagnation override behavior
        stagnation = consecutive_stagnant >= 1
        #Report the start of this attempt
        progress.attempt_start(request.column_name, attempt, max_attempts, stagnation)

        # Build the prompt using the latest request, failures, and critic guidance
        prompt = _build_cleaner_generation_prompt(
            dataset_name, request,
            previous_program=previous_program,
            validation_issues=validation_issues,
            repair_diagnosis=repair_diagnosis,
            attempt_number=attempt,
            stagnation_detected=stagnation,
        )
        # When stagnation is detected, slightly increase temperature to encourage a different solution
        model_settings = {"temperature": _stagnation_temperature(consecutive_stagnant)} if stagnation else None
        try:
            # Ask the generator agent for one cleaner program
            program = run_agent_with_backoff(
                column_cleaner_generator_agent, prompt,
                usage_limits=GENERATOR_USAGE_LIMITS,
                model_settings=model_settings,
            ).output
        #If the generator exceeded the allowed single code-execution check, raise ValueError( ... ) from error. Raise a clear user-facing error
        except UsageLimitExceeded as error:
            # The prompt explicitly allows only one code-execution check inside the generator call
            raise ValueError(
                "Generator exceeded the one-code-execution limit. "
                "This prevents hidden self-repair loops; simplify the prompt or raise the explicit limit if needed."
            ) from error

        # Validate the generated cleaner locally without using another model
        validation_issues = validate_generated_cleaner_program(request, program, column_values=column_values)
        #if there are no validation issues report success
        if not validation_issues:
            progress.generator_accepted(request.column_name, attempt)
            # Attach verified example transformations before returning the accepted program
            return rebuild_verified_program(request, program)
        #If there are issues, report the failure
        progress.validation_failed(request.column_name, attempt, max_attempts, validation_issues[0])

        # Detect whether the model is stuck by comparing the code and the failure fingerprint
        #Build a compact fingerprint of the issues
        fingerprint = validation_issue_fingerprint(validation_issues)
        #Check whether the new code is exactly the same as the previous code
        same_code = previous_program is not None and program.python_code.strip() == previous_program.python_code.strip()
        #Check whether the new set of failures is exactly the same as last time
        repeated_failure = last_fingerprint is not None and fingerprint == last_fingerprint
        if same_code or repeated_failure:
            #Build a human-readable reason
            reason = "repeated the same code" if same_code else "repeated the same host-side failures"
            #Log stagnation
            progress.stagnation_noted(request.column_name, attempt, max_attempts, reason)
            #Increase the stagnation counter
            consecutive_stagnant += 1
        #if there is a progress reseat stagnation counter to 0
        else:
            consecutive_stagnant = 0
        #Store the current program and current failure fingerprint for the next attempt
        previous_program, last_fingerprint = program, fingerprint

        # If attempts remain, ask the critic for a structured repair diagnosis.
        if attempt < max_attempts:
            #Report that the critic is starting
            progress.critic_started(request.column_name, len(validation_issues))
            #Ask the critic to analyze the failed cleaner
            repair_diagnosis = run_cleaner_repair_critic(dataset_name, request, program, validation_issues)
            #Log the critic’s diagnosis
            progress.critic_diagnosis(request.column_name, repair_diagnosis)
            # Stop early if the critic says another retry is not worthwhile
            if not repair_diagnosis.should_retry:
                #Build a readable summary of the first validation issues
                failure_lines = "\n".join(f"- {format_validation_issue(issue)}" for issue in validation_issues[:10])
                #Stop the loop and raise an error explaining why the critic advised against retrying
                raise ValueError(
                    f"Cleaner generation stopped for column '{request.column_name}' because the critic advised against retrying: "
                    f"{repair_diagnosis.root_cause}\n{failure_lines}"
                )

    # If all attempts fail, raise one final error containing the main validation problems
    # Build a final failure summary
    failure_lines = "\n".join(f"- {format_validation_issue(issue)}" for issue in validation_issues[:10])
    #Raise an error saying all attempts failed
    raise ValueError(
        f"Cleaner generation failed local validation for column '{request.column_name}' after {max_attempts} attempts:\n"
        f"{failure_lines}"
    )

#This is the async twin of the previous function
#It has almost identical logic, but the model calls are awaited.
async def run_column_cleaner_program_async(
    dataset_name: str,
    request: ColumnCleaningRequest,
    column_values,
    max_attempts: int = 10,
    on_event: ProgressCallback | None = None,
) -> ColumnCleanerProgram:
    '''Async version of the single-column generation loop used for parallel workers.'''
    progress = _GenerationProgress(on_event)
    previous_program: ColumnCleanerProgram | None = None
    validation_issues: list[CleanerValidationIssue] = []
    repair_diagnosis: CleanerRepairDiagnosis | None = None
    last_fingerprint: tuple[str, ...] | None = None
    consecutive_stagnant = 0

    # The logic mirrors the synchronous version, but the model calls are awaited.
    for attempt in range(1, max_attempts + 1):
        stagnation = consecutive_stagnant >= 1
        progress.attempt_start(request.column_name, attempt, max_attempts, stagnation)

        prompt = _build_cleaner_generation_prompt(
            dataset_name, request,
            previous_program=previous_program,
            validation_issues=validation_issues,
            repair_diagnosis=repair_diagnosis,
            attempt_number=attempt,
            stagnation_detected=stagnation,
        )
        model_settings = {"temperature": _stagnation_temperature(consecutive_stagnant)} if stagnation else None
        try:
            # Run the generator asynchronously so multiple columns can be processed in parallel
            result = await run_agent_with_backoff_async(
                column_cleaner_generator_agent, prompt,
                usage_limits=GENERATOR_USAGE_LIMITS,
                model_settings=model_settings,
            )
            program = result.output #Extract the program
        except UsageLimitExceeded as error:
            raise ValueError(
                "Generator exceeded the one-code-execution limit. "
                "This prevents hidden self-repair loops; simplify the prompt or raise the explicit limit if needed."
            ) from error

        # Validate the generated program locally on the host.
        validation_issues = validate_generated_cleaner_program(request, program, column_values=column_values)
        if not validation_issues:
            progress.generator_accepted(request.column_name, attempt)
            return rebuild_verified_program(request, program)
        progress.validation_failed(request.column_name, attempt, max_attempts, validation_issues[0])

        # Track repeated code or repeated failures to detect stagnation.
        fingerprint = validation_issue_fingerprint(validation_issues)
        same_code = previous_program is not None and program.python_code.strip() == previous_program.python_code.strip()
        repeated_failure = last_fingerprint is not None and fingerprint == last_fingerprint
        if same_code or repeated_failure:
            reason = "repeated the same code" if same_code else "repeated the same host-side failures"
            progress.stagnation_noted(request.column_name, attempt, max_attempts, reason)
            consecutive_stagnant += 1
        else:
            consecutive_stagnant = 0
        previous_program, last_fingerprint = program, fingerprint

        # Ask the async critic for repair guidance before the next retry.
        if attempt < max_attempts:
            progress.critic_started(request.column_name, len(validation_issues))
            #The critic call is also async
            repair_diagnosis = await run_cleaner_repair_critic_async(
                dataset_name, request, program, validation_issues
            )
            progress.critic_diagnosis(request.column_name, repair_diagnosis)
            if not repair_diagnosis.should_retry:
                failure_lines = "\n".join(f"- {format_validation_issue(issue)}" for issue in validation_issues[:10])
                raise ValueError(
                    f"Cleaner generation stopped for column '{request.column_name}' because the critic advised against retrying: "
                    f"{repair_diagnosis.root_cause}\n{failure_lines}"
                )

    # If all retries are exhausted, raise the collected failure summary.
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
    max_workers: int = 1,
    on_event: ProgressCallback | None = None,
) -> list[GeneratedCleanerArtifact]:
    '''Runs cleaner generation for one or more columns, protected by a dataset-level lock.'''
    # Validate user-controlled settings before starting any work
    if max_attempts < 1: #Validate the retry count
        raise ValueError("max_attempts must be at least 1.") #Reject invalid retry counts
    if max_workers < 1: #Validate the worker count
        raise ValueError("max_workers must be at least 1.") #Reject invalid worker counts

    # Protect the whole generation process so two runs do not write conflicting files
    with _generation_lock(path): #Acquire the dataset-level lock in order to ensure exclusive access to cache
    #Run the real generation workflow while holding the lock
        return _run_cleaner_generation_locked(path, reuse_consistency, column_name, max_attempts, max_workers, on_event)
    #It validates inputs and ensures the whole generation run is protected by a lock

async def _run_cleaner_generation_requests_async(
    path,
    generation_requests: list[tuple[int, str, ColumnCleaningRequest, object]],
    total_columns: int,
    max_attempts: int,
    max_workers: int,
    on_event: ProgressCallback | None = None,
) -> tuple[list[tuple[int, str, ColumnCleanerProgram]], list[str]]:
    '''Runs multiple column-generation requests asynchronously and collects successes and failures.'''
    # Limit how many async generation tasks run at the same time.
    #Create a semaphore so only max_workers tasks can run at the same time
    semaphore = asyncio.Semaphore(max_workers)

    #Inside it, there is an inner async function:
    async def _run_one_cleaner(
        idx: int,
        request_column_name: str,
        request: ColumnCleaningRequest,
        column_values,
    ) -> tuple[int, str, ColumnCleanerProgram | None, ValueError | None]:
        '''Runs the async single-column loop for one request inside the shared semaphore. It uns generation for one column and returns either a program or an error'''
        #Only run if a worker slot is available
        async with semaphore:
            # Log a short progress line showing which column is entering the sandbox loop
            print(
                f"\n[generate] '{request_column_name}' - {len(request.example_inconsistent_values)} outlier examples -> sandbox...",
                file=sys.stderr,
            )
            # Send a shorter progress message to the callback
            _emit(
                on_event,
                f"[{idx}/{total_columns}] '{request_column_name}' - {len(request.example_inconsistent_values)} outlier examples",
            )
            try:
                # Generate and validate a cleaner for this single column
                program = await run_column_cleaner_program_async( #Run the full async single-column loop
                    path.stem, request, column_values, max_attempts=max_attempts, on_event=on_event
                )
            except ValueError as error:
                # Return the failure instead of raising so the other tasks can continue
                return idx, request_column_name, None, error
            #If successful, return the program
            return idx, request_column_name, program, None
    
    #Back in the outer function:
    # Create one async task per requested column.
    tasks = [
        asyncio.create_task(_run_one_cleaner(idx, request_column_name, request, column_values))
        for idx, request_column_name, request, column_values in generation_requests
    ]
    #Start a list for successful results and a list for failed columns
    completed_results: list[tuple[int, str, ColumnCleanerProgram]] = []
    failed_columns: list[str] = []
    # Consume the tasks as they finish so progress is reported incrementally
    for task in asyncio.as_completed(tasks):
        #Wait for one completed task and unpack its result
        idx, request_column_name, program, error = await task
        if error is not None:
            #Store the failure summary
            failed_columns.append(f"{request_column_name}: {error}")
            print(f"  FAILED - {error}", file=sys.stderr)
            #Send a short callback failure message
            _emit(on_event, f"  x '{request_column_name}' failed: {str(error)[:120]}")
            continue
        #Save the successful result
        if program is not None:
            completed_results.append((idx, request_column_name, program))
#Return both successes and failures
    return completed_results, failed_columns
#It handles parallel generation across columns while allowing some columns to fail without stopping all the others

def _run_cleaner_generation_locked(
    path,
    reuse_consistency: bool,
    column_name: str | None,
    max_attempts: int,
    max_workers: int,
    on_event: ProgressCallback | None = None,
) -> list[GeneratedCleanerArtifact]:
    '''Performs the full generation workflow once the dataset lock has been acquired.'''
    # Load consistency findings and the raw dataset data needed to build requests
    consistency = run_format_consistency_validation(path, reuse_cache=reuse_consistency)
    df = load_dataset_frame(path)
    #Start an empty list of generated artifacts
    artifacts: list[GeneratedCleanerArtifact] = []

    # Try to load schema metadata so dtype and role hints can be added to each request
    schema_map = {}
    try:
        handoff = load_schema_handoff(path)
        #Build a fast lookup map by column name
        schema_map = {column.name: column for column in handoff.columns}
    except FileNotFoundError:
        # Schema hints are optional, so continue even if the handoff file does not exist
        pass

    # If the user asked for a single column, filter the findings to that column only
    #Store the optional requested single-column name
    requested_column_name = column_name
    #Start from all consistency findings
    findings = consistency.format_consistency_findings
    #If the user requested one specific column, Filter findings to only that column
    if requested_column_name is not None:
        findings = [finding for finding in findings if finding.column_name == requested_column_name]
        #If the requested column was not found, build a list of available columns
        if not findings:
            available = ", ".join(finding.column_name for finding in consistency.format_consistency_findings)
            #Raise a helpful error
            raise ValueError(
                f"Column {requested_column_name!r} not found among format-consistency findings for {path.stem!r}. "
                f"Available columns: {available}"
            )
    #Start a list of failed columns
    failed_columns: list[str] = []
    #Count how many columns will be processed
    total_columns = len(findings)
    # Notify the caller how many columns will be processed.
    _emit(on_event, f"{total_columns} column{'s' if total_columns != 1 else ''} to synthesize")

    # If parallel work is enabled and there is more than one column, prepare all requests first
    if max_workers > 1 and len(findings) > 1:
        #Start building async generation requests for all columns before running any of them, so the async workers can consume them immediately as they finish.
        generation_requests: list[tuple[int, str, ColumnCleaningRequest, object]] = []
        #Loop through each finding
        for idx, finding in enumerate(findings, start=1):
            request_column_name = finding.column_name
            # Build the request bundle that the generator agent will use for this column
            format_facts = build_column_format_facts(df, request_column_name)
            #Try to get schema metadata for the column
            schema_entry = schema_map.get(request_column_name)
            #Build the structured cleaning request
            request = build_column_cleaning_request(
                path.stem,
                request_column_name,
                finding,
                format_facts,
                schema_entry,
            )
            #Save the request
            generation_requests.append((idx, request_column_name, request, df[request_column_name]))

        # Do not create more workers than there are actual requests
        worker_count = min(max_workers, len(generation_requests))
        print(
            f"\n[generate] running {len(generation_requests)} cleaner generators with {worker_count} async workers",
            file=sys.stderr,
        )
        _emit(on_event, f"running cleaner generators with {worker_count} async workers")
        # Run the async batch and collect both successes and failures.
        completed_results, parallel_failed_columns = asyncio.run(
            _run_cleaner_generation_requests_async(
                path,
                generation_requests,
                total_columns,
                max_attempts,
                worker_count,
                on_event=on_event,
            )
        )
        failed_columns.extend(parallel_failed_columns)

        # Loop through successful results, sorted back into original column order
        for _idx, generated_column_name, program in sorted(completed_results, key=lambda item: item[0]):
            #Save the generated cleaner code to disk
            code_path = save_generated_cleaner(path, program)
            #Log where it was saved
            print(f"  saved: {code_path}", file=sys.stderr)
            #Log how many example transformations were verified
            print(f"  sandbox validation ({len(program.example_transformations)} transformations):", file=sys.stderr)
            #Print the transformation tabl
            _print_example_transformations(program)
            _emit(
                on_event,
                f"  saved cleaner ({len(program.example_transformations)} sandbox transformations verified)",
            )
            #Create and store an artifact describing the generated cleaner
            artifacts.append(
                GeneratedCleanerArtifact(
                    column_name=generated_column_name,
                    function_name=program.function_name,
                    code_path=str(code_path),
                    changed_rows=0,
                    summary=program.verification_summary,
                    example_transformations=list(program.example_transformations),
                )
            )

        # Clear findings so the sequential loop below does not re-run the same columns
        findings = []

    # Process any remaining columns sequentially
    for idx, finding in enumerate(findings, start=1):
        column_name = finding.column_name
        # Build the request bundle for this column using format facts and optional schema hints
        format_facts = build_column_format_facts(df, column_name)
        schema_entry = schema_map.get(column_name)
        request = build_column_cleaning_request(path.stem, column_name, finding, format_facts, schema_entry)

        #Log that generation for this column is starting
        print(
            f"\n[generate] '{column_name}' - {len(request.example_inconsistent_values)} outlier examples -> sandbox...",
            file=sys.stderr,
        )
        #Send short progress update
        _emit(
            on_event,
            f"[{idx}/{total_columns}] '{column_name}' — {len(request.example_inconsistent_values)} outlier examples",
        )
        try: #Try generating the cleane
            # Run the full single-column generator / validator / critic loop
            #Run the full synchronous single-column generation loop
            program = run_column_cleaner_program(
                path.stem, request, df[column_name], max_attempts=max_attempts, on_event=on_event
            )
            #If generation fails, save the failure
        except ValueError as error:
            # Keep the failure and continue with the remaining columns
            failed_columns.append(f"{column_name}: {error}")
            print(f"  FAILED - {error}", file=sys.stderr)
            _emit(on_event, f"  ✗ '{column_name}' failed: {str(error)[:120]}")
            continue
        # Save the verified cleaner program (code) to disk
        code_path = save_generated_cleaner(path, program)

        print(f"  saved: {code_path}", file=sys.stderr)
        print(f"  sandbox validation ({len(program.example_transformations)} transformations):", file=sys.stderr)
        _print_example_transformations(program)
        _emit(
            on_event,
            f"  saved cleaner ({len(program.example_transformations)} sandbox transformations verified)",
        )
        #Create and store the artifact
        artifacts.append(
            GeneratedCleanerArtifact(
                column_name=column_name,
                function_name=program.function_name,
                code_path=str(code_path),
                changed_rows=0,
                summary=program.verification_summary,
                example_transformations=list(program.example_transformations),
                )
            )

    # If the user requested one specific column and it failed, raise immediately with that failure
    if failed_columns and requested_column_name is not None:
        raise ValueError(f"Cleaner generation failed for requested column {requested_column_name!r}: {failed_columns[0]}")
    # If everything failed and no artifact was created, raise a combined failure summary
    if failed_columns and not artifacts:
        joined_failures = "\n".join(f"- {failure}" for failure in failed_columns)
        raise ValueError(
            "Cleaner generation failed for every format-consistency finding; "
            "the previous cleaner manifest was left unchanged.\n"
            f"{joined_failures}"
        )

    # Save the manifest describing all generated cleaner artifacts
    save_cleaner_manifest(path, artifacts)
    print(f"\n[generate] manifest saved -> {cleaner_manifest_path(path)}", file=sys.stderr)
    return artifacts
#This is the top-level internal workflow that loads validation findings, builds requests, runs generation, saves results, and handles both async and sequential execution
