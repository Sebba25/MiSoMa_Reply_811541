"""Cross-cutting helpers shared by the specialised ``tools/*`` modules.

This module centralizes four kinds of reusable primitives:
    * placeholder-token and attachment helpers for agent prompts
    * numeric/datetime parse-rate and value-shape utilities used by profiling
    * schema-pattern matchers used by the cleaner validator
    * retrying agent runners plus verbose terminal tracing for pydantic-ai calls

Nothing here contains stage-specific business logic; it exists to keep the
specialized tool modules small and consistent.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import os
from pathlib import Path
import re
import sys
import time

import pandas as pd
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    BinaryContent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
)
from pydantic_ai.usage import UsageLimits

try:
    from pydantic_ai.exceptions import ModelAPIError
except ImportError:  # pydantic-ai < 1.30
    ModelAPIError = RuntimeError

try:
    from pydantic_ai.messages import PartEndEvent
except ImportError:  # pydantic-ai 0.8.x does not expose this stream event
    PartEndEvent = ()

# Shared Constants

PLACEHOLDER_TOKENS = ("", "na", "n/a", "null", "none", "-", "--", "unknown", "n.d.", "?", "//")


# Shared Attachment And I/O Helpers

def attach_text_document(text: str) -> BinaryContent:
    """Wrap plain text as a BinaryContent document for pydantic-ai prompts."""
    return BinaryContent(data=text.encode("utf-8"), media_type="text/plain")


def attach_profile_text(profile: BaseModel) -> BinaryContent:
    """Serialize a Pydantic model to JSON and attach it as a text document."""
    return attach_text_document(profile.model_dump_json(indent=2))


def load_dataset_frame(path: Path, dtype: str | dict | None = None) -> pd.DataFrame:
    """Load one CSV dataset into a pandas DataFrame with an optional dtype override."""
    return pd.read_csv(path, dtype=dtype)


# Shared Encoding Helpers

def gzip_text_to_base64(text: str) -> str:
    """Compress text with gzip and return an ASCII-safe base64 string."""
    return base64.b64encode(gzip.compress(text.encode("utf-8"))).decode("ascii")


# Shared Profiling Helpers

def sample_non_null_values(series: pd.Series, limit: int = 5) -> list[str]:
    """Collect a few distinct rendered non-null values from a series."""
    values: list[str] = []
    for value in series.dropna():
        rendered = str(value).strip()
        # Skip blanks and preserve first-seen order so prompt examples stay intuitive.
        if not rendered or rendered in values:
            continue
        values.append(rendered[:80])
        if len(values) >= limit:
            break
    return values


def value_shape(value: str) -> str:
    """Convert a value into its structural shape signature.

    Digits become ``9``, letters become ``A``, and punctuation is preserved,
    so values with the same layout map to the same canonical shape.
    """
    parts: list[str] = []
    for char in value:
        if char.isdigit():
            parts.append("9")
        elif char.isalpha():
            parts.append("A")
        else:
            parts.append(char)
    return "".join(parts)


def compute_numeric_parse_pct(series: pd.Series) -> float:
    """Return the percentage of non-empty rendered values parseable as numeric."""
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    rendered = non_null.astype(str).str.strip()
    rendered = rendered[rendered != ""]
    if rendered.empty:
        return 0.0
    parsed = pd.to_numeric(rendered, errors="coerce")
    return float((parsed.notna().mean()) * 100)


def compute_datetime_parse_pct(series: pd.Series) -> float:
    """Return the percentage of rendered values that look and parse like datetimes.

    A lightweight signal mask is applied first so arbitrary text is not sent
    wholesale into pandas datetime parsing.
    """
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    rendered = non_null.astype(str).str.strip()
    rendered = rendered[rendered != ""]
    if rendered.empty:
        return 0.0

    # Pre-filter to strings with obvious date/time cues before asking pandas to parse them.
    signal_mask = (
        rendered.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", regex=True)
        | rendered.str.contains(r"\d{4}-\d{2}-\d{2}T\d", regex=True)
        | rendered.str.contains(r"[A-Za-z]{3,}\s+\d{1,2}\s+\d{2,4}", regex=True)
        | rendered.str.contains(r"\d{1,2}\s+[A-Za-z]{3,}\s+\d{2,4}", regex=True)
    )
    candidates = rendered[signal_mask]
    if candidates.empty:
        return 0.0

    parsed = pd.to_datetime(candidates, errors="coerce", format="mixed")
    return float((parsed.notna().sum() / len(rendered)) * 100)


def compute_empty_like_pct(series: pd.Series) -> float:
    """Return the share of rows that are null-like or placeholder-like."""
    if series.empty:
        return 0.0
    rendered = series.fillna("").astype(str).str.strip().str.lower()
    empty_like = rendered.isin(PLACEHOLDER_TOKENS)
    return float((empty_like.mean()) * 100)


def _normalized_pattern(pattern: str | None) -> str:
    """Normalize an optional pattern label for case-insensitive comparisons."""
    return (pattern or "").strip().lower()


def _matches_generic_digit_count(value: str, pattern: str) -> bool | None:
    """Match schema phrases such as ``4-digit`` or ``6-digit`` when present."""
    match = re.search(r"(\d+)-digit", pattern)
    if not match:
        return None
    digits = int(match.group(1))
    return bool(re.fullmatch(rf"\d{{{digits}}}", value))


def matches_numeric_schema_pattern(
    value: str,
    *,
    pandas_dtype: str | None = None,
    numeric_role: str | None = None,
    detected_pattern: str | None = None,
) -> bool | None:
    """Check whether one rendered value matches the schema's numeric pattern contract.

    Returns True/False when the current schema metadata is specific enough to make
    a decision, and None when the metadata does not define a concrete pattern.
    """
    stripped = value.strip()
    if not stripped:
        return False

    pattern = _normalized_pattern(detected_pattern)

    # Handle explicit semantic patterns first before falling back to dtype/role-based defaults.
    if pattern == "month number (1-12)":
        return bool(re.fullmatch(r"(?:[1-9]|1[0-2])", stripped))

    if pattern == "4-digit year":
        return bool(re.fullmatch(r"\d{4}", stripped))

    # Two-digit year contracts are intentionally fixed-width as well, so values
    # such as "3" or "2023" do not count as valid when the schema expects "23".
    if pattern == "2-digit year":
        return bool(re.fullmatch(r"\d{2}", stripped))

    if pattern == "yyyymm":
        return bool(re.fullmatch(r"\d{6}", stripped)) and 1 <= int(stripped[4:6]) <= 12

    generic_digit_match = _matches_generic_digit_count(stripped, pattern)
    if generic_digit_match is not None:
        return generic_digit_match

    if pattern in {"integer count", "integer code"}:
        return bool(re.fullmatch(r"\d+", stripped))

    if pandas_dtype == "Int64":
        if numeric_role in {"code", "indicator"}:
            return bool(re.fullmatch(r"\d+", stripped))
        if numeric_role == "measure":
            return bool(re.fullmatch(r"-?\d+", stripped))
        return bool(re.fullmatch(r"-?\d+", stripped))

    if pandas_dtype == "Float64":
        return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", stripped))

    return None


def numeric_pattern_allows_variable_width(
    *,
    pandas_dtype: str | None = None,
    numeric_role: str | None = None,
    detected_pattern: str | None = None,
) -> bool:
    """Return True when the numeric schema contract allows values of varying width."""
    pattern = _normalized_pattern(detected_pattern)

    # Fixed-width temporal codes and explicit N-digit patterns must preserve shape exactly.
    # Year and YYYYMM contracts are fixed-width temporal codes, so the cleaner
    # validator must enforce one exact visible shape instead of allowing width drift.
    if pattern in {"2-digit year", "4-digit year", "yyyymm"}:
        return False

    if re.search(r"(\d+)-digit", pattern):
        return False

    if pattern == "month number (1-12)":
        return True

    if pattern in {"integer count", "integer code"}:
        return True

    if pandas_dtype == "Float64":
        return True

    if pandas_dtype == "Int64" and numeric_role == "measure":
        return True

    return False


# Agent Runtime Helpers

def parse_retry_after_seconds(message: str, attempt: int) -> float:
    """Extract provider retry timing from an error message, else fall back to exponential backoff."""
    match = re.search(r"Please try again in ([0-9.]+)s", message)
    if match:
        return float(match.group(1)) + 0.5
    return min(2**attempt, 10)


def _verbose_agent_runs_enabled() -> bool:
    """Return True when terminal-level agent event tracing is enabled via env var."""
    return os.getenv("AGENT_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}


def _render_tool_args(args: object, limit: int = 400) -> str:
    """Render tool arguments into a single trimmed string for stderr logging."""
    rendered = str(args)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _render_tool_result_content(content: object, limit: int = 400) -> str:
    """Render tool results into a compact single-line stderr preview."""
    rendered = str(content)
    rendered = " ".join(rendered.split())
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _merge_tool_args(existing: object, delta: object) -> object:
    """Merge streaming tool-call argument deltas into one reconstructed payload."""
    if delta is None:
        return existing
    if existing is None:
        return delta
    if isinstance(existing, str) and isinstance(delta, str):
        return existing + delta
    if isinstance(existing, dict) and isinstance(delta, dict):
        merged = dict(existing)
        for key, value in delta.items():
            if key in merged and isinstance(merged[key], str) and isinstance(value, str):
                merged[key] += value
            else:
                merged[key] = value
        return merged
    return delta


def _format_multiline_block(text: str, *, max_lines: int = 20, max_chars: int = 1200) -> str:
    """Indent and truncate multiline content for readable terminal logging."""
    trimmed = text[:max_chars]
    lines = trimmed.splitlines()
    visible = lines[:max_lines]
    suffix = "\n    ..." if len(lines) > max_lines or len(text) > max_chars else ""
    if not visible:
        return "    <empty>"
    return "\n".join(f"    {line}" for line in visible) + suffix


def _render_tool_call_message(tool_name: str, args: object) -> str:
    """Pretty-print one tool call for verbose stderr traces."""
    if isinstance(args, dict):
        if tool_name == "code_execution":
            # Code-execution calls are rendered specially so the generated code block stays readable.
            meta = {k: v for k, v in args.items() if k != "code"}
            parts = [tool_name]
            if meta:
                parts.append(f"  meta: {json.dumps(meta, ensure_ascii=False)}")
            code = args.get("code")
            if isinstance(code, str):
                parts.append("  code:")
                parts.append(_format_multiline_block(code))
            return "\n".join(parts)

        rendered = json.dumps(args, ensure_ascii=False)
        return f"{tool_name} args={_render_tool_args(rendered)}"

    if args is None:
        return tool_name

    return f"{tool_name} args={_render_tool_args(args)}"


def build_terminal_event_stream_handler(agent_name: str):
    """Build an async event handler that streams pydantic-ai activity to stderr.

    This is only used in verbose mode to make agent thinking, tool calls, and
    final outputs inspectable while debugging the orchestration pipeline.
    """
    state = {"open_section": None, "tool_parts": {}}

    def _close_open_section() -> None:
        if state["open_section"] is not None:
            print("", file=sys.stderr, flush=True)
            state["open_section"] = None

    async def _handler(_, events) -> None:
        async for event in events:
            if isinstance(event, PartStartEvent):
                part_kind = getattr(event.part, "part_kind", None)
                if part_kind == "text":
                    _close_open_section()
                    print(f"[{agent_name}][text] ", end="", file=sys.stderr, flush=True)
                    content = getattr(event.part, "content", "")
                    if content:
                        print(content, end="", file=sys.stderr, flush=True)
                    state["open_section"] = "text"
                elif part_kind == "thinking":
                    _close_open_section()
                    print(f"[{agent_name}][thinking] ", end="", file=sys.stderr, flush=True)
                    content = getattr(event.part, "content", "")
                    if content:
                        print(content, end="", file=sys.stderr, flush=True)
                    state["open_section"] = "thinking"
                elif part_kind == "tool-call":
                    _close_open_section()
                    # Tool-call parts may stream their args incrementally, so keep mutable per-part state.
                    state["tool_parts"][event.index] = {
                        "kind": "tool-call",
                        "tool_name": getattr(event.part, "tool_name", "unknown"),
                        "args": getattr(event.part, "args", None),
                    }
                    print(
                        f"[{agent_name}][tool-call] {getattr(event.part, 'tool_name', 'unknown')}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif part_kind == "builtin-tool-call":
                    _close_open_section()
                    state["tool_parts"][event.index] = {
                        "kind": "builtin-tool-call",
                        "tool_name": getattr(event.part, "tool_name", "unknown"),
                        "args": getattr(event.part, "args", None),
                    }
                    print(
                        f"[{agent_name}][builtin-tool-call] {getattr(event.part, 'tool_name', 'unknown')}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif part_kind == "builtin-tool-return":
                    _close_open_section()
                    print(
                        f"[{agent_name}][builtin-tool-result] "
                        f"{_render_tool_result_content(getattr(event.part, 'content', None))}",
                        file=sys.stderr,
                        flush=True,
                    )
            elif isinstance(event, PartDeltaEvent):
                delta_kind = getattr(event.delta, "part_delta_kind", None)
                if delta_kind in {"text", "thinking"}:
                    content_delta = getattr(event.delta, "content_delta", None)
                    if content_delta:
                        print(content_delta, end="", file=sys.stderr, flush=True)
                        state["open_section"] = delta_kind
                elif delta_kind == "tool_call":
                    tool_name_delta = getattr(event.delta, "tool_name_delta", None)
                    args_delta = getattr(event.delta, "args_delta", None)
                    part_state = state["tool_parts"].setdefault(
                        event.index,
                        {"kind": "tool-call", "tool_name": "unknown", "args": None},
                    )
                    if tool_name_delta:
                        part_state["tool_name"] = str(part_state.get("tool_name", "")) + tool_name_delta
                    if args_delta is not None:
                        part_state["args"] = _merge_tool_args(part_state.get("args"), args_delta)
            elif isinstance(event, PartEndEvent):
                part_kind = getattr(event.part, "part_kind", None)
                if part_kind in {"text", "thinking"}:
                    _close_open_section()
                elif part_kind in {"tool-call", "builtin-tool-call"}:
                    _close_open_section()
                    part_state = state["tool_parts"].pop(event.index, None)
                    tool_name = getattr(event.part, "tool_name", "unknown")
                    args = getattr(event.part, "args", None)
                    if part_state is not None:
                        tool_name = part_state.get("tool_name", tool_name)
                        args = part_state.get("args", args)
                    # Emit the fully reconstructed tool call once the streaming part is complete.
                    print(
                        f"[{agent_name}][{part_kind}-args] {_render_tool_call_message(tool_name, args)}",
                        file=sys.stderr,
                        flush=True,
                    )
            elif isinstance(event, FunctionToolCallEvent):
                _close_open_section()
                print(
                    f"[{agent_name}][function-tool-call] {event.part.tool_name} "
                    f"args={_render_tool_args(event.part.args)}",
                    file=sys.stderr,
                    flush=True,
                )
            elif isinstance(event, FunctionToolResultEvent):
                _close_open_section()
                result = event.result
                print(
                    f"[{agent_name}][function-tool-result] {getattr(result, 'tool_name', 'unknown')} "
                    f"outcome={getattr(result, 'outcome', 'unknown')} "
                    f"content={_render_tool_result_content(getattr(result, 'content', None))}",
                    file=sys.stderr,
                    flush=True,
                )
            elif isinstance(event, FinalResultEvent):
                _close_open_section()
                print(
                    f"[{agent_name}][final-result] tool={event.tool_name or 'text'}",
                    file=sys.stderr,
                    flush=True,
                )

    return _handler


def run_agent_with_backoff(
    agent: Agent,
    prompt: str | list[object],
    max_attempts: int = 6,
    usage_limits: UsageLimits | None = None,
    model_settings: dict | None = None,
):
    """Run one agent synchronously with retries for rate limits and connection issues."""
    for attempt in range(max_attempts):
        try:
            event_stream_handler = None
            if _verbose_agent_runs_enabled():
                agent_name = getattr(agent, "name", None) or "agent"
                print(f"[{agent_name}] starting run (attempt {attempt + 1}/{max_attempts})", file=sys.stderr, flush=True)
                event_stream_handler = build_terminal_event_stream_handler(agent_name)
            run_kwargs: dict = {
                "event_stream_handler": event_stream_handler,
                "usage_limits": usage_limits,
            }
            if model_settings is not None:
                run_kwargs["model_settings"] = model_settings
            return agent.run_sync(prompt, **run_kwargs)
        except ModelHTTPError as error:
            # Retry only HTTP 429 rate limits; other HTTP failures should surface immediately.
            if error.status_code != 429 or attempt == max_attempts - 1:
                raise
            body_message = ""
            if isinstance(error.body, dict):
                body_message = str(error.body.get("message", ""))
            wait_seconds = parse_retry_after_seconds(body_message, attempt + 1)
            if _verbose_agent_runs_enabled():
                print(
                    f"[{getattr(agent, 'name', None) or 'agent'}] retrying after HTTP 429 in {wait_seconds:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(wait_seconds)
        except ModelAPIError as error:
            message = str(error)
            # Transport-level connection failures are retriable; semantic/API errors are not.
            is_connection_issue = "Connection error" in message or "connection" in message.lower()
            if not is_connection_issue or attempt == max_attempts - 1:
                raise
            wait_seconds = min(2**attempt, 10)
            if _verbose_agent_runs_enabled():
                print(
                    f"[{getattr(agent, 'name', None) or 'agent'}] connection issue, retrying in {wait_seconds:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(wait_seconds)


async def run_agent_with_backoff_async(
    agent: Agent,
    prompt: str | list[object],
    max_attempts: int = 6,
    usage_limits: UsageLimits | None = None,
    model_settings: dict | None = None,
):
    """Async variant of ``run_agent_with_backoff`` with identical retry policy."""
    for attempt in range(max_attempts):
        try:
            event_stream_handler = None
            if _verbose_agent_runs_enabled():
                agent_name = getattr(agent, "name", None) or "agent"
                print(f"[{agent_name}] starting run (attempt {attempt + 1}/{max_attempts})", file=sys.stderr, flush=True)
                event_stream_handler = build_terminal_event_stream_handler(agent_name)
            run_kwargs: dict = {
                "event_stream_handler": event_stream_handler,
                "usage_limits": usage_limits,
            }
            if model_settings is not None:
                run_kwargs["model_settings"] = model_settings
            return await agent.run(prompt, **run_kwargs)
        except ModelHTTPError as error:
            if error.status_code != 429 or attempt == max_attempts - 1:
                raise
            body_message = ""
            if isinstance(error.body, dict):
                body_message = str(error.body.get("message", ""))
            wait_seconds = parse_retry_after_seconds(body_message, attempt + 1)
            if _verbose_agent_runs_enabled():
                print(
                    f"[{getattr(agent, 'name', None) or 'agent'}] retrying after HTTP 429 in {wait_seconds:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
            await asyncio.sleep(wait_seconds)
        except ModelAPIError as error:
            message = str(error)
            is_connection_issue = "Connection error" in message or "connection" in message.lower()
            if not is_connection_issue or attempt == max_attempts - 1:
                raise
            wait_seconds = min(2**attempt, 10)
            if _verbose_agent_runs_enabled():
                print(
                    f"[{getattr(agent, 'name', None) or 'agent'}] connection issue, retrying in {wait_seconds:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
            await asyncio.sleep(wait_seconds)
