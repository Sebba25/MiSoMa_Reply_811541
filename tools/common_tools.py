"""Cross-cutting helpers shared by the specialised ``tools/*`` modules.

Groups four concerns:
    * placeholder token list + attachment helpers for agent prompts
    * numeric/datetime parse-rate and value-shape primitives
    * schema pattern matchers used by the cleaner validator
    * ``run_agent_with_backoff``: the single retrying entrypoint used by every
      pydantic-ai agent call in the pipeline

Nothing here talks to the agents directly — only pandas, pydantic, and the
pydantic-ai runtime types.
"""

from __future__ import annotations

import base64
import asyncio
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
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import (
    BinaryContent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
)
from pydantic_ai.usage import UsageLimits

# Shared Constants

PLACEHOLDER_TOKENS = ("", "na", "n/a", "null", "none", "-", "--", "unknown", "n.d.", "?", "//")


# Shared Attachment And I/O Helpers

def attach_text_document(text: str) -> BinaryContent:
    return BinaryContent(data=text.encode("utf-8"), media_type="text/plain")


def attach_profile_text(profile: BaseModel) -> BinaryContent:
    return attach_text_document(profile.model_dump_json(indent=2))


def load_dataset_frame(path: Path, dtype: str | dict | None = None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=dtype)


# Shared Encoding Helpers

def gzip_text_to_base64(text: str) -> str:
    return base64.b64encode(gzip.compress(text.encode("utf-8"))).decode("ascii")


# Shared Profiling Helpers

def sample_non_null_values(series: pd.Series, limit: int = 5) -> list[str]:
    values: list[str] = []
    for value in series.dropna():
        rendered = str(value).strip()
        if not rendered or rendered in values:
            continue
        values.append(rendered[:80])
        if len(values) >= limit:
            break
    return values


def value_shape(value: str) -> str:
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
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    rendered = non_null.astype(str).str.strip()
    rendered = rendered[rendered != ""]
    if rendered.empty:
        return 0.0

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
    if series.empty:
        return 0.0
    rendered = series.fillna("").astype(str).str.strip().str.lower()
    empty_like = rendered.isin(PLACEHOLDER_TOKENS)
    return float((empty_like.mean()) * 100)


def _normalized_pattern(pattern: str | None) -> str:
    return (pattern or "").strip().lower()


def _matches_generic_digit_count(value: str, pattern: str) -> bool | None:
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
    stripped = value.strip()
    if not stripped:
        return False

    pattern = _normalized_pattern(detected_pattern)

    if pattern == "month number (1-12)":
        return bool(re.fullmatch(r"(?:[1-9]|1[0-2])", stripped))

    if pattern == "4-digit year":
        return bool(re.fullmatch(r"\d{4}", stripped))

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
    pattern = _normalized_pattern(detected_pattern)

    if pattern in {"4-digit year", "yyyymm"}:
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
    match = re.search(r"Please try again in ([0-9.]+)s", message)
    if match:
        return float(match.group(1)) + 0.5
    return min(2**attempt, 10)


def _verbose_agent_runs_enabled() -> bool:
    return os.getenv("AGENT_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}


def _render_tool_args(args: object, limit: int = 400) -> str:
    rendered = str(args)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _render_tool_result_content(content: object, limit: int = 400) -> str:
    rendered = str(content)
    rendered = " ".join(rendered.split())
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _merge_tool_args(existing: object, delta: object) -> object:
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
    trimmed = text[:max_chars]
    lines = trimmed.splitlines()
    visible = lines[:max_lines]
    suffix = "\n    ..." if len(lines) > max_lines or len(text) > max_chars else ""
    if not visible:
        return "    <empty>"
    return "\n".join(f"    {line}" for line in visible) + suffix


def _render_tool_call_message(tool_name: str, args: object) -> str:
    if isinstance(args, dict):
        if tool_name == "code_execution":
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
