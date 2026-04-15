from __future__ import annotations

import base64
import gzip
from pathlib import Path
import re
import time

import pandas as pd
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import BinaryContent

# Shared Constants

PLACEHOLDER_TOKENS = ("", "na", "n/a", "null", "none", "-", "--", "unknown", "n.d.", "?")


# Shared Attachment And I/O Helpers

def attach_single_column_csv(df: pd.DataFrame, column_name: str) -> BinaryContent:
    return BinaryContent(data=df[[column_name]].to_csv(index=False).encode("utf-8"), media_type="text/csv")


def attach_text_document(text: str) -> BinaryContent:
    return BinaryContent(data=text.encode("utf-8"), media_type="text/plain")


def attach_profile_text(profile: BaseModel) -> BinaryContent:
    return attach_text_document(profile.model_dump_json(indent=2))


def load_dataset_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# Shared Encoding Helpers

def gzip_text_to_base64(text: str) -> str:
    return base64.b64encode(gzip.compress(text.encode("utf-8"))).decode("ascii")


def ungzip_base64_to_text(payload: str) -> str:
    normalized = payload.strip()
    if normalized.startswith("data:") and "," in normalized:
        normalized = normalized.split(",", 1)[1]
    normalized = "".join(normalized.split())
    padding = len(normalized) % 4
    if padding:
        normalized += "=" * (4 - padding)

    try:
        decoded = base64.b64decode(normalized.encode("ascii"))
    except Exception:
        decoded = base64.urlsafe_b64decode(normalized.encode("ascii"))

    return gzip.decompress(decoded).decode("utf-8")


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

    signal_mask = rendered.str.contains(r"[-/:T ]", regex=True)
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


# Agent Runtime Helpers

def parse_retry_after_seconds(message: str, attempt: int) -> float:
    match = re.search(r"Please try again in ([0-9.]+)s", message)
    if match:
        return float(match.group(1)) + 0.5
    return min(2**attempt, 10)


def run_agent_with_backoff(agent: Agent, prompt: str | list[object], max_attempts: int = 6):
    for attempt in range(max_attempts):
        try:
            return agent.run_sync(prompt)
        except ModelHTTPError as error:
            if error.status_code != 429 or attempt == max_attempts - 1:
                raise
            body_message = ""
            if isinstance(error.body, dict):
                body_message = str(error.body.get("message", ""))
            wait_seconds = parse_retry_after_seconds(body_message, attempt + 1)
            time.sleep(wait_seconds)
        except ModelAPIError as error:
            message = str(error)
            is_connection_issue = "Connection error" in message or "connection" in message.lower()
            if not is_connection_issue or attempt == max_attempts - 1:
                raise
            wait_seconds = min(2**attempt, 10)
            time.sleep(wait_seconds)
