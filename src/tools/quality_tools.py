"""Cross-column and dataset-wide quality heuristics for later validation stages.

Implements the deterministic detectors used by the anomaly, cross-column, and
duplicate-detection stages: numeric outliers, rare categories, duplicate-like
columns, semantic conflicts, period mismatches, date-order violations, and
exact / near-duplicate rows."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any

import pandas as pd

from src.tools.common_tools import PLACEHOLDER_TOKENS
from src.tools.schema_tools import normalized_schema_name


def normalize_cell_text(value: object) -> str:
    """Normalize a cell into a whitespace-collapsed lowercase comparison string."""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def render_cell_text(value: object) -> str | None:
    """Render a cell for human-readable evidence output, preserving original case."""
    if value is None or pd.isna(value):
        return None
    rendered = str(value).strip()
    return rendered or None


def _series_to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric with light normalization for decimal commas."""
    normalized = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    # Restore real nulls before coercion so they do not become the literal string "nan".
    normalized = normalized.where(~series.isna(), None)
    return pd.to_numeric(normalized, errors="coerce")


def detect_numeric_outlier_candidates(df: pd.DataFrame, schema_columns: list[Any]) -> list[dict[str, Any]]:
    """Detect robust-IQR numeric outliers for measure-like numeric columns only."""
    findings: list[dict[str, Any]] = []
    for column in schema_columns:
        # Codes and indicators may be numeric but should not be judged as continuous measures.
        if column.pandas_dtype not in {"Int64", "Float64"}:
            continue
        if column.numeric_role in {"code", "indicator"}:
            continue

        numeric = _series_to_numeric(df[column.name]).dropna()
        # Require enough sample size and spread to avoid noisy outlier labels on tiny columns.
        if len(numeric) < 20 or numeric.nunique() < 10:
            continue

        q1 = float(numeric.quantile(0.25))
        q3 = float(numeric.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            continue

        # Use a conservative 3*IQR band to reduce false positives on naturally skewed public-data distributions.
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        mask = (numeric < lower) | (numeric > upper)
        outlier_index = list(numeric[mask].index)
        if not outlier_index:
            continue

        raw_examples = [
            render_cell_text(df.at[index, column.name])
            for index in outlier_index[:10]
        ]
        example_values = [value for value in raw_examples if value is not None]
        severity = "high" if len(outlier_index) / max(len(df), 1) >= 0.02 else "medium"
        findings.append(
            {
                "column_name": column.name,
                "anomaly_type": "numeric_outlier",
                "severity": severity,
                "affected_rows": len(outlier_index),
                "example_values": list(dict.fromkeys(example_values))[:8],
                "evidence": (
                    f"{len(outlier_index)} rows fall outside the robust IQR band "
                    f"[{lower:.3f}, {upper:.3f}] computed from Q1={q1:.3f}, Q3={q3:.3f}."
                ),
                "suggested_action": (
                    "Review whether these values are genuine extreme cases or unit/format errors before imputation or removal."
                ),
            }
        )

    return findings


def detect_rare_category_candidates(df: pd.DataFrame, schema_columns: list[Any]) -> list[dict[str, Any]]:
    """Detect suspiciously rare labels in low-to-moderate-cardinality categorical columns."""
    findings: list[dict[str, Any]] = []
    for column in schema_columns:
        # Free text, names, and identifiers are expected to have many unique values and should not trigger this rule.
        if column.pandas_dtype not in {"string", "object"}:
            continue
        if column.string_role in {"free_text", "name", "identifier"}:
            continue

        rendered = df[column.name].dropna().astype(str).str.strip()
        rendered = rendered[rendered != ""]
        if rendered.empty:
            continue

        normalized = rendered.str.lower()
        rendered = rendered[~normalized.isin(PLACEHOLDER_TOKENS)]
        if rendered.empty:
            continue

        distinct = rendered.nunique()
        distinct_ratio = distinct / len(rendered)
        # Skip columns that are too small, too high-cardinality, or too diverse for rare-category heuristics to be meaningful.
        if distinct < 5 or distinct_ratio > 0.2 or distinct > 50:
            continue

        counts = Counter(rendered)
        # If no category is even moderately common, the column does not have a stable baseline to define "rare" against.
        if counts and (counts.most_common(1)[0][1] / len(rendered)) < 0.2:
            continue
        rare_threshold = max(1, math.floor(len(rendered) * 0.005))
        rare_values = [value for value, count in counts.items() if count <= rare_threshold]
        if not rare_values:
            continue

        affected_rows = sum(counts[value] for value in rare_values)
        example_values = sorted(rare_values, key=lambda value: (counts[value], value))[:8]
        severity = "medium" if affected_rows <= 5 else "low"
        findings.append(
            {
                "column_name": column.name,
                "anomaly_type": "rare_category",
                "severity": severity,
                "affected_rows": affected_rows,
                "example_values": example_values,
                "evidence": (
                    f"{len(rare_values)} rare categories occur at or below {rare_threshold} row(s) each "
                    f"out of {len(rendered)} non-null values."
                ),
                "suggested_action": (
                    "Review whether these rare labels are valid edge cases, spelling variants, or categories that should be consolidated."
                ),
            }
        )

    return findings


def _normalized_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Return one column normalized for case/whitespace-insensitive comparisons."""
    return df[column_name].map(normalize_cell_text)


def _schema_column_map(schema_columns: list[Any]) -> dict[str, Any]:
    """Build a name-to-schema-entry lookup map, skipping malformed entries."""
    return {column.name: column for column in schema_columns if getattr(column, "name", None)}


def _dtype_family(column: Any) -> str:
    """Collapse detailed pandas dtypes into broad comparison families."""
    pandas_dtype = getattr(column, "pandas_dtype", "") or ""
    if pandas_dtype in {"Int64", "Float64"}:
        return "numeric"
    if pandas_dtype == "datetime64[ns]":
        return "datetime"
    return "text"


def _eligible_column_pair(left: Any, right: Any) -> bool:
    """Return True when two schema-described columns are comparable for duplicate-like checks."""
    if left is None or right is None:
        return False
    if _dtype_family(left) != _dtype_family(right):
        return False
    if getattr(left, "string_role", None) == "free_text" or getattr(right, "string_role", None) == "free_text":
        return False
    if getattr(left, "numeric_role", None) == "measure" and getattr(right, "numeric_role", None) == "code":
        return False
    if getattr(left, "numeric_role", None) == "code" and getattr(right, "numeric_role", None) == "measure":
        return False
    return True


def detect_duplicate_like_columns(df: pd.DataFrame, schema_columns: list[Any]) -> list[dict[str, Any]]:
    """Detect exact and near-duplicate columns based on normalized row-wise agreement."""
    findings: list[dict[str, Any]] = []
    schema_map = _schema_column_map(schema_columns)
    seen_pairs: set[tuple[str, str]] = set()

    for left_index, left_name in enumerate(df.columns):
        left_meta = schema_map.get(left_name)
        for right_name in df.columns[left_index + 1:]:
            right_meta = schema_map.get(right_name)
            if not _eligible_column_pair(left_meta, right_meta):
                continue

            pair_key = tuple(sorted((left_name, right_name)))
            if pair_key in seen_pairs:
                continue

            left_norm = _normalized_series(df, left_name)
            right_norm = _normalized_series(df, right_name)
            # Compare only rows where both columns contain a real, non-placeholder value.
            comparable = (
                left_norm.ne("")
                & right_norm.ne("")
                & ~left_norm.isin(PLACEHOLDER_TOKENS)
                & ~right_norm.isin(PLACEHOLDER_TOKENS)
            )
            comparable_count = int(comparable.sum())
            if comparable_count < 20:
                continue

            left_present = int((left_norm.ne("") & ~left_norm.isin(PLACEHOLDER_TOKENS)).sum())
            right_present = int((right_norm.ne("") & ~right_norm.isin(PLACEHOLDER_TOKENS)).sum())
            # Require strong overlap so we do not compare columns that rarely co-occur.
            overlap_share = comparable_count / max(min(left_present, right_present), 1)
            if overlap_share < 0.8:
                continue

            agreement = comparable & left_norm.eq(right_norm)
            agreement_count = int(agreement.sum())
            similarity_pct = round((agreement_count / comparable_count) * 100, 2)
            mismatch_mask = comparable & ~left_norm.eq(right_norm)
            mismatch_indices = list(df.index[mismatch_mask])

            if agreement_count == comparable_count:
                findings.append(
                    {
                        "columns": [left_name, right_name],
                        "check_type": "exact_duplicate_columns",
                        "severity": "high",
                        "affected_rows": comparable_count,
                        "example_row_indices": list(df.index[comparable])[:8],
                        "similarity_pct": similarity_pct,
                        "evidence": (
                            f"Columns {left_name!r} and {right_name!r} match exactly on all {comparable_count} "
                            f"comparable non-missing rows ({similarity_pct:.2f}% similarity)."
                        ),
                        "suggested_action": (
                            "Review whether one column is redundant and can be dropped, or whether both should remain for lineage reasons."
                        ),
                    }
                )
                seen_pairs.add(pair_key)
                continue

            mismatch_count = len(mismatch_indices)
            if similarity_pct < 95 or mismatch_count > max(10, math.ceil(comparable_count * 0.05)):
                continue

            findings.append(
                {
                    "columns": [left_name, right_name],
                    "check_type": "near_duplicate_columns",
                    "severity": "medium",
                    "affected_rows": comparable_count,
                    "example_row_indices": mismatch_indices[:8],
                    "similarity_pct": similarity_pct,
                    "evidence": (
                        f"Columns {left_name!r} and {right_name!r} agree on {agreement_count} of {comparable_count} "
                        f"comparable non-missing rows ({similarity_pct:.2f}% similarity), with {mismatch_count} mismatches."
                    ),
                    "suggested_action": (
                        "Review the mismatching rows to decide whether one column is a stale copy, a lightly diverged duplicate, or a distinct field."
                    ),
                }
            )
            seen_pairs.add(pair_key)

    return findings


def detect_duplicate_semantic_conflicts(df: pd.DataFrame, duplicate_groups: list[Any]) -> list[dict[str, Any]]:
    """Detect conflicting values inside schema-level duplicate-name column groups."""
    findings: list[dict[str, Any]] = []
    for group in duplicate_groups:
        # Current heuristic only handles the simple and most actionable 2-column conflict case.
        if len(group.columns) != 2:
            continue
        left, right = group.columns
        if left not in df.columns or right not in df.columns:
            continue

        left_norm = _normalized_series(df, left)
        right_norm = _normalized_series(df, right)
        comparable = (
            left_norm.ne("")
            & right_norm.ne("")
            & ~left_norm.isin(PLACEHOLDER_TOKENS)
            & ~right_norm.isin(PLACEHOLDER_TOKENS)
        )
        comparable_count = int(comparable.sum())
        if comparable_count == 0:
            continue

        agreement = comparable & left_norm.eq(right_norm)
        agreement_count = int(agreement.sum())
        similarity_pct = round((agreement_count / comparable_count) * 100, 2)
        mismatch_mask = comparable & left_norm.ne(right_norm)
        mismatch_indices = list(df.index[mismatch_mask])
        if not mismatch_indices:
            continue

        findings.append(
            {
                "columns": [left, right],
                "check_type": "duplicate_semantic_conflict",
                "severity": "high",
                "affected_rows": len(mismatch_indices),
                "example_row_indices": mismatch_indices[:8],
                "similarity_pct": similarity_pct,
                "evidence": (
                    f"Columns {left!r} and {right!r} normalize to the same schema name but disagree on "
                    f"{len(mismatch_indices)} of {comparable_count} rows where both values are present "
                    f"({similarity_pct:.2f}% similarity)."
                ),
                "suggested_action": (
                    "Review whether one column should override the other, whether they need reconciliation rules, or whether both must be preserved separately."
                ),
            }
        )

    return findings


def _find_pattern_columns(schema_columns: list[Any], pattern: str) -> list[Any]:
    """Return schema columns whose detected pattern matches the given canonical label."""
    return [column for column in schema_columns if (column.detected_pattern or "").strip().lower() == pattern]


def _looks_like_period_column(column: Any) -> bool:
    """Heuristically identify YYYYMM-like period key columns from schema metadata."""
    pattern = (column.detected_pattern or "").strip().lower()
    if pattern == "yyyymm":
        return True
    tokens = set(normalized_schema_name(column.name).split("_"))
    sample_values = [str(value).strip() for value in getattr(column, "sample_values", []) if str(value).strip()]
    return (
        bool({"rata", "period", "periodo"} & tokens)
        and any(re.fullmatch(r"\d{6}", value) for value in sample_values)
    )


def detect_year_month_period_mismatches(df: pd.DataFrame, schema_columns: list[Any]) -> list[dict[str, Any]]:
    """Detect rows where year/month columns disagree with a companion YYYYMM period key."""
    findings: list[dict[str, Any]] = []
    year_columns = _find_pattern_columns(schema_columns, "4-digit year")
    month_columns = _find_pattern_columns(schema_columns, "month number (1-12)")
    period_columns = [column for column in schema_columns if _looks_like_period_column(column)]

    for year_column in year_columns:
        for month_column in month_columns:
            for period_column in period_columns:
                if len({year_column.name, month_column.name, period_column.name}) < 3:
                    continue

                year_values = pd.to_numeric(df[year_column.name], errors="coerce")
                month_values = pd.to_numeric(df[month_column.name], errors="coerce")
                period_values = df[period_column.name].astype(str).str.strip()
                valid_period = period_values.str.fullmatch(r"\d{6}")
                comparable = year_values.notna() & month_values.notna() & valid_period
                if not comparable.any():
                    continue

                # Rebuild the expected YYYYMM key from year + zero-padded month and compare it with the stored period.
                expected = (
                    year_values[comparable].astype(int).astype(str)
                    + month_values[comparable].astype(int).astype(str).str.zfill(2)
                )
                mismatch_mask = comparable.copy()
                mismatch_mask.loc[comparable] = period_values[comparable].ne(expected)
                mismatch_indices = list(df.index[mismatch_mask])
                if not mismatch_indices:
                    continue

                findings.append(
                    {
                        "columns": [year_column.name, month_column.name, period_column.name],
                        "check_type": "year_month_period_mismatch",
                        "severity": "high",
                        "affected_rows": len(mismatch_indices),
                        "example_row_indices": mismatch_indices[:8],
                        "evidence": (
                            f"Period column {period_column.name!r} disagrees with year/month columns "
                            f"{year_column.name!r}/{month_column.name!r} on {len(mismatch_indices)} comparable rows."
                        ),
                        "suggested_action": (
                            "Check whether the period key should be regenerated from year/month or whether one of the source columns is unreliable."
                        ),
                    }
                )

    return findings


def detect_date_order_violations(df: pd.DataFrame, schema_columns: list[Any]) -> list[dict[str, Any]]:
    """Detect start/end datetime pairs where the start date occurs after the end date."""
    findings: list[dict[str, Any]] = []
    datetime_columns = [column for column in schema_columns if column.pandas_dtype == "datetime64[ns]"]
    if len(datetime_columns) < 2:
        return findings

    name_map = {column.name: set(normalized_schema_name(column.name).split("_")) for column in datetime_columns}
    start_tokens = {"start", "from", "inizio", "begin"}
    end_tokens = {"end", "to", "fine", "stop"}

    for start_column in datetime_columns:
        for end_column in datetime_columns:
            # Identify likely temporal pairs from their normalized name tokens.
            if start_column.name == end_column.name:
                continue
            if not (name_map[start_column.name] & start_tokens):
                continue
            if not (name_map[end_column.name] & end_tokens):
                continue

            start_values = pd.to_datetime(df[start_column.name], errors="coerce", format="mixed")
            end_values = pd.to_datetime(df[end_column.name], errors="coerce", format="mixed")
            comparable = start_values.notna() & end_values.notna()
            violation_mask = comparable & start_values.gt(end_values)
            violation_indices = list(df.index[violation_mask])
            if not violation_indices:
                continue

            findings.append(
                {
                    "columns": [start_column.name, end_column.name],
                    "check_type": "date_order_violation",
                    "severity": "high",
                    "affected_rows": len(violation_indices),
                    "example_row_indices": violation_indices[:8],
                    "evidence": (
                        f"Start-like column {start_column.name!r} is later than end-like column {end_column.name!r} "
                        f"on {len(violation_indices)} rows."
                    ),
                    "suggested_action": (
                        "Review whether the columns are swapped, misparsed, or contain upstream date-entry errors."
                    ),
                }
            )

    return findings


def _normalized_row_signature(row: pd.Series) -> tuple[str, ...]:
    """Convert one full row into a normalized tuple suitable for duplicate grouping."""
    return tuple(normalize_cell_text(value) for value in row.tolist())


def detect_exact_duplicate_groups(df: pd.DataFrame, max_groups: int = 25) -> list[dict[str, Any]]:
    """Detect exact duplicate rows after case/whitespace normalization."""
    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, (_, row) in enumerate(df.iterrows()):
        grouped[_normalized_row_signature(row)].append(index)

    findings: list[dict[str, Any]] = []
    for row_indices in grouped.values():
        if len(row_indices) < 2:
            continue
        findings.append(
            {
                "duplicate_type": "exact_row",
                # Keep the full index list so downstream remediation can drop the
                # exact duplicate rows deterministically instead of only reporting
                # a truncated preview.
                "row_indices": row_indices,
                "key_columns": list(df.columns),
                "evidence": f"{len(row_indices)} rows are exact duplicates after whitespace/case normalization.",
                "suggested_action": "Review whether duplicate rows should be deduplicated or retained as legitimate repeated records.",
            }
        )
        if len(findings) >= max_groups:
            break

    return findings


def infer_duplicate_key_columns(schema_columns: list[Any], df: pd.DataFrame) -> list[str]:
    """Infer a small set of likely business-key columns for near-duplicate grouping."""
    selected: list[str] = []
    for column in schema_columns:
        tokens = set(normalized_schema_name(column.name).split("_"))
        # Prefer explicit identifiers/codes, then temporal key components, then common id/code tokens in the name.
        if column.string_role == "identifier" or column.numeric_role == "code":
            selected.append(column.name)
        elif (column.detected_pattern or "").strip().lower() in {"4-digit year", "month number (1-12)", "yyyymm"}:
            selected.append(column.name)
        elif {"id", "cod", "code", "rata"} & tokens:
            selected.append(column.name)

    selected = [column for column in dict.fromkeys(selected) if column in df.columns]
    return selected[:4]


def detect_near_duplicate_groups(
    df: pd.DataFrame,
    key_columns: list[str],
    max_groups: int = 25,
) -> list[dict[str, Any]]:
    """Detect groups of rows that share inferred business keys but differ elsewhere."""
    if not key_columns:
        return []

    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, (_, row) in enumerate(df[key_columns].iterrows()):
        key = tuple(normalize_cell_text(value) for value in row.tolist())
        # Skip groups whose inferred key is entirely empty; they are not meaningful identifiers.
        if not any(key):
            continue
        grouped[key].append(index)

    full_signatures = [_normalized_row_signature(row) for _, row in df.iterrows()]
    findings: list[dict[str, Any]] = []
    for row_indices in grouped.values():
        if len(row_indices) < 2:
            continue
        unique_rows = {full_signatures[index] for index in row_indices}
        if len(unique_rows) <= 1:
            continue
        findings.append(
            {
                "duplicate_type": "near_duplicate",
                "row_indices": row_indices[:20],
                "key_columns": key_columns,
                "evidence": (
                    f"{len(row_indices)} rows share the inferred key columns {key_columns} but differ elsewhere, "
                    "suggesting potential near-duplicate records."
                ),
                "suggested_action": "Review whether these rows are conflicting duplicates, valid multiple events, or records that need merge rules.",
            }
        )
        if len(findings) >= max_groups:
            break

    return findings
