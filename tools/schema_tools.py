from __future__ import annotations

import re

from pydantic import BaseModel, Field

from tools.common_tools import (
    compute_datetime_parse_pct,
    compute_empty_like_pct,
    compute_numeric_parse_pct,
    sample_non_null_values,
)

# Schema Constants

VALID_SCHEMA_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


# Schema Data Models

class ColumnProfile(BaseModel):
    column_name: str
    pandas_dtype: str
    non_null_rows: int = Field(ge=0)
    distinct_non_null_values: int = Field(ge=0)
    numeric_parse_pct: float = Field(ge=0, le=100)
    datetime_parse_pct: float = Field(ge=0, le=100)
    empty_like_pct: float = Field(ge=0, le=100)
    sample_values: list[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    dataset_name: str
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    columns: list[ColumnProfile] = Field(default_factory=list)


class SchemaDuplicateGroup(BaseModel):
    canonical_name: str
    columns: list[str] = Field(default_factory=list)


class SchemaLocalFacts(BaseModel):
    dataset_name: str
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    column_names: list[str] = Field(default_factory=list)
    invalid_naming_columns: list[str] = Field(default_factory=list)
    rename_suggestions: dict[str, str] = Field(default_factory=dict)
    naming_reasons: dict[str, str] = Field(default_factory=dict)
    data_type_risk_columns: list[str] = Field(default_factory=list)
    type_risk_rationales: dict[str, str] = Field(default_factory=dict)
    duplicate_semantic_columns: list[str] = Field(default_factory=list)
    duplicate_groups: list[SchemaDuplicateGroup] = Field(default_factory=list)


# Schema Helpers

def is_valid_schema_name(name: str) -> bool:
    if name.startswith("_"):
        body = name[1:]
        return bool(body) and bool(VALID_SCHEMA_NAME_RE.fullmatch(body))
    return bool(VALID_SCHEMA_NAME_RE.fullmatch(name))


def normalized_schema_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "column"


def suggest_schema_name(name: str) -> str:
    normalized = normalized_schema_name(name)
    if normalized[0].isdigit():
        match = re.match(r"(\d+)(.*)", normalized)
        if match:
            digits, remainder = match.groups()
            remainder = remainder.strip("_")
            if remainder:
                normalized = f"{remainder}_{digits}"
            else:
                normalized = f"col_{digits}"
    if normalized.startswith("_"):
        normalized = normalized.lstrip("_") or "column"
    if not is_valid_schema_name(normalized):
        normalized = f"col_{normalized}".strip("_")
    return normalized


def naming_rule_reason(name: str) -> str:
    reasons: list[str] = []
    if any(char.isupper() for char in name):
        reasons.append("uppercase letters")
    if any(char.isspace() for char in name):
        reasons.append("whitespace")
    if "-" in name:
        reasons.append("a hyphen")
    if "%" in name:
        reasons.append("a percent sign")
    if name and name[0].isdigit():
        reasons.append("a leading digit")
    if not reasons:
        return "Column name violates the lowercase snake_case naming rule."
    if len(reasons) == 1:
        return f"Column name contains {reasons[0]}, which violates the lowercase snake_case naming rule."
    return (
        "Column name contains "
        + ", ".join(reasons[:-1])
        + f", and {reasons[-1]}, which violates the lowercase snake_case naming rule."
    )


def infer_type_risk_rationale(column: ColumnProfile) -> str | None:
    normalized = normalized_schema_name(column.column_name)
    tokens = set(normalized.split("_"))

    is_code_like = bool({"id", "cod", "code"} & tokens)
    is_temporal_like = bool(
        {"date", "time", "timestamp", "datetime", "period", "month", "year", "rata", "aggregation"} & tokens
    )
    is_amount_like = bool(
        {"spesa", "amount", "importo", "price", "prezzo", "cost", "costo", "total", "totale"} & tokens
    )

    if is_code_like:
        return None
    if is_temporal_like and column.datetime_parse_pct < 40 and column.distinct_non_null_values > 0:
        return (
            f"Column name suggests temporal data but datetime_parse_pct is only {column.datetime_parse_pct:.2f}%, "
            "so the values may not align with the expected temporal meaning."
        )
    if is_amount_like and 0 < column.numeric_parse_pct < 60 and column.distinct_non_null_values > 0:
        return (
            f"Column name suggests an amount or total but numeric_parse_pct is only {column.numeric_parse_pct:.2f}%, "
            "so the values may need review before numeric cleaning."
        )
    return None


# Schema Profile Builders

def build_dataset_profile(df, dataset_name: str) -> DatasetProfile:
    return DatasetProfile(
        dataset_name=dataset_name,
        total_rows=len(df),
        total_columns=len(df.columns),
        columns=[
            ColumnProfile(
                column_name=column_name,
                pandas_dtype=str(df[column_name].dtype),
                non_null_rows=int(df[column_name].notna().sum()),
                distinct_non_null_values=int(df[column_name].nunique(dropna=True)),
                numeric_parse_pct=compute_numeric_parse_pct(df[column_name]),
                datetime_parse_pct=compute_datetime_parse_pct(df[column_name]),
                empty_like_pct=compute_empty_like_pct(df[column_name]),
                sample_values=sample_non_null_values(df[column_name]),
            )
            for column_name in df.columns
        ],
    )


def build_schema_local_facts(profile: DatasetProfile) -> SchemaLocalFacts:
    invalid_naming_columns: list[str] = []
    rename_suggestions: dict[str, str] = {}
    naming_reasons: dict[str, str] = {}
    data_type_risk_columns: list[str] = []
    type_risk_rationales: dict[str, str] = {}
    duplicate_groups_by_name: dict[str, list[str]] = {}

    for column in profile.columns:
        column_name = column.column_name
        if not is_valid_schema_name(column_name):
            invalid_naming_columns.append(column_name)
            rename_suggestions[column_name] = suggest_schema_name(column_name)
            naming_reasons[column_name] = naming_rule_reason(column_name)

        risk_rationale = infer_type_risk_rationale(column)
        if risk_rationale is not None:
            data_type_risk_columns.append(column_name)
            type_risk_rationales[column_name] = risk_rationale

        canonical_name = normalized_schema_name(column_name)
        duplicate_groups_by_name.setdefault(canonical_name, []).append(column_name)

    duplicate_groups = [
        SchemaDuplicateGroup(canonical_name=canonical_name, columns=columns)
        for canonical_name, columns in duplicate_groups_by_name.items()
        if len(columns) > 1
    ]
    duplicate_semantic_columns = [column_name for group in duplicate_groups for column_name in group.columns]

    return SchemaLocalFacts(
        dataset_name=profile.dataset_name,
        total_rows=profile.total_rows,
        total_columns=profile.total_columns,
        column_names=[column.column_name for column in profile.columns],
        invalid_naming_columns=invalid_naming_columns,
        rename_suggestions=rename_suggestions,
        naming_reasons=naming_reasons,
        data_type_risk_columns=data_type_risk_columns,
        type_risk_rationales=type_risk_rationales,
        duplicate_semantic_columns=duplicate_semantic_columns,
        duplicate_groups=duplicate_groups,
    )
