from __future__ import annotations
import re
from pydantic import BaseModel, Field
from tools.common_tools import (
    compute_datetime_parse_pct,
    compute_empty_like_pct,
    compute_numeric_parse_pct,
    sample_non_null_values,
)

# Valid snake_case name: starts with a lowercase letter, followed by lowercase letters, digits, or underscores only.
VALID_SCHEMA_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


# --- Data Models ---

class ColumnProfile(BaseModel):
    """Statistical snapshot of a single column, computed once from the raw DataFrame.
    Everything downstream — schema checks, completeness, format — derives from this."""
    column_name: str
    pandas_dtype: str                                    # Inferred dtype obtained from LLM call
    non_null_rows: int = Field(ge=0)                     
    distinct_non_null_values: int = Field(ge=0)          
    numeric_parse_pct: float = Field(ge=0, le=100)       
    datetime_parse_pct: float = Field(ge=0, le=100)     
    empty_like_pct: float = Field(ge=0, le=100)          
    sample_values: list[str] = Field(default_factory=list)  # small representative sample for LLM prompts

class DatasetProfile(BaseModel):
    """Full profile of a dataset: one ColumnProfile per column."""
    dataset_name: str
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    columns_profiles: list[ColumnProfile] = Field(default_factory=list)

class SchemaDuplicateGroup(BaseModel):
    """A group of columns that normalize to the same canonical name,
    suggesting possible semantic overlap (e.g. 'Provincia Sede' and 'provincia_sede')."""
    canonical_name: str                          # the shared normalized name
    columns: list[str] = Field(default_factory=list)  # original column names in this group



# --- Naming Convention Helpers ---

def is_valid_schema_name(name: str) -> bool:
    """Return True if the column name follows the lowercase snake_case convention.
    A leading underscore is allowed (e.g. '_internal'), but the rest must match."""
    if name.startswith("_"):
        body = name[1:]
        return bool(body) and bool(VALID_SCHEMA_NAME_RE.fullmatch(body))
    return bool(VALID_SCHEMA_NAME_RE.fullmatch(name))

def normalized_schema_name(name: str) -> str:
    """Convert any column name to its canonical snake_case form.
    Used both for rename suggestions and for duplicate detection:
    'Provincia Sede' and 'provincia_sede' both normalize to 'provincia_sede'."""
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower())  # replace invalid chars with _
    normalized = re.sub(r"_+", "_", normalized).strip("_")  # collapse repeated _ and trim edges
    return normalized or "column"                            # fallback if result is empty

def suggest_schema_name(name: str) -> str:
    """Produce a valid snake_case rename suggestion for an invalid column name.
    Handles edge cases: leading digits are moved to the end ('3descrizione' -> 'descrizione_3'),
    leading underscores are stripped, and any remaining invalid name gets a 'col_' prefix."""
    normalized = normalized_schema_name(name)

    # Leading digit: move the numeric prefix to the end so the name starts with a letter
    if normalized[0].isdigit():
        match = re.match(r"(\d+)(.*)", normalized)
        if match:
            digits, remainder = match.groups()
            remainder = remainder.strip("_")
            normalized = f"{remainder}_{digits}" if remainder else f"col_{digits}"

    # Strip any residual leading underscores
    if normalized.startswith("_"):
        normalized = normalized.lstrip("_") or "column"

    # Last resort: prefix with col_ if still invalid
    if not is_valid_schema_name(normalized):
        normalized = f"col_{normalized}".strip("_")

    return normalized

def naming_rule_reason(name: str) -> str:
    """Build a human-readable explanation of why a column name is invalid.
    Enumerates all violations found (uppercase, whitespace, hyphens, %, leading digit)."""
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


# --- Profile Builders ---
def build_dtype_inference_text(df, n_rows: int = 10) -> str:
    """Serialize the dataset into a column-by-column text profile for the dtype inference agent.
    Each line contains the column name, up to n_rows unique non-null sample values,
    and the numeric/datetime parse percentages computed over the full column.
    The parse percentages give the LLM statistical context about the whole column,
    preventing dirty sample values from misleading the type inference."""
    lines: list[str] = []
    for column_name in df.columns:
        series = df[column_name]
        samples = (
            series
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .drop_duplicates()
            .head(n_rows)
            .tolist()
        )
        samples_text = ", ".join(f'"{v}"' for v in samples) if samples else "(no samples)"
        num_pct = compute_numeric_parse_pct(series)
        dt_pct = compute_datetime_parse_pct(series)
        lines.append(
            f"{column_name}: {samples_text} "
            f"[numeric_parse_pct={num_pct:.1f}%, datetime_parse_pct={dt_pct:.1f}%]"
        )
    return "\n".join(lines)

def build_dataset_profile(df, dataset_name: str, dtype_overrides: dict[str, str] | None = None) -> DatasetProfile:
    """Build a DatasetProfile from a raw DataFrame.
    If dtype_overrides is provided (from the LLM dtype inference agent),
    those values replace the raw pandas dtype for the relevant columns."""
    return DatasetProfile(
        dataset_name=dataset_name,
        total_rows=len(df),
        total_columns=len(df.columns),
        columns_profiles=[
            ColumnProfile(
                column_name=column_name,
                pandas_dtype=(dtype_overrides or {}).get(column_name, str(df[column_name].dtype)),
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


