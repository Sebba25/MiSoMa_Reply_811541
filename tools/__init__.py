"""Public facade for the ``tools`` package.

Thin re-export layer so callers can do ``from tools import X`` without
caring which specialised submodule (common / schema / format / completeness /
quality) defines ``X``. Only symbols actually imported by code outside the
``tools`` package are surfaced here — internal helpers stay private to their
module to keep this facade small and the public API obvious.
"""

from tools.common_tools import (
    PLACEHOLDER_TOKENS,
    attach_profile_text,
    attach_text_document,
    gzip_text_to_base64,
    load_dataset_frame,
    matches_numeric_schema_pattern,
    numeric_pattern_allows_variable_width,
    run_agent_with_backoff,
    run_agent_with_backoff_async,
    value_shape,
)
from tools.completeness_tools import build_completeness_profile
from tools.format_tools import (
    ColumnFormatFacts,
    FormatOutlierExample,
    build_column_format_facts,
)
from tools.schema_tools import (
    SchemaDuplicateGroup,
    build_dataset_profile,
    build_dtype_inference_text,
    is_valid_schema_name,
    naming_rule_reason,
    normalized_schema_name,
    suggest_schema_name,
)
from tools.quality_tools import (
    detect_date_order_violations,
    detect_duplicate_like_columns,
    detect_duplicate_semantic_conflicts,
    detect_exact_duplicate_groups,
    detect_near_duplicate_groups,
    detect_numeric_outlier_candidates,
    detect_rare_category_candidates,
    detect_year_month_period_mismatches,
    infer_duplicate_key_columns,
)

__all__ = [
    "PLACEHOLDER_TOKENS",
    "SchemaDuplicateGroup",
    "ColumnFormatFacts",
    "FormatOutlierExample",
    "attach_profile_text",
    "attach_text_document",
    "build_column_format_facts",
    "build_completeness_profile",
    "build_dataset_profile",
    "build_dtype_inference_text",
    "detect_date_order_violations",
    "detect_duplicate_like_columns",
    "detect_duplicate_semantic_conflicts",
    "detect_exact_duplicate_groups",
    "detect_near_duplicate_groups",
    "detect_numeric_outlier_candidates",
    "detect_rare_category_candidates",
    "detect_year_month_period_mismatches",
    "gzip_text_to_base64",
    "infer_duplicate_key_columns",
    "is_valid_schema_name",
    "load_dataset_frame",
    "matches_numeric_schema_pattern",
    "naming_rule_reason",
    "normalized_schema_name",
    "numeric_pattern_allows_variable_width",
    "run_agent_with_backoff",
    "run_agent_with_backoff_async",
    "suggest_schema_name",
    "value_shape",
]
