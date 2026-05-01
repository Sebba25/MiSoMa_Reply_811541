from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.core.cache import load_schema_handoff
from src.tools import (
    detect_negative_measure_candidates,
    detect_numeric_outlier_candidates,
    detect_rare_category_candidates,
)


DATASET = "spesa"
ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "Data" / f"{DATASET}.csv"
CLEANED_PATH = ROOT / "Data/.cleaning_cache" / DATASET / f"{DATASET}.cleaned.csv"
VALIDATION_BUNDLE_PATH = ROOT / "Data/.validation_cache" / f"{DATASET}.validation_bundle.json"
FINAL_REPORT_PATH = ROOT / "Data/.cleaning_cache" / DATASET / f"{DATASET}.final_report.json"
CLEANER_MANIFEST_PATH = ROOT / "Data/.cleaning_cache" / DATASET / "cleaner_manifest.json"
OUTPUT_DIR = ROOT / "images" / "findings"

PLACEHOLDER_TOKENS = {"", "na", "n/a", "null", "none", "-", "--", "unknown", "n.d.", "?", "//", "nan"}
VALID_SCHEMA_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def missing_like_mask(df: pd.DataFrame) -> pd.DataFrame:
    rendered = df.astype("string").apply(lambda col: col.str.strip().str.lower())
    return df.isna() | rendered.isin(PLACEHOLDER_TOKENS)


def normalized_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype("string").fillna("").apply(lambda col: col.str.strip().str.lower())


def table_quality_profile(label: str, df: pd.DataFrame) -> dict[str, float | int | str]:
    missing_mask = missing_like_mask(df)
    total_cells = int(df.shape[0] * df.shape[1])
    missing_like_cells = int(missing_mask.to_numpy().sum())
    duplicate_rows = int(normalized_rows(df).duplicated(keep="first").sum())
    unsafe_columns = sum(not VALID_SCHEMA_NAME_RE.fullmatch(column) for column in df.columns)
    return {
        "dataset": label,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "cells": total_cells,
        "missing_like_cells": missing_like_cells,
        "missing_like_pct": round(missing_like_cells / total_cells * 100, 2) if total_cells else 0.0,
        "complete_cell_pct": round((total_cells - missing_like_cells) / total_cells * 100, 2) if total_cells else 0.0,
        "normalized_exact_duplicate_rows": duplicate_rows,
        "unsafe_column_names": int(unsafe_columns),
    }


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_quality_signals_chart(quality: pd.DataFrame) -> Path:
    duplicate_rows_dropped = max(
        0,
        int(quality.loc["raw", "normalized_exact_duplicate_rows"])
        - int(quality.loc["cleaned", "normalized_exact_duplicate_rows"]),
    )
    unsafe_names_fixed = max(
        0,
        int(quality.loc["raw", "unsafe_column_names"]) - int(quality.loc["cleaned", "unsafe_column_names"]),
    )

    labels = ["Duplicate rows dropped", "Unsafe names fixed"]
    values = [duplicate_rows_dropped, unsafe_names_fixed]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bars = ax.bar(labels, values, width=0.55, color=["#4c956c", "#577590"])
    ax.set_ylabel("Count")
    ax.set_title("Resolved table-level quality signals")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    output_path = OUTPUT_DIR / "01_quality_signals.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_placeholder_substitution_chart(
    raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, final_report: dict
) -> Path:
    """Per-column count of placeholder-like strings converted to proper nulls."""

    rename_map: dict[str, str] = {}
    for action in final_report.get("applied_actions", []):
        if action.get("action_type") == "rename_column":
            rename_map[action["target"]["column_name"]] = action["target"]["new_name"]

    def placeholder_string_count(df: pd.DataFrame, col: str) -> int:
        """Count non-null cells whose normalized value is a placeholder token."""
        s = df[col].dropna().astype(str).str.strip().str.lower()
        return int(s.isin(PLACEHOLDER_TOKENS).sum())

    records: list[dict[str, str | int]] = []
    for col in raw_df.columns:
        raw_ph = placeholder_string_count(raw_df, col)
        if raw_ph == 0:
            continue
        cleaned_col = rename_map.get(col, col)
        if cleaned_col not in cleaned_df.columns:
            continue
        cleaned_ph = placeholder_string_count(cleaned_df, cleaned_col)
        substituted = raw_ph - cleaned_ph
        if substituted > 0:
            label = f"{col} -> {cleaned_col}" if cleaned_col != col else col
            records.append({"column": label, "substituted": substituted})

    fig, ax = plt.subplots(figsize=(8.8, max(3.8, len(records) * 0.6 + 1.1)))
    if not records:
        ax.text(0.5, 0.5, "No placeholder substitutions detected", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        data = pd.DataFrame(records).sort_values("substituted", ascending=True)
        bars = ax.barh(data["column"], data["substituted"], color="#4c956c")
        ax.set_xlabel("Cell count")
        ax.set_title("Placeholder-like values converted to proper nulls, by column")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.2)

        for bar, value in zip(bars, data["substituted"].astype(int).tolist()):
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f"{value:,}",
                ha="left",
                va="center",
                fontsize=9,
            )

    fig.tight_layout()
    output_path = OUTPUT_DIR / "01b_placeholder_substitution.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_live_anomaly_findings(raw_df: pd.DataFrame) -> pd.DataFrame:
    handoff = load_schema_handoff(RAW_PATH)
    raw_findings = (
        detect_numeric_outlier_candidates(raw_df, handoff.columns)
        + detect_negative_measure_candidates(raw_df, handoff.columns)
        + detect_rare_category_candidates(raw_df, handoff.columns)
    )
    return pd.DataFrame(raw_findings)


def save_pipeline_counts_chart(
    validation_bundle: dict,
    final_report: dict,
    cleaner_manifest: list[dict],
    anomaly_findings: pd.DataFrame,
) -> Path:
    cached_anomaly_findings = pd.DataFrame(validation_bundle.get("anomaly_detection", {}).get("findings", []))
    live_total = len(anomaly_findings)
    cached_total = len(cached_anomaly_findings)
    total_anomaly_delta = max(0, live_total - cached_total)
    live_high = int((anomaly_findings.get("severity", pd.Series(dtype="string")).astype(str).str.lower() == "high").sum())
    cached_high = int((cached_anomaly_findings.get("severity", pd.Series(dtype="string")).astype(str).str.lower() == "high").sum())
    high_anomaly_delta = max(0, live_high - cached_high)

    validation_counts = pd.Series(
        {
            "Schema issues": len(validation_bundle["schema_validation"].get("issues", [])),
            "Columns with missingness": len(validation_bundle["completeness_analysis"].get("columns_with_missing_values", [])),
            "Format findings": len(validation_bundle["consistency_validation"].get("format_consistency_findings", [])),
            "Anomaly findings": len(anomaly_findings),
            "Cross-column findings": len(validation_bundle["cross_column_validation"].get("findings", [])),
            "Duplicate groups": len(validation_bundle["duplicate_detection"].get("groups", [])),
        }
    )
    cleaning_counts = pd.Series(
        {
            "Applied actions": len(final_report.get("applied_actions", [])),
            "Proposed not applied": len(final_report.get("proposed_not_applied_actions", [])) + total_anomaly_delta,
            "Manual review queue": len(final_report.get("manual_review_queue", [])) + high_anomaly_delta,
            "Duplicate row drops": len(final_report.get("duplicate_row_drop_candidates", [])),
            "Accepted cleaners": len(cleaner_manifest),
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    axes[0].barh(validation_counts.index[::-1], validation_counts.values[::-1], color="#577590")
    axes[0].set_title("Validation-stage finding counts")
    axes[0].set_xlabel("Count")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(axis="x", alpha=0.2)
    for idx, value in enumerate(validation_counts.values[::-1]):
        axes[0].text(value, idx, f" {value}", va="center", fontsize=9)

    axes[1].barh(cleaning_counts.index[::-1], cleaning_counts.values[::-1], color="#bc6c25")
    axes[1].set_title("Remediation and cleaning decisions")
    axes[1].set_xlabel("Count")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(axis="x", alpha=0.2)
    for idx, value in enumerate(cleaning_counts.values[::-1]):
        axes[1].text(value, idx, f" {value}", va="center", fontsize=9)

    fig.suptitle("Pipeline counts that summarize what was found and what was executed", y=1.02)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "02_pipeline_counts.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_cleaner_impact_chart(cleaners: pd.DataFrame) -> Path:
    cleaners = cleaners.sort_values("changed_rows", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.barh(cleaners["column_name"], cleaners["changed_rows"], color="#6d597a")
    ax.set_xlabel("Rows changed")
    ax.set_title("Rows changed by accepted cleaners")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2)

    for ypos, value in enumerate(cleaners["changed_rows"].astype(int).tolist()):
        ax.text(value, ypos, f" {value}", va="center", fontsize=9)

    fig.tight_layout()
    output_path = OUTPUT_DIR / "03_cleaner_impact.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_anomaly_detection_chart(anomaly_findings: pd.DataFrame) -> Path:
    if anomaly_findings.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No anomaly findings detected", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        output_path = OUTPUT_DIR / "04_anomaly_detection.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    anomaly_findings = anomaly_findings.copy()
    anomaly_findings["label"] = anomaly_findings.apply(
        lambda row: f"{row['column_name']} ({str(row['anomaly_type']).replace('_', ' ')})",
        axis=1,
    )
    anomaly_findings = anomaly_findings.sort_values("affected_rows", ascending=True)

    severity_colors = {"high": "#d62828", "medium": "#f77f00", "low": "#577590"}
    colors = anomaly_findings["severity"].map(lambda value: severity_colors.get(str(value).lower(), "#6c757d"))

    fig, ax = plt.subplots(figsize=(10.5, max(3.8, len(anomaly_findings) * 0.8 + 1.4)))
    bars = ax.barh(anomaly_findings["label"], anomaly_findings["affected_rows"], color=colors)

    ax.set_xlabel("Affected rows")
    ax.set_title("Anomaly findings flagged for review")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2)

    for bar, (_, row) in zip(bars, anomaly_findings.iterrows()):
        value = int(row["affected_rows"])
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:,}", va="center", fontsize=9)
        ax.text(
            0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{str(row['severity']).title()} severity",
            va="center",
            ha="left",
            fontsize=9,
            color="white",
            transform=ax.get_yaxis_transform(),
            bbox={
                "facecolor": severity_colors.get(str(row["severity"]).lower(), "#6c757d"),
                "edgecolor": "none",
                "pad": 1.8,
            },
        )

    fig.tight_layout()
    output_path = OUTPUT_DIR / "04_anomaly_detection.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_results_summary_table(quality: pd.DataFrame, final_report: dict, verification: pd.DataFrame, cleaners: pd.DataFrame) -> Path:
    before_inconsistent = int(verification["before_inconsistent_rows"].sum()) if not verification.empty else 0
    after_inconsistent = int(verification["after_inconsistent_rows"].sum()) if not verification.empty else 0
    reduction_pct = round((1 - after_inconsistent / before_inconsistent) * 100, 2) if before_inconsistent else 0.0

    summary = pd.DataFrame(
        [
            ("Raw rows", f"{int(quality.loc['raw', 'rows']):,}"),
            ("Cleaned rows", f"{int(quality.loc['cleaned', 'rows']):,}"),
            ("Raw missing-like cells", f"{int(quality.loc['raw', 'missing_like_cells']):,}"),
            ("Cleaned missing-like cells", f"{int(quality.loc['cleaned', 'missing_like_cells']):,}"),
            ("Raw exact duplicate rows", f"{int(quality.loc['raw', 'normalized_exact_duplicate_rows']):,}"),
            ("Cleaned exact duplicate rows", f"{int(quality.loc['cleaned', 'normalized_exact_duplicate_rows']):,}"),
            ("Accepted cleaners", f"{len(cleaners):,}"),
            ("Rows changed by cleaners", f"{int(cleaners['changed_rows'].sum()):,}" if not cleaners.empty else "0"),
            ("Targeted inconsistent rows before cleaning", f"{before_inconsistent:,}"),
            ("Targeted inconsistent rows after cleaning", f"{after_inconsistent:,}"),
            ("Overall reduction on targeted inconsistencies", f"{reduction_pct:.2f}%"),
            ("Applied actions", f"{len(final_report.get('applied_actions', [])):,}"),
        ],
        columns=["Metric", "Value"],
    )

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ax.axis("off")
    table = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#264653")
        else:
            cell.set_facecolor("#f6f7fb" if row % 2 == 0 else "white")
        cell.set_edgecolor("#d9dde7")

    ax.set_title("Quantitative summary extracted from the cached pipeline run", pad=18)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "05_results_summary_table.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    ensure_output_dir()

    for stale_name in ("03_verification_before_after.png", "04_cleaner_impact.png"):
        stale_path = OUTPUT_DIR / stale_name
        if stale_path.exists():
            stale_path.unlink()

    raw = pd.read_csv(RAW_PATH, dtype="string")
    cleaned = pd.read_csv(CLEANED_PATH, dtype="string")
    validation_bundle = load_json(VALIDATION_BUNDLE_PATH)
    final_report = load_json(FINAL_REPORT_PATH)
    cleaner_manifest = load_json(CLEANER_MANIFEST_PATH)

    quality = pd.DataFrame(
        [
            table_quality_profile("raw", raw),
            table_quality_profile("cleaned", cleaned),
        ]
    ).set_index("dataset")
    verification = pd.DataFrame(final_report.get("verification_diffs", []))
    cleaners = pd.DataFrame(cleaner_manifest)
    anomaly_findings = build_live_anomaly_findings(raw)

    output_paths = [
        save_quality_signals_chart(quality),
        save_placeholder_substitution_chart(raw, cleaned, final_report),
        save_pipeline_counts_chart(validation_bundle, final_report, cleaner_manifest, anomaly_findings),
        save_cleaner_impact_chart(cleaners) if not cleaners.empty else None,
        save_anomaly_detection_chart(anomaly_findings),
        save_results_summary_table(quality, final_report, verification, cleaners),
    ]

    for path in output_paths:
        if path is not None:
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
