from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
    metrics = [
        ("missing_like_cells", "Missing-like cells"),
        ("normalized_exact_duplicate_rows", "Exact duplicate rows"),
        ("unsafe_column_names", "Unsafe column names"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8))
    width = 0.55

    for ax, (key, label) in zip(axes, metrics):
        raw_value = int(quality.loc["raw", key])
        cleaned_value = int(quality.loc["cleaned", key])
        bars = ax.bar(["Raw", "Cleaned"], [raw_value, cleaned_value], width=width, color=["#8aa1c1", "#4c956c"])
        ax.set_title(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, [raw_value, cleaned_value]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:,}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.suptitle("Raw vs cleaned table-level quality signals", y=1.02)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "01_quality_signals.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_pipeline_counts_chart(validation_bundle: dict, final_report: dict, cleaner_manifest: list[dict]) -> Path:
    validation_counts = pd.Series(
        {
            "Schema issues": len(validation_bundle["schema_validation"].get("issues", [])),
            "Columns with missingness": len(validation_bundle["completeness_analysis"].get("columns_with_missing_values", [])),
            "Format findings": len(validation_bundle["consistency_validation"].get("format_consistency_findings", [])),
            "Anomaly findings": len(validation_bundle["anomaly_detection"].get("findings", [])),
            "Cross-column findings": len(validation_bundle["cross_column_validation"].get("findings", [])),
            "Duplicate groups": len(validation_bundle["duplicate_detection"].get("groups", [])),
        }
    )
    cleaning_counts = pd.Series(
        {
            "Applied actions": len(final_report.get("applied_actions", [])),
            "Proposed not applied": len(final_report.get("proposed_not_applied_actions", [])),
            "Manual review queue": len(final_report.get("manual_review_queue", [])),
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


def save_verification_chart(verification: pd.DataFrame) -> Path:
    verification = verification.copy()
    verification["display_name"] = verification["renamed_to"].fillna(verification["column_name"])
    verification = verification.sort_values("before_inconsistent_rows", ascending=True)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y = range(len(verification))
    height = 0.34

    before = verification["before_inconsistent_rows"].astype(int).tolist()
    after = verification["after_inconsistent_rows"].astype(int).tolist()

    ax.barh([i - height / 2 for i in y], before, height=height, label="Before cleaning", color="#d62828")
    ax.barh([i + height / 2 for i in y], after, height=height, label="After cleaning", color="#2a9d8f")

    ax.set_yticks(list(y))
    ax.set_yticklabels(verification["display_name"])
    ax.set_xlabel("Inconsistent rows")
    ax.set_title("Targeted format inconsistencies before and after cleaning")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2)

    for ypos, value in zip([i - height / 2 for i in y], before):
        ax.text(value, ypos, f" {value}", va="center", fontsize=9)
    for ypos, value in zip([i + height / 2 for i in y], after):
        ax.text(value, ypos, f" {value}", va="center", fontsize=9)

    fig.tight_layout()
    output_path = OUTPUT_DIR / "03_verification_before_after.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_cleaner_impact_chart(cleaners: pd.DataFrame) -> Path:
    cleaners = cleaners.sort_values("changed_rows", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.barh(cleaners["column_name"], cleaners["changed_rows"], color="#6d597a")
    ax.set_xlabel("Rows changed")
    ax.set_title("Accepted cleaner impact by column")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2)

    for ypos, value in enumerate(cleaners["changed_rows"].astype(int).tolist()):
        ax.text(value, ypos, f" {value}", va="center", fontsize=9)

    fig.tight_layout()
    output_path = OUTPUT_DIR / "04_cleaner_impact.png"
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

    output_paths = [
        save_quality_signals_chart(quality),
        save_pipeline_counts_chart(validation_bundle, final_report, cleaner_manifest),
        save_verification_chart(verification) if not verification.empty else None,
        save_cleaner_impact_chart(cleaners) if not cleaners.empty else None,
        save_results_summary_table(quality, final_report, verification, cleaners),
    ]

    for path in output_paths:
        if path is not None:
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
