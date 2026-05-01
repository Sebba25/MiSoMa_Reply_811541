"""
Results Visualization for the Agents AI Pipeline

Generates findings and results charts from cached pipeline artifacts and CSV data.
Run with:  python visualize_results.py
Requires:  matplotlib, pandas (both in requirements.txt)

Outputs (images/findings/):
  01_quality_signals.png          - Duplicate rows dropped and unsafe names fixed (raw vs cleaned)
  02_placeholder_substitution.png - Per-column placeholder-like values converted to proper nulls
  03_pipeline_counts.png          - Validation findings and remediation action counts
  04_verification_outcomes.png    - Format inconsistencies eliminated per column with before/after and outcome
  05_anomaly_detection.png        - Anomaly findings by column with severity
  06_results_summary_table.png    - Quantitative run summary table
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET      = "spesa"
ROOT         = Path(__file__).resolve().parent
RAW_PATH     = ROOT / "Data" / f"{DATASET}.csv"
CLEANED_PATH = ROOT / "Data/.cleaning_cache" / DATASET / f"{DATASET}.cleaned.csv"
BUNDLE_PATH  = ROOT / "Data/.validation_cache" / f"{DATASET}.validation_bundle.json"
REPORT_PATH  = ROOT / "Data/.cleaning_cache" / DATASET / f"{DATASET}.final_report.json"
MANIFEST_PATH= ROOT / "Data/.cleaning_cache" / DATASET / "cleaner_manifest.json"
CLEANED_CONS = ROOT / "Data/.cleaning_cache" / DATASET / ".validation_cache" / f"{DATASET}.cleaned.consistency.json"
OUTPUT_DIR   = ROOT / "images" / "findings"

PLACEHOLDER_TOKENS   = {"", "na", "n/a", "null", "none", "-", "--", "unknown", "n.d.", "?", "//", "nan"}
VALID_SCHEMA_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

# Reply brand palette
PALETTE = {
    "primary":    "#22A30A",   # Reply primary green  — dominant bars / main series
    "mid":        "#5DC73A",   # Reply light green    — secondary bars / positive outcomes
    "lime":       "#9FE870",   # Reply lime           — accent / warning-tier
    "dark_green": "#1A8A00",   # Reply dark green     — alternative / deep positive
    "light_bg":   "#F4F4F4",   # Reply light grey     — axes background / neutral bars
    "dark_text":  "#2D2D2D",   # Reply dark grey      — titles / labels
    "mid_text":   "#555555",   # Reply medium grey    — secondary text / neutral bars
    "near_black": "#141414",   # Reply near-black     — table headers
    "white":      "#FFFFFF",
    "red":        "#d62828",   # error / regression
    "severity":   {"high": "#d62828", "medium": "#9FE870", "low": "#5DC73A"},
}

TITLE_SIZE = 13
LABEL_SIZE = 10
TICK_SIZE  = 9


def _style(fig: plt.Figure, *axes) -> None:
    """Apply Reply brand styling to a figure and its axes."""
    fig.patch.set_facecolor(PALETTE["white"])
    for ax in axes:
        ax.set_facecolor(PALETTE["light_bg"])
        ax.title.set_color(PALETTE["dark_text"])
        ax.xaxis.label.set_color(PALETTE["mid_text"])
        ax.yaxis.label.set_color(PALETTE["mid_text"])
        ax.tick_params(colors=PALETTE["mid_text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["mid_text"])


def _load(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# CSV-level helpers (used by charts 01, 02, 07)
# ---------------------------------------------------------------------------

def _missing_like_mask(df: pd.DataFrame) -> pd.DataFrame:
    rendered = df.astype("string").apply(lambda col: col.str.strip().str.lower())
    return df.isna() | rendered.isin(PLACEHOLDER_TOKENS)


def _normalized_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype("string").fillna("").apply(lambda col: col.str.strip().str.lower())


def _table_profile(label: str, df: pd.DataFrame) -> dict:
    mask        = _missing_like_mask(df)
    total_cells = int(df.shape[0] * df.shape[1])
    missing     = int(mask.to_numpy().sum())
    duplicates  = int(_normalized_rows(df).duplicated(keep="first").sum())
    unsafe      = sum(not VALID_SCHEMA_NAME_RE.fullmatch(c) for c in df.columns)
    return {
        "dataset":               label,
        "rows":                  int(df.shape[0]),
        "columns":               int(df.shape[1]),
        "cells":                 total_cells,
        "missing_like_cells":    missing,
        "missing_like_pct":      round(missing / total_cells * 100, 2) if total_cells else 0.0,
        "duplicate_rows":        duplicates,
        "unsafe_column_names":   unsafe,
    }


def _placeholder_string_count(series: pd.Series) -> int:
    return int(series.dropna().astype(str).str.strip().str.lower().isin(PLACEHOLDER_TOKENS).sum())


# ---------------------------------------------------------------------------
# 01. Quality signals — duplicate rows dropped and unsafe names fixed
# ---------------------------------------------------------------------------

def plot_quality_signals(quality: pd.DataFrame) -> None:
    dup_dropped   = max(0, int(quality.loc["raw", "duplicate_rows"])  - int(quality.loc["cleaned", "duplicate_rows"]))
    unsafe_fixed  = max(0, int(quality.loc["raw", "unsafe_column_names"]) - int(quality.loc["cleaned", "unsafe_column_names"]))

    labels = ["Duplicate rows\ndropped", "Unsafe column\nnames fixed"]
    values = [dup_dropped, unsafe_fixed]
    colors = [PALETTE["primary"], PALETTE["dark_text"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, width=0.5, color=colors, edgecolor="white", linewidth=0.8)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:,}", ha="center", va="bottom", fontsize=LABEL_SIZE, fontweight="bold")

    row_raw     = int(quality.loc["raw",     "rows"])
    row_cleaned = int(quality.loc["cleaned", "rows"])
    ax.annotate(
        f"Rows: {row_raw:,} → {row_cleaned:,}",
        xy=(0.98, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
    )

    ax.set_ylabel("Count", fontsize=LABEL_SIZE)
    ax.set_title("Resolved Table-level Quality Signals", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.3 + 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    _style(fig, ax)
    fig.tight_layout()
    _save(fig, "01_quality_signals.png")


# ---------------------------------------------------------------------------
# 02. Placeholder substitution — per-column values converted to proper nulls
# ---------------------------------------------------------------------------

def plot_placeholder_substitution(raw: pd.DataFrame, cleaned: pd.DataFrame, report: dict) -> None:
    rename_map: dict[str, str] = {
        a["target"]["column_name"]: a["target"]["new_name"]
        for a in report.get("applied_actions", [])
        if a.get("action_type") == "rename_column"
    }

    records = []
    for col in raw.columns:
        raw_ph = _placeholder_string_count(raw[col])
        if raw_ph == 0:
            continue
        cleaned_col = rename_map.get(col, col)
        if cleaned_col not in cleaned.columns:
            continue
        substituted = raw_ph - _placeholder_string_count(cleaned[cleaned_col])
        if substituted > 0:
            label = f"{col} → {cleaned_col}" if cleaned_col != col else col
            records.append({"column": label, "substituted": substituted})

    fig, ax = plt.subplots(figsize=(9, max(4, len(records) * 0.65 + 1.2)))

    if not records:
        ax.text(0.5, 0.5, "No placeholder substitutions detected",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        data = pd.DataFrame(records).sort_values("substituted", ascending=True)
        bars = ax.barh(data["column"], data["substituted"],
                       color=PALETTE["primary"], edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, data["substituted"].astype(int)):
            ax.text(bar.get_width() + max(data["substituted"]) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:,}", va="center", fontsize=TICK_SIZE, fontweight="bold")
        ax.set_xlabel("Cells converted to null", fontsize=LABEL_SIZE)
        ax.set_xlim(0, max(data["substituted"]) * 1.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    ax.set_title("Placeholder-like Values Converted to Proper Nulls, by Column",
                 fontsize=TITLE_SIZE, fontweight="bold")
    _style(fig, ax)
    fig.tight_layout()
    _save(fig, "02_placeholder_substitution.png")


# ---------------------------------------------------------------------------
# 03. Pipeline counts — findings detected vs actions taken
# ---------------------------------------------------------------------------

def plot_pipeline_counts(bundle: dict, report: dict, manifest: list) -> None:
    finding_labels = [
        "Schema\nissues",
        "Columns\nwith missingness",
        "Format\nfindings",
        "Anomaly\nfindings",
        "Cross-column\nfindings",
        "Duplicate\ngroups",
    ]
    finding_vals = [
        len(bundle["schema_validation"].get("issues", [])),
        len(bundle["completeness_analysis"].get("columns_with_missing_values", [])),
        len(bundle["consistency_validation"].get("format_consistency_findings", [])),
        len(bundle["anomaly_detection"].get("findings", [])),
        len(bundle["cross_column_validation"].get("findings", [])),
        len(bundle["duplicate_detection"].get("groups", [])),
    ]

    action_labels = ["Applied\nauto", "Proposed\nnot applied", "Manual\nreview", "Accepted\ncleaners", "Failed"]
    action_vals   = [
        len(report.get("applied_actions", [])),
        len(report.get("proposed_not_applied_actions", [])),
        len(report.get("manual_review_queue", [])),
        len(manifest),
        len(report.get("failed_actions", [])),
    ]
    action_colors = [PALETTE["primary"], PALETTE["mid_text"], PALETTE["lime"], PALETTE["mid"], PALETTE["red"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = ax1.barh(finding_labels[::-1], finding_vals[::-1],
                     color=PALETTE["dark_text"], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars1, finding_vals[::-1]):
        ax1.text(bar.get_width() + max(finding_vals) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 str(v), va="center", fontsize=TICK_SIZE, fontweight="bold")
    ax1.set_title("Validation-stage Finding Counts", fontsize=TITLE_SIZE, fontweight="bold")
    ax1.set_xlabel("Count", fontsize=LABEL_SIZE)
    ax1.set_xlim(0, max(finding_vals) * 1.25)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)

    bars2 = ax2.barh(action_labels[::-1], action_vals[::-1],
                     color=action_colors[::-1], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars2, action_vals[::-1]):
        ax2.text(bar.get_width() + max(action_vals) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 str(v), va="center", fontsize=TICK_SIZE, fontweight="bold")
    ax2.set_title("Remediation and Cleaning Decisions", fontsize=TITLE_SIZE, fontweight="bold")
    ax2.set_xlabel("Count", fontsize=LABEL_SIZE)
    ax2.set_xlim(0, max(action_vals) * 1.25)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)

    fig.suptitle(f"Pipeline Counts — {DATASET}.csv", fontsize=14, fontweight="bold",
                 color=PALETTE["dark_text"])
    _style(fig, ax1, ax2)
    fig.tight_layout()
    _save(fig, "03_pipeline_counts.png")


# ---------------------------------------------------------------------------
# 05. Anomaly detection — affected rows per finding with severity
# ---------------------------------------------------------------------------

def plot_anomaly_detection(bundle: dict) -> None:
    findings = bundle["anomaly_detection"].get("findings", [])
    fig, ax  = plt.subplots(figsize=(10, max(4, len(findings) * 0.9 + 1.5)))

    if not findings:
        ax.text(0.5, 0.5, "No anomaly findings detected",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        df = pd.DataFrame(findings).copy()
        df["label"] = df.apply(
            lambda r: f"{r['column_name']}  ({str(r['anomaly_type']).replace('_', ' ')})", axis=1
        )
        df = df.sort_values("affected_rows", ascending=True)
        colors = df["severity"].map(lambda s: PALETTE["severity"].get(str(s).lower(), PALETTE["mid_text"]))

        bars = ax.barh(df["label"], df["affected_rows"],
                       color=colors.tolist(), edgecolor="white", linewidth=0.8)
        for bar, (_, row) in zip(bars, df.iterrows()):
            v = int(row["affected_rows"])
            ax.text(bar.get_width() + df["affected_rows"].max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:,}", va="center", fontsize=TICK_SIZE, fontweight="bold")

        legend = [mpatches.Patch(color=c, label=s.title())
                  for s, c in PALETTE["severity"].items()]
        ax.legend(handles=legend, fontsize=TICK_SIZE, title="Severity", title_fontsize=TICK_SIZE)
        ax.set_xlabel("Affected rows", fontsize=LABEL_SIZE)
        ax.set_xlim(0, df["affected_rows"].max() * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    ax.set_title("Anomaly Findings Flagged for Review", fontsize=TITLE_SIZE, fontweight="bold")
    _style(fig, ax)
    fig.tight_layout()
    _save(fig, "05_anomaly_detection.png")


# ---------------------------------------------------------------------------
# 06. Verification outcomes — inconsistent rows eliminated per column
# ---------------------------------------------------------------------------

def plot_verification_outcomes(report: dict) -> None:
    diffs = report.get("verification_diffs", [])
    if not diffs:
        print("  [skip] no verification_diffs in report")
        return

    status_color = {
        "resolved":  PALETTE["primary"],
        "improved":  PALETTE["mid"],
        "unchanged": PALETTE["mid_text"],
        "regressed": PALETTE["red"],
    }

    rows = sorted(diffs, key=lambda d: d["before_inconsistent_rows"], reverse=True)
    cols      = [d["column_name"]              for d in rows]
    before    = [d["before_inconsistent_rows"] for d in rows]
    after     = [d["after_inconsistent_rows"]  for d in rows]
    status    = [d.get("status", "resolved")   for d in rows]
    eliminated= [b - a for b, a in zip(before, after)]
    colors    = [status_color.get(s, PALETTE["primary"]) for s in status]

    fig, ax = plt.subplots(figsize=(9, max(4, len(cols) * 0.75 + 1.5)))
    bars = ax.barh(cols, eliminated, color=colors, edgecolor="white", linewidth=0.8)

    for bar, b, a, s in zip(bars, before, after, status):
        pct = 100 * (b - a) / b if b else 0
        label = f"{b} → {a}  (−{pct:.0f}%  {s})"
        ax.text(bar.get_width() + max(eliminated) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=TICK_SIZE)

    # Legend for statuses actually present
    seen = dict.fromkeys(status)
    legend = [mpatches.Patch(color=status_color[s], label=s.title()) for s in seen]
    ax.legend(handles=legend, fontsize=TICK_SIZE, title="Outcome", title_fontsize=TICK_SIZE)

    total_eliminated = sum(eliminated)
    ax.annotate(
        f"Total inconsistent rows eliminated: {total_eliminated:,}",
        xy=(0.98, 0.04), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
    )

    ax.set_xlabel("Inconsistent rows eliminated", fontsize=LABEL_SIZE)
    ax.set_title("Verification: Format Inconsistencies Eliminated by Cleaning",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlim(0, max(eliminated) * 1.55)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    _style(fig, ax)
    fig.tight_layout()
    _save(fig, "04_verification_outcomes.png")


# ---------------------------------------------------------------------------
# 07. Results summary table
# ---------------------------------------------------------------------------

def plot_results_summary_table(quality: pd.DataFrame, report: dict, manifest: list) -> None:
    diffs = pd.DataFrame(report.get("verification_diffs", []))
    before_inc = int(diffs["before_inconsistent_rows"].sum()) if not diffs.empty else 0
    after_inc  = int(diffs["after_inconsistent_rows"].sum())  if not diffs.empty else 0
    reduction  = round((1 - after_inc / before_inc) * 100, 2) if before_inc else 0.0

    cleaners_df   = pd.DataFrame(manifest)
    rows_by_clean = int(cleaners_df["changed_rows"].sum()) if not cleaners_df.empty else 0

    summary = pd.DataFrame([
        ("Raw rows",                                    f"{int(quality.loc['raw',     'rows']):,}"),
        ("Cleaned rows",                                f"{int(quality.loc['cleaned', 'rows']):,}"),
        ("Raw missing-like cells",                      f"{int(quality.loc['raw',     'missing_like_cells']):,}"),
        ("Cleaned missing-like cells",                  f"{int(quality.loc['cleaned', 'missing_like_cells']):,}"),
        ("Raw exact duplicate rows",                    f"{int(quality.loc['raw',     'duplicate_rows']):,}"),
        ("Cleaned exact duplicate rows",                f"{int(quality.loc['cleaned', 'duplicate_rows']):,}"),
        ("Accepted cleaners",                           f"{len(manifest):,}"),
        ("Rows changed by cleaners",                    f"{rows_by_clean:,}"),
        ("Targeted inconsistent rows before cleaning",  f"{before_inc:,}"),
        ("Targeted inconsistent rows after cleaning",   f"{after_inc:,}"),
        ("Overall reduction on targeted inconsistencies", f"{reduction:.2f}%"),
        ("Applied actions",                             f"{len(report.get('applied_actions', [])):,}"),
    ], columns=["Metric", "Value"])

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ax.axis("off")
    table = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        cellLoc="left", colLoc="left", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(PALETTE["near_black"])
        else:
            cell.set_facecolor("#f6f7fb" if row % 2 == 0 else "white")
        cell.set_edgecolor("#d9dde7")

    ax.set_title("Quantitative Summary — Cached Pipeline Run on spesa.csv",
                 pad=18, fontsize=TITLE_SIZE, fontweight="bold", color=PALETTE["dark_text"])
    fig.patch.set_facecolor(PALETTE["white"])
    fig.tight_layout()
    _save(fig, "06_results_summary_table.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Generating result charts into '{OUTPUT_DIR.relative_to(ROOT)}/'")

    raw     = pd.read_csv(RAW_PATH,     dtype="string")
    cleaned = pd.read_csv(CLEANED_PATH, dtype="string")
    bundle  = _load(BUNDLE_PATH)
    report  = _load(REPORT_PATH)
    manifest= _load(MANIFEST_PATH)

    quality = pd.DataFrame([
        _table_profile("raw",     raw),
        _table_profile("cleaned", cleaned),
    ]).set_index("dataset")

    plot_quality_signals(quality)
    plot_placeholder_substitution(raw, cleaned, report)
    plot_pipeline_counts(bundle, report, manifest)
    plot_verification_outcomes(report)
    plot_anomaly_detection(bundle)
    plot_results_summary_table(quality, report, manifest)

    print("Done.")


if __name__ == "__main__":
    main()
