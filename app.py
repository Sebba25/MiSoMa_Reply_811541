from __future__ import annotations

import io
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents import setup_logfire
from cache import load_validation_results
from cleaning_core.legacy import (
    _build_cleaning_requests,
    _resolve_remediation_plan,
)
from cleaning_core.paths import cleaned_dataset_path, final_report_path
from cleaning_core.reporting import (
    build_final_report,
    generate_narrative_report,
    narrative_report_path,
    save_final_report,
    save_narrative_report,
)
from cleaning_core.verification import run_verify
from cleaning_core.application import run_cleaner_application_with_plan
from cleaning_core.generation import run_cleaner_generation
from models import FinalPipelineReport
from pipeline import (
    build_validation_results,
    run_anomaly_detection,
    run_completeness_analysis,
    run_cross_column_validation,
    run_duplicate_detection,
    run_dtype_inference,
    run_format_consistency_validation,
    run_schema_validation,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NoiPA — Data Quality Atelier",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Custom styling — editorial / data-journalism aesthetic
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;0,9..144,700;0,9..144,900;1,9..144,400&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');

:root {
    --paper: #f5f1ea;
    --paper-deep: #ebe4d6;
    --ink: #1a1613;
    --ink-soft: #3a332d;
    --ink-muted: #8a7f72;
    --accent: #b54420;
    --accent-soft: #d97a4a;
    --success: #5a7a3a;
    --warning: #c4962b;
    --rule: #c7bfb2;
}

html, body, [class*="css"], .stApp {
    background-color: var(--paper) !important;
    color: var(--ink) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

.stApp {
    background-image:
        radial-gradient(circle at 15% 5%, rgba(181, 68, 32, 0.04) 0%, transparent 40%),
        radial-gradient(circle at 85% 95%, rgba(26, 22, 19, 0.03) 0%, transparent 40%);
    background-attachment: fixed;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1200px !important;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Fraunces', serif !important;
    color: var(--ink) !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
}

h1 { font-weight: 400 !important; }

/* Editorial masthead */
.masthead {
    border-top: 2px solid var(--ink);
    border-bottom: 1px solid var(--rule);
    padding: 1.2rem 0 1rem 0;
    margin-bottom: 2.5rem;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 1rem;
}
.masthead .title {
    font-family: 'Fraunces', serif;
    font-weight: 400;
    font-size: 2.6rem;
    line-height: 1;
    letter-spacing: -0.025em;
    font-style: italic;
}
.masthead .title em {
    color: var(--accent);
    font-style: italic;
}
.masthead .meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--ink-muted);
    text-align: right;
}
.masthead .meta b { color: var(--ink); font-weight: 500; }

/* Kicker / overline */
.kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--accent);
    margin-bottom: 0.6rem;
    display: inline-block;
    border-top: 1px solid var(--accent);
    padding-top: 0.4rem;
}

/* Section heads */
.section-head {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin: 2.5rem 0 1.4rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--rule);
}
.section-head .num {
    font-family: 'Fraunces', serif;
    font-size: 2.4rem;
    font-weight: 300;
    font-style: italic;
    color: var(--accent);
    line-height: 1;
    min-width: 3rem;
}
.section-head .title {
    font-family: 'Fraunces', serif;
    font-size: 1.7rem;
    font-weight: 400;
    letter-spacing: -0.015em;
    line-height: 1.1;
}
.section-head .title span {
    display: block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--ink-muted);
    margin-top: 0.3rem;
}

/* Pull quote / lede */
.lede {
    font-family: 'Fraunces', serif;
    font-size: 1.2rem;
    line-height: 1.55;
    color: var(--ink-soft);
    font-weight: 300;
    font-style: italic;
    padding: 0.5rem 0 0.5rem 1.5rem;
    border-left: 2px solid var(--accent);
    margin-bottom: 1.6rem;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 1px;
    background: var(--rule);
    border: 1px solid var(--rule);
    margin-bottom: 1.5rem;
}
.metric {
    background: var(--paper);
    padding: 1.1rem 1.2rem;
}
.metric .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--ink-muted);
    margin-bottom: 0.5rem;
}
.metric .value {
    font-family: 'Fraunces', serif;
    font-size: 2.2rem;
    font-weight: 400;
    line-height: 1;
    color: var(--ink);
}
.metric .unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--ink-muted);
    margin-left: 0.3rem;
}
.metric.accent .value { color: var(--accent); }
.metric.success .value { color: var(--success); }
.metric.warning .value { color: var(--warning); }

/* Stage strip — 6-column grid, 2 rows of 6 */
.stages {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    margin-bottom: 0.5rem;
    border: 1px solid var(--rule);
    border-bottom: none;
}
.stage {
    padding: 0.65rem 0.85rem;
    border-right: 1px solid var(--rule);
    border-bottom: 1px solid var(--rule);
    background: var(--paper);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    position: relative;
    transition: background 0.3s ease;
}
.stage:nth-child(6n) { border-right: none; }
.stage .num {
    font-size: 0.6rem;
    color: var(--ink-muted);
    display: block;
    margin-bottom: 0.3rem;
}
.stage .name { color: var(--ink); font-weight: 500; }
.stage.done { background: var(--paper-deep); }
.stage.done .name::before { content: "● "; color: var(--success); }
.stage.active { background: #fff; }
.stage.active .name::before {
    content: "◐ ";
    color: var(--accent);
    animation: pulse 1.2s infinite;
}
.stage.pending .name { color: var(--ink-muted); }
.stage.pending .name::before { content: "○ "; color: var(--ink-muted); }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Buttons */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.2em !important;
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.8rem 2rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: var(--rule) !important;
    color: var(--ink-muted) !important;
}

/* Download button override */
.stDownloadButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.2em !important;
    background: transparent !important;
    color: var(--ink) !important;
    border: 1px solid var(--ink) !important;
    border-radius: 0 !important;
    padding: 0.7rem 1.6rem !important;
}
.stDownloadButton > button:hover {
    background: var(--ink) !important;
    color: var(--paper) !important;
}

/* File uploader dropzone */
[data-testid="stFileUploaderDropzone"] {
    background: var(--paper-deep) !important;
    border: 1px dashed var(--ink-muted) !important;
    border-radius: 0 !important;
    padding: 1.2rem 1rem !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: #fff !important;
}
/* "Browse files" button — override both normal & nested text */
[data-testid="stFileUploaderDropzone"] button {
    background: var(--accent) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
    padding: 0.55rem 1.2rem !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploaderDropzone"] button span,
[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] button div {
    color: var(--paper) !important;
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: var(--ink) !important;
}
/* File size hint text ("200MB per file · CSV") */
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--ink-soft) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    background: transparent !important;
}

/* Select / inputs */
.stSelectbox > div > div, .stTextInput > div > div > input {
    background: #fff !important;
    border-radius: 0 !important;
    border: 1px solid var(--rule) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--ink) !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: #fff !important;
    color: var(--ink) !important;
}
.stSelectbox [data-baseweb="select"] [class*="ValueContainer"],
.stSelectbox [data-baseweb="select"] div[role="button"],
.stSelectbox [data-baseweb="select"] input {
    color: var(--ink) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
/* Placeholder styling */
.stSelectbox [data-baseweb="select"] [class*="placeholder"],
.stSelectbox [data-baseweb="select"] input::placeholder {
    color: var(--ink-muted) !important;
    font-style: italic !important;
    font-weight: 400 !important;
    opacity: 1 !important;
}

/* Selectbox dropdown arrow colour */
.stSelectbox svg { fill: var(--accent) !important; }

/* Dropdown popup — override Streamlit's dark theme */
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul {
    background: #fff !important;
    border: 1px solid var(--ink) !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0 rgba(26,22,19,0.08) !important;
}
[data-baseweb="popover"] [role="option"] {
    background: #fff !important;
    color: var(--ink) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    padding: 0.6rem 1rem !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background: var(--paper-deep) !important;
    color: var(--accent) !important;
}

/* Tables */
.stDataFrame {
    border: 1px solid var(--rule);
    border-radius: 0;
}
[data-testid="stTable"] table, .stDataFrame table {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}
[data-testid="stTable"] thead th, .stDataFrame thead th {
    background: var(--paper-deep) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--ink) !important;
    border-bottom: 1px solid var(--ink) !important;
}

/* Status boxes — high-specificity overrides with .stApp prefix to defeat emotion CSS */
.stApp details.stStatus,
.stApp details[data-testid="stStatus"],
.stApp [data-testid="stStatus"],
.stApp details.stStatus > summary,
.stApp details[data-testid="stStatus"] > summary,
.stApp details.stStatus summary,
.stApp details[data-testid="stStatus"] summary,
.stApp details[data-testid="stStatus"][open],
.stApp details[data-testid="stStatus"][open] > summary,
.stApp [data-testid="stStatus"][data-state="running"],
.stApp [data-testid="stStatus"][data-state="running"] summary,
.stApp [data-testid="stStatus"][data-state="running"] > summary,
.stApp [data-testid="stStatus"][data-state="complete"],
.stApp [data-testid="stStatus"][data-state="complete"] summary,
.stApp [data-testid="stStatus"][data-state="complete"] > summary,
.stApp [data-testid="stStatus"][data-state="error"],
.stApp [data-testid="stStatus"][data-state="error"] summary,
.stApp [data-testid="stStatus"][data-state="error"] > summary {
    background: var(--paper-deep) !important;
    background-color: var(--paper-deep) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
    border: 1px solid var(--rule) !important;
}

/* Force every descendant (including summary text nodes) to ink on transparent */
.stApp [data-testid="stStatus"] *:not(svg):not(path):not(g):not(circle):not(line):not(rect),
.stApp [data-testid="stStatus"] summary *,
.stApp [data-testid="stStatus"] > summary * {
    background-color: transparent !important;
    color: var(--ink) !important;
}
details.stStatus > summary,
details[data-testid="stStatus"] > summary {
    padding: 0.75rem 1rem !important;
    border: none !important;
    border-bottom: 1px solid var(--rule) !important;
}
details.stStatus[open] > summary,
details[data-testid="stStatus"][open] > summary {
    border-bottom: 1px solid var(--rule) !important;
}
details.stStatus summary,
details.stStatus summary *,
details[data-testid="stStatus"] summary,
details[data-testid="stStatus"] summary *,
[data-testid="stStatus"] label,
[data-testid="stStatus"] label * {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--ink) !important;
    background: transparent !important;
}
/* Expanded body content */
details.stStatus > div,
details[data-testid="stStatus"] > div,
[data-testid="stStatus"] [data-testid="stStatusContent"],
[data-testid="stStatus"] [data-testid="stVerticalBlock"] {
    background: var(--paper-deep) !important;
    padding: 0.6rem 1rem !important;
}
[data-testid="stStatus"] p,
[data-testid="stStatus"] span:not(svg *),
[data-testid="stStatus"] div[data-testid*="stMarkdown"] * {
    color: var(--ink-soft) !important;
    background: transparent !important;
}
/* Status icon/spinner colour */
[data-testid="stStatus"] svg { color: var(--accent) !important; }

/* Captions — ensure readable contrast */
.stCaption, [data-testid="stCaptionContainer"], small {
    color: var(--ink-soft) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* Info / success / error boxes */
.stAlert {
    border-radius: 0 !important;
    border-left: 3px solid var(--accent) !important;
    background: #fff !important;
    color: var(--ink) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.stAlert [data-testid="stAlertContentInfo"],
.stAlert [data-testid="stAlertContentSuccess"],
.stAlert [data-testid="stAlertContentError"] {
    color: var(--ink) !important;
}

/* Dataframe cell text */
.stDataFrame tbody td, [data-testid="stTable"] tbody td {
    color: var(--ink) !important;
}

/* Progress bar — sand track, green fill, transparent label */
.stApp .stProgress,
.stApp [data-testid="stProgress"],
.stApp .stProgress > div,
.stApp [data-testid="stProgress"] > div,
.stApp .stProgress p,
.stApp [data-testid="stProgress"] p,
.stApp .stProgress label,
.stApp [data-testid="stProgress"] label {
    background: transparent !important;
    background-color: transparent !important;
    border-radius: 0 !important;
}
.stApp .stProgress > div > div,
.stApp [data-testid="stProgress"] > div > div {
    background: #e8dfcb !important;
    background-color: #e8dfcb !important;
    border-radius: 0 !important;
}
.stApp .stProgress > div > div > div,
.stApp .stProgress > div > div > div > div,
.stApp [data-testid="stProgress"] > div > div > div,
.stApp [data-testid="stProgress"] div[role="progressbar"] > div,
.stApp .stProgress div[role="progressbar"] > div {
    background: var(--success) !important;
    background-color: var(--success) !important;
    background-image: none !important;
    border-radius: 0 !important;
    transition: width 0.4s ease !important;
}
.stApp .stProgress p,
.stApp [data-testid="stProgress"] p {
    color: var(--ink) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    padding: 0 !important;
    margin-top: 0.3rem !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--rule) !important;
    gap: 0 !important;
    background: transparent !important;
}
[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    background: transparent !important;
    border-radius: 0 !important;
    color: var(--ink-muted) !important;
    padding: 0.8rem 1.6rem !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: var(--ink) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* Narrative body */
.narrative {
    font-family: 'Fraunces', serif;
    font-size: 1.02rem;
    line-height: 1.65;
    color: var(--ink-soft);
    column-gap: 2.5rem;
}
.narrative h1, .narrative h2, .narrative h3 {
    font-family: 'Fraunces', serif !important;
    color: var(--ink) !important;
}
.narrative h2 {
    font-size: 1.4rem !important;
    margin-top: 2rem !important;
    padding-top: 0.6rem;
    border-top: 1px solid var(--rule);
    font-weight: 500 !important;
}
.narrative h3 {
    font-size: 1.05rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--accent) !important;
    margin-top: 1.3rem !important;
}
.narrative table {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
}
.narrative th {
    background: var(--paper-deep);
    text-align: left;
    padding: 0.5rem 0.8rem;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid var(--ink);
}
.narrative td {
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid var(--rule);
}

/* Dataset preview panel */
.panel {
    background: #fff;
    border: 1px solid var(--rule);
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.2rem;
}
.panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--ink-muted);
    margin-bottom: 0.6rem;
}

/* Sidebar CTA bar */
.action-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--ink);
    color: var(--paper);
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
}
.action-bar .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--paper-deep);
}
.action-bar .value {
    font-family: 'Fraunces', serif;
    font-size: 1.1rem;
    font-style: italic;
}

/* Edge ornament */
.ornament {
    text-align: center;
    font-family: 'Fraunces', serif;
    color: var(--accent);
    font-size: 1.4rem;
    margin: 2.5rem 0;
    letter-spacing: 1rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    ("01", "Dtype", "dtype inference"),
    ("02", "Schema", "naming & duplicates"),
    ("03", "Completeness", "missing & placeholders"),
    ("04", "Consistency", "format validation"),
    ("05", "Anomaly", "outliers & rare values"),
    ("06", "Cross-Column", "semantic checks"),
    ("07", "Duplicates", "row-level detection"),
    ("08", "Remediate", "action planning"),
    ("09", "Generate", "cleaner synthesis"),
    ("10", "Apply", "execution"),
    ("11", "Verify", "post-clean audit"),
    ("12", "Narrative", "report writing"),
]


def _init_state() -> None:
    defaults = {
        "dataset_path": None,
        "dataset_name": None,
        "pipeline_done": False,
        "current_stage": None,
        "completed_stages": set(),
        "final_report": None,
        "narrative_report": None,
        "validation_results": None,
        "logfire_setup": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _masthead() -> None:
    now = datetime.now().strftime("%d %b %Y · %H:%M").upper()
    st.markdown(
        f"""
        <div class="masthead">
            <div class="title">NoiPA · <em>Data Quality</em> Atelier</div>
            <div class="meta">
                No. {datetime.now().strftime("%Y.%m")}<br/>
                <b>{now}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section(number: str, title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="section-head">
            <div class="num">{number}</div>
            <div class="title">{title}<span>{caption}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _stage_strip(current: str | None, completed: set[str]) -> str:
    html = ['<div class="stages">']
    for num, name, _caption in PIPELINE_STAGES:
        if name in completed:
            status = "done"
        elif name == current:
            status = "active"
        else:
            status = "pending"
        html.append(
            f'<div class="stage {status}"><span class="num">{num}</span>'
            f'<span class="name">{name}</span></div>'
        )
    html.append("</div>")
    return "".join(html)


def _metric(label: str, value: str, unit: str = "", variant: str = "") -> str:
    variant_cls = f" {variant}" if variant else ""
    unit_html = f'<span class="unit">{unit}</span>' if unit else ""
    return (
        f'<div class="metric{variant_cls}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}{unit_html}</div>'
        f'</div>'
    )


def _metric_grid(metrics: list[str]) -> None:
    st.markdown(f'<div class="metric-grid">{"".join(metrics)}</div>', unsafe_allow_html=True)


def _save_uploaded_file(uploaded) -> Path:
    target_dir = Path(__file__).parent / "Data"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / uploaded.name
    target.write_bytes(uploaded.getbuffer())
    return target


def _list_available_datasets() -> list[Path]:
    data_dir = Path(__file__).parent / "Data"
    if not data_dir.exists():
        return []
    return sorted(p for p in data_dir.glob("*.csv") if p.is_file())


# ---------------------------------------------------------------------------
# Pipeline runner — executes stages sequentially with live status
# ---------------------------------------------------------------------------

def _render_progress(n_done: int, current: str | None, strip_ph, prog_ph) -> None:
    strip_ph.markdown(
        _stage_strip(current, st.session_state.completed_stages),
        unsafe_allow_html=True,
    )
    pct = int(n_done / len(PIPELINE_STAGES) * 100)
    label = f"Stage {n_done} of {len(PIPELINE_STAGES)} · {current or 'Done'} · {pct}%"
    prog_ph.progress(pct / 100, text=label)


def _mark_stage(name: str, strip_ph, prog_ph) -> None:
    st.session_state.current_stage = name
    _render_progress(len(st.session_state.completed_stages), name, strip_ph, prog_ph)


def _complete_stage(name: str, strip_ph, prog_ph) -> None:
    st.session_state.completed_stages.add(name)
    st.session_state.current_stage = None
    _render_progress(len(st.session_state.completed_stages), None, strip_ph, prog_ph)


def run_full_pipeline(dataset_path: Path, strip_ph, prog_ph, log_container) -> None:
    st.session_state.completed_stages = set()
    st.session_state.current_stage = None
    _render_progress(0, None, strip_ph, prog_ph)

    def stage_banner(stage_name: str, caption: str):
        return log_container.status(
            f"▸ {stage_name.upper()}  ·  {caption}", expanded=True
        )

    # Stage 1: dtype inference
    _mark_stage("Dtype", strip_ph, prog_ph)
    with stage_banner("Dtype", "inferring pandas dtypes from sample values") as s:
        handoff = run_dtype_inference(dataset_path)
        s.write(f"Inferred dtypes for {len(handoff.columns)} columns.")
    _complete_stage("Dtype", strip_ph, prog_ph)

    # Stage 2: schema
    _mark_stage("Schema", strip_ph, prog_ph)
    with stage_banner("Schema", "validating names & detecting duplicate columns") as s:
        schema = run_schema_validation(dataset_path)
        n_issues = len(schema.issues)
        s.write(f"{n_issues} schema issue{'s' if n_issues != 1 else ''} found.")
    _complete_stage("Schema", strip_ph, prog_ph)

    # Stage 3: completeness
    _mark_stage("Completeness", strip_ph, prog_ph)
    with stage_banner("Completeness", "detecting missing values & placeholder tokens") as s:
        completeness = run_completeness_analysis(dataset_path)
        n_cols = len(completeness.columns_with_missing_values)
        s.write(f"{n_cols} column{'s' if n_cols != 1 else ''} with missing or placeholder values.")
    _complete_stage("Completeness", strip_ph, prog_ph)

    # Stage 4: consistency
    _mark_stage("Consistency", strip_ph, prog_ph)
    with stage_banner("Consistency", "per-column format validation") as s:
        consistency = run_format_consistency_validation(dataset_path)
        n_findings = len(consistency.format_consistency_findings)
        s.write(f"{n_findings} format inconsistenc{'ies' if n_findings != 1 else 'y'} detected.")
    _complete_stage("Consistency", strip_ph, prog_ph)

    # Stage 5: anomaly
    _mark_stage("Anomaly", strip_ph, prog_ph)
    with stage_banner("Anomaly", "numeric outliers & rare categories") as s:
        anomaly = run_anomaly_detection(dataset_path)
        n_anom = len(anomaly.findings) if anomaly else 0
        s.write(f"{n_anom} anomaly finding{'s' if n_anom != 1 else ''}.")
    _complete_stage("Anomaly", strip_ph, prog_ph)

    # Stage 6: cross-column
    _mark_stage("Cross-Column", strip_ph, prog_ph)
    with stage_banner("Cross-Column", "semantic conflict detection") as s:
        cross_column = run_cross_column_validation(dataset_path)
        n_cc = len(cross_column.findings) if cross_column else 0
        s.write(f"{n_cc} cross-column finding{'s' if n_cc != 1 else ''}.")
    _complete_stage("Cross-Column", strip_ph, prog_ph)

    # Stage 7: duplicate rows
    _mark_stage("Duplicates", strip_ph, prog_ph)
    with stage_banner("Duplicates", "row-level exact & near duplicate groups") as s:
        duplicates = run_duplicate_detection(dataset_path)
        n_dup = len(duplicates.groups) if duplicates else 0
        s.write(f"{n_dup} duplicate group{'s' if n_dup != 1 else ''} identified.")
    _complete_stage("Duplicates", strip_ph, prog_ph)

    # Build validation bundle (reuse cached)
    validation_results = build_validation_results(
        dataset_path,
        reuse_schema=True,
        reuse_completeness=True,
        reuse_consistency=True,
    )
    st.session_state.validation_results = validation_results

    # Stage 8: remediation planning
    _mark_stage("Remediate", strip_ph, prog_ph)
    with stage_banner("Remediate", "mapping findings to concrete actions") as s:
        remediation_plan = _resolve_remediation_plan(
            dataset_path,
            validation_results,
            None,
            reuse_saved_validation=False,
            reuse_saved_remediation=False,
        )
        n_actions = len(remediation_plan.actions) if remediation_plan else 0
        s.write(f"{n_actions} remediation action{'s' if n_actions != 1 else ''} planned.")
    _complete_stage("Remediate", strip_ph, prog_ph)

    # Stage 9: cleaner generation
    _mark_stage("Generate", strip_ph, prog_ph)
    with stage_banner("Generate", "synthesizing per-column cleaning functions") as s:
        _build_cleaning_requests(dataset_path, validation_results)
        artifacts = run_cleaner_generation(dataset_path, reuse_consistency=True, max_attempts=10)
        s.write(f"{len(artifacts)} cleaner function{'s' if len(artifacts) != 1 else ''} generated.")
    _complete_stage("Generate", strip_ph, prog_ph)

    # Stage 10: application
    _mark_stage("Apply", strip_ph, prog_ph)
    with stage_banner("Apply", "executing cleaners, renames, casts") as s:
        cleaning_report, _exec_reports, remediation_plan = run_cleaner_application_with_plan(
            dataset_path, remediation_plan
        )
        n_applied = len([a for a in remediation_plan.actions if a.status == "applied"]) if remediation_plan else 0
        s.write(f"{n_applied} action{'s' if n_applied != 1 else ''} applied to dataset.")
    _complete_stage("Apply", strip_ph, prog_ph)

    # Stage 11: verification
    _mark_stage("Verify", strip_ph, prog_ph)
    with stage_banner("Verify", "re-checking consistency after cleaning") as s:
        verification_report = run_verify(dataset_path)
        s.write(verification_report.summary if verification_report else "Verification complete.")
    _complete_stage("Verify", strip_ph, prog_ph)

    # Final report
    final_report = build_final_report(
        validation_results, remediation_plan, cleaning_report, verification_report
    )
    save_final_report(dataset_path, final_report)
    st.session_state.final_report = final_report

    # Stage 12: narrative
    _mark_stage("Narrative", strip_ph, prog_ph)
    with stage_banner("Narrative", "agent writing the final quality report") as s:
        narrative = generate_narrative_report(final_report)
        save_narrative_report(dataset_path, narrative)
        st.session_state.narrative_report = narrative
        s.write(f"Report written: {len(narrative.sections)} sections, {len(narrative.recommendations)} recommendations.")
    _complete_stage("Narrative", strip_ph, prog_ph)

    st.session_state.pipeline_done = True


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def view_upload() -> None:
    _section("I", "The Dataset", "select or upload a source csv")

    st.markdown(
        '<div class="lede">This atelier transforms raw CSV files from the Public Administration into '
        'refined quality reports. Select an existing dataset or upload a new one.</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown('<div class="panel-label">Upload</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload a CSV",
            type=["csv"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            saved_path = _save_uploaded_file(uploaded)
            st.session_state.dataset_path = saved_path
            st.session_state.dataset_name = saved_path.stem
            st.session_state.pipeline_done = False
            st.session_state.completed_stages = set()

    with col_b:
        st.markdown('<div class="panel-label">Or select an existing dataset</div>', unsafe_allow_html=True)
        datasets = _list_available_datasets()
        if datasets:
            options = {f"{p.name}  ·  {p.stat().st_size // 1024} KB": p for p in datasets}
            current = st.session_state.get("dataset_path")
            current_label = next(
                (lbl for lbl, p in options.items() if p == current),
                None,
            )
            current_index = list(options.keys()).index(current_label) if current_label else None
            choice = st.selectbox(
                "existing",
                options=list(options.keys()),
                index=current_index,
                placeholder="Select a dataset…",
                label_visibility="collapsed",
                key="existing_dataset_select",
            )
            if choice and options[choice] != current:
                st.session_state.dataset_path = options[choice]
                st.session_state.dataset_name = options[choice].stem
                st.session_state.pipeline_done = False
                st.session_state.completed_stages = set()
                st.rerun()
        else:
            st.caption("No datasets found in the Data/ folder.")

    if st.session_state.dataset_path is not None:
        _dataset_preview_panel(st.session_state.dataset_path)


def _dataset_preview_panel(path: Path) -> None:
    try:
        df = pd.read_csv(path, nrows=200)
    except Exception as exc:
        st.error(f"Cannot read {path.name}: {exc}")
        return
    full_rows = sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1

    st.markdown(
        f"""
        <div class="action-bar">
            <div>
                <div class="label">Selected</div>
                <div class="value">{path.name}</div>
            </div>
            <div>
                <div class="label">Rows · Columns</div>
                <div class="value">{full_rows:,} × {len(df.columns)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _metric_grid([
        _metric("Rows", f"{full_rows:,}"),
        _metric("Columns", str(len(df.columns))),
        _metric("File size", f"{path.stat().st_size / 1024:.1f}", "KB"),
        _metric("Preview", "200", "rows"),
    ])

    st.markdown('<div class="panel-label">Preview (first 200 rows)</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=340)


def view_pipeline() -> None:
    _section("II", "The Pipeline", "twelve agents working in concert")

    strip_ph = st.empty()
    strip_ph.markdown(
        _stage_strip(
            st.session_state.current_stage,
            st.session_state.completed_stages,
        ),
        unsafe_allow_html=True,
    )

    n_done = len(st.session_state.completed_stages)
    pct = int(n_done / len(PIPELINE_STAGES) * 100)
    current = st.session_state.current_stage
    prog_label = (
        f"Stage {n_done} of {len(PIPELINE_STAGES)} · {current} · {pct}%"
        if current
        else (f"Complete · {pct}%" if n_done else "Waiting to start")
    )
    prog_ph = st.empty()
    prog_ph.progress(pct / 100, text=prog_label)

    col_run, col_reset = st.columns([3, 1])
    with col_run:
        disabled = st.session_state.dataset_path is None or st.session_state.pipeline_done
        run_label = "Run pipeline" if not st.session_state.pipeline_done else "Pipeline complete ✓"
        run_clicked = st.button(run_label, disabled=disabled, use_container_width=True)
    with col_reset:
        if st.button("Reset", use_container_width=True):
            st.session_state.pipeline_done = False
            st.session_state.completed_stages = set()
            st.session_state.final_report = None
            st.session_state.narrative_report = None
            st.rerun()

    log_container = st.container()

    if run_clicked and st.session_state.dataset_path is not None:
        if not st.session_state.logfire_setup:
            try:
                setup_logfire()
            except Exception:
                pass
            st.session_state.logfire_setup = True
        try:
            run_full_pipeline(
                st.session_state.dataset_path,
                strip_ph,
                prog_ph,
                log_container,
            )
            st.success("Pipeline completed successfully.")
            st.rerun()
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            raise


def view_report() -> None:
    final_report: FinalPipelineReport | None = st.session_state.final_report
    narrative = st.session_state.narrative_report
    if final_report is None:
        st.info("Run the pipeline to generate the report.")
        return

    _section("III", "The Verdict", "quality metrics at a glance")

    vs = final_report.validation_summary
    _metric_grid([
        _metric("Schema Issues", str(vs.get("schema_issues", 0)), variant="accent"),
        _metric("Missing Cols", str(vs.get("completeness_columns_with_missing", 0))),
        _metric("Format Issues", str(vs.get("consistency_findings", 0))),
        _metric("Anomalies", str(vs.get("anomaly_findings", 0))),
        _metric("Cross-Column", str(vs.get("cross_column_findings", 0))),
        _metric("Dup Groups", str(vs.get("duplicate_groups", 0))),
    ])

    n_manual = len(final_report.manual_review_queue)
    _metric_grid([
        _metric("Applied", str(len(final_report.applied_actions)), variant="success"),
        _metric("Manual Review", str(n_manual), variant="warning"),
        _metric("Failed", str(len(final_report.failed_actions))),
        _metric("Not needed", str(len(final_report.not_needed_actions))),
    ])

    tab_narrative, tab_actions, tab_data = st.tabs([
        "Narrative Report",
        "Action Ledger",
        "Cleaned Dataset",
    ])

    with tab_narrative:
        if narrative is not None:
            _render_narrative(narrative)
        else:
            st.caption("Narrative not available.")

    with tab_actions:
        _render_action_ledger(final_report)

    with tab_data:
        _render_cleaned_dataset(st.session_state.dataset_path)


def _render_narrative(narrative) -> None:
    st.markdown(
        f'<div class="kicker">Final Report</div>'
        f'<h1 style="font-family:Fraunces,serif; font-size:2.2rem; font-weight:400; font-style:italic; margin-top:0.3rem; margin-bottom:1.2rem;">{narrative.title}</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="lede">{narrative.executive_summary}</div>',
        unsafe_allow_html=True,
    )

    body_md = []
    for section in narrative.sections:
        body_md.append(f"## {section.heading}\n\n{section.body}")
    if narrative.recommendations:
        body_md.append("## Recommendations\n")
        for i, rec in enumerate(narrative.recommendations, 1):
            body_md.append(f"{i}. {rec}")

    st.markdown(
        f'<div class="narrative">\n\n{chr(10).join(body_md)}\n\n</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ornament">· · ·</div>', unsafe_allow_html=True)

    # Download buttons
    report_text = f"# {narrative.title}\n\n{narrative.executive_summary}\n\n" + "\n\n".join(
        f"## {s.heading}\n\n{s.body}" for s in narrative.sections
    )
    if narrative.recommendations:
        report_text += "\n\n## Recommendations\n\n" + "\n".join(
            f"{i}. {r}" for i, r in enumerate(narrative.recommendations, 1)
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "⬇ Download report (markdown)",
            data=report_text,
            file_name=f"{st.session_state.dataset_name}_quality_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_b:
        import json as _json
        st.download_button(
            "⬇ Download structured data (JSON)",
            data=_json.dumps(st.session_state.final_report.model_dump(), indent=2, ensure_ascii=False),
            file_name=f"{st.session_state.dataset_name}_final_report.json",
            mime="application/json",
            use_container_width=True,
        )


def _render_action_ledger(final_report: FinalPipelineReport) -> None:
    buckets = [
        ("Applied automatically", "success", final_report.applied_actions),
        ("Failed", "accent", final_report.failed_actions),
        ("Not needed", "", final_report.not_needed_actions),
    ]
    for label, _variant, actions in buckets:
        st.markdown(
            f'<div class="kicker" style="color:var(--ink-muted);">{label} · {len(actions)}</div>',
            unsafe_allow_html=True,
        )
        if not actions:
            st.caption("— no actions in this category —")
            continue
        rows = []
        for a in actions:
            target_str = ", ".join(f"{k}={v}" for k, v in (a.target or {}).items()) if a.target else "—"
            if len(target_str) > 80:
                target_str = target_str[:77] + "…"
            rows.append({
                "Type": a.action_type,
                "Source": a.source_check,
                "Confidence": a.confidence,
                "Risk": a.risk_level,
                "Target": target_str,
                "Reason": a.reason[:120] + "…" if len(a.reason) > 120 else a.reason,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="kicker" style="color:var(--ink-muted);">Manual Review Queue</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"{len(final_report.manual_review_queue)} items deferred to manual review "
        "(near-duplicate rows, semantic conflicts, outliers). See the Narrative Report for details."
    )


def _render_cleaned_dataset(dataset_path: Path | None) -> None:
    if dataset_path is None:
        st.caption("—")
        return
    cleaned_path = cleaned_dataset_path(dataset_path)
    if not cleaned_path.exists():
        st.caption("Cleaned dataset not yet available.")
        return
    try:
        df_clean = pd.read_csv(cleaned_path, nrows=500)
    except Exception as exc:
        st.error(str(exc))
        return

    total_rows = sum(1 for _ in open(cleaned_path, encoding="utf-8", errors="ignore")) - 1

    _metric_grid([
        _metric("Cleaned rows", f"{total_rows:,}", variant="success"),
        _metric("Cleaned columns", str(len(df_clean.columns))),
        _metric("Preview", "500", "rows"),
    ])

    st.dataframe(df_clean, use_container_width=True, height=420)

    with open(cleaned_path, "rb") as fh:
        st.download_button(
            "⬇ Download cleaned CSV",
            data=fh.read(),
            file_name=cleaned_path.name,
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()
    _masthead()
    view_upload()
    view_pipeline()
    view_report()

    st.markdown(
        '<div class="ornament">◆</div>'
        '<div style="text-align:center; font-family:IBM Plex Mono,monospace; '
        'font-size:0.68rem; text-transform:uppercase; letter-spacing:0.25em; '
        'color:var(--ink-muted);">Ministry of Economy and Finance · NoiPA · Data Quality Atelier</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
