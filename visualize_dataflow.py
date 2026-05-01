"""
Data Flow Visualization for the Agents AI Pipeline

Generates visual diagrams of the validation and cleaning pipeline.
Run with:  python visualize_dataflow.py
Requires:  pip install graphviz  (plus the Graphviz system binaries)
"""

import graphviz
from pathlib import Path

OUTPUT_DIR = "images/flow_diagrams"

# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

FONT      = "Helvetica-Bold"
FONT_BODY = "Helvetica-Bold"

NODE_DEFAULTS  = dict(shape="box", style="rounded,filled", fontname=FONT_BODY, fontsize="11",
                      margin="0.18,0.10", fontcolor="#2D2D2D", penwidth="1.6")
EDGE_DEFAULTS  = dict(fontname=FONT_BODY, fontsize="9", color="#555555", fontcolor="#555555",
                      penwidth="1.4")
GRAPH_DEFAULTS = dict(bgcolor="white", fontname=FONT, pad="0.4", nodesep="0.55", ranksep="0.75",
                      concentrate="true", splines="line")

# Reply brand palette
COLORS = {
    "source":    "#9FE870",   # Reply lime    – input data files
    "agent":     "#5DC73A",   # Reply light green – LLM-backed nodes
    "artifact":  "#F4F4F4",   # Reply light grey  – typed data objects
    "action":    "#E0F5D2",   # very light green  – deterministic code / host logic
    "output":    "#9FE870",   # Reply lime    – final output files
    "cluster_v": "#F2FBF0",   # near-white green  – validation cluster fill
    "cluster_c": "#F8FDF6",   # near-white lighter – cleaning cluster fill
}


def _base_graph(name: str, comment: str, rankdir: str = "TB", **kwargs) -> graphviz.Digraph:
    dot = graphviz.Digraph(name=name, comment=comment, format="png")
    attrs = {**GRAPH_DEFAULTS, "rankdir": rankdir, **kwargs}
    dot.attr("graph", **attrs)
    dot.attr("node",  **NODE_DEFAULTS)
    dot.attr("edge",  **EDGE_DEFAULTS)
    return dot


def _render(dot: graphviz.Digraph, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dot.directory = output_dir
    dot.render(cleanup=True)
    print(f"  {dot.name}.png")


# ---------------------------------------------------------------------------
# 1. Main pipeline overview  (left → right, two clusters)
# ---------------------------------------------------------------------------

def create_dataflow_diagram(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("01_pipeline_overview", "Main pipeline overview", rankdir="TB",
                      ranksep="1.1", nodesep="0.7", concentrate="false")

    # --- Input ---
    dot.node("csv",   "CSV input\n(Data/)",                 fillcolor=COLORS["source"], shape="folder")
    dot.node("entry", "Entrypoint\n(CLI / App / Notebook)", fillcolor=COLORS["action"])
    dot.edge("csv", "entry")

    # --- Validation cluster ---
    with dot.subgraph(name="cluster_validation") as v:
        v.attr(label="Validation half", style="rounded,filled", fillcolor=COLORS["cluster_v"],
               color="#22A30A", penwidth="1.5", fontname=FONT, fontsize="12", fontcolor="#22A30A")
        with v.subgraph() as same:
            same.attr(rank="same")
            same.node("schema",   "1. Schema\nvalidation",    fillcolor=COLORS["agent"])
            same.node("complete", "2. Completeness\nanalysis", fillcolor=COLORS["agent"])
            same.node("consist",  "3. Format\nconsistency",    fillcolor=COLORS["agent"])
            same.node("anomaly",  "4. Anomaly\ndetection",     fillcolor=COLORS["agent"])
            same.node("cross",    "5. Cross-column\nchecks",   fillcolor=COLORS["agent"])
            same.node("dupes",    "6. Duplicate\ndetection",   fillcolor=COLORS["agent"])
        # Invisible edges to preserve left-to-right order
        for a, b in [("schema","complete"),("complete","consist"),("consist","anomaly"),
                     ("anomaly","cross"),("cross","dupes")]:
            v.edge(a, b, style="invis")

    # --- Intermediate bundle node ---
    dot.node("bundle", "Validation bundle\n(OrchestrationStepResult)",
             fillcolor=COLORS["artifact"], style="rounded,filled,bold")

    # --- Cleaning cluster: row 1 = remediate/generate/apply, row 2 = verify/report ---
    with dot.subgraph(name="cluster_cleaning") as c:
        c.attr(label="Cleaning half", style="rounded,filled", fillcolor=COLORS["cluster_c"],
               color="#1A8A00", penwidth="1.5", fontname=FONT, fontsize="12", fontcolor="#1A8A00")
        # Row 1
        with c.subgraph() as row1:
            row1.attr(rank="same")
            row1.node("remediate", "1. Remediation\nplanning",  fillcolor=COLORS["agent"])
            row1.node("generate",  "2. Cleaner\ngeneration",    fillcolor=COLORS["agent"])
            row1.node("apply",     "3. Application\n(execute)", fillcolor=COLORS["action"])
        # Row 2
        with c.subgraph() as row2:
            row2.attr(rank="same")
            row2.node("verify", "4. Verification",      fillcolor=COLORS["agent"])
            row2.node("report", "5. Report\ngeneration", fillcolor=COLORS["agent"])
        # Sequence edges
        c.edge("remediate", "generate")
        c.edge("generate",  "apply")
        c.edge("apply",     "verify")
        c.edge("verify",    "report")
        # Invisible edge to keep row2 left-aligned under row1
        c.edge("remediate", "verify", style="invis")

    # --- Outputs ---
    dot.node("clean_csv", "Cleaned CSV",                  fillcolor=COLORS["output"], shape="folder")
    dot.node("report_md", "Narrative report\n(Markdown)", fillcolor=COLORS["output"], shape="folder")

    # --- Connections ---
    # Entry connects to every validation stage
    for stage in ["schema", "complete", "consist", "anomaly", "cross", "dupes"]:
        dot.edge("entry", stage)
    # All stages feed the bundle
    for stage in ["schema", "complete", "consist", "anomaly", "cross", "dupes"]:
        dot.edge(stage, "bundle")
    dot.edge("bundle", "remediate")
    dot.edge("report", "clean_csv")
    dot.edge("report", "report_md")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 2. Validation stage detail
# ---------------------------------------------------------------------------

def create_validation_flow_detail(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("06_validation_stage_pipeline", "Validation pipeline detail",
                      rankdir="TB", ranksep="1.0", nodesep="0.6", concentrate="false")

    dot.node("df", "Raw DataFrame", fillcolor=COLORS["source"])

    # Schema stage cluster — sch lives inside the cluster
    with dot.subgraph(name="cluster_schema") as s:
        s.attr(label="Schema stage", style="rounded,filled", fillcolor=COLORS["cluster_v"],
               color="#22A30A", penwidth="1.5", fontname=FONT, fontsize="10", fontcolor="#22A30A")
        s.node("prof",  "Deterministic\nprofiling",     fillcolor=COLORS["action"])
        s.node("dtype", "dtype-inference\nagent",       fillcolor=COLORS["agent"])
        s.node("name",  "Naming &\nduplication checks", fillcolor=COLORS["action"])
        s.node("sch",   "Schema handoff\n(SchemaHandoff)", fillcolor=COLORS["artifact"],
               style="rounded,filled,bold")
        s.edge("prof",  "dtype")
        s.edge("prof",  "name")
        s.edge("dtype", "sch")
        s.edge("name",  "sch")

    dot.edge("df", "prof")

    # Five report nodes on the same horizontal rank below the cluster
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("com", "Completeness report\n(CompletenessAnalysisReport)", fillcolor=COLORS["artifact"])
        s.node("con", "Consistency report\n(ConsistencyValidationReport)",  fillcolor=COLORS["artifact"])
        s.node("ano", "Anomaly report\n(AnomalyDetectionReport)",           fillcolor=COLORS["artifact"])
        s.node("cro", "Cross-column report\n(CrossColumnValidationReport)", fillcolor=COLORS["artifact"])
        s.node("dup", "Duplicate report\n(DuplicateDetectionReport)",       fillcolor=COLORS["artifact"])
    for a, b in [("com","con"),("con","ano"),("ano","cro"),("cro","dup")]:
        dot.edge(a, b, style="invis")

    # sch fans out to all five report nodes (dashed)
    for stage in ["com", "con", "ano", "cro", "dup"]:
        dot.edge("sch", stage, style="dashed", color="#888888")

    # All five feed the validation bundle
    dot.node("bun", "Validation bundle\n(OrchestrationStepResult)", fillcolor=COLORS["artifact"],
             style="rounded,filled,bold")
    for stage in ["com", "con", "ano", "cro", "dup"]:
        dot.edge(stage, "bun")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 3. Cleaning stage detail
# ---------------------------------------------------------------------------

def create_cleaning_flow_detail(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("09_cleaning_half_pipeline", "Application & verification detail",
                      rankdir="TB", ranksep="0.9", nodesep="0.7")

    # Inputs to application
    dot.node("cleaners", "Accepted cleaners\n(ColumnCleanerProgram[])", fillcolor=COLORS["artifact"])
    dot.node("auto",     "auto_apply actions\n(cast · rename · placeholder→null\n· exact duplicate drop)",
             fillcolor=COLORS["artifact"])

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("cleaners")
        s.node("auto")

    # Application stage
    dot.node("app", "Application\n(application.py)\n— ordered execution —",
             fillcolor=COLORS["action"], style="rounded,filled,bold")

    dot.edge("cleaners", "app")
    dot.edge("auto",     "app")

    with dot.subgraph(name="cluster_order") as o:
        o.attr(label="Application order", style="rounded,filled", fillcolor=COLORS["cluster_c"],
               color="#1A8A00", penwidth="1.5", fontname=FONT, fontsize="10", fontcolor="#1A8A00")
        o.node("s1", "1. Generated cleaners\n(column identities still intact)", fillcolor=COLORS["action"])
        o.node("s2", "2. Placeholder → null",                                    fillcolor=COLORS["action"])
        o.node("s3", "3. Drop exact duplicate columns",                          fillcolor=COLORS["action"])
        o.node("s4", "4. Column renames",                                        fillcolor=COLORS["action"])
        o.node("s5", "5. dtype casts",                                           fillcolor=COLORS["action"])
        with o.subgraph() as r1:
            r1.attr(rank="same")
            for n in ["s1", "s2", "s3"]:
                r1.node(n)
        with o.subgraph() as r2:
            r2.attr(rank="same")
            for n in ["s4", "s5"]:
                r2.node(n)
        o.edge("s1", "s2")
        o.edge("s2", "s3")
        o.edge("s3", "s4")
        o.edge("s4", "s5")

    dot.edge("app", "s1")

    dot.node("cleaned_csv", "Cleaned CSV", fillcolor=COLORS["output"], shape="folder")
    dot.edge("s5", "cleaned_csv")

    _render(dot, output_dir)



# ---------------------------------------------------------------------------
# 5. Generation–validation–critic cycle
# ---------------------------------------------------------------------------

def create_generation_validation_cycle(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("08_cleaner_generation_loop", "Code generation cycle", rankdir="TB", ranksep="1.1", nodesep="0.9", splines="polyline")

    dot.node("req",  "ColumnCleaningRequest\n(target dtype, valid examples,\ninconsistent examples)", fillcolor=COLORS["artifact"])
    dot.node("gen",  "Generator agent\n(writes Python cleaner fn)", fillcolor=COLORS["agent"])
    dot.node("self", "Self-test\n(CodeExecutionTool,\nbounded)", fillcolor=COLORS["action"])
    dot.node("hval", "Host-side validation\n(syntax · signature · security\n· preservation rules)", fillcolor=COLORS["action"])
    dot.node("ok",      "Cleaner accepted\n(ColumnCleanerProgram)", fillcolor=COLORS["output"])
    dot.node("rebuild", "rebuild_verified_program\n(re-parse · strip artefacts\n· final signature check)", fillcolor=COLORS["action"])
    dot.node("crit",    "Critic agent\n(diagnosis of\nfailed checks)", fillcolor=COLORS["agent"])
    dot.node("stag",    "Stagnation detector\n(repeated code / fingerprint\n→ raise temperature)", fillcolor=COLORS["action"])

    dot.edges([
        ("req",     "gen"),
        ("gen",     "self"),
        ("self",    "hval"),
        ("hval",    "ok"),
        ("ok",      "rebuild"),
    ])

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("ok")
        s.node("stag")

    # Failure loop
    dot.edge("hval", "crit", label="checks failed", style="dashed", color="#888888", fontcolor="#888888")
    dot.edge("crit", "stag", style="dashed", color="#888888")
    dot.edge("stag", "gen",  label="retry with repair prompt\n(max attempts)",
             style="dashed", color="#888888", fontcolor="#888888", constraint="false")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 6. Schema validation internals
# ---------------------------------------------------------------------------

def create_schema_validation_detail(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("03_schema_stage_internals", "Schema validation internals", rankdir="TB", ranksep="0.85")

    dot.node("df",      "Raw DataFrame",                         fillcolor=COLORS["source"])
    dot.node("prof",    "Deterministic profiler\n(schema_tools.py)\n\nnon-null count · distinct count\nnumeric parse % · datetime parse %\nrandom sample (≤5%, ≤500 values)",
             fillcolor=COLORS["action"])

    # Parallel branches
    dot.node("agent",   "dtype-inference agent\n\nreceives: column name,\nparse stats, bounded sample\n→ infers target dtype,\nsemantic role, detected_pattern",
             fillcolor=COLORS["agent"])
    dot.node("naming",  "Naming checks\n(deterministic)\n\nsnake_case · leading digit\nunsafe characters",
             fillcolor=COLORS["action"])
    dot.node("semdup",  "Duplicate-semantic\ncheck (deterministic)\n\ncolumns that normalise\nto the same name",
             fillcolor=COLORS["action"])

    dot.node("merge",   "Merge results",                         fillcolor=COLORS["action"])
    dot.node("sh",      "SchemaHandoff\n\npandas_dtype · detected_pattern\nsemantic role · rename suggestion\nnaming_valid · duplicate groups",
             fillcolor=COLORS["artifact"], style="rounded,filled,bold")

    dot.edge("df",    "prof")
    dot.edge("prof",  "agent",  label="column profile\n+ sample")
    dot.edge("prof",  "naming", label="column names")
    dot.edge("prof",  "semdup", label="column names")

    # force agent / naming / semdup on same rank
    with dot.subgraph() as s:
        s.attr(rank="same")
        for n in ["agent", "naming", "semdup"]:
            s.node(n)

    dot.edge("agent",  "merge")
    dot.edge("naming", "merge")
    dot.edge("semdup", "merge")
    dot.edge("merge",  "sh")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 7. Format consistency validation — fast path vs slow path
# ---------------------------------------------------------------------------

def create_consistency_detail(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("05_format_consistency_paths", "Format consistency fast vs slow path", rankdir="TB", ranksep="0.9")

    dot.node("sh",      "SchemaHandoff\n(detected_pattern, pandas_dtype)",  fillcolor=COLORS["artifact"])
    dot.node("df",      "Raw DataFrame column",                              fillcolor=COLORS["source"])
    dot.node("shape",   "Shape profiler\n(format_tools.py)\n\nrenders each value as an\nabstract shape (9→digit, A→letter)\ncounts shape frequencies\n→ dominant_shape, dominant_shape_pct",
             fillcolor=COLORS["action"])
    dot.node("gate",    "Entry gate\n\nmachine_format_candidate?\nnumeric_parse_pct ≥ 85?\nschema gate bypass?",
             fillcolor=COLORS["action"], shape="diamond")

    dot.node("skip",    "Column skipped\n(no finding emitted)",              fillcolor=COLORS["artifact"])

    dot.node("fast",    "Fast path\n(schema-guided)\n\nvalidate values directly\nagainst detected_pattern\nno LLM call",
             fillcolor=COLORS["action"])
    dot.node("slow",    "Slow path\n(agent-backed)\n\nColumnFormatFacts serialised:\ndominant_shape · outlier families\nparse stats · schema hints\n→ agent decides if actionable",
             fillcolor=COLORS["agent"])

    dot.node("finding", "FormatConsistencyFinding\n\nexpected_pattern · inconsistent_rows\nexample_inconsistent_values\nsuggested_strategy",
             fillcolor=COLORS["artifact"], style="rounded,filled,bold")
    dot.node("none",    "No finding\n(column is consistent\nor free-text)",  fillcolor=COLORS["artifact"])

    dot.edge("sh",    "shape")
    dot.edge("df",    "shape")
    dot.edge("shape", "gate")
    dot.edge("gate",  "skip",    label="not a candidate", style="dashed", color="#888888", fontcolor="#888888")
    dot.edge("gate",  "fast",    label="unambiguous\nschema pattern")
    dot.edge("gate",  "slow",    label="no stable\npattern")
    dot.edge("fast",  "finding", label="inconsistencies\nfound")
    dot.edge("fast",  "none",    label="all values\nvalid", style="dashed", color="#888888", fontcolor="#888888")
    dot.edge("slow",  "finding", label="agent flags\nas actionable")
    dot.edge("slow",  "none",    label="agent flags\nas non-actionable", style="dashed", color="#888888", fontcolor="#888888")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 9. Verification — before/after comparison
# ---------------------------------------------------------------------------

def create_verification_detail(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("10_post_cleaning_verification", "Post-cleaning verification", rankdir="TB", ranksep="0.9")

    dot.node("orig_csv",  "Original CSV",          fillcolor=COLORS["source"], shape="folder")
    dot.node("clean_csv", "Cleaned CSV",            fillcolor=COLORS["output"], shape="folder")
    dot.node("orig_find", "Original consistency\nfindings\n(from validation bundle)", fillcolor=COLORS["artifact"])

    dot.node("reread",    "Re-read cleaned CSV\nas raw strings\n(no dtype coercion)",  fillcolor=COLORS["action"])
    dot.node("reshape",   "Re-run shape profiler\non cleaned columns",                 fillcolor=COLORS["action"])

    dot.node("diff",      "Diff engine\n(verification.py)\n\ncompares new findings\nagainst original findings\nper targeted column",
             fillcolor=COLORS["action"])

    dot.node("agent",     "Verification agent\n\nsummarises diff results\ninto structured assessment",
             fillcolor=COLORS["agent"])

    # Outcome nodes (same rank)
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("resolved",   "resolved\n(issue gone)",              fillcolor=COLORS["agent"])
        s.node("improved",   "improved\n(inconsistencies reduced)", fillcolor=COLORS["action"])
        s.node("unchanged",  "unchanged\n(no effect)",              fillcolor=COLORS["artifact"])
        s.node("regressed",  "regressed\n(new issues introduced)",  fillcolor=COLORS["artifact"])

    dot.node("verif_out", "VerificationReport\n\nper-column outcome + evidence\noverall assessment",
             fillcolor=COLORS["artifact"], style="rounded,filled,bold")

    dot.edge("clean_csv", "reread")
    dot.edge("reread",    "reshape")
    dot.edge("reshape",   "diff")
    dot.edge("orig_find", "diff",  label="baseline")
    dot.edge("orig_csv",  "diff",  label="original\nrow count / dtypes", style="dashed", color="#888888", fontcolor="#888888")
    dot.edge("diff",      "agent")

    for outcome in ["resolved", "improved", "unchanged", "regressed"]:
        dot.edge("agent", outcome)
        dot.edge(outcome, "verif_out")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 10. Completeness detection flow  (null + empty + placeholder → missing mask)
# ---------------------------------------------------------------------------

def create_completeness_flow(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("04_completeness_detection_flow", "Completeness detection: building the missing-like mask",
                      rankdir="TB", ranksep="0.85")

    dot.node("df",    "Raw DataFrame column",              fillcolor=COLORS["source"])

    # Three parallel detection paths
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("null",  "True null check\n(pd.isna / pd.isnull)",                       fillcolor=COLORS["action"])
        s.node("empty", "Empty-string check\n(value.strip() == '')",                    fillcolor=COLORS["action"])
        s.node("phld",  "Placeholder normalisation\n(lowercase · strip → match against\nconfigured token list:\nN/A  –  //  ?  n.d.  unknown  …)", fillcolor=COLORS["action"])

    dot.node("mask",  "Unified missing-like mask\n(true null  OR  empty  OR  placeholder)",
             fillcolor=COLORS["artifact"], style="rounded,filled,bold")

    dot.node("prof",  "Completeness profiler\n(completeness_tools.py)\n\ncompleteness_pct · missing_like_count\nmissing_like_examples · sparse_candidate",
             fillcolor=COLORS["action"])

    dot.node("agent", "completeness-analysis agent\n\nreceives bounded column profile\n→ per-column recommendation\n(no raw column values sent)",
             fillcolor=COLORS["agent"])

    dot.node("out",   "CompletenessAnalysisReport\n\nper_column: completeness_pct\nmissing_like_count · missing_like_examples\nsparse_candidate · recommended_action",
             fillcolor=COLORS["artifact"], style="rounded,filled,bold")

    dot.edge("df",   "null")
    dot.edge("df",   "empty")
    dot.edge("df",   "phld")
    dot.edge("null",  "mask")
    dot.edge("empty", "mask")
    dot.edge("phld",  "mask")
    dot.edge("mask",  "prof")
    dot.edge("prof",  "agent")
    dot.edge("agent", "out")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 11. Remediation policy decision tree  (finding → action category)
# ---------------------------------------------------------------------------

def create_remediation_policy_tree(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("07_remediation_policy_tree", "Remediation policy: finding to action category",
                      rankdir="TB", ranksep="1.2", nodesep="0.9", concentrate="false")

    dot.node("finding", "Validation finding\n(any stage)", fillcolor=COLORS["artifact"])

    # Gate 1 — source stage
    dot.node("g1", "Source stage?", fillcolor=COLORS["action"], shape="diamond")
    dot.edge("finding", "g1")

    # Force the three second-level diamonds on the same rank with spacing
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("g2", "Action type?",              fillcolor=COLORS["action"], shape="diamond")
        s.node("g3", "Canonical target\nunambiguous?", fillcolor=COLORS["action"], shape="diamond")
        s.node("g4", "Risk level?",               fillcolor=COLORS["action"], shape="diamond")
    for a, b in [("g2","g3"),("g3","g4")]:
        dot.edge(a, b, style="invis")

    dot.edge("g1", "g2", xlabel="schema /\ncompleteness")
    dot.edge("g1", "g3", xlabel="consistency")
    dot.edge("g1", "g4", xlabel="anomaly /\ncross-column /\nduplicate")

    # Force all six leaf nodes on the same rank
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("auto_struct", "auto_apply = True\n\ndtype cast · safe rename\nplaceholder → null\nexact duplicate\ncolumn / row drop",
               fillcolor=COLORS["output"])
        s.node("manual_schema", "manual_review\n\nunsafe rename\nnear-duplicate columns\nsemantic conflict",
               fillcolor=COLORS["artifact"])
        s.node("format_fix", "format_fix action\n→ ColumnCleaningRequest\n→ cleaner generation loop",
               fillcolor=COLORS["agent"])
        s.node("report_only_c", "report_only\n(no safe target\ncan be defined)",
               fillcolor=COLORS["artifact"])
        s.node("auto_dup", "auto_apply = True\n\nexact duplicate rows\nexact duplicate columns",
               fillcolor=COLORS["output"])
        s.node("manual_amb", "manual_review\n\nnear-duplicate rows/columns\ndate-order violations\ntemporal mismatches\nrare categories · numeric outliers",
               fillcolor=COLORS["artifact"])
    for a, b in [("auto_struct","manual_schema"),("manual_schema","format_fix"),
                 ("format_fix","report_only_c"),("report_only_c","auto_dup"),("auto_dup","manual_amb")]:
        dot.edge(a, b, style="invis")

    dot.edge("g2", "auto_struct",   xlabel="structural,\nmechanical", minlen="2")
    dot.edge("g2", "manual_schema", xlabel="ambiguous /\nhigh impact", minlen="2")
    dot.edge("g3", "format_fix",    xlabel="yes", minlen="2")
    dot.edge("g3", "report_only_c", xlabel="no",  minlen="2")
    dot.edge("g4", "auto_dup",      xlabel="low risk\n(exact duplicate)", minlen="2")
    dot.edge("g4", "manual_amb",    xlabel="medium / high\nor ambiguous",  minlen="2")

    dot.node("plan", "RemediationPlan\n(RemediationAction[])",
             fillcolor=COLORS["artifact"], style="rounded,filled,bold")
    for src in ["auto_struct", "manual_schema", "format_fix", "report_only_c", "auto_dup", "manual_amb"]:
        dot.edge(src, "plan")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 12. Four-layer conceptual architecture
# ---------------------------------------------------------------------------

def create_four_layer_architecture(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("02_conceptual_architecture", "Four-layer conceptual architecture",
                      rankdir="TB", ranksep="1.0", nodesep="0.65", concentrate="false")

    # ── Layer 1 — Contract ──────────────────────────────────────────────────
    with dot.subgraph(name="cluster_L1") as c:
        c.attr(label="Layer 1 — Contract Layer",
               style="rounded,filled", fillcolor=COLORS["cluster_v"],
               color="#22A30A", penwidth="1.5", fontname=FONT, fontsize="12", fontcolor="#22A30A")
        with c.subgraph() as s:
            s.attr(rank="same")
            s.node("PM",  "Pydantic Models\n(models.py)",     fillcolor=COLORS["artifact"])
            s.node("ART", "Typed Artifacts\n(stage outputs)", fillcolor=COLORS["artifact"],
                   style="rounded,filled,bold")
        c.edge("PM", "ART", style="invis")

    # ── Layer 2 — Deterministic Evidence ───────────────────────────────────
    with dot.subgraph(name="cluster_L2") as c:
        c.attr(label="Layer 2 — Deterministic Evidence-Building Layer",
               style="rounded,filled", fillcolor=COLORS["cluster_c"],
               color="#1A8A00", penwidth="1.5", fontname=FONT, fontsize="12", fontcolor="#1A8A00")
        with c.subgraph() as s:
            s.attr(rank="same")
            s.node("PR", "Profiling\n(parse rates, shapes)",           fillcolor=COLORS["action"])
            s.node("CK", "Completeness Checks\n(nulls, placeholders)", fillcolor=COLORS["action"])
            s.node("DU", "Duplicate Detection",                        fillcolor=COLORS["action"])
            s.node("AN", "Anomaly Measurement\n(statistical)",         fillcolor=COLORS["action"])
        for a, b in [("PR","CK"),("CK","DU"),("DU","AN")]:
            c.edge(a, b, style="invis")
        c.node("EV", "Evidence Bundle", fillcolor=COLORS["agent"], style="rounded,filled,bold")
        for src in ["PR","CK","DU","AN"]:
            c.edge(src, "EV")

    # ── Layer 3 — Agent Layer ───────────────────────────────────────────────
    with dot.subgraph(name="cluster_L3") as c:
        c.attr(label="Layer 3 — Agent Layer  (LLM-backed)",
               style="rounded,filled", fillcolor=COLORS["cluster_v"],
               color="#22A30A", penwidth="1.5", fontname=FONT, fontsize="12", fontcolor="#22A30A")
        with c.subgraph() as s:
            s.attr(rank="same")
            s.node("VA", "Validation Agent\n(schema & format)",    fillcolor=COLORS["agent"])
            s.node("RA", "Remediation Agent\n(action planning)",   fillcolor=COLORS["agent"])
            s.node("CA", "Cleaner Agent\n(transformation gen.)",   fillcolor=COLORS["agent"])
            s.node("NA", "Narrative Agent\n(report prose)",        fillcolor=COLORS["agent"])
        c.edge("VA", "RA")
        c.edge("RA", "CA")
        c.edge("CA", "NA", style="invis")

    # ── Layer 4 — Host-Side Enforcement ────────────────────────────────────
    with dot.subgraph(name="cluster_L4") as c:
        c.attr(label="Layer 4 — Host-Side Enforcement Layer",
               style="rounded,filled", fillcolor=COLORS["cluster_c"],
               color="#0E5C00", penwidth="1.5", fontname=FONT, fontsize="12", fontcolor="#0E5C00")
        with c.subgraph() as s:
            s.attr(rank="same")
            s.node("VS", "Output Validation\n(Pydantic parse)",     fillcolor=COLORS["source"])
            s.node("CR", "Critic / Retry Loop\n(stagnation guard)", fillcolor=COLORS["source"])
            s.node("VF", "Post-Cleaning\nVerification",             fillcolor=COLORS["source"])
            s.node("FR", "Final Acceptance\n(diff engine)",         fillcolor=COLORS["agent"],
                   style="rounded,filled,bold")
        c.edge("VS", "CR")
        c.edge("CR", "VF")
        c.edge("VF", "FR")

    # ── Cross-layer flow (one clean edge per layer boundary) ────────────────
    dot.edge("ART", "PR")   # L1 → L2: typed schema into profiling
    dot.edge("EV",  "VA")   # L2 → L3: evidence bundle into validation agent
    dot.edge("VA",  "VS")   # L3 → L4: validated output into enforcement
    dot.edge("NA",  "VS")   # narrative agent also passes through enforcement
    # L4 → L1 feedback (dashed, non-constraining to avoid long stretch)
    dot.edge("FR", "ART", style="dashed", color="#555555", fontcolor="#555555",
             constraint="false")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 11. Final report assembly
# ---------------------------------------------------------------------------

def create_report_assembly(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("11_report_assembly", "Final report assembly",
                      rankdir="TB", ranksep="0.9", nodesep="0.7")

    # Inputs
    dot.node("verif_rep",  "VerificationReport\n(per-column outcomes\n+ overall assessment)",
             fillcolor=COLORS["artifact"])
    dot.node("manual_rep", "Manual review findings\n(forwarded unchanged\nfrom action router)",
             fillcolor=COLORS["artifact"])
    dot.node("val_bundle", "Validation bundle\n(OrchestrationStepResult\n— all stage reports)",
             fillcolor=COLORS["artifact"])

    with dot.subgraph() as s:
        s.attr(rank="same")
        for n in ["verif_rep", "manual_rep", "val_bundle"]:
            s.node(n)

    # Aggregation
    dot.node("agg", "Report aggregator\n(reporting.py)\n\ncollects per-stage findings\n+ remediation outcomes\n+ verification results",
             fillcolor=COLORS["action"], style="rounded,filled,bold")

    for n in ["verif_rep", "manual_rep", "val_bundle"]:
        dot.edge(n, "agg")

    # Structured report
    dot.node("final_rep", "FinalPipelineReport\n\nper-stage issue counts\nremediation summary\nverification outcomes\noverall status",
             fillcolor=COLORS["artifact"], style="rounded,filled,bold")
    dot.edge("agg", "final_rep")

    # Narrative agent
    dot.node("narr_agent", "Narrative report agent\n(LLM)\n\nreceives FinalPipelineReport\n→ writes human-readable\nMarkdown summary",
             fillcolor=COLORS["agent"])
    dot.edge("final_rep", "narr_agent")

    # Outputs
    dot.node("json_out", "FinalPipelineReport\n(structured JSON)",
             fillcolor=COLORS["output"], shape="folder")
    dot.node("md_out",   "Narrative report\n(Markdown)",
             fillcolor=COLORS["output"], shape="folder")

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("json_out")
        s.node("md_out")

    dot.edge("final_rep",  "json_out")
    dot.edge("narr_agent", "md_out")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Generating diagrams into '{OUTPUT_DIR}/'")
    create_dataflow_diagram()
    create_validation_flow_detail()
    create_cleaning_flow_detail()
    create_generation_validation_cycle()
    create_schema_validation_detail()
    create_consistency_detail()
    create_verification_detail()
    create_completeness_flow()
    create_remediation_policy_tree()
    create_four_layer_architecture()
    create_report_assembly()
    print("Done.")


if __name__ == "__main__":
    main()
