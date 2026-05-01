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

FONT = "Helvetica"

NODE_DEFAULTS  = dict(shape="box", style="rounded,filled", fontname=FONT, fontsize="11",
                      margin="0.18,0.10", fontcolor="#2D2D2D")
EDGE_DEFAULTS  = dict(fontname=FONT, fontsize="9", color="#555555", fontcolor="#555555")
GRAPH_DEFAULTS = dict(bgcolor="white", fontname=FONT, pad="0.4", nodesep="0.55", ranksep="0.75", concentrate="true")

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
    dot = _base_graph("PipelineOverview", "Main pipeline overview", rankdir="TB", ranksep="0.9")

    # --- Input ---
    dot.node("csv",   "CSV input\n(Data/)",                 fillcolor=COLORS["source"], shape="folder")
    dot.node("entry", "Entrypoint\n(CLI / App / Notebook)", fillcolor=COLORS["action"])
    dot.edge("csv", "entry")

    # --- Validation cluster ---
    with dot.subgraph(name="cluster_validation") as v:
        v.attr(label="Validation half", style="rounded,filled", fillcolor=COLORS["cluster_v"],
               color="#22A30A", penwidth="1.5", fontname=FONT, fontsize="12", fontcolor="#22A30A")
        # Force all stage nodes onto the same horizontal rank
        with v.subgraph() as same:
            same.attr(rank="same")
            same.node("schema",   "1. Schema\nvalidation",     fillcolor=COLORS["agent"])
            same.node("complete", "2. Completeness\nanalysis",  fillcolor=COLORS["agent"])
            same.node("consist",  "3. Format\nconsistency",     fillcolor=COLORS["agent"])
            same.node("anomaly",  "4. Anomaly\ndetection",      fillcolor=COLORS["agent"])
            same.node("cross",    "5. Cross-column\nchecks",    fillcolor=COLORS["agent"])
            same.node("dupes",    "6. Duplicate\ndetection",    fillcolor=COLORS["agent"])
        # Invisible edges to preserve left-to-right order
        for a, b in [("schema","complete"),("complete","consist"),("consist","anomaly"),
                     ("anomaly","cross"),("cross","dupes")]:
            v.edge(a, b, style="invis")

    # --- Intermediate bundle node (between the two halves) ---
    dot.node("bundle", "Validation bundle\n(OrchestrationStepResult)",
             fillcolor=COLORS["artifact"], style="rounded,filled,bold")

    # --- Cleaning cluster ---
    with dot.subgraph(name="cluster_cleaning") as c:
        c.attr(label="Cleaning half", style="rounded,filled", fillcolor=COLORS["cluster_c"],
               color="#1A8A00", penwidth="1.5", fontname=FONT, fontsize="12", fontcolor="#1A8A00")
        c.node("remediate", "1. Remediation\nplanning",  fillcolor=COLORS["agent"])
        c.node("generate",  "2. Cleaner\ngeneration",    fillcolor=COLORS["agent"])
        c.node("apply",     "3. Application\n(execute)", fillcolor=COLORS["action"])
        c.node("verify",    "4. Verification",           fillcolor=COLORS["agent"])
        c.node("report",    "5. Report\ngeneration",     fillcolor=COLORS["agent"])
        c.edges([
            ("remediate", "generate"),
            ("generate",  "apply"),
            ("apply",     "verify"),
            ("verify",    "report"),
        ])

    # --- Outputs ---
    dot.node("clean_csv", "Cleaned CSV",                  fillcolor=COLORS["output"], shape="folder")
    dot.node("report_md", "Narrative report\n(Markdown)", fillcolor=COLORS["output"], shape="folder")

    # --- Connections ---
    dot.edge("entry",    "schema",   label="dataset")
    for stage in ["schema", "complete", "consist", "anomaly", "cross", "dupes"]:
        dot.edge(stage, "bundle")
    dot.edge("bundle",   "remediate", label="all findings")
    dot.edge("report",   "clean_csv")
    dot.edge("report",   "report_md")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 2. Validation stage detail
# ---------------------------------------------------------------------------

def create_validation_flow_detail(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("ValidationStagePipeline", "Validation pipeline detail", rankdir="TB")

    dot.node("df",  "Raw DataFrame",    fillcolor=COLORS["source"])
    dot.node("sch", "Schema handoff\n(SchemaHandoff)",           fillcolor=COLORS["artifact"])
    dot.node("com", "Completeness report\n(CompletenessAnalysisReport)", fillcolor=COLORS["artifact"])
    dot.node("con", "Consistency report\n(ConsistencyValidationReport)", fillcolor=COLORS["artifact"])
    dot.node("ano", "Anomaly report\n(AnomalyDetectionReport)",  fillcolor=COLORS["artifact"])
    dot.node("cro", "Cross-column report\n(CrossColumnValidationReport)", fillcolor=COLORS["artifact"])
    dot.node("dup", "Duplicate report\n(DuplicateDetectionReport)", fillcolor=COLORS["artifact"])
    dot.node("bun", "Validation bundle\n(OrchestrationStepResult)", fillcolor=COLORS["artifact"],
             style="rounded,filled,bold")

    # Schema has an internal dtype-inference sub-step
    with dot.subgraph(name="cluster_schema") as s:
        s.attr(label="Schema stage", style="rounded,dashed", color="#555555", fontname=FONT, fontsize="10")
        s.node("prof",  "Deterministic\nprofiling",     fillcolor=COLORS["action"])
        s.node("dtype", "dtype-inference\nagent",       fillcolor=COLORS["agent"])
        s.node("name",  "Naming &\nduplication checks", fillcolor=COLORS["action"])
        s.edge("prof",  "dtype")
        s.edge("prof",  "name")
        s.edge("dtype", "sch")
        s.edge("name",  "sch")

    dot.edge("df",  "prof")
    dot.edge("sch", "com", label="schema passed\nforward to all stages", style="dashed", color="#888888")
    dot.edge("com", "con")
    dot.edge("con", "ano")
    dot.edge("ano", "cro")
    dot.edge("cro", "dup")
    dot.edge("dup", "bun")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 3. Cleaning stage detail
# ---------------------------------------------------------------------------

def create_cleaning_flow_detail(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("CleaningHalfPipeline", "Cleaning pipeline detail", rankdir="TB", ranksep="0.9")

    dot.node("bundle",  "Validation bundle\n(OrchestrationStepResult)",    fillcolor=COLORS["artifact"])
    dot.node("plan",    "RemediationPlan\n(RemediationAction[])",           fillcolor=COLORS["artifact"])
    dot.edge("bundle",  "plan")

    # Split auto-apply vs manual
    dot.node("split",   "Action router",                                    fillcolor=COLORS["action"], shape="diamond")
    dot.edge("plan",    "split")

    dot.node("manual",  "manual_review / report_only\n(no code generated,\nforwarded to report)",
             fillcolor=COLORS["artifact"])
    dot.node("auto",    "auto_apply actions\n(cast · rename · placeholder→null\n· exact duplicate drop)",
             fillcolor=COLORS["artifact"])
    dot.node("consist", "format_fix actions\n(FormatConsistencyFinding\nper column)",
             fillcolor=COLORS["artifact"])

    with dot.subgraph() as s:
        s.attr(rank="same")
        for n in ["manual", "auto", "consist"]:
            s.node(n)

    dot.edge("split",  "manual",  label="ambiguous /\nhigh risk",  style="dashed", color="#888888", fontcolor="#888888")
    dot.edge("split",  "auto",    label="low risk")
    dot.edge("split",  "consist", label="format inconsistency\nfound")

    # Consist path → generation loop
    dot.node("req",     "ColumnCleaningRequest\n\ntarget dtype · valid examples\ninconsistent examples · strategy",
             fillcolor=COLORS["artifact"])
    dot.node("gen",     "Generator + critic loop\n(see GenerationValidationCycle)",
             fillcolor=COLORS["agent"])
    dot.node("cleaners","Accepted cleaners\n(ColumnCleanerProgram[])",      fillcolor=COLORS["artifact"])

    dot.edge("consist", "req")
    dot.edge("req",     "gen")
    dot.edge("gen",     "cleaners")

    # Application stage — ordered execution
    dot.node("app_label", "Application\n(application.py)\n— ordered execution —",
             fillcolor=COLORS["action"], style="rounded,filled,bold")

    dot.edge("cleaners", "app_label", label="step 1: apply\ngenerated cleaners")
    dot.edge("auto",     "app_label", label="steps 2–5:\nstructural actions")

    with dot.subgraph(name="cluster_order") as o:
        o.attr(label="Application order", style="rounded,dashed", color="#555555", fontname=FONT, fontsize="10")
        o.node("s1", "1. Generated cleaners\n(column identities still intact)", fillcolor=COLORS["action"])
        o.node("s2", "2. Placeholder → null",                                    fillcolor=COLORS["action"])
        o.node("s3", "3. Drop exact duplicate columns",                          fillcolor=COLORS["action"])
        o.node("s4", "4. Column renames",                                        fillcolor=COLORS["action"])
        o.node("s5", "5. dtype casts",                                           fillcolor=COLORS["action"])
        o.edges([("s1","s2"),("s2","s3"),("s3","s4"),("s4","s5")])

    dot.edge("app_label", "s1")

    dot.node("cleaned_csv", "Cleaned CSV",                                   fillcolor=COLORS["output"], shape="folder")
    dot.edge("s5",          "cleaned_csv")

    # Verification feeds into report
    dot.node("verif",   "Verification\n(re-run consistency diff,\nbefore vs after)",  fillcolor=COLORS["agent"])
    dot.node("report",  "FinalPipelineReport\n+ Narrative report",                    fillcolor=COLORS["artifact"], style="rounded,filled,bold")
    dot.node("out",     "Cleaned CSV\n+ Markdown report",                             fillcolor=COLORS["output"], shape="folder")

    dot.edge("cleaned_csv", "verif")
    dot.edge("manual",      "report", style="dashed", color="#888888")
    dot.edge("verif",       "report")
    dot.edge("report",      "out")

    _render(dot, output_dir)



# ---------------------------------------------------------------------------
# 5. Generation–validation–critic cycle
# ---------------------------------------------------------------------------

def create_generation_validation_cycle(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("CleanerGenerationLoop", "Code generation cycle", rankdir="TB", ranksep="0.85")

    dot.node("req",  "ColumnCleaningRequest\n(target dtype, valid examples,\ninconsistent examples)", fillcolor=COLORS["artifact"])
    dot.node("gen",  "Generator agent\n(writes Python cleaner fn)", fillcolor=COLORS["agent"])
    dot.node("self", "Self-test\n(CodeExecutionTool,\nbounded)", fillcolor=COLORS["action"])
    dot.node("hval", "Host-side validation\n(syntax · signature · security\n· preservation rules)", fillcolor=COLORS["action"])
    dot.node("ok",   "Cleaner accepted\n(ColumnCleanerProgram)", fillcolor=COLORS["output"])
    dot.node("crit", "Critic agent\n(diagnosis of\nfailed checks)", fillcolor=COLORS["agent"])
    dot.node("stag", "Stagnation detector\n(repeated code / fingerprint\n→ raise temperature)", fillcolor=COLORS["action"])

    dot.edges([
        ("req",  "gen"),
        ("gen",  "self"),
        ("self", "hval"),
        ("hval", "ok",   ),
    ])

    # Failure loop
    dot.edge("hval", "crit", label="checks failed", color="#cc3300", fontcolor="#cc3300")
    dot.edge("crit", "stag", color="#cc3300")
    dot.edge("stag", "gen",  label="retry with repair prompt\n(max attempts)", color="#cc3300",
             fontcolor="#cc3300", style="dashed", constraint="false")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 6. Schema validation internals
# ---------------------------------------------------------------------------

def create_schema_validation_detail(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("SchemaStageInternals", "Schema validation internals", rankdir="TB", ranksep="0.85")

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
    dot = _base_graph("FormatConsistencyPaths", "Format consistency fast vs slow path", rankdir="TB", ranksep="0.9")

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
# 8. Remediation planning — findings → actions
# ---------------------------------------------------------------------------

def create_remediation_detail(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("RemediationPlanning", "Remediation planning", rankdir="TB", ranksep="0.85")

    dot.node("bundle", "Validation bundle\n(OrchestrationStepResult)", fillcolor=COLORS["artifact"])

    # Finding families (same rank)
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("schema_f",   "Schema findings\n(dtype cast, rename,\nduplicate columns)",  fillcolor=COLORS["artifact"])
        s.node("complete_f", "Completeness findings\n(placeholder tokens,\nsparse columns)", fillcolor=COLORS["artifact"])
        s.node("consist_f",  "Consistency findings\n(FormatConsistencyFinding)",            fillcolor=COLORS["artifact"])
        s.node("other_f",    "Anomaly · cross-column\n· duplicate findings",                fillcolor=COLORS["artifact"])

    dot.edge("bundle", "schema_f")
    dot.edge("bundle", "complete_f")
    dot.edge("bundle", "consist_f")
    dot.edge("bundle", "other_f")

    dot.node("planner", "Remediation planner\n(remediation.py)\n\nclassifies each finding\nby risk level and evidence strength",
             fillcolor=COLORS["agent"])

    for n in ["schema_f", "complete_f", "consist_f", "other_f"]:
        dot.edge(n, "planner")

    # Decision split
    dot.node("decision", "Risk classification",
             fillcolor=COLORS["action"], shape="diamond")
    dot.edge("planner", "decision")

    dot.node("auto",   "auto_apply = True\n\ndtype cast · safe rename\nplaceholder→null\nexact duplicate column/row drop",
             fillcolor=COLORS["output"])
    dot.node("manual", "manual_review / report_only\n\nanomalies · near-duplicate columns\nsemantic conflicts · date-order violations\nnear-duplicate rows",
             fillcolor=COLORS["artifact"])

    dot.edge("decision", "auto",   label="low risk,\nmechanically justified")
    dot.edge("decision", "manual", label="ambiguous or\nhigh impact")

    dot.node("plan", "RemediationPlan\n(RemediationAction[])", fillcolor=COLORS["artifact"], style="rounded,filled,bold")
    dot.edge("auto",   "plan")
    dot.edge("manual", "plan")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 9. Verification — before/after comparison
# ---------------------------------------------------------------------------

def create_verification_detail(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("PostCleaningVerification", "Post-cleaning verification", rankdir="TB", ranksep="0.9")

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
        s.node("resolved",   "resolved\n(issue gone)",              fillcolor="#22A30A", fontcolor="white")
        s.node("improved",   "improved\n(inconsistencies reduced)", fillcolor="#5DC73A")
        s.node("unchanged",  "unchanged\n(no effect)",              fillcolor=COLORS["artifact"])
        s.node("regressed",  "regressed\n(new issues introduced)",  fillcolor="#d62828", fontcolor="white")

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
    dot = _base_graph("CompletenessDetectionFlow", "Completeness detection: building the missing-like mask",
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
    dot.edge("prof",  "agent", label="bounded profile\n(counts, examples)")
    dot.edge("agent", "out")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 11. Remediation policy decision tree  (finding → action category)
# ---------------------------------------------------------------------------

def create_remediation_policy_tree(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("RemediationPolicyTree", "Remediation policy: finding to action category",
                      rankdir="TB", ranksep="0.9")

    dot.node("finding", "Validation finding\n(any stage)",             fillcolor=COLORS["artifact"])

    # Gate 1 — source stage
    dot.node("g1", "Source stage?",                                     fillcolor=COLORS["action"], shape="diamond")
    dot.edge("finding", "g1")

    # Schema / completeness branch
    dot.node("g2", "Action type?",                                      fillcolor=COLORS["action"], shape="diamond")
    dot.edge("g1", "g2", label="schema /\ncompleteness")

    dot.node("auto_struct", "auto_apply = True\n\ndtype cast\nsafe rename\nplaceholder → null\nexact duplicate\ncolumn / row drop",
             fillcolor=COLORS["output"])
    dot.edge("g2", "auto_struct", label="structural,\nmechanical")

    dot.node("manual_schema", "manual_review\n\nunsafe rename\nnear-duplicate columns\nsemantic conflict",
             fillcolor=COLORS["artifact"])
    dot.edge("g2", "manual_schema", label="ambiguous /\nhigh impact")

    # Consistency branch
    dot.node("g3", "Canonical target\nunambiguous?",                    fillcolor=COLORS["action"], shape="diamond")
    dot.edge("g1", "g3", label="consistency")

    dot.node("format_fix", "format_fix action\n→ ColumnCleaningRequest\n→ cleaner generation loop",
             fillcolor=COLORS["agent"])
    dot.edge("g3", "format_fix", label="yes")

    dot.node("report_only_c", "report_only\n(no safe target\ncan be defined)",
             fillcolor=COLORS["artifact"])
    dot.edge("g3", "report_only_c", label="no")

    # Anomaly / cross-column / duplicate branch
    dot.node("g4", "Risk level?",                                       fillcolor=COLORS["action"], shape="diamond")
    dot.edge("g1", "g4", label="anomaly /\ncross-column /\nduplicate")

    dot.node("auto_dup", "auto_apply = True\n\nexact duplicate rows\nexact duplicate columns",
             fillcolor=COLORS["output"])
    dot.edge("g4", "auto_dup", label="low risk\n(exact duplicate)")

    dot.node("manual_amb", "manual_review\n\nnear-duplicate rows/columns\ndate-order violations\ntemporal mismatches\nrare categories\nnumeric outliers",
             fillcolor=COLORS["artifact"])
    dot.edge("g4", "manual_amb", label="medium / high\nor ambiguous")

    # All paths feed into plan
    dot.node("plan", "RemediationPlan\n(RemediationAction[])",          fillcolor=COLORS["artifact"], style="rounded,filled,bold")
    for src in ["auto_struct", "manual_schema", "format_fix", "report_only_c", "auto_dup", "manual_amb"]:
        dot.edge(src, "plan")

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
    create_remediation_detail()
    create_verification_detail()
    create_completeness_flow()
    create_remediation_policy_tree()
    print("Done.")


if __name__ == "__main__":
    main()
