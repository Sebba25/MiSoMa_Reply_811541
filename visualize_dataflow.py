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

NODE_DEFAULTS  = dict(shape="box", style="rounded,filled", fontname=FONT, fontsize="11", margin="0.18,0.10")
EDGE_DEFAULTS  = dict(fontname=FONT, fontsize="9", color="#555555")
GRAPH_DEFAULTS = dict(bgcolor="white", fontname=FONT, pad="0.4", nodesep="0.55", ranksep="0.75", concentrate="true")

COLORS = {
    "source":    "#d5f5d5",   # green  – data files
    "agent":     "#dcd0ff",   # purple – LLM-backed nodes
    "artifact":  "#e8e8e8",   # grey   – typed data objects
    "action":    "#fff0c0",   # yellow – host-side logic / execution
    "output":    "#c8efc8",   # dark green – final outputs
    "cluster_v": "#f0f4ff",   # light blue cluster fill
    "cluster_c": "#fff8f0",   # light orange cluster fill
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
    dot = _base_graph("DataFlowPipeline", "Main pipeline overview", rankdir="TB", ranksep="0.9")

    # --- Input ---
    dot.node("csv",   "CSV input\n(Data/)",                 fillcolor=COLORS["source"], shape="folder")
    dot.node("entry", "Entrypoint\n(CLI / App / Notebook)", fillcolor=COLORS["action"])
    dot.edge("csv", "entry")

    # --- Validation cluster ---
    with dot.subgraph(name="cluster_validation") as v:
        v.attr(label="Validation half", style="rounded,filled", fillcolor=COLORS["cluster_v"],
               color="#3366cc", penwidth="1.5", fontname=FONT, fontsize="12", fontcolor="#3366cc")
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
               color="#cc6600", penwidth="1.5", fontname=FONT, fontsize="12", fontcolor="#cc6600")
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
    dot = _base_graph("ValidationFlow", "Validation pipeline detail", rankdir="TB")

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
        s.attr(label="Schema stage", style="rounded,dashed", color="#777777", fontname=FONT, fontsize="10")
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
    dot = _base_graph("CleaningFlow", "Cleaning pipeline detail", rankdir="TB")

    dot.node("in",  "Validation bundle",               fillcolor=COLORS["artifact"])
    dot.node("rem", "Remediation plan\n(RemediationPlan)", fillcolor=COLORS["artifact"])
    dot.node("req", "Cleaning requests\n(ColumnCleaningRequest[])", fillcolor=COLORS["artifact"])
    dot.node("gen", "Cleaner generation\n(generator + critic loop)", fillcolor=COLORS["agent"])
    dot.node("val", "Host-side\nvalidation",           fillcolor=COLORS["action"])
    dot.node("prg", "Accepted cleaners\n(ColumnCleanerProgram[])",  fillcolor=COLORS["artifact"])
    dot.node("app", "Application\n(execute on DataFrame)", fillcolor=COLORS["action"])
    dot.node("ver", "Verification\n(re-run consistency diff)", fillcolor=COLORS["agent"])
    dot.node("rpt", "Final report\n(FinalPipelineReport)", fillcolor=COLORS["artifact"])
    dot.node("out", "Cleaned CSV\n+ Narrative report", fillcolor=COLORS["output"], shape="folder")

    dot.edges([
        ("in",  "rem"),
        ("rem", "req"),
        ("req", "gen"),
        ("gen", "val"),
        ("val", "prg"),
        ("prg", "app"),
        ("app", "ver"),
        ("ver", "rpt"),
        ("rpt", "out"),
    ])

    # Reject path back to generator
    dot.edge("val", "gen", label="rejected →\nrepair prompt", style="dashed",
             color="#cc3300", fontcolor="#cc3300", constraint="false")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 4. Pydantic model / artifact flow
# ---------------------------------------------------------------------------

def create_schema_flow_diagram(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("SchemaFlow", "Pydantic model flow", rankdir="TB", ranksep="0.9")

    with dot.subgraph(name="cluster_val_models") as v:
        v.attr(label="Validation artifacts", style="rounded,filled", fillcolor=COLORS["cluster_v"],
               color="#3366cc", penwidth="1.2", fontname=FONT, fontsize="11", fontcolor="#3366cc")
        v.node("di",  "DatasetDtypeInference", fillcolor=COLORS["artifact"])
        v.node("sh",  "SchemaHandoff",         fillcolor=COLORS["artifact"])
        v.node("cr",  "CompletenessAnalysisReport",  fillcolor=COLORS["artifact"])
        v.node("cvr", "ConsistencyValidationReport", fillcolor=COLORS["artifact"])
        v.node("ar",  "AnomalyDetectionReport",      fillcolor=COLORS["artifact"])
        v.node("xr",  "CrossColumnValidationReport", fillcolor=COLORS["artifact"])
        v.node("dr",  "DuplicateDetectionReport",    fillcolor=COLORS["artifact"])
        v.node("osr", "OrchestrationStepResult",     fillcolor=COLORS["artifact"],
               style="rounded,filled,bold")
        v.edge("di",  "sh")
        for n in ["sh", "cr", "cvr", "ar", "xr", "dr"]:
            v.edge(n, "osr")

    with dot.subgraph(name="cluster_clean_models") as c:
        c.attr(label="Cleaning artifacts", style="rounded,filled", fillcolor=COLORS["cluster_c"],
               color="#cc6600", penwidth="1.2", fontname=FONT, fontsize="11", fontcolor="#cc6600")
        c.node("rp",  "RemediationPlan",          fillcolor=COLORS["artifact"])
        c.node("ccr", "ColumnCleaningRequest",     fillcolor=COLORS["artifact"])
        c.node("ccp", "ColumnCleanerProgram",      fillcolor=COLORS["artifact"])
        c.node("cpr", "CleaningPipelineResult",    fillcolor=COLORS["artifact"])
        c.node("fpr", "FinalPipelineReport",       fillcolor=COLORS["artifact"],
               style="rounded,filled,bold")
        c.edge("rp",  "ccr",  label="per column")
        c.edge("ccr", "ccp",  label="generation")
        c.edge("ccp", "cpr",  label="application")
        c.edge("cpr", "fpr",  label="narrative")

    dot.edge("osr", "rp", label="planning")

    _render(dot, output_dir)


# ---------------------------------------------------------------------------
# 5. Generation–validation–critic cycle
# ---------------------------------------------------------------------------

def create_generation_validation_cycle(output_dir: str = OUTPUT_DIR) -> None:
    dot = _base_graph("GenerationValidationCycle", "Code generation cycle", rankdir="TB", ranksep="0.85")

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
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Generating diagrams into '{OUTPUT_DIR}/'")
    create_dataflow_diagram()
    create_validation_flow_detail()
    create_cleaning_flow_detail()
    create_schema_flow_diagram()
    create_generation_validation_cycle()
    print("Done.")


if __name__ == "__main__":
    main()
