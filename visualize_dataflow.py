"""
Data Flow Visualization for the Agents AI Pipeline

This script generates a visual representation of how data flows through the entire
validation and cleaning pipeline. It shows:
- Data sources and destinations
- Pipeline stages (validation half and cleaning half)
- Intermediate artifacts and caches
- Agent involvement at each stage
"""

import graphviz
from pathlib import Path

OUTPUT_DIR = "images/flow_diagrams"


def create_dataflow_diagram(output_dir: str = OUTPUT_DIR) -> None:
    """
    Create a comprehensive data flow diagram showing the entire pipeline.
    
    Args:
        output_dir: Directory to save the diagram (default: images/flow_diagrams)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dot = graphviz.Digraph(
        name="DataFlowPipeline",
        comment="Data Flow Through Agents AI Pipeline",
        directory=output_dir,
        format="png",
    )
    
    dot.attr(rankdir="TB")  # Top to bottom
    dot.attr("graph", bgcolor="white", splines="ortho", nodesep="0.7", ranksep="0.9", size="12,18!")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue", fontname="Arial")
    dot.attr("edge", fontname="Arial", fontsize="9")
    
    # ==================== DATA SOURCES ====================
    dot.node("csv_source", "CSV Files\n(Data/)", fillcolor="lightgreen", shape="folder")
    # ==================== ENTRYPOINTS ====================
    dot.node("cli", "CLI Interface\n(cli.py)", fillcolor="lightyellow")
    dot.node("main", "Main Entrypoint\n(main.py)", fillcolor="lightyellow")

    # Connection to CLI
    dot.edge("csv_source", "cli", label="--stage, --column")
    dot.edge("main", "cli", label="dispatch")
    
    # ==================== VALIDATION HALF ====================
    dot.node("validation_group", "VALIDATION HALF", shape="rectangle", style="filled", fillcolor="lightcyan", fontsize="12", fontweight="bold")
    
    dot.node("schema_agent", "1. Schema Validation\n(Agent)\n(includes dtype inference)", fillcolor="lavender")
    dot.node("completeness_agent", "3. Completeness\n(Agent)", fillcolor="lavender")
    dot.node("consistency_agent", "4. Consistency\n(Agent)", fillcolor="lavender")
    dot.node("anomaly_agent", "5. Anomaly Detection\n(Agent)\n(always fresh)", fillcolor="lavender")
    dot.node("cross_column_agent", "6. Cross-Column\n(Agent)\n(always fresh)", fillcolor="lavender")
    dot.node("duplicates_agent", "7. Duplicates\n(Agent)\n(always fresh)", fillcolor="lavender")
    
    dot.node("validation_bundle", "OrchestrationStepResult\n(Bundled Validation)", fillcolor="lightgray")
    
    # Validation flow
    dot.edge("cli", "schema_agent", label="run_schema_validation()")
    dot.edge("schema_agent", "completeness_agent", label="SchemaHandoff")
    dot.edge("completeness_agent", "consistency_agent", label="CompletenessAnalysisReport")
    dot.edge("consistency_agent", "anomaly_agent", label="ConsistencyValidationReport")
    dot.edge("schema_agent", "anomaly_agent", label="schema cache")
    dot.edge("schema_agent", "cross_column_agent", label="schema cache")
    dot.edge("schema_agent", "duplicates_agent", label="schema cache")
    dot.edge("anomaly_agent", "cross_column_agent", label="AnomalyDetectionReport")
    dot.edge("cross_column_agent", "duplicates_agent", label="CrossColumnValidationReport")
    
    dot.edge("duplicates_agent", "validation_bundle", label="DuplicateDetectionReport")
    
    # ==================== CLEANING HALF ====================
    dot.node("cleaning_group", "CLEANING HALF", shape="rectangle", style="filled", fillcolor="lightcyan", fontsize="12", fontweight="bold")
    
    dot.node("remediation_stage", "1. Remediation Planning\n(Agent)", fillcolor="lavender")
    dot.node("generation_stage", "2. Code Generation\n(Generator + Critic)", fillcolor="lavender")
    dot.node("application_stage", "3. Cleaner Application\n(Execute Python)", fillcolor="lavender")
    dot.node("verification_stage", "4. Verification\n(Agent)", fillcolor="lavender")
    dot.node("reporting_stage", "5. Narrative Report\n(Agent)", fillcolor="lavender")
    
    dot.node("remediation_plan", "RemediationPlan", fillcolor="lightgray")
    dot.node("cleaner_programs", "ColumnCleanerProgram\n(Generated Python)", fillcolor="lightgray")
    dot.node("pipeline_result", "CleaningPipelineResult", fillcolor="lightgray")
    
    # Cleaning flow
    dot.edge("validation_bundle", "remediation_stage", label="run_remediation_planning()")
    dot.edge("remediation_stage", "remediation_plan", label="RemediationPlan")
    
    dot.edge("remediation_plan", "generation_stage", label="run_cleaner_generation()")
    dot.edge("generation_stage", "cleaner_programs", label="ColumnCleanerProgram[]")
    
    dot.edge("cleaner_programs", "application_stage", label="run_cleaner_application()")
    dot.edge("application_stage", "verification_stage", label="Cleaned CSV")
    
    dot.edge("verification_stage", "reporting_stage", label="run_verify()\n(diff consistency)")
    dot.edge("reporting_stage", "pipeline_result", label="FinalPipelineReport")
    
    # ==================== DATA OUTPUTS ====================
    dot.node("cleaned_csv", "Cleaned CSV Output", fillcolor="lightgreen", shape="folder")
    dot.node("report_md", "Narrative Report\n(Markdown)", fillcolor="lightgreen", shape="folder")
    dot.node("report_json", "JSON Report", fillcolor="lightgreen", shape="folder")
    
    dot.edge("pipeline_result", "cleaned_csv", label="save_cleaned_dataset()")
    dot.edge("pipeline_result", "report_md", label="save_narrative_report()")
    dot.edge("pipeline_result", "report_json", label="save_final_report()")
    
    # Save diagram
    dot.render(cleanup=True)
    print(f"✓ Data flow diagram created: {output_dir}/DataFlowPipeline.gv.png")


def create_validation_flow_detail(output_dir: str = OUTPUT_DIR) -> None:
    """Create a detailed diagram of just the validation stage."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dot = graphviz.Digraph(
        name="ValidationFlow",
        comment="Detailed Validation Pipeline",
        directory=output_dir,
        format="png",
    )
    
    dot.attr(rankdir="TB")  # Top to bottom
    dot.attr("graph", bgcolor="white", splines="ortho", nodesep="0.7", ranksep="0.9", size="10,16!")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="lavender", fontname="Arial")
    
    stages = [
        ("schema", "1. Schema Validation", "Build schema and naming fixes\n(includes dtype inference)"),
        ("completeness", "2. Completeness", "Check for missing/null\nvalues"),
        ("consistency", "3. Consistency", "Verify format consistency\nwithin columns"),
        ("anomaly", "4. Anomaly Detection", "Always fresh run\n(no cache reuse)"),
        ("cross_col", "5. Cross-Column", "Always fresh run\n(no cache reuse)"),
        ("duplicates", "6. Duplicates", "Always fresh run\n(no cache reuse)"),
    ]
    
    for i, (key, title, desc) in enumerate(stages):
        dot.node(key, f"{title}\n\n{desc}", fillcolor="lavender")
        if i > 0:
            prev_key = stages[i-1][0]
            dot.edge(prev_key, key)
    
    dot.node("dtype", "Dtype Inference\n(Agent)\n(standalone stage)", fillcolor="lavender")
    dot.edge("schema", "dtype", label="internal step", style="dashed")
    dot.edge("dtype", "schema", label="DatasetDtypeInference", style="dashed")

    dot.node("bundle", "Validation Bundle\n(OrchestrationStepResult)", fillcolor="lightgray", shape="box")
    dot.edge("duplicates", "bundle")
    
    dot.render(cleanup=True)
    print(f"✓ Validation detail diagram created: {output_dir}/ValidationFlow.gv.png")


def create_cleaning_flow_detail(output_dir: str = OUTPUT_DIR) -> None:
    """Create a detailed diagram of just the cleaning stage."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dot = graphviz.Digraph(
        name="CleaningFlow",
        comment="Detailed Cleaning Pipeline",
        directory=output_dir,
        format="png",
    )
    
    dot.attr(rankdir="TB")  # Top to bottom
    dot.attr("graph", bgcolor="white", splines="ortho", nodesep="0.7", ranksep="0.9", size="10,16!")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="lavender", fontname="Arial")
    
    stages = [
        ("planning", "Remediation Planning\n(Agent)", "Analyze validation issues\nand plan fixes"),
        ("generation", "Code Generation\n(Generator + Critic)", "Generate Python cleaner\nfunctions"),
        ("validation", "Cleaner Validation\n(Host-side checks)", "Verify generated code\nfor safety"),
        ("application", "Cleaner Application\n(Execute Python)", "Run cleaners on CSV\ndata"),
        ("verification", "Verification\n(Agent)", "Compare consistency findings\nbefore/after cleaning"),
        ("reporting", "Report Generation\n(Agent)", "Create narrative\nreport"),
    ]
    
    for i, (key, title, desc) in enumerate(stages):
        dot.node(key, f"{title}\n\n{desc}", fillcolor="lavender")
        if i > 0:
            prev_key = stages[i-1][0]
            dot.edge(prev_key, key)
    
    dot.node("output", "Final Output\n(CSV + Report)", fillcolor="lightgreen", shape="folder")
    dot.edge("reporting", "output")
    
    dot.render(cleanup=True)
    print(f"✓ Cleaning detail diagram created: {output_dir}/CleaningFlow.gv.png")


def create_schema_flow_diagram(output_dir: str = OUTPUT_DIR) -> None:
    """Create a diagram showing Pydantic model relationships."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dot = graphviz.Digraph(
        name="SchemaFlow",
        comment="Pydantic Model Flow",
        directory=output_dir,
        format="png",
    )
    
    dot.attr(rankdir="TB")
    dot.attr("graph", bgcolor="white", splines="ortho", nodesep="0.7", ranksep="0.9", size="12,18!")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightcyan", fontname="Arial", fontsize="9")
    
    # Validation side models
    dot.node("dtype_inf", "DatasetDtypeInference", fillcolor="lavender")
    dot.node("schema_ho", "SchemaHandoff", fillcolor="lavender")
    dot.node("complete_rep", "CompletenessAnalysisReport", fillcolor="lavender")
    dot.node("consist_rep", "ConsistencyValidationReport", fillcolor="lavender")
    dot.node("anomaly_rep", "AnomalyDetectionReport", fillcolor="lavender")
    dot.node("cross_rep", "CrossColumnValidationReport", fillcolor="lavender")
    dot.node("dup_rep", "DuplicateDetectionReport", fillcolor="lavender")
    
    dot.node("orch_result", "OrchestrationStepResult\n(Bundle)", fillcolor="lightgray", shape="rectangle")
    
    dot.edge("dtype_inf", "schema_ho")
    dot.edge("schema_ho", "orch_result")
    dot.edge("complete_rep", "orch_result")
    dot.edge("consist_rep", "orch_result")
    dot.edge("anomaly_rep", "orch_result")
    dot.edge("cross_rep", "orch_result")
    dot.edge("dup_rep", "orch_result")
    
    # Cleaning side models
    dot.node("rem_plan", "RemediationPlan", fillcolor="lightpink")
    dot.node("clean_req", "ColumnCleaningRequest", fillcolor="lightpink")
    dot.node("clean_prog", "ColumnCleanerProgram", fillcolor="lightpink")
    dot.node("clean_artifact", "GeneratedCleanerArtifact", fillcolor="lightpink")
    
    dot.edge("orch_result", "rem_plan", label="planning")
    dot.edge("rem_plan", "clean_req", label="per-column")
    dot.edge("clean_req", "clean_prog", label="generation")
    dot.edge("clean_prog", "clean_artifact", label="execution")
    
    # Final models
    dot.node("clean_result", "CleaningPipelineResult", fillcolor="lightgreen")
    dot.node("final_report", "FinalPipelineReport", fillcolor="lightgreen")
    
    dot.edge("clean_artifact", "clean_result")
    dot.edge("clean_result", "final_report", label="narrative")
    
    dot.render(cleanup=True)
    print(f"✓ Schema flow diagram created: {output_dir}/SchemaFlow.gv.png")


def create_generation_validation_cycle(output_dir: str = OUTPUT_DIR) -> None:
    """Create a detailed diagram of the generation-validation-critic cycle."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dot = graphviz.Digraph(
        name="GenerationValidationCycle",
        comment="Code Generation and Validation Cycle",
        directory=output_dir,
        format="png",
    )
    
    dot.attr(rankdir="TB")  # Top to bottom
    dot.attr("graph", bgcolor="white", splines="ortho", nodesep="0.7", ranksep="0.9", size="10,16!")
    dot.attr("node", shape="box", style="rounded,filled", fontname="Arial", fontsize="10")
    
    # Input
    dot.node("remediation_input", "Remediation Plan\n+ Format Facts\n+ Schema Info", fillcolor="lightblue")
    
    # Generation loop
    dot.node("column_cleaning_req", "ColumnCleaningRequest\n(Per-Column Task)", fillcolor="lightcyan")
    dot.edge("remediation_input", "column_cleaning_req")
    
    # Generator Agent
    dot.node("generator_agent", "Generator Agent\n(Create Python Cleaner)", fillcolor="lavender", shape="box")
    dot.edge("column_cleaning_req", "generator_agent", label="generate_cleaner()")
    
    # Generated Code
    dot.node("generated_code", "Generated Cleaner Code\n(Python Function)", fillcolor="lightyellow")
    dot.edge("generator_agent", "generated_code", label="ColumnCleanerProgram")
    
    # Critic Agent
    dot.node("critic_agent", "Critic Agent\n(Review & Refine)", fillcolor="lavender", shape="box")
    dot.edge("generated_code", "critic_agent", label="review_code()")
    
    # Critic Decision
    dot.node("critic_decision", "Critique Result", fillcolor="lightgray", shape="diamond")
    dot.edge("critic_agent", "critic_decision")
    
    # Retry loop
    dot.node("retry_count", "Attempt < Max?", fillcolor="lightyellow", shape="diamond")
    dot.edge("critic_decision", "retry_count", label="[Needs Improvement]")
    dot.edge("retry_count", "generator_agent", label="Yes: Refine", constraint="false")
    
    # Validation - Host-side
    dot.node("host_validation", "Host-Side Validation\n(Safety Checks)", fillcolor="lightpink")
    dot.edge("critic_decision", "host_validation", label="[Accepted]")
    
    # Validation checks
    dot.node("syntax_check", "Syntax Check", fillcolor="lightcoral", shape="ellipse")
    dot.node("security_check", "Security Check", fillcolor="lightcoral", shape="ellipse")
    dot.node("signature_check", "Signature Check", fillcolor="lightcoral", shape="ellipse")
    
    dot.edge("host_validation", "syntax_check")
    dot.edge("host_validation", "security_check")
    dot.edge("host_validation", "signature_check")
    
    # Validation result
    dot.node("validation_result", "Validation Result\n(fingerprint)", fillcolor="lightgray", shape="diamond")
    dot.edge("syntax_check", "validation_result")
    dot.edge("security_check", "validation_result")
    dot.edge("signature_check", "validation_result")
    
    # Success or Failure
    dot.node("approved", "CleanerValidationApproved\n(Ready for Application)", fillcolor="lightgreen")
    dot.node("rejected", "CleanerValidationIssue\n(Rejected)", fillcolor="lightcoral")
    
    dot.edge("validation_result", "approved", label="[Valid]")
    dot.edge("validation_result", "rejected", label="[Invalid]")

    # Stagnation detection
    dot.node("stagnation", "Stagnation Check\n(same code or\nvalidation fingerprint)", fillcolor="lightyellow", shape="box")
    dot.edge("generated_code", "stagnation", style="dashed")
    dot.edge("validation_result", "stagnation", style="dashed")
    dot.edge("stagnation", "retry_count", label="Rewrite + bump temp", constraint="false")

    dot.edge("rejected", "retry_count", label="Max retries exceeded?\nFail", constraint="false")
    
    # Application stage
    dot.node("application", "Cleaner Application\n(Execute on Data)", fillcolor="lightblue")
    dot.edge("approved", "application", label="run_cleaner()")
    
    # Output
    dot.node("cleaned_column", "Cleaned Column Data\n(CSV Output)", fillcolor="lightgreen")
    dot.edge("application", "cleaned_column")
    
    dot.render(cleanup=True)
    print(f"✓ Generation-Validation cycle diagram created: {output_dir}/GenerationValidationCycle.gv.png")


def main():
    """Generate all data flow diagrams."""
    print("=" * 60)
    print("Generating Data Flow Visualizations...")
    print("=" * 60)
    print()
    
    create_dataflow_diagram()
    create_validation_flow_detail()
    create_cleaning_flow_detail()
    create_schema_flow_diagram()
    create_generation_validation_cycle()
    
    print("\n" + "=" * 60)
    print("All diagrams generated successfully!")
    print("=" * 60)
    print(f"\nGenerated files in '{OUTPUT_DIR}/' directory:")
    print("  • DataFlowPipeline.gv.png         - Main data flow overview")
    print("  • ValidationFlow.gv.png           - Detailed validation pipeline")
    print("  • CleaningFlow.gv.png             - Detailed cleaning pipeline")
    print("  • GenerationValidationCycle.gv.png - Code generation & validation cycle")
    print("  • SchemaFlow.gv.png               - Pydantic model relationships")
    print("\nOpen the PNG files to visualize your data flow.")


if __name__ == "__main__":
    main()
