# NOIPA: Multi-agent system for data quality

**Authors:** Michele Turco, Mattia Sebastiani, Sofia Bruni

## Section 1 — Introduction

NOIPA is a multi-agent system for validating and cleaning CSV datasets from the Italian Public Administration. The problem it addresses is not the absence of data, but its inconsistency: values may be present in the file and still be difficult to use because they are encoded in different formats, grouped under unstable column names, or mixed with placeholders that obscure the real content. In that setting, manual cleanup is slow and fragile, while a single monolithic LLM prompt is too unconstrained to be trusted.

The project therefore proposes a staged pipeline that combines large language models with deterministic validation logic. The intent is to inspect the data, diagnose quality issues, and produce repairs in a way that remains reproducible and auditable. The notebook is written as a guided walkthrough of the production code, so that the transformation from raw CSV to cleaned output can be followed step by step.

## Section 2 — Methods

The system is built as a multi-agent pipeline in which each agent has a narrow responsibility and exchanges structured data with the next stage. This design choice was made to reduce ambiguity and to keep the workflow debuggable. Schema inference, completeness analysis, format consistency checking, anomaly detection, remediation planning, cleaning generation, and verification are separated so that each stage can solve one concrete subproblem instead of trying to reason about the whole pipeline at once. That separation also makes failures easier to localise, because a defect in one stage does not blur the responsibility of the next one.

### Use of Pydantic

Pydantic is the backbone of the architecture because the entire pipeline depends on reliable machine-to-machine handoffs. In a language-model workflow, unconstrained text output is a source of variability, and variability is especially risky when the output of one agent becomes the input of another. Pydantic data contracts reduce that risk by forcing every agent to emit a predefined structure. As a result, the interface between stages becomes deterministic, parseable, and easy to validate automatically.

The notebook and the production modules use this idea consistently. Pydantic models define not only the agent outputs, but also the validation artifacts, the cleaning requests, the remediation plan, the generated programs, the verification results, and the final narrative report. The same contract layer therefore supports both the factual pipeline and the presentation layer. This is valuable in a multi-agent system because a mistake at one boundary can otherwise cascade silently into later stages.

### Token efficiency and model choice

A second design choice concerns token efficiency. At most 500 values are passed to the LLM per interaction, which keeps the prompts deliberately compact and avoids unnecessary token consumption. Instead of sending an entire column or a large sample, the pipeline sends only representative values and local evidence that are sufficient for diagnosis. This keeps the system inexpensive to run and makes repeated experimentation practical.

The token budget also makes it possible to use a stronger lightweight model, GPT-4.5 nano, without turning the pipeline into a costly workflow. Deterministic Python code handles counting, profiling, and validation, while the model is reserved for interpretation, synthesis, and structured reasoning. That division of labour is central to the design.

### Pipeline flow

The overall pipeline follows a fixed sequence. First, schema inference estimates the structure of each column and produces a baseline schema handoff. Next, validation stages inspect completeness, format consistency, anomalies, and cross-column relationships. The resulting evidence is then converted into column-level cleaning requests, which feed a generation stage that proposes candidate cleaning functions. Those functions are checked by a deterministic validator before being accepted, and a critic agent is used when repairs are required. Finally, the accepted transformations are applied to the dataset and the outcome is reported through a verification and reporting layer.

The environment and dependency list are not reproduced in full here; the complete set of packages required to run the project is available in [requirements.txt](requirements.txt).

## Section 3 — Experimental Design

The development process relied on several iterative runs of the pipeline, primarily to fine-tune prompts and guardrails so that agent outputs became as deterministic as possible without losing the flexibility needed to handle heterogeneous public datasets. The experimental approach was therefore empirical rather than comparative: the main goal was to observe where the pipeline failed, understand why it failed, and refine the prompts, contracts, and validation rules accordingly.

No formal external baselines were used. Instead, the practical reference point was the behaviour of the same system under earlier prompt and guardrail configurations. Each agent call was monitored through Logfire, including its input, output, and reasoning trace, which made it possible to identify precisely which instruction or constraint had been ineffective in a given case. This monitoring process served as the main evaluation method for the project, because it allowed rapid iteration on failure modes such as inconsistent formatting, over-permissive transformations, and ambiguous repairs.

The main purpose of these experiments was to validate that the pipeline could remain stable across different datasets and different classes of data quality issues. The absence of formal baselines does not reduce the value of the evaluation in this setting, because the project objective was operational robustness rather than benchmark optimisation.

## Section 4 — Results

The pipeline produces a layered account of data quality rather than a single yes-or-no verdict. At the schema level, it identifies naming issues and dtype-related inconsistencies; at the completeness level, it distinguishes true missing values from placeholder-like tokens; at the consistency level, it isolates values that follow a different structural pattern from the dominant one; and at the remediation stage, it turns those findings into concrete cleaning actions that can be checked before they are applied. The practical result is a workflow that does not merely classify the dataset as “clean” or “dirty”, but explains which kinds of problems are present and how they are handled.

The most important operational result is that the generated cleaners are never trusted blindly. Each candidate function is validated against representative dominant values and outlier examples before being accepted, which keeps the system from silently introducing new errors while attempting to fix existing ones. The final verification stage then compares the pre-cleaning and post-cleaning consistency evidence, so the effect of the cleaning step remains visible in the report rather than being assumed.

The notebook also shows that the token-efficiency strategy is practical. Limiting each interaction to a compact set of representative values keeps the pipeline lightweight enough to support repeated experimentation, while still supplying enough evidence for the model to make meaningful structured decisions. The main result is therefore architectural as much as analytical: the pipeline demonstrates that a constrained LLM can be integrated into a deterministic data-quality workflow without sacrificing traceability.

The figures and tables for this section are generated from the code and should be inserted once the corresponding outputs are final.

**Figure 1.** [FIGURE X — pipeline overview generated from the code]

**Table 1.** [INSERT RESULT TABLE — summary of validation, cleaning, and verification outcomes]

**Placeholders for final reporting**

- [INSERT RESULT]: number of schema issues identified.
- [INSERT RESULT]: number of inconsistent columns repaired or improved.
- [INSERT RESULT]: verification outcome after the cleaning stage.

## Section 5 — Conclusions

The project shows that a multi-agent cleaning pipeline can be made reliable for public-administration CSV datasets when every LLM interaction is bounded by explicit Pydantic contracts and every generated transformation is checked by deterministic code. In that configuration, the model is used for interpretation and synthesis, while the host environment remains responsible for structure, validation, and correctness. That balance is what makes the workflow both practical and explainable.

Several questions remain open. It is still necessary to understand how the same architecture behaves on datasets with stronger domain drift, denser corruption, or more ambiguous value conventions. A natural next step would be to test the pipeline on additional public-administration datasets and to measure more systematically which stages fail, under what conditions, and with what kind of prompt changes. Another useful extension would be to refine the critic-and-repair loop so that difficult cleaning cases can be resolved with fewer retries while preserving the same deterministic safeguards.

### Concepts to specify

- The thought process: we started trying to implement langraph but pydantic revealed itself as a more practical way to enforce structure and traceability in the pipeline.
- Initially all the data was passe to the LLM: impractical and expensive, so we switched to a token-efficient strategy that sends only representative values and local evidence.
- All the process of how validation works: how many values are sent, how they are selected, what kind of evidence is generated, how the cleaning functions are validated. Especially, the thought process behind the design of the validation and cleaning stages, which is the core of the project.
- The cycle in the critic-and-repair loop: how the critic identifies the failure, how the repair is generated, and how it is checked before being applied.
- The verification stage: how the pre-cleaning and post-cleaning evidence is compared, and how the final report is generated.
- How the agents interact with each other, the cache system, why it is bettern than the agents talking to each other directly, how it supports traceability and debugging.
- The role of Logfire in monitoring the pipeline and supporting iterative refinement of prompts and guardrails.
- Every step of the pipeline we have to know, why it exists, what it does, how it does it, what it produces, why does it do it, why does it do in in a certain specific way, how it interacts with the next stage, what kind of issues it is designed to catch, and how it contributes to the overall goal of improving data quality.
- Why 100 values, are they sampled randomly, or are they selected based on some criteria? How does that selection process work, and how does it ensure that the values sent to the LLM are representative of the column's overall quality?
- 60 outliers: how are they defined, how are they selected, and what role do they play in the validation and cleaning process? Are they used to test the robustness of the generated cleaning functions, or do they serve a different purpose?
- validation of the generated cleaning functions: how does the deterministic validator work, what criteria does it use to accept or reject a proposed function, and how does it ensure that the cleaning actions are effective without introducing new errors?
- are there guardrails in the generation stage to prevent the model from proposing overly complex or risky transformations? If so, how are those guardrails designed and implemented?
- we are allowing an LLM to execute code, which is a potential risk. How do we mitigate that risk, especially in a public-administration context where data sensitivity and security are concerns? Are there specific constraints on the types of code that can be generated, or on the execution environment?
- how does the system handle cases where the model's proposed cleaning function fails validation? Is there a feedback loop to the critic agent, and how does that loop work in practice? Does the system allow for multiple iterations of critique and repair, and if so, how does it manage that process to avoid infinite loops or excessive retries?
- why is the final report generated in a narrative format, and how does that format contribute to the explainability and usability of the results? Does the report include specific sections for each stage of the pipeline, and how are the findings from each stage presented to the user? Is there a standard template for the report, and how does it ensure that the information is clear and actionable for stakeholders who may not be familiar with the technical details of the pipeline?