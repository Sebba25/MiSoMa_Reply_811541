# Pydantic Schema Flow

This document maps how the Pydantic classes in `models.py` relate to each other across the pipeline.

## High-level schema flow

```text
raw profiling / agent outputs
    -> validation models
    -> bundled validation result
    -> remediation models
    -> cleaning request / cleaner code models
    -> execution + verification models
    -> final report models
    -> narrative report models
```

## Validation-side schema flow

```text
ColumnDtypeInference
    -> DatasetDtypeInference

SchemaIssue
SchemaColumnEntry
SchemaDuplicateGroup
    -> SchemaHandoff

CompletenessColumnFinding
    -> CompletenessAnalysisReport

FormatConsistencyFinding
    -> ColumnConsistencyReport
    -> ConsistencyValidationReport

AnomalyFinding
    -> AnomalyDetectionReport

CrossColumnFinding
    -> CrossColumnValidationReport

DuplicateRecordGroup
    -> DuplicateDetectionReport
```

## Bundle and planning flow

```text
SchemaHandoff
+ CompletenessAnalysisReport
+ ConsistencyValidationReport
+ AnomalyDetectionReport
+ CrossColumnValidationReport
+ DuplicateDetectionReport
    -> OrchestrationStepResult

OrchestrationStepResult
    -> RemediationPlan
    -> CleaningPipelineResult
    -> FinalPipelineReport
```

## Cleaning-side schema flow

```text
FormatConsistencyFinding + format facts + schema info
    -> ColumnCleaningRequest

ColumnCleaningRequest
    -> CleanerRepairContext
    -> cleaner generator prompt contract
    -> host-side cleaner validation contract

ColumnCleanerProgram
    -> CleanerRepairContext
    -> runtime execution
    -> GeneratedCleanerArtifact
    -> CleaningPipelineResult.generated_programs

CleanerValidationIssue
    -> CleanerRepairContext
    -> CleanerRepairDiagnosis

CleanerRepairExample
    -> CleanerRepairDiagnosis

ExampleTransformation
    -> ColumnCleanerProgram
    -> GeneratedCleanerArtifact

CellUpdate
    -> ColumnCleanerExecutionReport

ColumnCleanerExecutionReport
    -> CleaningPipelineResult.execution_reports

GeneratedCleanerArtifact
    -> CleaningReport
    -> FinalPipelineReport

CleaningReport
    -> FinalPipelineReport
    -> CleaningPipelineResult
```

## Verification and reporting flow

```text
before/after consistency findings
    -> FindingDiff
    -> ConsistencyVerificationReport

RemediationAction
    -> RemediationPlan
    -> FinalPipelineReport action buckets

FinalPipelineReport
    -> NarrativeReport

NarrativeReportSection
    -> NarrativeReport
```

## Full schema walkthrough

```text
ColumnDtypeInference
    -> DatasetDtypeInference
    -> SchemaColumnEntry
    -> SchemaHandoff

SchemaIssue + SchemaDuplicateGroup
    -> SchemaHandoff

CompletenessColumnFinding
    -> CompletenessAnalysisReport

FormatConsistencyFinding
    -> ConsistencyValidationReport
    -> ColumnCleaningRequest

AnomalyFinding
    -> AnomalyDetectionReport

CrossColumnFinding
    -> CrossColumnValidationReport

DuplicateRecordGroup
    -> DuplicateDetectionReport

SchemaHandoff
+ CompletenessAnalysisReport
+ ConsistencyValidationReport
+ AnomalyDetectionReport
+ CrossColumnValidationReport
+ DuplicateDetectionReport
    -> OrchestrationStepResult

OrchestrationStepResult
    -> RemediationPlan
    -> ColumnCleaningRequest(s)

ColumnCleaningRequest
    -> ColumnCleanerProgram
    -> CleanerRepairContext

CleanerValidationIssue
    -> CleanerRepairContext
    -> CleanerRepairDiagnosis

ColumnCleanerProgram
    -> ExampleTransformation
    -> GeneratedCleanerArtifact
    -> ColumnCleanerExecutionReport

CellUpdate
    -> ColumnCleanerExecutionReport

GeneratedCleanerArtifact
+ ColumnCleanerExecutionReport
    -> CleaningReport

before/after consistency findings
    -> FindingDiff
    -> ConsistencyVerificationReport

OrchestrationStepResult
+ RemediationPlan
+ CleaningReport
+ ConsistencyVerificationReport
    -> FinalPipelineReport

NarrativeReportSection
    -> NarrativeReport

FinalPipelineReport
    -> NarrativeReport

all major outputs together
    -> CleaningPipelineResult
```

## Mental grouping of model families

### 1. Validation schemas

- `ColumnDtypeInference`
- `DatasetDtypeInference`
- `SchemaIssue`
- `SchemaColumnEntry`
- `SchemaHandoff`
- `CompletenessColumnFinding`
- `CompletenessAnalysisReport`
- `FormatConsistencyFinding`
- `ColumnConsistencyReport`
- `ConsistencyValidationReport`
- `AnomalyFinding`
- `AnomalyDetectionReport`
- `CrossColumnFinding`
- `CrossColumnValidationReport`
- `DuplicateRecordGroup`
- `DuplicateDetectionReport`

### 2. Bundle and planning schemas

- `OrchestrationStepResult`
- `RemediationAction`
- `RemediationPlan`

### 3. Cleaner-generation schemas

- `ColumnCleaningRequest`
- `ColumnCleanerProgram`
- `CleanerValidationIssue`
- `CleanerRepairContext`
- `CleanerRepairExample`
- `CleanerRepairDiagnosis`
- `ExampleTransformation`

### 4. Execution and verification schemas

- `CellUpdate`
- `ColumnCleanerExecutionReport`
- `GeneratedCleanerArtifact`
- `CleaningReport`
- `FindingDiff`
- `ConsistencyVerificationReport`

### 5. Final output schemas

- `FinalPipelineReport`
- `NarrativeReportSection`
- `NarrativeReport`
- `CleaningPipelineResult`

## Shortest possible summary

```text
small per-column finding models
    -> stage report models
    -> bundled validation model
    -> remediation + cleaning request models
    -> generated cleaner + execution models
    -> verification diff models
    -> final report models
    -> narrative report model
    -> one top-level CleaningPipelineResult
```
