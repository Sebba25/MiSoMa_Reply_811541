# CLI Reference

All commands run from the project root.

Use `python main.py ...`.
Default dataset: `Data/spesa.csv`

## Quick Model

The CLI is organized around two main entrypoints:

- `validate`: run the analysis pipeline and save validation artifacts
- `clean`: run validation, remediation planning, cleaner generation, cleaner application, verification, and final reporting

Everything else is a focused sub-step of one of those flows.

Legacy aliases are still accepted:

- `all` -> `validate`
- `pipeline` -> `clean`

## Common Commands

Build the validation bundle:

```bash
python main.py Data/attivazioniCessazioni.csv
python main.py Data/attivazioniCessazioni.csv --stage validate
python main.py Data/attivazioniCessazioni.csv --stage validate --reuse-schema --reuse-completeness --reuse-consistency
```

Run the full cleaning flow:

```bash
python main.py Data/attivazioniCessazioni.csv --stage clean
python main.py Data/attivazioniCessazioni.csv --stage clean --reuse-validation --reuse-remediation
```

Build the remediation plan only:

```bash
python main.py Data/attivazioniCessazioni.csv --stage remediate
python main.py Data/attivazioniCessazioni.csv --stage remediate --reuse-validation
```

Run one focused step:

```bash
python main.py Data/attivazioniCessazioni.csv --stage dtype
python main.py Data/attivazioniCessazioni.csv --stage schema
python main.py Data/attivazioniCessazioni.csv --stage completeness
python main.py Data/attivazioniCessazioni.csv --stage consistency
python main.py Data/attivazioniCessazioni.csv --stage remediate
python main.py Data/attivazioniCessazioni.csv --stage generate --column "aggregation-time"
python main.py Data/attivazioniCessazioni.csv --stage apply
python main.py Data/attivazioniCessazioni.csv --stage verify
```

## Stages

### `validate`

Builds and saves:

- schema handoff
- completeness report
- consistency report
- anomaly report
- cross-column report
- duplicate report
- validation bundle

### `remediate`

Builds and saves:

- remediation plan

The remediation plan is deterministic and separates:

- safe auto-apply actions
- proposed but not auto-applied actions
- manual-review items
- duplicate-row drop candidates that are reported but never auto-applied

### `clean`

Runs:

1. validation bundle load/build
2. remediation plan load/build
3. cleaner generation
4. cleaner application plus safe remediation actions
5. post-clean verification
6. final report write

### Focused stages

- `dtype`: print inferred dtype, role, pattern, and rationale per column
- `schema`: run schema analysis only
- `completeness`: run completeness analysis only
- `consistency`: run format consistency validation only
- `remediate`: build the remediation plan from the validation bundle
- `generate`: generate cleaner modules from consistency findings
- `apply`: apply generated cleaners and safe remediation/schema/completeness post-processing
- `verify`: compare original vs cleaned consistency findings

## Flags

| Flag | Applies to | Effect |
|---|---|---|
| `--stage` | all | Selects the stage. Default: `validate` |
| `--reuse-schema` | `schema`, `validate` | Reuse schema cache |
| `--reuse-completeness` | `completeness`, `validate` | Reuse completeness cache |
| `--reuse-consistency` | `consistency`, `validate`, `generate` | Reuse consistency cache |
| `--reuse-validation` | `remediate`, `clean` | Reuse the saved validation bundle |
| `--reuse-remediation` | `remediate`, `clean` | Reuse the saved remediation plan |
| `--verbose` | all | Stream live agent/tool events to stderr |
| `--column` | `generate` | Restrict generation to one exact column name |
| `--cleaner-attempts` | `clean`, `generate` | Max generator/critic attempts per column |

## Cache And Output Paths

Validation caches are saved next to the dataset file:

- `Data/.validation_cache/<dataset>.schema_handoff.json`
- `Data/.validation_cache/<dataset>.completeness.json`
- `Data/.validation_cache/<dataset>.consistency.json`
- `Data/.validation_cache/<dataset>.anomaly.json`
- `Data/.validation_cache/<dataset>.cross_column.json`
- `Data/.validation_cache/<dataset>.duplicates.json`
- `Data/.validation_cache/<dataset>.remediation_plan.json`
- `Data/.validation_cache/<dataset>.validation_bundle.json`

Cleaning outputs are also dataset-adjacent:

- `Data/.cleaning_cache/<dataset>/cleaner_manifest.json`
- `Data/.cleaning_cache/<dataset>/generated_cleaners/*.py`
- `Data/.cleaning_cache/<dataset>/<dataset>.cleaned.csv`
- `Data/.cleaning_cache/<dataset>/<dataset>.final_report.json`
