# CLI Reference

All commands run from the project root. Default dataset is `Data/spesa.csv`.

---

## Stages

### Dtype Inference
Infers pandas dtype, role, pattern, and rationale per column. Prints a formatted table.
```bash
python test.py Data/attivazioniCessazioni.csv --stage dtype
```

### Schema Validation
Naming convention checks, duplicate detection, dtype inference. Auto-saves to `.validation_cache/`.
```bash
python test.py Data/attivazioniCessazioni.csv --stage schema
python test.py Data/spesa.csv --stage schema
python test.py Data/attivazioniCessazioni.csv --stage schema --reuse-schema
```

### Completeness Analysis
Missing values, placeholder detection, sparse columns. Auto-saves to `.validation_cache/`.
```bash
python test.py Data/attivazioniCessazioni.csv --stage completeness
python test.py Data/attivazioniCessazioni.csv --stage completeness --reuse-completeness
```

### Consistency Validation
Format inconsistencies per column. Auto-saves to `.validation_cache/`.
```bash
python test.py Data/attivazioniCessazioni.csv --stage consistency
python test.py Data/attivazioniCessazioni.csv --stage consistency --reuse-consistency
python test.py Data/attivazioniCessazioni.csv --stage consistency --consistency-agent format
python test.py Data/spesa.csv --stage consistency --consistency-agent format
```

> **Schema handoff integration (automatic):** when `--consistency-agent format` is used,
> the format consistency stage automatically reads the cached schema handoff (if present) to skip
> `name`/`free_text` columns and pass `detected_pattern` as the authoritative expected format.
> Run `--stage schema` first to populate the cache, then run consistency — no extra flag needed:
> ```bash
> python test.py Data/attivazioniCessazioni.csv --stage schema
> python test.py Data/attivazioniCessazioni.csv --stage consistency --consistency-agent format
> ```

### Cleaning
Generates and applies Python cleaning functions for format issues found in consistency validation.
```bash
python test.py Data/attivazioniCessazioni.csv --stage clean
python test.py Data/attivazioniCessazioni.csv --stage clean --reuse-validation
python test.py Data/attivazioniCessazioni.csv --stage generate --verbose
python test.py Data/spesa.csv --stage generate --verbose --column "aggregation-time"
```

### Full Pipeline
Runs schema → completeness → consistency → saves bundle. Any stage can be reused from cache.
```bash
python test.py Data/attivazioniCessazioni.csv
python test.py Data/attivazioniCessazioni.csv --reuse-schema
python test.py Data/attivazioniCessazioni.csv --reuse-schema --reuse-completeness
python test.py Data/attivazioniCessazioni.csv --reuse-schema --reuse-completeness --reuse-consistency
```

---

## Flags

| Flag | Applies to | Effect |
|---|---|---|
| `--stage` | all | Which stage to run: `dtype`, `schema`, `completeness`, `consistency`, `clean`, `all` |
| `--reuse-schema` | `schema`, `all` | Load schema handoff from cache, skip LLM calls |
| `--reuse-completeness` | `completeness`, `all` | Load completeness result from cache, skip LLM calls |
| `--reuse-consistency` | `consistency`, `all` | Load consistency result from cache, skip LLM calls |
| `--reuse-validation` | `clean` | Load full validation bundle from cache before cleaning |
| `--consistency-agent` | `consistency` | Sub-agent to use: `all` (default) or `format` |
| `--verbose` | all | Stream live agent text/thinking/tool events to stderr while the run is in progress |
| `--column` | `generate` | Restrict cleaner generation to one exact column name, e.g. `--column "aggregation-time"` |

---

## Cache Files

All saved under `.validation_cache/<dataset_stem>/`:

| File | Written by |
|---|---|
| `<stem>.schema_handoff.json` | `--stage schema` |
| `<stem>.completeness.json` | `--stage completeness` |
| `<stem>.consistency.json` | `--stage consistency` |
| `<stem>.validation_bundle.json` | `--stage all` |

Cleaned output is saved under `.cleaning_cache/<dataset_stem>/`.
