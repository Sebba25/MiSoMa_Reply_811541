# Spiegazione Completa del Codice — AgentsAI

Questo documento raccoglie tutte le spiegazioni del codice, file per file, funzione per funzione.

---

# 1. validation/schema.py

Ecco una spiegazione dettagliata di [validation/schema.py](validation/schema.py), funzione per funzione.

---

## Docstring del modulo (righe 1–13)

Descrive lo scopo del file: fa **due chiamate LLM** per ogni esecuzione:

1. `dtype_inference_agent` — inferisce il tipo pandas, il ruolo semantico e il pattern rilevato per ogni colonna
2. `schema_summary_agent` — produce un riassunto leggibile dello `SchemaHandoff` già costruito (non rideriva nulla)

Oltre alle chiamate LLM, il modulo: costruisce il profilo statistico del dataset, rileva colonne duplicate via collisioni di nomi canonici, e produce `SchemaIssue` per violazioni di naming e colonne duplicate.

---

## Imports (righe 15–40)

```python
from agents import dtype_inference_agent, schema_summary_agent
```
Importa i due agenti Pydantic AI definiti in `agents.py`.

```python
from cache import load_schema_handoff, save_schema_handoff
```
Gestione della cache su disco (leggi/scrivi il risultato in JSON).

```python
from models import (DatasetDtypeInference, SchemaColumnEntry, SchemaHandoff, SchemaIssue)
```
I modelli Pydantic che definiscono la struttura dell'output (cosa restituisce ogni agente e cosa si passa tra le fasi).

```python
from tools import (...)
```
Funzioni di utilità: caricare il DataFrame, costruire testi da passare agli agenti, validare nomi di colonna, ecc.

---

## Funzione `build_schema_issues()` (righe 43–84)

**Input:** lista di `SchemaColumnEntry` (colonne già analizzate) + lista di `SchemaDuplicateGroup` (gruppi di colonne duplicate)

**Output:** lista di `SchemaIssue` — problemi trovati, con severità, evidenza e fix suggerito

**Non usa agenti** — è pura logica Python.

Fa **due passaggi**:

### Passaggio 1 — violazioni di naming (righe 49–61)
```python
for col in columns:
    if not col.naming_valid:
        issues.append(SchemaIssue(
            issue_type="naming_standard",
            severity="high",
            ...
        ))
```
Per ogni colonna che non rispetta `snake_case` minuscolo, crea un `SchemaIssue` con:
- `severity="high"` — il naming è un problema bloccante
- `suggested_fix` — il nome corretto suggerito (es. `"CustomerID"` → `"customer_id"`)
- `fix_confidence="high"` — la rinomina è meccanica, non ambigua

### Passaggio 2 — colonne semanticamente duplicate (righe 63–83)
```python
for group in duplicate_groups:
    for column_name in group.columns:
        peer_columns = [peer for peer in group.columns if peer != column_name]
        issues.append(SchemaIssue(
            issue_type="duplicate_column_semantics",
            severity="medium",
            ...
        ))
```
Se due colonne hanno lo stesso **nome canonico** (es. `"data_nascita"` e `"DataNascita"` → entrambe diventano `"data_nascita"`), sono probabilmente duplicate. Per ognuna crea un issue che dice "potresti avere overlap con le colonne peer X, Y".
- `severity="medium"` — richiede confronto manuale, non è certezza
- `fix_confidence="medium"` — non si può risolvere automaticamente senza guardare i valori

---

## Funzione `run_dtype_inference()` (righe 87–96)

**Input:** `path: Path` — percorso al file CSV

**Output:** `DatasetDtypeInference` — oggetto con un'entry per ogni colonna: dtype pandas suggerito, ruolo numerico/stringa, pattern rilevato, rationale

**Usa l'agente `dtype_inference_agent`.**

```python
df = load_dataset_frame(path)           # carica il DataFrame pandas
text = build_dtype_inference_text(df)   # costruisce un testo con campioni di valori per ogni colonna
prompt = [
    "Infer the correct pandas dtype...",
    attach_text_document(text),          # allega il testo come BinaryContent text/plain
]
result = run_agent_with_backoff(dtype_inference_agent, prompt)
return result.output                     # DatasetDtypeInference
```

Il testo allegato all'agente contiene campioni di valori reali per ogni colonna. L'agente riceve questo testo e risponde con un JSON strutturato (`DatasetDtypeInference`) che specifica per ogni colonna quale sia il tipo pandas più corretto.

`run_agent_with_backoff` gestisce i retry in caso di rate-limit (backoff esponenziale).

---

## Funzione `run_schema_validation()` (righe 99–165)

**Input:** `path: Path`, `reuse_cache: bool`

**Output:** `SchemaHandoff` — oggetto completo con profilo delle colonne, issues, gruppi duplicati e summary testuale

**Usa due agenti:** `dtype_inference_agent` (tramite `run_dtype_inference`) e `schema_summary_agent`.

### Fase 0 — cache hit (righe 100–101)
```python
if reuse_cache:
    return load_schema_handoff(path)
```
Se `--reuse-schema` è passato da CLI, carica il risultato precedente dal disco senza chiamare alcun agente.

### Fase 1 — dtype inference (righe 103–107)
```python
df = load_dataset_frame(path)
dtype_inference = run_dtype_inference(path)   # prima chiamata LLM
dtype_map = {col.column_name: col for col in dtype_inference.columns}
dtype_overrides = {name: col.pandas_dtype for name, col in dtype_map.items()}
```
Chiama il primo agente e costruisce un dizionario `colonna → dtype` per uso successivo.

### Fase 2 — profilo statistico (riga 109)
```python
profile = build_dataset_profile(df, path.stem, dtype_overrides=dtype_overrides)
```
Funzione Python (no LLM): calcola statistiche per ogni colonna — valori non nulli, valori distinti, percentuale di parse numerica/datetime, sample values, ecc. Usa i dtype inferiti dall'agente per fare parsing corretto.

### Fase 3 — rilevamento colonne duplicate (righe 111–119)
```python
duplicate_groups_by_name: dict[str, list[str]] = {}
for col_name in df.columns:
    canonical = normalized_schema_name(col_name)   # es. "DataNascita" → "data_nascita"
    duplicate_groups_by_name.setdefault(canonical, []).append(col_name)
duplicate_groups = [
    SchemaDuplicateGroup(canonical_name=cn, columns=cols)
    for cn, cols in duplicate_groups_by_name.items()
    if len(cols) > 1   # solo se ci sono almeno due colonne con lo stesso nome canonico
]
```
Logica Python pura: raggruppa colonne per nome normalizzato e tiene solo i gruppi con più di una colonna.

### Fase 4 — costruzione delle `SchemaColumnEntry` (righe 121–142)
```python
for col_profile in profile.columns_profiles:
    ...
    columns.append(SchemaColumnEntry(
        name=name,
        pandas_dtype=col_profile.pandas_dtype,
        numeric_role=dtype_col.numeric_role,    # dall'agente LLM
        string_role=dtype_col.string_role,       # dall'agente LLM
        detected_pattern=dtype_col.detected_pattern,  # dall'agente LLM
        non_null_rows=col_profile.non_null_rows, # dal profilo statistico
        naming_valid=is_valid_schema_name(name), # validazione naming
        rename_suggestion=suggest_schema_name(name),  # fix suggerito
        ...
    ))
```
Fonde: output dell'agente LLM (ruolo semantico, pattern) + statistiche Python (null count, sample values) + validazione naming.

### Fase 5 — `build_schema_issues()` (riga 144)
```python
issues = build_schema_issues(columns, duplicate_groups)
```
Chiama la funzione descritta sopra per produrre la lista di problemi.

### Fase 6 — costruzione `SchemaHandoff` (righe 146–153)
```python
handoff = SchemaHandoff(
    dataset_name=path.stem,
    total_rows=len(df),
    total_columns=len(df.columns),
    columns=columns,
    issues=issues,
    duplicate_groups=duplicate_groups,
)
```
Assembla l'oggetto finale (senza ancora il `summary` testuale).

### Fase 7 — summary agent (righe 154–162)
```python
result = run_agent_with_backoff(schema_summary_agent, [
    "Summarize the provided schema analysis for dataset ...",
    attach_profile_text(handoff),   # serializza handoff come testo allegato
])
handoff = handoff.model_copy(update={"summary": result.output.summary})
```
Il secondo agente riceve lo `SchemaHandoff` già costruito e produce **solo un testo riassuntivo** leggibile — non rideriva nulla. Il risultato viene usato per aggiornare il campo `summary` dell'handoff. Questo summary sarà letto dagli agenti delle fasi successive (completeness, consistency) come contesto.

### Fase 8 — salvataggio cache (riga 164)
```python
save_schema_handoff(path, handoff)
```
Salva il risultato in `Data/.validation_cache/<dataset>.schema.json` per uso futuro con `--reuse-schema`.

---

## Schema del flusso complessivo

```
CSV file
   │
   ├─► load_dataset_frame() ──────────────────────────────┐
   │                                                       │
   ├─► [LLM 1] dtype_inference_agent ─► dtype_map         │
   │                                        │              │
   ├─► build_dataset_profile(dtype_map) ─► profile        │
   │                                                       │
   ├─► normalized_schema_name() ─► duplicate_groups       │
   │                                                       │
   ├─► merge(dtype_map + profile + naming) ─► columns ────┘
   │
   ├─► build_schema_issues(columns, duplicate_groups) ─► issues
   │
   ├─► SchemaHandoff(columns, issues, duplicate_groups)
   │
   ├─► [LLM 2] schema_summary_agent ─► summary text
   │
   └─► SchemaHandoff completo → cache JSON
```

---

# 2. validation/completeness.py

Il file è molto più compatto di `schema.py` — una sola funzione pubblica, una sola chiamata LLM. Ecco la spiegazione completa.

---

## Docstring del modulo (righe 1–7)

Descrive lo scopo: **una sola chiamata LLM** tramite `completeness_analysis_agent`.

L'agente riceve un documento `CompletenessProfile` (costruito da Python) e restituisce un `CompletenessAnalysisReport` con:
- percentuale di valori mancanti per colonna
- token placeholder rilevati (es. `"N/A"`, `"NULL"`, `"-"`)
- flag per colonne sparse (candidate a rimozione o indagine)

---

## Imports (righe 9–22)

```python
from agents import completeness_analysis_agent
```
L'agente Pydantic AI definito in `agents.py` — l'unico usato in questo modulo.

```python
from cache import load_completeness, save_completeness
```
Lettura/scrittura del risultato in JSON su disco (cache).

```python
from models import CompletenessAnalysisReport
```
Il modello Pydantic che struttura l'output dell'agente.

```python
from tools import (
    attach_profile_text,
    build_completeness_profile,
    load_dataset_frame,
    run_agent_with_backoff,
)
```
- `load_dataset_frame` — carica il CSV come DataFrame pandas
- `build_completeness_profile` — costruisce il profilo di completezza (logica Python, no LLM)
- `attach_profile_text` — serializza il profilo come allegato `text/plain` per l'agente
- `run_agent_with_backoff` — chiama l'agente con retry su rate-limit

---

## Funzione `run_completeness_analysis()` (righe 25–44)

**Input:** `path: Path` (percorso al CSV), `reuse_cache: bool` (default `False`)

**Output:** `CompletenessAnalysisReport` — report strutturato per colonna

**Usa l'agente `completeness_analysis_agent`** — una sola chiamata LLM.

---

### Riga 26–27 — cache hit
```python
if reuse_cache:
    return load_completeness(path)
```
Se `--reuse-completeness` è passato da CLI, carica il risultato precedente da `Data/.validation_cache/<dataset>.completeness.json` e ritorna subito, senza chiamare l'agente.

---

### Riga 28 — caricamento dati
```python
df = load_dataset_frame(path)
```
Carica il CSV come DataFrame pandas. Non fa ancora nessuna analisi — è solo il raw data.

---

### Riga 29 — costruzione del profilo (no LLM)
```python
profile = build_completeness_profile(df, path.stem)
```
Funzione Python pura (in `tools/completeness_tools.py`). Calcola per ogni colonna:
- quanti valori sono `NaN` / `None`
- quanti valori sembrano placeholder (`"N/A"`, `"NULL"`, `"-"`, `"0"`, stringa vuota, ecc.)
- il tasso complessivo di "missing-like" (null + placeholder)
- i token placeholder effettivamente presenti

Questo lavoro è fatto **senza LLM** perché è deterministico: contare null e matchare stringhe note è compito di Python, non di un modello.

---

### Righe 30–39 — costruzione del prompt
```python
prompt = [
    (
        f"Analyze the attached completeness profile for dataset {path.stem}. "
        "Use Python in code execution to inspect the profile document. "
        "This is step 2 of the orchestration only: Completeness Analysis. "
        "Use the provided metrics to summarize per-column completeness, detect missing-like and placeholder values, "
        "identify actual placeholder tokens present in the dataset, and flag sparse columns..."
    ),
    attach_profile_text(profile),
]
```
Il prompt è una lista di due elementi:
1. **Istruzione testuale** — dice all'agente cosa fare (step 2 dell'orchestration, analizza il profilo allegato)
2. **Allegato** — il profilo costruito da Python, serializzato come `BinaryContent` con media type `text/plain`

La nota `"Use Python in code execution"` suggerisce all'agente di usare il tool di esecuzione codice (se disponibile nel runner) per ispezionare il documento — utile se il profilo è lungo e strutturato.

---

### Riga 40 — log a stderr
```python
print(f"[orchestrator][completeness] dataset='{path.stem}'", file=sys.stderr, flush=True)
```
Questa riga serve solo per **stampare un messaggio di controllo** mentre il programma sta girando, così puoi capire che l’orchestratore è arrivato alla fase di **completeness** e su quale dataset sta lavorando. Il testo viene mandato su **stderr** invece che su **stdout** perché stdout di solito viene usato per l’output “ufficiale” della pipeline, ad esempio risultati o file strutturati, mentre stderr è più adatto per log, messaggi di debug o informazioni di tracciamento. `path.stem` prende il nome del file senza estensione, quindi mostra solo il nome del dataset. Infine, `flush=True` forza Python a mostrare subito il messaggio, senza aspettare che il buffer (Un buffer è una specie di “area di attesa” temporanea dove il computer conserva dei dati prima di mostrarli o scriverli davvero.) venga svuotato più tardi.


---

### Riga 41 — chiamata all'agente (unica chiamata LLM)
```python
result = run_agent_with_backoff(completeness_analysis_agent, prompt)
```
Chiama l'agente con retry esponenziale in caso di `429 Rate Limit`. L'agente:
1. Legge il profilo allegato
2. Interpreta le metriche (missing %, placeholder tokens, sparse flags)
3. Restituisce un `CompletenessAnalysisReport` strutturato

---

### Righe 42–43 — estrazione e salvataggio
```python
report = result.output
save_completeness(path, report)
```
`result.output` è già un `CompletenessAnalysisReport` validato da Pydantic (grazie all'output type dichiarato nell'agente in `agents.py`). Viene salvato in cache JSON per uso futuro.

---

### Riga 44 — return
```python
return report
```
Il report viene restituito al chiamante (`validation/bundle.py`) che lo include nel `ValidationResults` complessivo.

---

## Schema del flusso

```
CSV file
   │
   ├─► load_dataset_frame() ─────────────► DataFrame
   │
   ├─► build_completeness_profile() ─────► CompletenessProfile
   │         (Python puro, no LLM)                │
   │                                               │
   │                                    attach_profile_text()
   │                                               │
   └─► [LLM] completeness_analysis_agent ◄─────────┘
                      │
                      ▼
         CompletenessAnalysisReport
                      │
              save_completeness()
                      │
                   return
```

---

## Confronto con `schema.py`

| Aspetto | `schema.py` | `completeness.py` |
|---|---|---|
| Chiamate LLM | 2 | 1 |
| Complessità | Alta (profilo + dtype + naming + duplicate detection) | Bassa (profilo → agente → report) |
| Preprocessing Python | Molto (profilo statistico, canonical names, issue building) | Poco (solo `build_completeness_profile`) |
| Cache | `SchemaHandoff` | `CompletenessAnalysisReport` |

`completeness.py` è il modulo più lineare del pipeline: costruisci il profilo con Python, lo passi all'agente, salvi il risultato. Tutta la logica interpretativa (capire se una colonna è davvero sparse, quali placeholder sono significativi) è delegata al LLM.

---

# 3. validation/consistency.py

Questo è il modulo più complesso dei tre — introduce due path distinti (fast/slow), esecuzione asincrona parallela, e una funzione helper per costruire il prompt del cleaner agent. Ecco la spiegazione completa.

---

## Docstring del modulo (righe 1–11)

Due path di esecuzione per ogni colonna:
- **Fast path** — guidato dallo schema, **nessuna chiamata LLM**: prende il pattern già inferito da `dtype_inference_agent` e conta direttamente gli outlier
- **Slow path** — `format_consistency_agent` inferisce il pattern dominante quando lo schema non ce l'ha

`_build_suggested_strategy` costruisce la stringa di strategia per il **cleaner agent** della fase successiva, enumerando ogni gruppo di outlier con esempi concreti.
Questa funzione costruisce una **strategia testuale** da dare al cleaner-agent, cioè un insieme di istruzioni su come sistemare i valori incoerenti di una colonna. Prima raggruppa gli esempi “sbagliati” in base alla loro forma (`shape`), ignorando quelli che hanno già la forma dominante valida. Se non trova errori, dice semplicemente di normalizzare tutto nel formato atteso e mettere `null` solo quando non si può convertire. Se invece ci sono valori incoerenti, crea un testo con il formato target, la forma valida dominante, alcuni esempi corretti e poi spiega come trattare ogni gruppo di valori anomali. La funzione distingue anche il caso in cui i numeri possano avere lunghezze diverse, per evitare conversioni troppo rigide. Alla fine aggiunge una regola importante: ogni valore incoerente deve essere gestito esplicitamente e bisogna preferire una conversione ragionevole invece di mettere `null`, quando il valore contiene informazioni recuperabili.


---

## Imports (righe 13–41)

```python
import asyncio
from collections import Counter, defaultdict
```
`asyncio` per il parallelismo colonna-per-colonna. `Counter` e `defaultdict` per raggruppare gli outlier per shape.

```python
from agents import format_consistency_agent
```
L'unico agente usato — solo nel slow path.

```python
from tools import (
    PLACEHOLDER_TOKENS,          # set di stringhe note come placeholder (N/A, NULL, -, ...)
    FormatOutlierExample,        # dataclass: value, shape, count
    build_column_format_facts,   # profila i formati di una colonna
    matches_numeric_schema_pattern,  # controlla se un valore rispetta il pattern numerico
    numeric_pattern_allows_variable_width,  # es. "year" accetta 2020 e 20 indifferentemente?
    run_agent_with_backoff,      # sync, con retry
    run_agent_with_backoff_async,  # async, con retry
    value_shape,                 # es. "12/03/2024" → "NN/NN/NNNN"
)
```

---

## Funzione `_build_suggested_strategy()` (righe 44–107)

**Input:**
- `expected_pattern` — il pattern atteso (es. `"YYYY-MM-DD"`)
- `dominant_shape` — la shape dominante della colonna (es. `"NNNN-NN-NN"`)
- `inconsistent_examples` — lista di `FormatOutlierExample` (valori non conformi)
- `dominant_example_values` — campioni di valori già validi
- `allow_variable_numeric_width` — se il pattern numerico ammette lunghezze diverse

**Output:** stringa di testo — la `suggested_strategy` che finirà nel `FormatConsistencyFinding` e sarà letta dal **cleaner agent** nella fase di cleaning.

**Non usa agenti** — è costruzione di testo puro.

### Logica interna

```python
groups: dict[str, list[str]] = defaultdict(list)
for ex in inconsistent_examples:
    if ex.shape != dominant_shape:
        groups[ex.shape].append(ex.value)
```
Raggruppa gli outlier per **shape** (es. tutti i valori con shape `"NN-NN-NNNN"` vanno insieme). Questo permette al cleaner agent di gestire ogni gruppo con una regola specifica.

Se non ci sono gruppi (nessun outlier con shape diversa dal dominante):
```python
return "Normalize all values to '{expected_pattern}'. Map to null when the value cannot be converted."
```
Strategia generica — basta normalizzare.

Se ci sono gruppi, costruisce un testo strutturato:
1. Pattern target e shape dominante (già valida, da preservare)
2. Esempi di valori già validi (il cleaner deve produrre output identico a questi)
3. Istruzioni specifiche per numeric vs. fixed-width:
   - `allow_variable_numeric_width=True` → non forzare stessa lunghezza, accetta qualsiasi numero parsabile. Caso speciale: mese 1-12, fuori range → null
   - `allow_variable_numeric_width=False` → stessa lunghezza, stesso ordine di campi, stessa struttura
4. Per ogni gruppo di outlier (ordinati per frequenza decrescente):
   ```
   shape 'NN/NN/NNNN': e.g. '03/12/2024', '14/01/2023'
   ```
5. Nota finale: ogni valore negli esempi inconsistenti deve essere gestito esplicitamente

Questo testo viene poi letto dal cleaner agent per scrivere una funzione Python che trasformi ogni outlier nel formato corretto.

---

## Funzione `_profile_schema_guided_inconsistencies()` (righe 110–147)

**Input:** `df`, `column_name`, `schema_entry: SchemaColumnEntry`

**Output:** `tuple[int, list[FormatOutlierExample]] | None`
- `None` se la colonna non è numerica (questo fast path funziona solo per `Int64` / `Float64`)
- `(count, examples)` — quanti valori non rispettano il pattern numerico e quali sono

**Non usa agenti** — validazione Python pura.

```python
if schema_entry.pandas_dtype not in {"Int64", "Float64"}:
    return None
```
Solo per colonne numeriche — le date, le stringhe ecc. vanno nel slow path dell'agente.

```python
rendered = df[column_name].dropna().astype(str).str.strip()
rendered = rendered[rendered != ""]
rendered = rendered[~rendered.str.lower().isin(PLACEHOLDER_TOKENS)]
```
Prepara i valori: rimuove null, stringa vuota, e placeholder noti (non sono errori, sono mancanze).

```python
invalid_values = [
    value for value in rendered
    if not matches_numeric_schema_pattern(value, pandas_dtype=..., numeric_role=..., detected_pattern=...)
]
```
Per ogni valore rimasto, controlla se rispetta il pattern numerico. Quelli che non lo rispettano sono gli outlier.

```python
counts = Counter(invalid_values)
examples = [
    FormatOutlierExample(value=value[:80], shape=value_shape(value), count=count)
    for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:60]
]
return len(invalid_values), examples
```
Conta le occorrenze, ordina per frequenza (più frequenti prima), prende al massimo 60 esempi (tronca i valori a 80 char).

---

## Funzione `run_column_format_check()` (righe 150–268) — versione sincrona

**Input:** `df`, `column_name`, `dataset_name`, `context_label`, `schema_entry`

**Output:** `ColumnConsistencyReport` — finding (se c'è un problema) + summary testuale

**Usa l'agente** solo nel slow path.

### Step 1 — skip per ruolo (righe 157–165)
```python
if schema_entry is not None and schema_entry.string_role in ("name", "free_text"):
    return ColumnConsistencyReport(finding=None, summary="Column skipped...")
```
Nomi propri e testo libero non hanno un formato macchina atteso — skip immediato senza costruire il profilo.

### Step 2 — profilo dei formati (righe 167–175)
```python
format_facts = build_column_format_facts(df, column_name)
if not format_facts.machine_format_candidate or format_facts.inconsistent_rows <= 0:
    return ColumnConsistencyReport(finding=None, summary="No actionable inconsistency...")
```
`build_column_format_facts` analizza i formati dei valori della colonna (via `value_shape`). Se la colonna non è candidata a un formato macchina (es. testo narrativo) oppure non ha righe inconsistenti, skip.

### Step 3 — FAST PATH (righe 177–234): schema ha già il pattern
```python
if schema_entry is not None and schema_entry.detected_pattern:
```
Se `dtype_inference_agent` aveva già identificato il pattern nella fase schema:

1. Prende le inconsistenze dal profilo del formato
2. **Prova il guided profiling** (`_profile_schema_guided_inconsistencies`) — se la colonna è `Int64/Float64`, ri-conta gli outlier con la validazione numerica precisa
3. Costruisce l'`evidence` (spiegazione testuale di cosa è stato trovato)
4. Costruisce il `FormatConsistencyFinding` con `_build_suggested_strategy`
5. **Ritorna senza chiamare l'agente** — tutto fatto in Python

Il summary include la nota `"(no LLM call)"` per tracciabilità.

### Step 4 — SLOW PATH (righe 236–268): nessun pattern dallo schema
```python
prompt = [
    f"Analyze the attached ColumnFormatFacts for dataset '{dataset_name}', column '{column_name}'."
    f" Dominant shape: '{format_facts.dominant_shape}' ({format_facts.dominant_shape_pct:.1f}%)."
    f" Inconsistent rows: {format_facts.inconsistent_rows}."
    " Determine whether a format inconsistency exists that a cleaning function could fix...",
    attach_profile_text(format_facts),
]
result = run_agent_with_backoff(format_consistency_agent, prompt)
```
L'agente riceve i `ColumnFormatFacts` (shape dominante, esempi, percentuali) e decide autonomamente se esiste una vera inconsistenza di formato e quale sia il pattern atteso.

```python
if output.finding is not None and output.finding.inconsistent_rows <= 0:
    return ColumnConsistencyReport(finding=None, summary=output.summary)
return output
```
Sanity check: se l'agente dice che ci sono inconsistenze ma `inconsistent_rows <= 0`, ignora il finding (probabilmente falso positivo dell'agente).

---

## Funzione `run_column_format_check_async()` (righe 271–388)

**Identica** a `run_column_format_check()` ma `async`. Le differenze sono solo due righe:
```python
# riga 381 — async slow path
result = await run_agent_with_backoff_async(format_consistency_agent, prompt)
```
Tutto il resto del codice è duplicato. Esiste per permettere il parallelismo con `asyncio`.

---

## Funzione `_run_column_format_checks_async()` (righe 391–416) — orchestratore async

**Input:** `df`, `column_names`, `dataset_name`, `schema_map`, `max_workers`

**Output:** `list[ColumnConsistencyReport]` nell'ordine originale delle colonne

**Non usa agenti direttamente** — chiama `run_column_format_check_async`.

```python
semaphore = asyncio.Semaphore(max_workers)
```
Limita le chiamate LLM concorrenti — es. con `max_workers=4`, al massimo 4 colonne vengono processate in parallelo.

```python
async def _run_one(index: int, column_name: str) -> tuple[int, ColumnConsistencyReport]:
    async with semaphore:
        report = await run_column_format_check_async(...)
        return index, report
```
Ogni task porta con sé il proprio `index` perché `asyncio.as_completed` restituisce i risultati fuori ordine.

```python
tasks = [asyncio.create_task(_run_one(index, column_name)) for ...]
reports_by_index: dict[int, ColumnConsistencyReport] = {}
for task in asyncio.as_completed(tasks):
    index, report = await task
    reports_by_index[index] = report
return [reports_by_index[index] for index in range(len(column_names))]
```
Avvia tutti i task insieme, raccoglie i risultati man mano che arrivano (non in ordine), poi riordina per index prima di restituire.

---

## Funzione `run_format_consistency_validation()` (righe 419–472) — entry point pubblico

**Input:** `path`, `reuse_cache`, `read_as_str`, `max_workers`

**Output:** `ConsistencyValidationReport` — lista di tutti i `FormatConsistencyFinding` trovati + summary

**Usa agenti** indirettamente (tramite le funzioni sopra).

```python
if reuse_cache:
    return load_consistency(path)
```
Cache hit — ritorna subito.

```python
df = load_dataset_frame(path, dtype=str if read_as_str else None)
```
`read_as_str=True` carica tutto come stringa — utile quando non vuoi che pandas faccia parsing automatico dei tipi (es. `"001"` non diventa `1`).

```python
try:
    handoff = load_schema_handoff(path)
    schema_map = {col.name: col for col in handoff.columns}
except Exception:
    pass
```
Carica lo `SchemaHandoff` se esiste. Il `try/except` silenzioso è intenzionale: la consistency può girare anche senza schema (andrà sempre nel slow path per tutte le colonne).

```python
if max_workers == 1 or len(column_names) <= 1:
    # path sincrono — colonna per colonna
    for column_name in column_names:
        reports.append(run_column_format_check(...))
else:
    # path asincrono — max_workers colonne in parallelo
    reports = asyncio.run(_run_column_format_checks_async(...))
```
Con una sola colonna o un solo worker, usa la versione sincrona più semplice. Altrimenti, lancia il runner asincrono.

```python
for result in reports:
    if result.finding is not None:
        format_findings.append(result.finding)
```
Filtra le colonne senza problemi — tiene solo i finding reali.

```python
report = ConsistencyValidationReport(
    dataset_name=path.stem,
    total_rows=len(df),
    format_consistency_findings=format_findings,
    summary=f"Analyzed all {len(df.columns)} columns... detected {len(format_findings)} format issues.",
)
save_consistency(path, report)
return report
```
Assembla il report finale e lo salva in cache.

---

## Schema complessivo del flusso

```
CSV file
   │
   ├─► load_dataset_frame()
   ├─► load_schema_handoff() ─► schema_map (opzionale)
   │
   │  per ogni colonna:
   │  ├─► string_role in (name, free_text)? → SKIP
   │  │
   │  ├─► build_column_format_facts() → format_facts
   │  │       └─► not machine_format_candidate? → SKIP
   │  │
   │  ├─► schema_entry.detected_pattern?
   │  │       ├─► YES → FAST PATH (Python puro)
   │  │       │         _profile_schema_guided_inconsistencies()
   │  │       │         _build_suggested_strategy()
   │  │       │         → FormatConsistencyFinding (no LLM)
   │  │       │
   │  │       └─► NO  → SLOW PATH (LLM)
   │  │                 [format_consistency_agent]
   │  │                 → ColumnConsistencyReport
   │  │
   │  └─► (async con semaphore se max_workers > 1)
   │
   ├─► filtra finding non null
   └─► ConsistencyValidationReport → cache JSON
```

---

## Confronto con i moduli precedenti

| Aspetto | `schema.py` | `completeness.py` | `consistency.py` |
|---|---|---|---|
| Chiamate LLM | 2 (sempre) | 1 (sempre) | 0–N (dipende dal fast/slow path) |
| Parallelismo | No | No | Sì (asyncio + semaphore) |
| Fast path | No | No | Sì (schema-guided, no LLM) |
| Granularità | Dataset intero | Dataset intero | Per colonna |
| Output usato da | Fase successiva + summary | Fase cleaning | Fase cleaning (suggested_strategy) |

---

# 4. validation/anomaly.py

Questo modulo è strutturalmente simile a `schema.py`: la **detection è Python puro**, l'LLM è usato solo per scrivere il summary finale. Ecco la spiegazione completa.

---

## Docstring del modulo (righe 1–8)

Due principi chiave:
1. I detector euristici (outlier numerici + categorie rare) costruiscono i finding localmente — **nessun LLM per la detection**
2. `anomaly_summary_agent` scrive solo il riassunto leggibile sul report già costruito
3. Le colonne marcate come **duplicate semantiche** nella fase schema vengono soppresse per l'alias non-preferito — evita di segnalare lo stesso anomaly due volte

---

## Imports (righe 10–25)

```python
from agents import anomaly_summary_agent
```
L'unico agente — usato solo per il summary finale.

```python
from cache import load_anomaly, load_schema_handoff, save_anomaly
```
Cache del report di anomaly + caricamento dello `SchemaHandoff` (per leggere i gruppi di duplicati).

```python
from models import AnomalyDetectionReport, AnomalyFinding, SchemaColumnEntry
```
- `AnomalyFinding` — singolo finding (colonna, tipo di anomalia, righe affette, esempi)
- `AnomalyDetectionReport` — container di tutti i finding + summary

```python
from tools import (
    detect_numeric_outlier_candidates,   # euristica: IQR / z-score per colonne numeriche
    detect_rare_category_candidates,      # euristica: categorie con frequenza < soglia
    load_dataset_frame,
    normalized_schema_name,              # "DataNascita" → "data_nascita"
)
```

```python
from validation._summary import summarize_validation_report
```
Questa riga importa la funzione `summarize_validation_report` dal file/modulo `validation._summary`. In pratica, questa funzione è un **aiutante comune** usato da più parti del progetto: prova a chiamare un agente per creare un riassunto del report di validazione; se qualcosa va storto, invece di bloccare tutto, usa un riassunto testuale più semplice come alternativa. Non si trova in `tools/` perché non è uno strumento operativo vero e proprio, ma una funzione interna di supporto. È stata creata per evitare di riscrivere lo stesso codice in ogni modulo di validazione.


---

## Funzione `_duplicate_semantic_suppressed_columns()` (righe 28–46)

**Input:** `columns: list[SchemaColumnEntry]`, `duplicate_groups` (dalla `SchemaHandoff`)

**Output:** `set[str]` — nomi di colonne da sopprimere nell'anomaly detection

**Non usa agenti** — logica di ranking Python.

**Perché esiste:** se `data_nascita` e `DataNascita` sono colonne duplicate semantiche, riportare anomaly su entrambe è ridondante. Si tiene solo la colonna "preferita" e si sopprimono le altre.

```python
by_name = {column.name: column for column in columns}
suppressed: set[str] = set()
for group in duplicate_groups:
    present = [name for name in group.columns if name in by_name]
    if len(present) < 2:
        continue
```
Per ogni gruppo di duplicati con almeno 2 colonne presenti nel DataFrame, si sceglie la preferita.

```python
def _sort_key(name: str) -> tuple[int, int, str]:
    column = by_name[name]
    return (
        0 if column.naming_valid else 1,       # priorità 1: nome valido (snake_case)
        0 if normalized_schema_name(name) == name else 1,  # priorità 2: già normalizzato
        name,                                   # priorità 3: alfabetico (tie-breaker)
    )

preferred = sorted(present, key=_sort_key)[0]
suppressed.update(name for name in present if name != preferred)
```
Il sort key è una tripla: si preferisce la colonna con naming valido, poi quella il cui nome è già il nome canonico, poi alfabetico. La prima dopo il sort è la preferita; tutte le altre vanno nel set di soppresse.

**Esempio:** gruppo `["DataNascita", "data_nascita"]`
- `"DataNascita"` → `(1, 1, "DataNascita")` (naming non valido, non è già normalizzato)
- `"data_nascita"` → `(0, 0, "data_nascita")` (naming valido, già normalizzato)
- Preferita: `"data_nascita"` → `"DataNascita"` viene soppressa

---

## Funzione `run_anomaly_detection()` (righe 49–98) — entry point pubblico

**Input:** `path: Path`, `reuse_cache: bool`

**Output:** `AnomalyDetectionReport` — lista di `AnomalyFinding` + summary testuale

**Usa l'agente `anomaly_summary_agent`** — una sola chiamata LLM, solo per il summary.

### Riga 50–51 — cache hit
```python
if reuse_cache:
    return load_anomaly(path)
```
Ritorna il risultato precedente senza fare nulla.

### Righe 53–60 — caricamento dati e schema
```python
df = load_dataset_frame(path)
try:
    handoff = load_schema_handoff(path)
    schema_columns = handoff.columns
    suppressed_columns = _duplicate_semantic_suppressed_columns(handoff.columns, handoff.duplicate_groups)
except FileNotFoundError:
    schema_columns = []
    suppressed_columns = set()
```
Questo blocco prima carica il dataset dentro `df`, cioè una tabella Python simile a un foglio Excel. Poi prova a caricare anche le informazioni prodotte dalla fase di schema, chiamate `SchemaHandoff`: da lì prende la lista delle colonne e capisce quali colonne duplicate devono essere ignorate. Il `try/except` serve perché questo file di schema potrebbe non esistere: se la fase schema non è stata fatta, il codice non si blocca, ma continua comunque l’anomaly detection usando `schema_columns = []` e `suppressed_columns = set()`, cioè senza informazioni extra sullo schema e senza colonne duplicate da escludere.


### Righe 62–69 — detection euristica (no LLM)
```python
findings = [
    AnomalyFinding(**finding)
    for finding in (
        detect_numeric_outlier_candidates(df, schema_columns)
        + detect_rare_category_candidates(df, schema_columns)
    )
    if finding["column_name"] not in suppressed_columns
]
```
Due liste di finding (dict) concatenate:
- `detect_numeric_outlier_candidates` — per colonne numeriche, rileva valori statisticamente anomali (IQR o z-score)
- `detect_rare_category_candidates` — per colonne categoriali, rileva valori con frequenza troppo bassa (es. una categoria presente in 3 righe su 10.000)

`schema_columns` è passato ai detector per usare il dtype inferito dall'agente — es. trattare una colonna come `Int64` anche se pandas l'ha caricata come `object`.

Il filtro `if finding["column_name"] not in suppressed_columns` scarta i finding delle colonne non-preferite nei gruppi di duplicati.

`AnomalyFinding(**finding)` converte ogni dict in un oggetto Pydantic validato.

### Riga 70 — ordinamento
```python
findings.sort(key=lambda finding: (-finding.affected_rows, finding.column_name, finding.anomaly_type))
```
Ordina per: righe affette decrescente (i problemi più gravi prima), poi nome colonna, poi tipo di anomalia. Questo ordine è usato nel report e nel testo del summary.

### Righe 71–82 — fallback summary e costruzione report
```python
fallback_summary = (
    f"Detected {len(findings)} anomaly findings across numeric outliers and rare categorical values."
    if findings
    else "No anomaly findings were detected by the current heuristic checks."
)
report = AnomalyDetectionReport(
    dataset_name=path.stem,
    total_rows=len(df),
    total_columns=len(df.columns),
    findings=findings,
    summary=fallback_summary,    # <— summary provvisorio
)
```
Il report viene costruito con un summary provvisorio (testuale, generato da Python). Se il summary agent fallisce, questo è il fallback.

### Righe 83–96 — summary agent (unica chiamata LLM)
```python
report = report.model_copy(
    update={
        "summary": summarize_validation_report(
            anomaly_summary_agent,
            f"Summarize the provided anomaly-detection findings for dataset {path.stem}. "
            "Do not infer new findings or alter the provided findings.",
            report,
            fallback_summary,
        )
    }
)
```
`summarize_validation_report` (da `validation/_summary.py`) fa tre cose:
1. Serializza il `report` e lo allega come testo
2. Chiama `anomaly_summary_agent` con il prompt + allegato
3. Se l'agente fallisce (eccezione o output vuoto), restituisce `fallback_summary`

Il pattern `report.model_copy(update={"summary": ...})` aggiorna solo il campo `summary` mantenendo tutto il resto immutato — è il pattern Pydantic per "aggiorna un campo su un oggetto immutabile".

L'agente **non rideriva nulla** — riceve i finding già costruiti e produce solo una narrazione leggibile. La nota `"Do not infer new findings"` nel prompt lo vincola esplicitamente.

### Righe 97–98 — salvataggio e return
```python
save_anomaly(path, report)
return report
```
Salva in `Data/.validation_cache/<dataset>.anomaly.json` e restituisce al chiamante (`bundle.py`).

---

## Schema del flusso

```
CSV file
   │
   ├─► load_dataset_frame() ──────────────────────────────► DataFrame
   │
   ├─► load_schema_handoff() ──────────────────────────────► SchemaHandoff
   │         (opzionale, FileNotFoundError silenzioso)              │
   │                                                                │
   │                               _duplicate_semantic_suppressed_columns()
   │                                                                │
   │                                                       suppressed_columns
   │
   ├─► detect_numeric_outlier_candidates(df, schema_columns)  ─┐
   ├─► detect_rare_category_candidates(df, schema_columns)    ─┤
   │                                                           │ concat + filter
   │                                                           ▼
   │                                                     AnomalyFinding list
   │                                                           │ sort
   │                                                           ▼
   │                                              AnomalyDetectionReport
   │                                           (summary = fallback testuale)
   │
   └─► [LLM] anomaly_summary_agent ──────────────────────────► summary testuale
                                                                      │
                                              report.model_copy(update={"summary": ...})
                                                                      │
                                                            save_anomaly() → return
```

---

## Confronto con i moduli precedenti

| Aspetto | `schema.py` | `completeness.py` | `consistency.py` | `anomaly.py` |
|---|---|---|---|---|
| Chiamate LLM | 2 | 1 | 0–N per colonna | 1 (solo summary) |
| Detection | Agente | Agente | Fast/slow path | **Python puro** |
| Summary LLM | Sì | No | No | Sì |
| Dipende da schema | No | No | Sì (fast path) | Sì (soppressione duplicati) |
| Fallback se LLM fallisce | No | No | No | **Sì** (fallback testuale) |

`anomaly.py` è il modulo che fa il **minor uso di LLM per la logica core**: tutta la detection è euristica Python, l'LLM è relegato esclusivamente alla narrativa finale.

---

# 5. validation/cross_column.py

Questo è il modulo più corto finora — una sola funzione pubblica, struttura quasi identica ad `anomaly.py`. Lo schema è sempre lo stesso: **Python trova i problemi, l'LLM scrive solo il riassunto**.

---

## Docstring del modulo (righe 1–7)

Quattro tipi di controlli **tra colonne** (non dentro una singola colonna):
- Colonne duplicate-like (stesso contenuto, nomi diversi)
- Conflitti semantici tra colonne duplicate
- Disallineamenti anno/mese/periodo
- Violazioni nell'ordine delle date (es. `data_fine` prima di `data_inizio`)

`cross_column_summary_agent` scrive solo il summary — **non aggiunge né modifica** i finding.

---

## Imports (righe 9–25)

```python
from agents import cross_column_summary_agent
```
L'unico agente — solo per il summary finale.

```python
from cache import load_cross_column, load_schema_handoff, save_cross_column
```
Cache del report + schema (per i gruppi di duplicati e i dtype delle colonne).

```python
from tools import (
    detect_duplicate_like_columns,         # colonne con valori quasi identici
    detect_duplicate_semantic_conflicts,   # colonne con stesso nome canonico ma valori diversi
    detect_year_month_period_mismatches,   # es. colonna "anno" non allineata con colonna "data"
    detect_date_order_violations,          # es. data_fine < data_inizio
    load_dataset_frame,
)
```
Quattro funzioni euristiche — tutte Python puro, nessun LLM.

```python
from validation._summary import summarize_validation_report
```
Stesso helper condiviso usato in `anomaly.py` — chiama l'agente per il summary con fallback testuale.

---

## Funzione `run_cross_column_validation()` (righe 28–77)

**Input:** `path: Path`, `reuse_cache: bool`

**Output:** `CrossColumnValidationReport` — lista di `CrossColumnFinding` + summary

**Usa l'agente `cross_column_summary_agent`** — una sola chiamata LLM, solo per il summary.

### Righe 29–30 — cache hit
```python
if reuse_cache:
    return load_cross_column(path)
```
Se il risultato è già in cache, ritorna subito.

### Righe 32–39 — caricamento dati e schema
```python
df = load_dataset_frame(path)
try:
    handoff = load_schema_handoff(path)
    schema_columns = handoff.columns
    duplicate_groups = handoff.duplicate_groups
except FileNotFoundError:
    schema_columns = []
    duplicate_groups = []
```
Carica il DataFrame e prova a leggere lo `SchemaHandoff`. Come in `anomaly.py`, il `try/except` è silenzioso: se la fase schema non è stata eseguita, i detector girano comunque ma senza contesto di dtype o gruppi di duplicati noti.

`duplicate_groups` è la lista già calcolata da `schema.py` (colonne con stesso nome canonico) — qui viene riutilizzata direttamente invece di ricalcolarla.

### Righe 41–49 — quattro detector euristici (no LLM)
```python
findings = [
    CrossColumnFinding(**finding)
    for finding in (
        detect_duplicate_like_columns(df, schema_columns)
        + detect_duplicate_semantic_conflicts(df, duplicate_groups)
        + detect_year_month_period_mismatches(df, schema_columns)
        + detect_date_order_violations(df, schema_columns)
    )
]
```
Le quattro liste di dict vengono concatenate, filtrate e convertite in oggetti Pydantic. Cosa fa ciascun detector:

| Detector | Cosa cerca | Esempio |
|---|---|---|
| `detect_duplicate_like_columns` | Colonne con valori quasi uguali (alta correlazione o % di match) | `codice_fiscale` e `cf` hanno gli stessi valori |
| `detect_duplicate_semantic_conflicts` | Colonne con stesso nome canonico ma valori diversi | `data_nascita` e `DataNascita` hanno date diverse → incoerenza |
| `detect_year_month_period_mismatches` | Anno/mese estratto da una colonna non combacia con un'altra | Colonna `anno` = 2023, ma `data` = 15/01/2024 |
| `detect_date_order_violations` | Data di fine prima di data di inizio | `data_cessazione` < `data_attivazione` |

`CrossColumnFinding` ha un campo `columns` (lista di nomi di colonne coinvolte) invece del singolo `column_name` degli altri finding — perché ogni finding riguarda **almeno due colonne**.

### Riga 50 — ordinamento
```python
findings.sort(key=lambda finding: (-finding.affected_rows, ",".join(finding.columns), finding.check_type))
```
Ordina per: righe affette decrescente (i problemi più diffusi prima), poi nomi colonne coinvolte (alfabetico), poi tipo di check. L'uso di `",".join(finding.columns)` è necessario perché `columns` è una lista, non una stringa.

### Righe 51–61 — fallback summary e costruzione report
```python
fallback_summary = (
    f"Detected {len(findings)} cross-column consistency findings."
    if findings
    else "No cross-column consistency findings were detected by the current rule set."
)
report = CrossColumnValidationReport(
    dataset_name=path.stem,
    total_rows=len(df),
    findings=findings,
    summary=fallback_summary,    # summary provvisorio
)
```
Il report viene costruito subito con un summary Python — se l'LLM fallisce, questo è il testo di riserva.

### Righe 63–75 — summary agent (unica chiamata LLM)
```python
report = report.model_copy(
    update={
        "summary": summarize_validation_report(
            cross_column_summary_agent,
            f"Summarize the provided cross-column validation findings for dataset {path.stem}. "
            "Do not infer new findings or alter the provided findings.",
            report,
            fallback_summary,
        )
    }
)
```
Identico al pattern di `anomaly.py`:
1. `summarize_validation_report` serializza il report, lo allega e chiama l'agente
2. L'agente scrive un testo narrativo leggibile sui finding già trovati
3. Se l'agente fallisce → usa `fallback_summary`
4. `model_copy(update=...)` aggiorna solo il campo `summary`

### Righe 76–77 — salvataggio e return
```python
save_cross_column(path, report)
return report
```
Salva in cache e restituisce al chiamante (`bundle.py`).

---

## Schema del flusso

```
CSV file
   │
   ├─► load_dataset_frame() ──────────────────────────────► DataFrame
   │
   ├─► load_schema_handoff() ──────────────────────────────► SchemaHandoff
   │         (opzionale, FileNotFoundError silenzioso)         │        │
   │                                                    schema_columns  duplicate_groups
   │
   ├─► detect_duplicate_like_columns(df, schema_columns)      ─┐
   ├─► detect_duplicate_semantic_conflicts(df, dup_groups)    ─┤
   ├─► detect_year_month_period_mismatches(df, schema_columns)─┤  concat
   ├─► detect_date_order_violations(df, schema_columns)       ─┘
   │                                                           │ sort
   │                                                           ▼
   │                                              CrossColumnFinding list
   │                                                           │
   │                                          CrossColumnValidationReport
   │                                        (summary = fallback testuale)
   │
   └─► [LLM] cross_column_summary_agent ─────────────────────► summary
                                                                    │
                                          report.model_copy(update={"summary": ...})
                                                                    │
                                                  save_cross_column() → return
```

---

## Confronto con i moduli simili

`cross_column.py` e `anomaly.py` sono quasi gemelli nella struttura. Le differenze:

| Aspetto | `anomaly.py` | `cross_column.py` |
|---|---|---|
| Cosa analizza | Singola colonna (outlier, rare categorie) | Relazione tra colonne |
| Finding key | `column_name` (stringa) | `columns` (lista di stringhe) |
| Suppression logic | Sì (duplicati semantici) | No |
| Schema usato per | Dtype + soppressione | Dtype + `duplicate_groups` |
| Detector | 2 (`numeric` + `rare_category`) | 4 (like, semantic, year/month, date-order) |

Entrambi usano esattamente lo stesso pattern LLM: **Python trova, LLM narra**.

---

# 6. validation/duplicates.py

Questo modulo è quasi identico a `cross_column.py` nella struttura — stessa architettura Python-trova/LLM-narra. Lo spiego in modo conciso evidenziando le parti nuove.

---

## Docstring (righe 1–6)

Due tipi di detection, entrambi Python puro:
- **Exact duplicates** — righe completamente identiche
- **Near duplicates** — righe simili, identificate tramite colonne chiave inferite (es. stesso codice fiscale, date diverse)

`duplicate_summary_agent` scrive solo il summary sul report già costruito.

---

## Imports (righe 8–23)

```python
from tools import (
    detect_exact_duplicate_groups,   # righe con tutti i valori uguali
    detect_near_duplicate_groups,    # righe simili su colonne chiave
    infer_duplicate_key_columns,     # capisce quali colonne usare come chiave
    load_dataset_frame,
)
```
La novità rispetto ai moduli precedenti è `infer_duplicate_key_columns` — serve per il near-duplicate detection e funziona così: guarda i `SchemaColumnEntry` (dtype, ruolo semantico) e decide quali colonne identificano "la stessa persona/entità" (es. codice fiscale, matricola). Senza schema usa euristiche sui nomi delle colonne.

---

## Funzione `run_duplicate_detection()` (righe 26–71)

**Input:** `path: Path`, `reuse_cache: bool`

**Output:** `DuplicateDetectionReport` — lista di `DuplicateRecordGroup` + summary

**Usa l'agente `duplicate_summary_agent`** — una sola chiamata LLM, solo per il summary.

### Righe 27–35 — cache hit e caricamento
```python
if reuse_cache:
    return load_duplicates(path)

df = load_dataset_frame(path)
try:
    handoff = load_schema_handoff(path)
    schema_columns = handoff.columns
except FileNotFoundError:
    schema_columns = []
```
Identico agli altri moduli: cache hit → ritorna subito. Schema opzionale — se manca, i detector girano senza contesto di dtype.

### Riga 37 — inferenza delle colonne chiave
```python
key_columns = infer_duplicate_key_columns(schema_columns, df)
```
Questa è la differenza principale rispetto ad `anomaly.py` e `cross_column.py`. Prima di cercare duplicati "simili", il sistema deve capire **quali colonne identificano un'entità unica** nel dataset. `infer_duplicate_key_columns` analizza i ruoli semantici delle colonne (es. `string_role="identifier"`, nomi come `id`, `codice`, `matricola`) e restituisce la lista di colonne da usare come chiave per il near-duplicate check.

**Perché è necessario:** senza una chiave, "near duplicate" non ha senso — qualsiasi riga con anche solo un campo uguale potrebbe sembrare duplicata. La chiave dice: "queste due righe parlano della stessa persona se hanno gli stessi valori in queste colonne".

### Righe 38–44 — due detector euristici (no LLM)
```python
groups = [
    DuplicateRecordGroup(**group)
    for group in (
        detect_exact_duplicate_groups(df)
        + detect_near_duplicate_groups(df, key_columns)
    )
]
```

| Detector | Cosa cerca | Come |
|---|---|---|
| `detect_exact_duplicate_groups` | Righe completamente identiche | `df.duplicated()` — tutti i campi uguali |
| `detect_near_duplicate_groups` | Righe con stessa chiave ma altri campi diversi | Raggruppa per `key_columns`, cerca gruppi con >1 riga |

`DuplicateRecordGroup` contiene: gli indici delle righe duplicate, le colonne usate come chiave, esempi di valori, quante righe sono coinvolte.

### Righe 45–55 — fallback summary e report
```python
fallback_summary = (
    f"Detected {len(groups)} duplicate-record groups."
    if groups
    else "No duplicate-record groups were detected..."
)
report = DuplicateDetectionReport(
    dataset_name=path.stem,
    total_rows=len(df),
    groups=groups,
    summary=fallback_summary,
)
```
Identico al pattern degli altri moduli — report costruito subito con summary Python di riserva.

### Righe 57–69 — summary agent (unica chiamata LLM)
```python
report = report.model_copy(
    update={
        "summary": summarize_validation_report(
            duplicate_summary_agent,
            f"Summarize the provided duplicate-detection findings for dataset {path.stem}. "
            "Do not infer new findings or alter the provided findings.",
            report,
            fallback_summary,
        )
    }
)
```
Stesso pattern di `anomaly.py` e `cross_column.py`: l'agente riceve il report serializzato, produce un testo narrativo, fallback se fallisce.

### Righe 70–71 — salvataggio e return
```python
save_duplicates(path, report)
return report
```
Cache in `Data/.validation_cache/<dataset>.duplicates.json`, poi ritorna a `bundle.py`.

---

## Schema del flusso

```
CSV file
   │
   ├─► load_dataset_frame() ────────────────────────► DataFrame
   │
   ├─► load_schema_handoff() ───────────────────────► schema_columns
   │         (opzionale)
   │
   ├─► infer_duplicate_key_columns(schema_columns, df)
   │         └─► key_columns (es. ["codice_fiscale", "matricola"])
   │
   ├─► detect_exact_duplicate_groups(df) ──────────────┐
   ├─► detect_near_duplicate_groups(df, key_columns) ──┘ concat
   │                                                    │
   │                                          DuplicateRecordGroup list
   │                                                    │
   │                                       DuplicateDetectionReport
   │                                     (summary = fallback testuale)
   │
   └─► [LLM] duplicate_summary_agent ─────────────────► summary
                                                              │
                                      report.model_copy(update={"summary": ...})
                                                              │
                                              save_duplicates() → return
```

---

## Confronto tra i tre moduli "gemelli"

| Aspetto | `anomaly.py` | `cross_column.py` | `duplicates.py` |
|---|---|---|---|
| Granularità finding | Per colonna | Per coppia/gruppo di colonne | Per gruppo di righe |
| Step extra pre-detection | Soppressione duplicati semantici | Nessuno | Inferenza `key_columns` |
| Detector | 2 | 4 | 2 |
| Schema usato per | Dtype + soppressione | Dtype + `duplicate_groups` | Dtype + ruoli semantici |
| LLM | Solo summary | Solo summary | Solo summary |
| Fallback summary | Sì | Sì | Sì |

Tutti e tre seguono esattamente lo stesso schema architetturale: **euristiche Python → report → LLM summary con fallback**. La differenza è solo nel tipo di problema che cercano e nello step di preparazione prima dei detector.

---

# 7. validation/bundle.py

Questo è il modulo più semplice di tutti — è solo il **collante** che chiama in sequenza tutti gli stage che abbiamo già visto.

---

## Docstring (righe 1–6)

`build_validation_results` esegue tutti e sei gli stage di validazione in ordine e salva il risultato combinato in un `OrchestrationStepResult`.

---

## Imports (righe 8–21)

```python
from cache import save_validation_results
from models import OrchestrationStepResult
```
`OrchestrationStepResult` è il contenitore Pydantic che tiene insieme tutti e sei i report. `save_validation_results` lo serializza in un unico JSON.

```python
from validation.anomaly import run_anomaly_detection
from validation.completeness import run_completeness_analysis
from validation.consistency import run_format_consistency_validation
from validation.cross_column import run_cross_column_validation
from validation.duplicates import run_duplicate_detection
from validation.schema import run_schema_validation
```
Importa le sei funzioni entry point, una per ogni modulo che abbiamo analizzato.

---

## Funzione `build_validation_results()` (righe 24–48)

**Input:**
- `path: Path` — percorso al CSV
- `reuse_schema`, `reuse_completeness`, `reuse_consistency` — flag per caricare dalla cache invece di rieseguire

**Output:** `OrchestrationStepResult` — tutti e sei i report dentro un unico oggetto

**Non usa agenti direttamente** — li chiamano i moduli che invoca.

### Righe 30–32 — i tre stage con cache opzionale
```python
schema_validation = run_schema_validation(path, reuse_cache=reuse_schema)
completeness_analysis = run_completeness_analysis(path, reuse_cache=reuse_completeness)
consistency_validation = run_format_consistency_validation(path, reuse_cache=reuse_consistency)
```
Schema, completeness e consistency ricevono il flag `reuse_cache` dalla CLI (`--reuse-schema`, `--reuse-completeness`, `--reuse-consistency`). Se il flag è `True`, caricano il JSON in cache invece di chiamare gli agenti.

### Righe 33–38 — i tre stage senza cache
```python
anomaly_detection = run_anomaly_detection(path)
cross_column_validation = run_cross_column_validation(path)
duplicate_detection = run_duplicate_detection(path)
```
Anomaly, cross-column e duplicates non hanno flag di cache esposti nella CLI — girano sempre. Le righe di `print` prima di ciascuno tracciano il progresso su stderr.

**Perché questa asimmetria?** I primi tre stage (schema, completeness, consistency) sono i più lenti — fanno molte chiamate LLM, soprattutto consistency che gira una per colonna. I secondi tre sono prevalentemente euristici e veloci, quindi non è stato necessario esporli come flag CLI.

### Righe 39–46 — assemblaggio del risultato
```python
validation_results = OrchestrationStepResult(
    schema_validation=schema_validation,
    completeness_analysis=completeness_analysis,
    consistency_validation=consistency_validation,
    anomaly_detection=anomaly_detection,
    cross_column_validation=cross_column_validation,
    duplicate_detection=duplicate_detection,
)
```
Raccoglie i sei report in un unico oggetto Pydantic. Questo oggetto è ciò che viene poi letto dalla fase di cleaning per costruire i `ColumnCleaningRequest`.

### Righe 47–48 — salvataggio e return
```python
save_validation_results(path, validation_results)
return validation_results
```
Salva l'intero bundle in un unico JSON e restituisce al chiamante (`cli.py`).

---

## Schema del flusso completo della validation

```
path + flags CLI
       │
       ▼
build_validation_results()
       │
       ├─► run_schema_validation()          → SchemaHandoff
       │       2 LLM calls
       │
       ├─► run_completeness_analysis()      → CompletenessAnalysisReport
       │       1 LLM call
       │
       ├─► run_format_consistency_validation() → ConsistencyValidationReport
       │       0–N LLM calls (per colonna, fast/slow path)
       │
       ├─► run_anomaly_detection()          → AnomalyDetectionReport
       │       1 LLM call (solo summary)
       │
       ├─► run_cross_column_validation()    → CrossColumnValidationReport
       │       1 LLM call (solo summary)
       │
       └─► run_duplicate_detection()        → DuplicateDetectionReport
               1 LLM call (solo summary)
                    │
                    ▼
          OrchestrationStepResult
          (tutti e sei i report)
                    │
          save_validation_results()
                    │
                 return
```

---

## Ruolo nell'architettura globale

`bundle.py` è l'unico punto in cui la CLI entra nella validation. Quando scrivi:

```bash
python main.py Data/file.csv --stage validate
```

`bundle.py` è il punto centrale che collega la **CLI** alla parte di **validation**. Quando lanci `python main.py Data/file.csv --stage validate`, il file `cli.py` chiama la funzione `build_validation_results()`, che esegue la validazione e restituisce un risultato organizzato (`OrchestrationStepResult`). Quando invece usi `--stage clean`, il progetto fa prima la validazione e poi usa quei risultati per decidere come costruire i cleaner, cioè le funzioni/agenti che sistemano i dati. Quindi `bundle.py` funziona come un ponte: da una parte raccoglie l’analisi dei problemi del dataset, dall’altra prepara le informazioni che serviranno alla fase di pulizia.


---

# 8. cleaning/remediation.py

Questo è il modulo **cerniera** tra validation e cleaning: legge tutti e sei i report di validazione e produce una lista piatta di azioni da eseguire. Non usa agenti LLM — è pura logica Python.

---

## Docstring (righe 1–8)

Cinque tipi di azione possibili:
- `rename_column` — rinomina colonne con naming non valido
- `replace_placeholders_with_null` — sostituisce token placeholder con null
- `cast_dtype` — converte la colonna al dtype inferito
- `generate_cleaner` — genera una funzione Python per pulire inconsistenze di formato
- `drop_exact_duplicate_column` — elimina una colonna identica a un'altra

Ogni azione ha un flag `auto_apply` che la fase di application usa per decidere se eseguirla automaticamente o solo segnalarla.

---

## Funzioni helper private

### `_action_id()` (righe 20–23)
```python
def _action_id(prefix: str, *parts: object) -> str:
    normalized_parts = [normalized_schema_name(str(part)) for part in parts if str(part).strip()]
    suffix = "__".join(part for part in normalized_parts if part)
    return f"{prefix}__{suffix}" if suffix else prefix
```
Genera un ID univoco e leggibile per ogni azione. Normalizza tutte le parti in snake_case e le unisce con `__`. Esempio: `_action_id("rename_column", "DataNascita", "data_nascita")` → `"rename_column__data_nascita__data_nascita"`.

### `_schema_column_map()` (righe 26–27)
```python
return {column.name: column for column in validation_results.schema_validation.columns}
```
Costruisce un dict `nome_colonna → SchemaColumnEntry` per accesso rapido per nome. Usato più avanti per sapere le proprietà di una colonna dato il suo nome.

### `_target_canonical_name()` (righe 30–31)
```python
return column.rename_suggestion or column.name
```
Ritorna il nome finale di una colonna: se ha una suggestion di rinomina, quella; altrimenti il nome attuale. Usato per costruire azioni che puntano al nome *post-rinomina*.

### `_planned_rename_map()` (righe 34–49)
**Input:** lista di `SchemaColumnEntry`
**Output:** `dict[str, str]` — mappa `nome_attuale → nome_finale`

```python
for column in columns:
    if column.naming_valid or not column.rename_suggestion:
        continue
    target = column.rename_suggestion
    if target in existing and target != column.name:
        suffix = 2
        while f"{target}_{suffix}" in existing:
            suffix += 1
        target = f"{target}_{suffix}"
    rename_map[column.name] = target
    existing.add(target)
```
Costruisce la mappa di rinomina gestendo i **conflitti**: se `"data_nascita"` è già il nome di un'altra colonna, la suggestion diventa `"data_nascita_2"`, poi `"data_nascita_3"`, ecc. Il set `existing` si aggiorna ad ogni iterazione per tenere traccia dei nomi già "occupati" (sia originali che già assegnati).

### `_column_keep_sort_key()` (righe 52–59)
**Input:** `SchemaColumnEntry`
**Output:** tupla per il sort — colonna "migliore" = sort key più bassa

```python
return (
    0 if column.naming_valid else 1,          # preferisce naming valido
    -column.non_null_rows,                     # preferisce più valori non nulli
    0 if normalized_schema_name(target_name) == target_name else 1,  # preferisce già normalizzato
    column.name,                               # tie-breaker alfabetico
)
```
Definisce cosa rende una colonna "preferibile" rispetto a un'altra quando sono duplicate.

### `_choose_keep_drop_columns()` (righe 62–71)
**Input:** due nomi di colonna + schema_map
**Output:** `(keep_name, drop_name)` — quale tenere e quale eliminare

```python
keep = sorted([left, right], key=_column_keep_sort_key)[0]
drop = right if keep.name == left.name else left
return keep.name, drop.name
```
Usa il sort key sopra per scegliere la colonna preferita tra due esatte duplicate.

### `_build_summary()` (righe 74–79)
Conta le azioni auto-apply vs. manuali e restituisce una stringa tipo `"Planned 12 remediation actions: 8 auto-apply and 4 review/report actions."`.

---

## Funzione `build_remediation_plan()` (righe 82–272) — cuore del modulo

**Input:** `OrchestrationStepResult` — i sei report di validazione
**Output:** `RemediationPlan` — lista ordinata di `RemediationAction`
**Nessun LLM** — pura trasformazione di report in azioni.

Itera sui sei report in ordine e produce azioni diverse per ognuno:

### 1. Schema → rename + cast (righe 87–121)

```python
for column in validation_results.schema_validation.columns:
```

**Rename** (se naming non valido):
```python
RemediationAction(
    action_type="rename_column",
    auto_apply=True,       # eseguita automaticamente
    risk_level="low",      # rinominare una colonna è sicuro
    confidence="high",
    ...
)
```

**Cast dtype** (se il dtype inferito è uno dei tipi "sicuri"):
```python
if column.pandas_dtype in {"datetime64[ns]", "Int64", "Float64", "boolean", "string"}:
    final_column_name = rename_map.get(column.name, column.name)  # usa il nome post-rinomina
    RemediationAction(action_type="cast_dtype", auto_apply=True, ...)
```
Nota: `final_column_name` usa già il nome rinominato — perché la rinomina viene applicata prima del cast nella fase application.

### 2. Completeness → replace placeholders (righe 123–143)

```python
for column_finding in validation_results.completeness_analysis.per_column:
    if column_finding.missing_like_count <= 0 or not column_finding.missing_like_examples:
        continue
    RemediationAction(
        action_type="replace_placeholders_with_null",
        auto_apply=True,    # sicuro: converte "N/A" → NaN
        risk_level="low",
        confidence="high",
        ...
    )
```

### 3. Consistency → generate cleaner (righe 145–163)

```python
for finding in validation_results.consistency_validation.format_consistency_findings:
    RemediationAction(
        action_type="generate_cleaner",
        auto_apply=True,      # il cleaner viene generato e applicato
        risk_level="medium",  # più rischioso — il cleaner potrebbe sbagliare
        confidence="high",
        target={"column_name": ..., "expected_pattern": ...},
        ...
    )
```
`risk_level="medium"` perché il cleaner è codice generato da LLM — meno prevedibile di una rinomina.

### 4. Cross-column → drop o manual review (righe 165–214)

Due casi distinti:

**Colonne esatte duplicate** → drop automatico:
```python
if finding.check_type == "exact_duplicate_columns" and len(finding.columns) == 2:
    keep_name, drop_name = _choose_keep_drop_columns(...)
    RemediationAction(
        action_type="drop_exact_duplicate_column",
        auto_apply=True,
        risk_level="low",
        ...
    )
```

**Tutti gli altri tipi** (`near_duplicate_columns`, `duplicate_semantic_conflict`, `year_month_period_mismatch`, `date_order_violation`) → solo segnalazione:
```python
RemediationAction(
    action_type="manual_review",
    auto_apply=False,   # NON eseguita automaticamente
    status="proposed_not_applied",
    risk_level="medium" if near_duplicate else "high",
    confidence="medium",
    ...
)
```
Questi problemi richiedono giudizio umano — il sistema non può sapere quale dei due valori discordanti sia quello corretto.

### 5. Anomaly → manual review o report only (righe 216–236)

```python
action_type = "manual_review" if finding.severity == "high" else "report_only"
RemediationAction(
    auto_apply=False,
    status="proposed_not_applied",
    ...
)
```
Gli outlier non vengono mai corretti automaticamente — potrebbero essere dati legittimi o errori, impossibile saperlo senza contesto di dominio.

### 6. Duplicati → drop candidate o manual review (righe 238–265)

```python
if group.duplicate_type == "exact_row":
    action_type = "drop_rows_candidate"
    confidence = "medium"
    risk_level = "medium"
else:
    action_type = "manual_review"
    confidence = "low"
    risk_level = "high"
RemediationAction(auto_apply=False, ...)
```
Anche i duplicati esatti di riga non vengono eliminati automaticamente (`auto_apply=False`) — la decisione finale è dell'utente. Il sistema li marca come `"drop_rows_candidate"` per indicare che sono buoni candidati alla rimozione.

### Ordinamento finale (riga 267)
```python
actions.sort(key=lambda action: (0 if action.auto_apply else 1, action.action_type, action.action_id))
```
Le azioni auto-apply vengono messe prima, poi ordinate per tipo e ID. Questo è l'ordine in cui la fase application le esegue.

---

## Funzioni di supporto per l'entry point

### `_resolve_validation_results()` (righe 275–287)
Tre strategie per ottenere i validation results, in ordine di priorità:
1. Già in memoria (passati come argomento) → usali direttamente
2. In cache su disco (`reuse_saved_validation=True`) → carica il JSON
3. Nessuno dei due → riesegui tutta la validation

### `run_remediation_planning()` (righe 290–305) — entry point pubblico
```python
if reuse_saved_remediation:
    try:
        return load_remediation_plan(path)
    except FileNotFoundError:
        pass

resolved_validation = _resolve_validation_results(path, validation_results, reuse_saved_validation)
plan = build_remediation_plan(resolved_validation)
save_remediation_plan(path, plan)
return plan
```
Gestisce la cache del piano stesso (non solo dei validation results), poi chiama `build_remediation_plan` e salva il risultato.

---

## Schema del flusso

```
OrchestrationStepResult (6 report)
           │
           ▼
build_remediation_plan()
           │
           ├─► schema.columns ──────────────────► rename_column (auto, low risk)
           │                                       cast_dtype    (auto, low risk)
           │
           ├─► completeness.per_column ──────────► replace_placeholders_with_null (auto, low risk)
           │
           ├─► consistency.findings ────────────► generate_cleaner (auto, medium risk)
           │
           ├─► cross_column.findings
           │       ├─ exact_duplicate_columns ──► drop_exact_duplicate_column (auto, low risk)
           │       └─ altri tipi ───────────────► manual_review (NOT auto, medium/high risk)
           │
           ├─► anomaly.findings ────────────────► manual_review / report_only (NOT auto)
           │
           └─► duplicates.groups
                   ├─ exact_row ───────────────► drop_rows_candidate (NOT auto, medium risk)
                   └─ near_duplicate ──────────► manual_review (NOT auto, high risk)
                                │
                                ▼
                    RemediationPlan (lista ordinata)
                    auto_apply first, poi manual
```

---

## Tabella `auto_apply` per tipo di azione

| Azione | Auto apply | Risk | Perché |
|---|---|---|---|
| `rename_column` | Sì | low | Meccanico, reversibile |
| `cast_dtype` | Sì | low | Meccanico, guidato da LLM affidabile |
| `replace_placeholders_with_null` | Sì | low | Token noti, nessuna ambiguità |
| `generate_cleaner` | Sì | medium | Codice LLM, ma validato prima dell'apply |
| `drop_exact_duplicate_column` | Sì | low | Colonne identiche al 100% |
| `manual_review` | No | medium/high | Richiede giudizio umano |
| `report_only` | No | low/medium | Solo informativo |
| `drop_rows_candidate` | No | medium | L'utente decide quali righe eliminare |

---

# 9. cleaning/generation.py + cleaning/validation.py + cleaning/request.py

Questo è il cuore del sistema — il loop più sofisticato di tutto il progetto. Spiego prima il concetto, poi file per file, poi come si intersecano.

---

## Il concetto del loop in una frase

Per ogni colonna con inconsistenze di formato: un agente **genera** una funzione Python, Python la **testa** senza LLM, se fallisce un secondo agente la **diagnostica**, e il generatore **riprova** — fino a 10 volte.

```
[Generator Agent] → funzione Python
        ↓
[Host Validator]  → lista di errori (no LLM)
        ↓ (se fallisce)
[Critic Agent]    → diagnosi del bug
        ↓
[Generator Agent] → funzione corretta (retry)
        ↓ (se passa)
     DONE ✓
```

---

## cleaning/request.py — "prepara il brief per il generatore"

Questo file costruisce il `ColumnCleaningRequest` — il documento che il generatore legge per capire cosa deve fare. **Nessun agente**, pura logica Python.

### `_build_datetime_expected_pattern()` (righe 18–31)
```python
if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}", stripped):
    return "ISO timestamp YYYY-MM-DDTHH:MM:SS.ffffff"
```
Guarda il primo esempio di valore dominante e ne deduce il pattern atteso leggibile. Invece di passare al generatore il valore grezzo `"2024-03-11T02:01:04.421000"`, gli passa la stringa `"ISO timestamp YYYY-MM-DDTHH:MM:SS.ffffff"` — molto più utile per scrivere la funzione di pulizia.

### `_augment_datetime_strategy()` (righe 34–52)
```python
guidance = (
    "Datetime output contract:\n"
    f"- Preserve already-valid dominant timestamps unchanged, for example {example!r}.\n"
    "- The cleaned output must use that same canonical datetime layout..."
)
return guidance + "\n\nExisting shape notes:\n" + suggested_strategy
```
Per le date c'è un problema speciale: se un valore è già `"2024-03-11T02:01:04.421000"`, il cleaner non deve toccarlo — ma se c'è un `if '-' in s:` come primo branch, lo riscrive comunque. Questa funzione **prepende** un contratto esplicito alla strategia già costruita da `consistency.py`, ricordando al generatore: "preserva i dominanti, non swappare solo il separatore, riordina le componenti".

### `build_column_cleaning_request()` (righe 55–88)

**Input:** `finding` (da consistency), `format_facts` (profilo della colonna), `schema_entry` (da schema)

**Output:** `ColumnCleaningRequest` — il bundle completo che va al generatore

```python
example_inconsistent_values = list(dict.fromkeys(finding.example_inconsistent_values))
```
`dict.fromkeys` rimuove i duplicati preservando l'ordine — più pulito di `set()`.

```python
target_dtype = schema_entry.pandas_dtype if schema_entry else None
target_role = schema_entry.numeric_role or schema_entry.string_role if schema_entry else None
```
Fonde le informazioni dello schema (dtype, ruolo semantico) con quelle del finding di consistency (pattern atteso, strategia) e del profilo di formato (shape dominante, esempi).

```python
if target_dtype == "datetime64[ns]":
    expected_pattern = _build_datetime_expected_pattern(format_facts, expected_pattern)
    suggested_strategy = _augment_datetime_strategy(format_facts, suggested_strategy)
```
Solo per le date: aggiorna il pattern e augmenta la strategia. Per tutti gli altri tipi il request è costruito direttamente senza augmentazione.

---

## cleaning/validation.py — "Python testa la funzione generata"

**Nessun LLM.** Questo modulo esegue la funzione generata contro esempi reali e restituisce una lista di `CleanerValidationIssue`. È il giudice oggettivo del loop.

### Funzioni di supporto

**`dominant_output_shape()`** (righe 30–34)
Trova la shape strutturale più comune tra i valori dominanti (es. `"NNNN-NN-NN"`). Usata per verificare che gli output degli outlier abbiano la stessa struttura.

**`is_parseable_output()`** (righe 37–50)
Verifica che il valore prodotto dal cleaner sia effettivamente parsabile nel dtype target:
- `datetime64[ns]` → prova `pd.to_datetime`
- `Int64` → prova `pd.to_numeric` + controlla che sia intero
- `Float64` → prova `pd.to_numeric`
- `boolean` → controlla se è in `{"true","false","1","0","yes","no","si","sì"}`

**`detect_shadowed_delimiter_branches()`** (righe 213–261)
Analisi **statica** del codice generato (senza eseguirlo). Cerca il pattern anti-bug classico:
```python
if '/' in s:      # branch generico — troppo largo
    ...
elif re.fullmatch(r'\d{2}/\d{4}', s):  # branch specifico — MAI raggiunto
    ...
```
Se trova un branch generico con `'sep' in s` prima di un branch specifico con lo stesso separatore, emette un issue `shadowed_specific_branch`. Questo bug farebbe sì che tutti i valori con `/` finiscano nel branch generico, senza mai raggiungere quello più preciso.

### `validate_generated_cleaner_program()` (righe 264–424) — cuore del validatore

**Input:** `request` (il brief), `program` (la funzione generata dall'agente)

**Output:** `list[CleanerValidationIssue]` — vuota = pass, non vuota = fail

```python
try:
    cleaner = load_cleaner_callable(program)
except Exception as error:
    if isinstance(error, NameError):
        return [build_validation_issue(category="non_self_contained_function", ...)]
    return [build_validation_issue(category="runtime_exception", ...)]
```
Prima prova a **caricare** la funzione in un modulo fresco. Se il codice generato referenzia variabili esterne (es. `dominant`, `request_data`) che non esistono in un modulo pulito, la funzione non si carica e il loop lo segnala come `non_self_contained_function`.

```python
issues.extend(detect_shadowed_delimiter_branches(program))
```
Controllo statico prima ancora di eseguire.

**Test sui valori dominanti** (righe 313–331):
```python
for value in request.dominant_example_values:
    cleaned = cleaner(value)
    if cleaned != value:
        issues.append(build_validation_issue(
            category="dominant_value_modified",
            severity="high",
            ...
        ))
```
I valori dominanti sono già validi — il cleaner non deve toccarli. Se anche uno solo cambia, è un `dominant_value_modified` — errore grave.

**Test sugli outlier** (righe 333–423):
```python
for value in request.example_inconsistent_values:
    cleaned = cleaner(value)
    if cleaned is None:
        continue  # null è accettato per valori irrecuperabili
    if cleaned == value:
        issues.append(...categoria "outlier_unchanged"...)   # non ha trasformato nulla
    if require_fixed_shape and value_shape(cleaned_str) != target_shape:
        issues.append(...categoria "wrong_output_shape"...)  # forma sbagliata
    if not is_parseable_output(cleaned_str, request.target_dtype):
        issues.append(...categoria "not_parseable_as_target_dtype"...)  # non parsabile
    if matches_request_target_pattern(cleaned_str, request) is False:
        issues.append(...categoria "not_matching_target_pattern"...)  # pattern sbagliato
```
Quattro controlli in cascata su ogni outlier. Per le date c'è anche il controllo del formato canonico: l'output deve avere la stessa struttura dei dominanti (es. se i dominanti sono `"2024-03-11"`, l'output non può essere `"11/03/2024"`).

### `rebuild_verified_program()` (righe 427–465)
Chiamata **solo quando il programma passa** tutti i controlli. Riesegue il cleaner sui valori di esempio e aggiorna le `example_transformations` con i risultati reali (non le trasformazioni inventate dall'agente). Questo garantisce che le trasformazioni mostrate nel report finale siano reali.

### Funzioni di utilità per il loop

**`validation_issue_fingerprint()`** (righe 491–495)
```python
return tuple(
    f"{issue.category}|{issue.input_value}|{issue.actual_output}|{issue.expected_behavior}"
    for issue in issues[:limit]
)
```
Produce una "firma" degli errori attuali. Usata in `generation.py` per rilevare la stagnazione: se la firma degli errori al tentativo N è uguale a quella al tentativo N-1, il generatore sta producendo lo stesso codice sbagliato.

---

## cleaning/generation.py — "il loop genera, valida, ripara"

Questo è il regista. Coordina tutti gli altri.

### `_generation_lock()` (righe 73–99)
```python
fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
```
Lock su filesystem con `O_EXCL` — se il file di lock esiste già, fallisce immediatamente invece di aspettare. Impedisce che due processi (es. due run di Streamlit) generino cleaner per lo stesso dataset contemporaneamente e si sovrascrivano.

### `_build_stagnation_unblock_brief()` (righe 161–226)
Quando il generatore produce lo stesso codice sbagliato per due tentativi consecutivi, viene iniettato questo testo nel prompt. Contiene uno **skeleton obbligatorio** con:
1. Canonical guard all'inizio (early-exit per i valori già validi)
2. Branch mutuamente esclusivi per gli outlier

```python
skeleton = (
    "def clean_column(value):\n"
    "    import re\n"
    "    ...\n"
    "    # STEP 1 — CANONICAL GUARD (MANDATORY)\n"
    "    canonical_patterns = [...]\n"
    "    if any(p.fullmatch(s) for p in canonical_patterns):\n"
    "        return s\n"
    "    # STEP 2 — MUTUALLY EXCLUSIVE FORMAT BRANCHES\n"
    "    ...\n"
)
```
L'idea: se il modello è bloccato in un loop deterministico, dargli una struttura obbligatoria lo forza ad abbandonare il controllo di flusso precedente.

### `_build_cleaner_generation_prompt()` (righe 229–298)
Costruisce il prompt per il generatore. La struttura del prompt **cambia** a seconda del contesto:

**Primo tentativo:** solo il `ColumnCleaningRequest` allegato + istruzione di base.

**Tentativi successivi (repair):** aggiunge tre elementi extra:
1. Lista degli errori di validazione host-side (`error_lines`)
2. La funzione precedente fallita (`previous_program.python_code`)
3. La diagnosi del critic (`repair_diagnosis`) — se disponibile
4. Se stagnazione rilevata: lo skeleton obbligatorio

```python
if stagnation_detected:
    prompt.append(attach_text_document(_build_stagnation_unblock_brief(request)))
```

### `run_column_cleaner_program()` (righe 301–426) — il loop per una singola colonna

```python
for attempt in range(1, max_attempts + 1):
```

**Ogni iterazione:**

1. **Rileva stagnazione:**
```python
stagnation_detected = consecutive_stagnant_attempts >= 1
```

2. **Scala la temperatura se stagnante:**
```python
if stagnation_detected:
    bumped_temp = min(0.2 + 0.1 * (consecutive_stagnant_attempts - 1), 0.5)
    override_settings = {"temperature": bumped_temp}
```
Il generatore parte a `temperature=0` (deterministico). Se si blocca, la temperatura sale progressivamente: `0.2` → `0.3` → `0.4` → max `0.5`. Più temperatura = più variabilità nell'output = più probabilità di uscire dal loop.

3. **Chiama il generatore:**
```python
program = run_agent_with_backoff(
    column_cleaner_generator_agent,
    prompt,
    usage_limits=GENERATOR_USAGE_LIMITS,  # max 1 code execution
    model_settings=override_settings,
).output
```
`GENERATOR_USAGE_LIMITS = UsageLimits(tool_calls_limit=1)` — il generatore può eseguire codice **esattamente una volta**. Se prova a farne di più, viene interrotto. Questo previene che il generatore faccia da solo il loop di repair che invece spetta al sistema esterno.

4. **Valida senza LLM:**
```python
validation_issues = validate_generated_cleaner_program(request, program)
if not validation_issues:
    program = rebuild_verified_program(request, program)
    return program  # SUCCESSO
```

5. **Rileva se è stagnante:**
```python
same_code_as_previous = (previous_program is not None and
    program.python_code.strip() == previous_program.python_code.strip())
repeated_failure = (last_issue_fingerprint is not None and
    issue_fingerprint == last_issue_fingerprint)
if same_code_as_previous or repeated_failure:
    consecutive_stagnant_attempts += 1
else:
    consecutive_stagnant_attempts = 0
```

6. **Chiama il critic:**
```python
repair_diagnosis = run_cleaner_repair_critic(dataset_name, request, program, validation_issues)
if not repair_diagnosis.should_retry:
    raise ValueError("Critic advised against retrying...")
```
Se il critic dice `should_retry=False` (il problema è irrisolvibile), il loop si interrompe subito invece di sprecare altri tentativi.

7. **Se finisce i tentativi:**
```python
raise ValueError(f"Cleaner generation failed after {max_attempts} attempts...")
```

### `run_cleaner_generation()` (righe 558–572) — entry point pubblico
```python
with _generation_lock(path):
    return _run_cleaner_generation_locked(...)
```
Acquisisce il lock e delega a `_run_cleaner_generation_locked`.

### `_run_cleaner_generation_locked()` (righe 626–772)
Carica il consistency report, il DataFrame, lo schema map, poi itera su ogni finding:
- `max_workers == 1` → loop sincrono, una colonna alla volta
- `max_workers > 1` → loop asincrono con semaphore (stesso pattern di `consistency.py`)

Per ogni colonna che passa, salva il file `.py` del cleaner e aggiunge un `GeneratedCleanerArtifact` alla lista. Alla fine salva il `cleaner_manifest.json`.

---

## Come si intersecano con agents.py

### `column_cleaner_generator_agent` (agents.py righe 274–421)
```python
column_cleaner_generator_agent = Agent(
    MODEL,
    builtin_tools=[CodeExecutionTool()],  # può eseguire codice
    output_type=PromptedOutput(ColumnCleanerProgram),
    model_settings={"temperature": 0},
    ...
)
```
Ha `CodeExecutionTool` — può eseguire Python durante la risposta. Le istruzioni impongono:
- **Un solo blocco di esecuzione** — il loop di repair è fuori, non dentro
- **Funzione self-contained** — nessuna variabile esterna
- **Canonical guard obbligatorio** — prima logica dopo il check null/empty
- **Branch mutuamente esclusivi** — no `if '-' in s` generico prima di branch specifici

### `cleaner_repair_critic_agent` (agents.py righe 424–494)
```python
cleaner_repair_critic_agent = Agent(
    MODEL,
    output_type=PromptedOutput(CleanerRepairDiagnosis),
    model_settings={"temperature": 0},
    ...
)
```
**Nessun CodeExecutionTool** — non esegue codice, solo ragiona. Le istruzioni dicono:
- Trata gli issue del validator come verità assoluta
- Preferisce `minimal_edit` per bug localizzati, `targeted_rewrite` per bug strutturali
- Se sia `shadowed_specific_branch` che `dominant_value_modified` sono presenti → sono lo stesso bug (canonical guard mancante) → `targeted_rewrite`
- `should_retry=false` solo quando la situazione è irrisolvibile

---

## Schema dell'interazione tra i quattro file

```
consistency finding + format_facts + schema_entry
              │
   [request.py] build_column_cleaning_request()
              │
       ColumnCleaningRequest
              │
   ┌──────────────────────────────────────────┐
   │         LOOP (max 10 tentativi)          │
   │                                          │
   │  [generation.py]                         │
   │  _build_cleaner_generation_prompt()      │
   │         │ prompt con request + contesto  │
   │         ▼                                │
   │  [agents.py] column_cleaner_generator    │
   │         │ ColumnCleanerProgram           │
   │         ▼                                │
   │  [validation.py]                         │
   │  validate_generated_cleaner_program()    │
   │         │                                │
   │    ┌────┴────┐                           │
   │  PASS       FAIL                         │
   │    │          │                          │
   │    │    [agents.py] cleaner_repair_critic│
   │    │          │ CleanerRepairDiagnosis   │
   │    │          │                          │
   │    │    should_retry=False?              │
   │    │          │ Yes → raise ValueError   │
   │    │          │ No  → loop              │
   │    ▼                                     │
   │  [validation.py]                         │
   │  rebuild_verified_program()              │
   └──────────────────────────────────────────┘
              │
      ColumnCleanerProgram validato
              │
      save_generated_cleaner() → file .py
```

---

# 10. cleaning/application.py

Questo modulo è il più "concreto" di tutto il progetto — **nessun agente LLM**, solo pandas che trasforma il DataFrame reale in cinque passi sequenziali e produce il CSV pulito.

---

## Docstring (righe 1–7)

Cinque passi in ordine fisso:
1. Format cleaners (funzioni Python generate dal loop generator/critic)
2. Placeholder → null (da completeness)
3. Drop colonne duplicate esatte (da cross-column)
4. Rinomina colonne (da schema)
5. Cast dei dtype (da schema)

Output finale: `CleaningReport` con un `ColumnCleanerExecutionReport` per ogni cleaner applicato.

---

## Funzioni helper private

### `_apply_column_renames()` (righe 32–63)

**Input:** `df`, `path`
**Output:** `(df rinominato, rename_map dict)`
**Nessun LLM**

```python
handoff = load_schema_handoff(path)
for column in handoff.columns:
    if column.naming_valid or not column.rename_suggestion:
        continue
    target = column.rename_suggestion
    if target in existing and target != column.name:
        suffix = 2
        while f"{target}_{suffix}" in existing:
            suffix += 1
        target = f"{target}_{suffix}"
    rename_map[column.name] = target
df = df.rename(columns=rename_map)
```

Legge lo `SchemaHandoff` dalla cache (prodotto da `schema.py`), costruisce la mappa di rinomina gestendo i conflitti (stesso identico meccanismo di `_planned_rename_map` in `remediation.py`), e applica la rinomina con `df.rename()`.

**Perché viene fatto come step 4 e non step 1?** I cleaner (step 1) lavorano ancora sui **nomi originali** delle colonne — i finding di consistency usano i nomi originali. La rinomina deve avvenire dopo che i cleaner hanno già operato.

---

### `_apply_placeholder_nulls()` (righe 66–96)

**Input:** `df`, `path`
**Output:** `(df con null al posto dei placeholder, totale_rimpiazzi, dict colonna→count)`
**Nessun LLM**

```python
completeness = load_completeness(path)
for column_finding in completeness.per_column:
    placeholders = {
        example.strip().lower()
        for example in column_finding.missing_like_examples
        if example.strip()
    }
    mask = df[column_name].astype(str).str.strip().str.lower().isin(placeholders)
    count = int(mask.sum())
    if count > 0:
        df.loc[mask, column_name] = pd.NA
```

Legge il `CompletenessAnalysisReport` dalla cache. Per ogni colonna che aveva placeholder, costruisce un set di token da sostituire (es. `{"n/a", "null", "-", ""}`) e usa una maschera booleana pandas per sostituire tutte le occorrenze con `pd.NA`.

`.strip().lower()` garantisce che `"N/A "` e `"n/a"` vengano trattati allo stesso modo.

---

### `_coerce_boolean()` (righe 99–105)

**Input:** qualsiasi valore
**Output:** `True`, `False`, o `pd.NA`

```python
normalized = str(value).strip().lower()
if normalized in ("true", "1", "yes", "si", "sì"):
    return True
if normalized in ("false", "0", "no"):
    return False
return pd.NA
```

Helper usato nel cast a `boolean`. Gestisce le varianti italiane (`"si"`, `"sì"`) oltre a quelle standard. Tutto ciò che non è riconoscibile diventa `pd.NA` invece di sollevare un errore.

---

### `_apply_dtype_casts()` (righe 108–164)

**Input:** `df`, `path`
**Output:** `(df con dtype corretti, dict colonna→"applied"/"failed"/"not_needed")`
**Nessun LLM**

```python
handoff = load_schema_handoff(path)
rename_map = {col.name: col.rename_suggestion for col in handoff.columns if ...}
schema_by_current = {}
for column in handoff.columns:
    current_name = rename_map.get(column.name, column.name)  # usa il nome post-rinomina
    if current_name in df.columns:
        schema_by_current[current_name] = column
```

Prima costruisce la mappa dei nomi post-rinomina per sapere come si chiama ogni colonna **nel DataFrame attuale** (che è già stato rinominato allo step 4... aspetta, ma qui siamo allo step 5!). In realtà il cast avviene dopo la rinomina, quindi si deve cercare la colonna con il suo **nuovo nome**.

Per ogni colonna, applica il cast corretto:

| dtype target | Come lo applica |
|---|---|
| `datetime64[ns]` | `pd.to_datetime(..., errors="coerce")` — valori non parsabili → NaT |
| `Int64` | `pd.to_numeric(..., errors="coerce").astype("Int64")` |
| `Float64` | Prima controlla se tutti i valori sono interi → upgradia a `Int64` automaticamente |
| `boolean` | `df.map(_coerce_boolean).astype("boolean")` |
| `string` | `.where(notna, pd.NA).astype("string")` — preserva i null esistenti |

**Nota intelligente sul Float64:** se l'agente ha detto `Float64` ma tutti i valori sono interi (es. `1.0`, `2.0`, `3.0`), il sistema upgradia automaticamente a `Int64` — più corretto e più efficiente.

```python
non_null = numeric.dropna()
if not non_null.empty and (non_null == non_null.round()).all():
    df[column_name] = numeric.astype("Int64")  # upgrade automatico
```

---

### `_load_artifact_program()` (righe 167–178)

```python
def _load_artifact_program(artifact: GeneratedCleanerArtifact) -> ColumnCleanerProgram | None:
    code_path = Path(artifact.code_path)
    if not code_path.exists():
        return None
    return ColumnCleanerProgram(
        python_code=code_path.read_text(encoding="utf-8"),
        ...
    )
```

Carica il file `.py` del cleaner dal disco (salvato da `generation.py`) e lo avvolge in un `ColumnCleanerProgram` per passarlo ad `apply_cleaner_to_series`. Se il file non esiste (es. è stato cancellato), ritorna `None` e il rischio viene registrato in `unresolved_risks`.

---

### `_apply_exact_duplicate_column_drops()` (righe 187–215)

**Input:** `df`, lista di `RemediationAction`
**Output:** `(df senza le colonne droppate, lista di rischi non risolti)`

```python
for action in actions:
    if action.action_type != "drop_exact_duplicate_column" or not action.auto_apply:
        continue
    keep_column = action.target["keep_column"]
    drop_column = action.target["drop_column"]
    df = df.drop(columns=[drop_column])
    action.status = "applied"
```

Itera sulle azioni del piano di remediation, filtra solo quelle di tipo `drop_exact_duplicate_column` con `auto_apply=True`, e droppa le colonne. Aggiorna lo `status` dell'azione nel piano (da `"planned"` a `"applied"`, `"failed"`, o `"not_needed"`).

---

## Funzione principale: `run_cleaner_application_with_plan()` (righe 218–378)

**Input:** `path`, `remediation_plan` (opzionale), `on_event` (callback per UI)

**Output:** `(CleaningReport, list[ColumnCleanerExecutionReport], RemediationPlan aggiornato)`

**Nessun LLM** — trasformazioni pandas pure.

```python
remediation_plan = _clone_remediation_plan(remediation_plan)
```
Fa una copia profonda del piano prima di modificarlo — non vuole alterare l'originale passato dal chiamante.

### Step 1 — Format cleaners (righe 250–303)
```python
for idx, artifact in enumerate(artifacts, start=1):
    program = _load_artifact_program(artifact)
    cleaned_series, report = apply_cleaner_to_series(df[artifact.column_name], program)
    if report.execution_ok and cleaned_series is not None:
        df[artifact.column_name] = cleaned_series
```
Per ogni cleaner nel manifest:
1. Carica il codice dal file `.py`
2. Applica la funzione a tutta la serie pandas (`apply_cleaner_to_series` è in `runtime.py`)
3. Se va bene, sostituisce la colonna nel DataFrame
4. Aggiorna lo status dell'azione corrispondente nel piano

### Step 2 — Placeholder → null (righe 305–314)
```python
df, total_replaced, placeholder_by_column = _apply_placeholder_nulls(df, path)
for action in actions:
    if action.action_type != "replace_placeholders_with_null":
        continue
    action.status = "applied" if placeholder_by_column.get(column_name, 0) > 0 else "not_needed"
```
Applica la sostituzione placeholder e aggiorna il piano. `"not_needed"` se non c'era nulla da sostituire in quella colonna.

### Step 3 — Drop colonne duplicate (righe 316–322)
```python
df, drop_risks = _apply_exact_duplicate_column_drops(df, actions)
```
Droppa le colonne esatte duplicate identificate da cross-column validation.

### Step 4 — Rinomina colonne (righe 324–339)
```python
df, rename_map = _apply_column_renames(df, path)
```
Applica le rinominine da schema. **Deve avvenire dopo i cleaner** (step 1) perché i cleaner usano ancora i nomi originali.

### Step 5 — Cast dtype (righe 341–352)
```python
df, cast_results = _apply_dtype_casts(df, path)
```
Converte ogni colonna al suo dtype corretto. **Deve avvenire per ultimo** perché: i cleaner producono stringhe, la rinomina cambia i nomi, solo alla fine si può fare il cast sicuro ai tipi finali.

### Salvataggio (righe 354–378)
```python
cleaned_path = cleaned_dataset_path(path)
df.to_csv(cleaned_path, index=False)

cleaning_report = CleaningReport(
    rows_before=rows_before,
    rows_after=len(df),
    columns_before=columns_before,
    columns_after=len(df.columns),
    generated_cleaners=applied_artifacts,
    unresolved_risks=unresolved_risks,
    cleaned_csv_gzip_base64=gzip_text_to_base64(df.to_csv(index=False)),
    ...
)
```
Salva il CSV pulito in `Data/.cleaning_cache/<dataset>/<dataset>.cleaned.csv`.

`cleaned_csv_gzip_base64` comprime e codifica in base64 l'intero CSV — usato per passare il file via JSON al report narrativo senza scrivere altri file.

---

## `run_cleaner_application()` (righe 381–383)
```python
def run_cleaner_application(path: Path, on_event=None) -> CleaningReport:
    cleaning_report, _, _ = run_cleaner_application_with_plan(path, on_event=on_event)
    return cleaning_report
```
Wrapper semplificato per chi non ha bisogno degli `execution_reports` e del piano aggiornato — restituisce solo il `CleaningReport`.

---

## Schema dell'ordine degli step e perché

```
DataFrame grezzo (CSV originale)
         │
  Step 1 — Format cleaners
         │  Applicano funzioni Python generate dal loop generator/critic
         │  Lavorano sui nomi ORIGINALI delle colonne
         │  Producono sempre stringhe in output
         │
  Step 2 — Placeholder → null
         │  Sostituisce "N/A", "-", "NULL" con pd.NA
         │  Prima del cast (altrimenti i placeholder bloccherebbero il cast)
         │
  Step 3 — Drop colonne duplicate
         │  Elimina le colonne identiche tenendo la "migliore"
         │  Prima della rinomina (usa ancora i nomi originali dal piano)
         │
  Step 4 — Rinomina colonne
         │  "DataNascita" → "data_nascita"
         │  Prima del cast (il cast usa i nomi post-rinomina)
         │
  Step 5 — Cast dtype
         │  Converte ogni colonna al suo tipo finale
         │  Per ultimo: i dati sono già puliti (cleaner + null + rinomina)
         │
  df.to_csv() → Data/.cleaning_cache/<dataset>/<dataset>.cleaned.csv
```

---

## Perché non usa LLM?

Tutti e cinque gli step sono trasformazioni **deterministiche e reversibili**:
- Le decisioni (cosa rinominare, quali placeholder, quale dtype) sono già state prese dagli agenti nelle fasi precedenti e salvate nelle cache
- `application.py` esegue soltanto — non decide nulla
- L'unico "rischio" che gestisce è che un file di cleaner manchi dal disco o che un cast fallisca — entrambi vengono registrati in `unresolved_risks` senza bloccare il resto

---

# 11. cleaning/verification.py

Questo modulo è il **giudice finale** del cleaning — risponde alla domanda: "il cleaner ha davvero migliorato la situazione?". Nessun LLM, solo confronto numerico.

---

## Docstring (righe 1–7)

Ri-esegue la validazione di consistency sul CSV **pulito**, allinea i finding con quelli originali tenendo conto delle rinominate, e produce un diff per colonna con cinque stati possibili: `resolved`, `improved`, `unchanged`, `regressed`, `new`.

---

## Funzioni helper private

### `_schema_rename_map()` (righe 24–34)
```python
return {
    column.name: column.rename_suggestion
    for column in handoff.columns
    if not column.naming_valid and column.rename_suggestion
}
```
**Input:** `path`
**Output:** `dict[str, str]` — mappa `"DataNascita" → "data_nascita"`

Legge lo `SchemaHandoff` e costruisce la mappa delle rinominate. Serve per allineare i finding: nel CSV originale la colonna si chiamava `"DataNascita"`, nel CSV pulito si chiama `"data_nascita"` — sono la stessa colonna, bisogna riconoscerlo.

---

### `_numeric_original_names()` (righe 37–43)
```python
numeric_columns = {
    column for column in cleaned_df.columns
    if pd.api.types.is_integer_dtype(...) or pd.api.types.is_float_dtype(...)
}
return {reverse_rename.get(column, column) for column in numeric_columns}
```
**Input:** DataFrame pulito + mappa inversa delle rinominate
**Output:** set di nomi originali delle colonne che nel CSV pulito sono diventate numeriche

**Perché esiste:** dopo il cleaning, una colonna come `"anno"` che era stringa è diventata `Int64`. Se rilanci la consistency sul CSV pulito leggendo tutto come stringa (`read_as_str=True`), `"2024"` ha una shape diversa da come era rappresentata prima — potresti vedere false inconsistenze su una colonna che è stata corretta. Questo set viene usato per **filtrare** i finding del dopo che riguardano colonne numeriche, evitando falsi positivi.

---

### `_print_diff_table()` (righe 46–61)
Stampa su stderr una tabella leggibile tipo:

```
  COLUMN            STATUS      BEFORE    AFTER   REDUCTION
  ────────────────  ──────────  ──────    ─────   ─────────
  data_nascita      resolved      4523        0     100.0%
  rata              improved       312       41      86.9%
  codice_fiscale    unchanged       18       18       0.0%
```
Se lo status non è `resolved`, mostra anche campioni dei valori ancora sbagliati.

---

### `_diff_summary()` (righe 64–87)
Costruisce la stringa di summary del report. Esempio:
```
"3 resolved (data_nascita→data_nascita, rata, mese); 1 improved; 2 unchanged"
```
Per le colonne risolte con rinomina, mostra la freccia `"DataNascita→data_nascita"` per chiarezza.

---

## Funzione principale: `run_verify()` (righe 90–183)

**Input:** `path` (al CSV originale), `on_event` (callback UI), `max_workers`

**Output:** `ConsistencyVerificationReport` — lista di `FindingDiff` + summary

**Nessun LLM** — confronto numerico puro.

### Riga 102–106 — controlla che il CSV pulito esista
```python
cleaned_path = cleaned_dataset_path(path)
if not cleaned_path.exists():
    raise FileNotFoundError("Cleaned dataset not found. Run --stage apply first.")
```
La verifica non ha senso senza il CSV pulito — errore esplicito invece di fallire silenziosamente.

### Righe 108–112 — carica i finding originali (PRIMA del cleaning)
```python
original = load_consistency(path)
original_map = {finding.column_name: finding for finding in original.format_consistency_findings}
rename_map = _schema_rename_map(path)
reverse_rename = {new: old for old, new in rename_map.items()}
```
`original_map` è un dict `nome_colonna → finding` per accesso rapido.
`reverse_rename` è la mappa inversa — `"data_nascita" → "DataNascita"` — serve per tradurre i nomi del CSV pulito ai nomi originali.

### Righe 114–115 — rileva colonne numeriche nel CSV pulito
```python
cleaned_df = load_dataset_frame(cleaned_path)
numeric_original_names = _numeric_original_names(cleaned_df, reverse_rename)
```
Identifica quali colonne sono già state castate a numerico nel CSV pulito, per escluderle dai falsi positivi di consistency.

### Righe 119–124 — riesegue consistency sul CSV PULITO
```python
after = run_format_consistency_validation(
    cleaned_path,
    reuse_cache=False,   # SEMPRE fresco, mai dalla cache
    read_as_str=True,    # legge tutto come stringa
    max_workers=max_workers,
)
```
Richiama esattamente la stessa funzione di `validation/consistency.py` ma sul CSV pulito. `reuse_cache=False` è obbligatorio — deve sempre ricalcolare. `read_as_str=True` è importante: il CSV pulito ha colonne già castata a `Int64`, ma la consistency lavora su rappresentazioni stringa dei valori.

### Righe 125–129 — costruisce la mappa dei finding DOPO il cleaning
```python
after_map = {
    reverse_rename.get(finding.column_name, finding.column_name): finding
    for finding in after.format_consistency_findings
    if reverse_rename.get(finding.column_name, finding.column_name) not in numeric_original_names
}
```
Traduce i nomi del CSV pulito ai nomi originali (via `reverse_rename`) e filtra via le colonne numeriche (falsi positivi). Così `after_map` usa gli stessi nomi di `original_map` e i due possono essere confrontati direttamente.

### Righe 131–157 — costruisce il diff per ogni finding originale
```python
for column_name, before_finding in original_map.items():
    after_finding = after_map.get(column_name)
    before_rows = before_finding.inconsistent_rows
    after_rows = after_finding.inconsistent_rows if after_finding else 0
    reduction_pct = round((before_rows - after_rows) / before_rows * 100, 1)

    if after_finding is None or after_rows == 0:
        status = "resolved"    # il problema è sparito del tutto
    elif after_rows < before_rows:
        status = "improved"    # migliorato ma non risolto
    elif after_rows == before_rows:
        status = "unchanged"   # nessun effetto
    else:
        status = "regressed"   # peggiorato (il cleaner ha introdotto nuovi errori)
```

| Status | Condizione | Esempio |
|---|---|---|
| `resolved` | after == 0 o finding scomparso | 4523 → 0 righe inconsistenti |
| `improved` | after < before | 312 → 41 righe |
| `unchanged` | after == before | 18 → 18 righe |
| `regressed` | after > before | 50 → 87 righe (il cleaner ha rotto qualcosa) |

```python
diffs.append(FindingDiff(
    column_name=column_name,
    status=status,
    before_inconsistent_rows=before_rows,
    after_inconsistent_rows=after_rows,
    reduction_pct=reduction_pct,
    remaining_examples=after_finding.example_inconsistent_values if after_finding else [],
    renamed_to=rename_map.get(column_name),  # es. "data_nascita" se rinominata
))
```

### Righe 161–173 — rileva finding completamente nuovi
```python
for column_name, after_finding in after_map.items():
    if column_name in original_map:
        continue  # già gestito sopra
    diffs.append(FindingDiff(
        column_name=column_name,
        status="new",
        before_inconsistent_rows=0,
        after_inconsistent_rows=after_finding.inconsistent_rows,
        reduction_pct=-100.0,  # negativo = peggiorato del 100%
        ...
    ))
```
Se nel CSV pulito appare un finding per una colonna che prima non ne aveva, viene marcato come `"new"` — il cleaner potrebbe aver introdotto un nuovo tipo di inconsistenza in un'altra colonna.

### Righe 177–183 — costruisce e ritorna il report
```python
return ConsistencyVerificationReport(
    dataset_name=path.stem,
    original_finding_count=len(original_map),
    remaining_finding_count=len(after_map),
    diffs=diffs,
    summary=_diff_summary(diffs),
)
```

---

## Schema del flusso

```
Data originale (CSV originale)           CSV pulito (da application.py)
         │                                         │
load_consistency() (cache)               load_dataset_frame()
         │                                         │
   original_map                        run_format_consistency_validation()
(finding prima)                           (reuse_cache=False, read_as_str=True)
         │                                         │
         │                                   after_map
         │                              (finding dopo, nomi tradotti)
         │                                         │
         └──────────── confronto ─────────────────┘
                           │
                  per ogni colonna originale:
                  ├── resolved   (dopo = 0)
                  ├── improved   (dopo < prima)
                  ├── unchanged  (dopo = prima)
                  └── regressed  (dopo > prima)
                           │
                  per ogni finding nuovo:
                  └── new        (non era nel prima)
                           │
              ConsistencyVerificationReport
```

---

## Posizione nel pipeline complessivo

```
validate → generate → apply → VERIFY
                                │
                         Risponde a:
                         "il cleaning ha funzionato?"
                         per ogni colonna, prima e dopo
```

È l'unica fase che **rilancia una validation completa** (`run_format_consistency_validation`) — non si fida del fatto che il cleaner abbia detto di funzionare, lo verifica direttamente sui dati reali. Questo è il motivo per cui `reuse_cache=False` è obbligatorio: deve sempre vedere lo stato attuale del CSV pulito, non una cache vecchia.

---

# 12. cleaning/reporting.py

Questo è l'ultimo modulo del pipeline — raccoglie tutto quello che è stato fatto e produce il report finale in due forme: un JSON strutturato e un documento Markdown narrativo scritto dall'agente.

---

## Docstring (righe 1–7)

Due compiti:
1. `build_final_report` — assembla tutti gli artifact (validation, remediation, cleaning, verification) in un unico `FinalPipelineReport`
2. `generate_narrative_report` — passa quel report all'agente narrativo che produce il Markdown leggibile salvato accanto al CSV pulito

---

## Funzioni semplici

### `save_final_report()` (righe 27–31)
```python
report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
```
Salva il `FinalPipelineReport` come JSON indentato. Nulla di speciale — serializzazione Pydantic standard.

### `narrative_report_path()` + `save_narrative_report()` (righe 137–155)
```python
lines: list[str] = [f"# {report.title}", "", report.executive_summary, ""]
for section in report.sections:
    lines += [f"## {section.heading}", "", section.body, ""]
if report.recommendations:
    lines += ["## Raccomandazioni", ""]
    for i, rec in enumerate(report.recommendations, 1):
        lines.append(f"{i}. {rec}")
output_path.write_text("\n".join(lines), encoding="utf-8")
```
Converte il `NarrativeReport` (struttura Pydantic con `title`, `executive_summary`, `sections`, `recommendations`) in un file Markdown leggibile. L'agente produce JSON strutturato; questo codice lo formatta come `.md`.

---

## `_compute_cleaned_non_null_counts()` (righe 34–55)

**Input:** `dataset_path` (percorso al CSV originale)
**Output:** `(total_rows, dict[colonna → count_non_null])`
**Nessun LLM**

```python
df = load_dataset_frame(cleaned_path)
total = len(df)
counts = {str(col): int(df[col].notna().sum()) for col in df.columns}
return total, counts
```

**Perché esiste:** il report narrativo deve mostrare la colonna "% Non-Null" per ogni cast di dtype. Senza questa funzione, l'agente dovrebbe stimare la percentuale — e inventerebbe numeri. Invece, il sistema legge il CSV pulito reale, conta i valori non nulli per ogni colonna, e passa questi numeri precisi al briefing. L'agente può solo copiare, non può inventare.

---

## `build_final_report()` (righe 58–134)

**Input:** i quattro grandi artifact del pipeline:
- `validation_results` — i sei report di validazione
- `remediation_plan` — il piano con tutte le azioni e i loro status aggiornati
- `cleaning_report` — il report di application (righe cambiate, cleaner applicati)
- `verification_report` — il diff prima/dopo consistency

**Output:** `FinalPipelineReport` — un unico oggetto che contiene tutto

**Nessun LLM** — puro assemblaggio Python.

### Righe 65–72 — conteggio sommario della validation
```python
validation_summary = {
    "schema_issues": len(validation_results.schema_validation.issues),
    "completeness_columns_with_missing": len(...),
    "consistency_findings": len(...),
    "anomaly_findings": len(...),
    "cross_column_findings": len(...),
    "duplicate_groups": len(...),
}
```
Un dict con i conteggi di alto livello per ogni stage — usato nel summary testuale.

### Righe 74–85 — classificazione delle azioni per status
```python
applied_actions     = [a for a in remediation_plan.actions if a.status == "applied"]
proposed_not_applied = [a for a in remediation_plan.actions if a.status == "proposed_not_applied"]
failed_actions      = [a for a in remediation_plan.actions if a.status == "failed"]
not_needed_actions  = [a for a in remediation_plan.actions if a.status == "not_needed"]
```
Filtra le azioni del piano in quattro bucket distinti. Lo status è stato aggiornato da `application.py` durante l'esecuzione. Questa separazione è importante: `not_needed` (azione ridondante) e `proposed_not_applied` (azione deliberatamente non eseguita) sono due cose diverse che il report narrativo deve tenere separate.

### Righe 94–133 — assemblaggio del FinalPipelineReport
Raccoglie tutti i pezzi in un unico oggetto, includendo:
- I conteggi reali dal CSV pulito (`non_null_counts_cleaned`)
- I finding di anomalia, cross-column, duplicati (per il report narrativo)
- I dettagli di completeness per colonna
- Le trasformazioni di esempio dei cleaner (per il report narrativo)
- I diff di verification

---

## `_build_narrative_briefing()` (righe 158–399) — la funzione più importante

**Input:** `FinalPipelineReport`
**Output:** stringa di testo — il briefing passato all'agente narrativo

**Nessun LLM** — costruzione di testo strutturato.

Questa funzione è il cuore del reporting. Il suo scopo è costruire un documento testuale **blindato contro le allucinazioni** dell'agente: ogni sezione del briefing è chiaramente etichettata come "GROUND TRUTH", e le istruzioni nel testo dicono esplicitamente all'agente di non inventare nulla.

### Struttura del briefing

**Intestazione generale:**
```
DATASET: attivazioniCessazioni
VALIDATION SUMMARY: {'schema_issues': 3, 'consistency_findings': 7, ...}
CLEANING SUMMARY: Applied 7 format cleaners, replaced 1240 placeholders...
VERIFICATION SUMMARY: 5 resolved; 1 improved; 1 unchanged
UNRESOLVED RISKS: None
OVERALL: ...
```

**Blocco NON-NULL COUNTS** (righe 188–201):
```
========================================
NON-NULL COUNTS — cleaned CSV (TOTAL ROWS = 125430) — GROUND TRUTH
========================================
Use (non_null_count / TOTAL ROWS * 100)...
  data_nascita: 124891/125430 = 99.6%
  codice_fiscale: 125430/125430 = 100.0%
```
Calcoli già fatti da Python — l'agente li copia e basta.

**Blocco COMPLETENESS DETAILS** (righe 204–216):
```
========================================
COMPLETENESS DETAILS (GROUND TRUTH) — quote verbatim
========================================
  rata: missing_like_count=847, completeness=99.3%, tokens=['N/A', '-', '//']
```

**Blocco ANOMALY FINDINGS** (righe 219–236):
```
========================================
ANOMALY FINDINGS (GROUND TRUTH)
========================================
Do NOT invent columns, row counts, or example values.
  [numeric_outlier] 'importo' — severity=high, affected_rows=23
      examples: '999999', '0', '-1'
      evidence: IQR range [100, 5000]; outliers outside [...]
```

**Blocco CROSS-COLUMN FINDINGS** (righe 240–260):
```
Every similarity %, mismatch count, and column pair MUST come from this block.
  [near_duplicate_columns] data_a & data_b — severity=medium, affected_rows=12450, similarity_pct=99.2%
```

**Blocco DUPLICATE ROW GROUPS** (righe 263–281):
Solo i primi 8 gruppi + conteggio totale.

**Blocco CLEANER EXAMPLE TRANSFORMATIONS** (righe 285–312) — il più critico:
```
========================================
CLEANER EXAMPLE TRANSFORMATIONS (GROUND TRUTH — QUOTE VERBATIM)
========================================
For every 'Clean example: X → Y' line, copy an entry BELOW verbatim.
Do NOT invent outputs, drop characters (e.g. '.000' suffix), or guess.

--- column: data_nascita ---
  '11/03/1985' -> '1985-03-11T00:00:00.000'
  '03-11-85' -> '1985-11-03T00:00:00.000'
  (plus 8 already-valid values preserved unchanged)
```
Le trasformazioni reali vengono da `rebuild_verified_program()` in `validation.py` — sono esecuzioni reali, non invenzioni.

**Blocco APPLIED ACTIONS** (righe 315–330):
Lista dettagliata di ogni azione applicata, con ID, target, motivo e statistiche.

**Blocco NOT NEEDED / FAILED** (righe 332–349):
Separati e distinti dal blocco DEFERRED per evitare che l'agente li confonda.

**Blocco DEFERRED / MANUAL REVIEW** (righe 352–398):
Le azioni `proposed_not_applied` raggruppate per tipo. I `drop_rows_candidate` sono condensati per non occupare tutto il briefing.

---

## `generate_narrative_report()` (righe 402–415) — unica chiamata LLM

**Input:** `FinalPipelineReport`
**Output:** `NarrativeReport` — struttura Pydantic con titolo, summary esecutivo, sezioni, raccomandazioni

**Usa `narrative_report_agent`** — l'unico agente di questo modulo.

```python
briefing = _build_narrative_briefing(final_report)
result = run_agent_with_backoff(
    narrative_report_agent,
    [
        f"Generate an exhaustive narrative quality report for dataset '{final_report.dataset_name}'. "
        "The attached briefing contains all findings, actions, and verification results.",
        attach_text_document(briefing),
    ],
)
return result.output
```

L'agente riceve il briefing come allegato e deve produrre un report Markdown professionale in 10 sezioni obbligatorie (definite nelle istruzioni in `agents.py`). Il briefing è costruito appositamente per essere il più blindato possibile: ogni numero, ogni esempio, ogni valore viene fornito esplicitamente con l'etichetta "GROUND TRUTH" e l'istruzione "copia verbatim".

---

## Schema del flusso complessivo

```
validation_results  +  remediation_plan  +  cleaning_report  +  verification_report
         │                    │                     │                    │
         └────────────────────┴─────────────────────┴────────────────────┘
                                        │
                              build_final_report()
                                        │
                              FinalPipelineReport
                              (tutto in un oggetto)
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
               save_final_report()          _build_narrative_briefing()
               → JSON su disco               → testo strutturato blindato
                                                       │
                                          [LLM] narrative_report_agent
                                                       │
                                              NarrativeReport
                                          (JSON strutturato con sezioni)
                                                       │
                                          save_narrative_report()
                                          → file .md su disco
```

---

## Il principio chiave di questo modulo

L'intero `_build_narrative_briefing()` esiste per risolvere un problema fondamentale dei LLM: tendono ad **allucinare numeri** quando vengono chiesti di scrivere report. La soluzione adottata qui è di non chiedere all'agente di calcolare nulla — Python calcola tutto, l'agente copia e narra. Ogni sezione del briefing è etichettata "GROUND TRUTH" e accompagnata da istruzioni esplicite come "quote verbatim", "do NOT invent", "MUST come from this block". L'agente diventa essenzialmente un formattatore intelligente di dati già calcolati.

---

# 13. cli.py

Questo è il **punto di ingresso** di tutto il sistema — il file che viene eseguito quando scrivi `python main.py ...`. Non contiene logica di business, fa solo una cosa: **legge i parametri dalla riga di comando e smista al modulo giusto**.

---

## Costanti (righe 22–39)

```python
VISIBLE_STAGES = (
    "validate", "dtype", "schema", "completeness",
    "consistency", "remediate", "generate", "apply",
    "verify", "clean", "report",
)

STAGE_ALIASES = {
    "all": "validate",
    "pipeline": "clean",
}
```

`VISIBLE_STAGES` è la lista degli stage validi che l'utente può passare con `--stage`. `STAGE_ALIASES` sono alias per compatibilità: se qualcuno scrive `--stage all` viene trattato come `--stage validate`.

---

## `build_parser()` (righe 42–120) — definisce i parametri CLI

Costruisce il parser `argparse`. Ogni `add_argument` definisce un parametro accettato dalla riga di comando:

| Parametro | Tipo | Default | Cosa fa |
|---|---|---|---|
| `dataset` | posizionale | `Data/spesa.csv` | Percorso al CSV da analizzare |
| `--stage` | stringa | `validate` | Quale stage eseguire |
| `--reuse-schema` | flag | False | Carica schema dalla cache invece di rieseguire |
| `--reuse-completeness` | flag | False | Carica completeness dalla cache |
| `--reuse-consistency` | flag | False | Carica consistency dalla cache |
| `--reuse-validation` | flag | False | Carica tutto il bundle di validation dalla cache |
| `--reuse-remediation` | flag | False | Carica il piano di remediation dalla cache |
| `--verbose` | flag | False | Mostra eventi degli agenti su stderr in tempo reale |
| `--column` | stringa | None | Per `--stage generate`: genera il cleaner solo per questa colonna |
| `--cleaner-attempts` | intero | 10 | Max tentativi del loop generator/critic per colonna |
| `--concurrent-agents` | flag | False | Esegue colonne in parallelo dove supportato |
| `--agent-workers` | intero | 3 | Numero massimo di worker quando `--concurrent-agents` è attivo |

`--consistency-agent` (riga 61) ha `argparse.SUPPRESS` — è un parametro nascosto non mostrato nell'help, usato solo internamente.

---

## `normalize_stage()` (righe 123–129)

```python
def normalize_stage(stage: str) -> str:
    normalized = stage.strip().lower()
    normalized = STAGE_ALIASES.get(normalized, normalized)
    if normalized not in VISIBLE_STAGES:
        raise ValueError(f"Unknown stage {stage!r}. Supported stages: ...")
    return normalized
```

Normalizza lo stage: rimuove spazi, mette in minuscolo, risolve gli alias, e valida che sia uno stage conosciuto. Se non lo è, solleva `ValueError` che `main()` converte in `SystemExit` con un messaggio chiaro.

---

## `print_dtype_inference()` (righe 132–150)

```python
def print_dtype_inference(dataset_path: Path) -> None:
    inference = run_dtype_inference(dataset_path)
    ...
    for column in inference.columns:
        print(f"{column.column_name:<{column_width}}  {column.pandas_dtype:<{dtype_width}}  ...")
```

Stampa la tabella del dtype inference direttamente su stdout, **non come JSON**. È l'unico stage che ha un formato di output speciale — tabella allineata con colonne fisse. Usato per ispezionare rapidamente cosa l'agente ha dedotto per ogni colonna. Dopo la stampa, chiama `raise SystemExit(0)` per uscire pulitamente senza passare da `print_result()`.

---

## `run_validation_bundle()` (righe 153–159)

```python
def run_validation_bundle(args: argparse.Namespace, dataset_path: Path):
    return build_validation_results(
        dataset_path,
        reuse_schema=args.reuse_schema,
        reuse_completeness=args.reuse_completeness,
        reuse_consistency=args.reuse_consistency,
    )
```

Semplice wrapper che passa i flag di cache da `args` a `build_validation_results()` (il bundle di validation che abbiamo visto in `bundle.py`).

---

## `run_narrative_report()` (righe 162–179)

```python
report_path = final_report_path(dataset_path)
if not report_path.exists():
    raise SystemExit("Final report not found. Run --stage clean first.")
final_report = FinalPipelineReport.model_validate_json(report_path.read_text(...))
narrative = generate_narrative_report(final_report)
output_path = save_narrative_report(dataset_path, narrative)
sys.stdout.buffer.write(output_path.read_bytes())
```

Carica il `FinalPipelineReport` JSON dal disco (prodotto da `--stage clean`), chiama l'agente narrativo, salva il `.md`, e poi scrive il file Markdown direttamente su `stdout.buffer` (binario, non testo — per gestire correttamente UTF-8 su tutti i sistemi).

---

## `run_stage()` (righe 182–222) — il dispatcher

Questa è la funzione più importante della CLI. Usa un **dict di funzioni** invece di una lunga catena di `if/elif`:

```python
agent_workers = args.agent_workers if args.concurrent_agents else 1

stage_handlers: dict[str, Callable] = {
    "validate":     run_validation_bundle,
    "schema":       lambda args, path: run_schema_validation(path, ...),
    "completeness": lambda args, path: run_completeness_analysis(path, ...),
    "consistency":  lambda args, path: run_format_consistency_validation(path, ..., max_workers=agent_workers),
    "remediate":    lambda args, path: run_remediation_planning(path, ...),
    "clean":        lambda args, path: run_cleaning(path, ..., cleaner_workers=agent_workers),
    "generate":     lambda args, path: run_cleaner_generation(path, ..., column_name=args.column),
    "apply":        lambda args, path: run_cleaner_application(path),
    "verify":       lambda args, path: run_verify(path, max_workers=agent_workers),
    "report":       run_narrative_report,
}
return stage_handlers[args.stage](args, dataset_path)
```

`agent_workers` vale `args.agent_workers` (default 3) se `--concurrent-agents` è attivo, altrimenti `1` (sequenziale). Viene passato agli stage che supportano il parallelismo (consistency, generate, verify).

Ogni entry del dict è una lambda che adatta la firma `(args, path)` alla firma specifica della funzione del modulo. È più pulito di `if/elif` perché aggiungere un nuovo stage richiede solo una riga nel dict.

---

## `print_result()` (righe 225–243)

```python
def _strip_large_payloads(value):
    if isinstance(value, dict):
        stripped = {}
        for key, item in value.items():
            if key == "cleaned_csv_gzip_base64":
                continue   # rimuove il CSV compresso in base64 dall'output
            ...
    ...
print(json.dumps(dump, ensure_ascii=False, indent=2))
```

Serializza il risultato come JSON indentato su stdout. Prima passa per `_strip_large_payloads` che rimuove ricorsivamente il campo `cleaned_csv_gzip_base64` — quel campo contiene l'intero CSV compresso in base64 e occuperebbe megabyte di output inutile nel terminale. Viene rimosso solo dall'output su schermo, non dal file JSON su disco.

---

## `main()` (righe 246–267) — entry point

```python
def main() -> None:
    load_dotenv()           # carica .env (OPENAI_API_KEY, ecc.)
    parser = build_parser()
    args = parser.parse_args()

    args.stage = normalize_stage(args.stage)  # valida e normalizza

    if args.verbose:
        os.environ["AGENT_VERBOSE"] = "1"  # flag per il logging degli agenti

    if args.agent_workers < 1:
        raise SystemExit("--agent-workers must be at least 1.")

    setup_logfire()  # configura l'osservabilità Logfire

    dataset_path = Path(__file__).parent / args.dataset  # percorso assoluto
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    print_result(run_stage(args, dataset_path))
```

Sequenza di avvio in 7 passi:
1. Carica `.env` con le API key
2. Parsa gli argomenti da riga di comando
3. Valida lo stage
4. Se `--verbose`, imposta la variabile d'ambiente che gli agenti leggono per decidere se fare streaming degli eventi
5. Valida `--agent-workers`
6. Inizializza Logfire (osservabilità)
7. Costruisce il percorso assoluto del dataset, verifica che esista, esegue lo stage e stampa il risultato

---

## Come viene usato — esempi pratici

```bash
# Esegue tutti e sei gli stage di validation
python main.py Data/attivazioniCessazioni.csv --stage validate

# Riusa schema e completeness dalla cache, riesegue solo consistency
python main.py Data/attivazioniCessazioni.csv --stage validate \
    --reuse-schema --reuse-completeness

# Solo dtype inference, output tabellare
python main.py Data/attivazioniCessazioni.csv --stage dtype

# Pipeline completo: validate → generate → apply → verify
python main.py Data/attivazioniCessazioni.csv --stage clean

# Genera cleaner solo per una colonna, max 5 tentativi, in parallelo
python main.py Data/attivazioniCessazioni.csv --stage generate \
    --column "data_nascita" --cleaner-attempts 5

# Tutto in parallelo con 4 worker, output verboso
python main.py Data/attivazioniCessazioni.csv --stage clean \
    --concurrent-agents --agent-workers 4 --verbose
```

---

## Posizione nel sistema

```
terminale → main() → normalize_stage() → run_stage()
                                              │
                         ┌────────────────────┼────────────────────┐
                         │                    │                    │
                    "validate"             "clean"            "generate"
                         │                    │                    │
                  build_validation_     run_cleaning()    run_cleaner_
                    results()          (orchestrator.py)  generation()
                  (bundle.py)
```

`cli.py` non contiene logica — è solo il cancello di ingresso che smista ogni comando al modulo responsabile.

---

# 14. agents.py

## Struttura comune a tutti gli agenti

Prima di tutto, il pattern che si ripete in **ogni** agente:

```python
Agent(
    MODEL,                              # "openai-responses:gpt-4o-mini" per tutti
    name="nome-agente",                 # nome per Logfire/logging
    output_type=PromptedOutput(Model),  # il modello Pydantic che deve restituire
    retries=4,                          # riprova fino a 4 volte se l'output non è JSON valido
    model_settings={"temperature": 0},  # risposta deterministica (no casualità)
    instructions="...",                 # le istruzioni fisse che l'agente legge sempre
)
```

`PromptedOutput` dice a Pydantic AI: "includi automaticamente lo schema JSON del modello nelle istruzioni all'agente, così sa esattamente come deve essere strutturato il suo output". Senza questo, l'agente potrebbe restituire JSON con campi sbagliati.

`retries=4` gestisce i casi in cui l'agente restituisce JSON malformato o con campi mancanti — Pydantic AI riprova automaticamente fino a 4 volte prima di sollevare un errore.

---

## setup_logfire() (righe 28–38)

```python
logfire.configure(...)
logfire.instrument_pydantic_ai()
```

Configura l'osservabilità: ogni chiamata a un agente viene tracciata su Logfire (piattaforma cloud di monitoring). `instrument_pydantic_ai()` aggancia automaticamente tutti gli agenti — non serve strumentare ogni agente manualmente. Chiamata una sola volta da `cli.py:main()` all'avvio.

---

## I quattro "summary agent" — stessa logica, spiego solo il pattern

`schema_summary_agent`, `anomaly_summary_agent`, `cross_column_summary_agent`, `duplicate_summary_agent` seguono tutti lo stesso schema:

**Compito:** ricevono un report già costruito da Python, scrivono solo un testo narrativo leggibile. Non derivano nulla di nuovo.

**Istruzioni tipiche:**
```
"Inspect the provided findings document and write a short, concrete summary.
Return valid JSON only. Do not infer new findings. Do not use markdown."
```

**Dove vengono usati:**

| Agente | Usato in | Quando |
|---|---|---|
| `schema_summary_agent` | `validation/schema.py` | Dopo aver costruito lo `SchemaHandoff` |
| `anomaly_summary_agent` | `validation/anomaly.py` | Dopo aver trovato gli outlier con Python |
| `cross_column_summary_agent` | `validation/cross_column.py` | Dopo i quattro detector euristici |
| `duplicate_summary_agent` | `validation/duplicates.py` | Dopo exact + near duplicate detection |

**Perché esiste questo pattern:** trovare i problemi è compito di Python (deterministico, verificabile). Spiegare i problemi in linguaggio naturale è compito dell'LLM. I due ruoli sono tenuti separati di proposito.

---

## dtype_inference_agent — istruzioni complesse, nessun tool

**Usato in:** `validation/schema.py` → `run_dtype_inference()`

**Compito:** guardare i campioni di valori di ogni colonna e decidere quale dtype pandas la colonna *dovrebbe avere dopo la pulizia* (non come è adesso).

**Perché è complesso:** è l'unico agente che deve fare ragionamento statistico + semantico in un unico passaggio. Le istruzioni definiscono una **gerarchia di priorità** rigida:

```
1. Prima guarda i parse percentages (numeric_parse_pct, datetime_parse_pct)
2. Tratta i valori sporchi come rumore, non come evidenza del tipo
3. Solo dopo scegli il dtype, poi raffina il ruolo semantico e il pattern
4. Il nome della colonna non può MAI sovrascrivere l'evidenza dei dati
```

**Le soglie di evidenza** sono esplicite nelle istruzioni:
```
numeric_parse_pct >= 80 → forte evidenza per Int64/Float64
datetime_parse_pct >= 60 → forte evidenza per datetime64[ns]
```

**Le "hard gates"** bloccano scelte sbagliate comuni:
```
NON scegliere string solo perché i valori sono memorizzati come stringhe
NON scegliere string quando numeric/datetime domina
object solo come ultima risorsa
```

Questo livello di dettaglio nelle istruzioni serve perché GPT-4o-mini tende a scegliere `string` per default su dati sporchi — le regole esplicite correggono questo bias.

---

## completeness_analysis_agent — unico con CodeExecutionTool nella validation

**Usato in:** `validation/completeness.py`

```python
completeness_analysis_agent = Agent(
    MODEL,
    builtin_tools=[CodeExecutionTool()],   # ← può eseguire Python
    ...
)
```

**Perché ha il CodeExecutionTool:** il profilo di completeness allegato è un documento testuale strutturato che può essere lungo. Invece di chiedere all'agente di leggere e interpretare il testo a memoria, gli si dà la possibilità di eseguire Python per ispezionarlo programmaticamente. Le istruzioni dicono esplicitamente:
```
"Always use the code execution tool to inspect the attached completeness profile document."
```

**Compito:** legge il profilo (missing %, placeholder examples, sparse flags), produce un `CompletenessAnalysisReport` strutturato con `recommended_action` per ogni colonna.

**Regola speciale nelle istruzioni:**
```
"If completeness_pct is 100 and missing_like_count is 0,
recommended_action must be exactly 'No action needed'."
```
Questa regola impedisce che l'agente scriva raccomandazioni generiche quando non c'è nulla da fare.

---

## format_consistency_agent — decisione binaria per colonna

**Usato in:** `validation/consistency.py` → slow path (quando lo schema non ha già il pattern)

**Compito:** riceve i `ColumnFormatFacts` di una colonna e decide: c'è un'inconsistenza di formato che una funzione Python potrebbe correggere? Se sì, descrivila precisamente.

**La regola più importante nelle istruzioni:**
```
"suggested_strategy: this is the most important field — the downstream cleaner reads it as its normalization contract.
List every outlier shape group with 2-3 concrete examples and the exact transformation needed.
'shape YYYY-MM (e.g. 2023-09): remove dash, concatenate to YYYYMM' is good.
'normalize dates' is not acceptable."
```

La `suggested_strategy` che questo agente scrive diventa l'input principale del `column_cleaner_generator_agent` tre stage dopo. Se è vaga, il cleaner produrrà codice sbagliato. Per questo le istruzioni insistono tanto sulla concretezza.

**Condizioni per restituire `finding=null`** (nessun problema):
- `machine_format_candidate` è falso
- `dominant_shape_pct` sotto il 70%
- colonne con `string_role` di tipo `name` o `free_text` (variazione di contenuto, non di formato)

---

## column_cleaner_generator_agent — il più complesso

**Usato in:** `cleaning/generation.py` → `run_column_cleaner_program()`

**Caratteristiche uniche:**

```python
column_cleaner_generator_agent = Agent(
    MODEL,
    builtin_tools=[CodeExecutionTool()],   # può eseguire Python
    model_settings={"temperature": 0},     # deterministico di default
    ...
)
```

In `generation.py`, la temperature viene alzata dinamicamente se il generatore è bloccato in stagnazione — ma l'agente parte sempre da 0.

**Il vincolo più critico — un solo blocco di esecuzione:**
```
"Use code execution exactly once for one grouped test over ALL examples.
Do not patch and re-run inside the same model run.
FORBIDDEN: repeated rewrites or a second code-execution call."
```

Questo vincolo esiste perché senza di esso l'agente farebbe il proprio loop interno di repair (genera → testa → corregge → testa di nuovo), bypassando il loop esterno generator/critic che il sistema vuole controllare. `GENERATOR_USAGE_LIMITS = UsageLimits(tool_calls_limit=1)` in `generation.py` lo impone a livello di codice.

**Le due regole strutturali obbligatorie:**

1. **Canonical guard obbligatorio** — prima cosa dopo il check null/empty:
```
"The FIRST logical step MUST be a canonical-pattern early-exit.
Build the guard by deriving a regex from one dominant example.
This guard is NON-NEGOTIABLE for datetime columns."
```
Senza questo, un `if '-' in s:` riscrive anche i valori già corretti.

2. **Branch mutuamente esclusivi:**
```
"Never write `if '<sep>' in s:` above another branch that re-inspects the same separator.
The host validator rejects any program where a generic branch precedes a more specific one."
```

**Perché le istruzioni sono così lunghe:** ogni regola risolve un bug specifico che si è manifestato durante lo sviluppo. La lunghezza delle istruzioni è proporzionale al numero di edge case reali incontrati.

---

## cleaner_repair_critic_agent — ragionamento puro, zero codice

**Usato in:** `cleaning/generation.py` → `run_cleaner_repair_critic()`

```python
cleaner_repair_critic_agent = Agent(
    MODEL,
    # NO CodeExecutionTool ← deliberato
    output_type=PromptedOutput(CleanerRepairDiagnosis),
    ...
)
```

**Compito:** riceve la funzione fallita + la lista di errori del validator Python → produce una diagnosi precisa del bug → il generatore usa la diagnosi nel prossimo tentativo.

**Perché non ha CodeExecutionTool:** il critic non deve scrivere o testare codice. Deve solo ragionare. Dargli il tool lo porterebbe a provare a fixare da solo, sovrapponendosi al generatore.

**Le regole più raffinate nelle istruzioni:**

*Gestione dei fallimenti compositi:*
```
"When BOTH 'shadowed_specific_branch' AND 'dominant_value_modified' are present,
they are usually the same root cause: a generic delimiter branch appears before
the canonical-value guard. root_cause MUST explicitly name BOTH."
```

*Component-order rewrite (il bug più subdolo):*
```
"When the output has the correct delimiters but wrong component order —
e.g. '11/01/2024' becoming '11-01-2024T...' instead of '2024-01-11T...' —
DO NOT say 'change the output format to YYYY-MM-DD' — that phrasing is ambiguous.
MUST state: 'the branch emits components in source order with a new delimiter
instead of reordering them'."
```

Questa regola esiste perché il generatore, ricevendo "cambia il formato in YYYY-MM-DD", lo interpreta come "sostituisci il separatore" invece di "riordina le componenti" — e produce lo stesso bug.

**`should_retry=false`** — il critic può fermare il loop:
```
"Set should_retry=false only when another retry is unlikely to help
because the evidence is contradictory, missing, or underspecified."
```
Se il critic dice stop, `generation.py` solleva `ValueError` immediatamente senza aspettare altri tentativi.

---

## narrative_report_agent — l'unico con temperature > 0

**Usato in:** `cleaning/reporting.py` → `generate_narrative_report()`

```python
narrative_report_agent = Agent(
    MODEL,
    model_settings={"temperature": 0.3},  # ← unico con temperatura > 0
    ...
)
```

**Perché temperature 0.3:** gli altri agenti devono essere deterministici (stessa input → stesso output). L'agente narrativo deve scrivere testo professionale fluido — un po' di variabilità migliora la qualità della prosa senza compromettere la correttezza (che è garantita dal briefing ground truth).

**Compito:** riceve il briefing costruito da `_build_narrative_briefing()` e produce un report Markdown professionale in 10 sezioni obbligatorie.

**La regola anti-allucinazione più importante:**
```
"The % Non-Null value MUST come from the NON-NULL COUNTS block in the briefing.
Copy that percentage verbatim. Never estimate, interpolate, or invent these numbers."
```

E per le trasformazioni dei cleaner:
```
"The 'Clean example' line MUST be copied verbatim from the CLEANER EXAMPLE TRANSFORMATIONS block.
Do NOT reformat the cleaned value. Do NOT trim it to 'look like ISO 8601'."
```

Questo agente è il più esposto al rischio di allucinazione perché deve scrivere numeri (percentuali, conteggi) in un testo narrativo — e i LLM tendono ad "arrotondare" o "stimare" i numeri invece di copiarli esattamente.

---

## Mappa completa: agente → dove viene usato → perché

```
agents.py
   │
   ├── dtype_inference_agent ──────────► schema.py (run_dtype_inference)
   │      Inferisce dtype target da campioni di valori
   │
   ├── schema_summary_agent ───────────► schema.py (run_schema_validation)
   │      Narra lo SchemaHandoff già costruito
   │
   ├── completeness_analysis_agent ────► completeness.py (run_completeness_analysis)
   │      Analizza il profilo di completeness con CodeExecutionTool
   │
   ├── format_consistency_agent ───────► consistency.py (slow path)
   │      Inferisce il pattern dominante quando lo schema non ce l'ha
   │
   ├── anomaly_summary_agent ──────────► anomaly.py (run_anomaly_detection)
   │      Narra i finding già trovati da Python
   │
   ├── cross_column_summary_agent ─────► cross_column.py (run_cross_column_validation)
   │      Narra i finding dei quattro detector euristici
   │
   ├── duplicate_summary_agent ────────► duplicates.py (run_duplicate_detection)
   │      Narra i gruppi di duplicati trovati da Python
   │
   ├── column_cleaner_generator_agent ─► generation.py (run_column_cleaner_program)
   │      Scrive + testa la funzione Python di pulizia (1 esecuzione di codice)
   │
   ├── cleaner_repair_critic_agent ────► generation.py (run_cleaner_repair_critic)
   │      Diagnosi del bug quando il generatore fallisce (nessun codice)
   │
   └── narrative_report_agent ─────────► reporting.py (generate_narrative_report)
          Scrive il report Markdown finale in 10 sezioni
```

---

# 15. models.py

`models.py` è il **dizionario comune** di tutto il sistema — definisce la forma esatta di ogni dato che passa tra i moduli, tra gli agenti, e verso il disco. Nessuna logica qui, solo strutture dati.

---

## Cos'è un modello Pydantic

```python
class SchemaIssue(BaseModel):
    column_name: str
    severity: str = Field(description="Use low, medium, or high.")
```

Ogni classe che estende `BaseModel` è un **contenitore di dati validato**. Pydantic controlla automaticamente che i tipi siano corretti, che i campi obbligatori ci siano, e che i vincoli (`ge=0`, `le=100`) siano rispettati. Se un agente restituisce JSON con un campo sbagliato, Pydantic solleva errore prima che il dato entri nel sistema.

---

## Type Literals (righe 12–32)

```python
VALID_PANDAS_DTYPE = Literal["Int64", "Float64", "datetime64[ns]", "string", "boolean", "object"]
NUMERIC_ROLE = Literal["measure", "code", "indicator"]
STRING_ROLE = Literal["identifier", "categorical", "name", "free_text"]
```

`Literal` significa "solo questi valori esatti sono ammessi". Se un agente restituisce `"int64"` invece di `"Int64"`, Pydantic lo rifiuta. Queste costanti sono usate come tipo in più modelli per garantire coerenza in tutto il sistema.

---

## Gruppo Schema (righe 37–97)

### `SchemaIssue`
Un singolo problema trovato nello schema. Usato in `schema.py:build_schema_issues()`.

```python
issue_type: str   # "naming_standard" o "duplicate_column_semantics"
severity: str     # "low", "medium", "high"
fix_confidence: str
suggested_fix: str
suggested_strategy: str
```

### `SchemaColumnEntry`
**Il modello più denso del sistema** — fonde in un unico oggetto tutto quello che si sa di una colonna: dtype inferito dall'agente + statistiche Python + validazione naming.

```python
name: str
pandas_dtype: str                   # dall'agente
numeric_role: NUMERIC_ROLE | None   # dall'agente
string_role: STRING_ROLE | None     # dall'agente
detected_pattern: str | None        # dall'agente
non_null_rows: int = Field(ge=0)    # da Python (ge=0 = valore >= 0)
distinct_non_null_values: int       # da Python
numeric_parse_pct: float            # da Python (0-100)
datetime_parse_pct: float           # da Python (0-100)
sample_values: list[str]            # da Python
naming_valid: bool                  # da Python
rename_suggestion: str | None       # da Python
```

Costruita in `schema.py:run_schema_validation()` fondendo `dtype_map` (agente) + `col_profile` (Python).

### `SchemaHandoff`
Il risultato completo della fase schema. "Handoff" perché è ciò che viene passato agli stage successivi.

```python
columns: list[SchemaColumnEntry]
issues: list[SchemaIssue]
duplicate_groups: list[SchemaDuplicateGroup]
summary: str = ""   # default vuoto — viene riempito dall'agente summary
```

Salvato come JSON in `Data/.validation_cache/<dataset>.schema.json`.

### Quattro "summary output" (righe 83–97)
```python
class SchemaSummaryOutput(BaseModel):
    summary: str

class AnomalySummaryOutput(BaseModel):
    summary: str
# ... stessi per CrossColumn e Duplicate
```
Modelli minimi — un solo campo. Sono l'output strutturato dei quattro summary agent. Il motivo per cui sono modelli Pydantic invece di semplici stringhe: `PromptedOutput` richiede un `BaseModel`, e Pydantic valida che l'agente abbia davvero restituito il JSON corretto.

### `ColumnDtypeInference` + `DatasetDtypeInference`
L'output del `dtype_inference_agent` per una singola colonna e per l'intero dataset.

```python
class ColumnDtypeInference(BaseModel):
    column_name: str
    pandas_dtype: VALID_PANDAS_DTYPE  # validato dal Literal
    numeric_role: NUMERIC_ROLE | None
    string_role: STRING_ROLE | None
    detected_pattern: str | None
    rationale: str

class DatasetDtypeInference(BaseModel):
    columns: list[ColumnDtypeInference]
```

---

## Gruppo Completeness (righe 129–148)

### `CompletenessColumnFinding`
Un finding per singola colonna dal `completeness_analysis_agent`.

```python
completeness_pct: float    # % valori non null
missing_like_count: int    # quanti valori sono "null-like"
missing_like_examples: list[str]  # es. ["N/A", "-", "NULL"]
sparse_candidate: bool     # quasi completamente vuota?
recommended_action: str    # "No action needed" o istruzione concreta
```

### `CompletenessAnalysisReport`
Il report completo — contiene `per_column` (una lista di `CompletenessColumnFinding`) più aggregati globali.

```python
columns_with_missing_values: list[str]  # nomi colonne con problemi
sparse_columns: list[str]               # nomi colonne quasi vuote
placeholder_values_detected: list[str]  # tutti i token trovati
per_column: list[CompletenessColumnFinding]
```

---

## Gruppo Consistency (righe 152–188)

### `FormatConsistencyFinding`
Un problema di formato per una colonna — il dato chiave per il cleaning.

```python
column_name: str
expected_pattern: str          # es. "YYYY-MM-DD"
inconsistent_rows: int         # quante righe non rispettano il pattern
example_inconsistent_values: list[str]  # campioni di valori sbagliati
evidence: str                  # spiegazione testuale
suggested_strategy: str        # istruzioni per il cleaner agent
```

`suggested_strategy` è il campo più critico: costruito da `_build_suggested_strategy()` in `consistency.py` (fast path) o dall'agente (slow path), viene poi letto dal `column_cleaner_generator_agent`.

### `ColumnConsistencyReport`
Wrapper che include il finding (o `None` se non c'è problema) + un summary testuale. Usato per i risultati per-colonna prima di aggregarli nel report finale.

```python
finding: FormatConsistencyFinding | None = None  # None = nessun problema
summary: str
```

### `FindingDiff`
Il confronto prima/dopo usato da `verification.py`.

```python
status: Literal["resolved", "improved", "unchanged", "regressed", "new"]
before_inconsistent_rows: int
after_inconsistent_rows: int
reduction_pct: float = Field(ge=-100)  # negativo se regressed
remaining_examples: list[str]
renamed_to: str | None  # se la colonna è stata rinominata
```

`reduction_pct` può essere negativo (`ge=-100` invece di `ge=0`) perché un `regressed` significa che il numero di righe problematiche è aumentato.

---

## Gruppo Anomaly (righe 193–209)

### `AnomalyFinding`
Un outlier o categoria rara trovata dagli euristici Python.

```python
anomaly_type: Literal["numeric_outlier", "rare_category"]
severity: Literal["low", "medium", "high"]
affected_rows: int
example_values: list[str]
evidence: str         # es. "IQR range [100, 5000]"
suggested_action: str # sempre manual_review — non si auto-corregge
```

---

## Gruppo Cross-Column (righe 213–234)

### `CrossColumnFinding`
Differisce dagli altri finding perché coinvolge **più colonne**.

```python
columns: list[str]   # ← lista, non singolo column_name
check_type: Literal[
    "duplicate_semantic_conflict",
    "exact_duplicate_columns",
    "near_duplicate_columns",
    "year_month_period_mismatch",
    "date_order_violation",
]
similarity_pct: float | None  # presente solo per exact/near duplicate
example_row_indices: list[int]  # indici di righe esempio
```

---

## Gruppo Remediation (righe 256–313)

### I type literal di remediation

```python
REMEDIATION_ACTION_TYPE = Literal[
    "rename_column", "replace_placeholders_with_null",
    "generate_cleaner", "drop_exact_duplicate_column",
    "cast_dtype", "manual_review", "report_only", "drop_rows_candidate",
]
REMEDIATION_STATUS = Literal[
    "planned",              # creata da remediation.py, non ancora eseguita
    "applied",              # eseguita da application.py
    "proposed_not_applied", # non eseguita — richiede giudizio umano
    "failed",               # tentata ma fallita
    "not_needed",           # ridondante, già soddisfatta
]
```

### `RemediationAction`
L'unità atomica del piano — un'azione da eseguire o proporre.

```python
action_id: str                    # es. "rename_column__data_nascita__data_nascita"
action_type: REMEDIATION_ACTION_TYPE
object_type: REMEDIATION_OBJECT_TYPE  # "column", "column_pair", "row_group"
target: dict[str, Any]           # flessibile: {"column_name": ..., "new_name": ...}
source_check: str                 # da quale stage viene (es. "schema_validation")
confidence: REMEDIATION_CONFIDENCE
risk_level: REMEDIATION_RISK_LEVEL
auto_apply: bool                  # True = eseguita da application.py
status: REMEDIATION_STATUS        # aggiornato durante l'esecuzione
reason: str
preview_stats: dict[str, Any]     # statistiche contestuali
```

`target` è `dict[str, Any]` (flessibile) invece di un tipo fisso perché ogni `action_type` ha target diversi: `rename_column` ha `{"column_name": ..., "new_name": ...}`, `drop_rows_candidate` ha `{"row_indices": [...]}`, ecc.

### `FinalPipelineReport`
Il modello più grande del sistema — aggrega tutto per il report narrativo.

```python
applied_actions: list[RemediationAction]
proposed_not_applied_actions: list[RemediationAction]
failed_actions: list[RemediationAction]
not_needed_actions: list[RemediationAction]
verification_diffs: list[FindingDiff]
generated_cleaners: list[GeneratedCleanerArtifact]
non_null_counts_cleaned: dict[str, int]  # per il % Non-Null nel report
completeness_details: list[CompletenessColumnFinding]
anomaly_findings: list[AnomalyFinding]
cross_column_findings: list[CrossColumnFinding]
duplicate_groups: list[DuplicateRecordGroup]
```

È essenzialmente una "fotografia" dello stato finale del pipeline, costruita da `reporting.py:build_final_report()` e passata al `narrative_report_agent`.

---

## Gruppo Cleaning (righe 350–506)

### `ColumnCleaningRequest`
Il "brief" passato al generatore — tutto quello che serve per scrivere il cleaner.

```python
expected_pattern: str              # es. "YYYY-MM-DD"
semantic_hint: str                 # contesto semantico della colonna
target_dtype: str | None           # dtype target (da schema)
dominant_shape: str | None         # shape strutturale dominante
dominant_example_values: list[str] # valori già validi (da preservare)
example_inconsistent_values: list[str]  # valori da correggere
suggested_strategy: str            # istruzioni dettagliate per il cleaner
```

### `ExampleTransformation`
Una singola trasformazione di esempio — usata nei report e nel `ColumnCleanerProgram`.

```python
original_value: str
cleaned_value: str | None  # None = valore irrecuperabile → null
rationale: str

@field_validator("original_value", "cleaned_value", mode="before")
@classmethod
def coerce_to_str(cls, v):
    if v is None:
        return None
    return str(v)
```

Il `field_validator` è l'unico validator custom del file. Converte qualsiasi valore (int, float) in stringa prima della validazione — necessario perché l'agente a volte restituisce `cleaned_value: 2024` (intero) invece di `"2024"` (stringa). Senza questo validator, Pydantic rifiuterebbe il JSON.

### `ColumnCleanerProgram`
L'output del generatore — il codice della funzione di pulizia.

```python
column_name: str
function_name: str
python_code: str = Field(
    description=(
        "Pure Python source code containing exactly one function definition. "
        "Must be importable as-is: no test code, no print statements..."
    )
)
example_transformations: list[ExampleTransformation]
verification_summary: str
residual_risks: list[str]
```

La `description` del campo `python_code` è letta da `PromptedOutput` e inclusa nello schema JSON che l'agente riceve — quindi funziona anche come istruzione all'agente su cosa deve contenere quel campo.

### `VALIDATION_FAILURE_CATEGORY`
```python
VALIDATION_FAILURE_CATEGORY = Literal[
    "program_mismatch",
    "non_self_contained_function",  # referenzia variabili esterne
    "runtime_exception",            # crasha durante l'esecuzione
    "shadowed_specific_branch",     # branch generico prima di specifico
    "dominant_value_modified",      # ha modificato un valore già valido
    "outlier_unchanged",            # non ha trasformato un outlier
    "wrong_output_shape",           # output ha struttura sbagliata
    "not_parseable_as_target_dtype",# output non parsabile nel tipo target
    "not_matching_target_pattern",  # output non rispetta il pattern
]
```

Ogni categoria corrisponde a un controllo specifico in `cleaning/validation.py:validate_generated_cleaner_program()`. Il critic in `agents.py` legge queste categorie per decidere il tipo di diagnosi.

### `CleanerRepairDiagnosis`
L'output del critic agent — la diagnosi del bug.

```python
should_retry: bool = True          # False = ferma il loop
primary_category: VALIDATION_FAILURE_CATEGORY
root_cause: str                    # spiegazione del bug
bug_location: str                  # dove nel codice (es. "date-parsing branch")
planned_fix: str                   # istruzioni concrete per il generatore
patch_style: Literal["minimal_edit", "targeted_rewrite"]
exact_repairs: list[CleanerRepairExample]  # input→output atteso, cosa cambiare
confidence: Literal["low", "medium", "high"]
```

### `ColumnCleanerExecutionReport`
Prodotto da `cleaning/runtime.py` quando il cleaner viene applicato al dataset reale.

```python
execution_ok: bool = True       # False se crasha su dati reali
changed_rows: int               # quante righe sono state modificate
sample_updates: list[CellUpdate]  # campioni di celle cambiate
unresolved_risks: list[str]
```

Distinto da `ColumnCleanerProgram` (il codice generato) — questo è il **risultato dell'applicazione** del codice sui dati reali.

---

## Gruppo Orchestration (righe 486–506)

### `OrchestrationStepResult`
Il bundle dei sei report di validation — passato da `bundle.py` a `remediation.py` e poi a `reporting.py`.

```python
schema_validation: SchemaHandoff           # obbligatorio
completeness_analysis: CompletenessAnalysisReport  # obbligatorio
consistency_validation: ConsistencyValidationReport  # obbligatorio
anomaly_detection: AnomalyDetectionReport | None = None      # opzionale
cross_column_validation: CrossColumnValidationReport | None = None  # opzionale
duplicate_detection: DuplicateDetectionReport | None = None  # opzionale
```

I tre stage opzionali (`anomaly`, `cross_column`, `duplicate`) hanno default `None` — il sistema può girare anche senza di loro (es. `--stage validate` senza gli stage avanzati).

### `CleaningPipelineResult`
Il risultato completo dell'intero pipeline clean — dal CSV originale al report finale.

```python
source_path: str
cleaned_path: str
validation_results: OrchestrationStepResult
remediation_plan: RemediationPlan | None
generated_programs: list[ColumnCleanerProgram]
execution_reports: list[ColumnCleanerExecutionReport]
cleaning_report: CleaningReport
verification_report: ConsistencyVerificationReport | None
final_report: FinalPipelineReport | None
```

---

## Mappa modello → dove viene usato

```
models.py
   │
   ├── SchemaColumnEntry ──────────────► schema.py (prodotto), consistency.py (letto),
   │                                     remediation.py (letto), anomaly.py (letto)
   │
   ├── SchemaHandoff ──────────────────► schema.py (prodotto + cache JSON)
   │                                     → consistency.py, anomaly.py, cross_column.py,
   │                                       duplicates.py, remediation.py, application.py
   │
   ├── CompletenessAnalysisReport ─────► completeness.py (prodotto + cache JSON)
   │                                     → application.py (per placeholder replacement)
   │
   ├── FormatConsistencyFinding ───────► consistency.py (prodotto)
   │                                     → generation.py (per build_column_cleaning_request)
   │
   ├── ColumnCleaningRequest ──────────► request.py (prodotto)
   │                                     → generation.py (passato al generator agent)
   │
   ├── ColumnCleanerProgram ───────────► generation.py (output del generator agent)
   │                                     → validation.py (validato), application.py (eseguito)
   │
   ├── CleanerRepairDiagnosis ─────────► generation.py (output del critic agent)
   │                                     → _build_cleaner_generation_prompt() (usato nel retry)
   │
   ├── RemediationAction ──────────────► remediation.py (prodotto)
   │                                     → application.py (eseguito + status aggiornato)
   │
   ├── FinalPipelineReport ────────────► reporting.py:build_final_report() (prodotto)
   │                                     → narrative_report_agent (input)
   │
   └── OrchestrationStepResult ────────► bundle.py (prodotto)
                                          → remediation.py, reporting.py (input)
```

---

