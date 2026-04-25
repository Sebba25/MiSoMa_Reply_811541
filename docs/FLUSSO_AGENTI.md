# Flusso degli Agenti — Chi Riceve Cosa e Cosa Produce

Questo documento segue il dato dall'inizio alla fine.
Per ogni passaggio: cosa entra, chi lo elabora, cosa esce, dove viene salvato.

---

## La Mappa Completa del Flusso

```
FILE CSV ORIGINALE
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                  FASE 1: VALIDA                     │
│                                                     │
│  Step 1 → dtype_inference_agent                     │
│  Step 2 → schema_summary_agent                      │
│  Step 3 → completeness_analysis_agent               │
│  Step 4 → format_consistency_agent  (per colonna)   │
│  Step 5 → anomaly_summary_agent                     │
│  Step 6 → cross_column_summary_agent                │
│  Step 7 → duplicate_summary_agent                   │
│                                                     │
│  Output finale → validation_bundle.json             │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                 FASE 2: REMEDITA                    │
│                                                     │
│  Nessun agente AI. Solo logica deterministica.      │
│  Legge il bundle → costruisce il piano d'azione     │
│                                                     │
│  Output → remediation_plan.json                     │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                  FASE 3: PULISCI                    │
│                                                     │
│  Step 8 → column_cleaner_generator_agent            │
│            + validatore interno (no AI)             │
│            + cleaner_repair_critic_agent            │
│           (loop per ogni colonna inconsistente)     │
│                                                     │
│  Step 9 → Applica tutto al CSV (no AI)              │
│                                                     │
│  Output → dataset.cleaned.csv                       │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                 FASE 4: VERIFICA                    │
│                                                     │
│  Nessun agente AI. Confronto automatico.            │
│  Rilegge il CSV pulito → confronta con il prima     │
│                                                     │
│  Output → ConsistencyVerificationReport (in memoria)│
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                  FASE 5: REPORT                     │
│                                                     │
│  Step 10 → narrative_report_agent                   │
│                                                     │
│  Output → final_report.json + report.md             │
└─────────────────────────────────────────────────────┘
```

---

## FASE 1 — VALIDA

---

### Step 1 — dtype_inference_agent

**File coinvolto:** `validation/schema.py`

**Cosa riceve:**
Un campione del CSV in formato testo — le prime righe di ogni colonna, con i valori reali.
Viene preparato dalla funzione `build_dtype_inference_text()` in `tools/schema_tools.py`.

**Cosa fa:**
Legge il campione e per ogni colonna decide:
- Che tipo di dato è (numero intero, numero decimale, data, testo, vero/falso)
- Qual è il ruolo semantico (es. "identificativo", "importo monetario", "anno")
- Se c'è un pattern riconoscibile (es. "YYYYMM", "DD/MM/YYYY")

**Cosa produce:**
Un oggetto `DatasetDtypeInference` — una lista con una riga per ogni colonna del CSV.
Non viene salvato su file. Passa direttamente allo step successivo.

---

### Step 2 — schema_summary_agent

**File coinvolto:** `validation/schema.py`

**Cosa riceve:**
- Il risultato dello Step 1 (`DatasetDtypeInference`)
- Il profilo del dataset costruito da `build_dataset_profile()` in `tools/schema_tools.py`
  (contiene statistiche per colonna: quanti valori unici, quanti null, esempi rappresentativi)

**Cosa fa:**
Legge il profilo e il risultato dell'inferenza e produce un riassunto dei problemi di schema:
- Quali colonne hanno nomi non validi (spazi, maiuscole, caratteri speciali)
- Quali colonne dovrebbero essere rinominate e come
- Quali colonne sono probabilmente duplicate (stesso contenuto, nome diverso)
- Quale tipo di dato definitivo assegnare a ogni colonna

**Cosa produce:**
Un oggetto `SchemaSummaryOutput`.

**Salvato in:**
`Data/.validation_cache/<nome_dataset>.schema.json`

---

### Step 3 — completeness_analysis_agent

**File coinvolto:** `validation/completeness.py`

**Cosa riceve:**
Il profilo di completezza del dataset, costruito da `build_completeness_profile()` in `tools/completeness_tools.py`.
Questo profilo contiene, per ogni colonna:
- Quante celle sono vuote davvero (null)
- Quante celle contengono parole false come "N/D", "—", "NULL", "n.d.", ecc.
- Esempi di questi valori falsi trovati nel dataset

**Cosa fa:**
Analizza il profilo e decide:
- Quali colonne hanno un problema di completezza reale
- Quali token specifici vanno sostituiti con il vero valore nullo
- Quanto è grave il problema (percentuale di celle affette)

**Cosa produce:**
Un oggetto `CompletenessAnalysisReport` con una lista di `CompletenessFinding`.

**Salvato in:**
`Data/.validation_cache/<nome_dataset>.completeness.json`

---

### Step 4 — format_consistency_agent (uno per ogni colonna)

**File coinvolto:** `validation/consistency.py`

Questo step viene eseguito **una volta per ogni colonna** del dataset.
Per ogni colonna, il sistema sceglie tra due percorsi.

**Percorso rapido (senza AI):**
Se lo Step 1 ha già rilevato un pattern preciso per quella colonna (es. "YYYY-MM-DD"),
il sistema confronta direttamente ogni cella con quel pattern.
Non chiama nessun agente. È veloce.

**Percorso lento (con AI):**
Se non c'è un pattern riconosciuto, il sistema prepara i "fatti di formato" della colonna
usando `build_column_format_facts()` in `tools/format_tools.py`.
Questi fatti includono:
- La forma più comune dei valori (es. `9999-99-99` → struttura di una data)
- La percentuale di celle che seguono quella forma
- Gli esempi di celle che non seguono la forma dominante (le "inconsistenti")

Poi chiama il `format_consistency_agent`.

**Cosa riceve il format_consistency_agent:**
I "fatti di formato" della colonna (`ColumnFormatFacts`).

**Cosa fa:**
Decide se esiste davvero un'inconsistenza, di che tipo è, e quale strategia usare per correggerla.

**Cosa produce:**
Un oggetto `ColumnConsistencyReport` per ogni colonna analizzata.

**Salvato in:**
`Data/.validation_cache/<nome_dataset>.consistency.json`

---

### Step 5 — anomaly_summary_agent

**File coinvolto:** `validation/anomaly.py`

**Cosa riceve:**
I risultati dei rilevatori automatici in `tools/quality_tools.py`:
- `detect_numeric_outlier_candidates()` — trova numeri molto lontani dalla media (metodo IQR)
- `detect_rare_category_candidates()` — trova categorie che compaiono pochissime volte

**Cosa fa:**
Legge questi risultati e produce un riassunto leggibile: quali anomalie sono probabilmente errori,
quali potrebbero essere dati legittimi, quali meritano attenzione.

**Cosa produce:**
Un oggetto `AnomalySummaryOutput`.

**Salvato in:**
`Data/.validation_cache/<nome_dataset>.anomaly.json`

---

### Step 6 — cross_column_summary_agent

**File coinvolto:** `validation/cross_column.py`

**Cosa riceve:**
I risultati di quattro rilevatori automatici in `tools/quality_tools.py`:
- `detect_duplicate_like_columns()` — trova colonne con contenuto quasi identico
- `detect_duplicate_semantic_conflicts()` — trova colonne simili ma con valori in conflitto
- `detect_year_month_period_mismatches()` — trova incoerenze tra anno, mese e periodo
- `detect_date_order_violations()` — trova date in ordine sbagliato (es. fine prima dell'inizio)

**Cosa fa:**
Riassume questi problemi in linguaggio comprensibile, indicando quali sono i più critici
e quali richiedono revisione manuale.

**Cosa produce:**
Un oggetto `CrossColumnSummaryOutput`.

**Salvato in:**
`Data/.validation_cache/<nome_dataset>.cross_column.json`

---

### Step 7 — duplicate_summary_agent

**File coinvolto:** `validation/duplicates.py`

**Cosa riceve:**
I risultati di tre rilevatori automatici in `tools/quality_tools.py`:
- `detect_exact_duplicate_groups()` — trova righe identiche al 100%
- `detect_near_duplicate_groups()` — trova righe quasi identiche (stessi valori nelle colonne chiave)
- `infer_duplicate_key_columns()` — identifica quali colonne definiscono "unicità" nel dataset

**Cosa fa:**
Riassume quanti duplicati ci sono, quali righe sono coinvolte, e quanto è grave il problema.

**Cosa produce:**
Un oggetto `DuplicateSummaryOutput`.

**Salvato in:**
`Data/.validation_cache/<nome_dataset>.duplicates.json`

---

### Fine della Fase 1 — Il Bundle

**File coinvolto:** `validation/bundle.py`

Dopo tutti e sette gli step, la funzione `build_validation_results()` raccoglie
tutti i risultati in un unico oggetto e lo salva.

**Salvato in:**
`Data/.validation_cache/<nome_dataset>.validation_bundle.json`

Questo file è il punto di partenza di tutte le fasi successive.

---

## FASE 2 — REMEDITA

**File coinvolto:** `cleaning/remediation.py`

**Nessun agente AI in questa fase.**

**Cosa riceve:**
Il `validation_bundle.json` prodotto dalla Fase 1.

**Cosa fa:**
Scorre ogni finding e costruisce un'azione corrispondente seguendo regole fisse:

| Se trova questo... | Crea questa azione | Automatica? |
|---|---|---|
| Nome colonna non valido | `rename_column` | Sì |
| Placeholder falsi (N/D, ecc.) | `replace_placeholders_with_null` | Sì |
| Colonna con formato inconsistente | `generate_cleaner` | Sì |
| Colonna duplicata esatta | `drop_exact_duplicate_column` | Sì |
| Tipo di dato da correggere | `cast_dtype` | Sì |
| Colonna quasi duplicata | `manual_review` | No |
| Conflitto semantico tra colonne | `manual_review` | No |
| Anomalia numerica | `manual_review` | No |
| Righe duplicate | `drop_rows_candidate` | No |
| Data in ordine sbagliato | `manual_review` | No |

Per le colonne duplicate esatte, sceglie quale tenere con questa priorità:
1. Preferisce il nome valido rispetto al nome non valido
2. A parità, preferisce la colonna con più valori non nulli
3. A parità, preferisce la colonna con il nome più canonico
4. A parità, usa l'ordine originale delle colonne

**Cosa produce:**
Un oggetto `RemediationPlan` con la lista completa di tutte le azioni.

**Salvato in:**
`Data/.validation_cache/<nome_dataset>.remediation_plan.json`

---

## FASE 3 — PULISCI

---

### Step 8 — Il Loop Generator/Critic (per ogni colonna inconsistente)

**File coinvolti:** `cleaning/generation.py`, `cleaning/validation.py`, `agents.py`

Questo step si ripete per ogni colonna che ha una finding di consistenza.

#### 8a — Preparazione della richiesta

**File coinvolto:** `cleaning/request.py`

Prima di chiamare qualsiasi agente, il sistema prepara un oggetto `ColumnCleaningRequest`
che descrive esattamente il problema di quella colonna.
Contiene:
- Il nome della colonna
- Il tipo di dato atteso
- Il formato dominante (quello corretto)
- Gli esempi di valori corretti (che non devono essere toccati)
- Gli esempi di valori sbagliati (che devono essere corretti)
- La strategia suggerita per correggere ogni tipo di eccezione

#### 8b — column_cleaner_generator_agent

**Cosa riceve:**
La `ColumnCleaningRequest` della colonna.

**Cosa fa:**
Scrive una funzione Python autonoma che, dato un valore di cella, lo corregge.
Può fare UNA sola chiamata a strumenti esterni (un'esecuzione di codice) — poi deve consegnare.
Questo impedisce che l'agente si auto-corregga internamente senza essere monitorato.

**Cosa produce:**
Un oggetto `ColumnCleanerProgram` — il codice della funzione + metadati.

#### 8c — Il Validatore Interno (senza AI)

**File coinvolto:** `cleaning/validation.py`

Non è un agente AI. È codice normale che testa la funzione prodotta.

**Cosa riceve:**
La funzione generata (`ColumnCleanerProgram`) + la `ColumnCleaningRequest`.

**Cosa fa:**
Esegue la funzione su ogni esempio di valore corretto e sbagliato e controlla:
- I valori già corretti non vengono modificati? (se li modifica, è un errore)
- I valori sbagliati vengono corretti? (se non li tocca, è un errore)
- L'output ha la forma giusta? (es. una data deve restare una data)
- L'output è parseable nel tipo di dato corretto?
- La funzione non usa variabili esterne? (deve essere autonoma)
- Ci sono branch di codice nell'ordine sbagliato? (un caso generico prima di uno specifico)

Se tutto è a posto → la funzione è approvata. Si va allo Step 9.
Se ci sono problemi → si va al Step 8d.

#### 8d — cleaner_repair_critic_agent

**Cosa riceve:**
- La funzione che ha fallito
- La lista di errori prodotta dal validatore (con esempi specifici)
- Il contesto della colonna

**Cosa fa:**
Legge gli errori e produce una diagnosi precisa:
- Qual è la causa radice del problema?
- Quale riga di codice è sbagliata?
- Che tipo di correzione serve? (riscrivere la logica, cambiare l'ordine dei branch, ecc.)
- Vale la pena riprovare o è un caso irrisolvibile?

**Cosa produce:**
Un oggetto `CleanerRepairDiagnosis` con la spiegazione e le istruzioni per la correzione.

#### 8e — Il Ciclo Riparte

La diagnosi viene passata di nuovo al `column_cleaner_generator_agent`,
che riscrive la funzione tenendo conto degli errori segnalati.

Si ritesta. Se fallisce ancora, si chiama di nuovo il critic. E così via.
**Massimo 10 tentativi per colonna.**

Se dopo molti tentativi la funzione continua a sbagliare nello stesso modo
(stessa funzione o stesso tipo di errore), il sistema rileva la **stagnazione**:
- Fornisce al generator uno schema strutturale diverso da seguire
- Aumenta leggermente la "temperatura" dell'AI (0.2 → 0.3 → 0.4 → massimo 0.5)
  così l'agente è spinto a provare soluzioni meno ovvie

#### Output dello Step 8

Per ogni colonna approvata:
- La funzione viene salvata in `Data/.cleaning_cache/<dataset>/generated_cleaners/<colonna>.py`
- Viene aggiunta a `Data/.cleaning_cache/<dataset>/cleaner_manifest.json`

---

### Step 9 — Applicazione (senza AI)

**File coinvolto:** `cleaning/application.py`

**Cosa riceve:**
- Il CSV originale
- Il `RemediationPlan` (dalla Fase 2)
- Il `cleaner_manifest.json` (dallo Step 8)

**Cosa fa (in questo ordine):**
1. Carica le funzioni cleaner dal disco e le esegue su ogni colonna inconsistente
2. Sostituisce i placeholder falsi con null nelle colonne identificate
3. Elimina le colonne duplicate esatte (quelle marcate `auto_apply=true`)
4. Rinomina le colonne con nomi non validi
5. Converte le colonne al tipo di dato corretto

**Cosa produce:**
Il CSV pulito + un `CleaningReport` che registra cosa è stato fatto colonna per colonna.

**Salvato in:**
`Data/.cleaning_cache/<nome_dataset>/<nome_dataset>.cleaned.csv`

---

## FASE 4 — VERIFICA

**File coinvolto:** `cleaning/verification.py`

**Nessun agente AI in questa fase.**

**Cosa riceve:**
- Il percorso del CSV pulito prodotto dallo Step 9
- La mappa dei rename applicati (per allineare i nomi vecchi con i nuovi)

**Cosa fa:**
- Rilegge il CSV pulito come testo puro (senza che pandas interpreti nulla)
- Riesegue la stessa analisi di consistenza della Fase 1
- Confronta i risultati colonna per colonna

**Cosa produce:**
Un `ConsistencyVerificationReport` con, per ogni colonna:
- `resolved` — il problema è sparito completamente
- `improved` — il problema è ridotto ma non eliminato
- `unchanged` — nessun cambiamento
- `regressed` — il problema è peggiorato (da investigare)
- `new` — è apparso un problema che prima non c'era

---

## FASE 5 — REPORT

**File coinvolto:** `cleaning/reporting.py`

---

### Step 10a — Costruzione del Report Tecnico (senza AI)

**Cosa riceve:**
Tutti i risultati di tutte le fasi precedenti:
- Il bundle di validazione
- Il piano di remediation
- Il CleaningReport
- Il ConsistencyVerificationReport

**Cosa fa:**
Li unisce in un unico oggetto `FinalPipelineReport`.

---

### Step 10b — narrative_report_agent

**Cosa riceve:**
Il `FinalPipelineReport` completo.

**Cosa fa:**
Scrive un testo narrativo in linguaggio naturale che racconta:
- Cosa è stato trovato nella validazione (sezione per sezione)
- Quali azioni sono state eseguite automaticamente
- Quali azioni sono state proposte ma non eseguite (richiedono revisione umana)
- Com'è cambiata la qualità dei dati prima e dopo
- Quali rischi restano aperti

**Cosa produce:**
Un oggetto `NarrativeReport` — testo Markdown + metadati.

**Salvato in:**
- `Data/.cleaning_cache/<nome_dataset>/<nome_dataset>.final_report.json`
- `Data/.cleaning_cache/<nome_dataset>/<nome_dataset>.report.md`

---

## Tabella Riepilogativa — Tutti i File

### File di Codice

| File | Cosa riceve | Cosa produce | Agente AI? |
|---|---|---|---|
| `main.py` | Niente (entrypoint) | Lancia `cli.py` | No |
| `cli.py` | Argomenti da terminale | Chiama la fase giusta | No |
| `validation/bundle.py` | CSV originale | `validation_bundle.json` | No (orchestra gli altri) |
| `validation/schema.py` | Campione CSV testo | `schema.json` | Sì (2 agenti) |
| `validation/completeness.py` | Profilo completezza | `completeness.json` | Sì (1 agente) |
| `validation/consistency.py` | Fatti di formato per colonna | `consistency.json` | Sì (1 agente, solo slow path) |
| `validation/anomaly.py` | Risultati rilevatori | `anomaly.json` | Sì (1 agente) |
| `validation/cross_column.py` | Risultati rilevatori | `cross_column.json` | Sì (1 agente) |
| `validation/duplicates.py` | Risultati rilevatori | `duplicates.json` | Sì (1 agente) |
| `cleaning/remediation.py` | `validation_bundle.json` | `remediation_plan.json` | No |
| `cleaning/request.py` | Schema + consistency finding | `ColumnCleaningRequest` (in memoria) | No |
| `cleaning/generation.py` | `ColumnCleaningRequest` | `<colonna>.py` + `cleaner_manifest.json` | Sì (2 agenti in loop) |
| `cleaning/validation.py` | Funzione generata + request | Lista di errori | No |
| `cleaning/application.py` | CSV + piano + manifest | `dataset.cleaned.csv` | No |
| `cleaning/verification.py` | CSV pulito | `ConsistencyVerificationReport` | No |
| `cleaning/reporting.py` | Tutti i risultati | `final_report.json` + `report.md` | Sì (1 agente) |
| `cleaning/runtime.py` | Funzione `.py` + Series pandas | `ColumnCleanerExecutionReport` | No |
| `cleaning/paths.py` | Nome dataset | Percorsi file (in memoria) | No |
| `tools/common_tools.py` | Varie | Helpers, retry AI, shape analisi | No |
| `tools/schema_tools.py` | DataFrame | Testo profilo dataset | No |
| `tools/format_tools.py` | DataFrame + nome colonna | `ColumnFormatFacts` | No |
| `tools/completeness_tools.py` | DataFrame | `CompletenessProfile` | No |
| `tools/quality_tools.py` | DataFrame | Liste di anomalie/duplicati/problemi | No |
| `agents.py` | — | Definizioni di tutti gli agenti | Sì (definisce i 10 agenti) |
| `models.py` | — | Definizioni di tutti i modelli dati | No |
| `cache.py` | — | Funzioni load/save per i JSON di cache | No |

---

### File di Dati Prodotti (in ordine di creazione)

| Ordine | File | Prodotto da | Consumato da |
|---|---|---|---|
| 1 | `.validation_cache/<ds>.schema.json` | `schema_summary_agent` | `bundle.py`, `request.py` |
| 2 | `.validation_cache/<ds>.completeness.json` | `completeness_analysis_agent` | `bundle.py`, `application.py` |
| 3 | `.validation_cache/<ds>.consistency.json` | `format_consistency_agent` | `bundle.py`, `generation.py` |
| 4 | `.validation_cache/<ds>.anomaly.json` | `anomaly_summary_agent` | `bundle.py` |
| 5 | `.validation_cache/<ds>.cross_column.json` | `cross_column_summary_agent` | `bundle.py` |
| 6 | `.validation_cache/<ds>.duplicates.json` | `duplicate_summary_agent` | `bundle.py` |
| 7 | `.validation_cache/<ds>.validation_bundle.json` | `bundle.py` | `remediation.py`, `orchestrator.py` |
| 8 | `.validation_cache/<ds>.remediation_plan.json` | `remediation.py` | `application.py` |
| 9 | `.cleaning_cache/<ds>/generated_cleaners/<col>.py` | `column_cleaner_generator_agent` | `runtime.py` |
| 10 | `.cleaning_cache/<ds>/cleaner_manifest.json` | `generation.py` | `application.py` |
| 11 | `.cleaning_cache/<ds>/<ds>.cleaned.csv` | `application.py` | `verification.py` |
| 12 | `.cleaning_cache/<ds>/<ds>.final_report.json` | `reporting.py` | — (output finale) |
| 13 | `.cleaning_cache/<ds>/<ds>.report.md` | `narrative_report_agent` | — (output finale) |

---

## Sequenza Completa in Ordine

```
 1. Carica il CSV originale in memoria
 2. [dtype_inference_agent]    → inferisce tipi e pattern per ogni colonna
 3. [schema_summary_agent]     → valida schema, nomi colonne, tipi definitivi
                                  → salva: schema.json
 4. [completeness_analysis_agent] → trova valori mancanti e placeholder falsi
                                  → salva: completeness.json
 5. Per ogni colonna del dataset:
    [format_consistency_agent] (solo se serve)
                                  → trova celle con formato inconsistente
                                  → salva: consistency.json
 6. [anomaly_summary_agent]    → riassume anomalie numeriche e categoriche
                                  → salva: anomaly.json
 7. [cross_column_summary_agent] → riassume problemi incrociati tra colonne
                                  → salva: cross_column.json
 8. [duplicate_summary_agent]  → riassume righe duplicate
                                  → salva: duplicates.json
 9. build_validation_results() → raccoglie tutto
                                  → salva: validation_bundle.json
10. run_remediation_planning() → costruisce il piano d'azione (no AI)
                                  → salva: remediation_plan.json
11. Per ogni colonna inconsistente (loop):
    a. Prepara ColumnCleaningRequest (no AI)
    b. [column_cleaner_generator_agent] → scrive la funzione Python
    c. Validatore interno (no AI) → testa la funzione
    d. Se fallisce:
       [cleaner_repair_critic_agent] → diagnostica il problema
       Torna al punto b (max 10 volte)
    e. Se passa: salva <colonna>.py
                                  → salva: generated_cleaners/<col>.py
12. Salva la lista di tutti i cleaner
                                  → salva: cleaner_manifest.json
13. Applica tutto al CSV (no AI):
    - Esegue i cleaner
    - Sostituisce placeholder con null
    - Elimina colonne duplicate
    - Rinomina colonne
    - Converte tipi di dato
                                  → salva: dataset.cleaned.csv
14. Rilegge il CSV pulito, confronta con prima (no AI)
                                  → produce: ConsistencyVerificationReport
15. Assembla il report finale (no AI)
                                  → produce: FinalPipelineReport
16. [narrative_report_agent]   → scrive il testo narrativo
                                  → salva: final_report.json + report.md
```

---

## I 10 Agenti in Ordine di Utilizzo

| # | Agente | Quando viene chiamato | Max chiamate |
|---|---|---|---|
| 1 | `dtype_inference_agent` | Una volta, all'inizio della validazione | 1 |
| 2 | `schema_summary_agent` | Una volta, subito dopo il dtype | 1 |
| 3 | `completeness_analysis_agent` | Una volta, dopo lo schema | 1 |
| 4 | `format_consistency_agent` | Una volta per ogni colonna senza pattern (solo slow path) | N colonne |
| 5 | `anomaly_summary_agent` | Una volta, dopo la consistency | 1 |
| 6 | `cross_column_summary_agent` | Una volta, dopo le anomalie | 1 |
| 7 | `duplicate_summary_agent` | Una volta, dopo il cross-column | 1 |
| 8 | `column_cleaner_generator_agent` | Più volte per colonna inconsistente (max 10 per colonna) | N colonne × 10 |
| 9 | `cleaner_repair_critic_agent` | Solo quando il generator produce una funzione sbagliata | Variabile |
| 10 | `narrative_report_agent` | Una volta, alla fine di tutto | 1 |
