@ -0,0 +1,697 @@
# Spiegazione Format Consistency

Questo file gestisce la fase di **format consistency**, cioè controlla se i valori di ogni colonna hanno un formato coerente. Per esempio, se una colonna contiene mesi, controlla che i valori siano tutti simili, come `1`, `2`, `12`, e segnala valori strani come `January`, `mese 3`, `99` o `unknown`. 

### Docstring iniziale

La parte tra triple virgolette `""" ... """` spiega lo scopo del file. Dice che questa fase lavora **colonna per colonna** e può usare due modalità: una modalità veloce, senza chiamare l’LLM, quando lo schema ha già capito qual è il formato corretto; e una modalità lenta, in cui viene chiamato un agente AI per capire meglio il formato dominante. Dice anche che `_build_suggested_strategy` costruisce una strategia scritta per il cleaner-agent, cioè per l’agente che poi dovrà correggere i dati.

### Import

```python
from __future__ import annotations
```

Serve per rendere più flessibili le annotazioni dei tipi, cioè quelle parti come `str`, `list[str]`, `SchemaColumnEntry | None`. In pratica aiuta Python a leggere meglio i tipi usati nel codice.

```python
import asyncio
import sys
from collections import Counter, defaultdict
from pathlib import Path
```

Qui vengono importati strumenti standard di Python. `asyncio` serve per eseguire più controlli in parallelo. `sys` serve qui soprattutto per stampare messaggi su `stderr`. `Counter` conta quante volte compare un valore. `defaultdict` è un dizionario che crea automaticamente una lista vuota quando una chiave non esiste ancora. `Path` serve per gestire percorsi di file in modo più ordinato.

```python
import pandas as pd
```

Importa `pandas`, la libreria usata per lavorare con tabelle e dataset.

Gli import successivi prendono agenti, funzioni e modelli da altri file del progetto. Per esempio `format_consistency_agent` è l’agente che analizza i formati, mentre `ColumnConsistencyReport`, `FormatConsistencyFinding` e `ConsistencyValidationReport` sono strutture dati usate per salvare i risultati del controllo.

---

## Funzione `_build_suggested_strategy`

```python
def _build_suggested_strategy(...):
```

Questa funzione costruisce una **stringa di istruzioni** per il cleaner-agent. Una stringa è semplicemente testo. Lo scopo è dire all’agente come correggere i valori incoerenti.

```python
expected_pattern: str
```

È il formato atteso. Per esempio: `"YYYY-MM-DD"` oppure `"month number (1-12)"`.

```python
dominant_shape: str | None
```

È la forma più frequente nella colonna. Per esempio, se quasi tutti i valori sono tipo `2024-01`, quella è la forma dominante. `None` significa che potrebbe non esserci una forma dominante chiara.

```python
inconsistent_examples
```

Sono esempi di valori incoerenti, cioè valori che non seguono il formato principale.

```python
dominant_example_values: list[str] | None = None
```

Sono esempi di valori già corretti. Servono per far capire all’agente come deve apparire l’output finale.

```python
allow_variable_numeric_width: bool = False
```

È un booleano, quindi può essere `True` o `False`. Indica se i numeri possono avere lunghezze diverse. Per esempio, per i mesi `1` e `12` sono entrambi validi anche se uno ha una cifra e l’altro due.

```python
groups: dict[str, list[str]] = defaultdict(list)
```

Crea un dizionario chiamato `groups`. La chiave sarà la forma del valore, mentre il valore sarà una lista di esempi. Per esempio: forma `"letters"` → valori `["Jan", "Feb"]`.

```python
for ex in inconsistent_examples:
    if ex.shape != dominant_shape:
        groups[ex.shape].append(ex.value)
```

Qui il codice guarda ogni esempio incoerente. Se la sua forma è diversa dalla forma dominante, lo mette nel gruppo corrispondente. Quindi raggruppa gli errori per tipo di forma.

```python
if not groups:
```

Significa: se non ci sono gruppi di valori incoerenti.

```python
return (
    f"Normalize all values to '{expected_pattern}'. "
    "Map to null when the value cannot be converted."
)
```

Se non ci sono errori particolari, la funzione restituisce una frase semplice: normalizza tutti i valori al formato atteso e metti `null` solo quando il valore non può essere convertito.

```python
lines = [...]
```

Crea una lista di righe di testo. Alla fine queste righe verranno unite per formare una strategia completa.

```python
f"Target format: '{expected_pattern}'. "
```

Dice qual è il formato finale desiderato.

```python
f"Dominant valid shape: '{dominant_shape}' ..."
```

Dice qual è la forma già valida e specifica che i valori con quella forma vanno conservati.

```python
if dominant_example_values:
```

Se ci sono esempi di valori validi, allora il codice li aggiunge alla strategia.

```python
dominant_sample = ", ".join(repr(v) for v in dominant_example_values[:5])
```

Prende al massimo i primi 5 esempi validi e li trasforma in una stringa leggibile. `repr(v)` mostra il valore con le virgolette, per esempio `'2024-01'`.

```python
lines.append(...)
```

Aggiunge una nuova riga di testo alla strategia.

```python
if allow_variable_numeric_width:
```

Se i numeri possono avere lunghezze diverse, il codice aggiunge istruzioni meno rigide. Per esempio, non obbliga tutti i valori ad avere la stessa lunghezza.

```python
"Do not force every output to have the same string length..."
```

Questa parte dice all’agente: non correggere i valori solo copiando la lunghezza degli esempi. Se il valore ha senso, può essere valido anche con una lunghezza diversa.

```python
if expected_pattern.strip().lower() == "month number (1-12)":
```

Qui controlla se il formato atteso è un numero di mese da 1 a 12. `strip()` elimina spazi iniziali e finali. `lower()` mette tutto in minuscolo, così il confronto è più sicuro.

```python
"Pure numeric values that are outside the valid month range..."
```

Se la colonna rappresenta mesi, allora numeri come `0`, `13`, `99` o negativi devono diventare `null`, perché non sono mesi validi.

```python
else:
```

Se invece non è permessa una lunghezza numerica variabile, allora il codice usa regole più rigide.

```python
"same length, same character structure, same field order..."
```

Qui dice all’agente che l’output deve avere la stessa struttura degli esempi validi: stessa lunghezza, stesso ordine dei campi, stessa forma generale. Per esempio, se il formato valido è `YYYY-MM`, non bisogna trasformarlo in `MM-YYYY`.

```python
for shape, values in sorted(groups.items(), key=lambda x: -len(x[1])):
```

Qui il codice passa su ogni gruppo di errori. Li ordina mettendo prima i gruppi più numerosi. `lambda x: -len(x[1])` significa: ordina in base alla lunghezza della lista dei valori, dal gruppo più grande al più piccolo.

```python
examples = ", ".join(repr(v) for v in values[:5])
```

Per ogni gruppo prende massimo 5 esempi e li prepara come testo.

```python
lines.append(f"  shape '{shape}': e.g. {examples}")
```

Aggiunge alla strategia una riga che dice: questa forma ha questi esempi.

```python
lines.append(...)
```

Aggiunge una regola finale importante: ogni valore incoerente deve essere gestito esplicitamente e non bisogna lasciarlo invariato se non è già corretto.

```python
return "\n".join(lines)
```

Unisce tutte le righe in un unico testo finale, separate da `\n`, cioè andando a capo.

---

## Funzione `_profile_schema_guided_inconsistencies`

Questa funzione controlla gli errori di formato usando lo **schema**, cioè informazioni già note sulla colonna.

```python
if schema_entry.pandas_dtype not in {"Int64", "Float64"}:
    return None
```

Se la colonna non è numerica, la funzione si ferma e restituisce `None`. `Int64` indica numeri interi, `Float64` indica numeri decimali.

```python
rendered = df[column_name].dropna().astype(str).str.strip()
```

Prende la colonna, elimina i valori mancanti con `dropna()`, converte tutto in stringa con `astype(str)` e rimuove spazi inutili con `str.strip()`.

```python
rendered = rendered[rendered != ""]
```

Elimina le stringhe vuote.

```python
rendered = rendered[~rendered.str.lower().isin(PLACEHOLDER_TOKENS)]
```

Elimina valori placeholder, cioè valori che indicano “mancante” o “non disponibile”, come potrebbero essere `unknown`, `n/a`, `null`.

```python
invalid_values = [...]
```

Crea una lista di valori non validi.

```python
for value in rendered
```

Controlla ogni valore pulito della colonna.

```python
if not matches_numeric_schema_pattern(...)
```

Se il valore non rispetta il formato numerico atteso, viene inserito tra gli invalidi.

```python
if not invalid_values:
    return 0, []
```

Se non ci sono valori invalidi, restituisce `0` e una lista vuota.

```python
counts = Counter(invalid_values)
```

Conta quante volte compare ogni valore invalido.

```python
examples = [...]
```

Costruisce una lista di esempi di outlier, cioè valori fuori formato.

```python
FormatOutlierExample(
    value=value[:80],
    shape=value_shape(value),
    count=count,
)
```

Per ogni valore invalido salva tre informazioni: il valore, la sua forma, e quante volte compare. `value[:80]` limita il valore ai primi 80 caratteri.

```python
return len(invalid_values), examples
```

Restituisce il numero totale di valori invalidi e gli esempi rappresentativi.

---

## Funzione `run_column_format_check`

Questa funzione controlla **una singola colonna**.

```python
if schema_entry is not None and schema_entry.string_role in ("name", "free_text"):
```

Se la colonna è un nome o testo libero, viene saltata. Per esempio, una colonna con nomi di persone non deve avere un formato rigido.

```python
return ColumnConsistencyReport(...)
```

Restituisce un report dicendo che la colonna è stata saltata.

```python
format_facts = build_column_format_facts(df, column_name)
```

Qui costruisce un riassunto del formato della colonna: forma dominante, esempi validi, esempi incoerenti, numero di righe problematiche.

```python
if not format_facts.machine_format_candidate or format_facts.inconsistent_rows <= 0:
```

Se la colonna non è una candidata a formato macchina, oppure non ha righe incoerenti, non c’è nulla da correggere.

```python
return ColumnConsistencyReport(finding=None, ...)
```

Restituisce un report senza finding. `finding=None` significa: nessun problema trovato.

```python
if schema_entry is not None and schema_entry.detected_pattern:
```

Questa è la **fast path**, cioè la strada veloce. Se lo schema ha già identificato un formato, non serve chiamare l’LLM.

```python
inconsistent_rows = format_facts.inconsistent_rows
inconsistent_example_profiles = format_facts.inconsistent_examples
used_schema_override = False
```

Salva il numero di righe incoerenti, gli esempi incoerenti e una variabile che indica se verrà usato un controllo guidato dallo schema.

```python
guided_profile = _profile_schema_guided_inconsistencies(...)
```

Prova a fare un controllo più preciso usando lo schema.

```python
if guided_profile is not None:
```

Se il controllo guidato ha prodotto un risultato, usa quello.

```python
inconsistent_rows, inconsistent_example_profiles = guided_profile
used_schema_override = True
```

Aggiorna numero di errori ed esempi, e segna che è stato usato lo schema.

```python
if inconsistent_rows <= 0:
```

Se dopo il controllo non ci sono più errori, restituisce un report senza problemi.

```python
inconsistent_examples = [ex.value for ex in inconsistent_example_profiles]
```

Estrae solo i valori dagli esempi incoerenti.

```python
evidence = (...)
```

Costruisce una spiegazione del problema trovato: quale pattern era atteso, quante righe sono fuori formato, qual è la forma dominante.

```python
if used_schema_override:
```

Se è stato usato il controllo guidato dallo schema, cambia il testo dell’evidenza per renderlo più preciso.

```python
return ColumnConsistencyReport(...)
```

Restituisce il report finale della colonna.

```python
finding=FormatConsistencyFinding(...)
```

Qui crea il problema vero e proprio: nome colonna, formato atteso, numero di righe incoerenti, esempi, evidenza e strategia suggerita.

```python
suggested_strategy=_build_suggested_strategy(...)
```

Chiama la funzione spiegata prima per costruire le istruzioni da dare al cleaner-agent.

```python
allow_variable_numeric_width=numeric_pattern_allows_variable_width(...)
```

Decide se il formato numerico può avere lunghezze diverse.

---

## Slow path: quando serve l’agente AI

```python
schema_context = ""
```

Crea una stringa vuota che conterrà informazioni aggiuntive sullo schema.

```python
if schema_entry is not None:
```

Se ci sono informazioni di schema, le aggiunge.

```python
schema_context = (...)
```

Qui inserisce nel prompt il tipo atteso della colonna e il ruolo semantico. Il ruolo semantico indica cosa rappresenta la colonna, per esempio mese, prezzo, codice, testo.

```python
prompt = [...]
```

Costruisce il prompt per l’agente AI. Il prompt contiene il nome del dataset, il nome della colonna, il numero di righe, la forma dominante e il numero di righe incoerenti.

```python
attach_profile_text(format_facts)
```

Aggiunge al prompt il profilo della colonna in formato testuale.

```python
print(..., file=sys.stderr, flush=True)
```

Stampa un log tecnico su `stderr`, non su `stdout`, per non mischiare i messaggi di controllo con l’output principale. `flush=True` forza la stampa immediata.

```python
result = run_agent_with_backoff(format_consistency_agent, prompt)
```

Chiama l’agente AI. `with_backoff` significa che, se la chiamata fallisce temporaneamente, il sistema può riprovare aspettando un po’.

```python
output = result.output
```

Prende l’output prodotto dall’agente.

```python
if output.finding is not None and output.finding.inconsistent_rows <= 0:
```

Se l’agente ha prodotto un finding ma dice che le righe incoerenti sono zero, il codice lo considera come nessun problema reale.

```python
return output
```

Altrimenti restituisce direttamente il risultato dell’agente.

---

## Funzione `run_column_format_check_async`

Questa funzione fa la stessa cosa di `run_column_format_check`, ma in versione **asincrona**. Asincrona significa che può essere eseguita insieme ad altre operazioni, senza aspettare che una finisca prima di iniziare la successiva.

La logica è quasi identica: salta colonne non adatte, costruisce `format_facts`, usa la fast path se c’è uno schema, oppure usa la slow path con l’agente AI.

La differenza principale è questa riga:

```python
result = await run_agent_with_backoff_async(format_consistency_agent, prompt)
```

`await` significa: aspetta il risultato di questa operazione asincrona, ma senza bloccare inutilmente tutto il programma.

---

## Funzione `_run_column_format_checks_async`

Questa funzione serve per controllare **più colonne in parallelo**.

```python
semaphore = asyncio.Semaphore(max_workers)
```

Crea un semaforo. In programmazione, un semaforo serve a limitare quante operazioni possono avvenire contemporaneamente. Per esempio, se `max_workers = 3`, controlla massimo 3 colonne alla volta.

```python
async def _run_one(index: int, column_name: str):
```

Definisce una funzione interna che controlla una singola colonna.

```python
async with semaphore:
```

Dice: esegui questa parte solo se c’è uno slot libero nel semaforo.

```python
report = await run_column_format_check_async(...)
```

Esegue il controllo asincrono della colonna.

```python
return index, report
```

Restituisce sia l’indice della colonna sia il report. L’indice serve per rimettere i risultati nell’ordine originale.

```python
tasks = [...]
```

Crea una lista di task asincroni, uno per ogni colonna.

```python
for task in asyncio.as_completed(tasks):
```

Processa i task man mano che finiscono, anche se finiscono in ordine diverso.

```python
reports_by_index[index] = report
```

Salva ogni report usando l’indice originale.

```python
return [reports_by_index[index] for index in range(len(column_names))]
```

Restituisce i report nello stesso ordine delle colonne iniziali.

---

## Funzione `run_format_consistency_validation`

Questa è la funzione principale del file. Controlla tutto il dataset.

```python
if max_workers < 1:
    raise ValueError("max_workers must be at least 1.")
```

Controlla che il numero di worker sia almeno 1. Se è 0 o negativo, genera errore.

```python
if reuse_cache:
    return load_consistency(path)
```

Se `reuse_cache=True`, non rifà i controlli: carica un risultato già salvato. La cache è una memoria temporanea di risultati precedenti.

```python
df = load_dataset_frame(path, dtype=str if read_as_str else None)
```

Carica il dataset in un DataFrame pandas. Un DataFrame è una tabella con righe e colonne. Se `read_as_str=True`, legge tutti i valori come stringhe.

```python
format_findings: list[FormatConsistencyFinding] = []
```

Crea una lista vuota dove verranno messi tutti i problemi trovati.

```python
schema_map: dict[str, SchemaColumnEntry] = {}
```

Crea un dizionario che collegherà ogni nome colonna alle informazioni di schema.

```python
try:
    handoff = load_schema_handoff(path)
    schema_map = {col.name: col for col in handoff.columns}
except Exception:
    pass
```

Prova a caricare lo schema prodotto da una fase precedente. Se non ci riesce, non blocca il programma: continua senza schema.

```python
column_names = list(df.columns)
```

Prende tutti i nomi delle colonne del dataset.

```python
if max_workers == 1 or len(column_names) <= 1:
```

Se c’è un solo worker o una sola colonna, esegue i controlli uno alla volta.

```python
reports = []
for column_name in column_names:
```

Crea una lista di report e passa colonna per colonna.

```python
schema_entry = schema_map.get(column_name)
```

Cerca le informazioni di schema per quella colonna.

```python
reports.append(run_column_format_check(...))
```

Esegue il controllo della colonna e aggiunge il report alla lista.

```python
else:
```

Se invece ci sono più worker e più colonne, usa la versione asincrona.

```python
worker_count = min(max_workers, len(column_names))
```

Decide quanti worker usare davvero. Non ha senso usare più worker del numero di colonne.

```python
print(..., file=sys.stderr, flush=True)
```

Stampa un log che dice quante colonne saranno controllate e con quanti worker.

```python
reports = asyncio.run(...)
```

Avvia i controlli asincroni.

```python
for result in reports:
    if result.finding is not None:
        format_findings.append(result.finding)
```

Guarda tutti i report e conserva solo quelli che hanno trovato un problema.

```python
report = ConsistencyValidationReport(...)
```

Crea il report finale dell’intero dataset.

```python
dataset_name=path.stem
```

Usa il nome del file senza estensione come nome del dataset.

```python
total_rows=len(df)
```

Salva il numero totale di righe.

```python
format_consistency_findings=format_findings
```

Inserisce tutti i problemi di formato trovati.

```python
summary=(...)
```

Scrive un riassunto finale: quante colonne sono state analizzate e quanti problemi sono stati trovati.

```python
save_consistency(path, report)
```

Salva il report nella cache o in un file di output.

```python
return report
```

Restituisce il report finale.