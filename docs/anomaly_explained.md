# Spiegazione File: Anomaly Detection

Questo documento spiega in modo semplice il file anomaly.py, blocco per blocco.

---

## Scopo del file

Questo file serve per individuare **anomalie nei dati**, cioè valori “strani” o rari.

Esempi:
- numeri molto diversi dagli altri (outliers)
- categorie che compaiono pochissime volte

IMPORTANTE:
Le anomalie vengono trovate con regole (non AI). L’AI viene usata solo per scrivere il riassunto finale.

---

## Docstring iniziale

Spiega che:
- le anomalie vengono trovate con metodi locali (heuristic)
- l’agente AI scrive solo il summary
- colonne duplicate vengono filtrate per evitare duplicati

---

## Import

- `sys`: per log
- `Path`: gestione file
- `anomaly_summary_agent`: AI per riassunto
- `load_anomaly / save_anomaly`: cache
- `detect_numeric_outlier_candidates`: trova outlier numerici
- `detect_rare_category_candidates`: trova categorie rare

---

## Funzione `_duplicate_semantic_suppressed_columns`

Serve a evitare duplicati tra colonne simili.

### Input:
- lista colonne schema
- gruppi di colonne duplicate

### Logica:

```python
by_name = {column.name: column for column in columns}
```
Crea dizionario nome → colonna

```python
suppressed = set()
```
Crea insieme colonne da ignorare

```python
for group in duplicate_groups:
```
Analizza ogni gruppo duplicato

```python
present = [name for name in group.columns if name in by_name]
```
Prende solo colonne presenti nel dataset

```python
if len(present) < 2:
    continue
```
Se meno di 2 → niente duplicato

---

### Scelta colonna migliore

```python
def _sort_key(name):
```

Definisce una regola per scegliere la colonna migliore.

Criteri:
1. nome valido
2. nome normalizzato
3. ordine alfabetico

```python
preferred = sorted(present, key=_sort_key)[0]
```

Sceglie la migliore

```python
suppressed.update(name for name in present if name != preferred)
```

Tutte le altre vengono ignorate

---

## Funzione `run_anomaly_detection`

È la funzione principale.

---

### 1. Cache

```python
if reuse_cache:
    return load_anomaly(path)
```

Se già calcolato → carica risultato

---

### 2. Caricamento dati

```python
df = load_dataset_frame(path)
```

Carica dataset

---

### 3. Schema

```python
handoff = load_schema_handoff(path)
```

Carica informazioni schema

```python
suppressed_columns = _duplicate_semantic_suppressed_columns(...)
```

Trova colonne da ignorare

Se schema non esiste:
```python
except FileNotFoundError:
```
continua senza schema

---

### 4. Detect anomalie

```python
findings = [...]
```

Qui succede la parte importante:

```python
detect_numeric_outlier_candidates(df, schema_columns)
```
Trova numeri anomali

```python
detect_rare_category_candidates(df, schema_columns)
```
Trova categorie rare

```python
if finding["column_name"] not in suppressed_columns
```

Filtra colonne duplicate

---

### 5. Ordinamento

```python
findings.sort(...)
```

Ordina anomalie per:
1. più righe colpite
2. nome colonna
3. tipo anomalia

---

### 6. Summary base

```python
fallback_summary = ...
```

Se ci sono anomalie:
→ dice quante sono

Se no:
→ dice che non ci sono anomalie

---

### 7. Creazione report

```python
report = AnomalyDetectionReport(...)
```

Contiene:
- nome dataset
- numero righe
- numero colonne
- anomalie trovate
- summary base

---

### 8. Log

```python
print(..., file=sys.stderr, flush=True)
```

Scrive log tecnico

---

### 9. AI summary

```python
report = report.model_copy(update={...})
```

Aggiorna report con summary AI

```python
summarize_validation_report(...)
```

AI:
- legge anomalie
- scrive summary leggibile
- NON modifica dati

---

### 10. Salvataggio

```python
save_anomaly(path, report)
```

Salva risultato

---

### 11. Return

```python
return report
```

Restituisce report finale

---

## Concetti chiave

- **Outlier**: valore numerico molto diverso
- **Categoria rara**: valore che appare poche volte
- **Schema**: informazioni sulle colonne
- **Duplicate suppression**: evitare doppioni
- **Heuristic**: regole semplici, non AI
- **AI summary**: solo spiegazione, non calcolo

---

## Idea generale

Questo file:
1. Carica dataset
2. Trova anomalie con regole
3. Rimuove duplicati
4. Ordina risultati
5. Usa AI per riassunto
6. Salva report
