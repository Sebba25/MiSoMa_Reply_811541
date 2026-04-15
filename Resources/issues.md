# schema_tools.py — Known Issues

## 1. `pandas_dtype` comment is misleading
**Location:** `ColumnProfile.pandas_dtype` (line 21)

The comment says "Inferred dtype obtained from LLM call", but in `build_dataset_profile` the value falls back to `str(df[column_name].dtype)` (raw pandas dtype) when no `dtype_overrides` are provided. A downstream consumer reading the comment might wrongly trust this as always being a semantic LLM-inferred type.

**Severity:** Low — documentation only, no runtime impact.

---

## 2. `naming_rule_reason` has incomplete violation detection
**Location:** `naming_rule_reason` (lines 98–121)

Only 5 specific violations are enumerated: uppercase, whitespace, `-`, `%`, and leading digit. Any other invalid character (e.g. `.`, `@`, `!`, `(`, `)`) falls through to the generic fallback message:
> "Column name violates the lowercase snake_case naming rule."

So `"price@unit"` would not mention the `@` character explicitly. The column is still correctly flagged as invalid — this is purely a clarity issue.

**Severity:** Low — UX/clarity only, no incorrect behavior.

---

## 3. `is_valid_schema_name` / `suggest_schema_name` leading-underscore inconsistency
**Location:** `is_valid_schema_name` (line 58), `suggest_schema_name` (lines 74–96)

`is_valid_schema_name` allows names starting with `_` (e.g. `_internal` is valid). However, `suggest_schema_name` strips leading underscores, so calling `suggest_schema_name("_foo")` returns `"foo"` — discarding the underscore that was deemed acceptable by the validator.

In the current flow this is harmless (only invalid names enter `suggest_schema_name`), but it is a latent trap if the function is reused elsewhere.

**Severity:** Low — no current runtime impact, but a potential future bug.

---

## 4. `suggest_schema_name` silent collision on empty/whitespace column names
**Location:** `suggest_schema_name` (line 74), `normalized_schema_name` (line 66)

If `name = ""` or a whitespace-only string is passed, `normalized_schema_name` returns the fallback `"column"`, and `suggest_schema_name` propagates it. If a dataset has multiple empty-named columns, they all receive the same suggestion `"column"`, producing a silent naming collision with no deduplication (e.g. `"column_1"`, `"column_2"`).

**Severity:** Medium — produces misleading output for datasets with unnamed columns.
