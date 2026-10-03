# LLM-SQL deterministic evaluator

## Current default: accelerated evaluator

`evaluator.py` is now the accelerated serial implementation, version
`llm_sql_serial_fast_v1`. The GPT and GLM launchers select it by default.
`evaluator_prev.py` preserves the original concurrent implementation.
`evaluator_deterministic.py` remains the slow serial reference and provides
the exact LCP kernel imported by the accelerated evaluator; retain both files.
Historical scripts naming `evaluator.py` now select the new implementation:
use `evaluator_prev.py` explicitly to reproduce old concurrent scoring.

Full-input checks on all five datasets confirmed identical character counts,
serialized rows and prefix hits. CSV scalar serialization and character checks
are accelerated; exotic types fall back to the original serialization path.
Inputs are reloaded each evaluation, not cached, avoiding mutation leakage.
The score formula, timed region and legacy validity checks are unchanged.
Runtime-dependent total scores can still fluctuate.

The following describes the retained slower serial reference.

Use `evaluator_deterministic.py` for all newly rerun methods. Its entry point is
`evaluate(program_path)`, and every result includes
`evaluator_version: llm_sql_serial_v1`.

This is a separate benchmark version, not a replacement for historical results.
It preserves the original five datasets, reorder arguments, pandas row
serialization, validity checks, timed region and score formula. All datasets
must succeed. It replaces only concurrent Trie scoring with exact serial Trie
reuse, computed by summing adjacent longest common prefixes after lexical sort.
No separator is inserted between cells. Missing values serialize as empty strings.
For a fixed multiset of serialized rows the hit count is independent of row order.

Score = 0.95 * mean(hit_rates) + 0.05 * (12 - min(12, mean(runtime))) / 12.
Runtime measures reorder execution, not validation or prefix scoring. Prefix
counting is deterministic; program execution and its runtime need not be.

## Validity limitations

Compatibility mode intentionally retains the original row-count and character-
count checks. These are NOT a complete data integrity guarantee. The original
initial program actually merges columns and returns empty ordering metadata;
requiring unchanged shape or complete ordering metadata would invalidate it.
Do not describe this evaluator as enforcing exact cell preservation or all
column-merge constraints. A stricter validator would require a separately
agreed output/provenance contract.

## Evaluator files

- `evaluator_deterministic.py`: deterministic serial evaluator for new runs and
  final reporting.
- `fast_search_proxy.py`: faster, opt-in search proxy. It uses vectorized
  serialization and validation, so its scores are not final-reporting scores.
- `evaluator.py`: original concurrent evaluator, retained only for historical
  compatibility. Do not compare its scores directly with deterministic scores.

## Fair reruns

- Start every method from the same original program, without warm starts.
- Use the same task system prompt, budget, dataset files and seed set.
- Update the shared task prompt to describe serial character-prefix reuse;
  remove claims about concurrent Trie scoring. Do not give EE alone the v5
  algorithm-specific hints when making a search-method comparison.
- Pin Python/pandas dependencies and record evaluator/dataset hashes.
- Keep evaluator concurrency and machine load comparable. Repeat final
  measurements, report every result and their mean, not only the maximum.
- Do not compare new scores directly against old concurrent-evaluator scores.

`evaluator_deterministic.py` loads `evaluator.py` into a private namespace to
reuse its evaluation protocol without changing the historical module. Changes
to `evaluator.py` therefore require revalidating and versioning the deterministic
evaluator.
