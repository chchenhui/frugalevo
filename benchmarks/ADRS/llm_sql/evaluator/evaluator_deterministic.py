"""Deterministic prefix metric, version llm_sql_serial_v1.

Keeps the original evaluator's datasets, serialization, validity checks and
runtime formula. Only replaces concurrent Trie scoring with exact serial reuse.
Total score still varies with measured program runtime. No proxy imports.
"""
from pathlib import Path
import runpy
import sys

VERSION = "llm_sql_serial_v1"
DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY))


def serial_trie_hit_count(strings):
    """Sum of adjacent sorted LCPs equals total query-before-insert Trie hits.

    Every distinct prefix contributes occurrences minus one hits, regardless
    of insertion order. Empty strings, duplicates and Unicode are supported.
    """
    ordered = sorted(strings)
    hits = 0
    for left, right in zip(ordered, ordered[1:]):
        low, high = 0, min(len(left), len(right))
        while low < high:
            middle = (low + high + 1) // 2
            if left[:middle] == right[:middle]:
                low = middle
            else:
                high = middle - 1
        hits += low
    return hits


def evaluate_df_prefix_hit_cnt(frame):
    # Deliberately retain the original pandas row serialization, not a
    # vectorized approximation: mixed dtypes and missing values matter.
    strings = ["".join(row.fillna("").astype(str).values)
               for _, row in frame.iterrows()]
    total = sum(map(len, strings))
    hits = serial_trie_hit_count(strings)
    return hits, 100.0 * hits / total if total else 0.0


# run_path creates a private namespace: never mutate the imported original
# evaluator module, so old and new evaluators can coexist in one process.
_namespace = runpy.run_path(str(DIRECTORY / "evaluator_prev.py"))
_evaluate = _namespace["evaluate"]
_evaluate.__globals__["evaluate_df_prefix_hit_cnt"] = evaluate_df_prefix_hit_cnt


def evaluate(program_path):
    result = _evaluate(program_path)
    result["evaluator_version"] = VERSION
    return result


if __name__ == "__main__":
    from wrapper import run as run_wrapper
    run_wrapper(evaluate)
