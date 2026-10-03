"""Opt-in search proxy with vectorized character-count validation (formerly v2).

Uses the same five datasets, reorder arguments, score formula, and timed code
region as evaluator.py. Character-Trie concurrency is replaced by ideal serial
LCP reuse, so final reporting MUST use the unchanged official evaluator.
"""
import importlib.util
from pathlib import Path
import sys
import time

import pandas as pd
import numpy as np

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY))


def serial_trie_hit_count(strings):
    """Original v5 counting kernel, inlined to remove the legacy proxy dependency."""
    ordered = sorted(strings)
    hit = 0
    for previous, current in zip(ordered, ordered[1:]):
        if previous == current:
            hit += len(current)
            continue
        length = 0
        for left, right in zip(previous, current):
            if left != right:
                break
            length += 1
        hit += length
    return hit

MERGES = {
    "movies": [["movieinfo", "movietitle", "rottentomatoeslink"]],
    "beer": [["beer/beerId", "beer/name"]],
    "BIRD": [["PostId", "Body"]],
    "PDMX": [["path", "metadata"], ["hasmetadata", "isofficial", "isuserpublisher", "isdraft", "hasannotations", "subsetall"]],
    "products": [["product_title", "parent_asin"]],
}


def character_count(frame):
    # Same astype(str) values as the original row-wise sum, cheaper reduction.
    # pandas 3 string dtype retains missing sentinels after astype(str);
    # the official .str.len().sum() skips those sentinels.
    return sum(len(value) for value in frame.astype(str).to_numpy().ravel()
               if isinstance(value, str))


def serialize_rows(frame):
    # iterrows draws rows from this same common-dtype matrix. Preserve its
    # numeric upcasting, but avoid one Series/fillna/astype allocation per row.
    values = frame.to_numpy()
    missing = pd.isna(values)
    text = np.frompyfunc(str, 1, 1)(values)
    text[missing] = ""
    return ["".join(row) for row in text]


def evaluate_df_prefix_hit_cnt(frame):
    strings = serialize_rows(frame)
    total = sum(map(len, strings))
    hit = serial_trie_hit_count(strings)
    return hit, 100.0 * hit / total if total else 0.0


def evaluate(program_path):
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        hits = []
        runtime = 0.0
        for name, merge in MERGES.items():
            frame = pd.read_csv(DIRECTORY / "datasets" / f"{name}.csv")
            count_before = character_count(frame)
            rows_before = len(frame)
            start = time.time()
            reordered, _ = program.Evolved().reorder(
                frame, early_stop=100000, distinct_value_threshold=0.7,
                row_stop=4, col_stop=2, col_merge=merge,
            )
            runtime += time.time() - start
            if len(reordered) != rows_before:
                raise ValueError("Row count changed")
            if character_count(reordered) < count_before:
                raise ValueError("Character count decreased")
            hits.append(evaluate_df_prefix_hit_cnt(reordered)[1] / 100.0)
        score = 0.95 * sum(hits) / len(hits) + 0.05 * (12 - min(12, runtime / len(hits))) / 12
        return {"combined_score": score, "runs_successfully": 1.0,
                "hit_rates": hits, "total_runtime": runtime}
    except Exception as exc:
        return {"combined_score": 0.0, "runs_successfully": 0.0,
                "error": f"{type(exc).__name__}: {exc}"}
