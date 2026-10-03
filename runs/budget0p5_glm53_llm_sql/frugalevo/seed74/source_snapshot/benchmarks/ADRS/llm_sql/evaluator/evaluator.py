"""Accelerated serial evaluator; same scoring and legacy validity rules.

No input cache: every evaluation reloads data, preventing candidate mutation leakage.
"""
import importlib.util
from pathlib import Path
import sys
import time

import pandas as pd
import numpy as np

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY))
from evaluator_deterministic import serial_trie_hit_count
VERSION = "llm_sql_serial_fast_v1"


MERGES = {
    "movies": [["movieinfo", "movietitle", "rottentomatoeslink"]],
    "beer": [["beer/beerId", "beer/name"]],
    "BIRD": [["PostId", "Body"]],
    "PDMX": [["path", "metadata"], ["hasmetadata", "isofficial", "isuserpublisher", "isdraft", "hasannotations", "subsetall"]],
    "products": [["product_title", "parent_asin"]],
}


def character_count(frame):
    if frame.empty:
        return frame.astype(str).apply(lambda row: row.str.len().sum(), axis=1).sum()
    # Same astype(str) values as the original row-wise sum, cheaper reduction.
    # pandas 3 string dtype retains missing sentinels after astype(str);
    # the official .str.len().sum() skips those sentinels.
    return sum(len(value) for value in frame.astype(str).to_numpy().ravel()
               if isinstance(value, str))


def serialize_rows(frame):
    # iterrows draws rows from this same common-dtype matrix. Preserve its
    # numeric upcasting, but avoid one Series/fillna/astype allocation per row.
    values = frame.to_numpy()
    # Exotic dtypes retain the exact pandas path. Fast path targets CSV scalar data.
    if values.dtype.kind not in ("O", "U", "S") or any(
        not isinstance(v, (str, int, float, bool, np.integer, np.floating, np.bool_))
        and v is not None and v is not pd.NA
        for v in values.ravel()
    ):
        return ["".join(row.fillna("").astype(str).values) for _, row in frame.iterrows()]
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
                "hit_rates": hits, "total_runtime": runtime, "evaluator_version": VERSION}
    except Exception as exc:
        return {"combined_score": 0.0, "runs_successfully": 0.0,
                "error": f"{type(exc).__name__}: {exc}", "evaluator_version": VERSION}


if __name__ == "__main__":
    from wrapper import run as run_wrapper
    run_wrapper(evaluate)
