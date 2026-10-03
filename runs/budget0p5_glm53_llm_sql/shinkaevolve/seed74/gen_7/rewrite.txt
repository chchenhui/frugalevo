# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
import numpy as np


class Evolved(Algorithm):
    """
    Simple, safe reordering optimized for character prefix reuse.
    - Global column order ranked by length-weighted repetition.
    - Per-row column orderings consistent with returned data.
    - Rows sorted by serialized string to maximize trie prefix sharing.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    def _serialize(self, v) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and pd.isna(v):
            return ""
        if v is pd.NA:
            return ""
        return str(v)

    def _cell_strings(self, df: pd.DataFrame):
        """Return list-of-lists of serialized cell strings (rows x cols)."""
        cols = df.columns.tolist()
        data = []
        for row in df.itertuples(index=False, name=None):
            data.append([self._serialize(v) for v in row])
        return cols, data

    def _global_order(self, cols, data, col_merge):
        """Rank columns by sum over values v of len(v)*count*(count-1)."""
        n_rows = len(data)
        n_cols = len(cols)
        scores = [0] * n_cols
        for j in range(n_cols):
            counts = {}
            for i in range(n_rows):
                s = data[i][j]
                counts[s] = counts.get(s, 0) + 1
            sc = 0
            for s, c in counts.items():
                if c > 1:
                    sc += len(s) * c * (c - 1)
            scores[j] = sc
        # deterministic tie-break: score desc, then original column position
        order = sorted(range(n_cols), key=lambda j: (-scores[j], j))

        # honor col_merge: keep each merge group contiguous in given order
        merged = [g for g in col_merge if g]
        merged_set = set()
        for g in merged:
            merged_set.update(g)
        # expand merge groups into actual column names present
        expanded = []
        for g in merged:
            names = [c for c in cols if c in set(g)]
            if names:
                expanded.append(names)
        result = []
        used = set()
        for g in expanded:
            for c in g:
                if c in used:
                    continue
                result.append(c)
                used.add(c)
        for j in order:
            c = cols[j]
            if c not in used:
                result.append(c)
                used.add(c)
        return result

    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: int = None,
        col_stop: int = None,
        col_merge: List[List[str]] = [],
        one_way_dep: List[Tuple[str, str]] = [],
        distinct_value_threshold: float = 0.8,
        parallel: bool = True,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        original_cols = df.columns.tolist()
        if len(df) == 0 or len(original_cols) == 0:
            return df.copy(), [[] for _ in range(len(df))]

        cols, data = self._cell_strings(df)

        # global column order with merge handling
        ordered_cols = self._global_order(original_cols, data, col_merge)
        col_pos = {c: j for j, c in enumerate(original_cols)}
        order_idx = [col_pos[c] for c in ordered_cols]

        # build reordered row data and per-row orderings
        reordered_data = []
        for i in range(len(data)):
            row = data[i]
            reordered_data.append([row[j] for j in order_idx])

        # sort rows by serialized string to maximize trie prefix reuse
        row_keys = ["".join(r) for r in reordered_data]
        sort_idx = sorted(range(len(row_keys)), key=lambda i: row_keys[i])

        # build output dataframe preserving values and shape
        out_df = pd.DataFrame(index=range(len(df)), columns=ordered_cols)
        out_df = out_df.astype(object)
        column_orderings = []
        for pos, i in enumerate(sort_idx):
            src_row = df.iloc[i]
            for k, j in enumerate(order_idx):
                out_df.iat[pos, k] = src_row.iat[j]
            column_orderings.append(list(ordered_cols))

        return out_df, column_orderings


# EVOLVE-BLOCK-END