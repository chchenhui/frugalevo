# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict
import os


class Evolved(Algorithm):
    """
    Global character-Trie aware column reordering.

    The evaluator serializes each output row as the concatenation of its cell
    strings in the DataFrame's column order and measures character reuse via a
    shared Trie. For a fixed multiset of serialized row strings the reuse is
    insertion-order independent, so the only decision that matters is the
    (global) column order. We build a handful of cheap candidate orders,
    measure their true Trie reuse with the sorted-string adjacent-LCP sum, and
    return the best one.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.dep_graph = None
        self.num_rows = 0
        self.num_cols = 0
        self.column_stats = None
        self.val_len = None
        self.row_stop = None
        self.col_stop = None
        self.base = 2000

    # ------------------------------------------------------------------
    # serialization helpers
    # ------------------------------------------------------------------
    def _serialize_columns(self, work: pd.DataFrame) -> Dict[str, pd.Series]:
        """Per-column string representation matching the evaluator:
        missing -> "", everything else -> str(value)."""
        ser = {}
        for c in work.columns:
            col = work[c]
            s = col.astype(str)
            try:
                mask = col.isna()
                if mask.any():
                    s = s.copy()
                    s[mask] = ""
            except Exception:
                pass
            ser[c] = s
        return ser

    def _row_strings(self, order: List[str], ser: Dict[str, pd.Series]) -> List[str]:
        arrays = [ser[c].to_numpy(dtype=object) for c in order]
        n = len(work_len_marker := arrays[0]) if arrays else 0
        if not arrays:
            return []
        if len(arrays) == 1:
            return list(arrays[0])
        # join per row
        cols = np.empty((len(arrays), len(arrays[0])), dtype=object)
        for i, a in enumerate(arrays):
            cols[i] = a
        return ["".join(row) for row in cols.T]

    def _lcp_score(self, strs: List[str]) -> float:
        """Exact ideal Trie reuse: sum of adjacent LCPs after sorting."""
        if len(strs) < 2:
            return 0.0
        ss = sorted(strs)
        total = 0.0
        cp = os.path.commonprefix
        for i in range(1, len(ss)):
            a, b = ss[i - 1], ss[i]
            if a == b:
                total += len(a)
            else:
                total += len(cp((a, b)))
        return total

    # ------------------------------------------------------------------
    # candidate column orders
    # ------------------------------------------------------------------
    def _candidate_orders(self, work: pd.DataFrame, ser: Dict[str, pd.Series]) -> List[List[str]]:
        cols = list(work.columns)
        if len(cols) <= 1:
            return [cols]

        gains = {}
        gains2 = {}
        nuniq = {}
        avg_len = {}
        for c in cols:
            s = ser[c]
            vc = s.value_counts()
            n = len(s)
            g1 = 0
            g2 = 0
            lens = s.str.len()
            len_map = dict(zip(vc.index, lens.value_counts().index.tolist() and
                               [len(str(v)) for v in vc.index]))
            for v, cnt in vc.items():
                L = len(str(v))
                if cnt > 1:
                    g1 += L * cnt * (cnt - 1)
                    g2 += L * cnt
            gains[c] = g1
            gains2[c] = g2
            nuniq[c] = int(len(vc))
            avg_len[c] = float(lens.mean()) if n else 0.0

        order1 = sorted(cols, key=lambda c: (-gains[c], nuniq[c], c))
        order2 = sorted(cols, key=lambda c: (nuniq[c], -avg_len[c], c))
        order3 = sorted(cols, key=lambda c: (-gains2[c], nuniq[c], c))
        # dedupe identical orders
        seen = []
        for o in (order1, order2, order3):
            if o not in seen:
                seen.append(o)
        return seen

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
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
        # Work on a copy; never mutate the caller's frame.
        work = df.copy()
        n_rows = len(work)

        # Honor required column merges using the parent API semantics.
        if col_merge:
            cols_now = list(work.columns)
            for grp in col_merge:
                present = [c for c in cols_now if c in grp]
                if len(present) > 1 and hasattr(self, "mererging_columns"):
                    work = self.merging_columns(work, present, prepended=False)
                elif len(present) > 1 and hasattr(self, "merging_columns"):
                    work = self.merging_columns(work, present, prepended=False)
                cols_now = list(work.columns)

        cols = list(work.columns)
        if len(cols) == 0:
            return work.copy(), []
        if n_rows == 0:
            return work.copy(), []

        # Serialized per-column strings (evaluator representation, values untouched).
        ser = self._serialize_columns(work)

        # Build a small set of cheap candidate global orders and pick the one
        # with the best true character-Trie reuse.
        candidates = self._candidate_orders(work, ser)
        best_order = None
        best_score = -1.0
        for order in candidates:
            strs = self._row_strings(order, ser)
            score = self._lcp_score(strs)
            if score > best_score:
                best_score = score
                best_order = order

        # Materialize the reordered frame preserving every original value.
        out = work[best_order].copy()
        if out.dtypes.apply(lambda d: d == object).any():
            out = out.astype(object)

        column_orderings = [list(best_order) for _ in range(n_rows)]
        self.num_rows = n_rows
        self.num_cols = len(best_order)
        return out, column_orderings


work_len_marker = None  # placeholder removed; see _row_strings

# EVOLVE-BLOCK-END