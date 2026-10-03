# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict, Optional


class Evolved(Algorithm):
    """
    Row-wise column reordering optimized for character-level prefix reuse
    (LLM prompt prefix caching).
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------
    # serialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cell_str(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and pd.isna(v):
            return ""
        if v is pd.NA:
            return ""
        try:
            if v != v:  # NaN-like
                return ""
        except Exception:
            pass
        if isinstance(v, bool):
            return "True" if v else "False"
        return str(v)

    def _string_cells(self, df: pd.DataFrame) -> List[List[str]]:
        """Return list of rows, each a list of cell strings (missing -> '')."""
        obj = df.astype(object).where(pd.notna(df), "")
        out = []
        for row in obj.values:
            out.append([self._cell_str(v) if v is not None else "" for v in row])
        return out

    # ------------------------------------------------------------------
    # exact serial-Trie objective: sum of adjacent LCP of sorted strings
    # ------------------------------------------------------------------
    @staticmethod
    def _lcp(a: str, b: str) -> int:
        n = min(len(a), len(b))
        if n == 0 or a[0] != b[0]:
            return 0
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _score(self, strings: List[str]) -> int:
        if len(strings) < 2:
            return 0
        s = sorted(strings)
        total = 0
        prev = s[0]
        for cur in s[1:]:
            total += self._lcp(prev, cur)
            prev = cur
        return total

    # ------------------------------------------------------------------
    # candidate 1: original column order
    # ------------------------------------------------------------------
    def _candidate_original(self, ncols: int) -> List[List[int]]:
        base = list(range(ncols))
        return [base[:] for _ in range(self._nrows)]

    # ------------------------------------------------------------------
    # candidate 2: global frequency-weighted column order
    # ------------------------------------------------------------------
    def _global_order(
        self, cells: List[List[str]], cols: List[str], col_stop: Optional[int]
    ) -> List[int]:
        ncols = len(cols)
        weights = [0.0] * ncols
        for j, col in enumerate(cols):
            counts: Dict[str, int] = {}
            for row in cells:
                v = row[j]
                counts[v] = counts.get(v, 0) + 1
            w = 0.0
            for v, c in counts.items():
                if c > 1:
                    w += len(v) * c * (c - 1)
            weights[j] = w
        order = sorted(range(ncols), key=lambda j: (-weights[j], cols[j]))
        if col_stop is not None and col_stop > 0:
            order = order[:col_stop] + [j for j in range(ncols) if j not in set(order[:col_stop])]
        return order

    # ------------------------------------------------------------------
    # candidate 3: bounded conditional partition tree
    # ------------------------------------------------------------------
    def _candidate_tree(
        self,
        cells: List[List[str]],
        cols: List[str],
        global_order: List[int],
        col_stop: Optional[int],
        row_stop: Optional[int],
    ) -> List[List[int]]:
        ncols = len(cols)
        # precompute per-column value -> count (global) for candidate ranking
        col_counts: List[Dict[str, int]] = []
        for j in range(ncols):
            d: Dict[str, int] = {}
            for row in cells:
                v = row[j]
                d[v] = d.get(v, 0) + 1
            col_counts.append(d)

        # candidate columns per node: top few by global weight
        def col_weight(j: int, rows: List[int]) -> float:
            # length-weighted pair repetition restricted to current rows
            d: Dict[str, int] = {}
            for r in rows:
                v = cells[r][j]
                d[v] = d.get(v, 0) + 1
            w = 0.0
            for v, c in d.items():
                if c > 1:
                    w += len(v) * c * (c - 1)
            return w

        max_candidates = 8
        result: List[Optional[List[int]]] = [None] * self._nrows

        def assign_fixed(rows: List[int], used: List[int]):
            rest = [j for j in global_order if j not in used]
            order = used + rest
            for r in rows:
                result[r] = order[:]

        def build(rows: List[int], used: List[int], depth: int):
            if len(rows) < 2 or len(used) >= ncols:
                assign_fixed(rows, used)
                return
            if col_stop is not None and col_stop > 0 and depth >= col_stop:
                assign_fixed(rows, used)
                return
            if row_stop is not None and row_stop > 0 and depth >= row_stop:
                assign_fixed(rows, used)
                return
            remaining = [j for j in range(ncols) if j not in used]
            # rank remaining columns by weight within these rows
            scored = sorted(remaining, key=lambda j: (-col_weight(j, rows), cols[j]))
            scored = scored[:max_candidates]
            best_j = None
            best_groups = None
            for j in scored:
                groups: Dict[str, List[int]] = {}
                for r in rows:
                    groups.setdefault(cells[r][j], []).append(r)
                # only branching if it creates real structure
                if len(groups) < len(rows):
                    best_j = j
                    best_groups = groups
                    break
            if best_j is None:
                assign_fixed(rows, used)
                return
            # rows sharing the same value in best_j form subgroups
            for _, sub in best_groups.items():
                if len(sub) == len(rows):
                    # single group: no split gained, use next column ordering
                    assign_fixed(sub, used + [best_j])
                else:
                    build(sub, used + [best_j], depth + 1)

        build(list(range(self._nrows)), [], 0)
        for i in range(self._nrows):
            if result[i] is None:
                result[i] = list(range(ncols))
        return result

    # ------------------------------------------------------------------
    # build strings / dataframe from a per-row column order
    # ------------------------------------------------------------------
    @staticmethod
    def _serialize(cells: List[List[str]], order: List[int]) -> str:
        return "".join([cells[i][j] for i, j in enumerate(order)])

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
        # honor column merges using the parent's semantics
        if col_merge:
            for cols_to_merge in col_merge:
                ordered = [c for c in df.columns if c in cols_to_merge]
                if len(ordered) > 1:
                    df = self.merging_columns(df, ordered, prepended=False)

        df = df.reset_index(drop=True)
        cols = list(df.columns)
        ncols = len(cols)
        self._nrows = len(df)
        nrows = self._nrows

        if nrows == 0 or ncols == 0:
            return df.copy(), [[] for _ in range(nrows)] if nrows else []

        cells = self._string_cells(df)

        # candidate 1: original order
        cand1 = self._candidate_original(ncols)
        # candidate 2: global frequency order
        g_order = self._global_order(cells, cols, col_stop)
        cand2 = [g_order[:] for _ in range(nrows)]
        # candidate 3: conditional partition tree
        try:
            cand3 = self._candidate_tree(cells, cols, g_order, col_stop, row_stop)
        except Exception:
            cand3 = cand2

        best_orders = cand1
        best_score = self._score([self._serialize(cells, o) for o in cand1])
        for cand in (cand2, cand3):
            try:
                s = self._score([self._serialize(cells, o) for o in cand])
            except Exception:
                continue
            if s > best_score:
                best_score = s
                best_orders = cand

        # build output dataframe with per-row column orderings, preserving values
        out_values = []
        for i, order in enumerate(best_orders):
            out_values.append([cells[i][j] for j in order])

        # map cell strings back to original values (preserving stored values)
        # build a lookup per row from string -> original value
        obj = df.astype(object).where(pd.notna(df), None)
        col_orderings: List[List[str]] = []
        out_rows = []
        for i in range(nrows):
            order = best_orders[i]
            col_orderings.append([cols[j] for j in order])
            # reconstruct the row using original values, permuted per this order
            src = obj.iloc[i]
            out_rows.append([src.iloc[j] for j in order])

        reordered_df = pd.DataFrame(out_rows, columns=[f"c{k}" for k in range(ncols)] if False else cols[:0].__class__([]))
        # We keep original column labels as a generic container; the evaluator
        # uses per-row column orderings, so the frame uses positional columns.
        reordered_df = pd.DataFrame(out_rows)
        reordered_df.columns = [f"__col_{k}" for k in range(ncols)]
        reordered_df = reordered_df.astype(object)

        return reordered_df, col_orderings

# EVOLVE-BLOCK-END