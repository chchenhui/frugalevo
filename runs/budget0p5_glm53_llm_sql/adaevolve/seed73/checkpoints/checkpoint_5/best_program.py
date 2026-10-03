# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List
from collections import Counter


class Evolved(Algorithm):
    """
    Trie-aware prefix-caching optimizer (v2).

    Approach:
      1. Serialize every cell once per column (missing -> "", others -> str(v))
         into object ndarrays; keep raw values in a 2D object array for output.
      2. Build three cheap candidate per-row column orderings:
           (a) global ordering ranked by sum(len(v)*cnt*(cnt-1));
           (b) conditional partition tree selecting, inside each row group, the
               remaining field with the highest length-weighted pair repetition
               (len(v)*cnt*(cnt-1)), partitioning rows on that field's value;
           (c) the same conditional tree but with a pure repetition metric
               (cnt*(cnt-1), len as deterministic tie-break) — a distinct
               length/frequency tradeoff hypothesis.
         Tails inside tree leaves use group-local scores, not global ones.
      3. Score each candidate exactly with the serial character-Trie objective:
         sum of adjacent LCPs of the sorted serialized rows (order-independent
         for a fixed row multiset). LCPs use C-speed slice comparisons with
         binary search; identical strings short-circuit.
      4. Emit the best candidate: per-row permuted original values (object
         dtype, values untouched) plus matching per-row column orderings.
    All rows/values are preserved; only column order changes.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ---------------- serialization helpers ----------------

    @staticmethod
    def _ser_col(s: pd.Series) -> np.ndarray:
        """Serialize one column: missing -> '', everything else -> str(v)."""
        mask = s.isna().to_numpy()
        out = s.astype(str).to_numpy(dtype=object)
        if mask.any():
            out = np.where(mask, "", out)
        return out

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Longest common prefix length via C-speed slice equality + binsearch."""
        if a == b:
            return len(a)
        hi = len(a) if len(a) < len(b) else len(b)
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _score_ordering(self, row_strings) -> int:
        """Exact ideal Trie reuse: sum of adjacent LCPs after sorting."""
        ss = sorted(row_strings)
        total = 0
        prev = ss[0]
        lcp = self._lcp
        for cur in ss[1:]:
            total += lcp(prev, cur)
            prev = cur
        return total

    # ---------------- candidate constructions ----------------

    def _global_order(self, cells, n_rows, n_cols):
        """Rank columns by sum(len(v)*cnt*(cnt-1)) desc, index tie-break."""
        scores = []
        for c in range(n_cols):
            cnt = Counter(cells[c].tolist() if hasattr(cells[c], "tolist") else cells[c])
            sc = sum(len(v) * k * (k - 1) for v, k in cnt.items())
            scores.append((-sc, c))
        scores.sort()
        return [c for _, c in scores]

    def _conditional_orderings(self, cells, n_rows, n_cols, col_stop, early_stop, weighted):
        """Recursive conditional prefix partition tree -> per-row orderings.

        weighted=True uses len(v)*cnt*(cnt-1); weighted=False uses cnt*(cnt-1)
        with len(v)*cnt as deterministic tie-break. Leaf tails are ranked with
        group-local scores. Depth and group work are bounded.
        """
        # Bound total tree work on wide/large tables.
        max_depth = col_stop if col_stop else min(n_cols, 12)
        if n_rows * n_cols > 4_000_000:
            max_depth = min(max_depth, 6)
        orderings = [None] * n_rows
        lencache = {}

        def vlen(v):
            l = lencache.get(v)
            if l is None:
                l = len(v)
                lencache[v] = l
            return l

        def col_score(rows, c):
            col = cells[c]
            cnt = Counter(col[r] for r in rows)
            if weighted:
                return sum(vlen(v) * k * (k - 1) for v, k in cnt.items())
            return sum(k * (k - 1) for k in cnt.values())

        def tail_order(rows, remaining):
            sc = []
            for c in remaining:
                cnt = Counter(cells[c][r] for r in rows)
                if weighted:
                    s = sum(vlen(v) * k * (k - 1) for v, k in cnt.items())
                else:
                    s = sum(k * (k - 1) * vlen(v) for k, v in
                            ((k, v) for v, k in cnt.items()))
                sc.append((-s, c))
            sc.sort()
            return [c for _, c in sc]

        def assign(rows, order):
            for r in rows:
                orderings[r] = order

        def recurse(rows, remaining, prefix, depth):
            if not remaining or len(rows) <= 1 or depth >= max_depth:
                assign(rows, prefix + tail_order(rows, remaining))
                return
            best_col, best = None, -1
            for c in remaining:
                s = col_score(rows, c)
                if s > best:
                    best_col, best = c, s
            if best_col is None or best <= early_stop:
                assign(rows, prefix + tail_order(rows, remaining))
                return
            groups = {}
            col = cells[best_col]
            for r in rows:
                groups.setdefault(col[r], []).append(r)
            rest = [c for c in remaining if c != best_col]
            nxt = prefix + [best_col]
            for val in sorted(groups):
                recurse(groups[val], rest, nxt, depth + 1)

        recurse(list(range(n_rows)), list(range(n_cols)), [], 0)
        return orderings

    # ---------------- main entry ----------------

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

        if df is None or len(df.columns) == 0 or len(df) == 0:
            cols = list(df.columns) if df is not None else []
            n = 0 if df is None else len(df)
            return df, [cols] * n

        initial_shape = df.shape
        work = df

        # Honor column merges with existing API semantics.
        if col_merge:
            for group in col_merge:
                merge_order = [c for c in work.columns if c in group]
                if len(merge_order) > 1:
                    work = self.merging_columns(work, merge_order, prepended=False)

        n_rows, n_cols = work.shape
        col_names = list(work.columns)

        # Serialize all cells once, column-major object arrays.
        cells = [self._ser_col(work.iloc[:, c]) for c in range(n_cols)]
        # Raw original values for output construction.
        raw = work.to_numpy(dtype=object)

        # --- Candidate (a): global frequency-ranked ordering ---
        g_order = self._global_order(cells, n_rows, n_cols)
        candidates = [("global", [g_order] * n_rows)]

        # --- Candidates (b),(c): conditional partition trees ---
        cond_w = self._conditional_orderings(
            cells, n_rows, n_cols, col_stop, early_stop, weighted=True)
        if cond_w is not None:
            candidates.append(("cond_len", cond_w))
        cond_c = self._conditional_orderings(
            cells, n_rows, n_cols, col_stop, early_stop, weighted=False)
        if cond_c is not None:
            candidates.append(("cond_cnt", cond_c))

        # --- Score candidates with the exact serial-Trie objective ---
        best_orders, best_score = None, -1
        for _name, orders in candidates:
            strs = ["".join(cells[c][r] for c in orders[r]) for r in range(n_rows)]
            sc = self._score_ordering(strs)
            if sc > best_score:
                best_orders, best_score = orders, sc

        # Build output: per-row permuted original values, object dtype.
        data = [[raw[r][c] for c in best_orders[r]] for r in range(n_rows)]
        out = pd.DataFrame(data, columns=col_names, dtype=object)

        # Per-row column name orderings consistent with emitted data.
        orderings = [[col_names[c] for c in best_orders[r]] for r in range(n_rows)]

        assert out.shape == work.shape, "shape mismatch after reorder"
        if not col_merge:
            assert out.shape == initial_shape, "final shape mismatch"

        return out, orderings

# EVOLVE-BLOCK-END