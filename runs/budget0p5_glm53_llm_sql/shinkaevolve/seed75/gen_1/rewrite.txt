# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict


class Evolved(Algorithm):
    """
    Serial character-Trie oriented row/column reordering.

    Objective: maximize total shared prefix characters across the collection
    of serialized rows ("".join of str cells, missing -> ""), measured exactly
    as the sum of longest-common-prefix lengths of lexicographically adjacent
    serialized strings. Row order does not change this total, so we only need
    good per-row field orderings.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------ #
    # serialization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ser(v) -> str:
        """Scoring serialization of a single cell (does not alter stored data)."""
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
        return str(v)

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Longest common prefix via binary search with C-speed slice compares."""
        n = min(len(a), len(b))
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _score_orders(self, ser, orders, R) -> int:
        """Exact serial-Trie reuse: sum of LCPs of sorted serialized rows."""
        strs = ["".join(ser[r][c] for c in orders[r]) for r in range(R)]
        strs.sort()
        tot = 0
        lcp = self._lcp
        prev = None
        for s in strs:
            if prev is not None:
                tot += lcp(prev, s)
            prev = s
        return tot

    # ------------------------------------------------------------------ #
    # candidate constructions
    # ------------------------------------------------------------------ #
    def _global_freq_order(self, sercols, C, names) -> List[int]:
        """Order columns by sum(len(v) * count * (count-1)) over serialized values."""
        scores = []
        for c in range(C):
            cnt = {}
            col = sercols[c]
            for v in col:
                if v:
                    cnt[v] = cnt.get(v, 0) + 1
            sc = sum(len(v) * n * (n - 1) for v, n in cnt.items())
            scores.append(sc)
        order = sorted(range(C), key=lambda c: (-scores[c], names[c]))
        return order

    def _global_len_order(self, sercols, C, names) -> List[int]:
        """Alternative tradeoff: total serialized characters, descending."""
        scores = [sum(len(v) for v in sercols[c]) for c in range(C)]
        return sorted(range(C), key=lambda c: (-scores[c], names[c]))

    def _tree_orders(self, ser, R, C, base_order, max_depth, max_nodes=4000):
        """Conditional prefix partition tree.

        Within each row group, pick the remaining field with the highest
        length-weighted pair repetition, put it first for the whole group,
        then partition rows by that field's serialized value and choose
        suffix orders separately inside each partition.
        """
        orders = [None] * R
        base_set = list(base_order)
        nodes = [0]

        def emit(rows, cols):
            o = [c for c in base_set if c in cols]
            for r in rows:
                orders[r] = o

        def rec(rows, cols, depth):
            nodes[0] += 1
            if (
                len(rows) <= 1
                or not cols
                or depth >= max_depth
                or nodes[0] > max_nodes
            ):
                emit(rows, cols)
                return
            best, best_sc = None, 0
            for c in cols:
                cnt = {}
                for r in rows:
                    v = ser[r][c]
                    if v:
                        cnt[v] = cnt.get(v, 0) + 1
                sc = sum(len(v) * n * (n - 1) for v, n in cnt.items())
                if sc > best_sc:
                    best_sc, best = sc, c
            if best is None or best_sc <= 0:
                emit(rows, cols)
                return
            rest = [c for c in cols if c != best]
            head = [best] + [c for c in base_set if c in rest]
            for r in rows:
                orders[r] = head
            groups: Dict[str, List[int]] = {}
            for r in rows:
                groups.setdefault(ser[r][best], []).append(r)
            for g in groups.values():
                if len(g) > 1:
                    rec(g, rest, depth + 1)

        rec(list(range(R)), list(range(C)), 0)
        # safety: every row must have an order
        for r in range(R):
            if orders[r] is None:
                orders[r] = list(base_set)
        return orders

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
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
        # honor column merges with the existing API semantics
        work = df.copy()
        if col_merge:
            for col_to_merge in col_merge:
                present = [c for c in work.columns if c in col_to_merge]
                if len(present) > 1:
                    work = self.merging_columns(work, present, prepended=False)

        R, C = work.shape
        if R == 0:
            return work.copy(), []
        if C == 0:
            return work.copy(), [[] for _ in range(R)]

        names = [str(c) for c in work.columns]

        # serialize every cell once, reuse across all candidate constructions
        sercols = [work.iloc[:, c].map(self._ser).tolist() for c in range(C)]
        ser = [[sercols[c][r] for c in range(C)] for r in range(R)]

        # candidate 1: global frequency-ranked order
        freq_order = self._global_freq_order(sercols, C, names)
        candidates = [("freq", [freq_order] * R)]

        # candidate 2: deterministic length-ranked order (different tradeoff)
        if C > 1:
            len_order = self._global_len_order(sercols, C, names)
            if len_order != freq_order:
                candidates.append(("len", [len_order] * R))

        # candidate 3: conditional partition tree (bounded)
        max_depth = col_stop if col_stop else 10
        if row_stop is not None:
            max_depth = min(max_depth, max(1, int(row_stop)))
        budget = R * C
        if C > 1 and R > 1 and budget <= 4_000_000:
            tree = self._tree_orders(ser, R, C, freq_order, max_depth)
            if any(tree[r] != freq_order for r in range(R)):
                candidates.append(("tree", tree))

        # select using the real serial-Trie character objective
        best_orders, best_score = candidates[0][1], -1
        for _, orders in candidates:
            sc = self._score_orders(ser, orders, R)
            if sc > best_score:
                best_score, best_orders = sc, orders

        # emit: permute each row's original values per its ordering
        data = [[work.iat[r, c] for c in best_orders[r]] for r in range(R)]
        out = pd.DataFrame(data, columns=list(work.columns), index=work.index)
        out = out.astype(object)

        column_orderings = [
            [work.columns[c] for c in best_orders[r]] for r in range(R)
        ]
        return out, column_orderings

# EVOLVE-BLOCK-END