# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from collections import Counter
from solver import Algorithm
from typing import Tuple, List


class Evolved(Algorithm):
    """
    Character-Trie oriented reordering, lean runtime version.

    Ideal serial Trie reuse for a fixed multiset of serialized row strings is
    insertion-order independent and equals the sum of LCPs of lexicographically
    adjacent sorted strings. Since the combined score's runtime term rewards
    speed, this version keeps only the cheap, highly effective candidates and
    scores them exactly:

    1. Serialize every cell once per column (missing -> "", otherwise str),
       matching the evaluator's fillna("").astype(str) semantics.
    2. Build two cheap candidate per-row column orderings:
         A) global order ranked by sum(len(v) * count(v) * (count(v)-1)),
            deterministic tie-break by column index;
         C) the original column order as a reliable baseline.
    3. Score each candidate with the exact sorted-string adjacent-LCP sum
       (identical strings deduplicated and counted analytically; LCP via
       binary search on C-speed slice equality) and return the best.

    The returned DataFrame preserves every original cell value (object dtype
    for mixed types); only per-row column order changes. col_merge groups
    are kept contiguous in every row ordering.
    """

    # ---------- scoring helpers ----------

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Longest common prefix via binary search on slice equality (C-level)."""
        hi = min(len(a), len(b))
        if hi == 0:
            return 0
        if a == b:
            return hi
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _score(self, S, orders, n, m):
        """Exact ideal Trie reuse = adjacent-LCP sum over sorted distinct row strings.

        A run of k copies of a string of length L contributes (k-1)*L analytically;
        only distinct strings go through sort + pairwise LCP.
        """
        cnt = Counter(
            "".join(S[orders[r][c]][r] for c in range(m)) for r in range(n)
        )
        total = 0
        strs = []
        for s, k in cnt.items():
            if k > 1:
                total += (k - 1) * len(s)
            strs.append(s)
        if len(strs) < 2:
            return total
        strs.sort()
        lcp = self._lcp
        total += sum(lcp(strs[i - 1], strs[i]) for i in range(1, len(strs)))
        return total

    # ---------- col_merge adjacency ----------

    @staticmethod
    def _fix_adjacency(order, col_pos, col_merge):
        """Keep each col_merge group contiguous (group order preserved)."""
        order = list(order)
        for group in col_merge:
            gset = [col_pos[c] for c in group if c in col_pos]
            if len(gset) < 2:
                continue
            gset_set = set(gset)
            pos = min(i for i, j in enumerate(order) if j in gset_set)
            order = [j for j in order if j not in gset_set]
            order[pos:pos] = gset
        return order

    # ---------- public API ----------

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
        n, m = df.shape
        if n == 0 or m == 0:
            return df.copy(), [list(df.columns)] * n
        if m == 1:
            return df.copy(), [list(df.columns)] * n

        cols = list(df.columns)
        col_pos = {c: i for i, c in enumerate(cols)}

        # Serialize each cell once, vectorized per column (missing -> "").
        S = []
        for j in range(m):
            s = df.iloc[:, j]
            S.append(s.astype(str).where(s.notna(), "").tolist())

        # Global length-weighted pair-repetition ranking (candidate A key):
        # sum over serialized values v of len(v) * count(v) * (count(v)-1).
        col_score = []
        for j in range(m):
            c = Counter(S[j])
            col_score.append(sum(len(v) * k * (k - 1) for v, k in c.items()))
        orderA = sorted(range(m), key=lambda j: (-col_score[j], j))
        orderC = list(range(m))

        # Score both cheap candidates exactly; best wins.
        candidates = [("A", [orderA] * n)]
        if orderC != orderA:
            candidates.append(("C", [orderC] * n))

        best_sc, best_orders = -1.0, [orderA] * n
        for _, orders in candidates:
            try:
                sc = self._score(S, orders, n, m)
            except Exception:
                continue
            if sc > best_sc:
                best_sc, best_orders = sc, orders

        orders = best_orders
        if col_merge:
            orders = [self._fix_adjacency(o, col_pos, col_merge) for o in orders]

        # Build output preserving original cell values, per-row permutation.
        out = np.empty((n, m), dtype=object)
        colnames = []
        row_vals = df.values  # object view of original values
        for r in range(n):
            o = orders[r]
            rv = row_vals[r]
            for c in range(m):
                out[r, c] = rv[o[c]]
            colnames.append([cols[j] for j in o])

        result = pd.DataFrame(out, columns=cols, index=df.index)
        return result, colnames
# EVOLVE-BLOCK-END