# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from collections import Counter
from solver import Algorithm
from typing import Tuple, List


class Evolved(Algorithm):
    """
    Character-Trie oriented reordering.

    The evaluator measures serial character-Trie reuse, which for a fixed
    multiset of row strings equals total length minus distinct character
    edges, computable as the sum of LCPs of lexicographically adjacent
    sorted row strings (insertion-order independent). We:

    1. Serialize every cell once (missing -> "", str otherwise) into a
       column-major string table S.
    2. Build up to three cheap candidate per-row column orderings:
        A) global order ranked by sum(len(v) * count(v) * (count(v)-1)),
           deterministic tie-break by column index;
        B) a bounded conditional prefix partition tree: inside each row
           group, pick the remaining column with the largest
           length-weighted pair repetition (count*(count-1)), partition
           rows by its factorized value codes, and recurse (depth/group
           size bounded, distinct-heavy columns excluded via
           distinct_value_threshold, candidate columns capped);
        C) the original column order as a reliable baseline.
    3. Score each candidate with the exact sorted-string adjacent-LCP sum
       (binary-search LCP using C-speed slice equality), return the best.

    The returned DataFrame preserves every original cell value (object
    dtype for mixed types); only per-row column order changes. col_merge
    groups are kept contiguous in every row ordering.
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
        """Exact ideal Trie reuse = sum of adjacent LCPs of sorted row strings."""
        strs = ["".join(S[orders[r][c]][r] for c in range(m)) for r in range(n)]
        strs.sort()
        lcp = self._lcp
        return sum(lcp(strs[i - 1], strs[i]) for i in range(1, n))

    # ---------- candidate B: conditional grouping ----------

    def _group_orders(self, S, col_score, n, m, thr, orderA):
        """Bounded conditional partition tree producing per-row column orders."""
        codes = []       # per column: int code per row
        uniq_lens = []   # per column: serialized length per code
        for j in range(m):
            d = {}
            cd = np.empty(n, dtype=np.int64)
            ls = []
            for r in range(n):
                v = S[j][r]
                k = d.get(v)
                if k is None:
                    k = len(ls)
                    d[v] = k
                    ls.append(len(v))
                cd[r] = k
            codes.append(cd)
            uniq_lens.append(ls)

        # Exclude highly distinct columns from being branching keys.
        excl = {j for j in range(m) if len(uniq_lens[j]) > thr * n}

        orders = [None] * n
        maxdepth = min(8, m)
        all_cols = list(range(m))

        def fixed(rem):
            rem_set = set(rem)
            return [j for j in orderA if j in rem_set]

        def rec(rows, rem, depth):
            if not rem or len(rows) < 2 or depth >= maxdepth:
                fo = fixed(rem)
                for r in rows:
                    orders[r] = fo
                return
            cand = [j for j in rem if j not in excl]
            cand.sort(key=lambda j: -col_score[j])
            cand = cand[:10]
            bestj, bestsc = -1, 0
            for j in cand:
                cd = codes[j][rows]
                cnt = np.bincount(cd)
                ls = uniq_lens[j]
                sc = 0
                for k in range(len(cnt)):
                    c = cnt[k]
                    if c > 1:
                        sc += ls[k] * c * (c - 1)
                if sc > bestsc:
                    bestsc, bestj = sc, j
            if bestj < 0:
                fo = fixed(rem)
                for r in rows:
                    orders[r] = fo
                return
            cd = codes[bestj][rows]
            newrem = [j for j in rem if j != bestj]
            groups = {}
            for idx, r in enumerate(rows):
                groups.setdefault(cd[idx], []).append(r)
            for g in groups.values():
                rec(g, newrem, depth + 1)
                for r in g:
                    orders[r] = [bestj] + orders[r]

        rec(list(range(n)), all_cols, 0)
        for r in range(n):
            if orders[r] is None:
                orders[r] = list(orderA)
        return orders

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

        cols = list(df.columns)
        col_pos = {c: i for i, c in enumerate(cols)}

        # Serialize each cell once (column-major table S).
        S = []
        for j in range(m):
            S.append(
                ["" if pd.isna(v) else (v if isinstance(v, str) else str(v))
                 for v in df.iloc[:, j]]
            )

        # Global frequency/length ranking (candidate A key).
        col_score = []
        for j in range(m):
            c = Counter(S[j])
            col_score.append(sum(len(v) * k * (k - 1) for v, k in c.items()))
        orderA = sorted(range(m), key=lambda j: (-col_score[j], j))

        # Candidate orderings.
        candidates = [("A", [orderA] * n)]
        try:
            orderB = self._group_orders(S, col_score, n, m,
                                        distinct_value_threshold, orderA)
            candidates.append(("B", orderB))
        except Exception:
            pass
        orderC = list(range(m))
        candidates.append(("C", [orderC] * n))

        best_sc, best_orders = -1.0, [orderA] * n
        for _, orders in candidates:
            sc = self._score(S, orders, n, m)
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