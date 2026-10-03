# EVOLVE-BLOCK-START
"""
Optimized for exact serial character-Trie prefix reuse.

Build at most three cheap candidate per-row column orderings, score each with
the true Trie objective (sum of adjacent LCPs over sorted serialized row
strings), and return the best. Candidates:
  A) global frequency-ranked order (per-column score sum_v len(v)*cnt*(cnt-1))
  B) conditional prefix partition tree over precomputed integer codes, with a
     GLOBAL frequency tail when branching stops
  C) same partition tree but with a LOCAL frequency tail: at each leaf group,
     remaining columns are ranked by their pair repetition *within that group*
  D) original column order (cheap baseline, skipped on huge inputs)
Row order does not affect the exact Trie total, so rows are left in place.
Data is never altered: only column order changes, plus per-row orderings.
Missing values serialize to '' exactly like the scorer.
"""
import numpy as np
import pandas as pd
from solver import Algorithm
from typing import Tuple, List
from collections import Counter


class Evolved(Algorithm):

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ---------- helpers ----------

    @staticmethod
    def _cell_str(v) -> str:
        """Serialize one cell exactly like the scorer: missing -> ''."""
        if v is None:
            return ""
        try:
            if v != v:  # NaN / NaT / pd.NA sentinels
                return ""
        except Exception:
            pass
        return str(v)

    @staticmethod
    def _lcp_sum(strings: List[str]) -> int:
        """Exact Trie reuse = sum of adjacent LCPs of sorted strings.

        Identical strings handled directly; otherwise binary search the
        common prefix length using prefix-slice equality so comparisons run
        in C (no per-character Python loop).
        """
        if len(strings) < 2:
            return 0
        ss = sorted(strings)
        total = 0
        prev = ss[0]
        for cur in ss[1:]:
            if cur == prev:
                total += len(cur)
            else:
                lo, hi = 0, min(len(prev), len(cur))
                while lo < hi:
                    mid = (lo + hi + 1) >> 1
                    if prev[:mid] == cur[:mid]:
                        lo = mid
                    else:
                        hi = mid - 1
                total += lo
            prev = cur
        return total

    # ---------- main API ----------

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
        out = df.copy()
        # honor column merges with existing API semantics
        if col_merge:
            try:
                for grp in col_merge:
                    final = [c for c in out.columns if c in grp]
                    if len(final) > 1:
                        out = self.merging_columns(out, final, prepended=False)
            except Exception:
                pass

        n, p = out.shape
        col_names = list(out.columns)
        if n == 0 or p == 0:
            return out, [list(col_names) for _ in range(n)]

        arr = out.to_numpy(dtype=object)
        # serialized cells per column (reused across all candidates)
        col_strs = [[self._cell_str(v) for v in arr[:, j].tolist()] for j in range(p)]

        # ---- precompute per-column: factorization codes + per-code lengths ----
        codes = [None] * p
        vlen = [None] * p
        gscore = [0] * p
        for j in range(p):
            colj = col_strs[j]
            cnt = Counter(colj)
            gscore[j] = sum(len(v) * k * (k - 1) for v, k in cnt.items())
            uniq = sorted(cnt)
            code_map = {v: i for i, v in enumerate(uniq)}
            codes[j] = np.fromiter((code_map[v] for v in colj), dtype=np.int64, count=n)
            vlen[j] = np.array([len(v) for v in uniq], dtype=np.int64)

        # ---- Candidate A: global frequency-ranked column order ----
        global_order = sorted(range(p), key=lambda j: (-gscore[j], j))
        cand_global = [global_order] * n

        # ---- Candidates B / C: conditional prefix partition tree ----
        max_depth = col_stop if col_stop else 8
        max_depth = max(1, min(max_depth, p, 12))

        def pair_rep(j: int, rows: np.ndarray) -> int:
            """Length-weighted pair repetition of column j within rows (vectorized)."""
            k = np.bincount(codes[j][rows], minlength=len(vlen[j]))
            return int((vlen[j] * k * (k - 1)).sum())

        def local_tail(rows: np.ndarray, avail: List[int]) -> List[int]:
            """Rank remaining columns by their pair repetition inside this group."""
            return sorted(avail, key=lambda c: (-pair_rep(c, rows), c))

        def build_tree(local: bool) -> List[List[int]]:
            orders_idx = [global_order] * n

            def build(rows: np.ndarray, avail: List[int], depth: int):
                if len(rows) <= 1 or depth == 0 or not avail:
                    if local and len(rows) > 1:
                        tail = local_tail(rows, avail)
                    else:
                        aset = set(avail)
                        tail = [c for c in global_order if c in aset]
                    for r in rows.tolist():
                        orders_idx[r] = tail
                    return
                best_c, best_s = -1, early_stop
                for c in avail:
                    s = pair_rep(c, rows)
                    if s > best_s:
                        best_s, best_c = s, c
                if best_c < 0:
                    if local and len(rows) > 1:
                        tail = local_tail(rows, avail)
                    else:
                        aset = set(avail)
                        tail = [c for c in global_order if c in aset]
                    for r in rows.tolist():
                        orders_idx[r] = tail
                    return
                cvals = codes[best_c][rows]
                navail = [c for c in avail if c != best_c]
                order = np.argsort(cvals, kind="stable")
                sorted_vals = cvals[order]
                bounds = np.flatnonzero(np.diff(sorted_vals)) + 1
                for grp_rows in np.split(rows[order], bounds):
                    build(grp_rows, navail, depth - 1)
                    for r in grp_rows.tolist():
                        orders_idx[r] = [best_c] + orders_idx[r]

            build(np.arange(n, dtype=np.int64), list(range(p)), max_depth)
            return orders_idx

        cand_cond_g = build_tree(local=False)
        cand_cond_l = build_tree(local=True)

        # ---- score candidates with the exact Trie objective ----
        def score(orders):
            strs = ["".join(col_strs[c][i] for c in orders[i]) for i in range(n)]
            return self._lcp_sum(strs)

        big = n * p > 5_000_000
        s_global = score(cand_global)
        best_orders, best_score = cand_global, s_global
        for cand in (cand_cond_g, cand_cond_l):
            if cand is cand_global or cand == cand_global:
                continue
            s = score(cand)
            if s > best_score:
                best_orders, best_score = cand, s
        if not big and p > 1:
            orig = [list(range(p))] * n
            s_orig = score(orig)
            if s_orig > best_score:
                best_orders, best_score = orig, s_orig

        # ---- build output ----
        # physical column layout: the most common per-row order (deterministic)
        most_common_order = Counter(tuple(o) for o in best_orders).most_common(1)[0][0]
        phys = list(most_common_order)
        result = out.iloc[:, phys].copy()
        result.columns = [col_names[j] for j in phys]
        column_orderings = [[col_names[j] for j in best_orders[i]] for i in range(n)]

        assert result.shape == out.shape
        return result, column_orderings

# EVOLVE-BLOCK-END