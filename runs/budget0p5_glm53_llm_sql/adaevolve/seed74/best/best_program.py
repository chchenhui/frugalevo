# EVOLVE-BLOCK-START
"""
Optimized for exact serial character-Trie prefix reuse.

Build a small bounded set of cheap candidate per-row column orderings, score
each once (deduped by signature) with the true Trie objective (sum of adjacent
LCPs over sorted serialized row strings), and return the best. Candidates:
  A)  global pair-repetition order (per-column score sum_v len(v)*cnt*(cnt-1))
  A') global savings order (per-column score sum_v len(v)*(cnt-1)) — a
      different length/frequency tradeoff: favors columns with many distinct
      repeated values over one hugely duplicated value
  B)  conditional partition tree, split by PAIR repetition, GLOBAL tail
  C)  same tree with a LOCAL frequency tail at leaf groups
  D)  tree split by SAVINGS, GLOBAL tail
  D') tree split by SAVINGS, LOCAL tail (skipped on very large inputs)
  E)  original column order (cheap baseline, skipped on huge inputs)
All group-level statistics run on precomputed integer factorization codes
(vectorized bincount), never pandas per-group operations. The exact LCP
objective uses C-speed prefix-slice binary search, never per-character loops.
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
        g_pair = [0] * p
        g_sav = [0] * p
        for j in range(p):
            colj = col_strs[j]
            cnt = Counter(colj)
            g_pair[j] = sum(len(v) * k * (k - 1) for v, k in cnt.items())
            g_sav[j] = sum(len(v) * (k - 1) for v, k in cnt.items())
            uniq = sorted(cnt)
            code_map = {v: i for i, v in enumerate(uniq)}
            codes[j] = np.fromiter((code_map[v] for v in colj), dtype=np.int64, count=n)
            vlen[j] = np.array([len(v) for v in uniq], dtype=np.int64)

        # ---- Candidates A / A': two global frequency-ranked orders ----
        pair_order = sorted(range(p), key=lambda j: (-g_pair[j], j))
        sav_order = sorted(range(p), key=lambda j: (-g_sav[j], j))
        cand_global_pair = [pair_order] * n
        cand_global_sav = [sav_order] * n

        # ---- conditional partition trees ----
        max_depth = col_stop if col_stop else 8
        max_depth = max(1, min(max_depth, p, 12))

        def make_scorer(mode: str):
            """Group-level column score over precomputed integer codes.

            'pair':    sum len*k*(k-1)  (length-weighted pair repetition)
            'savings': sum len*(k-1)    (per-value savings; favors many
                                         distinct repeated values)
            """
            if mode == "pair":
                def f(j: int, rows: np.ndarray) -> int:
                    k = np.bincount(codes[j][rows], minlength=len(vlen[j]))
                    return int((vlen[j] * k * (k - 1)).sum())
            else:
                def f(j: int, rows: np.ndarray) -> int:
                    k = np.bincount(codes[j][rows], minlength=len(vlen[j]))
                    return int((vlen[j] * np.maximum(k - 1, 0)).sum())
            return f

        pair_rep = make_scorer("pair")
        sav_rep = make_scorer("savings")

        def build_tree(tail_local: bool, split_score, base_order):
            """Recursive conditional prefix partition tree.

            At each node, pick the remaining column with the highest
            split_score within the current row group, partition rows by its
            factorized value, and recurse on remaining columns. Leaves get a
            global (or local) frequency-ranked tail over remaining columns.
            """
            orders_idx = [base_order] * n

            def leaf_tail(rows: np.ndarray, avail: List[int]) -> List[int]:
                if tail_local and len(rows) > 1:
                    return sorted(avail, key=lambda c: (-pair_rep(c, rows), c))
                aset = set(avail)
                return [c for c in base_order if c in aset]

            def build(rows: np.ndarray, avail: List[int], depth: int):
                if len(rows) <= 1 or depth == 0 or not avail:
                    tail = leaf_tail(rows, avail)
                    for r in rows.tolist():
                        orders_idx[r] = tail
                    return
                best_c, best_s = -1, early_stop
                for c in avail:
                    s = split_score(c, rows)
                    if s > best_s:
                        best_s, best_c = s, c
                if best_c < 0:
                    tail = leaf_tail(rows, avail)
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

        big = n * p > 5_000_000

        candidates = [cand_global_pair, cand_global_sav,
                      build_tree(False, pair_rep, pair_order),
                      build_tree(True, pair_rep, pair_order),
                      build_tree(False, sav_rep, sav_order)]
        if not big:
            candidates.append(build_tree(True, sav_rep, sav_order))

        # ---- score candidates with the exact Trie objective (deduped) ----
        def score(orders):
            strs = ["".join(col_strs[c][i] for c in orders[i]) for i in range(n)]
            return self._lcp_sum(strs)

        best_orders, best_score = cand_global_pair, score(cand_global_pair)
        scored_signatures = {tuple(pair_order)}

        for cand in candidates:
            # cheap signature dedup: if every row shares one ordering,
            # score it only once across candidates
            first = cand[0]
            uniform = True
            for o in cand:
                if o is not first and o != first:
                    uniform = False
                    break
            if uniform:
                sig = tuple(first)
                if sig in scored_signatures:
                    continue
                scored_signatures.add(sig)
            s = score(cand)
            if s > best_score:
                best_orders, best_score = cand, s

        if not big and p > 1:
            orig = [list(range(p))] * n
            sig = tuple(range(p))
            if sig not in scored_signatures:
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