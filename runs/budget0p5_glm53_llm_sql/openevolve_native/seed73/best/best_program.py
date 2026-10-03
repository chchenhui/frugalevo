# EVOLVE-BLOCK-START
import numpy as np
import pandas as pd
from solver import Algorithm
from typing import Tuple, List


def _lcp(a: str, b: str) -> int:
    """LCP via binary search on prefix slices (C-speed comparisons)."""
    if a == b:
        return len(a)
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class Evolved(Algorithm):
    """
    Character-Trie prefix caching optimizer.

    Serialize cells once, precompute per-column value counts once, then build
    a bounded set of cheap candidate per-row column orderings:
      - identity, global pair-ranked (both directions), char-ranked (both),
      - mode-first per-row ranking,
      - frequency-ranked per-row ordering (each row puts its most repeated /
        longest values first, weighted by count*(count-1)*len),
      - conditional partition trees (pair- and char-weighted) with two
        different deterministic tails,
      - a greedy global column order selected by actual prefix LCP score
        (only when cheap enough).
    Every candidate is scored exactly once with the ideal serial-Trie
    objective (sum of adjacent LCPs over sorted serialized rows) and the
    best is returned. Original cell values are preserved (object dtype).
    """

    def _serialize(self, df: pd.DataFrame) -> np.ndarray:
        """Cell-string matrix matching evaluator serialization; NaN/None/NaT -> ''."""
        arr = df.to_numpy(dtype=object)
        n, p = arr.shape
        out = np.empty((n, p), dtype=object)
        isna = pd.isna
        for j in range(p):
            col = arr[:, j]
            out[:, j] = ["" if isna(v) else str(v) for v in col]
        return out

    @staticmethod
    def _col_counts(S: np.ndarray) -> List[dict]:
        """Per-column value -> count maps, computed once and shared."""
        return [{v: c for v, c in pd.Series(S[:, j].tolist()).value_counts().items()}
                for j in range(S.shape[1])]

    def _score(self, row_strings: List[str]) -> int:
        """Exact ideal Trie reuse: sum of adjacent LCPs of sorted row strings."""
        s = sorted(row_strings)
        tot = 0
        for i in range(1, len(s)):
            tot += _lcp(s[i - 1], s[i])
        return tot

    def _tree_orders(self, S: np.ndarray, early_stop: int, max_depth: int,
                     eligible: List[int], global_pos: List[int],
                     char_weighted: bool) -> List[List[int]]:
        """Conditional prefix partition tree producing per-row column orders.

        At each node, pick the eligible remaining column with the highest
        length-weighted repetition (pair- or char-weighted), partition rows by
        its serialized value, and recurse. Leaves order remaining columns by
        the precomputed global ranking restricted to the remaining set.
        """
        n, p = S.shape
        orders: List[List[int]] = [None] * n
        eligible_set = set(eligible) if eligible else set(range(p))
        tail_cache = {}

        def tail_order(remaining: List[int]) -> List[int]:
            key = tuple(remaining)
            t = tail_cache.get(key)
            if t is None:
                rem = set(remaining)
                t = [j for j in global_pos if j in rem]
                tail_cache[key] = t
            return t

        def rec(rows: List[int], remaining: List[int], depth: int):
            if len(rows) <= 1 or not remaining or depth >= max_depth:
                tail = tail_order(remaining)
                for r in rows:
                    orders[r] = tail
                return
            sub = S[np.ix_(rows, remaining)]
            best_k, best_sc = -1, 0
            for k in range(len(remaining)):
                if remaining[k] not in eligible_set:
                    continue
                sc = 0
                cnt = {}
                for v in sub[:, k].tolist():
                    cnt[v] = cnt.get(v, 0) + 1
                for v, c in cnt.items():
                    if char_weighted:
                        sc += len(v) * c
                    elif c > 1:
                        sc += len(v) * c * (c - 1)
                if sc > best_sc:
                    best_sc, best_k = sc, k
            if best_k < 0 or best_sc <= early_stop:
                tail = tail_order(remaining)
                for r in rows:
                    orders[r] = tail
                return
            j = remaining[best_k]
            rest = remaining[:best_k] + remaining[best_k + 1:]
            groups = {}
            for r, v in zip(rows, sub[:, best_k].tolist()):
                groups.setdefault(v, []).append(r)
            for grows in groups.values():
                if len(grows) == 1:
                    orders[grows[0]] = [j] + tail_order(rest)
                else:
                    rec(grows, rest, depth + 1)
                    for r in grows:
                        orders[r] = [j] + orders[r]

        rec(list(range(n)), list(range(p)), 0)
        for r in range(n):
            if orders[r] is None:
                orders[r] = list(range(p))
        return orders

    def _mode_first_orders(self, S: np.ndarray, gpos: List[int],
                           counts: List[dict]) -> List[List[int]]:
        """Per-row ordering: columns whose value equals the column mode come
        first (ordered by global pair rank), then the rest (same rank)."""
        n, p = S.shape
        modes = []
        for j in range(p):
            best_v, best_c = None, -1
            for v, c in counts[j].items():
                if c > best_c or (c == best_c and (best_v is None or v < best_v)):
                    best_v, best_c = v, c
            modes.append(best_v)
        orders = []
        for r in range(n):
            first = [j for j in gpos if S[r, j] == modes[j]]
            rest = [j for j in gpos if S[r, j] != modes[j]]
            orders.append(first + rest)
        return orders

    def _freq_orders(self, S: np.ndarray, counts: List[dict],
                     pair_weighted: bool) -> List[List[int]]:
        """Per-row ordering by the row's own value repetition weight.

        Each row places columns whose value is heavily repeated (and long)
        first: weight = len(v)*c*(c-1) if pair_weighted else len(v)*c.
        Deterministic tie break on column index.
        """
        n, p = S.shape
        W = np.empty((n, p), dtype=np.int64)
        for j in range(p):
            cj = counts[j]
            if pair_weighted:
                W[:, j] = [len(v) * (c * (c - 1) if c > 1 else 0)
                           for v, c in ((v, cj[v]) for v in S[:, j].tolist())]
            else:
                W[:, j] = [len(v) * cj[v] for v in S[:, j].tolist()]
        orders = []
        Wl = W.tolist()
        for r in range(n):
            orders.append(sorted(range(p), key=lambda j: (-Wl[r][j], j)))
        return orders

    def _greedy_global(self, S: np.ndarray) -> List[List[int]]:
        """Greedy global column order chosen by actual prefix LCP score.

        Iteratively append the column that maximizes the exact serial-Trie
        reuse of the rows serialized with the chosen prefix columns only.
        Bounded: skipped when the estimated work is too large.
        """
        n, p = S.shape
        avg_len = sum(len(S[0, j]) for j in range(p)) if p else 0
        # rough op estimate: p^2 scoring rounds * n rows * avg prefix length
        if n * p * p * max(avg_len, 1) > 3e8:
            return None
        chosen: List[int] = []
        remaining = list(range(p))
        # cache of current prefix strings per row
        cur = [""] * n
        while remaining:
            best_j, best_sc, best_strs = None, -1, None
            for j in remaining:
                col = S[:, j]
                cand = [cur[r] + col[r] for r in range(n)]
                sc = self._score(cand)
                if sc > best_sc:
                    best_sc, best_j, best_strs = sc, j, cand
            chosen.append(best_j)
            remaining.remove(best_j)
            cur = best_strs
        # deterministic tail: remaining is empty; order is global
        return [chosen] * n if len(chosen) == p else None

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
        df = df.copy()
        if col_merge and hasattr(self, "merging_columns"):
            for cols in col_merge:
                present = [c for c in df.columns if c in cols]
                if len(present) > 1:
                    df = self.merging_columns(df, present, prepended=False)

        n, p = df.shape
        if n == 0 or p == 0:
            if n == 0:
                return df.copy(), []
            return df.copy(), [list(df.columns) for _ in range(n)]

        S = self._serialize(df)
        col_names = list(df.columns)
        counts = self._col_counts(S)

        pair_scores = [sum(len(v) * (c * (c - 1) if c > 1 else 0)
                           for v, c in counts[j].items()) for j in range(p)]
        char_scores = [sum(len(v) * c for v, c in counts[j].items())
                       for j in range(p)]

        identity = [list(range(p)) for _ in range(n)]

        gpos = sorted(range(p), key=lambda j: (-pair_scores[j], j))
        gorder = [gpos] * n
        rpos = list(reversed(gpos))
        rorder = [rpos] * n

        cpos = sorted(range(p), key=lambda j: (-char_scores[j], j))
        corder = [cpos] * n
        crev = list(reversed(cpos))
        crevorder = [crev] * n

        morder = self._mode_first_orders(S, gpos, counts)
        fq_pair = self._freq_orders(S, counts, pair_weighted=True)
        fq_char = self._freq_orders(S, counts, pair_weighted=False)

        max_depth = min(p, 12)
        if col_stop is not None:
            max_depth = min(max_depth, max(1, int(col_stop)))
        thresh = max(1, int(distinct_value_threshold * n))
        eligible = [j for j in range(p) if len(counts[j]) <= thresh]
        if not eligible:
            eligible = list(range(p))
        tree_pair = self._tree_orders(S, early_stop, max_depth, eligible, gpos,
                                      char_weighted=False)
        tree_char = self._tree_orders(S, early_stop, max_depth, eligible, gpos,
                                      char_weighted=True)
        tree_pair_c = self._tree_orders(S, early_stop, max_depth, eligible, cpos,
                                        char_weighted=False)

        candidates = [identity, gorder, rorder, corder, crevorder, morder,
                      fq_pair, fq_char, tree_pair, tree_char, tree_pair_c]
        greedy = self._greedy_global(S) if p > 1 else None
        if greedy is not None:
            candidates.append(greedy)

        best_orders, best_score = identity, -1
        for cand in candidates:
            rows_str = ["".join(S[r, cand[r]].tolist()) for r in range(n)]
            sc = self._score(rows_str)
            if sc > best_score:
                best_score, best_orders = sc, cand

        # build output from ORIGINAL values (object dtype, no coercion)
        vals = df.to_numpy(dtype=object)
        out = np.empty((n, p), dtype=object)
        for r in range(n):
            out[r] = vals[r, best_orders[r]]
        result = pd.DataFrame(out, columns=col_names, index=df.index)
        column_orderings = [[col_names[j] for j in best_orders[r]] for r in range(n)]
        return result, column_orderings

# EVOLVE-BLOCK-END