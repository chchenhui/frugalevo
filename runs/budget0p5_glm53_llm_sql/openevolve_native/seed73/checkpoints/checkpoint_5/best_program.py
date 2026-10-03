# EVOLVE-BLOCK-START
import numpy as np
import pandas as pd
from collections import Counter
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

    Serialize cells once (missing -> ""), compute per-column pair-weighted
    repetition scores once, build a bounded set of candidate orderings
    (original; global pair-weighted descending; its reverse; total-character
    weighted descending; a conditional partition tree with per-row orders),
    score each with the exact serial-Trie objective (sum of adjacent LCPs of
    sorted serialized rows), and return the best. Output keeps every original
    cell value (object dtype).
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

    def _col_pair_scores(self, S: np.ndarray) -> List[int]:
        """Per-column sum of len(v)*c*(c-1) over repeated serialized values."""
        scores = []
        for j in range(S.shape[1]):
            sc = 0
            for v, c in Counter(S[:, j].tolist()).items():
                if c > 1:
                    sc += len(v) * c * (c - 1)
            scores.append(sc)
        return scores

    def _col_char_scores(self, S: np.ndarray) -> List[int]:
        """Per-column total serialized characters len(v)*count (distinct tradeoff)."""
        scores = []
        for j in range(S.shape[1]):
            sc = 0
            for v, c in Counter(S[:, j].tolist()).items():
                sc += len(v) * c
            scores.append(sc)
        return scores

    def _score(self, row_strings: List[str]) -> int:
        """Exact ideal Trie reuse: sum of adjacent LCPs of sorted row strings."""
        s = sorted(row_strings)
        tot = 0
        for i in range(1, len(s)):
            tot += _lcp(s[i - 1], s[i])
        return tot

    def _tree_orders(self, S: np.ndarray, early_stop: int, max_depth: int,
                     eligible: List[int], global_pos: List[int]) -> List[List[int]]:
        """Conditional prefix partition tree producing per-row column orders.

        At each node, pick the eligible remaining column with the highest
        length-weighted pair repetition, partition rows by its value, and
        recurse. Leaves order remaining columns by the precomputed global
        pair-score ranking restricted to the remaining set (deterministic,
        avoids recomputing counts at every leaf).
        """
        n, p = S.shape
        orders: List[List[int]] = [list(range(p)) for _ in range(n)]
        eligible_set = set(eligible) if eligible else set(range(p))

        def tail_order(remaining: List[int]) -> List[int]:
            rem = set(remaining)
            return [j for j in global_pos if j in rem]

        def rec(rows: List[int], remaining: List[int], depth: int):
            if len(rows) <= 1 or not remaining or depth >= max_depth:
                tail = tail_order(remaining)
                for r in rows:
                    orders[r] = list(tail)
                return
            sub = S[np.ix_(rows, remaining)]
            best_k, best_sc = -1, 0
            for k in range(len(remaining)):
                if remaining[k] not in eligible_set:
                    continue
                sc = 0
                for v, c in Counter(sub[:, k].tolist()).items():
                    if c > 1:
                        sc += len(v) * c * (c - 1)
                if sc > best_sc:
                    best_sc, best_k = sc, k
            if best_k < 0 or best_sc <= early_stop:
                tail = tail_order(remaining)
                for r in rows:
                    orders[r] = list(tail)
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
        return orders

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

        pair_scores = self._col_pair_scores(S)
        char_scores = self._col_char_scores(S)

        # candidate 1: original order
        identity = [list(range(p)) for _ in range(n)]

        # candidate 2/3: global pair-weighted order and its reverse
        gpos = sorted(range(p), key=lambda j: (-pair_scores[j], j))
        gorder = [list(gpos) for _ in range(n)]
        rorder = [list(reversed(gpos)) for _ in range(n)]

        # candidate 4: total-character weighted order (distinct tradeoff)
        cpos = sorted(range(p), key=lambda j: (-char_scores[j], j))
        corder = [list(cpos) for _ in range(n)]

        # candidate 5: conditional partition tree with per-row orders
        max_depth = min(p, 12)
        if col_stop is not None:
            max_depth = min(max_depth, max(1, int(col_stop)))
        thresh = max(1, int(distinct_value_threshold * n))
        eligible = [j for j in range(p) if len(set(S[:, j].tolist())) <= thresh]
        if not eligible:
            eligible = list(range(p))
        tree = self._tree_orders(S, early_stop, max_depth, eligible, gpos)

        best_orders, best_score = identity, -1
        for cand in (identity, gorder, rorder, corder, tree):
            rows_str = ["".join(S[r, cand[r]].tolist()) for r in range(n)]
            sc = self._score(rows_str)
            if sc > best_score:
                best_score, best_orders = sc, cand

        vals = df.to_numpy(dtype=object)
        out = np.empty((n, p), dtype=object)
        for r in range(n):
            out[r] = vals[r, best_orders[r]]
        result = pd.DataFrame(out, columns=col_names, index=df.index)
        column_orderings = [[col_names[j] for j in best_orders[r]] for r in range(n)]
        return result, column_orderings

# EVOLVE-BLOCK-END