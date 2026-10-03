# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict, Optional


class Evolved(Algorithm):
    """
    Trie-prefix-cache oriented reordering.

    Build <=4 cheap candidate per-row column orderings, score each with the
    exact serial character-Trie reuse objective (sum of adjacent LCPs over
    sorted serialized rows, duplicates counted at full length), and return
    the best.

    Candidates:
      A) Global frequency ordering: columns ranked by
         sum(len(v)*count*(count-1)), over-distinct columns pushed last.
      B) Conditional partition tree, split column chosen by pair repetition
         len(v)*c*(c-1); suffixes ordered LOCALLY per group.
      D) Conditional partition tree, split column chosen by immediate
         savings len(v)*(c-1); suffixes ordered by the GLOBAL order.
      C) Original column order (deterministic baseline).

    Only column order changes; every cell value, row and column preserved.
    """

    # ---------------- serialization ----------------

    def _serialize(self, df: pd.DataFrame) -> List[List[str]]:
        """Per-column scoring strings: '' for missing, str(v) otherwise."""
        n = len(df)
        grid: List[List[str]] = [[] for _ in range(n)]
        for col in df.columns:
            series = df[col]
            try:
                mask = pd.isna(series).values
            except Exception:
                mask = np.array([v is None or (isinstance(v, float) and np.isnan(v))
                                 for v in series.values])
            vals = series.values
            if mask.any():
                for i in range(n):
                    grid[i].append("" if mask[i] else str(vals[i]))
            else:
                for i in range(n):
                    grid[i].append(str(vals[i]))
        return grid

    # ---------------- scoring ----------------

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Longest common prefix via binary search on slice equality (C speed)."""
        if a == b:
            return len(a)
        hi = min(len(a), len(b))
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    @classmethod
    def _score(cls, strings: List[str]) -> int:
        """Exact ideal Trie reuse: sum of adjacent LCPs over sorted strings.

        Identical strings are collapsed with a multiplicity weight so the
        sort and LCP work only touches distinct strings.
        """
        if len(strings) < 2:
            return 0
        cnt: Dict[str, int] = {}
        for s in strings:
            cnt[s] = cnt.get(s, 0) + 1
        arr = sorted(cnt)
        total = 0
        lcp = cls._lcp
        prev = arr[0]
        c0 = cnt[prev]
        if c0 > 1:
            total += len(prev) * (c0 - 1)
        for i in range(1, len(arr)):
            cur = arr[i]
            total += lcp(prev, cur)
            c = cnt[cur]
            if c > 1:
                total += len(cur) * (c - 1)
            prev = cur
        return total

    # ---------------- column scoring ----------------

    @staticmethod
    def _col_scores(S: List[List[str]], rows: List[int], cols: List[int],
                    mode: int) -> Dict[int, int]:
        """Per-column repetition score over the given rows.

        mode 0: len(v)*c*(c-1)  (pair repetition)
        mode 1: len(v)*(c-1)    (immediate savings)
        """
        scores = {}
        for j in cols:
            cnt: Dict[str, int] = {}
            for i in rows:
                v = S[i][j]
                cnt[v] = cnt.get(v, 0) + 1
            s = 0
            for v, c in cnt.items():
                if c > 1:
                    if mode == 0:
                        s += len(v) * c * (c - 1)
                    else:
                        s += len(v) * (c - 1)
            scores[j] = s
        return scores

    def _global_order(self, S: List[List[str]], n: int, m: int,
                      distinct_value_threshold: float) -> List[int]:
        """Column order by length-weighted pair repetition, distinct cols last."""
        col_score = self._col_scores(S, range(n), range(m), 0)
        distinct_last = []
        for j in range(m):
            nuniq = len({S[i][j] for i in range(n)})
            distinct_last.append(0 if (nuniq / n) <= distinct_value_threshold else 1)
        return sorted(range(m), key=lambda j: (distinct_last[j], -col_score[j], j))

    # ---------------- conditional partition tree ----------------

    def _tree_orders(self, S: List[List[str]], n: int, m: int,
                     distinct_value_threshold: float, mode: int,
                     local_tail: bool) -> List[List[int]]:
        """Recursive conditional prefix partition tree -> per-row orders.

        mode selects the split-column criterion; local_tail selects whether
        leaves order remaining columns by group-local repetition (True) or
        by the global order (False). Depth, group size and candidate columns
        are bounded.
        """
        global_tail = self._global_order(S, n, m, distinct_value_threshold)
        max_depth = 8
        min_group = 8
        max_cands = 24
        orders: List[Optional[List[int]]] = [None] * n

        def tail(rows: List[int], remaining: List[int]) -> List[int]:
            if local_tail and len(rows) >= 2 * len(remaining) and len(rows) >= 16:
                sc = self._col_scores(S, rows, remaining, 0)
                return sorted(remaining, key=lambda j: (-sc[j], j))
            rem_set = set(remaining)
            return [j for j in global_tail if j in rem_set]

        def recurse(rows: List[int], remaining: List[int], prefix: List[int], depth: int):
            if len(rows) < min_group or depth >= max_depth or not remaining:
                full = prefix + tail(rows, remaining)
                for i in rows:
                    orders[i] = full
                return
            scores = self._col_scores(S, rows, remaining, mode)
            ranked = sorted(remaining, key=lambda j: (-scores[j], j))[:max_cands]
            best_j, best_s = ranked[0], scores[ranked[0]]
            if best_s <= 0:
                full = prefix + tail(rows, remaining)
                for i in rows:
                    orders[i] = full
                return
            groups: Dict[str, List[int]] = {}
            for i in rows:
                groups.setdefault(S[i][best_j], []).append(i)
            rem = [j for j in remaining if j != best_j]
            new_prefix = prefix + [best_j]
            for v in sorted(groups):
                recurse(groups[v], rem, new_prefix, depth + 1)

        recurse(list(range(n)), list(range(m)), [], 0)
        return [o if o is not None else list(range(m)) for o in orders]

    # ---------------- public API ----------------

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
        cols = list(df.columns)

        if n == 0 or m == 0:
            return df.copy(), [list(cols) for _ in range(n)]

        # Honor col_merge semantics via base class if available; otherwise
        # place merged column groups adjacently (values unchanged).
        work = df
        if col_merge:
            try:
                merged_df = df.copy()
                stats_cols = list(df.columns)
                for group in col_merge:
                    ordered = [c for c in stats_cols if c in group]
                    if hasattr(self, "merging_columns") and len(ordered) > 1:
                        merged_df = self.merging_columns(merged_df, ordered, prepended=False)
                work = merged_df
                cols = list(work.columns)
                m = len(cols)
            except Exception:
                flat = [c for g in col_merge for c in g if c in cols]
                rest = [c for c in cols if c not in flat]
                cols = flat + rest
                work = df[cols]
                m = len(cols)

        # Serialize cells once; reuse across all candidates.
        S = self._serialize(work)

        order_a = self._global_order(S, n, m, distinct_value_threshold)
        orders_b = self._tree_orders(S, n, m, distinct_value_threshold, 0, True)
        orders_d = self._tree_orders(S, n, m, distinct_value_threshold, 1, False)
        order_c = list(range(m))

        cand_orders = [order_a, orders_b, orders_d, order_c]

        def row_strings(orders) -> List[str]:
            if isinstance(orders[0], int):
                return ["".join([S[i][j] for j in orders]) for i in range(n)]
            return ["".join([S[i][j] for j in orders[i]]) for i in range(n)]

        best_orders, best_score = order_c, -1
        for od in cand_orders:
            try:
                sc = self._score(row_strings(od))
            except Exception:
                sc = -1
            if sc > best_score:
                best_score, best_orders = sc, od

        # Build output: same columns, per-row permuted values, object dtype.
        raw = work.values
        out_arr = np.empty((n, m), dtype=object)
        col_orderings: List[List[str]] = []
        is_uniform = isinstance(best_orders[0], int)
        for i in range(n):
            od = best_orders if is_uniform else best_orders[i]
            row_vals = raw[i]
            for k, j in enumerate(od):
                out_arr[i, k] = row_vals[j]
            col_orderings.append([cols[j] for j in od])

        out_df = pd.DataFrame(out_arr, columns=cols,
                              index=df.index[:n] if len(df.index) == n else None)
        return out_df, col_orderings
# EVOLVE-BLOCK-END