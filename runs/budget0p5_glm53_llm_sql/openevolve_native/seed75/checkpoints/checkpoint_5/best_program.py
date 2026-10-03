# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict, Optional


class Evolved(Algorithm):
    """
    Trie-prefix-cache oriented reordering.

    Build <=3 cheap candidate per-row column orderings, score each with the
    exact serial character-Trie reuse objective (sum of adjacent LCPs over the
    sorted serialized rows), and return the best.

    Candidates:
      A) Global frequency ordering: columns ranked by
         sum(len(v) * count * (count-1)) over distinct serialized values,
         with over-distinct columns pushed last.
      B) Conditional partition tree: recursively pick, inside each row group,
         the remaining column with the largest length-weighted pair repetition,
         partition rows by that column's value, and order the remaining suffix
         LOCALLY per group (by group-local repetition scores).
      C) Original column order (cheap deterministic baseline).

    Only column order changes; every cell value, row and column is preserved.
    """

    # ---------------- serialization ----------------

    def _serialize(self, df: pd.DataFrame) -> List[List[str]]:
        """Per-column serialization matching the evaluator.

        For each column compute a missing mask with pd.isna on that Series
        (safe for int64/float/object/datetime/nullable dtypes), then emit ''
        for missing cells and str(v) otherwise. Returns an n x m nested list
        of scoring strings.
        """
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
        """Exact ideal Trie reuse = sum of adjacent LCPs after sorting."""
        if len(strings) < 2:
            return 0
        arr = sorted(strings)
        total = 0
        prev = arr[0]
        lcp = cls._lcp
        for i in range(1, len(arr)):
            cur = arr[i]
            if prev == cur:
                total += len(prev)
            else:
                total += lcp(prev, cur)
            prev = cur
        return total

    # ---------------- column scoring ----------------

    @staticmethod
    def _col_scores(S: List[List[str]], rows: List[int], cols: List[int]) -> Dict[int, int]:
        """For each column, sum(len(v)*c*(c-1)) over the given rows."""
        scores = {}
        for j in cols:
            cnt: Dict[str, int] = {}
            for i in rows:
                v = S[i][j]
                cnt[v] = cnt.get(v, 0) + 1
            s = 0
            for v, c in cnt.items():
                if c > 1:
                    s += len(v) * c * (c - 1)
            scores[j] = s
        return scores

    def _global_order(self, S: List[List[str]], n: int, m: int,
                      distinct_value_threshold: float) -> List[int]:
        """Column order by length-weighted repetition, distinct columns last."""
        col_score = self._col_scores(S, range(n), range(m))
        distinct_last = []
        for j in range(m):
            nuniq = len({S[i][j] for i in range(n)})
            distinct_last.append(0 if (nuniq / n) <= distinct_value_threshold else 1)
        return sorted(range(m), key=lambda j: (distinct_last[j], -col_score[j], j))

    # ---------------- conditional partition tree ----------------

    def _conditional_orders(self, S: List[List[str]], n: int, m: int,
                            distinct_value_threshold: float) -> List[List[int]]:
        """Recursive conditional prefix partition tree -> per-row column orders.

        At each node, score remaining columns on this row subset, pick the best
        split column, partition rows by its value, recurse. Leaves order the
        remaining suffix by LOCAL repetition scores (not the global order),
        which captures residual sharing inside each group. Depth, minimum
        group size and candidate columns are bounded to keep runtime low.
        """
        global_tail = self._global_order(S, n, m, distinct_value_threshold)
        max_depth = 8
        min_group = 8
        max_cands = 24
        orders: List[Optional[List[int]]] = [None] * n

        def local_tail(rows: List[int], remaining: List[int]) -> List[int]:
            """Order remaining columns by repetition within this row group."""
            if len(rows) >= 2 * len(remaining) and len(rows) >= 16:
                sc = self._col_scores(S, rows, remaining)
                return sorted(remaining, key=lambda j: (-sc[j], j))
            rem_set = set(remaining)
            return [j for j in global_tail if j in rem_set]

        def recurse(rows: List[int], remaining: List[int], prefix: List[int], depth: int):
            if len(rows) < min_group or depth >= max_depth or not remaining:
                full = prefix + local_tail(rows, remaining)
                for i in rows:
                    orders[i] = full
                return
            scores = self._col_scores(S, rows, remaining)
            ranked = sorted(remaining, key=lambda j: (-scores[j], j))[:max_cands]
            best_j, best_s = ranked[0], scores[ranked[0]]
            if best_s <= 0:
                full = prefix + local_tail(rows, remaining)
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

        # Candidate A: global frequency ordering (same order for all rows).
        order_a = self._global_order(S, n, m, distinct_value_threshold)
        # Candidate B: conditional partition tree (per-row orders, local tails).
        orders_b = self._conditional_orders(S, n, m, distinct_value_threshold)
        # Candidate C: original column order.
        order_c = list(range(m))

        cand_orders = [order_a, orders_b, order_c]

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
        # Extract raw values once (fast C-level access) instead of per-cell iat.
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