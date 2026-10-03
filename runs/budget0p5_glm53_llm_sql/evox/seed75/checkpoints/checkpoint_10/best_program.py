# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict


class Evolved(Algorithm):
    """
    Prefix-cache oriented reordering (lean v3.1).

    Serialize each cell once (missing -> ""), build three cheap candidate
    per-row column orderings:
      (a) global pair-repetition order  (score = sum len(v)*cnt*(cnt-1))
      (b) total-serialized-length order (score = sum len(v)*cnt)
      (c) conditional partition tree over integer-factorized codes,
          choosing per-group the column with best length-weighted pair
          repetition, bounded depth / group size
    Score each with the exact serial-Trie objective (sum of adjacent LCPs
    of the sorted serialized rows, via C-speed prefix-slice binary search)
    and return the best. Rows keep identity and original values; only the
    per-row column order changes. Per-column uniques/counts are computed
    once and shared across all candidates.
    """

    # ---------- serialization helpers ----------

    def _serialize(self, df: pd.DataFrame) -> np.ndarray:
        """Return an (n_rows, n_cols) object array of scoring strings.

        Missing values become '' exactly as the evaluator does; stored
        cell values are never modified.
        """
        obj = df.astype(object)
        obj = obj.where(pd.notna(obj), "")
        return obj.astype(str).values.astype(object)

    def _row_strings(self, ser: np.ndarray, orders: List[List[int]]) -> List[str]:
        """Join serialized cells per row following each row's column order."""
        return ["".join([ser[i, o] for o in orders[i]]) for i in range(len(orders))]

    def _lcp_sum(self, strs: List[str]) -> int:
        """Exact serial-Trie reuse: sum of LCPs of lexicographically adjacent strings.

        Identical strings are handled directly; otherwise the common prefix
        length is found by binary search on prefix slices so comparisons run in C.
        """
        if not strs:
            return 0
        s = sorted(strs)
        total = 0
        prev = s[0]
        for cur in s[1:]:
            if cur == prev:
                total += len(cur)
            else:
                lo, hi = 0, min(len(prev), len(cur))
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if prev[:mid] == cur[:mid]:
                        lo = mid
                    else:
                        hi = mid - 1
                total += lo
            prev = cur
        return total

    # ---------- candidate constructions ----------

    def _col_stats(self, ser: np.ndarray):
        """Single pass per column: unique values, counts, value lengths.

        Returns (uniq_list, counts_list, lens_list) reused by all candidates.
        """
        m = ser.shape[1]
        uniqs, counts_l, lens_l = [], [], []
        for j in range(m):
            uniq, counts = np.unique(ser[:, j], return_counts=True)
            lens = np.fromiter((len(v) for v in uniq), dtype=np.int64, count=len(uniq))
            uniqs.append(uniq)
            counts_l.append(counts)
            lens_l.append(lens)
        return uniqs, counts_l, lens_l

    def _global_order(self, cols: List[str], counts_l, lens_l, mode: str) -> List[int]:
        """Column order by repetition statistics; modes 'pairs' or 'total'."""
        m = len(cols)
        scores = []
        for j in range(m):
            counts, lens = counts_l[j], lens_l[j]
            if mode == "pairs":
                sc = float((lens * counts * (counts - 1)).sum())
            else:
                sc = float((lens * counts).sum())
            scores.append(sc)
        return sorted(range(m), key=lambda j: (-scores[j], cols[j]))

    def _partition_orders(
        self,
        ser: np.ndarray,
        cols: List[str],
        base_order: List[int],
        eligible: List[int],
        max_depth: int,
        early_stop: int,
        min_group: int,
    ) -> List[List[int]]:
        """Conditional prefix partition tree producing per-row column orders.

        Within each group pick the eligible column maximizing length-weighted
        pair repetition (len*cnt*(cnt-1)) using precomputed integer codes and
        bincount; partition rows by its value; recurse. Leaves append the
        remaining columns in the global base order.
        """
        n, m = ser.shape
        codes = {}
        lens = {}
        for j in range(m):
            uniq, inv = np.unique(ser[:, j], return_inverse=True)
            codes[j] = inv.astype(np.int64)
            lens[j] = np.fromiter((len(v) for v in uniq), dtype=np.int64,
                                  count=len(uniq))

        orders = [None] * n

        def assign(rows: np.ndarray, used: List[int], remaining: List[int]):
            rem_set = set(remaining)
            suffix = used + [c for c in base_order if c in rem_set]
            for r in rows:
                orders[r] = suffix

        def rec(rows: np.ndarray, used: List[int], remaining: List[int], depth: int):
            if len(rows) < min_group or depth >= max_depth or not remaining:
                assign(rows, used, remaining)
                return
            best_col, best_score = -1, -1.0
            rem_set = set(remaining)
            for j in eligible:
                if j not in rem_set:
                    continue
                c = codes[j][rows]
                cnt = np.bincount(c, minlength=len(lens[j]))
                mask = cnt > 0
                sc = float((lens[j][mask] * cnt[mask] * (cnt[mask] - 1)).sum())
                if sc > best_score:
                    best_col, best_score = j, sc
            if best_col < 0 or best_score <= early_stop:
                assign(rows, used, remaining)
                return
            c = codes[best_col][rows]
            srt = np.argsort(c, kind="stable")
            sorted_rows = rows[srt]
            sorted_c = c[srt]
            new_used = used + [best_col]
            new_remaining = [x for x in remaining if x != best_col]
            start = 0
            L = len(sorted_c)
            for k in range(1, L + 1):
                if k == L or sorted_c[k] != sorted_c[start]:
                    rec(sorted_rows[start:k], new_used, new_remaining, depth + 1)
                    start = k

        rec(np.arange(n), [], list(range(m)), 0)
        for i in range(n):
            if orders[i] is None:
                orders[i] = list(base_order)
        return orders

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
        cols = list(df.columns)
        n = len(df)

        if n == 0 or len(cols) == 0:
            return df.copy(), [[] for _ in range(n)]

        # Honor column merges using parent semantics when available.
        work = df
        if col_merge:
            if hasattr(self, "merging_columns"):
                try:
                    _, stats = self.calculate_col_stats(df, enable_index=True)
                    ordered = [c for c, _, _, _ in stats]
                    for group in col_merge:
                        merge_order = [c for c in ordered if c in group]
                        if merge_order:
                            work = self.merging_columns(work, merge_order, prepended=False)
                except Exception:
                    work = df
            cols = list(work.columns)

        ser = self._serialize(work)
        vals = work.astype(object).values
        m = len(cols)

        # Shared per-column statistics (one pass each).
        uniqs, counts_l, lens_l = self._col_stats(ser)

        # Candidate (a): global pair-repetition order
        base_order = self._global_order(cols, counts_l, lens_l, "pairs")
        # Candidate (b): total-length order
        total_order = self._global_order(cols, counts_l, lens_l, "total")

        # Candidate (c): conditional partition tree
        nunique = np.array([len(uniqs[j]) for j in range(m)])
        thresh = max(1.0, distinct_value_threshold * n)
        eligible = [j for j in range(m) if nunique[j] <= thresh]
        if not eligible:
            eligible = list(range(m))
        if len(eligible) > 12:
            eligible = sorted(eligible, key=lambda j: (-1, cols[j]))[:12]
        max_depth = col_stop if col_stop is not None else 8
        if row_stop is not None:
            max_depth = min(max_depth, int(row_stop))
        max_depth = max(1, min(max_depth, m - 1, 10))
        tree_orders = self._partition_orders(
            ser, cols, base_order, eligible, max_depth, early_stop, min_group=4
        )

        candidates = [
            [list(base_order) for _ in range(n)],
            [list(total_order) for _ in range(n)],
            [list(o) for o in tree_orders],
        ]

        best_orders, best_score = candidates[0], -1
        for cand in candidates:
            sc = self._lcp_sum(self._row_strings(ser, cand))
            if sc > best_score:
                best_score, best_orders = sc, cand

        # Build output: original values, per-row permutation, object dtype.
        out = np.empty((n, m), dtype=object)
        for i in range(n):
            out[i, :] = vals[i, best_orders[i]]

        result = pd.DataFrame(out, columns=cols, index=work.index).infer_objects()
        column_orderings = [[cols[j] for j in best_orders[i]] for i in range(n)]
        return result, column_orderings


# EVOLVE-BLOCK-END