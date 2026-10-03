# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List


class Evolved(Algorithm):
    """
    Prefix-cache oriented reordering (lean v6).

    Serialize each cell once (missing -> ""). Build a bounded set of candidate
    per-row column orderings:
      (a) global pair-repetition order  (sum len(v)*cnt*(cnt-1))
      (b) total-serialized-length order (sum len(v)*cnt)
      (c) conditional partition tree over integer-factorized codes
      (d) greedy incremental order: columns appended one at a time, each step
          choosing the column that maximizes the EXACT serial-Trie reuse
          (sorted-string adjacent-LCP sum) of the resulting full order
      (e) insertion hill-climb starting from the best uniform candidate,
          bounded passes, exact scoring
      (f) per-row variant of the best uniform order with empty ("") cells
          pushed to the tail
    Search phases (d)/(e) are scored on a deterministic row sample when the
    table is large, keeping cost bounded; the FINAL selection among all
    candidates always uses exact full-data scoring. Rows keep identity and
    original values; only the per-row column order changes.
    """

    # ---------- serialization helpers ----------

    def _serialize(self, df: pd.DataFrame) -> np.ndarray:
        """(n_rows, n_cols) object array of scoring strings; missing -> ''."""
        obj = df.astype(object)
        obj = obj.where(pd.notna(obj), "")
        return obj.astype(str).values.astype(object)

    def _lcp_sum(self, strs: List[str]) -> int:
        """Exact serial-Trie reuse: sum of LCPs of lexicographically adjacent
        strings. Identical strings handled directly; otherwise binary search
        on prefix slices so comparisons run in C."""
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

    def _score_uniform(self, ser: np.ndarray, order: List[int]) -> int:
        """Exact objective for a single uniform column order."""
        sub = ser[:, order]
        return self._lcp_sum(["".join(r) for r in sub])

    def _score_orders(self, ser: np.ndarray, orders: List[List[int]]) -> int:
        """Exact objective for per-row column orders."""
        return self._lcp_sum(
            ["".join([ser[i, o] for o in orders[i]]) for i in range(len(orders))]
        )

    # ---------- statistics / cheap constructions ----------

    def _col_stats(self, ser: np.ndarray):
        """One pass per column: unique values, counts, value lengths."""
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
        self, ser, cols, base_order, eligible, max_depth, early_stop, min_group
    ) -> List[List[int]]:
        """Conditional prefix partition tree: within each group pick the
        eligible column maximizing length-weighted pair repetition using
        precomputed integer codes + bincount; partition rows by its value and
        recurse. Leaves append remaining columns in the global base order."""
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

    # ---------- exact-score-guided search (bounded) ----------

    def _greedy_order(self, sub: np.ndarray, base_order: List[int]) -> List[int]:
        """Greedy incremental construction: append columns one at a time, each
        step selecting the column whose addition maximizes the exact LCP-sum
        of the resulting full order (unplaced columns fill the tail in base
        order). Cost O(m^2) scorings on the (possibly sampled) rows."""
        m = sub.shape[1]
        rem = list(range(m))
        order: List[int] = []
        rem_set = set(rem)
        while rem:
            best_j, best_sc = rem[0], -1
            for j in rem:
                rest = [c for c in base_order if c in rem_set and c != j]
                sc = self._score_uniform(sub, order + [j] + rest)
                if sc > best_sc:
                    best_sc, best_j = sc, j
            order.append(best_j)
            rem.remove(best_j)
            rem_set.discard(best_j)
        return order

    def _hill_climb(self, sub: np.ndarray, order: List[int], max_passes: int) -> List[int]:
        """Insertion hill-climb: repeatedly try moving each column to every
        other position, accepting strict improvements, for at most max_passes.
        Cost O(m^2) scorings per pass on the (possibly sampled) rows."""
        order = list(order)
        m = len(order)
        cur = self._score_uniform(sub, order)
        for _ in range(max_passes):
            improved = False
            for pos in range(m):
                col = order[pos]
                without = order[:pos] + order[pos + 1:]
                for newpos in range(m):
                    if newpos == pos:
                        continue
                    cand = without[:newpos] + [col] + without[newpos:]
                    sc = self._score_uniform(sub, cand)
                    if sc > cur:
                        cur = sc
                        order = cand
                        improved = True
                        pos = newpos
                        without = order[:pos] + order[pos + 1:]
            if not improved:
                break
        return order

    def _empty_tail_orders(self, ser: np.ndarray, base_order: List[int]) -> List[List[int]]:
        """Per-row variant of base_order: columns whose serialized value is
        '' (missing) are moved to the tail, preserving relative order."""
        n = ser.shape[0]
        orders = []
        for i in range(n):
            row = ser[i]
            head = [j for j in base_order if row[j] != ""]
            tail = [j for j in base_order if row[j] == ""]
            orders.append(head + tail)
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

        # Deterministic row sample for bounded search-phase scoring.
        SAMPLE_N = 4000
        if n > SAMPLE_N:
            rng = np.random.RandomState(0)
            sample_idx = rng.choice(n, SAMPLE_N, replace=False)
            sub = ser[sample_idx]
        else:
            sub = ser

        # Candidate (d): greedy incremental order (bounded by width).
        greedy_order = None
        if m <= 16:
            try:
                greedy_order = self._greedy_order(sub, base_order)
            except Exception:
                greedy_order = None

        # Candidate (e): insertion hill-climb from the better of (a)/(b).
        climb_order = None
        if m <= 12:
            try:
                start = base_order
                if self._score_uniform(sub, total_order) > self._score_uniform(sub, base_order):
                    start = total_order
                climb_order = self._hill_climb(sub, start, max_passes=2)
            except Exception:
                climb_order = None

        # Candidate (f): per-row empty-tail variant of the best uniform order.
        uniform_cands = [base_order, total_order]
        if greedy_order is not None:
            uniform_cands.append(greedy_order)
        if climb_order is not None:
            uniform_cands.append(climb_order)
        best_uniform = uniform_cands[0]
        best_usc = -1
        for uo in uniform_cands:
            sc = self._score_uniform(ser, uo)  # exact, full data
            if sc > best_usc:
                best_usc, best_uniform = sc, uo
        empty_tail = self._empty_tail_orders(ser, best_uniform)

        # Final exact full-data selection among all candidates.
        candidates = [[list(best_uniform) for _ in range(n)], tree_orders, empty_tail]
        best_orders, best_score = candidates[0], -1
        for cand in candidates:
            sc = self._score_orders(ser, cand)
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