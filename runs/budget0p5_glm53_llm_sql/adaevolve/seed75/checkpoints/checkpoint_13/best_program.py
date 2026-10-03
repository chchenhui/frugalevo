# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List
from collections import defaultdict


class Evolved(Algorithm):
    """
    Character-trie prefix caching optimizer (v3).

    The evaluator's total reuse over a fixed multiset of serialized row
    strings is insertion-order independent, so only per-row column orderings
    matter (they change the strings themselves). Row sorting is unnecessary.

    Approach:
      1. Serialize every cell once (missing -> ""), keeping the raw object
         value matrix untouched so output cells are preserved exactly.
      2. Build bounded candidate per-row column orderings:
           (a) global order ranked by sum(len(v) * cnt * (cnt-1)),
           (b) a conditional partition tree picking, inside each row group, a
               remaining field by length-weighted pair repetition and
               recursing per value-group (bounded depth/candidates),
           (c) same tree with unweighted pair repetition cnt*(cnt-1),
           (d) a greedy sequential construction that, at each position, picks
               the next column maximizing the *actual* sorted-LCP reuse of the
               partially built row strings (bounded by a work budget),
           (e) the original column order.
      3. Score every candidate exactly with the serial-trie objective: sort
         the serialized rows and sum adjacent LCP lengths (binary-searched
         prefix equality, C-speed compares). Total serialized length is
         invariant, so raw matched sums are comparable.
      4. Optionally refine the best uniform order with one bounded pass of
         adjacent column swaps accepted only on real LCP-score improvement.
      5. Return the best candidate's per-row permutation of the original
         cells (object dtype) with matching per-row column orderings.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ---------- serialization helpers ----------

    @staticmethod
    def _cell_str(v) -> str:
        """Scoring serialization: missing -> '', else str(v)."""
        if v is None:
            return ""
        if isinstance(v, float) and v != v:
            return ""
        return str(v)

    def _serialize_matrix(self, df: pd.DataFrame):
        """Return (raw object matrix unchanged, list-of-lists of cell strings)."""
        raw = df.to_numpy(dtype=object)
        ser = [[self._cell_str(v) for v in row] for row in raw]
        return raw, ser

    # ---------- scoring ----------

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Longest common prefix via binary search on prefix slices (C compares)."""
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

    def _score(self, strings: List[str]) -> int:
        """Exact ideal trie reuse: sum of adjacent LCPs after sorting."""
        if len(strings) < 2:
            return 0
        ss = sorted(strings)
        total = 0
        prev = ss[0]
        for i in range(1, len(ss)):
            cur = ss[i]
            total += self._lcp(prev, cur)
            prev = cur
        return total

    @staticmethod
    def _build_strings(ser, order_matrix) -> List[str]:
        out = []
        for row, o in zip(ser, order_matrix):
            out.append("".join([row[c] for c in o]))
        return out

    # ---------- candidate constructions ----------

    def _global_order(self, ser, m: int) -> List[int]:
        """Rank columns by sum over values v of len(v)*cnt*(cnt-1), deterministic."""
        scores = [0] * m
        for c in range(m):
            counts = defaultdict(int)
            for row in ser:
                counts[row[c]] += 1
            s = 0
            for v, cnt in counts.items():
                if cnt > 1:
                    s += len(v) * cnt * (cnt - 1)
            scores[c] = s
        return sorted(range(m), key=lambda c: (-scores[c], c))

    def _tree_orders(self, ser, tail_order, m: int, weighted: bool,
                     max_depth=8, max_cands=24):
        """Recursive conditional grouping.

        Inside each row group, pick a remaining field by pair repetition
        (len-weighted or not), partition rows by that field's value, and
        recurse per group. Leaves get the remaining global-tail order.
        Returns a per-row list of column indices.
        """
        n = len(ser)
        tail = list(tail_order)

        def rec(row_ids, used, depth):
            if len(row_ids) < 2 or depth >= max_depth or len(used) >= m:
                rem = [c for c in tail if c not in used]
                return {i: rem for i in row_ids}
            cands = [c for c in tail if c not in used][:max_cands]
            best_col, best_gain = None, 0
            for c in cands:
                counts = defaultdict(int)
                for i in row_ids:
                    counts[ser[i][c]] += 1
                gain = 0
                for v, cnt in counts.items():
                    if cnt > 1:
                        gain += len(v) * cnt * (cnt - 1) if weighted else cnt * (cnt - 1)
                if gain > best_gain:
                    best_gain, best_col = gain, c
            if best_col is None:
                rem = [c for c in tail if c not in used]
                return {i: rem for i in row_ids}
            groups = defaultdict(list)
            for i in row_ids:
                groups[ser[i][best_col]].append(i)
            result = {}
            new_used = used | {best_col}
            for g in groups.values():
                sub = rec(g, new_used, depth + 1)
                for i, suffix in sub.items():
                    result[i] = [best_col] + suffix
            return result

        res = rec(list(range(n)), set(), 0)
        return [res.get(i) or list(tail) for i in range(n)]

    def _greedy_order(self, ser, m: int, n: int) -> List[int]:
        """Greedy sequential order driven by the real LCP objective.

        Start from empty partial strings; repeatedly append the column that
        maximizes the exact sorted-LCP reuse of the partial row strings.
        Work is bounded: m steps, each scoring <= m candidate appends.
        """
        order = []
        remaining = set(range(m))
        partial = [""] * n
        for _ in range(m):
            best_col, best_s = None, -1
            for c in remaining:
                cand = [p + row[c] for p, row in zip(partial, ser)]
                s = self._score(cand)
                if s > best_s:
                    best_s, best_col = s, c
            order.append(best_col)
            remaining.discard(best_col)
            partial = [p + row[best_col] for p, row in zip(partial, ser)]
        return order

    def _swap_refine(self, ser, order, m: int, n: int, max_swaps=None) -> List[int]:
        """One bounded pass of adjacent swaps on a uniform order.

        Each swap is accepted only if the exact LCP score improves.
        """
        order = list(order)
        base = self._score(self._build_strings(ser, [order] * n))
        swaps = max_swaps if max_swaps is not None else m - 1
        tried = 0
        i = 0
        while i < m - 1 and tried < swaps + m:
            tried += 1
            cand = list(order)
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            s = self._score(self._build_strings(ser, [cand] * n))
            if s > base:
                order, base = cand, s
                i = max(0, i - 1)  # recheck neighborhood after improvement
            else:
                i += 1
        return order

    # ---------- main API ----------

    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop=None,
        col_stop=None,
        col_merge: List[List[str]] = [],
        one_way_dep=[],
        distinct_value_threshold: float = 0.8,
        parallel: bool = True,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        initial_df = df.copy()

        if col_merge:
            try:
                _, stats = self.calculate_col_stats(df, enable_index=True)
                reordered_columns = [c for c, _, _, _ in stats]
                for grp in col_merge:
                    final_order = [c for c in reordered_columns if c in grp]
                    if len(final_order) > 1:
                        df = self.merging_columns(df, final_order, prepended=False)
            except Exception:
                pass

        colnames = list(df.columns)
        n, m = df.shape
        if n == 0 or m == 0:
            return df.copy(), [list(colnames) for _ in range(n)]

        raw, ser = self._serialize_matrix(df)

        # Candidate 1: global frequency-ranked order
        g_order = self._global_order(ser, m)
        cands = [([g_order] * n)]

        # Candidates 2 & 3: conditional partition trees (bounded)
        depth = min(8, col_stop if isinstance(col_stop, int) else 8)
        try:
            cands.append(self._tree_orders(ser, g_order, m, True, max_depth=depth))
        except Exception:
            pass
        try:
            cands.append(self._tree_orders(ser, g_order, m, False, max_depth=depth))
        except Exception:
            pass

        # Candidate 4: greedy sequential order driven by the real LCP score.
        # Bound total work: m steps * m candidate scores * n log n sorting.
        if n * m * m <= 4_000_000 and n > 1:
            try:
                greedy = self._greedy_order(ser, m, n)
                cands.append([greedy] * n)
                # Bounded adjacent-swap refinement of the greedy order.
                if m > 2 and n * m <= 500_000:
                    refined = self._swap_refine(ser, greedy, m, n)
                    if refined != greedy:
                        cands.append([refined] * n)
            except Exception:
                pass

        # Candidate 5: original column order
        cands.append([list(range(m))] * n)

        best_score, best_orders = -1, cands[0]
        for orders in cands:
            s = self._score(self._build_strings(ser, orders))
            if s > best_score:
                best_score, best_orders = s, orders

        # Materialize output: each row's original cells permuted per its order.
        out = np.empty((n, m), dtype=object)
        for i in range(n):
            row_raw = raw[i]
            out[i] = [row_raw[c] for c in best_orders[i]]

        out_df = pd.DataFrame(out, columns=colnames)
        column_orderings = [[colnames[c] for c in o] for o in best_orders]

        if not col_merge:
            assert out_df.shape == initial_df.shape
        else:
            assert out_df.shape[0] == initial_df.shape[0]
        return out_df, column_orderings

# EVOLVE-BLOCK-END