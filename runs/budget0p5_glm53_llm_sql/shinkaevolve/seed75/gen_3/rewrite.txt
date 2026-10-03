# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict, Optional


class Evolved(Algorithm):
    """
    Character-Trie prefix-reuse optimized reordering.

    Objective: maximize total matched characters when serialized rows
    ("".join(row.fillna("").astype(str))) are inserted into a character Trie.
    For a fixed multiset of row strings the ideal reuse equals the sum of
    adjacent longest-common-prefix lengths of the sorted row strings.
    We construct a few cheap candidate per-row column orderings, score each
    with this exact deterministic objective, and return the best.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------ #
    # serialization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ser(v) -> str:
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(v, bool):
            return "True" if v else "False"
        return str(v)

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Longest common prefix length; binary search with C-speed slices."""
        if a == b:
            return len(a)
        hi = min(len(a), len(b))
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _score_orders(self, ser, orders) -> int:
        """Exact ideal Trie reuse for per-row positional orders."""
        strs = ["".join([row[p] for p in ord_]) for row, ord_ in zip(ser, orders)]
        strs.sort()
        total = 0
        prev = None
        for s in strs:
            if prev is not None:
                total += self._lcp(prev, s)
            prev = s
        return total

    # ------------------------------------------------------------------ #
    # candidate constructions
    # ------------------------------------------------------------------ #
    @staticmethod
    def _col_stats_ser(ser, n, m):
        """Per column: weight = sum len(v)*cnt*(cnt-1), freq = sum cnt*(cnt-1), total len."""
        weights = [0] * m
        freqs = [0] * m
        totlen = [0] * m
        nuniques = [0] * m
        for c in range(m):
            cnt = {}
            tl = 0
            for r in range(n):
                s = ser[r][c]
                cnt[s] = cnt.get(s, 0) + 1
                tl += len(s)
            w = 0
            f = 0
            for s, k in cnt.items():
                w += len(s) * k * (k - 1)
                f += k * (k - 1)
            weights[c] = w
            freqs[c] = f
            totlen[c] = tl
            nuniques[c] = len(cnt)
        return weights, freqs, totlen, nuniques

    def _cond_orders(self, ser, weights, rows, remaining, depth, max_depth,
                     early_stop, distinct_cols):
        """Conditional prefix partition tree -> per-row positional orders."""
        m = self._m
        if (len(rows) <= 1 or not remaining or depth >= max_depth):
            tail = sorted(remaining,
                          key=lambda c: (c in distinct_cols, -weights[c], c))
            return [list(tail) for _ in rows]

        # pick branching column by length-weighted pair repetition
        best_col, best_gain = -1, early_stop  # must beat early_stop
        for c in remaining:
            if c in distinct_cols:
                continue
            cnt = {}
            lens = {}
            for r in rows:
                s = ser[r][c]
                cnt[s] = cnt.get(s, 0) + 1
                lens[s] = len(s)
            gain = sum(lens[s] * (k - 1) for s, k in cnt.items())
            if gain > best_gain:
                best_gain = gain
                best_col = c
        if best_col < 0:
            tail = sorted(remaining,
                          key=lambda c: (c in distinct_cols, -weights[c], c))
            return [list(tail) for _ in rows]

        rest = [c for c in remaining if c != best_col]
        # partition rows by serialized value of best_col
        groups = {}
        for r in rows:
            groups.setdefault(ser[r][best_col], []).append(r)

        orders = [None] * len(rows)
        idx_of = {r: i for i, r in enumerate(rows)}
        for _, grp in groups.items():
            sub = self._cond_orders(ser, weights, grp, rest, depth + 1,
                                    max_depth, early_stop, distinct_cols)
            for i, r in enumerate(grp):
                orders[idx_of[r]] = [best_col] + sub[i]
        return orders

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: Optional[int] = None,
        col_stop: Optional[int] = None,
        col_merge: List[List[str]] = [],
        one_way_dep: List[Tuple[str, str]] = [],
        distinct_value_threshold: float = 0.8,
        parallel: bool = True,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        work = df.copy()

        # honor required column merges using existing API semantics
        if col_merge:
            for col_to_merge in col_merge:
                final_col_order = [c for c in work.columns if c in col_to_merge]
                if len(final_col_order) > 1:
                    work = self.merging_columns(work, final_col_order, prepended=False)

        cols = list(work.columns)
        n, m = work.shape
        if n == 0:
            return work.copy(), []
        if m == 0:
            return pd.DataFrame(index=range(n)), [[] for _ in range(n)]

        # raw values (preserve exactly) and serialized cells
        raw = [[work.iat[r, c] for c in range(m)] for r in range(n)]
        ser = [[self._ser(v) for v in row] for row in raw]

        weights, freqs, totlen, nuniques = self._col_stats_ser(ser, n, m)
        self._m = m

        # highly distinct columns -> pushed to the tail of every order
        thr = n * distinct_value_threshold
        distinct_cols = {c for c in range(m) if nuniques[c] > thr}

        # candidate 1: global order by length-weighted repetition
        order1 = sorted(range(m),
                        key=lambda c: (c in distinct_cols, -weights[c], cols[c]))
        cand1 = [list(order1) for _ in range(n)]

        # candidate 2: frequency-ranked order with length tiebreak
        order2 = sorted(range(m),
                        key=lambda c: (c in distinct_cols, -freqs[c],
                                       -(totlen[c] / max(n, 1)), cols[c]))
        cand2 = [list(order2) for _ in range(n)]

        # candidate 3: conditional partition tree (bounded depth)
        max_depth = col_stop if col_stop else 6
        if row_stop is not None:
            max_depth = min(max_depth, max(1, int(row_stop)))
        cand3 = self._cond_orders(ser, weights, list(range(n)), list(range(m)),
                                  0, max_depth, early_stop, distinct_cols)

        # enforce one-way dependencies: dep column must precede dependent column
        dep_pairs = []
        for dep in (one_way_dep or []):
            try:
                a, b = dep[0], dep[1]
            except Exception:
                continue
            ca = [c for c in cols if a in str(c)]
            cb = [c for c in cols if b in str(c)]
            if len(ca) == 1 and len(cb) == 1:
                dep_pairs.append((cols.index(ca[0]), cols.index(cb[0])))

        def fix_deps(orders):
            if not dep_pairs:
                return orders
            out = []
            for od in orders:
                od = list(od)
                for a, b in dep_pairs:
                    if a in od and b in od:
                        pa, pb = od.index(a), od.index(b)
                        if pa > pb:
                            od[pa], od[pb] = od[pb], od[pa]
                out.append(od)
            return out

        cand1 = fix_deps(cand1)
        cand2 = fix_deps(cand2)
        cand3 = fix_deps(cand3)

        # select by the exact ideal Trie character objective
        best_orders, best_score = cand1, self._score_orders(ser, cand1)
        for cand in (cand2, cand3):
            s = self._score_orders(ser, cand)
            if s > best_score:
                best_orders, best_score = cand, s

        # build output: fixed output columns = order1 labels; per-row
        # orderings map output positions to original column names.
        out_cols = [cols[i] for i in order1]
        # ensure unique output labels (defensive for duplicate column names)
        if len(set(out_cols)) != m:
            out_cols = [f"col_{j}" for j in range(m)]
            orderings = [[f"col_{p}" for p in od] for od in best_orders]
        else:
            orderings = [[cols[p] for p in od] for od in best_orders]

        data = [[raw[r][p] for p in od] for r, od in enumerate(best_orders)]
        final_df = pd.DataFrame(data, columns=out_cols, dtype=object)

        assert final_df.shape == (n, m)
        return final_df, orderings

# EVOLVE-BLOCK-END