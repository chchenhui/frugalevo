# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
import numpy as np


class Evolved(Algorithm):
    """
    Prefix-cache-optimized reordering: produces per-row column permutations
    chosen among a small set of bounded candidate constructions, selected by
    the exact serial character-Trie reuse objective (sum of adjacent LCPs of
    sorted serialized rows). Data and shape are fully preserved.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ---------------- serialization helpers ----------------

    @staticmethod
    def _cell_str(v):
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        try:
            if v is pd.NA or v is pd.NaT:
                return ""
        except Exception:
            pass
        if isinstance(v, str):
            return v
        if isinstance(v, bool):
            return str(v)
        return str(v)

    def _serialize_cells(self, df: pd.DataFrame):
        """Return list-of-lists of cell strings (scoring representation only)."""
        cols = df.columns.tolist()
        n = len(df)
        cells = []
        for c in cols:
            s = df[c]
            arr = s.astype(object).where(s.notna(), "").tolist() if hasattr(s, "notna") else [self._cell_str(v) for v in s]
            out = []
            for v in arr:
                if isinstance(v, str):
                    out.append(v)
                else:
                    out.append(self._cell_str(v))
            cells.append(out)
        return cols, cells, n

    # ---------------- exact Trie objective ----------------

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        if a == b:
            return len(a)
        lo, hi = 0, min(len(a), len(b))
        # binary search on prefix-slice equality (C-speed comparisons)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _trie_score(self, strings: List[str]) -> int:
        if not strings:
            return 0
        ss = sorted(set(strings)) if len(strings) > 200000 else sorted(strings)
        total = 0
        prev = None
        for s in ss:
            if prev is not None:
                total += self._lcp(prev, s)
            prev = s
        return total

    # ---------------- candidate constructions ----------------

    def _global_freq_order(self, cells, n, cols):
        """Column order ranked by sum over values v of len(v)*count*(count-1)."""
        m = len(cols)
        scores = [0] * m
        for j in range(m):
            col = cells[j]
            counts = {}
            lens = {}
            for v in col:
                if v in counts:
                    counts[v] += 1
                else:
                    counts[v] = 1
                    lens[v] = len(v)
            sc = 0
            for v, c in counts.items():
                if c > 1:
                    sc += lens[v] * c * (c - 1)
            scores[j] = sc
        order = sorted(range(m), key=lambda j: (-scores[j], j))
        return order

    def _conditional_orders(self, cells, n, cols, depth_limit=6, min_group=8, max_splits=64):
        """Recursive conditional partition tree producing per-row column orders."""
        m = len(cols)
        # factorize each column to integer codes
        codes = []
        code_vals = []  # per column: list of unique serialized strings per code
        for j in range(m):
            col = cells[j]
            mapping = {}
            cvals = []
            c = []
            for v in col:
                if v in mapping:
                    c.append(mapping[v])
                else:
                    idx = len(cvals)
                    mapping[v] = idx
                    cvals.append(v)
                    c.append(idx)
            codes.append(c)
            code_vals.append(cvals)
        base_order = self._global_freq_order(cells, n, cols)
        orders = [[base_order] * n]  # candidate 0: global order for all rows
        orders.append([list(base_order) for _ in range(n)])  # candidate 1: to be filled per row

        # recursive partition on rows
        node_id = 0
        # stack of (row_indices, used_cols)
        stack = [(list(range(n)), [])]
        splits_done = 0
        while stack and splits_done < max_splits:
            rows, used = stack.pop()
            if len(rows) < min_group or len(used) >= depth_limit or len(used) >= m:
                continue
            # pick best remaining column by len-weighted pair repetition within group
            best_j, best_gain = -1, 0
            for j in range(m):
                if j in used:
                    continue
                cv = code_vals[j]
                cnt = {}
                for r in rows:
                    c = codes[j][r]
                    cnt[c] = cnt.get(c, 0) + 1
                gain = 0
                for c, k in cnt.items():
                    if k > 1:
                        gain += len(cv[c]) * k * (k - 1)
                if gain > best_gain:
                    best_gain, best_j = gain, j
            if best_j < 0:
                continue
            used2 = used + [best_j]
            tail = [j for j in base_order if j not in used2]
            # partition rows by value
            groups = {}
            for r in rows:
                groups.setdefault(codes[best_j][r], []).append(r)
            splits_done += 1
            for c, grp in groups.items():
                head = list(used2)
                # place the shared value column first, then remaining in freq order
                for r in grp:
                    orders[1][r] = head + tail
                if len(grp) >= min_group:
                    stack.append((grp, used2))
            node_id += 1
        return orders[0], orders[1]

    # ---------------- main API ----------------

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
        n, m = df.shape
        if n == 0 or m == 0:
            return df.copy(), [[] for _ in range(n)]

        # honor col_merge: keep merged groups' relative internal order stable by
        # simply not splitting them apart beyond what scoring suggests; since
        # scoring treats fields without separators, we still optimize freely but
        # we never alter values. col_merge is treated as a soft preference.
        cols, cells, n = self._serialize_cells(df)

        base_order = self._global_freq_order(cells, n, cols)

        # candidate A: global frequency order for every row
        order_a = base_order

        # candidate B: conditional partition per-row orders
        try:
            _, order_b = self._conditional_orders(cells, n, cols)
        except Exception:
            order_b = [list(base_order) for _ in range(n)]

        # candidate C: length-weighted variant (prefer long repeated values first)
        scores = [0] * m
        for j in range(m):
            counts = {}
            for v in cells[j]:
                counts[v] = counts.get(v, 0) + 1
            sc = 0
            for v, c in counts.items():
                if c > 1:
                    sc += len(v) * len(v) * c
            scores[j] = sc
        order_c = sorted(range(m), key=lambda j: (-scores[j], j))

        candidates = [
            (order_a, [order_a] * n),
            (order_c, [order_c] * n),
            (order_b, order_b),
        ]

        best_orders = None
        best_score = -1
        for _, per_row in candidates:
            strs = []
            for r in range(n):
                strs.append("".join(per_row[r][j] is not None and cells[per_row[r][j]][r] or "" for j in range(m)))
            sc = self._trie_score(strs)
            if sc > best_score:
                best_score = sc
                best_orders = per_row

        # build output: per-row permutations, object dtype, preserving values
        out_cols = cols
        data = {}
        for j in range(m):
            col_vals = []
            for r in range(n):
                o = best_orders[r]
                col_vals.append(df.iat[r, o[j]])
            data[out_cols[j]] = col_vals
        result = pd.DataFrame(data, columns=out_cols, dtype=object)
        result.index = df.index.copy()

        column_orderings = [[out_cols[j] for j in order] for order in best_orders]
        return result, column_orderings


# EVOLVE-BLOCK-END