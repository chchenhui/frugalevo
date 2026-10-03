# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List
from collections import Counter


class Evolved(Algorithm):
    """
    Trie-reuse oriented column reordering.

    Strategy: build a small number (<=3) of cheap candidate per-row column
    orderings, measure each with the exact serial character-Trie reuse
    objective (sum of adjacent LCPs of the sorted serialized rows), and return
    the best. All original cell values are preserved; only their per-row
    permutation changes.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _cell_str(v) -> str:
        """Serialization used for scoring: missing -> '' else str(v)."""
        if v is None:
            return ""
        if isinstance(v, float) and v != v:
            return ""
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        return str(v)

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        n = min(len(a), len(b))
        i = 0
        step = 64
        while i < n and a[i:i + step] == b[i:i + step]:
            i += step
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _trie_reuse(self, strings) -> int:
        """Ideal trie reuse = sum of adjacent LCPs after sorting."""
        s = sorted(strings)
        total = 0
        lcp = self._lcp
        prev = None
        for cur in s:
            if prev is not None:
                total += lcp(prev, cur)
            prev = cur
        return total

    # ------------------------------------------------------------- candidates

    def _global_orders(self, cols, gscore, gscore2, col_pos):
        order_a = sorted(cols, key=lambda c: (-gscore[c], col_pos[c]))
        order_b = sorted(cols, key=lambda c: (-gscore2[c], col_pos[c]))
        return order_a, order_b

    def _tree_orders(self, cols, col_pos, code_arr, lens_arr, gscore,
                     rows, max_depth=10, cand_cap=12, group_cap=2):
        """Recursive conditional partition producing row-specific orders."""
        n_rows_total = len(code_arr[cols[0]])

        def tail_order(avail):
            return sorted(avail, key=lambda c: (-gscore[c], col_pos[c]))

        def rec(rows, avail, depth):
            if len(rows) < 2 or len(avail) <= 1 or depth >= max_depth:
                t = tail_order(avail)
                return {int(r): t for r in rows}
            if len(avail) > cand_cap:
                avail = sorted(avail, key=lambda c: (-gscore[c], col_pos[c]))[:cand_cap]
            best_col = None
            best_save = 0.0
            best_counts = None
            for c in avail:
                cc = code_arr[c][rows]
                counts = np.bincount(cc)
                nz = counts > 1
                if not nz.any():
                    continue
                wsum = np.bincount(cc, weights=lens_arr[c][rows])
                save = float(((counts[nz] - 1) * (wsum[nz] / counts[nz])).sum())
                if save > best_save + 1e-12:
                    best_save = save
                    best_col = c
                    best_counts = counts
            if best_col is None:
                t = tail_order(avail)
                return {int(r): t for r in rows}
            cc = code_arr[best_col][rows]
            rest = [c for c in avail if c != best_col]
            out = {}
            for code in np.unique(cc):
                sub_rows = rows[cc == code]
                if len(sub_rows) > group_cap * len(rows) and len(np.unique(cc)) <= 1:
                    sub = {int(r): rest for r in sub_rows}
                else:
                    sub = rec(sub_rows, rest, depth + 1)
                for r, order in sub.items():
                    out[r] = [best_col] + order
            return out

        try:
            return rec(rows, set(cols), 0)
        except Exception:
            t = tail_order(cols)
            return {int(r): t for r in rows}

    # ------------------------------------------------------------ merge fixes

    def _apply_merges(self, ordering, merge_map, col_pos):
        """Force each col_merge group to be contiguous (in given order)."""
        out = []
        seen = set()
        for c in ordering:
            if c in merge_map:
                g = merge_map[c]
                key = id(g)
                if key in seen:
                    continue
                seen.add(key)
                out.extend(g)
            else:
                out.append(c)
        return out

    # ------------------------------------------------------------------ main

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
        n_rows, n_cols = df.shape
        if n_rows == 0 or n_cols == 0:
            return df.copy(), [[] for _ in range(n_rows)]

        cols = list(df.columns)
        col_pos = {c: i for i, c in enumerate(cols)}

        # Merge groups: lists of existing columns, kept contiguous in order.
        merge_groups = []
        for group in (col_merge or []):
            g = [c for c in group if c in col_pos]
            if len(g) >= 2:
                merge_groups.append(g)
        merge_map = {}
        for g in merge_groups:
            for c in g:
                merge_map[c] = g

        # Serialize every cell once (scoring representation only).
        cellstr = {}
        for c in cols:
            cellstr[c] = [self._cell_str(v) for v in df[c].tolist()]

        # Global column scores.
        gscore = {}
        gscore2 = {}
        for c in cols:
            cnt = Counter(cellstr[c])
            s1 = 0
            s2 = 0
            for v, n in cnt.items():
                lv = len(v)
                if n > 1:
                    s1 += lv * n * (n - 1)
                    s2 += lv * (n - 1)
            gscore[c] = s1
            gscore2[c] = s2

        # Candidate 1 & 2: global orders.
        order_a, order_b = self._global_orders(cols, gscore, gscore2, col_pos)

        # Candidate 3: conditional partition tree (bounded).
        use_tree = (n_rows * n_cols) <= 3_000_000
        candidates = [(order_a, None), (order_b, None)]
        if use_tree:
            code_arr = {}
            lens_arr = {}
            for c in cols:
                codes, _ = pd.factorize(pd.Series(cellstr[c], dtype=object), sort=False)
                code_arr[c] = np.asarray(codes, dtype=np.int64)
                lens_arr[c] = np.fromiter((len(s) for s in cellstr[c]),
                                          dtype=np.float64, count=n_rows)
            all_rows = np.arange(n_rows, dtype=np.int64)
            tree = self._tree_orders(cols, col_pos, code_arr, lens_arr,
                                     gscore, all_rows)
            candidates.append((None, tree))

        # Evaluate candidates with the exact trie objective.
        col_lists = [cellstr[c] for c in cols]
        best_score = -1
        best_orders = None

        for flat, tree in candidates:
            if flat is not None:
                if merge_groups:
                    flat = self._apply_merges(flat, merge_map, col_pos)
                orders = [flat] * n_rows
            else:
                flat0 = tail = None
                orders = []
                for i in range(n_rows):
                    o = tree.get(i) or sorted(cols, key=lambda c: (-gscore[c], col_pos[c]))
                    if merge_groups:
                        o = self._apply_merges(o, merge_map, col_pos)
                    orders.append(o)
            # Serialize rows under this ordering.
            strings = ["".join(cellstr[c][i] for c in orders[i])
                       for i in range(n_rows)]
            score = self._trie_reuse(strings)
            if score > best_score:
                best_score = score
                best_orders = orders

        # Build output: original values permuted per row.
        orig_vals = {c: df[c].tolist() for c in cols}
        data = []
        for i in range(n_rows):
            data.append([orig_vals[c][i] for c in best_orders[i]])
        out = pd.DataFrame(data, columns=cols)
        out = out.astype(object) if n_cols else out

        return out, best_orders

# EVOLVE-BLOCK-END