# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict
from collections import defaultdict


class Evolved(Algorithm):
    """
    Reorders columns (per row) to maximize character-Trie prefix reuse of the
    serialized rows, while preserving every cell value and row identity.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.dep_graph = None  # not used, kept for API compatibility
        self.num_rows = 0
        self.num_cols = 0
        self.base = 2000

    # ------------------------------------------------------------------
    # serialization (matches evaluator representation; does not alter data)
    # ------------------------------------------------------------------
    def _serialize(self, df: pd.DataFrame) -> pd.DataFrame:
        obj = df.astype(object)
        obj = obj.where(pd.notna(obj), "")
        return obj.astype(str)

    # ------------------------------------------------------------------
    # one-way dependency: stable topological order for locked columns
    # ------------------------------------------------------------------
    def _topo_order(self, locked, pairs, cols):
        order = [c for c in cols if c in locked]
        if not order:
            return []
        for _ in range(len(order) * len(order) + 1):
            changed = False
            for a, b in pairs:
                if a in order and b in order:
                    ia, ib = order.index(a), order.index(b)
                    if ia > ib:
                        order.remove(a)
                        order.insert(ib, a)
                        changed = True
            if not changed:
                break
        return order

    # ------------------------------------------------------------------
    # ideal Trie reuse: sum of adjacent LCPs of sorted strings
    # ------------------------------------------------------------------
    @staticmethod
    def _lcp(a: str, b: str) -> int:
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

    def _trie_reuse(self, strings) -> int:
        if not strings:
            return 0
        ss = sorted(strings)
        total = 0
        prev = ss[0]
        for s in ss[1:]:
            total += self._lcp(prev, s)
            prev = s
        return total

    # ------------------------------------------------------------------
    # conditional partition tree producing per-row orders
    # ------------------------------------------------------------------
    def _assign_tree(self, free_cols, rows, prefix, depth, max_depth,
                     early_stop, col_vals, cidx, base_rank, orders):
        if (not free_cols or len(rows) < 2 or depth >= max_depth):
            suffix = sorted(free_cols, key=lambda c: base_rank[c])
            full = prefix + suffix
            for r in rows:
                orders[r] = full
            return
        best_col, best_gain, best_groups = None, 0, None
        for c in free_cols:
            vals = col_vals[cidx[c]]
            groups = defaultdict(list)
            for r in rows:
                groups[vals[r]].append(r)
            gain = 0
            for v, g in groups.items():
                k = len(g)
                if k > 1:
                    gain += len(v) * k * (k - 1)
            if gain > best_gain:
                best_gain, best_col, best_groups = gain, c, groups
        if best_col is None or best_gain <= max(0, early_stop):
            suffix = sorted(free_cols, key=lambda c: base_rank[c])
            full = prefix + suffix
            for r in rows:
                orders[r] = full
            return
        rest = [c for c in free_cols if c != best_col]
        nxt = prefix + [best_col]
        for _, g in best_groups.items():
            self._assign_tree(rest, g, nxt, depth + 1, max_depth,
                              early_stop, col_vals, cidx, base_rank, orders)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
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

        n_rows = len(df)
        if n_rows == 0 or df.shape[1] == 0:
            cols = list(df.columns)
            return df.astype(object), [list(cols) for _ in range(n_rows)]

        # duplicate labels -> cannot build unambiguous per-row orders
        if len(set(map(str, df.columns))) != df.shape[1]:
            cols = list(df.columns)
            return df.astype(object), [list(cols) for _ in range(n_rows)]

        work = df.copy()

        # honor column merges with existing API semantics
        if col_merge and hasattr(self, "merging_columns"):
            for grp in col_merge:
                present = [c for c in work.columns if c in grp]
                if len(present) > 1:
                    try:
                        work = self.merging_columns(work, present, prepended=False)
                    except Exception:
                        pass

        cols = list(work.columns)
        n_cols = len(cols)
        self.num_rows, self.num_cols = n_rows, n_cols

        # exact serialized representation for scoring only
        ser = self._serialize(work)
        col_vals = [ser.iloc[:, j].tolist() for j in range(n_cols)]
        cidx = {c: j for j, c in enumerate(cols)}

        # per-column repetition statistics (length-weighted pairs)
        stats = {}
        for c in cols:
            cnt = defaultdict(int)
            for v in col_vals[cidx[c]]:
                cnt[v] += 1
            rep = sum(len(v) * k * (k - 1) for v, k in cnt.items())
            stats[c] = (len(cnt), rep)

        # one-way dependency locking
        dep_pairs = []
        for dep in (one_way_dep or []):
            if not isinstance(dep, (list, tuple)) or len(dep) < 2:
                continue
            ca = [c for c in cols if dep[0] in c]
            cb = [c for c in cols if dep[1] in c]
            if len(ca) == 1 and len(cb) == 1 and ca[0] != cb[0]:
                dep_pairs.append((ca[0], cb[0]))
        locked = set()
        for a, b in dep_pairs:
            locked.add(a)
            locked.add(b)
        locked_order = self._topo_order(locked, dep_pairs, cols)
        free_cols = [c for c in cols if c not in locked]

        # global candidate order A: fewest distinct values first, then repetition
        orderA = sorted(free_cols, key=lambda c: (stats[c][0], -stats[c][1], c))
        orderA = orderA + locked_order
        # global candidate order B: pure length-weighted repetition ranking
        orderB = sorted(free_cols, key=lambda c: (-stats[c][1], stats[c][0], c))
        orderB = orderB + locked_order

        base_rank = {c: i for i, c in enumerate(orderA)}

        # depth bounds
        max_depth = min(len(free_cols), 12)
        if col_stop:
            max_depth = min(max_depth, int(col_stop))
        if row_stop:
            max_depth = min(max_depth, int(row_stop))

        # candidate per-row orderings
        candidates = {
            "A": [orderA] * n_rows,
            "B": [orderB] * n_rows,
        }

        # bounded budget: skip the tree (and full scoring) on very large inputs
        budget = n_rows * n_cols
        use_tree = budget <= 2_000_000
        do_score = budget <= 6_000_000

        if use_tree and free_cols:
            tree_orders = {}
            self._assign_tree(
                free_cols, list(range(n_rows)), [], 0, max_depth,
                early_stop, col_vals, cidx, base_rank, tree_orders,
            )
            fallback = orderA
            candidates["T"] = [tree_orders.get(i, fallback) for i in range(n_rows)]

        def row_strings(per_row_orders):
            return [
                "".join(col_vals[cidx[c]][i] for c in per_row_orders[i])
                for i in range(n_rows)
            ]

        best_name, best_orders, best_score = None, candidates["A"], -1
        if do_score:
            for name, orders in candidates.items():
                score = self._trie_reuse(row_strings(orders))
                if score > best_score:
                    best_name, best_score, best_orders = name, score, orders
        else:
            best_orders = candidates["A"]

        # build output with ORIGINAL values permuted per row (object dtype)
        raw = work.astype(object)
        raw = raw.to_numpy(dtype=object)
        data = [[None] * n_cols for _ in range(n_rows)]
        for i in range(n_rows):
            order = best_orders[i]
            assert len(order) == n_cols and set(order) == set(cols)
            for j, c in enumerate(order):
                data[i][j] = raw[i, cidx[c]]

        out = pd.DataFrame(data, columns=cols, index=df.index).astype(object)
        column_orderings = [list(o) for o in best_orders]
        return out, column_orderings

# EVOLVE-BLOCK-END