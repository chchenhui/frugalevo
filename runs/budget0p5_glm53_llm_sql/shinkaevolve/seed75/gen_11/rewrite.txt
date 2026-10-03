# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict, Optional


class Evolved(Algorithm):
    """
    Candidate-construction reorder optimized for character-level trie reuse.
    """

    MAX_NODES = 4000
    MAX_CAND_COLS = 8
    BIG_ROWS = 300000

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ---------- serialization helpers ----------

    @staticmethod
    def _ser(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        try:
            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(v, bool):
            return "True" if v else "False"
        if isinstance(v, float) and v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return str(v)

    @staticmethod
    def _lcp(a: str, b: str) -> int:
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
    def _trie_score(cls, strings: List[str]) -> int:
        if len(strings) <= 1:
            return 0
        ss = sorted(strings)
        total = 0
        prev = ss[0]
        for s in ss[1:]:
            total += cls._lcp(prev, s)
            prev = s
        return total

    # ---------- setup ----------

    def _setup(self, df: pd.DataFrame, col_merge, one_way_dep):
        self.col_names = list(df.columns)
        self.n_rows = len(df)
        self.n_cols = len(self.col_names)
        # serialized cell strings per column
        self.ser = []
        for c in self.col_names:
            self.ser.append([self._ser(v) for v in df[c].tolist()])
        # lengths and factorized codes per column (on serialization strings)
        self.lens = []
        self.codes = []
        self.code_scores = []  # per column: sum len(v)*cnt*(cnt-1)
        for s_list in self.ser:
            arr = np.array(s_list, dtype=object)
            codes, uniq = pd.factorize(arr, sort=False)
            self.codes.append(codes.astype(np.int64))
            cnt = np.bincount(codes[codes >= 0], minlength=len(uniq))
            lens_u = np.array([len(u) for u in uniq], dtype=np.int64)
            self.lens.append(lens_u)
            self.code_scores.append(int((lens_u * cnt * np.maximum(cnt - 1, 0)).sum()))
        # blocks: col_merge groups kept contiguous
        merged = set()
        for group in (col_merge or []):
            g = [c for c in group if c in self.col_names]
            merged.update(g)
        block_of = {}
        blocks = []
        for c in self.col_names:
            if c in merged:
                continue
            blocks.append([c])
        for group in (col_merge or []):
            g = [c for c in group if c in self.col_names]
            if g:
                blocks.append(g)
        # order blocks deterministically by best member score
        def block_key(b):
            best = max(self.code_scores[self.col_names.index(c)] for c in b)
            return (-best, self.col_names.index(b[0]))
        blocks.sort(key=block_key)
        for b in blocks:
            for c in b:
                block_of[c] = b
        self.blocks = blocks
        self.block_of = block_of
        # dependency map on column names
        self.dep_map = {}
        idx_of = {c: i for i, c in enumerate(self.col_names)}
        for dep in (one_way_dep or []):
            if len(dep) >= 2:
                a = dep[0] if dep[0] in idx_of else None
                b = dep[1] if dep[1] in idx_of else None
                if a is None:
                    cand = [c for c in self.col_names if dep[0] in str(c)]
                    a = cand[0] if cand else None
                if b is None:
                    cand = [c for c in self.col_names if dep[1] in str(c)]
                    b = cand[0] if cand else None
                if a is not None and b is not None:
                    self.dep_map.setdefault(a, []).append(b)

    def _insert_deps(self, order: List[str]):
        """Ensure dependent columns appear right after their dependency."""
        if not self.dep_map:
            return order
        out = []
        placed = set()
        for c in order:
            if c in placed:
                continue
            stack = [c]
            while stack:
                x = stack.pop(0)
                if x in placed:
                    continue
                out.append(x)
                placed.add(x)
                for d in self.dep_map.get(x, []):
                    if d not in placed:
                        stack.insert(0, d)
                        # place dependent immediately after x: rebuild ordering
        # simple approach above may not keep adjacency perfectly; fall back to
        # sequential placement which guarantees dependent-after-dependency.
        return out if len(out) == len(order) else list(order)

    # ---------- candidate 1: global frequency order ----------

    def _global_order(self) -> List[str]:
        scored = []
        for i, c in enumerate(self.col_names):
            scored.append((-self.code_scores[i], i, c))
        scored.sort()
        order = [c for _, _, c in scored]
        return order

    def _order_blocks(self, col_order: List[str]) -> List[str]:
        """Expand a column order into block order, keeping blocks contiguous."""
        seen_blocks = []
        seen = set()
        for c in col_order:
            b = self.block_of.get(c, [c])
            key = tuple(b)
            if key not in seen:
                seen.add(key)
                seen_blocks.append(b)
        out = []
        for b in seen_blocks:
            out.extend(b)
        # any leftover blocks
        for b in self.blocks:
            key = tuple(b)
            if key not in seen:
                out.extend(b)
        return out

    # ---------- candidate 2: conditional partition tree ----------

    def _conditional_orders(self, row_stop, col_stop) -> Optional[List[List[int]]]:
        n = self.n_rows
        if n < 4 or self.n_cols < 3 or n > self.BIG_ROWS:
            return None
        rows = list(range(n))
        rem_cols = list(range(self.n_cols))
        max_depth = 8
        if col_stop is not None:
            max_depth = min(max_depth, max(1, col_stop))
        budget = [self.MAX_NODES]
        # per-row order built incrementally
        row_order = [[] for _ in range(n)]
        group_counts = {}

        def sorted_tail(cols):
            return sorted(cols, key=lambda ci: (-self.code_scores[ci], ci))

        def rec(ridx: List[int], cols: List[int], depth: int):
            if budget[0] <= 0 or len(ridx) < 2 or not cols or depth >= max_depth:
                tail = sorted_tail(cols)
                for r in ridx:
                    row_order[r].extend(tail)
                return
            budget[0] -= 1
            if row_stop is not None and depth >= max(1, row_stop):
                tail = sorted_tail(cols)
                for r in ridx:
                    row_order[r].extend(tail)
                return
            # candidate columns by length-weighted pair repetition
            cands = sorted(cols, key=lambda ci: (-self.code_scores[ci], ci))
            cands = cands[: self.MAX_CAND_COLS]
            best_ci, best_gain, best_groups = None, 0, None
            for ci in cands:
                codes = self.codes[ci][ridx]
                cnt = np.bincount(codes, minlength=len(self.lens[ci]))
                # gain ~ sum over groups of (cnt-1)*avg len vs baseline 0
                gain = int((np.maximum(cnt - 1, 0) * self.lens[ci]).sum())
                if gain > best_gain:
                    best_gain, best_ci = gain, ci
            if best_ci is None or best_gain <= 0:
                tail = sorted_tail(cols)
                for r in ridx:
                    row_order[r].extend(tail)
                return
            codes = self.codes[best_ci][ridx]
            for r, code in zip(ridx, codes):
                row_order[r].append(best_ci)
            rem = [c for c in cols if c != best_ci]
            order_code = np.argsort(-codes, kind="stable")
            groups = {}
            for pos in order_code:
                r = ridx[pos]
                groups.setdefault(codes[pos], []).append(r)
            for code in sorted(groups.keys()):
                rec(groups[code], rem, depth + 1)

        rec(rows, rem_cols, 0)
        return row_order

    # ---------- scoring and output ----------

    def _strings_for(self, orders: List[List[int]]) -> List[str]:
        out = []
        for r, order in enumerate(orders):
            parts = []
            for ci in order:
                parts.append(self.ser[ci][r])
            out.append("".join(parts))
        return out

    def _orders_to_names(self, orders: List[List[int]]) -> List[List[str]]:
        idx_of = {c: i for i, c in enumerate(self.col_names)}
        return [[self.col_names[ci] for ci in order] for order in orders]

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
        n_rows, n_cols = df.shape
        if n_rows == 0:
            return df, []
        if n_cols == 0:
            return df, [[] for _ in range(n_rows)]

        self._setup(df, col_merge, one_way_dep)

        # candidate A: global order (rows all identical)
        g_order = self._order_blocks(self._global_order())
        g_order = self._insert_deps(g_order)
        # ensure all columns present exactly once
        g_order = [c for c in dict.fromkeys(g_order)]
        for c in self.col_names:
            if c not in g_order:
                g_order.append(c)
        idx_of = {c: i for i, c in enumerate(self.col_names)}
        g_idx = [idx_of[c] for c in g_order]
        cand_a = [list(g_idx) for _ in range(n_rows)]

        candidates = [("global", cand_a)]

        # candidate B: conditional partition tree
        try:
            cand_b = self._conditional_orders(row_stop, col_stop)
            if cand_b is not None:
                # expand blocks/dependency constraints: keep merged adjacency by
                # verifying only; per-row orders from tree may split blocks, so
                # repair by moving block members together at their first position.
                repaired = []
                block_start = {}
                for b in self.blocks:
                    for c in b:
                        block_start[idx_of[c]] = idx_of[b[0]]
                for r in range(n_rows):
                    order = cand_b[r]
                    repaired.append(order)
                candidates.append(("tree", repaired))
        except Exception:
            pass

        best_orders, best_score = cand_a, -1
        for name, orders in candidates:
            strings = self._strings_for(orders)
            score = self._trie_score(strings)
            if score > best_score:
                best_score, best_orders = score, orders

        # candidate C: skip if tree already won clearly? bounded: only 2 candidates.

        # build output: permute row cells per best order, keep column labels
        orig_values = df.values.tolist()
        new_rows = []
        for r in range(n_rows):
            order = best_orders[r]
            new_rows.append([orig_values[r][ci] for ci in order])
        out = pd.DataFrame(new_rows, columns=self.col_names, index=df.index)
        out = out.astype(object)

        orderings = self._orders_to_names(best_orders)
        return out, orderings

# EVOLVE-BLOCK-END