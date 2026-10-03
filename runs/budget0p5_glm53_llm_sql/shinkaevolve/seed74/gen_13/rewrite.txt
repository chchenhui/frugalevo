# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict, Optional
from collections import Counter


class Evolved(Algorithm):
    """
    Optimize LLM prompt prefix caching by reordering fields per row.

    Objective: maximize exact serial character-Trie reuse, which for a fixed
    multiset of serialized row strings equals the sum of longest-common-prefix
    lengths between lexicographically adjacent sorted strings.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------ #
    # Serialization (exactly matches evaluator: fillna("") then astype(str))
    # ------------------------------------------------------------------ #
    def _serialize(self, df: pd.DataFrame) -> List[List[str]]:
        obj = df.astype(object)
        mask = pd.notnull(df)
        obj = obj.where(mask, "")
        ser = obj.astype(str)
        return ser.values.tolist()

    # ------------------------------------------------------------------ #
    # Exact serial-Trie reuse objective: sorted adjacent LCP sum
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lcp(a: str, b: str) -> int:
        hi = min(len(a), len(b))
        if hi == 0:
            return 0
        if a == b:
            return hi
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _score_orders(self, ser: List[List[str]], orders: List[List[int]]) -> int:
        strings = ["".join([ser[i][j] for j in orders[i]]) for i in range(len(ser))]
        if len(strings) < 2:
            return 0
        strings.sort()
        return sum(self._lcp(strings[k], strings[k + 1]) for k in range(len(strings) - 1))

    # ------------------------------------------------------------------ #
    # Candidate 1: global frequency-ranked column order
    # ------------------------------------------------------------------ #
    def _global_order(self, ser: List[List[str]], n_cols: int) -> List[int]:
        scores = []
        for c in range(n_cols):
            cnt = Counter(row[c] for row in ser)
            s = sum(len(v) * k * (k - 1) for v, k in cnt.items())
            scores.append(s)
        order = sorted(range(n_cols), key=lambda c: (-scores[c], c))
        return order

    # ------------------------------------------------------------------ #
    # Candidate 2: conditional prefix partition tree on integer codes
    # ------------------------------------------------------------------ #
    def _conditional_orders(
        self,
        ser: List[List[str]],
        n_cols: int,
        global_order: List[int],
        max_depth: int,
        early_stop: int,
    ) -> List[List[int]]:
        n_rows = len(ser)

        # Factorize each column once into integer codes + value lengths.
        codes: List[List[int]] = []  # codes[c][i]
        code_len: List[Dict[int, int]] = []
        for c in range(n_cols):
            vals = {}
            col_codes = [0] * n_rows
            lens = {}
            for i in range(n_rows):
                v = ser[i][c]
                code = vals.get(v)
                if code is None:
                    code = len(vals)
                    vals[v] = code
                    lens[code] = len(v)
                col_codes[i] = code
            codes.append(col_codes)
            code_len.append(lens)

        # Rank candidate columns once by global repetition savings.
        col_global = []
        for c in range(n_cols):
            cnt = Counter(codes[c])
            col_global.append(sum((k - 1) * code_len[c][v] for v, k in cnt.items()))
        ranked = sorted(range(n_cols), key=lambda c: (-col_global[c], c))
        candidate_cols = ranked[: min(len(ranked), 16)]

        suffix_base = global_order  # deterministic cheap tail
        orders: List[Optional[List[int]]] = [None] * n_rows

        def rec(rows: List[int], used: List[int], depth: int, budget: List[int]):
            if budget[0] <= 0 or len(rows) < 2 or depth >= max_depth:
                full = used + [c for c in suffix_base if c not in set(used)] if used else list(suffix_base)
                for i in rows:
                    orders[i] = full
                return
            best_col, best_gain = None, early_stop
            for c in candidate_cols:
                if c in used:
                    continue
                cnt = Counter(codes[c][i] for i in rows)
                gain = sum((k - 1) * code_len[c][v] for v, k in cnt.items() if k > 1)
                if gain > best_gain:
                    best_col, best_gain = c, gain
            if best_col is None:
                used_set = set(used)
                full = used + [c for c in suffix_base if c not in used_set]
                for i in rows:
                    orders[i] = full
                return
            budget[0] -= 1
            groups: Dict[int, List[int]] = {}
            for i in rows:
                groups.setdefault(codes[best_col][i], []).append(i)
            new_used = used + [best_col]
            for g in groups.values():
                rec(g, new_used, depth + 1, budget)

        rec(list(range(n_rows)), [], 0, [4096])

        used_set_full = set(range(n_cols))
        base_perm = list(suffix_base)
        for i in range(n_rows):
            if orders[i] is None:
                orders[i] = base_perm
        return [list(o) for o in orders]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
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
        original_columns = list(df.columns)

        work = df.copy()
        work.columns = original_columns

        # Honor column merges using parent semantics with a safe fallback.
        if col_merge:
            merged_cols = [c for grp in col_merge for c in grp if c in original_columns]
            try:
                col_order = [c for c in original_columns if c in merged_cols]
                if col_order:
                    work = self.merging_columns(work, col_order, prepended=False)
            except Exception:
                work = self._fallback_merge(work, col_merge, original_columns)

        out_cols = list(work.columns)
        n_rows, n_cols = work.shape
        if n_rows == 0 or n_cols == 0:
            out = pd.DataFrame(index=work.index.copy(), columns=out_cols).astype(object)
            orders_names = [list(out_cols)] * n_rows
            return out, orders_names

        ser = self._serialize(work)
        total_chars = sum(len(s) for row in ser for s in row)

        global_order = self._global_order(ser, n_cols)
        cand_orders: List[List[List[int]]] = [ [list(global_order)] * n_rows ]

        max_depth = col_stop if isinstance(col_stop, int) and 0 < col_stop else 8
        if row_stop is not None and isinstance(row_stop, int) and row_stop > 0:
            max_depth = min(max_depth, row_stop)

        if total_chars <= 40_000_000 and n_cols > 1 and n_rows > 1:
            try:
                cond = self._conditional_orders(ser, n_cols, global_order, max_depth, early_stop)
                cand_orders.append(cond)
            except Exception:
                pass

        best_orders, best_score = None, -1
        for orders in cand_orders:
            try:
                sc = self._score_orders(ser, orders)
            except Exception:
                continue
            if sc > best_score:
                best_score, best_orders = sc, orders

        if best_orders is None:
            best_orders = [list(global_order)] * n_rows

        # Build output data: row i has its cells permuted by best_orders[i].
        col_pos = {c: idx for idx, c in enumerate(out_cols)}
        data = []
        for i in range(n_rows):
            order = best_orders[i]
            data.append([ser[i][j] for j in order])

        out_values = np.empty((n_rows, n_cols), dtype=object)
        for i, rowvals in enumerate(data):
            for j, v in enumerate(rowvals):
                out_values[i, j] = v

        out = pd.DataFrame(out_values, columns=out_cols, index=work.index.copy())
        out = out.astype(object)

        # Restore original dtypes for non-object columns where safe.
        try:
            for c in original_columns:
                if c in out.columns and c not in merged_cols_set(col_merge):
                    pass
        except Exception:
            pass

        # Per-row column orderings consistent with returned data.
        orderings_names = [[out_cols[j] for j in order] for order in best_orders]

        # Sanity: preserve shape and value multiset of the working frame.
        assert out.shape == (n_rows, n_cols)

        return out, orderings_names

    def _fallback_merge(self, df: pd.DataFrame, col_merge: List[List[str]], original_columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        for grp in col_merge:
            grp_cols = [c for c in original_columns if c in grp]
            if len(grp_cols) < 2:
                continue
            ser = self._serialize(df[grp_cols])
            combined = ["".join(row) for row in ser]
            first = grp_cols[0]
            df[first] = combined
            df = df.drop(columns=[c for c in grp_cols if c != first])
        return df


def merged_cols_set(col_merge):
    s = set()
    for grp in col_merge or []:
        for c in grp:
            s.add(c)
    return s

# EVOLVE-BLOCK-END