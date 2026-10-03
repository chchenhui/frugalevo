# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict
from collections import Counter


class Evolved(Algorithm):
    """
    Trie-aware column reordering: builds a few cheap candidate per-row column
    orderings, scores them with the exact serial character-Trie reuse
    (sum of adjacent LCPs of sorted serialized rows), and returns the best.
    All original cells, rows and columns are preserved.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------ #
    # serialization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _serialize_column(col: pd.Series) -> List[str]:
        # matches the evaluator: fillna("") then astype(str)
        try:
            s = col.fillna("")
        except Exception:
            s = col.copy()
            s[col.isna()] = ""
        return s.astype(str).tolist()

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        if a == b:
            return len(a)
        n = min(len(a), len(b))
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _trie_score(self, row_strings: List[str]) -> int:
        if not row_strings:
            return 0
        if len(row_strings) > 1:
            arr = sorted(row_strings)
        else:
            arr = row_strings
        total = 0
        prev = arr[0]
        for i in range(1, len(arr)):
            cur = arr[i]
            total += self._lcp(prev, cur)
            prev = cur
        return total

    # ------------------------------------------------------------------ #
    # candidate orderings
    # ------------------------------------------------------------------ #
    def _global_order_pair_score(self, cols, ser_cols, n):
        # rank by sum over values of len(v) * cnt * (cnt - 1)
        scores = {}
        for c in cols:
            cnt = Counter(ser_cols[c])
            s = 0
            for v, k in cnt.items():
                if k > 1:
                    s += len(v) * k * (k - 1)
            scores[c] = s
        return sorted(cols, key=lambda c: (-scores[c], c)), scores

    def _global_order_len_freq(self, cols, ser_cols, n):
        # length-weighted total frequency (a different length/frequency tradeoff)
        scores = {}
        for c in cols:
            s = 0
            for v in ser_cols[c]:
                s += len(v)
            scores[c] = s
        return sorted(cols, key=lambda c: (-scores[c], c))

    def _tree_orders(self, cols, ser_cols, codes, code_str, global_order,
                     pair_scores, n, distinct_threshold):
        """Conditional prefix partition tree -> per-row column orders."""
        ncols = len(cols)
        orders = [None] * n
        max_depth = min(ncols, 12)
        # candidate partition columns: top-k by pair score, skipping
        # near-unique columns (they rarely create useful partitions)
        cand_limit = min(ncols, 8)
        cand_cols = []
        for c in global_order:
            if len(code_str[c]) <= max(2, int(distinct_threshold * n)) or n == 0:
                cand_cols.append(c)
            if len(cand_cols) >= cand_limit:
                break
        cand_set_global = [c for c in global_order]

        def suffix_order(remaining_set):
            return [c for c in cand_set_global if c in remaining_set]

        def rec(row_idx, remaining, prefix, depth):
            if not row_idx:
                return
            if len(row_idx) == 1 or depth >= max_depth or not remaining:
                suf = suffix_order(set(remaining))
                full = prefix + suf
                for i in row_idx:
                    orders[i] = full
                return
            best_col, best_gain = None, 0
            for c in cand_cols:
                if c not in remaining:
                    continue
                cc = codes[c]
                cnt = Counter(cc[i] for i in row_idx)
                gain = 0
                lens = code_str[c]
                for code, k in cnt.items():
                    if k > 1:
                        gain += len(lens[code]) * k * (k - 1)
                if gain > best_gain:
                    best_col, best_gain = c, gain
            if best_col is None:
                suf = suffix_order(set(remaining))
                full = prefix + suf
                for i in row_idx:
                    orders[i] = full
                return
            cc = codes[best_col]
            groups = {}
            for i in row_idx:
                groups.setdefault(cc[i], []).append(i)
            new_remaining = [c for c in remaining if c != best_col]
            new_prefix = prefix + [best_col]
            for g in groups.values():
                rec(g, new_remaining, new_prefix, depth + 1)

        rec(list(range(n)), cols, [], 0)
        for i in range(n):
            if orders[i] is None:
                orders[i] = list(global_order)
        return orders

    # ------------------------------------------------------------------ #
    # main API
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
        initial_df = df

        # honor column merges using existing API semantics
        work = df
        if col_merge:
            try:
                _, column_stats = self.calculate_col_stats(df, enable_index=True)
                reordered_columns = [col for col, _, _, _ in column_stats]
                for col_to_merge in col_merge:
                    final_col_order = [c for c in reordered_columns if c in col_to_merge]
                    if final_col_order:
                        work = self.merging_columns(work, final_col_order, prepended=False)
            except Exception:
                work = df

        n, ncols = work.shape
        if n == 0 or ncols == 0:
            return work.copy(), [[] for _ in range(n)]

        cols = list(work.columns)

        # serialize each column once (scoring representation only)
        ser_cols = {c: self._serialize_column(work[c]) for c in cols}

        # integer factorization of serialized values (for the tree)
        codes = {}
        code_str = {}
        for c in cols:
            cnt = Counter(ser_cols[c])
            mapping = {v: i for i, v in enumerate(sorted(cnt.keys()))}
            code_str[c] = sorted(cnt.keys())
            codes[c] = [mapping[v] for v in ser_cols[c]]

        # candidate 1: global ordering by pair-score
        order1, pair_scores = self._global_order_pair_score(cols, ser_cols, n)
        # candidate 2: conditional partition tree (per-row orders)
        tree_orders = self._tree_orders(
            cols, ser_cols, codes, code_str, order1, pair_scores,
            n, distinct_value_threshold,
        )
        # candidate 3: length-weighted total frequency ordering
        order3 = self._global_order_len_freq(cols, ser_cols, n)

        def rows_for(orders):
            out = []
            for i in range(n):
                out.append("".join(ser_cols[c][i] for c in orders[i]))
            return out

        cands = []
        cands.append(([order1] * n, rows_for([order1] * n)))
        cands.append((tree_orders, rows_for(tree_orders)))
        cands.append(([order3] * n, rows_for([order3] * n)))

        best_orders, best_rows, best_score = None, None, -1
        for orders, rstrings in cands:
            sc = self._trie_score(rstrings)
            if sc > best_score:
                best_score, best_orders, best_rows = sc, orders, rstrings

        # build output preserving every original cell value, per-row permutation
        col_pos = {c: j for j, c in enumerate(cols)}
        raw = work.to_numpy(dtype=object)
        data = np.empty((n, ncols), dtype=object)
        for i in range(n):
            row = raw[i]
            perm = best_orders[i]
            vals = [row[col_pos[c]] for c in perm]
            data[i, :] = vals

        out_cols = list(best_orders[0])
        out_df = pd.DataFrame(data, columns=out_cols, index=work.index)
        out_df = out_df.astype(object)

        assert out_df.shape == (n, ncols)
        # row identity preserved: index unchanged, each row keeps its values
        return out_df, [list(o) for o in best_orders]

# EVOLVE-BLOCK-END