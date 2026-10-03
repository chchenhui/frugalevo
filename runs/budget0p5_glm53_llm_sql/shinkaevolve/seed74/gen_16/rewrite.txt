# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict, Optional


class Evolved(Algorithm):
    """
    Character-Trie prefix-caching optimizer.

    Builds a few cheap candidate per-row column orderings, scores each with
    the true ideal serial-Trie reuse (sum of adjacent LCPs of sorted serialized
    row strings), and returns the best. Data, shape, and row identity are
    preserved exactly; only column order (per row) changes.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.dep_graph = None

    # ------------------------------------------------------------------ #
    # Serialization helpers (match evaluator: fillna("") then astype(str)) #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cell_str(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        try:
            if v != v:  # NaN of any dtype
                return ""
        except Exception:
            pass
        return str(v)

    def _serialize_columns(self, df: pd.DataFrame):
        """Return (col_strs, col_vals): per-column list of str for scoring and
        per-column list of original values for output."""
        col_strs = {}
        col_vals = {}
        for c in df.columns:
            s = df[c]
            strs = [self._cell_str(v) for v in s.tolist()]
            col_strs[c] = strs
            col_vals[c] = s.tolist()
        return col_strs, col_vals

    # ------------------------------------------------------------------ #
    # Column scoring: sum over values v of len(v) * count(v) * (count(v)-1) #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _col_scores(col_strs: Dict, columns: List, use_len: bool = True) -> Dict:
        scores = {}
        for c in columns:
            strs = col_strs[c]
            counts = {}
            for s in strs:
                counts[s] = counts.get(s, 0) + 1
            sc = 0
            if use_len:
                for s, n in counts.items():
                    if n > 1:
                        sc += len(s) * n * (n - 1)
            else:
                for s, n in counts.items():
                    if n > 1:
                        sc += n * (n - 1)
            scores[c] = sc
        return scores

    # ------------------------------------------------------------------ #
    # Ideal Trie objective: sort strings, sum adjacent LCPs                  #
    # ------------------------------------------------------------------ #

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

    @classmethod
    def _trie_reuse(cls, strings: List[str]) -> int:
        if len(strings) <= 1:
            return 0
        ss = sorted(strings)
        total = 0
        prev = ss[0]
        for cur in ss[1:]:
            total += cls._lcp(prev, cur)
            prev = cur
        return total

    # ------------------------------------------------------------------ #
    # Candidate orderings                                                   #
    # ------------------------------------------------------------------ #

    def _global_order(self, columns: List, scores: Dict) -> List:
        return sorted(columns, key=lambda c: (-scores.get(c, 0), str(c)))

    def _tree_orders(
        self,
        n: int,
        columns: List,
        col_strs: Dict,
        scores: Dict,
        max_depth: int,
        min_group: int,
        nuniq_limit: Optional[int],
    ) -> List[List]:
        """Conditional prefix partition tree producing per-row column orders."""
        orders = [None] * n
        rows = list(range(n))

        # factorized codes per column (once)
        codes = {}
        for c in columns:
            uniq = {}
            cc = []
            for s in col_strs[c]:
                if s not in uniq:
                    uniq[s] = len(uniq)
                cc.append(uniq[s])
            codes[c] = cc

        def assign(rows_idx, remaining):
            tail = sorted(remaining, key=lambda c: (-scores.get(c, 0), str(c)))
            for r in rows_idx:
                orders[r] = tail

        def rec(rows_idx, remaining, depth):
            if (depth >= max_depth or len(rows_idx) < min_group
                    or not remaining):
                assign(rows_idx, remaining)
                return
            # choose branch column: best length-weighted repetition
            best_col, best_sc = None, -1
            for c in remaining:
                sc = scores.get(c, 0)
                if sc > best_sc:
                    if nuniq_limit is not None:
                        uq = len(set(col_strs[c][r] for r in rows_idx))
                        if uq > nuniq_limit:
                            continue
                    best_col, best_sc = c, sc
            if best_col is None or best_sc <= 0:
                assign(rows_idx, remaining)
                return
            # partition rows by this column's value within group
            groups = {}
            col_code = codes[best_col]
            for r in rows_idx:
                groups.setdefault(col_code[r], []).append(r)
            if len(groups) <= 1:
                # column constant within group: keep as prefix, drop it
                rest = [c for c in remaining if c != best_col]
                for r in rows_idx:
                    orders[r] = None  # placeholder, filled below
                # recurse on remainder with prefix fixed
                prefix = [best_col]
                sub = {k: v for k, v in [(c, codes[c]) for c in rest]}
                _ = sub
                saved = [orders[r] for r in rows_idx]
                rec(rows_idx, rest, depth + 1)
                for r in rows_idx:
                    orders[r] = prefix + orders[r]
                return
            rest = [c for c in remaining if c != best_col]
            for _, g_rows in sorted(groups.items(), key=lambda kv: kv[0]):
                rec(g_rows, rest, depth + 1)
                for r in g_rows:
                    orders[r] = [best_col] + orders[r]

        rec(rows, list(columns), 0)
        for r in range(n):
            if orders[r] is None:
                orders[r] = list(columns)
        return orders

    # ------------------------------------------------------------------ #
    # Public API                                                            #
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
        initial_df = df

        # Honor column merges with existing API semantics.
        if col_merge:
            try:
                _, col_stats = self.calculate_col_stats(df, enable_index=True)
                reordered_columns = [c for c, _, _, _ in col_stats]
                for col_to_merge in col_merge:
                    final_col_order = [c for c in reordered_columns if c in col_to_merge]
                    if final_col_order:
                        df = self.merging_columns(df, final_col_order, prepended=False)
            except Exception:
                # If merging infrastructure differs, proceed without merging
                # but never alter data.
                df = initial_df.copy()

        columns = list(df.columns)
        n = len(df)

        if n == 0 or len(columns) == 0:
            return df.copy(), [[] for _ in range(n)] if columns else []

        col_strs, col_vals = self._serialize_columns(df)
        scores_len = self._col_scores(col_strs, columns, use_len=True)
        scores_cnt = self._col_scores(col_strs, columns, use_len=False)

        # Bounds
        max_depth = col_stop if (col_stop is not None and col_stop > 0) else 6
        max_depth = min(max_depth, len(columns))
        min_group = 8
        if row_stop is not None and row_stop > 0:
            min_group = max(min_group, 1)  # row_stop bounds overall; depth capped below
        if early_stop and early_stop > 0:
            min_group = max(min_group, early_stop)
        nuniq_limit = max(1, int(n * distinct_value_threshold))

        # Candidate 1: global frequency-ranked (length-weighted), desc
        g1 = self._global_order(columns, scores_len)
        cand1 = [g1] * n

        # Candidate 2: conditional partition tree (length-weighted)
        cand2 = self._tree_orders(
            n, columns, col_strs, scores_len, max_depth, min_group, nuniq_limit
        )

        # Candidate 3: partition tree using count*(count-1) without lengths
        cand3 = self._tree_orders(
            n, columns, col_strs, scores_cnt, max_depth, min_group, nuniq_limit
        )

        def row_strings(orders):
            out = []
            for i in range(n):
                o = orders[i]
                out.append("".join(col_strs[c][i] for c in o))
            return out

        best_orders, best_score = cand1, self._trie_reuse(row_strings(cand1))
        for cand in (cand2, cand3):
            if cand is cand1:
                continue
            sc = self._trie_reuse(row_strings(cand))
            if sc > best_score:
                best_orders, best_score = cand, sc

        # Build output DataFrame preserving every cell value and row identity.
        # Column labels keep the original set; row i position j holds the value
        # of column best_orders[i][j].
        data = []
        for i in range(n):
            o = best_orders[i]
            data.append([col_vals[c][i] for c in o])
        out_df = pd.DataFrame(data, columns=columns, index=df.index)
        out_df = out_df.astype(object)

        assert out_df.shape == df.shape, "shape mismatch"
        column_orderings = [list(o) for o in best_orders]

        if col_merge:
            # Merging changes column count; shape assertions follow merge semantics
            if out_df.shape[0] != initial_df.shape[0]:
                raise AssertionError("row count changed")
        else:
            assert out_df.shape == initial_df.shape, "final shape mismatch"

        return out_df, column_orderings

# EVOLVE-BLOCK-END