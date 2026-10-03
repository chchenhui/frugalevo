# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List


class Evolved(Algorithm):
    """
    Reorders columns to maximize character-level prefix reuse of serialized
    rows in a shared Trie, while preserving all data and shapes.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------ #
    # serialization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cell_str(v) -> str:
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
            return str(v)
        return str(v)

    def _serialize_matrix(self, df: pd.DataFrame, order: List[str]) -> List[str]:
        """Serialize rows with a given column order, evaluator-style."""
        cols = [df[c].tolist() for c in order]
        n = len(df)
        out = []
        for i in range(n):
            out.append("".join(self._cell_str(c[i]) for c in cols))
        return out

    # ------------------------------------------------------------------ #
    # ideal serial-trie reuse scoring
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lcp(a: str, b: str) -> int:
        if a == b:
            return len(a)
        lo, hi = 0, min(len(a), len(b))
        while lo < hi:
            mid = (lo + hi) // 2
            if a[:mid] == b[:mid]:
                lo = mid + 1
            else:
                hi = mid
        # lo is first index where prefixes may differ; verify
        while lo > 0 and a[:lo] != b[:lo]:
            lo -= 1
        return lo

    def _score(self, row_strings: List[str]) -> int:
        if len(row_strings) <= 1:
            return sum(len(s) for s in row_strings)
        srt = sorted(row_strings)
        total = len(srt[0])
        prev = srt[0]
        for s in srt[1:]:
            if s != prev:
                total += self._lcp(prev, s)
                prev = s
            # identical strings share full prefix via trie; count once
        # count identical duplicates as full reuse
        from collections import Counter
        cnt = Counter(row_strings)
        for s, c in cnt.items():
            if c > 1:
                total += (c - 1) * len(s)
        return total

    # ------------------------------------------------------------------ #
    # candidate column orders
    # ------------------------------------------------------------------ #
    def _candidate_orders(self, df: pd.DataFrame) -> List[List[str]]:
        cols = list(df.columns)
        if len(cols) <= 1:
            return [cols]

        # value counts per column (on serialized representation)
        str_cols = {c: [self._cell_str(v) for v in df[c].tolist()] for c in cols}
        counts = {c: {} for c in cols}
        for c in cols:
            d = counts[c]
            for v in str_cols[c]:
                d[v] = d.get(v, 0) + 1

        # candidate 1: original order
        cands = [list(cols)]

        # candidate 2: sum len(v) * count * (count-1), desc
        def weighted_score(c):
            return sum(len(v) * n * (n - 1) for v, n in counts[c].items())
        cands.append(sorted(cols, key=lambda c: (-weighted_score(c), cols.index(c))))

        # candidate 3: frequency of most common value, then weighted score
        def freq_score(c):
            return max(counts[c].values()) if counts[c] else 0
        cands.append(sorted(cols, key=lambda c: (-freq_score(c), -weighted_score(c), cols.index(c))))

        # dedupe, keep order
        seen = set()
        unique = []
        for cand in cands:
            key = tuple(cand)
            if key not in seen:
                seen.add(key)
                unique.append(cand)
        return unique

    # ------------------------------------------------------------------ #
    # public API
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

        # honor column merges using the parent API semantics
        if col_merge:
            try:
                stats = self.calculate_col_stats(df, enable_index=True)
                ordered = [col for col, _, _, _ in stats]
                for group in col_merge:
                    merge_order = [c for c in ordered if c in group]
                    if merge_order:
                        df = self.merging_columns(df, merge_order, prepended=False)
            except AttributeError:
                # parent lacks merging_columns; fall back to no merge
                pass

        n_rows = len(df)
        cols = list(df.columns)

        # trivial cases
        if n_rows == 0 or len(cols) == 0:
            out = df.copy()
            if len(cols) > 0:
                out = out.astype(object) if any(
                    not pd.api.types.is_dtype_equal(out[c].dtype, out[cols[0]].dtype) for c in cols
                ) else out
            return out, [list(cols) for _ in range(n_rows)]

        candidates = self._candidate_orders(df)

        # bound work: score each candidate once and pick the best
        best_order = None
        best_score = -1
        total_chars = None
        for order in candidates:
            strs = self._serialize_matrix(df, order)
            s = self._score(strs)
            if total_chars is None:
                total_chars = sum(len(x) for x in strs)
            if s > best_score:
                best_score = s
                best_order = order

        reordered = df.loc[:, best_order].copy()
        # object dtype for mixed-type safety
        try:
            reordered = reordered.astype(object)
        except Exception:
            pass

        orderings = [list(best_order) for _ in range(n_rows)]

        assert reordered.shape == df.shape
        assert list(reordered.columns) == best_order
        return reordered, orderings

# EVOLVE-BLOCK-END