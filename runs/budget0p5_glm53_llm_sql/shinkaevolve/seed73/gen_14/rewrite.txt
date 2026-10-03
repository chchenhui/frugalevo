# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict, Optional


class Evolved(Algorithm):
    """
    Column-reordering algorithm optimized for serial character-Trie prefix
    caching. Builds a small set of cheap global column-order candidates,
    scores each with the true objective (sum of adjacent LCPs over sorted
    serialized rows), and returns the best. Preserves all data exactly and
    returns valid per-row column orderings.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------ #
    # Serialization helpers (scoring representation only; never mutates
    # stored cell values).
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cell_str(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and pd.isna(v):
            return ""
        try:
            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(v, bool):
            return str(v)
        return str(v)

    def _serialize_columns(self, df: pd.DataFrame, cols: List[str]):
        """Return {col: list of serialized cell strings}."""
        ser = {}
        for c in cols:
            ser[c] = [self._cell_str(v) for v in df[c].tolist()]
        return ser

    # ------------------------------------------------------------------ #
    # LCP scoring: ideal Trie reuse for a fixed multiset of strings equals
    # the sum of longest-common-prefix lengths of lexicographically
    # adjacent sorted strings.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lcp(a: str, b: str) -> int:
        n = min(len(a), len(b))
        if n == 0 or a[0] != b[0]:
            return 0
        # Binary search on prefix-slice equality; slices compare in C.
        lo, hi = 1, n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        # lo is at least 1 because a[:1] == b[:1]
        return lo

    def _row_strings(self, ser: Dict[str, List[str]], order: List[str], n_rows: int):
        idx = {c: i for i, c in enumerate(order)}
        cols_arrays = [ser[c] for c in order]
        rows = ["".join(arr[i] for arr in cols_arrays) for i in range(n_rows)]
        return rows

    def _lcp_score(self, ser: Dict[str, List[str]], order: List[str], n_rows: int) -> int:
        rows = self._row_strings(ser, order, n_rows)
        rows.sort()
        total = 0
        prev = rows[0]
        for cur in rows[1:]:
            if cur == prev:
                total += len(cur)
            else:
                total += self._lcp(prev, cur)
            prev = cur
        return total

    # ------------------------------------------------------------------ #
    # Candidate column-order constructions (cheap, bounded).
    # ------------------------------------------------------------------ #
    def _candidate_orders(self, ser: Dict[str, List[str]], cols: List[str]) -> List[List[str]]:
        from collections import Counter

        cands: List[List[str]] = []

        # Candidate A: sum over serialized values v of len(v) * count * (count-1)
        score_a = {}
        for c in cols:
            cnt = Counter(ser[c])
            score_a[c] = sum(len(v) * k * (k - 1) for v, k in cnt.items())
        cands.append(sorted(cols, key=lambda c: (-score_a.get(c, 0), str(c))))

        # Candidate B: length-weighted pair repetition len(v) * (count-1)
        score_b = {}
        for c in cols:
            cnt = Counter(ser[c])
            score_b[c] = sum(len(v) * (k - 1) for v, k in cnt.items())
        cands.append(sorted(cols, key=lambda c: (-score_b.get(c, 0), str(c))))

        # Candidate C: duplicate-pair count ranking with tie-break on length
        score_c = {}
        for c in cols:
            cnt = Counter(ser[c])
            score_c[c] = sum((k - 1) for v, k in cnt.items() if k > 1)
        cands.append(sorted(cols, key=lambda c: (-score_c.get(c, 0), str(c))))

        # Candidate D: original column order (reliable fallback)
        cands.append(list(cols))

        # Deduplicate candidates
        unique: List[List[str]] = []
        for cand in cands:
            if cand not in unique:
                unique.append(cand)
        return unique

    # ------------------------------------------------------------------ #
    # col_merge handling: keep merged groups contiguous without altering
    # any stored values or the DataFrame shape.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_col_merge(order: List[str], col_merge: List[List[str]]) -> List[str]:
        if not col_merge:
            return order
        order = list(order)
        cols_set = set(order)
        for group in col_merge:
            members = [c for c in group if c in cols_set]
            if len(members) <= 1:
                continue
            member_set = set(members)
            # find position of first member in current order
            first_pos = None
            for i, c in enumerate(order):
                if c in member_set:
                    first_pos = i
                    break
            if first_pos is None:
                continue
            others = [c for c in order if c not in member_set]
            new_order = others[:first_pos] + members + others[first_pos:]
            if sorted(new_order) == sorted(order):
                order = new_order
        return order

    # ------------------------------------------------------------------ #
    # Public API
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
        df = df.copy()
        n_rows, n_cols = df.shape

        if n_rows == 0:
            return df.copy(), []
        if n_cols == 0:
            return df.copy(), [[] for _ in range(n_rows)]

        cols = list(df.columns)
        if len(set(cols)) != len(cols):
            # Duplicate column names: cannot permute safely per column name.
            return df.copy(), [list(cols) for _ in range(n_rows)]

        # Serialize cells once (scoring representation only).
        ser = self._serialize_columns(df, cols)

        # Build candidate column orders.
        candidates = self._candidate_orders(ser, cols)

        # Score candidates with the true serial-Trie objective, with a
        # bounded-cost guard for very large tables.
        max_score_rows = 300_000
        best_order = candidates[0]
        if n_rows <= max_score_rows:
            best_score = -1
            for cand in candidates:
                try:
                    s = self._lcp_score(ser, cand, n_rows)
                except Exception:
                    s = -1
                if s > best_score:
                    best_score = s
                    best_order = cand
        else:
            best_order = candidates[0]

        # Honor column-merge adjacency constraints.
        order = self._apply_col_merge(best_order, col_merge)

        # Build output DataFrame with the chosen column order.
        reordered_df = df.loc[:, order].copy()
        # Preserve mixed types: do not force casting. Reindexing with .loc
        # keeps per-column dtypes as-is.

        # Valid per-row column orderings consistent with returned data.
        row_order = list(order)
        column_orderings = [list(row_order) for _ in range(n_rows)]

        assert reordered_df.shape == df.shape
        assert len(column_orderings) == n_rows
        for i in range(n_rows):
            assert len(column_orderings[i]) == n_cols
            assert set(column_orderings[i]) == set(cols)

        return reordered_df, column_orderings

# EVOLVE-BLOCK-END