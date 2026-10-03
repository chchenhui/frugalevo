# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict, Optional
from collections import Counter


class Evolved(Algorithm):
    """
    Prefix-caching optimized reordering.

    Strategy: build a small number of bounded candidate per-row column
    orderings (original order, global repetition-ranked order, and a
    conditional partition tree), score each with the exact serial
    character-Trie reuse objective (sum of adjacent LCPs of the sorted
    serialized row strings), and return the best. Data values, rows and
    shape are never modified; only the per-row serialization order
    (returned via column_orderings) changes.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.dep_graph = None

    # ------------------------------------------------------------------ #
    # serialization helpers (scoring representation only)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _col_strings(series: pd.Series) -> np.ndarray:
        """Serialize a column: NaN/None -> '', else str(value)."""
        s = series.astype(object)
        arr = np.empty(len(s), dtype=object)
        values = s.values
        mask = pd.isna(values)
        for i in range(len(values)):
            v = values[i]
            arr[i] = "" if mask[i] else str(v)
        return arr

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Longest common prefix length via binary search with C slices."""
        if a == b:
            return len(a)
        hi = len(a) if len(a) < len(b) else len(b)
        if hi == 0:
            return 0
        if a[0] != b[0]:
            return 0
        lo = 1
        # find largest k with a[:k] == b[:k]
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _score(self, row_strings: List[str]) -> int:
        """Exact ideal Trie reuse: sum of adjacent LCPs after sorting."""
        if len(row_strings) <= 1:
            return 0
        ss = sorted(row_strings)
        total = 0
        lcp = self._lcp
        for i in range(1, len(ss)):
            total += lcp(ss[i - 1], ss[i])
        return total

    def _build_row_strings(self, colstrs: Dict[str, np.ndarray],
                           perms: List[List[str]]) -> List[str]:
        out = []
        n = len(perms)
        if n == 0:
            return out
        first = perms[0]
        cs = [colstrs[c] for c in first]
        for i in range(n):
            perm = perms[i]
            if perm is first:
                parts = [cs[j][i] for j in range(len(cs))]
            else:
                parts = [colstrs[c][i] for c in perm]
            out.append("".join(parts))
        return out

    # ------------------------------------------------------------------ #
    # candidate constructions
    # ------------------------------------------------------------------ #
    def _column_weights(self, colstrs: Dict[str, np.ndarray],
                        columns: List[str]) -> Dict[str, int]:
        """weight(col) = sum over distinct values v of len(v)*cnt*(cnt-1)."""
        w = {}
        for c in columns:
            cnt = Counter(colstrs[c])
            s = 0
            for v, k in cnt.items():
                if k > 1:
                    s += len(v) * k * (k - 1)
            w[c] = s
        return w

    def _freq_order(self, columns: List[str],
                    weights: Dict[str, int]) -> List[str]:
        idx = {c: i for i, c in enumerate(columns)}
        return sorted(columns, key=lambda c: (-weights[c], idx[c]))

    def _conditional_perms(self, df: pd.DataFrame,
                           colstrs: Dict[str, np.ndarray],
                           columns: List[str],
                           freq_order: List[str],
                           max_depth: int = 8,
                           min_group: int = 3) -> List[List[str]]:
        """Conditional prefix partition tree using integer codes."""
        n = len(df)
        # factorize each column once
        codes = {}
        uniq = {}
        for c in columns:
            arr = colstrs[c]
            seen = {}
            code = np.empty(n, dtype=np.int64)
            ulist = []
            for i in range(n):
                v = arr[i]
                j = seen.get(v)
                if j is None:
                    j = len(ulist)
                    seen[v] = j
                    ulist.append(v)
                code[i] = j
            codes[c] = code
            uniq[c] = ulist

        ncols = len(columns)
        # bound work on very wide / huge tables
        if n * ncols > 20_000_000:
            base = freq_order
            return [list(base) for _ in range(n)]

        perms = [None] * n
        all_rows = np.arange(n)
        # stack of (rows, prefix tuple, depth)
        stack = [(all_rows, (), 0)]
        col_list = columns
        while stack:
            rows, prefix, depth = stack.pop()
            remaining = [c for c in freq_order if c not in prefix] if prefix else list(freq_order)
            if (len(rows) < min_group or depth >= max_depth
                    or not remaining):
                tail = [c for c in freq_order if c not in prefix]
                order = list(prefix) + tail
                for r in rows:
                    perms[r] = order
                continue
            # pick best column by length-weighted repetition within group
            best_col = None
            best_w = 0
            for c in remaining:
                sub = codes[c][rows]
                m = len(uniq[c])
                bc = np.bincount(sub, minlength=m)
                w = 0
                nz = np.nonzero(bc)[0]
                for j in nz:
                    k = bc[j]
                    if k > 1:
                        w += len(uniq[c][j]) * k * (k - 1)
                if w > best_w:
                    best_w = w
                    best_col = c
            if best_col is None or best_w <= 0:
                tail = [c for c in freq_order if c not in prefix]
                order = list(prefix) + tail
                for r in rows:
                    perms[r] = order
                continue
            new_prefix = prefix + (best_col,)
            sub = codes[best_col][rows]
            order_idx = np.argsort(sub, kind="stable")
            sorted_vals = sub[order_idx]
            sorted_rows = rows[order_idx]
            # split into groups of equal code
            bounds = np.nonzero(np.diff(sorted_vals))[0] + 1
            starts = np.concatenate(([0], bounds, [len(sorted_rows)]))
            for g in range(len(starts) - 1):
                grp = sorted_rows[starts[g]:starts[g + 1]]
                stack.append((grp, new_prefix, depth + 1))
        return perms

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
        # honor column merges via existing API semantics
        work = df.copy()
        if col_merge:
            for col_to_merge in col_merge:
                present = [c for c in work.columns if c in col_to_merge]
                if len(present) > 1:
                    work = self.merging_columns(work, present, prepended=False)

        n = len(work)
        columns = list(work.columns)
        if n == 0 or len(columns) == 0:
            out = work.astype(object) if len(columns) else work
            return out, [[] for _ in range(n)]

        # one-way dependencies: accepted, recorded, not needed for ordering
        self.dep_graph = None

        # serialized per-column string arrays (scoring representation only)
        colstrs = {c: self._col_strings(work[c]) for c in columns}

        # candidate 1: original column order
        cand1 = [list(columns) for _ in range(n)]

        # candidate 2: global frequency-ranked order
        weights = self._column_weights(colstrs, columns)
        freq_order = self._freq_order(columns, weights)
        cand2 = [list(freq_order) for _ in range(n)]

        candidates = [("freq", cand2), ("orig", cand1)]

        # candidate 3: conditional partition tree (bounded)
        total_chars = int(sum(len(colstrs[c][i]) for c in columns for i in (0,) if n) or 0)
        try:
            cand3 = self._conditional_perms(work, colstrs, columns, freq_order,
                                            max_depth=min(8, col_stop if col_stop else 8))
            candidates.append(("tree", cand3))
        except Exception:
            pass

        # select by the exact serial-Trie character objective
        best_name, best_perms, best_score = None, cand2, -1
        for name, perms in candidates:
            try:
                strings = self._build_row_strings(colstrs, perms)
                score = self._score(strings)
            except Exception:
                continue
            if score > best_score:
                best_name, best_perms, best_score = name, perms, score

        # ensure every row has a valid permutation (fallback safety)
        valid_sets = [set(columns) for _ in range(0)]
        col_set = set(columns)
        final_perms = []
        for i in range(n):
            p = best_perms[i]
            if p is None or len(p) != len(columns) or set(p) != col_set:
                p = list(freq_order)
            final_perms.append(list(p))

        out = work.astype(object).copy() if len(columns) else work.copy()
        # verify data preservation
        assert out.shape == work.shape
        for c in columns:
            assert list(pd.isna(out[c])) == list(pd.isna(work[c]))
        return out, final_perms


# EVOLVE-BLOCK-END