# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List


def _cell_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    if v is pd.NA:
        return ""
    return str(v)


def _lcp(a: str, b: str) -> int:
    """Longest common prefix length; C-speed via slice comparisons."""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _trie_score(strings) -> int:
    """Ideal trie reuse = sum of adjacent LCPs after sorting (order-independent)."""
    s = sorted(set(strings)) if False else sorted(strings)
    total = 0
    prev = None
    for cur in s:
        if prev is not None:
            total += _lcp(prev, cur)
        prev = cur
    return total


class Evolved(Algorithm):
    def __init__(self, df: pd.DataFrame = None):
        self.df = df

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
        num_rows, num_cols = df.shape
        cols = list(df.columns)

        # Keep col_merge groups adjacent: place merged columns contiguously at
        # the position of the first member, in their given relative order.
        merged_flat = [c for grp in col_merge for c in grp if c in cols]
        order0 = []
        for c in cols:
            if c in merged_flat:
                if c == next((m for m in merged_flat if m in cols and m not in order0), None):
                    for grp in col_merge:
                        if c in grp:
                            order0.extend([g for g in grp if g in cols and g not in order0])
                            break
            elif c not in order0:
                order0.append(c)
        # ensure any leftover
        for c in cols:
            if c not in order0:
                order0.append(c)

        # Serialize cells once
        cell_str = {}
        col_vals = {}
        for c in cols:
            vals = [_cell_str(v) for v in df[c].tolist()]
            col_vals[c] = vals
            cell_str[c] = vals

        def col_score(c, weighted=False):
            counter = {}
            for v in col_vals[c]:
                counter[v] = counter.get(v, 0) + 1
            if weighted:
                return sum(len(v) * n * (n - 1) for v, n in counter.items())
            return sum(len(v) * (n - 1) for v, n in counter.items() if n > 1)

        # Candidate 1: original order (with merge adjacency)
        cand1 = order0
        # Candidate 2: global frequency-ranked (count-1 savings)
        cand2 = sorted(cols, key=lambda c: (-col_score(c), cols.index(c)))
        # respect merge adjacency inside cand2 too
        cand2 = self._apply_merge_order(cand2, col_merge, cols)
        # Candidate 3: length-weighted count*(count-1)
        cand3 = sorted(cols, key=lambda c: (-col_score(c, weighted=True), cols.index(c)))
        cand3 = self._apply_merge_order(cand3, col_merge, cols)

        def serialize_with(order):
            return ["".join(col_vals[c][i] for c in order) for i in range(num_rows)]

        best_order, best_score = None, -1
        for cand in (cand1, cand2, cand3):
            s = serialize_with(cand)
            score = _trie_score(s)
            if score > best_score:
                best_score, best_order = score, cand

        final_order = best_order
        strings = serialize_with(final_order)
        column_orderings = [list(final_order) for _ in range(num_rows)]

        # Build output dataframe with per-row permutation (same order for all
        # rows here, but expressed per-row for API compliance).
        out = pd.DataFrame(index=df.index)
        for c in final_order:
            out[c] = df[c].values
        if out.shape[1] and any(str(t) == "object" for t in out.dtypes.astype(str)):
            pass
        # preserve mixed types
        out = out.astype(object) if any(
            len(set(type(v).__name__ for v in out[c].tolist())) > 1 for c in out.columns
        ) else out

        # Order rows: sorted serialized strings maximize adjacent LCP sharing
        sort_idx = sorted(range(num_rows), key=lambda i: strings[i])
        out = out.iloc[sort_idx].reset_index(drop=True)
        column_orderings = [column_orderings[i] for i in sort_idx]

        assert out.shape == df.shape
        return out, column_orderings

    def _apply_merge_order(self, order, col_merge, cols):
        """Force each merge group to be contiguous, keeping given relative order."""
        groups = {c: i for i, grp in enumerate(col_merge) for c in grp if c in cols}
        if not groups:
            return order
        placed = []
        seen_groups = set()
        for c in order:
            g = groups.get(c)
            if g is None:
                if c not in placed:
                    placed.append(c)
            else:
                if g in seen_groups:
                    continue
                seen_groups.add(g)
                for gc in col_merge[g]:
                    if gc in cols and gc not in placed:
                        placed.append(gc)
        for c in cols:
            if c not in placed:
                placed.append(c)
        return placed

# EVOLVE-BLOCK-END