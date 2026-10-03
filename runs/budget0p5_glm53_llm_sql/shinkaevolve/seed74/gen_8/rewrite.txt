# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict, Optional


class Evolved(Algorithm):
    """
    Prefix-cache optimized column reordering.
    Selects among a small set of cheap candidate column orderings by
    measuring the true serial character-Trie reuse (sum of adjacent LCPs of
    the sorted serialized rows), then applies the best ordering.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------
    # Serialization helpers (matches evaluator: "".join(fillna("").astype(str)))
    # ------------------------------------------------------------------
    @staticmethod
    def _serialize_matrix(values: np.ndarray, missing: np.ndarray, order: List[int]) -> List[str]:
        """Serialize rows given a column order (list of positional indices)."""
        rows = []
        n_rows = values.shape[0]
        for i in range(n_rows):
            parts = []
            for j in order:
                if missing[i, j]:
                    continue
                v = values[i, j]
                if v is None:
                    continue
                parts.append(v)
            rows.append("".join(parts))
        return rows

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Longest common prefix length using C-speed slice comparisons."""
        if a == b:
            return len(a)
        lo, hi = 0, min(len(a), len(b))
        # binary search on prefix equality
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _score(self, rows: List[str]) -> int:
        """Ideal Trie reuse: sum of adjacent LCPs over sorted strings."""
        if len(rows) <= 1:
            return 0
        s = sorted(rows)
        total = 0
        prev = s[0]
        for cur in s[1:]:
            total += self._lcp(prev, cur)
            prev = cur
        return total

    # ------------------------------------------------------------------
    # Candidate constructions
    # ------------------------------------------------------------------
    def _rank_columns(self, codes: List[np.ndarray], weights: List[np.ndarray],
                      n_cols: int, use_len: bool) -> List[int]:
        """Rank columns by sum over values of len(v)*c*(c-1) (or c*(c-1))."""
        scores = []
        for j in range(n_cols):
            code = codes[j]
            w = weights[j]
            counts = np.bincount(code, minlength=int(w.max(initial=0)) + 1 if w.size else 1)
            pairs = counts * (counts - 1)
            if use_len:
                s = float(np.dot(pairs[:len(w)] if len(pairs) >= len(w) else pairs, w[:len(pairs)] if len(w) >= len(pairs) else w))
            else:
                s = float(pairs.sum())
            scores.append((s, -j))
        scores.sort(reverse=True)
        return [j for j, _ in scores]

    def _conditional_order(self, codes: List[np.ndarray], weights: List[np.ndarray],
                           lens: List[np.ndarray], n_cols: int, n_rows: int,
                           col_stop: Optional[int], row_stop: Optional[int]) -> List[int]:
        """Greedy conditional partition tree producing a global column order.

        Repeatedly pick the column with the largest length-weighted pair
        repetition among the remaining ones, place it next, and restrict the
        statistics to rows sharing the most common value of that column.
        Bounded depth; falls back to plain frequency ranking for the tail.
        """
        order = []
        remaining = set(range(n_cols))
        row_idx = np.arange(n_rows)
        depth = 0
        max_depth = col_stop if col_stop else 12
        if row_stop is not None:
            max_depth = min(max_depth, row_stop)
        while remaining and row_idx.size > 0 and depth < max_depth:
            best_j, best_gain = None, 0.0
            for j in remaining:
                code = codes[j][row_idx]
                w = weights[j][row_idx]
                if code.size == 0:
                    continue
                counts = np.bincount(code, minlength=int(w.max(initial=0)) + 1 if w.size else 1)
                pairs = counts * (counts - 1)
                if pairs.size and w.size:
                    gain = float(np.dot(pairs[:w.size] if pairs.size >= w.size else np.pad(pairs, (0, w.size - pairs.size)),
                                        w if pairs.size >= w.size else w[:pairs.size]))
                else:
                    gain = 0.0
                if gain > best_gain:
                    best_gain, best_j = gain, j
            if best_j is None or best_gain <= 0:
                break
            order.append(best_j)
            remaining.discard(best_j)
            # restrict to rows having the most frequent value of best_j
            code = codes[best_j][row_idx]
            counts = np.bincount(code)
            if counts.size == 0:
                break
            common = int(counts.argmax())
            row_idx = row_idx[code == common]
            depth += 1
        # cheap deterministic tail
        if remaining:
            tail = self._rank_columns([codes[j][row_idx] if row_idx.size else codes[j] for j in remaining],
                                      [weights[j] for j in remaining],
                                      len(remaining), use_len=True)
            order.extend([list(remaining)[t] for t in tail])
        return order

    # ------------------------------------------------------------------
    # Public API
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
        df = df.copy()
        n_rows, n_cols = df.shape
        col_names = [str(c) for c in df.columns]

        if n_rows == 0 or n_cols == 0:
            return df, [[] for _ in range(n_rows)]

        # Precompute per-cell string representation (scoring only; stored
        # values in the output remain untouched).
        str_df = df.fillna("").astype(str)
        values = str_df.to_numpy(dtype=object)
        missing = ~np.asarray(df.notna().to_numpy(), dtype=bool)

        # Factorize each column once into integer codes + weight arrays.
        codes, weights = [], []
        for j in range(n_cols):
            col_vals = values[:, j]
            uniq, code = np.unique(col_vals, return_inverse=True)
            w = np.array([len(u) for u in uniq], dtype=float)
            codes.append(code)
            weights.append(w)

        # Candidate global column orders (positional indices).
        candidates = []
        c1 = self._rank_columns(codes, weights, n_cols, use_len=True)
        candidates.append(c1)
        c2 = self._rank_columns(codes, weights, n_cols, use_len=False)
        if c2 != c1:
            candidates.append(c2)
        c3 = self._conditional_order(codes, weights, None, n_cols, n_rows,
                                     col_stop, row_stop)
        if c3 != c1 and c3 != c2:
            candidates.append(c3)

        # Honor col_merge: keep each merge group contiguous at the position of
        # its best-ranked member (no cell values are altered).
        if col_merge:
            merged = []
            merged_flat = set()
            for grp in col_merge:
                merged.append([col_names.index(c) for c in grp if c in col_names])
                merged_flat.update(merged[-1])

            def apply_merge(order):
                pos = {j: k for k, j in enumerate(order)}
                out, used = [], set()
                for j in order:
                    if j in used:
                        continue
                    if j in merged_flat:
                        for grp in merged:
                            if j in grp:
                                for g in sorted(grp, key=lambda x: pos.get(x, 10**9)):
                                    if g not in used:
                                        out.append(g)
                                        used.add(g)
                                break
                    else:
                        out.append(j)
                        used.add(j)
                return out

            candidates = [apply_merge(c) for c in candidates]

        # Deduplicate candidate orders.
        uniq_candidates = []
        for c in candidates:
            if sorted(c) != list(range(n_cols)):
                continue  # safety: must be a permutation
            if c not in uniq_candidates:
                uniq_candidates.append(c)
        if not uniq_candidates:
            uniq_candidates = [list(range(n_cols))]

        # Score candidates with the true ideal Trie objective.
        best_order, best_score = uniq_candidates[0], -1
        total_chars = None
        for order in uniq_candidates:
            rows = self._serialize_matrix(values, missing, order)
            if total_chars is None:
                total_chars = sum(len(r) for r in rows)
            s = self._score(rows)
            if s > best_score:
                best_score, best_order = s, order

        # Build output: same rows, same values, columns permuted per row.
        perm_cols = [df.columns[j] for j in best_order]
        out_df = df.loc[:, perm_cols].copy()
        if out_df.dtypes.apply(lambda d: d == object).any():
            out_df = out_df.astype(object)

        order_names = [col_names[j] for j in best_order]
        column_orderings = [list(order_names) for _ in range(n_rows)]

        assert out_df.shape == df.shape
        return out_df, column_orderings

# EVOLVE-BLOCK-END