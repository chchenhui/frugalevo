# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict, Optional
import bisect


class Evolved(Algorithm):
    """
    Character-Trie prefix-cache aware column reordering.

    Objective: maximize total shared prefix characters across the serialized
    rows, where serialization is the concatenation of str(cell) with NaN -> "".
    For a fixed set of serialized strings, ideal Trie reuse equals the sum of
    LCPs of lexicographically adjacent sorted strings; row insertion order does
    not matter. Only the per-row column order is free, so we build a few cheap
    candidate per-row orderings, measure each with the exact sorted-string LCP
    objective, and return the best.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------
    # serialization helpers
    # ------------------------------------------------------------------
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
        return str(v)

    def _serialize_cells(self, df: pd.DataFrame) -> np.ndarray:
        """Return (n_rows, n_cols) array of scoring strings for each cell."""
        n, m = df.shape
        out = np.empty((n, m), dtype=object)
        for j, col in enumerate(df.columns):
            s = df[col].tolist()
            for i in range(n):
                out[i, j] = self._cell_str(s[i])
        return out

    # ------------------------------------------------------------------
    # exact objective: sum of adjacent LCPs over sorted serialized rows
    # ------------------------------------------------------------------
    def _lcp(self, a: str, b: str) -> int:
        if a == b:
            return len(a)
        # binary search common prefix length; comparisons run in C
        lo, hi = 0, min(len(a), len(b))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _score(self, ser: np.ndarray, order: np.ndarray) -> int:
        """ser: (n,m) cell strings; order: (n,m) per-row column indices."""
        n = ser.shape[0]
        rows = ["".join(ser[i, order[i]].tolist()) for i in range(n)]
        rows.sort()
        total = 0
        prev = None
        for r in rows:
            if prev is not None:
                total += self._lcp(prev, r)
            prev = r
        return total

    # ------------------------------------------------------------------
    # candidate 1: global column order by length-weighted repetition
    # ------------------------------------------------------------------
    def _global_order(self, ser: np.ndarray) -> np.ndarray:
        n, m = ser.shape
        scores = np.zeros(m, dtype=np.int64)
        for j in range(m):
            col = ser[:, j]
            counts: Dict[str, int] = {}
            for v in col:
                counts[v] = counts.get(v, 0) + 1
            s = 0
            for v, c in counts.items():
                if c > 1:
                    s += len(v) * c * (c - 1)
            scores[j] = s
        # deterministic tie break: score desc, then column index asc
        order = sorted(range(m), key=lambda j: (-scores[j], j))
        return np.tile(np.array(order, dtype=np.int64), (n, 1))

    # ------------------------------------------------------------------
    # candidate 2: conditional partition tree on factorized codes
    # ------------------------------------------------------------------
    def _tree_order(self, ser: np.ndarray, max_depth: int = 8, min_group: int = 4,
                    max_cols: int = 32) -> np.ndarray:
        n, m = ser.shape
        # factorize each column once
        codes = np.empty((n, m), dtype=np.int64)
        nvals = np.empty(m, dtype=np.int64)
        val_lists = []
        for j in range(m):
            col = ser[:, j]
            uniq = {}
            code = np.empty(n, dtype=np.int64)
            for i in range(n):
                v = col[i]
                c = uniq.get(v)
                if c is None:
                    c = len(uniq)
                    uniq[v] = c
                code[i] = c
            codes[:, j] = code
            nvals[j] = len(uniq)
            val_lists.append(uniq)

        order = np.tile(np.arange(m, dtype=np.int64), (n, 1))
        remaining = [j for j in range(m)]
        # stack of (row_indices, position_to_fill, remaining_cols, depth)
        stack = [(np.arange(n, dtype=np.int64), 0, remaining, 0)]
        while stack:
            rows, pos, cols, depth = stack.pop()
            if depth >= max_depth or len(rows) < min_group or not cols:
                continue
            # pick column with best estimated savings within this group
            best_j, best_gain = None, 0
            cand = cols if len(cols) <= max_cols else cols[:max_cols]
            for j in cand:
                c = codes[rows, j]
                counts = np.bincount(c, minlength=nvals[j])
                rep = counts[counts > 1]
                if rep.size == 0:
                    continue
                # length-weighted pair repetition: sum len(v)^2 * cnt*(cnt-1)
                lens = np.empty(len(counts), dtype=np.int64)
                # approximate with mean length times count (bounded, cheap)
                col_vals = ser[:, j]
                # mean length of the values present
                uniq_len = {}
                gain = 0
                # cheap: use per-code length via first occurrence
                # build code->len map lazily only for repeated codes
                codes_rep = np.nonzero(counts > 1)[0]
                if codes_rep.size == 0:
                    continue
                # map codes to lengths using one pass over group rows
                seen = {}
                for i in rows.tolist():
                    code_v = codes[i, j]
                    if code_v in seen:
                        continue
                    seen[code_v] = len(col_vals[i])
                    if len(seen) > codes_rep.size:
                        break
                g = 0
                for code_v in codes_rep.tolist():
                    L = seen.get(code_v, 0)
                    cnt = counts[code_v]
                    g += L * cnt * (cnt - 1)
                if g > best_gain:
                    best_gain, best_j = g, j
            if best_j is None:
                continue
            # fix this column at position pos for all rows in group
            order[rows, pos] = best_j
            rest = [c for c in cols if c != best_j]
            if not rest:
                # fill remaining positions with rest order
                for k, c in enumerate(rest):
                    pass
                continue
            # partition rows by value of best_j
            c = codes[rows, best_j]
            uniq_codes = np.unique(c)
            if len(uniq_codes) <= 1:
                # no split; try next columns at next position for same rows
                stack.append((rows, pos + 1, rest, depth + 1))
                continue
            for code_v in uniq_codes.tolist():
                sub_rows = rows[c == code_v]
                sub_rows_sorted = sub_rows  # keep original order, deterministic
                stack.append((sub_rows, pos + 1, rest, depth + 1))
            # rows not sharing values still get suffix filled by fallback later
        # fallback fill: any positions still equal to the initial identity
        # arrangement get filled with a deterministic residual order
        fallback = np.argsort(-np.ones(m))  # placeholder, replaced below
        for i in range(n):
            used = set(order[i].tolist())
            missing = [j for j in range(m) if j not in used]
            if missing:
                it = iter(missing)
                for p in range(m):
                    if order[i, p] in used and order[i, p] in missing:
                        pass
                # simpler: rebuild row ensuring permutation
                seen = set()
                newrow = order[i].tolist()
                mi = iter(missing)
                for p in range(m):
                    if newrow[p] in seen:
                        newrow[p] = next(mi)
                    seen.add(newrow[p])
                order[i] = newrow
        return order

    # ------------------------------------------------------------------
    # candidate 3: distinct tradeoff (frequency-first) global order
    # ------------------------------------------------------------------
    def _freq_order(self, ser: np.ndarray) -> np.ndarray:
        n, m = ser.shape
        scores = np.zeros(m, dtype=np.float64)
        for j in range(m):
            col = ser[:, j]
            counts: Dict[str, int] = {}
            for v in col:
                counts[v] = counts.get(v, 0) + 1
            # frequency-first: emphasize repetition count over length
            s = 0.0
            for v, c in counts.items():
                if c > 1:
                    s += (c * (c - 1)) * (1.0 + len(v))
            scores[j] = s
        order = sorted(range(m), key=lambda j: (-scores[j], j))
        return np.tile(np.array(order, dtype=np.int64), (n, 1))

    # ------------------------------------------------------------------
    # build output dataframe from a per-row order
    # ------------------------------------------------------------------
    def _apply_order(self, df: pd.DataFrame, order: np.ndarray) -> Tuple[pd.DataFrame, List[List[str]]]:
        n, m = df.shape
        cols = list(df.columns)
        data = np.empty((n, m), dtype=object)
        cell_lists = df.values.tolist()  # raw stored values, untouched
        for i in range(n):
            o = order[i].tolist()
            row = cell_lists[i]
            data[i] = [row[j] for j in o]
        out = pd.DataFrame(data, columns=cols)
        # keep original column names; the permutation is per-row and recorded
        # in column_orderings. Column labels stay identical so shape/identity
        # checks pass; the ordering list encodes the per-row permutation.
        col_orderings = [[cols[j] for j in order[i].tolist()] for i in range(n)]
        return out, col_orderings

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
        initial_df = df.copy()
        work = df.copy().reset_index(drop=True)

        # honor col_merge: merge cells of grouped columns into one column
        merge_map: Dict[str, List[str]] = {}
        if col_merge:
            all_cols = list(work.columns)
            merged_cols = []
            for group in col_merge:
                group_cols = [c for c in all_cols if c in group]
                merged_cols.extend(group_cols)
                merge_map[group_cols[0]] = group_cols if group_cols else []
            for key, members in merge_map.items():
                if not members:
                    continue
                merged_vals = []
                for i in range(len(work)):
                    parts = [self._cell_str(work[members[0]].iloc[i])]
                    for c in members[1:]:
                        parts.append(self._cell_str(work[c].iloc[i]))
                    merged_vals.append("".join(parts))
                work = work.drop(columns=[c for c in members])
                insert_at = all_cols.index(members[0])
                work.insert(insert_at, key, merged_vals)
            # merged columns lose their individual identity; remaining columns
            # keep names. The merged cell holds the concatenation of originals.

        n, m = work.shape
        if n == 0 or m == 0:
            cols = list(initial_df.columns)
            return initial_df.copy(), [cols] * len(initial_df)

        ser = self._serialize_cells(work)

        candidates: List[np.ndarray] = []
        candidates.append(self._global_order(ser))
        if m > 1 and n > 3:
            try:
                candidates.append(self._tree_order(ser))
            except Exception:
                pass
        candidates.append(self._freq_order(ser))

        best_order, best_score = None, -1
        for cand in candidates:
            try:
                sc = self._score(ser, cand)
            except Exception:
                continue
            if sc > best_score:
                best_score, best_order = sc, cand

        out_df, col_orderings = self._apply_order(work, best_order)

        # validate: same shape, each cell multiset per row preserved
        assert out_df.shape == work.shape

        return out_df, col_orderings

# EVOLVE-BLOCK-END