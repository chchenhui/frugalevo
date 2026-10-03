# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import traceback


class Evolved(Algorithm):
    """
    Row-wise column reordering optimized for character-level prefix reuse
    (LLM prompt prefix caching), preserving all data and row identity.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------ #
    # Serialization / scoring helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ser_cell(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        try:
            if v is pd.NA or v is pd.NaT:
                return ""
        except Exception:
            pass
        return str(v)

    def _serialize_rows(self, arr: np.ndarray, orders: List[List[int]]) -> List[str]:
        n, m = arr.shape
        out = []
        for i in range(n):
            perm = orders[i]
            parts = []
            for j in perm:
                parts.append(self._ser_cell(arr[i, j]))
            out.append("".join(parts))
        return out

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        if a == b:
            return len(a)
        hi = min(len(a), len(b))
        if hi == 0:
            return 0
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _score_serial(self, strs: List[str]) -> int:
        """Ideal Trie reuse = sum of adjacent LCP over sorted strings."""
        if len(strs) <= 1:
            return 0
        ss = sorted(strs)
        total = 0
        prev = ss[0]
        for cur in ss[1:]:
            total += self._lcp(prev, cur)
            prev = cur
        return total

    # ------------------------------------------------------------------ #
    # Candidate constructions
    # ------------------------------------------------------------------ #

    def _global_freq_order(self, codes: Dict[int, np.ndarray], n_rows: int,
                           ncols: int, serlens) -> List[int]:
        """Rank columns by sum over values v of len(v)*count*(count-1)."""
        scores = []
        for c in range(ncols):
            col_codes = codes[c]
            # per-code representative serialized length
            uniq, counts = np.unique(col_codes, return_counts=True)
            s = 0.0
            for u, cnt in zip(uniq.tolist(), counts.tolist()):
                if cnt > 1:
                    L = serlens[c].get(u, 1)
                    s += L * cnt * (cnt - 1)
            scores.append((s, -c))
        scores.sort(reverse=True)
        return [c for c, _ in scores]

    def _conditional_order(self, codes: Dict[int, np.ndarray], n_rows: int,
                           ncols: int, serlens, base_order: List[int],
                           blocks: List[List[int]], max_depth: int = 8,
                           min_group: int = 4) -> List[List[int]]:
        """Recursive conditional prefix partition: per-group leading field by
        length-weighted pair repetition; per-row suffix orders."""
        orders = [None] * n_rows
        remaining_sets = [frozenset(range(ncols)) for _ in range(0)]
        all_cols = frozenset(range(ncols))

        def rec(rows: np.ndarray, avail: frozenset, depth: int):
            if len(rows) == 0:
                return
            if len(rows) == 1 or depth >= max_depth or len(avail) <= 2:
                fixed = [c for c in base_order if c in avail]
                for i in rows:
                    if orders[i] is None:
                        orders[i] = fixed + [c for c in base_order if c not in avail]
                        orders[i] = [c for c in orders[i]]
                return
            # choose best column by sum len(v)*cnt*(cnt-1)
            best_col, best_score = None, -1.0
            for c in avail:
                col_codes = codes[c][rows]
                uniq, counts = np.unique(col_codes, return_counts=True)
                s = 0.0
                cl = serlens[c]
                for u, cnt in zip(uniq.tolist(), counts.tolist()):
                    if cnt > 1:
                        s += cl.get(u, 1) * cnt * (cnt - 1)
                if s > best_score:
                    best_score, best_col = s, cnt if False else s, c
                    best_col = c
            if best_col is None or best_score <= 0:
                fixed = [c for c in base_order if c in avail]
                for i in rows:
                    if orders[i] is None:
                        orders[i] = [c for c in base_order if c in avail] + \
                                    [c for c in base_order if c not in avail]
                return
            # partition on best_col
            col_codes = codes[best_col][rows]
            uniq = np.unique(col_codes)
            next_avail = avail - {best_col}
            head = best_col
            for u in uniq.tolist():
                sub_rows = rows[col_codes == u]
                if len(sub_rows) < min_group or len(next_avail) == 0:
                    fixed_suffix = [c for c in base_order if c in next_avail]
                    for i in sub_rows:
                        if orders[i] is None:
                            orders[i] = [head] + fixed_suffix + \
                                        [c for c in base_order if c not in avail and c != head]
                    continue
                rec(sub_rows, next_avail, depth + 1)
                # prefix the settled head column
                for i in sub_rows:
                    if orders[i] is not None and best_col not in orders[i]:
                        orders[i] = [head] + orders[i]

        rows_all = np.arange(n_rows)
        rec(rows_all, all_cols, 0)
        # fill any leftovers
        for i in range(n_rows):
            if orders[i] is None:
                orders[i] = list(base_order)
        return orders

    def _apply_blocks(self, orders: List[List[int]], blocks_by_col: Dict[int, List[int]]) -> List[List[int]]:
        """Expand block-aware orders: merged columns stay contiguous in fixed order."""
        out = []
        for perm in orders:
            new = []
            used = set()
            for c in perm:
                if c in used:
                    continue
                blk = blocks_by_col.get(c)
                if blk is not None:
                    for b in blk:
                        if b not in used:
                            new.append(b)
                            used.add(b)
                else:
                    new.append(c)
                    used.add(c)
            out.append(new)
        return out

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
        try:
            return self._reorder_impl(df, col_merge)
        except Exception:
            try:
                return self._reorder_impl(df, [])
            except Exception:
                # Last-resort: identity ordering (always valid, preserves data).
                arr = df.to_numpy(dtype=object)
                cols = list(df.columns)
                out = pd.DataFrame(arr.copy(), columns=cols)
                return out, [list(cols) for _ in range(len(df))]

    def _reorder_impl(self, df: pd.DataFrame, col_merge: List[List[str]]) -> Tuple[pd.DataFrame, List[List[str]]]:
        df = df.copy()
        original_cols = list(df.columns)
        n_rows, ncols = df.shape
        if n_rows == 0 or ncols == 0:
            return df.copy(), [list(original_cols) for _ in range(n_rows)]

        # Column name -> index (handle duplicate names via positional lookup)
        arr = df.to_numpy(dtype=object)

        # Blocks for col_merge: groups of columns kept contiguous in given order.
        merged_names = set()
        blocks: List[List[int]] = []
        name_to_first_idx: Dict[str, int] = {}
        for c in original_cols:
            if c not in name_to_first_idx:
                name_to_first_idx[c] = original_cols.index(c) if False else None
        # positional index builder
        first_pos: Dict[str, int] = {}
        for i, c in enumerate(original_cols):
            if c not in first_pos:
                first_pos[c] = i

        for group in col_merge or []:
            idxs = []
            for nm in group:
                if nm in first_pos:
                    idxs.append(first_pos[nm])
                    merged_names.add(first_pos[nm])
            if len(idxs) >= 2:
                blocks.append(idxs)

        blocks_by_col: Dict[int, List[int]] = {}
        for blk in blocks:
            for b in blk:
                blocks_by_col[b] = blk

        # "Free" units for ordering: singleton columns + merge blocks.
        block_members = set()
        for blk in blocks:
            block_members.update(blk)
        units: List[List[int]] = []
        for i in range(ncols):
            if i in block_members:
                blk = blocks_by_col[i]
                if blk and blk[0] == i:
                    units.append(blk)
            else:
                units.append([i])
        unit_of_col: Dict[int, int] = {}
        for ui, blk in enumerate(units):
            for c in blk:
                unit_of_col[c] = ui

        # Factorize each column into integer codes once; record serialized lengths.
        codes: Dict[int, np.ndarray] = {}
        serlens: Dict[int, Dict[int, int]] = {}
        for c in range(ncols):
            col_vals = arr[:, c]
            key_map: Dict[object, int] = {}
            code_arr = np.empty(n_rows, dtype=np.int64)
            lens: Dict[int, int] = {}
            for i, v in enumerate(col_vals):
                if v not in key_map:
                    key_map[v] = len(key_map)
                    lens[key_map[v]] = len(self._ser_cell(v))
                code_arr[i] = key_map[v]
            codes[c] = code_arr
            serlens[c] = lens

        # Bound work on wide tables
        max_units = 64
        order_units = units if len(units) <= max_units else units[:max_units]

        # ---- Candidate 1: global frequency-ranked unit order ----
        unit_scores = []
        for ui, blk in enumerate(order_units):
            s = 0.0
            for c in blk:
                uniq, counts = np.unique(codes[c], return_counts=True)
                cl = serlens[c]
                for u, cnt in zip(uniq.tolist(), counts.tolist()):
                    if cnt > 1:
                        s += cl.get(u, 1) * cnt * (cnt - 1)
            unit_scores.append((s, -ui, ui))
        unit_scores.sort(reverse=True)
        freq_unit_order = [ui for _, _, ui in unit_scores]

        def units_to_cols(unit_order):
            out = []
            for ui in unit_order:
                if ui < len(units):
                    out.extend(units[ui])
            # keep any truncated columns at the end (wide tables)
            placed = set(out)
            for c in range(ncols):
                if c not in placed:
                    out.append(c)
            return out

        freq_col_order = units_to_cols(freq_unit_order)

        # ---- Candidate 2: identity order ----
        ident_col_order = list(range(ncols))

        candidates: List[List[List[int]]] = []
        names_order: List[str] = []

        candidates.append([[c for c in freq_col_order] for _ in range(n_rows)])
        candidates.append([[c for c in ident_col_order] for _ in range(n_rows)])

        # ---- Candidate 3: conditional partition (bounded) ----
        try:
            # operate at column level but respect blocks: choose order over units
            # simple per-column version with block expansion afterwards
            base = freq_col_order
            unit_codes: Dict[int, np.ndarray] = {}
            unit_serlens: Dict[int, Dict[int, int]] = {}
            for ui, blk in enumerate(order_units):
                # use first column of block as partition key
                c0 = blk[0]
                unit_codes[ui] = codes[c0]
                unit_serlens[ui] = serlens[c0]
            nu = len(order_units)
            cond_orders = self._conditional_orders_units(
                unit_codes, unit_serlens, n_rows, nu, freq_unit_order
            )
            cond_col_orders = []
            for i in range(n_rows):
                perm = []
                for ui in cond_orders[i]:
                    perm.extend(units[ui])
                placed = set(perm)
                for c in range(ncols):
                    if c not in placed:
                        perm.append(c)
                cond_col_orders.append(perm)
            candidates.append(cond_col_orders)
        except Exception:
            pass

        # Enforce merge blocks on every candidate.
        fixed_candidates = [self._apply_blocks(cand, blocks_by_col) for cand in candidates]

        # Score each candidate with the true serial Trie objective.
        best_orders, best_score, best_strs = None, -1, None
        for cand in fixed_candidates:
            strs = self._serialize_rows(arr, cand)
            sc = self._score_serial(strs)
            if sc > best_score:
                best_score, best_orders, best_strs = sc, cand, strs

        # Build output DataFrame.
        out_arr = np.empty((n_rows, ncols), dtype=object)
        for i in range(n_rows):
            perm = best_orders[i]
            for j, c in enumerate(perm):
                out_arr[i, j] = arr[i, c]

        out_cols = [original_cols[c] for c in best_orders[0]]
        # Ensure unique column labels (duplicate names possible)
        final_cols = list(out_cols)
        seen: Dict[str, int] = {}
        for k, nm in enumerate(final_cols):
            if nm in seen:
                seen[nm] += 1
                final_cols[k] = f"{nm}.{seen[nm]}"
            else:
                seen[nm] = 0

        out_df = pd.DataFrame(out_arr, columns=final_cols)
        out_df.index = df.index.copy() if len(df.index) == n_rows else pd.RangeIndex(n_rows)

        column_orderings = [[original_cols[c] for c in perm] for perm in best_orders]

        assert out_df.shape == (n_rows, ncols)
        return out_df, column_orderings

    def _conditional_orders_units(self, unit_codes, unit_serlens, n_rows, n_units,
                                  base_unit_order, max_depth: int = 8,
                                  min_group: int = 4) -> List[List[int]]:
        orders: List[List[int] or None] = [None] * n_rows
        all_units = frozenset(range(n_units))

        def rec(rows: np.ndarray, avail: frozenset, depth: int):
            if len(rows) == 0:
                return
            if len(rows) == 1 or depth >= max_depth or len(avail) <= 1:
                fixed = [u for u in base_unit_order if u in avail]
                rest = [u for u in base_unit_order if u not in avail]
                for i in rows:
                    if orders[i] is None:
                        orders[i] = fixed + rest
                return
            best_u, best_score = None, -1.0
            for u in avail:
                uc = unit_codes[u][rows]
                uniq, counts = np.unique(uc, return_counts=True)
                sl = unit_serlens[u]
                s = 0.0
                for code, cnt in zip(uniq.tolist(), counts.tolist()):
                    if cnt > 1:
                        s += sl.get(code, 1) * cnt * (cnt - 1)
                if s > best_score:
                    best_score, best_u = s, u
            if best_u is None or best_score <= 0:
                fixed = [x for x in base_unit_order if x in avail]
                rest = [x for x in base_unit_order if x not in avail]
                for i in rows:
                    if orders[i] is None:
                        orders[i] = fixed + rest
                return
            uc = unit_codes[best_u][rows]
            next_avail = avail - {best_u}
            for u_val in np.unique(uc).tolist():
                sub_rows = rows[uc == u_val]
                if len(sub_rows) < min_group or len(next_avail) == 0:
                    fixed = [x for x in base_unit_order if x in next_avail]
                    rest = [x for x in base_unit_order
                            if x not in avail and x != best_u]
                    for i in sub_rows:
                        if orders[i] is None:
                            orders[i] = [best_u] + fixed + rest
                    continue
                rec(sub_rows, next_avail, depth + 1)
                for i in sub_rows:
                    if orders[i] is not None and best_u not in orders[i]:
                        orders[i] = [best_u] + orders[i]

        rec(np.arange(n_rows), all_units, 0)
        for i in range(n_rows):
            if orders[i] is None:
                orders[i] = list(base_unit_order)
        return orders

# EVOLVE-BLOCK-END