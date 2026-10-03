# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict, Optional


class Evolved(Algorithm):
    """
    Character-Trie-aware column reordering.

    Builds a small number of deterministic candidate per-row column orderings
    (global frequency-ranked, an alternate length/frequency tradeoff, and a
    conditional partition tree), scores each with the exact ideal Trie reuse
    objective (sum of adjacent LCPs over sorted serialized rows), and returns
    the best. All original cells are preserved; only their per-row placement
    changes.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _serialize_value(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        if v is pd.NA or v is pd.NaT:
            return ""
        try:
            if v != v:  # NaN check for objects
                return ""
        except Exception:
            pass
        if isinstance(v, bool):
            return "True" if v else "False"
        if isinstance(v, float) and float(v).is_integer():
            # match pandas str conversion style used by astype(str)
            s = str(v)
            return s
        return str(v)

    def _serialize_column(self, col_series) -> np.ndarray:
        out = []
        for v in col_series.tolist():
            out.append(self._serialize_value(v))
        return np.array(out, dtype=object)

    # ------------------------------------------------------------------
    # Exact Trie objective: sum of adjacent LCPs over sorted strings
    # ------------------------------------------------------------------
    @staticmethod
    def _lcp(a: str, b: str) -> int:
        if a == b:
            return len(a)
        lo, hi = 0, min(len(a), len(b))
        # binary search on prefix equality (C-speed slicing comparisons)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _trie_score(self, strings: List[str]) -> int:
        if len(strings) <= 1:
            return 0
        arr = sorted(strings)
        total = 0
        prev = arr[0]
        for s in arr[1:]:
            if s != prev:
                total += self._lcp(prev, s)
                prev = s
        return total

    # ------------------------------------------------------------------
    # Column block handling (col_merge: keep groups adjacent, given order)
    # ------------------------------------------------------------------
    def _build_blocks(self, columns: List[str], col_merge: List[List[str]]) -> List[List[str]]:
        merged = {}
        for group in col_merge:
            present = [c for c in group if c in set(columns)]
            if len(present) >= 2:
                for c in present:
                    merged[c] = present
        blocks = []
        seen = set()
        for c in columns:
            if c in merged:
                if c in seen:
                    continue
                blocks.append(list(merged[c]))
                seen.update(merged[c])
            else:
                blocks.append([c])
        return blocks

    # ------------------------------------------------------------------
    # Column statistics for ranking
    # ------------------------------------------------------------------
    def _col_weight(self, ser: np.ndarray, mode: str) -> float:
        # ser: serialized strings of a column (aligned with rows)
        uniq, inv, counts = np.unique(ser, return_inverse=True, return_counts=True)
        lens = np.array([len(u) for u in uniq], dtype=np.int64)
        cnt = counts.astype(np.int64)
        if mode == "freq":
            w = float(np.sum(lens * cnt * (cnt - 1)))
        else:  # "len" tradeoff
            w = float(np.sum(lens * (cnt - 1)))
        return w

    # ------------------------------------------------------------------
    # Candidate constructions
    # ------------------------------------------------------------------
    def _global_order(self, blocks, block_sers, mode: str):
        weights = [self._col_weight(np.concatenate([block_sers[b] for b in block]) if len(block) > 1 else block_sers[block[0]], mode)
                   for block in blocks]
        order_idx = sorted(range(len(blocks)), key=lambda i: (-weights[i], i))
        ordered_blocks = [blocks[i] for i in order_idx]
        return ordered_blocks

    def _conditional_tree_orders(self, blocks, block_sers, n_rows):
        # Recursive conditional partition: per-group next-column selection.
        orders = [None] * n_rows
        all_block_idx = list(range(len(blocks)))

        def recurse(row_indices, remaining_blocks):
            if len(remaining_blocks) == 0:
                return
            if len(remaining_blocks) == 1 or len(row_indices) <= 1:
                bl = remaining_blocks
                for r in row_indices:
                    if orders[r] is None:
                        orders[r] = [b for block in bl for b in block]
                return
            # choose column with best length-weighted repetition within group
            best_b, best_w = None, -1.0
            for b in remaining_blocks:
                block = blocks[b]
                ser = np.concatenate([block_sers[c][row_indices] for c in block]) if len(block) > 1 else block_sers[block[0]][row_indices]
                uniq, inv, counts = np.unique(ser, return_inverse=True, return_counts=True)
                lens = np.array([len(u) for u in uniq], dtype=np.int64)
                cnt = counts.astype(np.int64)
                w = float(np.sum(lens * cnt * (cnt - 1)))
                if w > best_w:
                    best_w, best_b = w, b
            if best_w <= 0:
                bl = remaining_blocks
                for r in row_indices:
                    if orders[r] is None:
                        orders[r] = [c for block in bl for c in block]
                return
            # assign this block first for these rows, partition by its values
            block = blocks[best_b]
            ser = np.concatenate([block_sers[c][row_indices] for c in block]) if len(block) > 1 else block_sers[block[0]][row_indices]
            rest = [b for b in remaining_blocks if b != best_b]
            uniq, inv = np.unique(ser, return_inverse=True)
            groups: Dict[int, List[int]] = {}
            for pos, r in enumerate(row_indices):
                groups.setdefault(int(inv[pos]), []).append(r)
            for g_rows in groups.values():
                for r in g_rows:
                    if orders[r] is None:
                        orders[r] = [c for c in block]  # first block
                # recurse for suffix orders
                suffix_holder = {}
                def build(r, remaining, out):
                    pass
                # simpler: recursive suffix assignment via temporary orders
                sub_orders = {}
                self._assign_suffix(g_rows, rest, blocks, block_sers, orders, list(block))
            return

        def _assign(row_indices, remaining_blocks):
            if not remaining_blocks:
                return
            if len(remaining_blocks) == 1 or len(row_indices) <= 1:
                bl = [blocks[i] for i in remaining_blocks]
                for r in row_indices:
                    orders[r] = (orders[r] or []) + [c for block in bl for c in block]
                return
            best_b, best_w = None, -1.0
            for b in remaining_blocks:
                block = blocks[b]
                ser = np.concatenate([block_sers[c][row_indices] for c in block]) if len(block) > 1 else block_sers[block[0]][row_indices]
                uniq, inv, counts = np.unique(ser, return_inverse=True, return_counts=True)
                lens = np.array([len(u) for u in uniq], dtype=np.int64)
                cnt = counts.astype(np.int64)
                w = float(np.sum(lens * cnt * (cnt - 1)))
                if w > best_w:
                    best_w, best_b = w, b
            if best_w <= 0:
                bl = [blocks[i] for i in remaining_blocks]
                for r in row_indices:
                    orders[r] = (orders[r] or []) + [c for block in bl for c in block]
                return
            block = blocks[best_b]
            rest = [b for b in remaining_blocks if b != best_b]
            if len(block) > 1:
                ser = np.concatenate([block_sers[c][row_indices] for c in block])
            else:
                ser = block_sers[block[0]][row_indices]
            uniq, inv = np.unique(ser, return_inverse=True)
            groups: Dict[int, List[int]] = {}
            for pos, r in enumerate(row_indices):
                groups.setdefault(int(inv[pos]), []).append(r)
            for r in row_indices:
                orders[r] = (orders[r] or []) + list(block)
            for g_rows in groups.values():
                _assign(g_rows, rest)

        try:
            _assign(list(range(n_rows)), all_block_idx)
        except RecursionError:
            # fall back: fill any None with global order
            pass
        ordered_blocks_fallback = [blocks[i] for i in all_block_idx]
        fallback = [c for block in ordered_blocks_fallback for c in block]
        for r in range(n_rows):
            if orders[r] is None or len(orders[r]) != len(fallback):
                orders[r] = list(fallback)
        return orders

    # ------------------------------------------------------------------
    # Build candidate row strings / dataframes
    # ------------------------------------------------------------------
    def _rows_for_order(self, row_orders, block_sers, n_rows, columns):
        strings = []
        for r in range(n_rows):
            parts = [block_sers[c][r] for c in row_orders[r]]
            strings.append("".join(parts))
        return strings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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
        original_columns = list(df.columns)
        n_rows, n_cols = df.shape

        # Edge cases
        if n_cols == 0:
            return df.copy(), []
        if n_rows == 0:
            return df.copy(), []

        # Serialize columns once, reused by all constructions
        block_sers: Dict[str, np.ndarray] = {}
        for c in original_columns:
            block_sers[c] = self._serialize_column(df[c])

        blocks = self._build_blocks(original_columns, col_merge)

        # Candidate 1: global frequency-ranked order
        ordered_blocks_a = self._global_order(blocks, block_sers, "freq")
        order_a = [c for block in ordered_blocks_a for c in block]

        candidates: List[List[List[str]]] = []  # per-row orders

        # Candidate A: same order for all rows
        candidates.append([list(order_a) for _ in range(n_rows)])

        # Candidate B: alternate length/frequency tradeoff
        ordered_blocks_b = self._global_order(blocks, block_sers, "len")
        order_b = [c for block in ordered_blocks_b for c in block]
        if order_b != order_a:
            candidates.append([list(order_b) for _ in range(n_rows)])

        # Candidate C: conditional partition tree (per-row orders)
        # bound work: skip for very wide tables to keep runtime bounded
        if n_cols <= 64:
            try:
                cand_c = self._conditional_tree_orders_safe(blocks, block_sers, n_rows)
                valid = all(
                    sorted(o) == sorted(original_columns) and len(o) == n_cols
                    for o in cand_c
                )
                if valid:
                    candidates.append(cand_c)
            except Exception:
                pass

        # Score each candidate with the exact Trie objective
        best_orders = candidates[0]
        best_score = -1
        total_chars = sum(len(s) for s in self._rows_for_order(candidates[0], block_sers, n_rows, original_columns))
        for cand in candidates:
            strings = self._rows_for_order(cand, block_sers, n_rows, original_columns)
            sc = self._trie_score(strings)
            if sc > best_score:
                best_score = sc
                best_orders = cand

        # Build output DataFrame with per-row column order, preserving values
        col_pos = {c: i for i, c in enumerate(original_columns)}
        out_arr = np.empty((n_rows, n_cols), dtype=object)
        # gather raw values per column once
        raw_vals = {c: df[c].tolist() for c in original_columns}
        for r in range(n_rows):
            o = best_orders[r]
            for j, c in enumerate(o):
                out_arr[r, j] = raw_vals[c][r]

        out_df = pd.DataFrame(out_arr, columns=original_columns, index=df.index)
        # ensure mixed types kept as object
        out_df = out_df.astype(object)

        # Per-row column orderings must be valid permutations of the columns
        column_orderings = [list(o) for o in best_orders]
        return out_df, column_orderings

    # ------------------------------------------------------------------
    # Safe wrapper for conditional tree (avoids deep recursion issues)
    # ------------------------------------------------------------------
    def _conditional_tree_orders_safe(self, blocks, block_sers, n_rows):
        orders = [None] * n_rows
        all_block_idx = list(range(len(blocks)))
        max_depth = 40

        def assign(row_indices, remaining_blocks, depth):
            if not remaining_blocks:
                return
            if len(remaining_blocks) == 1 or len(row_indices) <= 2 or depth >= max_depth:
                bl = [blocks[i] for i in remaining_blocks]
                for r in row_indices:
                    orders[r] = (orders[r] or []) + [c for block in bl for c in block]
                return
            best_b, best_w = None, -1.0
            for b in remaining_blocks:
                block = blocks[b]
                if len(block) > 1:
                    ser = np.concatenate([block_sers[c][row_indices] for c in block])
                else:
                    ser = block_sers[block[0]][row_indices]
                uniq, inv, counts = np.unique(ser, return_inverse=True, return_counts=True)
                lens = np.array([len(u) for u in uniq], dtype=np.int64)
                cnt = counts.astype(np.int64)
                w = float(np.sum(lens * cnt * (cnt - 1)))
                if w > best_w:
                    best_w, best_b = w, b
            if best_w <= 0:
                bl = [blocks[i] for i in remaining_blocks]
                for r in row_indices:
                    orders[r] = (orders[r] or []) + [c for block in bl for c in block]
                return
            block = blocks[best_b]
            rest = [b for b in remaining_blocks if b != best_b]
            if len(block) > 1:
                ser = np.concatenate([block_sers[c][row_indices] for c in block])
            else:
                ser = block_sers[block[0]][row_indices]
            uniq, inv = np.unique(ser, return_inverse=True)
            groups: Dict[int, List[int]] = {}
            for pos, r in enumerate(row_indices):
                groups.setdefault(int(inv[pos]), []).append(r)
            for r in row_indices:
                orders[r] = (orders[r] or []) + list(block)
            for g_rows in groups.values():
                assign(g_rows, rest, depth + 1)

        assign(list(range(n_rows)), all_block_idx, 0)
        fallback = [c for block in blocks for c in block]
        for r in range(n_rows):
            if orders[r] is None or len(orders[r]) != len(fallback):
                orders[r] = list(fallback)
        return orders
# EVOLVE-BLOCK-END