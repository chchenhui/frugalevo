# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List
from collections import defaultdict


class Evolved(Algorithm):
    """
    Character-Trie prefix caching optimizer.
    Produces per-row column permutations, selects among a few cheap candidate
    constructions using the exact serial Trie reuse objective (sum of adjacent
    LCPs over sorted serialized rows), while preserving all data and shape.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _serialize_cell(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        if v is pd.NA:
            return ""
        if isinstance(v, str):
            return v
        return str(v)

    def _serialize_matrix(self, df: pd.DataFrame, codes) -> np.ndarray:
        """Serialize each cell once. codes[i][j] maps to string table."""
        n_rows = len(df)
        ser = np.empty((n_rows, len(codes)), dtype=object)
        col_arrays = [df[c].to_numpy() for c in df.columns]
        for j in range(len(codes)):
            table = codes[j]["table"]
            arr = col_arrays[j]
            col_ser = np.empty(n_rows, dtype=object)
            for i in range(n_rows):
                v = arr[i]
                col_ser[i] = table[id(v)] if False else None
            ser[:, j] = col_ser
        return ser

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Longest common prefix length; C-speed comparisons via slicing."""
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

    def _score_orders(self, row_strings) -> int:
        """Exact ideal Trie reuse: sum of adjacent LCPs over sorted strings."""
        strings = sorted(row_strings)
        total = len(strings[0])
        prev = strings[0]
        for s in strings[1:]:
            total += len(s) - self._lcp(prev, s)
            prev = s
        # reuse = total_chars - total (total = number of trie edges = distinct chars)
        n_chars = sum(len(s) for s in strings)
        return n_chars - total

    # ------------------------------------------------------------------
    # candidate constructions
    # ------------------------------------------------------------------
    def _global_order(self, df, units):
        """Rank columns by sum over serialized values of len(v)*count*(count-1)."""
        scores = []
        for unit in units:
            s = 0
            for col in unit:
                vals = df[col]
                counts = vals.value_counts(dropna=False)
                for v, c in counts.items():
                    sv = self._serialize_cell(v)
                    s += len(sv) * c * (c - 1)
            scores.append((-s, len(unit), tuple(unit)))
        scores.sort()
        return [list(u[-1]) for u in scores]

    def _conditional_order(self, df, units, depth_budget=12):
        """Recursive conditional grouping: pick a leading column by length-weighted
        pair repetition, partition, recurse on remaining columns per group."""
        order_cache = {}

        def build(row_idx, remaining_units, depth):
            key = tuple(remaining_units)
            if not remaining_units or depth >= depth_budget:
                return [list(remaining_units)]
            # choose column with best len*count*(count-1) among this row subset
            best_col, best_score = None, 0
            for unit in remaining_units:
                for col in unit:
                    vals = df[col].iloc[row_idx]
                    counts = vals.value_counts(dropna=False)
                    s = 0
                    for v, c in counts.items():
                        s += len(self._serialize_cell(v)) * c * (c - 1)
                    if s > best_score:
                        best_score, best_col = s, col
            if best_col is None or best_score <= 0:
                return [list(remaining_units)]
            best_unit = [u for u in remaining_units if best_col in u][0]
            rest = [u for u in remaining_units if best_col not in u]
            # order columns within best_unit: best_col first then by repetition
            sub = [best_unit] + rest
            orders = []
            groups = df[best_col].iloc[row_idx].fillna("\x00__NA__\x00") if False else df[best_col].iloc[row_idx]
            groups = groups.astype(object)
            gmap = defaultdict(list)
            for pos, i in enumerate(row_idx):
                v = groups.iloc[pos]
                try:
                    hash(v)
                except TypeError:
                    v = str(v)
                gmap[v].append(i)
            if len(gmap) <= 1:
                return [list(remaining_units)]
            for v, idxs in gmap.items():
                for sub_order in build(idxs, rest, depth + 1):
                    orders.append([best_unit] + sub_order)
            return orders

        n = len(df)
        row_orders = []
        all_units = list(units)
        # get group orders, then map each row to its order
        # simpler: compute order per row via recursion tracking membership
        row_order = [None] * n

        def assign(row_idx, remaining_units, prefix, depth):
            if not remaining_units or depth >= depth_budget:
                for i in row_idx:
                    row_order[i] = prefix + list(remaining_units)
                return
            best_col, best_score = None, 0
            for unit in remaining_units:
                for col in unit:
                    vals = df[col].iloc[row_idx]
                    counts = vals.value_counts(dropna=False)
                    s = 0
                    for v, c in counts.items():
                        s += len(self._serialize_cell(v)) * c * (c - 1)
                    if s > best_score:
                        best_score, best_col = s, col
            if best_col is None or best_score <= 0:
                for i in row_idx:
                    row_order[i] = prefix + list(remaining_units)
                return
            best_unit = [u for u in remaining_units if best_col in u][0]
            rest = [u for u in remaining_units if best_col not in u]
            if len(row_idx) <= 2 or len(df[best_col].iloc[row_idx].value_counts(dropna=False)) <= 1:
                for i in row_idx:
                    row_order[i] = prefix + list(remaining_units)
                return
            groups = df[best_col].iloc[row_idx].astype(object)
            gmap = defaultdict(list)
            for pos, i in enumerate(row_idx):
                v = groups.iloc[pos]
                try:
                    hash(v)
                except TypeError:
                    v = str(v)
                gmap[v].append(i)
            if len(gmap) <= 1:
                for i in row_idx:
                    row_order[i] = prefix + list(remaining_units)
                return
            for v, idxs in gmap.items():
                assign(idxs, rest, prefix + [best_unit], depth + 1)

        assign(list(range(n)), all_units, [], 0)
        return row_order

    # ------------------------------------------------------------------
    # main API
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
        original_columns = list(df.columns)
        n_rows, n_cols = df.shape
        if n_rows == 0 or n_cols == 0:
            return df.copy(), [[] for _ in range(n_rows)] if n_cols else [
                [] for _ in range(n_rows)
            ]

        # ---------- build units: merged columns + dependencies stay contiguous ----------
        merged_map = {}
        for group in col_merge:
            for c in group:
                merged_map[c] = group

        dep_children = defaultdict(list)
        dep_in_unit = set()
        for a, b in one_way_dep:
            ca = [c for c in original_columns if a in c]
            cb = [c for c in original_columns if b in c]
            if len(ca) == 1 and len(cb) == 1:
                dep_children[ca[0]].append(cb[0])
                dep_in_unit.add(cb[0])

        units = []
        used = set()
        for c in original_columns:
            if c in used:
                continue
            group = merged_map.get(c, [c])
            unit = []
            for gc in group:
                if gc in original_columns and gc not in used:
                    unit.append(gc)
                    used.add(gc)
            # attach dependents into the same unit after their source
            for src in list(unit):
                for dep in dep_children.get(src, []):
                    if dep in original_columns and dep not in used:
                        unit.append(dep)
                        used.add(dep)
            for dep in dep_children.get(c, []):
                if dep in original_columns and dep not in used:
                    unit.append(dep)
                    used.add(dep)
            if unit:
                units.append(unit)
        # any leftovers
        for c in original_columns:
            if c not in used:
                units.append([c])
                used.add(c)

        # ---------- candidates ----------
        candidate_orders = []  # each: list (len n_rows) of column lists

        # Candidate A: global frequency-ranked unit order
        global_units = self._global_order(df, units)
        candidate_orders.append([list(global_units) for _ in range(n_rows)])

        # Candidate B: conditional partition tree
        try:
            cond = self._conditional_order(df, units)
            if cond and all(o is not None for o in cond):
                flat_cond = []
                for o in cond:
                    flat = []
                    for u in o:
                        flat.extend(u)
                    if len(flat) == n_cols:
                        flat_cond.append(flat)
                if len(flat_cond) == n_rows:
                    candidate_orders.append(flat_cond)
        except Exception:
            pass

        # Candidate C: original column order (reliable fallback)
        candidate_orders.append([list(original_columns) for _ in range(n_rows)])

        # ---------- score candidates with exact Trie objective ----------
        cell_str = {}
        for c in original_columns:
            arr = df[c].to_numpy(dtype=object)
            strs = [self._serialize_cell(v) for v in arr]
            cell_str[c] = strs

        def rows_for(order_per_row):
            out = []
            for i in range(n_rows):
                out.append("".join(cell_str[c][i] for c in order_per_row[i]))
            return out

        best_orders, best_score = candidate_orders[0], -1
        for orders in candidate_orders:
            try:
                rows = rows_for(orders)
                score = self._score_orders(rows)
            except Exception:
                continue
            if score > best_score:
                best_score, best_orders = score, orders

        # ---------- build output preserving every value ----------
        col_pos = {c: j for j, c in enumerate(original_columns)}
        out_vals = np.empty((n_rows, n_cols), dtype=object)
        source_arrays = {c: df[c].to_numpy(dtype=object) for c in original_columns}
        for i in range(n_rows):
            order = best_orders[i]
            for j, c in enumerate(order):
                out_vals[i, j] = source_arrays[c][i]
        result = pd.DataFrame(out_vals, columns=original_columns)
        result = result.astype(object)
        # preserve index
        result.index = df.index

        return result, [list(o) for o in best_orders]

# EVOLVE-BLOCK-END