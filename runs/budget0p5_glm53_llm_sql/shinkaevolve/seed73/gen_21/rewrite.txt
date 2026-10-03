# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List
from collections import Counter, defaultdict


class Evolved(Algorithm):
    """
    Serial character-Trie prefix caching optimizer.

    Produces per-row column orderings that maximize shared serialized
    prefixes across the whole row collection, scored with the exact
    Trie objective (sum of adjacent LCPs of sorted serialized rows).
    All original cells, rows and columns are preserved exactly.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------ #
    # serialization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ser(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and v != v:
            return ""
        try:
            if v is pd.NaT:
                return ""
        except Exception:
            pass
        try:
            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
        return str(v)

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        if a == b:
            return len(a)
        m = min(len(a), len(b))
        if m == 0:
            return 0
        lo, hi = 0, m
        # binary search on prefix equality (comparisons run in C)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _score(self, strings: List[str]) -> int:
        """Exact ideal Trie reuse: sum of adjacent LCPs after sorting."""
        if not strings:
            return 0
        s = sorted(strings)
        total = 0
        prev = s[0]
        for cur in s[1:]:
            if cur != prev:
                total += self._lcp(prev, cur)
            else:
                total += len(cur)
            prev = cur
        return total

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
        n, p = df.shape
        if n == 0 or p == 0:
            return df.copy(), [list(original_columns) for _ in range(n)]

        work = df.copy()
        cols = list(work.columns)
        col_index = {c: i for i, c in enumerate(cols)}

        # ---------------- units (merged columns stay adjacent) ---------- #
        merged = set()
        units: List[List[str]] = []
        for grp in col_merge:
            g = [c for c in grp if c in col_index]
            if len(g) > 1:
                units.append(g)
                merged.update(g)
        for c in cols:
            if c not in merged:
                units.append([c])
        covered = [c for u in units for c in u]
        if len(covered) != p or set(covered) != set(cols):
            units = [[c] for c in cols]
        num_units = len(units)
        unit_pos = [[col_index[c] for c in u] for u in units]
        unit_name = [u[0] for u in units]

        # ---------------- serialized matrix ----------------------------- #
        ser_cols: List[List[str]] = []
        for c in cols:
            vals = work[c].tolist()
            ser_cols.append([self._ser(v) for v in vals])
        # per-row value tuples (for row-string construction)
        row_ser = [list(t) for t in zip(*ser_cols)]

        def flat_positions(unit_order):
            return [j for u in unit_order for j in unit_pos[u]]

        def row_string(i, flat):
            r = row_ser[i]
            return "".join([r[j] for j in flat])

        # ---------------- global unit statistics ------------------------ #
        unit_score = [0] * num_units
        unit_chars = [0] * num_units
        col_code: List[List[int]] = []
        col_vlen: List[List[int]] = []
        for u in range(num_units):
            s = 0
            ch = 0
            for j in unit_pos[u]:
                cnt = Counter(ser_cols[j])
                s += sum(len(v) * c * (c - 1) for v, c in cnt.items())
                ch += sum(len(v) * c for v, c in cnt.items())
            unit_score[u] = s
            unit_chars[u] = ch
        for j in range(p):
            mapping = {}
            codes = []
            lens = []
            for v in ser_cols[j]:
                code = mapping.get(v)
                if code is None:
                    code = len(lens)
                    mapping[v] = code
                    lens.append(len(v))
                codes.append(code)
            col_code.append(codes)
            col_vlen.append(lens)

        def global_order(unit_ids):
            return sorted(unit_ids, key=lambda u: (-unit_score[u], unit_name[u]))

        all_units = list(range(num_units))
        cand_A = global_order(all_units)
        cand_C = sorted(all_units, key=lambda u: (-unit_chars[u], -unit_score[u], unit_name[u]))

        max_depth = col_stop if (col_stop is not None and col_stop > 0) else min(num_units, 8)
        max_depth = min(max_depth, num_units)
        if row_stop is not None and row_stop > 0:
            max_depth = min(max_depth, row_stop)
        max_cands = 12

        # ---------------- candidate B: conditional partition tree -------- #
        orderings_B: List[List[int]] = [[] for _ in range(n)]

        def build_tree(unit_ids, row_idx, prefix, depth):
            if len(row_idx) < 2 or not unit_ids or depth >= max_depth:
                rest = global_order(unit_ids)
                full = prefix + rest
                for i in row_idx:
                    orderings_B[i] = full
                return
            cands = global_order(unit_ids)[:max_cands]
            best_u = None
            best_sav = 0
            for u in cands:
                sav = 0
                for j in unit_pos[u]:
                    codes = col_code[j]
                    cnt = Counter(codes[i] for i in row_idx)
                    lens = col_vlen[j]
                    for code, c in cnt.items():
                        if c > 1:
                            sav += lens[code] * c * (c - 1)
                if sav > best_sav:
                    best_sav = sav
                    best_u = u
            if best_u is None or best_sav <= max(0, early_stop):
                rest = global_order(unit_ids)
                full = prefix + rest
                for i in row_idx:
                    orderings_B[i] = full
                return
            rest_ids = [u for u in unit_ids if u != best]
            rest_ids = [u for u in unit_ids if u != best_u]
            groups = defaultdict(list)
            j0 = unit_pos[best_u][0]
            codes0 = col_code[j0]
            for i in row_idx:
                groups[codes0[i]].append(i)
            new_prefix = prefix + [best_u]
            if len(groups) == 1:
                # no real split; avoid infinite descent
                rest = global_order(rest_ids)
                full = new_prefix + rest
                for i in row_idx:
                    orderings_B[i] = full
                return
            for g in groups.values():
                build_tree(rest_ids, g, new_prefix, depth + 1)

        # ---------------- build and score candidates --------------------- #
        candidates = []  # (name, per-row unit orders)
        candidates.append(("global_freq", [cand_A for _ in range(n)]))
        candidates.append(("global_chars", [cand_C for _ in range(n)]))

        use_tree = n <= 300000 and num_units > 1
        if use_tree:
            build_tree(all_units, list(range(n)), [], 0)
            if all(len(o) == num_units for o in orderings_B):
                candidates.append(("cond_tree", orderings_B))

        best_orders = None
        best_score = -1
        for _, orders in candidates:
            strs = [row_string(i, flat_positions(orders[i])) for i in range(n)]
            sc = self._score(strs)
            if sc > best_score:
                best_score = sc
                best_orders = orders

        # ---------------- enforce one-way dependencies -------------------- #
        dep_pairs = []
        for dep in one_way_dep or []:
            if isinstance(dep, (list, tuple)) and len(dep) == 2:
                a = [c for c in cols if dep[0] in c]
                b = [c for c in cols if dep[1] in c]
                if len(a) == 1 and len(b) == 1:
                    dep_pairs.append((a[0], b[0]))

        # ---------------- expand to per-row column orderings ------------- #
        column_orderings: List[List[str]] = []
        for i in range(n):
            names = [cols[j] for u in best_orders[i] for j in unit_pos[u]]
            for a, b in dep_pairs:
                ia = names.index(a)
                ib = names.index(b)
                if ia > ib:
                    names.pop(ia)
                    names.insert(ib, a)
            column_orderings.append(names)

        # ---------------- build output DataFrame -------------------------- #
        out_arr = np.empty((n, p), dtype=object)
        src = work
        src_values = [src[c].tolist() for c in cols]
        for i in range(n):
            order = column_orderings[i]
            for j, cname in enumerate(order):
                out_arr[i, j] = src_values[col_index[cname]][i]
        out_df = pd.DataFrame(out_arr, columns=cols, index=work.index)
        for c in cols:
            try:
                out_df[c] = out_df[c].astype(work[c].dtype, errors="ignore")
            except Exception:
                pass

        # sanity: every per-row ordering is a permutation of all columns
        assert all(sorted(o) == sorted(original_columns) for o in column_orderings)
        assert out_df.shape == df.shape

        return out_df, column_orderings

# EVOLVE-BLOCK-END