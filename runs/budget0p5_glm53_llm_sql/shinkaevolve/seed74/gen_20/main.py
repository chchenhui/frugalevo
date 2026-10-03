# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict
from collections import defaultdict


class Evolved(Algorithm):
    """
    Optimizes per-row column orderings for serialized character-prefix reuse.

    The scoring objective is exact Trie prefix reuse over serialized row
    strings, computed internally as the sum of adjacent LCPs after sorting.
    We build a small number of bounded candidate orderings and select the
    best by that exact objective.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------
    # serialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cell_str(v) -> str:
        try:
            if v is None:
                return ""
            if isinstance(v, float) and np.isnan(v):
                return ""
        except (TypeError, ValueError):
            pass
        try:
            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(v, str):
            return v
        return str(v)

    def _build_cell_strings(self, df: pd.DataFrame):
        """Return (col_names, cells) where cells[r][c] is the scoring string."""
        col_names = [str(c) for c in df.columns]
        n = len(df)
        m = len(col_names)
        cells = [["" for _ in range(m)] for _ in range(n)]
        values = df.values
        for r in range(n):
            row = values[r]
            for c in range(m):
                cells[r][c] = self._cell_str(row[c])
        return col_names, cells

    # ------------------------------------------------------------------
    # exact serial-Trie objective: sorted adjacent LCP sum
    # ------------------------------------------------------------------
    @staticmethod
    def _lcp(a: str, b: str) -> int:
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

    def _score_orderings(self, orderings, cells) -> int:
        rows = []
        for order in orderings:
            rows.append("".join(cells[r][c] for c in order))
        rows.sort()
        total = 0
        prev = None
        for s in rows:
            if prev is not None:
                total += self._lcp(prev, s)
            prev = s
        return total

    # ------------------------------------------------------------------
    # candidate constructions
    # ------------------------------------------------------------------
    def _global_scores(self, cells, m, n):
        """Per-column score: sum(len(v) * count * (count-1)) and variants."""
        score_len = [0] * m          # length-weighted pair repetition
        score_count = [0] * m        # frequency-only
        for c in range(m):
            cnt = defaultdict(int)
            for r in range(n):
                cnt[cells[r][c]] += 1
            sl = 0
            sc = 0
            for v, k in cnt.items():
                if k > 1:
                    sl += len(v) * k * (k - 1)
                    sc += k * (k - 1)
            score_len[c] = sl
            score_count[c] = sc
        return score_len, score_count

    def _flatten_units(self, unit_order, unit_map):
        out = []
        for u in unit_order:
            out.extend(unit_map[u])
        return out

    def _candidate_global(self, unit_cols, score_len, score_count, cells, n, weighted=True):
        """Order ordering-units by (score, name) descending."""
        score = score_len if weighted else score_count
        order = sorted(
            unit_cols,
            key=lambda u: (score[u[0]] if not isinstance(u, tuple) else max(score[c] for c in u), str(u)),
            reverse=True,
        )
        return order

    def _candidate_conditional(
        self,
        unit_cols,
        unit_map,
        score_len,
        cells,
        n,
        max_depth,
    ):
        """
        Recursive conditional partition tree over factorized integer codes.
        Each row gets its own ordering: chosen unit first, then the ordering
        chosen within its group.
        """
        # flatten current units to column indices
        flat = []
        for u in unit_cols:
            for c in u:
                flat.append(c)

        # factorize each column's values once (per call on current column set)
        codes = {}
        uniques = {}
        for c in flat:
            mapping = {}
            code_list = []
            for r in range(n):
                v = cells[r][c]
                if v not in mapping:
                    mapping[v] = len(mapping)
                code_list.append(mapping[v])
            codes[c] = code_list
            uniques[c] = mapping

        orderings = [None] * n
        # tail ordering: by global length-weighted score descending
        tail = sorted(flat, key=lambda c: (score_len[c], str(c)), reverse=True)

        def recurse(row_idx, units_left, depth):
            if len(units_left) == 0:
                for r in row_idx:
                    orderings[r] = list(tail_remaining(row_idx, units_left))
                return
            if len(row_idx) <= 1 or depth >= max_depth:
                tail_order = self._flatten_units(
                    sorted(units_left, key=lambda u: (score_len[u[0]], str(u[0])), reverse=True),
                    unit_map,
                )
                for r in row_idx:
                    orderings[r] = list(tail_order)
                return
            # pick best unit by length-weighted pair repetition within group
            best_u = None
            best_gain = 0
            for u in units_left:
                gain = 0
                for c in u:
                    if c not in codes:
                        continue
                    cnt = defaultdict(int)
                    cl = codes[c]
                    for r in row_idx:
                        cnt[cl[r]] += 1
                    for k in cnt.values():
                        if k > 1:
                            vlen = 0
                            # average length of the value in this group
                            inv = {code: val for val, code in uniques[c].items()}
                            for r in row_idx:
                                if cl[r] in cnt and cl[r] == cl[r]:
                                    pass
                            # cheap: use unique value length
                            gain += 0
                    # recompute gain properly but bounded
                    cnt2 = defaultdict(int)
                    for r in row_idx:
                        cnt2[cl[r]] += 1
                    g = 0
                    for code, k in cnt2.items():
                        if k > 1:
                            g += len(uniques_lookup(c, code)) * k * (k - 1)
                    if g > best_gain:
                        best_gain = g
                        best_u = u
            if best_u is None or best_gain <= 0:
                tail_order = self._flatten_units(
                    sorted(units_left, key=lambda u: (score_len[u[0]], str(u[0])), reverse=True),
                    unit_map,
                )
                for r in row_idx:
                    orderings[r] = list(tail_order)
                return
            # partition rows by best unit's first column value
            head = list(best_u)
            rest = [u for u in units_left if u is not best_u]
            cl = codes[best_u[0]]
            groups = defaultdict(list)
            for r in row_idx:
                groups[cl[r]].append(r)
            # recurse into each group for the suffix ordering
            suffix_orders = {}
            for code, idxs in groups.items():
                if len(idxs) <= 1 or not rest:
                    suffix_orders[code] = self._flatten_units(
                        sorted(rest, key=lambda u: (score_len[u[0]], str(u[0])), reverse=True),
                        unit_map,
                    )
                else:
                    # simple deterministic suffix: by global score (bounded)
                    suffix_orders[code] = self._flatten_units(
                        sorted(rest, key=lambda u: (score_len[u[0]], str(u[0])), reverse=True),
                        unit_map,
                    )
            for r in row_idx:
                orderings[r] = head + list(suffix_orders[cl[r]])

        def tail_remaining(row_idx, units_left):
            return []

        def uniques_lookup(c, code):
            inv = uniques[c]
            for val, cd in inv.items():
                if cd == code:
                    return val
            return ""

        # Simplify: bounded depth recursion using head-unit selection only
        max_depth_eff = min(max_depth, len(unit_cols))
        # initial call
        all_rows = list(range(n))
        self._conditional_build(
            all_rows, unit_cols, unit_map, score_len, codes, uniques,
            orderings, 0, max_depth_eff,
        )
        return orderings

    def _conditional_build(self, row_idx, unit_cols, unit_map, score_len, codes, uniques, orderings, depth, max_depth):
        tail_order = self._flatten_units(
            sorted(unit_cols, key=lambda u: (score_len[u[0]], str(u[0])), reverse=True),
            unit_map,
        )
        if not unit_cols:
            for r in row_idx:
                orderings[r] = []
            return
        if len(row_idx) <= 1 or depth >= max_depth or len(unit_cols) <= 1:
            for r in row_idx:
                orderings[r] = list(tail_order)
            return
        best_u = None
        best_gain = 0
        for u in unit_cols:
            c = u[0]
            cl = codes.get(c)
            if cl is None:
                continue
            cnt = defaultdict(int)
            for r in row_idx:
                cnt[cl[r]] += 1
            g = 0
            inv = {}
            for val, code in uniques[c].items():
                inv[code] = val
            for code, k in cnt.items():
                if k > 1:
                    g += len(inv[code]) * k * (k - 1)
            if g > best_gain:
                best_gain = g
                best_u = u
        if best_u is None or best_gain <= 0:
            for r in row_idx:
                orderings[r] = list(tail_order)
            return
        head = list(best_u)
        rest = [u for u in unit_cols if u != best_u]
        cl = codes[best_u[0]]
        groups = defaultdict(list)
        for r in row_idx:
            groups[cl[r]].append(r)
        for code, idxs in groups.items():
            self._conditional_build(
                idxs, rest, unit_map, score_len, codes, uniques, orderings, depth + 1, max_depth
            )
        for r in row_idx:
            orderings[r] = head + orderings[r]

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
        out = df.copy()
        n, m = out.shape
        if n == 0 or m == 0:
            return out, [[] for _ in range(n)]

        col_names, cells = self._build_cell_strings(out)

        # Build ordering units: single columns, except merged groups which
        # must stay contiguous (atomic units).
        merged_flat = set()
        units = []
        for group in (col_merge or []):
            grp = [c for c in out.columns if c in group]
            if len(grp) > 1:
                units.append(tuple(str(c) for c in grp))
                merged_flat.update(str(c) for c in grp)
            elif len(grp) == 1:
                units.append((str(grp[0]),))
                merged_flat.add(str(grp[0]))
        for c in out.columns:
            cs = str(c)
            if cs not in merged_flat:
                units.append((cs,))
        if len(units) != m:
            # Fallback: treat all columns as single units
            units = [(str(c),) for c in out.columns]
            merged_flat = set()
            for group in (col_merge or []):
                grp = [str(c) for c in out.columns if c in group]
                for g in grp:
                    merged_flat.add(g)

        unit_map = {u: list(u) for u in units}
        col_index = {str(c): i for i, c in enumerate(out.columns)}
        score_len, score_count = self._global_scores(cells, m, n)
        # unit scores keyed by first column index
        def unit_score(u, score):
            return sum(score[col_index[c]] for c in u)

        # bounds
        max_depth = row_stop if isinstance(row_stop, int) else 8
        max_depth = max(1, min(max_depth, 24))
        if isinstance(col_stop, int):
            max_depth = min(max_depth, max(1, col_stop))

        candidates = []

        # Candidate 1: global length-weighted frequency order
        order1 = sorted(units, key=lambda u: (unit_score(u, score_len), str(u[0])), reverse=True)
        flat1 = self._flatten_units(order1, unit_map)
        candidates.append(([flat1] * n, "global_len"))

        # Candidate 2: global frequency-only order
        order2 = sorted(units, key=lambda u: (unit_score(u, score_count), str(u[0])), reverse=True)
        flat2 = self._flatten_units(order2, unit_map)
        candidates.append(([flat2] * n, "global_count"))

        # Candidate 3: conditional partition tree (row-specific orders)
        try:
            codes = {}
            uniques = {}
            flat_all = [c for u in units for c in u]
            for cname in flat_all:
                ci = col_index[cname]
                mapping = {}
                code_list = []
                for r in range(n):
                    v = cells[r][ci]
                    if v not in mapping:
                        mapping[v] = len(mapping)
                    code_list.append(mapping[v])
                codes[ci] = code_list
                uniques[ci] = mapping
            # index units by first column index
            unit_by_first = {col_index[u[0]]: u for u in units}
            unit_cols_idx = [col_index[u[0]] for u in units]

            orderings3 = [None] * n
            # map unit tuples to lists of column indices
            unit_map_idx = {col_index[u[0]]: [col_index[c] for c in u] for u in units}

            def build(row_idx, unit_firsts, depth):
                if not unit_firsts:
                    for r in row_idx:
                        orderings3[r] = []
                    return
                tail = []
                for uf in sorted(unit_firsts, key=lambda uf: (score_len[uf], str(uf)), reverse=True):
                    tail.extend(unit_map_idx[uf])
                if len(row_idx) <= 1 or depth >= max_depth or len(unit_firsts) == 1:
                    for r in row_idx:
                        orderings3[r] = list(tail)
                    return
                best_uf = None
                best_gain = 0
                for uf in unit_firsts:
                    cl = codes[uf]
                    cnt = defaultdict(int)
                    for r in row_idx:
                        cnt[cl[r]] += 1
                    inv = {code: val for val, code in uniques[uf].items()}
                    g = 0
                    for code, k in cnt.items():
                        if k > 1:
                            g += len(inv[code]) * k * (k - 1)
                    if g > best_gain:
                        best_gain = g
                        best_uf = uf
                if best_uf is None or best_gain <= 0:
                    for r in row_idx:
                        orderings3[r] = list(tail)
                    return
                head = list(unit_map_idx[best_uf])
                rest = [uf for uf in unit_firsts if uf != best_uf]
                cl = codes[best_uf]
                groups = defaultdict(list)
                for r in row_idx:
                    groups[cl[r]].append(r)
                for code, idxs in groups.items():
                    build(idxs, rest, depth + 1)
                for r in row_idx:
                    orderings3[r] = head + orderings3[r]

            build(list(range(n)), unit_cols_idx, 0)
            # convert index orderings to name orderings
            names = [str(c) for c in out.columns]
            orderings3 = [[names[c] for c in order] for order in orderings3]
            candidates.append((orderings3, "conditional"))
        except Exception:
            pass

        # Score candidates by exact serial-Trie objective
        col_pos = {str(c): i for i, c in enumerate(out.columns)}
        best_orders = None
        best_score = -1
        for orderings, tag in candidates:
            try:
                idx_orders = [[col_pos[c] for c in order] for order in orderings]
                s = self._score_orderings(idx_orders, cells)
                if s > best_score:
                    best_score = s
                    best_orders = orderings
            except Exception:
                continue

        if best_orders is None:
            best_orders = [[str(c) for c in out.columns]] * n

        # Ensure output as object dtype to preserve mixed types exactly
        try:
            if out.shape[1] > 0:
                out = out.astype(object)
        except Exception:
            pass

        return out, [list(o) for o in best_orders]


# EVOLVE-BLOCK-END