# EVOLVE-BLOCK-START
import numpy as np
import pandas as pd
from solver import Algorithm
from typing import Tuple, List


class Evolved(Algorithm):
    """
    Serial character-Trie prefix caching optimizer (multi-candidate variant).

    1. Serialize every cell exactly like the evaluator ("" for missing,
       str() otherwise) once per column; reuse arrays everywhere.
    2. Build several cheap candidate row-field orderings:
       (A) global frequency-ranked order: columns sorted by
           sum(len(v) * count(v) * (count(v)-1)) over serialized values;
       (C) the original column order (safe fallback);
       (B1) conditional partition tree, split column chosen by
            length-weighted pair repetition len*count*(count-1),
            deterministic global-order tail;
       (B2) same tree but split by *immediate* savings len*(count-1)
            (the true approximate Trie gain of putting a column first),
            with group-local frequency tails at stop nodes;
       (B3) pair-repetition tree with the alternate global tail,
            only when the table is small enough.
       col_merge groups stay adjacent as ordering units.
    3. Score each candidate exactly once with the exact ideal Trie reuse
       objective: serialize rows, sort, sum adjacent LCPs (binary search
       on prefix-slice equality so comparisons run in C).
    4. Return the best candidate; no cell values, rows or columns altered.
    """

    def _serialize_columns(self, df: pd.DataFrame, cols: List[str]):
        """Return dict col -> object array of serialized cell strings."""
        ser = {}
        for c in cols:
            s = df[c]
            arr = s.astype(str).values.astype(object)
            na = s.isna().values
            if na.any():
                arr = arr.copy()
                arr[na] = ""
            ser[c] = arr
        return ser

    def _lcp(self, a: str, b: str) -> int:
        """Longest common prefix length via binary search (C-speed slices)."""
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

    def _trie_score(self, strs: List[str]) -> int:
        """Ideal Trie reuse = sum of adjacent LCPs after sorting."""
        if len(strs) < 2:
            return 0
        ss = sorted(strs)
        tot = 0
        prev = ss[0]
        lcp = self._lcp
        for cur in ss[1:]:
            tot += lcp(prev, cur)
            prev = cur
        return tot

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
        n, m = df.shape
        cols = list(df.columns)

        if n == 0 or m == 0:
            orders = [list(cols) for _ in range(n)]
            return df.copy().astype(object), orders

        # ---- col_merge: grouped columns stay adjacent as ordering units ----
        merged = {}
        for group in (col_merge or []):
            g = [c for c in group if c in cols]
            for c in g:
                merged[c] = g
        units = []
        seen = set()
        for c in cols:
            if c in merged and c not in seen:
                g = [x for x in merged[c] if x in cols and x not in seen]
                units.append(g)
                seen.update(g)
            elif c not in seen:
                units.append([c])
                seen.add(c)
        unit_key = {id(u): i for i, u in enumerate(units)}

        # ---- serialization (shared across all candidates) ----
        ser = self._serialize_columns(df, cols)

        # ---- per-column statistics: freq score + integer factorization ----
        scores = {}
        codes = {}
        codelen = {}
        for c in cols:
            arr = ser[c]
            uniq, inv = np.unique(arr, return_inverse=True)
            cnt = np.bincount(inv, minlength=len(uniq))
            lens = np.fromiter((len(x) for x in uniq), dtype=np.int64, count=len(uniq))
            scores[c] = int((lens * cnt * (cnt - 1)).sum())
            codes[c] = inv
            codelen[c] = lens

        # ---- candidate A: global frequency-ranked unit order ----
        unit_freq = {id(u): sum(scores[c] for c in u) for u in units}
        order_A_units = sorted(units, key=lambda u: (-unit_freq[id(u)], unit_key[id(u)]))
        global_order = [c for u in order_A_units for c in u]

        # ---- candidate C: original order ----
        order_C = list(cols)

        # ---- serialization helpers ----
        def serialize_global(col_order):
            return ["".join(t) for t in zip(*[ser[c] for c in col_order])]

        def serialize_rows(row_orders):
            out = []
            for r in range(n):
                o = row_orders[r]
                out.append("".join([ser[c][r] for c in o]))
            return out

        scA = self._trie_score(serialize_global(global_order))
        scC = self._trie_score(serialize_global(order_C))
        if scA >= scC:
            tail_order, alt_tail, best_global = global_order, order_C, scA
        else:
            tail_order, alt_tail, best_global = order_C, global_order, scC

        # ---- bounded conditional partition tree factory ----
        depth_cap = 10 if col_stop is None else max(1, min(int(col_stop), 10))
        max_split_candidates = 12
        col_pos = {c: i for i, c in enumerate(cols)}
        all_rows = np.arange(n)
        local_tail_max_rows = 20000
        local_tail_max_cols = 60

        def build_tree(tail, split_mode, local_tail):
            """Assign per-row column orders via conditional prefix partition.

            split_mode 'pair': score = sum len(v)*cnt*(cnt-1) (pair repetition).
            split_mode 'imm':  score = sum len(v)*(cnt-1)  (immediate savings,
                               the true approximate Trie gain of leading col).
            local_tail: at stop nodes order remaining columns by within-group
                        repetition instead of the global order.
            """
            order_arr = np.empty((n, m), dtype=object)

            def local_order(rows, remaining):
                if (len(rows) > local_tail_max_rows
                        or len(remaining) > local_tail_max_cols):
                    return [c for c in tail if c in remaining]
                loc = {}
                for c in remaining:
                    nb = np.bincount(codes[c][rows], minlength=len(codelen[c]))
                    loc[c] = int((codelen[c] * (nb - 1).clip(0)).sum())
                return sorted(remaining, key=lambda c: (-loc[c], col_pos[c]))

            def build(rows, remaining, pos, depth):
                if pos >= m or len(rows) == 0 or not remaining:
                    return
                if len(rows) < 2 or depth >= depth_cap:
                    tl = local_order(rows, remaining) if local_tail else \
                        [c for c in tail if c in remaining]
                    for r in rows:
                        order_arr[r, pos:pos + len(tl)] = tl
                    return
                if len(remaining) <= 25:
                    cands = remaining
                else:
                    cands = sorted(
                        remaining, key=lambda c: (-scores[c], col_pos[c])
                    )[:max_split_candidates]
                best, best_sc = None, -1
                for c in cands:
                    nb = np.bincount(codes[c][rows], minlength=len(codelen[c]))
                    if split_mode == "pair":
                        sc = int((codelen[c] * nb * (nb - 1)).sum())
                    else:
                        sc = int((codelen[c] * (nb - 1).clip(0)).sum())
                    if sc > best_sc:
                        best_sc, best = sc, c
                if best is None or best_sc <= 0:
                    tl = local_order(rows, remaining) if local_tail else \
                        [c for c in tail if c in remaining]
                    for r in rows:
                        order_arr[r, pos:pos + len(tl)] = tl
                    return
                for r in rows:
                    order_arr[r, pos] = best
                rest = [c for c in remaining if c != best]
                if not rest:
                    return
                cd = codes[best][rows]
                for v in np.unique(cd):
                    build(rows[cd == v], rest, pos + 1, depth + 1)

            build(all_rows, list(cols), 0, 0)
            return [list(order_arr[r]) for r in range(n)]

        # ---- build & score candidates (each scored exactly once) ----
        candidates = []  # (score, orders)

        tree1 = build_tree(tail_order, "pair", False)
        scB1 = self._trie_score(serialize_rows(tree1))
        candidates.append((scB1, tree1))

        tree2 = build_tree(tail_order, "imm", True)
        scB2 = self._trie_score(serialize_rows(tree2))
        candidates.append((scB2, tree2))

        if n * m <= 2_000_000:
            tree3 = build_tree(alt_tail, "pair", False)
            scB3 = self._trie_score(serialize_rows(tree3))
            candidates.append((scB3, tree3))

        candidates.append((best_global, [list(tail_order) for _ in range(n)]))

        orders = max(candidates, key=lambda t: t[0])[1]

        # ---- materialize output preserving every value ----
        colidx = {c: i for i, c in enumerate(cols)}
        vals = df.to_numpy(dtype=object) if m else np.empty((n, 0), dtype=object)
        out = np.empty((n, m), dtype=object)
        for r in range(n):
            vrow = vals[r]
            out[r] = [vrow[colidx[c]] for c in orders[r]]

        result = pd.DataFrame(out, columns=cols, index=df.index).astype(object)
        return result, orders
# EVOLVE-BLOCK-START