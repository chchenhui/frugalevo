# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict, Optional


class Evolved(Algorithm):
    """
    Character-Trie prefix-reuse optimized column reordering.
    Serializes rows exactly like the evaluator (join of str values, NaN -> ""),
    builds a small number of cheap candidate per-row column orderings, scores
    each with the exact ideal-Trie reuse (sorted-string adjacent LCP sum), and
    returns the best. Never modifies stored cell values.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------ #
    # Serialization helpers (scoring representation only; values untouched)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cell_str(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        if v is pd.NA or v is pd.NaT:
            return ""
        try:
            if v != v:  # NaN-like
                return ""
        except Exception:
            pass
        return str(v)

    def _serialize_rows(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df.columns)
        out = []
        for col in df.columns:
            s = df[col]
            out.append(s.map(self._cell_str).to_numpy())
        if n == 0:
            return np.array([""] * len(df), dtype=object)
        arr = out[0]
        for i in range(1, n):
            arr = np.char.add(np.asarray(arr, dtype=str), np.asarray(out[i], dtype=str))
        return arr.astype(object)

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        if a == b:
            return len(a)
        lo, hi = 0, min(len(a), len(b))
        # binary search on prefix-slice equality (C-speed compare)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _trie_reuse(self, strings) -> int:
        """Exact ideal Trie reuse = sum of adjacent LCPs after sorting."""
        ss = sorted(strings)
        total = 0
        prev = None
        for s in ss:
            if prev is not None:
                total += self._lcp(prev, s)
            prev = s
        return total

    # ------------------------------------------------------------------ #
    # Column statistics (factorized once per column)
    # ------------------------------------------------------------------ #
    def _factorize_columns(self, df: pd.DataFrame) -> Dict:
        col_info = {}
        for col in df.columns:
            s = df[col]
            vals = s.map(self._cell_str).to_numpy()
            # factorize on serialized values: deterministic and hashable
            codes, uniques = pd.factorize(pd.Series(vals), sort=False)
            codes = np.asarray(codes, dtype=np.int64)
            counts = np.bincount(codes, minlength=len(uniques)) if len(uniques) else np.zeros(0, dtype=np.int64)
            vlen = np.array([len(u) for u in uniques], dtype=np.int64)
            # length-weighted pair repetition: len * count * (count-1)
            w = vlen * counts * np.maximum(counts - 1, 0)
            col_info[col] = {
                "strs": vals,
                "codes": codes,
                "uniques": list(uniques),
                "counts": counts,
                "vlen": vlen,
                "weight": int(w.sum()),
                "nunique": int(len(uniques)),
            }
        return col_info

    # ------------------------------------------------------------------ #
    # Candidate 1: global frequency-ranked column order
    # ------------------------------------------------------------------ #
    def _global_order(self, df: pd.DataFrame, info: Dict, cols: List) -> List:
        def key(c):
            d = info[c]
            # weight desc, then avg len desc, then nunique asc, then name for determinism
            return (-d["weight"], -int(d["vlen"].sum()) if len(df) else 0, d["nunique"], str(c))
        return sorted(cols, key=key)

    # ------------------------------------------------------------------ #
    # Candidate 2: conditional partition tree (per-group suffix orders)
    # ------------------------------------------------------------------ #
    def _partition_order(self, df: pd.DataFrame, info: Dict, cols: List,
                          max_depth: int = 8, min_group: int = 8,
                          max_candidates: int = 12) -> List[List]:
        n = len(df)
        order = [None] * n  # per-row column ordering (as lists of column names)
        rows = np.arange(n)
        used = [False] * len(cols)
        col_idx = {c: i for i, c in enumerate(cols)}

        def build(rows: np.ndarray, prefix: List, depth: int):
            if len(rows) == 0:
                return
            remaining = [c for c in cols if not used[col_idx[c]]]
            # deterministic cheap tail
            if (depth >= max_depth or len(rows) < min_group or len(remaining) <= 1
                    or len(remaining) > 3 * max_candidates):
                tail = self._global_order(df, info, remaining)
                full = prefix + tail
                for r in rows:
                    order[r] = full
                return
            # candidate columns by length-weighted pair repetition among these rows
            cand_scores = []
            for c in remaining:
                codes = info[c]["codes"][rows]
                cnt = np.bincount(codes)
                cnt = cnt[cnt > 0]
                vl = info[c]["vlen"]
                w = int((vl * cnt * np.maximum(cnt - 1, 0)).sum()) if len(cnt) else 0
                cand_scores.append((w, c))
            cand_scores.sort(key=lambda x: (-x[0], str(x[1])))
            best_col, best_w = cand_scores[0]
            if best_w <= 0:
                tail = self._global_order(df, info, remaining)
                full = prefix + tail
                for r in rows:
                    order[r] = full
                return
            used[col_idx[best_col]] = True
            codes = info[best_col]["codes"][rows]
            # partition rows by value of best_col, groups sorted deterministically
            order_by_code = {}
            for r, code in zip(rows, codes):
                order_by_code.setdefault(int(code), []).append(r)
            keys = sorted(order_by_code.keys(), key=lambda k: str(info[best_col]["uniques"][k]))
            for k in keys:
                build(np.array(order_by_code[k], dtype=np.int64), prefix + [best_col], depth + 1)
            used[col_idx[best_col]] = False

        build(rows, [], 0)
        # fill any missed rows defensively
        fallback = self._global_order(df, info, cols)
        for i in range(n):
            if order[i] is None:
                order[i] = fallback
        return order

    # ------------------------------------------------------------------ #
    # Candidate 3: length-first then frequency ordering
    # ------------------------------------------------------------------ #
    def _length_first_order(self, df: pd.DataFrame, info: Dict, cols: List) -> List:
        n = len(df)
        def key(c):
            d = info[c]
            avg = (int(d["vlen"].sum()) / n) if n else 0.0
            return (-avg, -d["weight"], str(c))
        return sorted(cols, key=key)

    # ------------------------------------------------------------------ #
    # Build output dataframe from per-row orderings
    # ------------------------------------------------------------------ #
    def _apply_orders(self, df: pd.DataFrame, orders: List[List], columns: List) -> pd.DataFrame:
        # Build a values matrix in original column space, per row permuted.
        n, m = len(df), len(columns)
        if n == 0:
            return df.copy()
        pos = {c: i for i, c in enumerate(columns)}
        # Extract raw values once per column (object dtype, verbatim)
        col_vals = {c: df[c].tolist() for c in columns}
        # For per-row output, we need a rectangular ordering; rows may differ.
        # Represent orders as index arrays into `columns`.
        idx_orders = []
        for o in orders:
            idx_orders.append([pos[c] for c in o])
        # Materialize rows
        data = [[None] * m for _ in range(n)]
        # column position -> (row, source col index) lists
        for r in range(n):
            row_vals = col_vals
            io = idx_orders[r]
            for j in range(m):
                data[r][j] = col_vals[columns[io[j]]][r]
        out = pd.DataFrame(data, columns=columns, dtype=object)
        return out

    def _orders_matrix(self, orders: List[List], columns: List) -> List[List[str]]:
        return [[str(c) for c in o] for o in orders]

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
        original_columns = [str(c) for c in df.columns]
        work = df.copy()
        work.columns = original_columns
        n, m = len(work), len(original_columns)
        if m == 0:
            return df.copy(), []
        if n == 0:
            return df.copy(), [[] for _ in range(0)]

        # Honor col_merge: keep merged columns adjacent (positioning only,
        # stored values are never altered).
        merge_groups = []
        merged_cols = set()
        for grp in (col_merge or []):
            g = [str(c) for c in grp if str(c) in original_columns]
            if g:
                merge_groups.append(g)
                merged_cols.update(g)

        # Column pools: recursion columns = all columns; merged blocks act as units
        # for ordering purposes but remain separate stored columns.
        units = []
        for g in merge_groups:
            units.append(list(g))
        for c in original_columns:
            if c not in merged_cols:
                units.append([c])

        info = self._factorize_columns(work)

        # Candidate A: global frequency-ranked unit order
        unit_order_a = self._order_units(work, info, units, mode="freq")
        orders_a = self._expand_unit_orders(unit_order_a, n)

        # Candidate B: conditional partition tree over units
        try:
            orders_b = self._partition_units(work, info, units, n)
        except Exception:
            orders_b = orders_a

        # Candidate C: length-first unit order
        unit_order_c = self._order_units(work, info, units, mode="len")
        orders_c = self._expand_unit_orders(unit_order_c, n)

        candidates = [("a", orders_a), ("b", orders_b), ("c", orders_c)]
        best_orders, best_score = None, -1
        total_chars = 0
        for name, orders in candidates:
            strs = self._serialize_with_orders(work, orders, units)
            total_chars = sum(len(s) for s in strs)
            score = self._trie_reuse(strs)
            if score > best_score:
                best_score, best_orders = score, orders

        # Build final dataframe with per-row orders (full column names)
        out = self._apply_orders_full(work, best_orders)
        col_orderings = [list(o) for o in best_orders]

        # Ensure object dtype for mixed types
        out = out.astype(object) if out.dtypes.apply(lambda d: d == object).any() else out
        return out, col_orderings

    # ---------------- unit-level helpers ---------------- #
    def _unit_stats(self, work, info, unit: List):
        """Aggregate stats over the columns of a unit (serialized concatenation)."""
        total_w = 0
        total_len = 0
        for c in unit:
            d = info[c]
            total_w += d["weight"]
            total_len += int(d["vlen"].sum())
        return total_w, total_len

    def _order_units(self, work, info, units, mode="freq"):
        n = len(work)
        scored = []
        for u in units:
            w, lsum = self._unit_stats(work, info, u)
            avg = lsum / n if n else 0.0
            if mode == "freq":
                key = (-w, -avg, tuple(str(c) for c in u))
            else:
                key = (-avg, -w, tuple(str(c) for c in u))
            scored.append((key, u))
        scored.sort(key=lambda x: x[0])
        return [u for _, u in scored]

    def _expand_unit_orders(self, unit_order, n) -> List[List[str]]:
        flat = []
        for u in unit_order:
            flat.extend(u)
        return [list(flat) for _ in range(n)]

    def _partition_units(self, work, info, units, n,
                         max_depth: int = 8, min_group: int = 8,
                         max_candidates: int = 12) -> List[List[str]]:
        orders = [None] * n
        unit_used = [False] * len(units)
        rows = np.arange(n)

        def unit_codes(unit):
            # combined serialized codes for the unit's columns
            parts = []
            for c in unit:
                parts.append(info[c]["codes"])
            if len(parts) == 1:
                return parts[0]
            # combine tuple of codes into a single code array
            stacked = np.stack(parts, axis=1)
            # factorize rows of the stacked codes
            _, inv = np.unique(stacked, axis=0, return_inverse=True)
            return np.asarray(inv, dtype=np.int64)

        unit_code_cache = [unit_codes(u) for u in units]

        def build(rws: np.ndarray, prefix: List, depth: int):
            if len(rws) == 0:
                return
            remaining = [i for i in range(len(units)) if not unit_used[i]]
            if (depth >= max_depth or len(rws) < min_group or len(remaining) <= 1):
                tail_units = self._order_units_subset(work, info, units, remaining, rws)
                full = prefix + tail_units
                for r in rws:
                    orders[r] = full
                return
            # score each remaining unit on these rows
            cand = []
            for i in remaining:
                codes = unit_code_cache[i][rws]
                cnt = np.bincount(codes)
                cnt = cnt[cnt > 0]
                if len(cnt) == 0:
                    w = 0
                    vl = 1
                else:
                    # approximating unit value length by mean column length sum
                    lsum = sum(int(info[c]["vlen"].sum()) for c in units[i])
                    avg = lsum / n if n else 0.0
                    w = int((cnt * np.maximum(cnt - 1, 0)).sum() * avg)
                cand.append((w, i))
            cand.sort(key=lambda x: (-x[0], x[1]))
            best_w, best_i = cand[0]
            if best_w <= 0:
                tail_units = self._order_units_subset(work, info, units, remaining, rws)
                full = prefix + tail_units
                for r in rws:
                    orders[r] = full
                return
            unit_used[best_i] = True
            codes = unit_code_cache[best_i][rws]
            groups = {}
            for r, code in zip(rws, codes):
                groups.setdefault(int(code), []).append(int(r))
            for k in sorted(groups.keys()):
                build(np.array(groups[k], dtype=np.int64), prefix + units[best_i], depth + 1)
            unit_used[best_i] = False

        build(rows, [], 0)
        fallback = self._expand_unit_orders(self._order_units(work, info, units), n)
        for i in range(n):
            if orders[i] is None:
                orders[i] = fallback[i]
        return orders

    def _order_units_subset(self, work, info, units, idxs, rws):
        if not idxs:
            return []
        sub = [units[i] for i in idxs]
        scored = []
        for u in sub:
            w = 0
            lsum = 0
            for c in u:
                d = info[c]
                codes = d["codes"][rws]
                cnt = np.bincount(codes)
                cnt = cnt[cnt > 0]
                w += int((d["vlen"][:len(cnt)] * cnt * np.maximum(cnt - 1, 0)).sum()) if len(cnt) else 0
                lsum += int(d["vlen"].sum())
            key = (-w, -lsum, tuple(str(c) for c in u))
            scored.append((key, u))
        scored.sort(key=lambda x: x[0])
        return [u for _, u in scored]

    # ---------------- serialization with orders ---------------- #
    def _serialize_with_orders(self, work, orders, units) -> List[str]:
        # cached cell strings per column
        cache = {c: work[c].map(self._cell_str).tolist() for c in work.columns}
        n = len(work)
        out = []
        for r in range(n):
            parts = []
            for c in orders[r]:
                parts.append(cache[c][r])
            out.append("".join(parts))
        return out

    def _apply_orders_full(self, work, orders) -> pd.DataFrame:
        n = len(work)
        cols = [str(c) for c in work.columns]
        if n == 0:
            return work.copy()
        vals = {c: work[c].tolist() for c in cols}
        data = []
        for r in range(n):
            row = [vals[c][r] for c in orders[r]]
            data.append(row)
        out = pd.DataFrame(data, columns=cols)
        return out

# EVOLVE-BLOCK-END