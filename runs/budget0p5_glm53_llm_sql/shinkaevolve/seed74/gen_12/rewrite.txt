# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict
from collections import Counter
import networkx as nx


class Evolved(Algorithm):
    """
    Trie-prefix-reuse oriented reordering.

    Strategy (bounded construction selection):
      1. Serialize every cell exactly as the evaluator does (missing -> "").
      2. Build up to three candidate per-row column orderings:
         A) global order ranked by sum(len(v) * count(v) * (count(v)-1))
         B) global order ranked by sum(len(v) * (count(v)-1))
         C) a bounded conditional partition tree: inside each group pick the
            remaining column with the largest length-weighted pair repetition,
            partition rows on its value, recurse (depth <= 3, groups >= 4 rows),
            with a deterministic cheap tail.
      3. Score each candidate with the exact Trie objective: sum of adjacent
         LCP lengths over the sorted serialized row strings.
      4. Return the best candidate. Data values are never modified.
    """

    MAX_SCORED_CHARS = 3_000_000  # skip expensive scoring on very large data
    COND_DEPTH = 3
    COND_MIN_GROUP = 4

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.dep_graph = None

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #
    def _serialize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of str cells, with missing values as ''."""
        obj = df.astype(object)
        mask = df.isna()
        try:
            s = obj.astype(str)
        except Exception:
            s = pd.DataFrame(
                [["" if pd.isna(v) else str(v) for v in row]
                 for row in obj.itertuples(index=False, name=None)],
                index=df.index, columns=df.columns,
            )
        s = s.where(~mask, "")
        return s

    def _lcp_sum(self, strings: List[str]) -> int:
        """Exact Trie reuse: sum of adjacent LCPs after sorting."""
        if not strings:
            return 0
        ss = sorted(strings)
        total = 0
        for a, b in zip(ss, ss[1:]):
            if a == b:
                total += len(a)
                continue
            lo, hi = 0, min(len(a), len(b))
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if a[:mid] == b[:mid]:
                    lo = mid
                else:
                    hi = mid - 1
            total += lo
        return total

    # ------------------------------------------------------------------ #
    # Column scoring
    # ------------------------------------------------------------------ #
    def _column_scores(self, ser: pd.DataFrame):
        """Per-column scores: s1 = sum len*c*(c-1), s2 = sum len*(c-1)."""
        s1: Dict = {}
        s2: Dict = {}
        for col in ser.columns:
            cnt = Counter(ser[col].tolist())
            a = 0
            b = 0
            for v, n in cnt.items():
                if n > 1:
                    a += len(v) * n * (n - 1)
                    b += len(v) * (n - 1)
            s1[col] = a
            s2[col] = b
        return s1, s2

    def _global_order(self, cols, score: Dict) -> List:
        return sorted(cols, key=lambda c: (-score.get(c, 0), str(c)))

    # ------------------------------------------------------------------ #
    # Candidate C: conditional partition tree (per-row orders)
    # ------------------------------------------------------------------ #
    def _conditional_orders(self, ser: pd.DataFrame) -> List[List]:
        n = len(ser)
        cols = list(ser.columns)
        orders: List[List] = [None] * n
        if n == 0 or not cols:
            return [list(cols) for _ in range(n)]
        self._arr = {c: ser[c].to_numpy(dtype=object) for c in cols}
        self._cond_rec(np.arange(n), cols, [], orders, 0)
        for i in range(n):
            if orders[i] is None:
                orders[i] = list(cols)
        return orders

    def _cond_rec(self, row_idx: np.ndarray, cols: List, prefix: List,
                  orders: List[List], depth: int) -> None:
        if len(cols) <= 1 or depth >= self.COND_DEPTH or len(row_idx) < self.COND_MIN_GROUP:
            tail = self._global_order(cols, self._s1)
            full = prefix + tail
            for i in row_idx.tolist():
                orders[i] = list(full)
            return

        best_col = None
        best_gain = -1
        for c in cols:
            vals = self._arr[c][row_idx]
            try:
                u, cnt = np.unique(vals, return_counts=True)
            except Exception:
                u = np.array(list(dict.fromkeys(vals.tolist())), dtype=object)
                cnt = np.array([np.count_nonzero(vals == v) for v in u])
            g = 0
            for v, k in zip(u.tolist(), cnt.tolist()):
                if k > 1:
                    g += len(str(v)) * k * (k - 1)
            if g > best_gain or (g == best_gain and best_col is not None
                                 and str(c) < str(best_col)):
                best_col, best_gain = c, g

        if best_col is None or best_gain <= 0:
            tail = self._global_order(cols, self._s1)
            full = prefix + tail
            for i in row_idx.tolist():
                orders[i] = list(full)
            return

        rest = [c for c in cols if c != best_col]
        bc_arr = self._arr[best_col]
        groups: Dict = {}
        for i in row_idx.tolist():
            groups.setdefault(bc_arr[i], []).append(i)
        new_prefix = prefix + [best_col]
        for _v, idxs in sorted(groups.items(), key=lambda kv: str(kv[0])):
            self._cond_rec(np.array(idxs, dtype=np.int64), rest,
                            new_prefix, orders, depth + 1)

    # ------------------------------------------------------------------ #
    # Candidate scoring
    # ------------------------------------------------------------------ #
    def _row_strings_global(self, ser: pd.DataFrame, order: List) -> List[str]:
        parts = [ser[c].to_numpy(dtype=object) for c in order]
        n = len(ser)
        return ["".join(str(parts[j][i]) for j in range(len(parts))) for i in range(n)]

    def _row_strings_per_row(self, ser: pd.DataFrame, orders: List[List]) -> List[str]:
        arrs = {c: ser[c].to_numpy(dtype=object) for c in ser.columns}
        out = []
        for i, order in enumerate(orders):
            out.append("".join(str(arrs[c][i]) for c in order))
        return out

    # ------------------------------------------------------------------ #
    # Dependency handling (hint only; keeps orderings valid)
    # ------------------------------------------------------------------ #
    def _apply_dep_order(self, order: List) -> List:
        if self.dep_graph is None:
            return list(order)
        order = list(order)
        for a, b in self._dep_pairs:
            if a in order and b in order:
                ia, ib = order.index(a), order.index(b)
                if ia > ib:
                    order.pop(ia)
                    order.insert(order.index(b) + 1 if b in order else ib, a)
                    order = self._apply_dep_order(order)
                    break
        return order

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
        initial_df = df.copy()

        # ---- required column merges (existing API semantics) ----
        if col_merge:
            _, column_stats = self.calculate_col_stats(df, enable_index=True)
            reordered_columns = [col for col, _, _, _ in column_stats]
            for col_to_merge in col_merge:
                final_col_order = [c for c in reordered_columns if c in col_to_merge]
                if final_col_order:
                    df = self.merging_columns(df, final_col_order, prepended=False)

        # ---- one-way dependencies (hint) ----
        self.dep_graph = None
        self._dep_pairs = []
        if one_way_dep:
            self.dep_graph = nx.DiGraph()
            for dep in one_way_dep:
                try:
                    col1 = [c for c in df.columns if dep[0] in str(c)]
                    col2 = [c for c in df.columns if dep[1] in str(c)]
                    if len(col1) == 1 and len(col2) == 1:
                        self.dep_graph.add_edge(col1[0], col2[0])
                        self._dep_pairs.append((col1[0], col2[0]))
                except Exception:
                    pass

        n = len(df)
        cols = list(df.columns)

        # Trivial cases
        if n == 0 or not cols:
            return df.copy(), [list(cols) for _ in range(n)]

        ser = self._serialize_frame(df)
        self._s1, self._s2 = self._column_scores(ser)

        order_a = self._apply_dep_order(self._global_order(cols, self._s1))
        order_b = self._apply_dep_order(self._global_order(cols, self._s2))

        total_chars = int(sum(len(v) for v in ser.values.ravel())) if n else 0

        best_orders = [list(order_a) for _ in range(n)]
        if total_chars <= self.MAX_SCORED_CHARS and n > 1:
            strings_a = self._row_strings_global(ser, order_a)
            score_a = self._lcp_sum(strings_a)
            best_score = score_a

            strings_b = self._row_strings_global(ser, order_b)
            score_b = self._lcp_sum(strings_b)
            if score_b > best_score:
                best_score = score_b
                best_orders = [list(order_b) for _ in range(n)]

            # Candidate C: conditional partition tree (per-row orders)
            try:
                cand_c = self._conditional_orders(ser)
                cand_c = [list(o) for o in cand_c]
                ok = all(sorted(o) == sorted(map(str, cols)) or
                         sorted(map(str, o)) == sorted(map(str, cols))
                         for o in cand_c[:1])
                if cand_c and ok:
                    strings_c = self._row_strings_per_row(ser, cand_c)
                    score_c = self._lcp_sum(strings_c)
                    if score_c > best_score:
                        best_score = score_c
                        best_orders = cand_c
            except Exception:
                pass
        else:
            best_orders = [list(order_a) for _ in range(n)]

        # Output: original values, original column layout untouched.
        out = df.copy()
        # Normalize mixed types safely to object where needed.
        if any(dtype == object for dtype in out.dtypes) and out.dtypes.nunique() > 1:
            out = out.astype(object)

        column_orderings = [list(o) for o in best_orders]
        assert len(column_orderings) == n

        if not col_merge:
            assert out.shape == initial_df.shape, "shape mismatch"
        else:
            assert out.shape[0] == initial_df.shape[0], "row count mismatch"

        return out, column_orderings

# EVOLVE-BLOCK-END