# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict
from collections import defaultdict


class Evolved(Algorithm):
    """
    Trie-oriented row-wise column reordering.

    Selects among a small bounded set of per-row column-ordering candidates by
    measuring the exact ideal character-Trie reuse (sum of adjacent LCPs of the
    sorted serialized rows), while never modifying stored cell values.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.dep_graph = None
        self.num_rows = 0
        self.num_cols = 0
        self.column_stats = None
        self.val_len = None
        self.row_stop = None
        self.col_stop = None
        self.base = 2000

    # ------------------------------------------------------------------ #
    # serialization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ser_series(col: pd.Series) -> np.ndarray:
        # exactly match the evaluator's scoring serialization:
        # missing -> "", everything else -> str(value)
        obj = col.astype(object)
        mask = col.isna().values if hasattr(col, "isna") else None
        if mask is None:
            mask = pd.isna(obj.values)
        out = np.empty(len(obj), dtype=object)
        vals = obj.values
        for i in range(len(vals)):
            v = vals[i]
            out[i] = "" if (v is None or (v is not v and isinstance(v, float)) or (mask[i] if mask is not None else False)) else str(v)
        return out

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        if a == b:
            return len(a)
        hi = min(len(a), len(b))
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    @classmethod
    def _trie_reuse(cls, strings: List[str]) -> int:
        if len(strings) <= 1:
            return 0
        ss = sorted(strings)
        total = 0
        prev = ss[0]
        for s in ss[1:]:
            total += cls._lcp(prev, s)
            prev = s
        return total

    # ------------------------------------------------------------------ #
    # candidate constructions
    # ------------------------------------------------------------------ #
    def _global_order(self, scores: List[float]) -> List[int]:
        return sorted(range(len(scores)), key=lambda j: (-scores[j], j))

    def _conditional_orders(
        self,
        row_ids: List[int],
        avail: frozenset,
        codes: List[np.ndarray],
        scores: List[float],
        depth: int,
        max_depth: int,
        cand_limit: int,
    ) -> Dict[int, List[int]]:
        base = sorted(avail, key=lambda j: (-scores[j], j))
        if not avail or len(row_ids) <= 1 or depth >= max_depth:
            return {r: list(base) for r in row_ids}

        # bound candidate columns on wide tables
        cands = sorted(avail, key=lambda j: (-scores[j], j))[:cand_limit]
        best_j = None
        best_gain = -1.0
        for j in cands:
            counts: Dict[int, int] = defaultdict(int)
            cj = codes[j]
            for r in row_ids:
                counts[cj[r]] += 1
            gain = 0.0
            for c, cnt in counts.items():
                if cnt > 1:
                    gain += self._val_len_cache[j].get(c, 0) * cnt * (cnt - 1)
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_j = j
        if best_j is None or best_gain <= 0:
            return {r: list(base) for r in row_ids}

        groups: Dict[int, List[int]] = defaultdict(list)
        cb = codes[best_j]
        for r in row_ids:
            groups[cb[r]].append(r)
        remaining = avail - {best_j}
        result: Dict[int, List[int]] = {}
        for g in groups.values():
            sub = self._conditional_orders(
                g, remaining, codes, scores, depth + 1, max_depth, cand_limit
            )
            for r, order in sub.items():
                result[r] = [best_j] + order
        return result

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
        initial_df = df.copy()

        # honor column merges using existing API semantics
        if col_merge:
            try:
                stats = self.calculate_col_stats(df, enable_index=True)
                reordered_columns = [col for col, _, _, _ in stats]
                for col_to_merge in col_merge:
                    final_col_order = [c for c in reordered_columns if c in col_to_merge]
                    if len(final_col_order) > 1:
                        df = self.merging_columns(df, final_col_order, prepended=False)
            except Exception:
                # if merging infrastructure is unavailable, keep data as-is
                pass

        n_rows, n_cols = df.shape
        if n_rows == 0 or n_cols == 0:
            return df.copy().astype(object), [[] for _ in range(n_rows)] if n_cols == 0 else [[c for c in df.columns] for _ in range(n_rows)]

        col_names = list(df.columns)

        # per-column serialized strings + integer codes
        ser_cols: List[np.ndarray] = []
        codes: List[np.ndarray] = []
        for c in col_names:
            s = self._ser_series(df[c])
            ser_cols.append(s)
            uniq, code = pd.factorize(s, sort=False)
            codes.append(np.asarray(code))
        # length-weighted pair repetition score per column
        scores: List[float] = []
        self._val_len_cache: List[Dict[int, int]] = []
        for j in range(n_cols):
            codes_arr = codes[j]
            uniq_vals = pd.unique(ser_cols[j])
            uidx = {v: i for i, v in enumerate(uniq_vals)}
            # map codes -> unique value index
            try:
                mapping = np.array([uidx[v] for v in pd.unique(ser_cols[j])], dtype=np.int64)
            except Exception:
                mapping = None
            lens = np.array([len(v) for v in uniq_vals], dtype=np.float64)
            cnts = np.bincount(
                np.array([uidx[v] for v in ser_cols[j]], dtype=np.int64),
                minlength=len(uniq_vals),
            )
            scores.append(float(np.sum(lens * cnts * np.maximum(cnts - 1, 0))))
            if mapping is not None:
                self._val_len_cache.append(
                    {i: int(lens[i]) for i in range(len(uniq_vals))}
                )
            else:
                self._val_len_cache.append({})

        # map code values to unique-value index for group length lookups
        # (codes[j][r] indexes into the column's unique value list)
        # rebuild val_len_cache keyed by code:
        self._val_len_cache = []
        for j in range(n_cols):
            uniq_vals = pd.unique(ser_cols[j])
            lens = {i: len(v) for i, v in enumerate(uniq_vals)}
            self._val_len_cache.append(lens)

        all_rows = list(range(n_rows))

        candidates: List[List[List[int]]] = []

        # candidate 1: global frequency-ranked order
        go = self._global_order(scores)
        candidates.append([list(go) for _ in range(n_rows)])

        # candidate 2: recursive conditional grouping
        max_depth = 12
        cand_limit = 24
        try:
            cond = self._conditional_orders(
                all_rows,
                frozenset(range(n_cols)),
                codes,
                scores,
                0,
                max_depth,
                cand_limit,
            )
            orders2 = [cond.get(r, list(go)) for r in all_rows]
        except Exception:
            orders2 = [list(go) for _ in range(n_rows)]
        candidates.append(orders2)

        # candidate 3: identity order (safe fallback)
        candidates.append([[j for j in range(n_cols)] for _ in range(n_rows)])

        def serialize(orders: List[List[int]]) -> List[str]:
            out = []
            for i in range(n_rows):
                out.append("".join(ser_cols[j][i] for j in orders[i]))
            return out

        best_orders = candidates[0]
        best_score = self._trie_reuse(serialize(best_orders))
        for cand in candidates[1:]:
            try:
                sc = self._trie_reuse(serialize(cand))
            except Exception:
                continue
            if sc > best_score:
                best_score = sc
                best_orders = cand

        # build output preserving every original cell value and row identity
        raw = df.astype(object).values
        out_arr = np.empty((n_rows, n_cols), dtype=object)
        for i in range(n_rows):
            order = best_orders[i]
            row = raw[i]
            for pos, j in enumerate(order):
                out_arr[i, pos] = row[j]

        result_df = pd.DataFrame(out_arr, columns=col_names, index=df.index)
        result_df = result_df.astype(object)

        column_orderings = [
            [col_names[j] for j in best_orders[i]] for i in range(n_rows)
        ]

        assert result_df.shape == df.shape
        return result_df, column_orderings

    # ------------------------------------------------------------------ #
    # retained compatibility helpers (unused slow paths removed)
    # ------------------------------------------------------------------ #
    def fixed_reorder(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[List[str]]]:
        order = list(df.columns)
        return df.astype(object).copy(), [list(order) for _ in range(len(df))]

    def calculate_length(self, value):
        if isinstance(value, bool):
            return 4
        if isinstance(value, (int, float, str)):
            return len(str(value))
        return 0

# EVOLVE-BLOCK-END