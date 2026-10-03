# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict, Optional
from collections import Counter, defaultdict


class Evolved(Algorithm):
    """
    Serial character-Trie prefix-cache aware reordering.

    Objective: maximize total matched characters of "".join(row.fillna("").astype(str))
    under an exact serial Trie, which equals sum of adjacent LCPs of the sorted
    serialized row strings for a fixed multiset of strings. We construct a small
    number of cheap candidate per-row column orderings, score each with the exact
    LCP objective, and return the best. All cell values are preserved exactly.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------
    # serialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cell_to_str(v) -> str:
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
        return str(v)

    def _serialize_columns(self, df: pd.DataFrame) -> List[List[str]]:
        """Return, per column, a list of serialized strings (one per row)."""
        out = []
        for col in df.columns:
            out.append([self._cell_to_str(v) for v in df[col].tolist()])
        return out

    @staticmethod
    def _lcp_sum(strings: List[str]) -> int:
        """Exact ideal Trie reuse: sum of adjacent LCPs of sorted strings."""
        n = len(strings)
        if n < 2:
            return 0
        s = sorted(strings)
        total = 0
        prev = s[0]
        for cur in s[1:]:
            if cur == prev:
                total += len(cur)
            else:
                lo = 0
                hi = min(len(prev), len(cur))
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if prev[:mid] == cur[:mid]:
                        lo = mid
                    else:
                        hi = mid - 1
                total += lo
            prev = cur
        return total

    # ------------------------------------------------------------------
    # candidate constructions
    # ------------------------------------------------------------------
    def _global_scores(self, col_strs: List[List[str]]):
        """score1 = sum(len(v)*c*(c-1)); score2 = sum(c*(c-1)) (freq-only alt)."""
        score1 = []
        score2 = []
        for strs in col_strs:
            c = Counter(strs)
            s1 = 0
            s2 = 0
            for v, cnt in c.items():
                if cnt > 1:
                    s1 += len(v) * cnt * (cnt - 1)
                    s2 += cnt * (cnt - 1)
            score1.append(s1)
            score2.append(s2)
        return score1, score2

    def _conditional_orders(
        self,
        n_rows: int,
        col_strs: List[List[str]],
        avail: List[int],
        rank_key,
        score1: List[int],
        distinct_ok: List[bool],
        max_depth: int,
        min_group: int = 4,
        max_cands: int = 12,
    ) -> List[List[int]]:
        """Recursive conditional prefix partition; returns per-row column orders."""
        base = sorted(avail, key=rank_key)

        def build(rows: List[int], avail_set: frozenset, depth: int):
            if not avail_set or depth >= max_depth or len(rows) < min_group:
                rest = sorted(avail_set, key=rank_key)
                return {r: rest for r in rows}
            # candidate partition columns: top by global score, eligible distinctness
            cands = [c for c in sorted(avail_set, key=rank_key) if distinct_ok[c]]
            cands = cands[:max_cands]
            best_col = -1
            best_gain = 0
            for c in cands:
                groups = defaultdict(list)
                colvals = col_strs[c]
                for r in rows:
                    groups[colvals[r]].append(r)
                gain = 0
                for g_rows in groups.values():
                    cnt = len(g_rows)
                    if cnt > 1:
                        gain += len(next(iter(
                            None for _ in [0])) or "") * 0  # placeholder, replaced below
                # recompute gain properly
                gain = 0
                for v, g_rows in groups.items():
                    cnt = len(g_rows)
                    if cnt > 1:
                        gain += len(v) * cnt * (cnt - 1)
                if gain > best_gain:
                    best_gain = gain
                    best_col = c
            if best_col < 0:
                rest = sorted(avail_set, key=rank_key)
                return {r: rest for r in rows}
            groups = defaultdict(list)
            colvals = col_strs[best_col]
            for r in rows:
                groups[colvals[r]].append(r)
            result = {}
            nxt = avail_set - {best_col}
            for g_rows in groups.values():
                sub = build(g_rows, nxt, depth + 1)
                for r, order in sub.items():
                    result[r] = [best_col] + order
            return result

        orders_map = build(list(range(n_rows)), frozenset(avail), 0)
        return [orders_map[r] for r in range(n_rows)]

    # ------------------------------------------------------------------
    # constraint enforcement (col_merge adjacency, one_way_dep ordering)
    # ------------------------------------------------------------------
    @staticmethod
    def _fix_order(order: List[int], merge_groups: List[List[int]], dep_pairs: List[Tuple[int, int]]) -> List[int]:
        order = list(order)
        for group in merge_groups:
            g = [c for c in group if c in order]
            if len(g) > 1:
                pos = order.index(g[0])
                order = [c for c in order if c not in g[1:]]
                pos = order.index(g[0])
                order = order[: pos + 1] + g[1:] + order[pos + 1 :]
        for a, b in dep_pairs:
            if a in order and b in order:
                ia = order.index(a)
                ib = order.index(b)
                if ib < ia:
                    order.pop(ib)
                    order.insert(order.index(a) + 1, b)
        return order

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
        original = df.copy()
        n_rows, n_cols = original.shape
        col_names = list(original.columns)

        if n_rows == 0 or n_cols == 0:
            return original, [list(col_names) for _ in range(n_rows)]

        name_to_idx = {c: i for i, c in enumerate(col_names)}
        merge_groups = []
        for group in (col_merge or []):
            g = [name_to_idx[c] for c in group if c in name_to_idx]
            if len(g) > 1:
                merge_groups.append(g)
        dep_pairs = []
        for dep in (one_way_dep or []):
            if len(dep) >= 2 and dep[0] in name_to_idx and dep[1] in name_to_idx:
                dep_pairs.append((name_to_idx[dep[0]], name_to_idx[dep[1]]))

        # serialized cells (shared across all candidate constructions / scoring)
        col_strs = self._serialize_columns(original)
        score1, score2 = self._global_scores(col_strs)

        all_cols = list(range(n_cols))

        # --- candidate A: global length-weighted repetition ordering
        orderA = sorted(all_cols, key=lambda c: (-score1[c], c))
        orderA = self._fix_order(orderA, merge_groups, dep_pairs)
        candA = [orderA] * n_rows

        # --- candidate C: frequency-first alternative tradeoff
        orderC = sorted(all_cols, key=lambda c: (-score2[c], -score1[c], c))
        orderC = self._fix_order(orderC, merge_groups, dep_pairs)
        candC = [orderC] * n_rows

        # --- candidate B: bounded conditional partition tree
        max_depth = col_stop if col_stop else min(n_cols, 8)
        if row_stop is not None:
            max_depth = min(max_depth, max(1, int(row_stop)))
        # columns eligible as partition fields: not overly distinct
        limit = n_rows * (distinct_value_threshold if distinct_value_threshold is not None else 0.8)
        distinct_ok = []
        for c in all_cols:
            distinct_ok.append(len(set(col_strs[c])) <= max(limit, 1.0))
        rank_key = lambda c: (-score1[c], c)
        try:
            candB = self._conditional_orders(
                n_rows, col_strs, all_cols, rank_key, score1, distinct_ok,
                max_depth=max_depth,
            )
            candB = [self._fix_order(o, merge_groups, dep_pairs) for o in candB]
        except Exception:
            candB = candA

        # --- score candidates with the exact serial-Trie objective
        def row_strings(orders: List[List[int]]) -> List[str]:
            return [
                "".join(col_strs[c][r] for c in orders[r]) for r in range(n_rows)
            ]

        candidates = [candA, candB, candC]
        # dedupe identical candidates
        seen = set()
        unique = []
        for cand in candidates:
            key = tuple(map(tuple, cand[: min(n_rows, 64)]))
            if key not in seen:
                seen.add(key)
                unique.append(cand)

        best_orders = candA
        best_score = -1
        for cand in unique:
            try:
                sc = self._lcp_sum(row_strings(cand))
            except Exception:
                sc = -1
            if sc > best_score:
                best_score = sc
                best_orders = cand

        # --- build output
        orderings_names = [[col_names[c] for c in order] for order in best_orders]

        # modal ordering for the DataFrame's physical column layout (hedge:
        # if a consumer serializes the df directly, use the most common order)
        cnt = Counter(tuple(order) for order in best_orders)
        modal = list(cnt.most_common(1)[0][0])
        modal_names = [col_names[c] for c in modal]
        out_df = original[modal_names].copy()
        if out_df.shape != original.shape:
            out_df = original.copy()
            modal_names = list(col_names)

        # preserve dtypes / mixed types with object dtype when mixed
        if any(str(original[c].dtype) == "object" for c in original.columns):
            try:
                out_df = out_df.astype(object)
            except Exception:
                pass

        return out_df, orderings_names

# EVOLVE-BLOCK-END