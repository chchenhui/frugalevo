# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List
from collections import Counter, defaultdict


class Evolved(Algorithm):
    """
    Column/row reordering optimized for character-Trie prefix caching.

    Objective: for a fixed multiset of serialized row strings, total reuse is
    the sum of adjacent LCP lengths after sorting the strings. We construct a
    few cheap candidate per-row column orderings and select the best via this
    exact (bounded) objective.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------ #
    # serialization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Longest common prefix via binary search on slices (C-speed compare)."""
        if a == b:
            return len(a)
        hi = len(a) if len(a) < len(b) else len(b)
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    @classmethod
    def _trie_reuse(cls, strings) -> int:
        """Exact ideal trie reuse: sum of adjacent LCPs over sorted strings."""
        s = sorted(strings)
        total = 0
        prev = None
        for cur in s:
            if prev is not None:
                total += cls._lcp(prev, cur)
            prev = cur
        return total

    @staticmethod
    def _serialize_columns(df: pd.DataFrame):
        """Per-column list of serialized strings, matching the evaluator:
        ''.join(row.fillna('').astype(str).values) with no separators."""
        ser = []
        for c in df.columns:
            try:
                col = df[c].fillna("").astype(str)
            except Exception:
                col = df[c].map(lambda v: "" if pd.isna(v) else str(v))
            ser.append(col.tolist())
        return ser

    # ------------------------------------------------------------------ #
    # block construction (merged / dependent columns stay adjacent)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_blocks(n_cols, col_index, col_merge, one_way_dep):
        parent = list(range(n_cols))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for group in (col_merge or []):
            idxs = [col_index[c] for c in group if c in col_index]
            for j in idxs[1:]:
                union(idxs[0], j)
        for a, b in (one_way_dep or []):
            if a in col_index and b in col_index:
                union(col_index[a], col_index[b])

        groups = defaultdict(list)
        for i in range(n_cols):
            groups[find(i)].append(i)
        # blocks in original column order, each block internally in original order
        blocks = [sorted(v) for _, v in sorted(groups.items(), key=lambda kv: min(kv[1]))]
        return blocks

    # ------------------------------------------------------------------ #
    # candidate constructions
    # ------------------------------------------------------------------ #
    @staticmethod
    def _expand(blocks_order):
        out = []
        for b in blocks_order:
            out.extend(b)
        return out

    def _global_block_order(self, blocks, block_weight):
        return sorted(blocks, key=lambda b: (-block_weight[id(b)], min(b)))

    def _candidate_global(self, n_rows, blocks, col_weight):
        block_weight = {id(b): sum(col_weight[c] for c in b) for b in blocks}
        go = self._global_block_order(blocks, block_weight)
        order = self._expand(go)
        return [list(order) for _ in range(n_rows)]

    def _candidate_recursive(
        self, ser, blocks, col_weight, n_rows, max_depth=10, work_budget=4_000_000
    ):
        n_cols = sum(len(b) for b in blocks)
        block_weight = {id(b): sum(col_weight[c] for c in b) for b in blocks}
        global_order = self._global_block_order(blocks, block_weight)
        col_to_block = {}
        for bi, b in enumerate(blocks):
            for c in b:
                col_to_block[c] = bi

        # per-column value -> serialized length
        strlen = []
        for c in range(n_cols):
            d = {}
            for s in ser[c]:
                if s not in d:
                    d[s] = len(s)
            strlen.append(d)

        orders = [None] * n_rows
        all_blocks = list(range(len(blocks)))

        def rec(ridx, rem_blocks, depth, work):
            if (
                len(ridx) < 2
                or not rem_blocks
                or depth >= max_depth
                or work > work_budget
            ):
                rest = [blocks[bi] for bi in global_order if bi in rem_blocks]
                flat = self._expand(rest)
                for i in ridx:
                    orders[i] = list(flat)
                return

            rem_cols = [c for bi in rem_blocks for c in blocks[bi]]
            # estimated savings per column: sum cnt*(cnt-1)*len(str)
            best_col, best_sav = -1, 0
            for c in rem_cols:
                cnt = Counter(ser[c][i] for i in ridx)
                sav = 0
                for v, k in cnt.items():
                    if k > 1:
                        sav += k * (k - 1) * strlen[c].get(v, len(v))
                if sav > best_sav:
                    best_sav, best_col = sav, c

            if best_col < 0 or best_sav <= 0:
                rest = [blocks[bi] for bi in global_order if bi in rem_blocks]
                flat = self._expand(rest)
                for i in ridx:
                    orders[i] = list(flat)
                return

            bi = col_to_block[best_col]
            new_rem = [b for b in rem_blocks if b != bi]
            head = blocks[bi]
            groups = defaultdict(list)
            for i in ridx:
                groups[ser[best_col][i]].append(i)
            w2 = work + len(ridx) * max(1, len(rem_cols))
            if len(groups) == 1:
                rest = [blocks[b] for b in global_order if b in new_rem]
                flat = head + self._expand(rest)
                for i in ridx:
                    orders[i] = list(flat)
                return
            for _, g in groups.items():
                if len(g) < 2:
                    rest = [blocks[b] for b in global_order if b in new_rem]
                    flat = head + self._expand(rest)
                    for i in g:
                        orders[i] = list(flat)
                    continue
                rec(g, new_rem, depth + 1, w2)
                if orders[g[0]] is not None:
                    pre = head
                    for i in g:
                        orders[i] = pre + orders[i]

        rec(list(range(n_rows)), all_blocks, 0, 0)

        # fallback for any unassigned row
        flat = self._expand(global_order)
        for i in range(n_rows):
            if orders[i] is None:
                orders[i] = list(flat)
        return orders

    def _candidate_length_bias(self, n_rows, blocks, ser, n_cols):
        # rank blocks by total serialized characters descending
        def block_key(b):
            tot = 0
            for c in b:
                tot += sum(len(s) for s in ser[c])
            return (-tot, min(b))

        go = sorted(blocks, key=block_key)
        flat = self._expand(go)
        return [list(flat) for _ in range(n_rows)]

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: int = None,
        col_stop: int = None,
        col_merge=[],
        one_way_dep=[],
        distinct_value_threshold: float = 0.8,
        parallel: bool = True,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        df = df.copy()
        n_rows, n_cols = df.shape
        col_names = list(df.columns)
        col_index = {c: i for i, c in enumerate(col_names)}

        if n_rows == 0 or n_cols == 0:
            return df.copy(), [[] for _ in range(n_rows)]

        ser = self._serialize_columns(df)

        # column weight: sum over unique values of len(v) * count * (count-1)
        col_weight = []
        for c in range(n_cols):
            cnt = Counter(ser[c])
            col_weight.append(
                sum(len(v) * k * (k - 1) for v, k in cnt.items())
            )

        blocks = self._build_blocks(n_cols, col_index, col_merge, one_way_dep)

        # candidate constructions (bounded to three)
        candidates = []
        candidates.append(("global", self._candidate_global(n_rows, blocks, col_weight)))
        try:
            candidates.append(
                ("recursive", self._candidate_recursive(ser, blocks, col_weight, n_rows))
            )
        except Exception:
            pass
        candidates.append(
            ("length", self._candidate_length_bias(n_rows, blocks, ser, n_cols))
        )

        # serialize rows per candidate and score with exact trie objective;
        # bound total scoring work for very large datasets
        total_chars = sum(sum(len(s) for s in col) for col in ser) * n_cols
        budget_ok = total_chars * max(1, n_rows // max(1, n_cols)) <= 200_000_000

        best_orders = None
        best_score = -1
        for name, orders in candidates:
            if not budget_ok and best_orders is not None:
                break
            try:
                row_strs = [
                    "".join(ser[j][i] for j in orders[i]) for i in range(n_rows)
                ]
                score = self._trie_reuse(row_strs)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_orders = orders

        if best_orders is None:
            best_orders = candidates[0][1]

        # build output: same shape, same index, same cell values (object dtype),
        # each row permuted according to its per-row ordering
        raw = df.to_numpy(dtype=object)
        data = []
        for i in range(n_rows):
            data.append([raw[i][j] for j in best_orders[i]])

        out = pd.DataFrame(data, index=df.index, columns=col_names)
        out = out.astype(object)

        column_orderings = [[col_names[j] for j in best_orders[i]] for i in range(n_rows)]

        assert out.shape == df.shape
        return out, column_orderings

# EVOLVE-BLOCK-END