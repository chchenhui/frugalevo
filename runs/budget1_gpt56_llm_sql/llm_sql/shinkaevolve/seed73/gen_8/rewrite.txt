# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from collections import Counter, defaultdict
from functools import lru_cache


class Evolved(Algorithm):
    """
    Prefix-Trie-oriented column reordering.

    The DataFrame retains its original shape, index, column labels, and values.
    For row-specific layouts, ``column_orderings[row]`` describes which source
    column supplied each positional value in the returned row.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.num_rows = 0
        self.num_cols = 0

    @staticmethod
    def _cell_string(value) -> str:
        """Match evaluator serialization without changing the stored value."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, bool) and missing:
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        if left == right:
            return len(left)
        limit = min(len(left), len(right))
        lo, hi = 0, limit
        # Slice equality is implemented in C and avoids Python character loops.
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if left[:mid] == right[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _trie_score(self, strings: List[str]) -> int:
        if len(strings) < 2:
            return 0
        ordered = sorted(strings)
        return sum(self._lcp(ordered[i - 1], ordered[i])
                   for i in range(1, len(ordered)))

    def _make_constraint_normalizer(self, columns, col_merge, one_way_dep):
        """
        Build a stable normalizer.  Merged columns are treated as an adjacent
        block, while one-way dependencies place the source block before the
        destination block.  This preserves all individual cells.
        """
        ncols = len(columns)
        parent = list(range(ncols))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a

        name_to_positions = defaultdict(list)
        for i, name in enumerate(columns):
            name_to_positions[name].append(i)

        for merge_group in col_merge or []:
            positions = []
            for name in merge_group:
                positions.extend(name_to_positions.get(name, []))
            if positions:
                root = positions[0]
                for pos in positions[1:]:
                    union(root, pos)

        groups = defaultdict(list)
        for i in range(ncols):
            groups[find(i)].append(i)
        blocks = list(groups.values())
        block_for_col = {}
        for block_id, block in enumerate(blocks):
            for col_id in block:
                block_for_col[col_id] = block_id

        edges = set()
        for source_text, target_text in one_way_dep or []:
            sources = [i for i, name in enumerate(columns)
                       if source_text in str(name)]
            targets = [i for i, name in enumerate(columns)
                       if target_text in str(name)]
            for source in sources:
                for target in targets:
                    a, b = block_for_col[source], block_for_col[target]
                    if a != b:
                        edges.add((a, b))

        cache = {}

        def normalize(order):
            key = tuple(order)
            cached = cache.get(key)
            if cached is not None:
                return list(cached)

            rank = {col: i for i, col in enumerate(order)}
            block_rank = {
                block_id: min(rank.get(col, ncols + col) for col in block)
                for block_id, block in enumerate(blocks)
            }
            outgoing = defaultdict(list)
            indegree = [0] * len(blocks)
            for source, target in edges:
                outgoing[source].append(target)
                indegree[target] += 1

            available = [i for i in range(len(blocks)) if indegree[i] == 0]
            available.sort(key=lambda i: (block_rank[i], i))
            block_order = []
            while available:
                current = available.pop(0)
                block_order.append(current)
                for target in outgoing[current]:
                    indegree[target] -= 1
                    if indegree[target] == 0:
                        available.append(target)
                available.sort(key=lambda i: (block_rank[i], i))

            # A cyclic dependency cannot satisfy every edge; retain a stable,
            # complete permutation rather than dropping or duplicating columns.
            if len(block_order) != len(blocks):
                seen = set(block_order)
                block_order.extend(sorted(
                    (i for i in range(len(blocks)) if i not in seen),
                    key=lambda i: (block_rank[i], i),
                ))

            result = []
            for block_id in block_order:
                result.extend(sorted(blocks[block_id], key=lambda col: rank[col]))
            result = tuple(result)
            if len(cache) < 4096:
                cache[key] = result
            return list(result)

        return normalize

    def _global_order(self, value_strings: List[List[str]]) -> List[int]:
        scores = []
        for col_id, values in enumerate(value_strings):
            counts = Counter(values)
            score = sum(len(value) * count * (count - 1)
                        for value, count in counts.items())
            scores.append((score, col_id))
        return [col_id for _, col_id in sorted(scores, key=lambda x: (-x[0], x[1]))]

    def _conditional_orders(self, value_strings, global_order, normalize):
        """
        Bounded conditional prefix partition tree.  It uses value repetition
        weighted by character length and selects only a limited number of
        promising fields at each node.
        """
        ncols = len(value_strings)
        nrows = len(value_strings[0]) if ncols else 0
        if nrows == 0:
            return []

        candidate_limit = min(ncols, 48)
        candidates = set(global_order[:candidate_limit])
        orders = [None] * nrows
        node_budget = 128
        max_depth = min(ncols, 16)
        nodes_used = [0]

        def finish(rows, prefix, remaining):
            tail = [col for col in global_order if col in remaining]
            order = normalize(prefix + tail)
            for row in rows:
                orders[row] = order

        def visit(rows, prefix, remaining, depth):
            if (not rows or not remaining or depth >= max_depth or
                    nodes_used[0] >= node_budget):
                finish(rows, prefix, remaining)
                return
            nodes_used[0] += 1

            best_col = None
            best_score = 0
            for col in remaining:
                if col not in candidates:
                    continue
                counts = Counter(value_strings[col][row] for row in rows)
                score = sum(len(value) * count * (count - 1)
                            for value, count in counts.items())
                if score > best_score or (score == best_score and
                                          best_col is not None and col < best_col):
                    best_col, best_score = col, score

            if best_col is None or best_score <= 0:
                finish(rows, prefix, remaining)
                return

            partitions = defaultdict(list)
            for row in rows:
                partitions[value_strings[best_col][row]].append(row)
            next_remaining = [col for col in remaining if col != best_col]

            # A field with no actual split only consumes useful prefix depth
            # while it has repeated character value; continue once.
            for value in sorted(partitions):
                visit(partitions[value], prefix + [best_col],
                      next_remaining, depth + 1)

        visit(list(range(nrows)), [], list(range(ncols)), 0)
        fallback = normalize(global_order)
        return [order if order is not None else fallback for order in orders]

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True):
        """Compatibility helper with a deterministic global ordering."""
        values = [[self._cell_string(df.iloc[row, col])
                   for row in range(len(df))]
                  for col in range(df.shape[1])]
        order = self._global_order(values)
        result = df.iloc[:, order].copy()
        if row_sort and len(result):
            # Stable sorting is only cosmetic; Trie score is insertion invariant.
            strings = [
                "".join(values[col][row] for col in order)
                for row in range(len(df))
            ]
            positions = sorted(range(len(df)), key=lambda row: strings[row])
            result = result.iloc[positions]
        return result, [[df.columns[col] for col in order] for _ in range(len(result))]

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
        # No helper/index column is inserted: index and every original value stay intact.
        self.df = df
        self.num_rows, self.num_cols = df.shape
        if df.empty or self.num_cols == 0:
            return df.copy(), [[] for _ in range(len(df))]

        columns = list(df.columns)
        stored_values = [
            [df.iloc[row, col] for col in range(self.num_cols)]
            for row in range(self.num_rows)
        ]
        value_strings = [
            [self._cell_string(stored_values[row][col])
             for row in range(self.num_rows)]
            for col in range(self.num_cols)
        ]

        normalize = self._make_constraint_normalizer(
            columns, col_merge or [], one_way_dep or []
        )
        original_order = normalize(list(range(self.num_cols)))
        global_order = normalize(self._global_order(value_strings))
        conditional_orders = self._conditional_orders(
            value_strings, global_order, normalize
        )

        candidates = [
            [original_order for _ in range(self.num_rows)],
            [global_order for _ in range(self.num_rows)],
            conditional_orders,
        ]

        best_orders = candidates[0]
        best_score = -1
        for candidate in candidates:
            strings = [
                "".join(value_strings[col][row] for col in candidate[row])
                for row in range(self.num_rows)
            ]
            score = self._trie_score(strings)
            if score > best_score:
                best_score = score
                best_orders = candidate

        output_rows = [
            [stored_values[row][col] for col in best_orders[row]]
            for row in range(self.num_rows)
        ]
        result = pd.DataFrame(output_rows, index=df.index, columns=df.columns,
                              dtype=object)
        column_orderings = [
            [columns[col] for col in order] for order in best_orders
        ]
        return result, column_orderings


# EVOLVE-BLOCK-END