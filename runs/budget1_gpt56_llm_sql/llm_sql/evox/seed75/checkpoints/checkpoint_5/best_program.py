# EVOLVE-BLOCK-START
"""
Prefix-cache-oriented dataframe column permutation.

The evaluator serializes every output row by concatenating cell strings without
separators.  This implementation preserves every source cell exactly once while
choosing bounded row-specific column permutations that maximize the exact ideal
serial-Trie objective when practical.

Candidates:
1. Global ordering by length * count * (count - 1), emphasizing repeated values.
2. Global ordering by actual character-prefix reuse inside each column.
3. A bounded conditional partition tree which chooses reusable leading fields
   independently inside groups of rows.

The final candidate is selected using exact sorted-row LCP scoring.
"""

from typing import List, Tuple
import math

import numpy as np
import pandas as pd

from solver import Algorithm


class Evolved(Algorithm):
    """Choose safe, bounded source-column permutations for prefix reuse."""

    def __init__(self, df: pd.DataFrame = None):
        """Store the optional dataframe for Algorithm compatibility."""
        self.df = df

    @staticmethod
    def _serial_value(value) -> str:
        """Convert one scalar to the evaluator's missing-normalized string form."""
        if value is None or value is pd.NA:
            return ""

        if isinstance(value, float):
            try:
                if math.isnan(value):
                    return ""
            except Exception:
                pass

        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and missing:
                return ""
        except Exception:
            pass

        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Return character LCP length using bounded C-level slice comparisons."""
        limit = min(len(left), len(right))
        if limit == 0:
            return 0
        if left == right:
            return limit

        low, high = 0, limit
        while low < high:
            middle = (low + high + 1) // 2
            if left[:middle] == right[:middle]:
                low = middle
            else:
                high = middle - 1
        return low

    @classmethod
    def _trie_score(cls, strings: List[str]) -> int:
        """Compute exact ideal Trie reuse from lexicographically adjacent strings."""
        if len(strings) < 2:
            return 0

        ordered = sorted(strings)
        score = 0
        previous = ordered[0]
        for current in ordered[1:]:
            score += cls._lcp(previous, current)
            previous = current
        return score

    @staticmethod
    def _serialize(text_cells: List[List[str]], orders: List[List[int]]) -> List[str]:
        """Serialize all rows under their corresponding source-column permutations."""
        return [
            "".join(text_cells[row_id][column_id] for column_id in order)
            for row_id, order in enumerate(orders)
        ]

    @staticmethod
    def _resolve_constraints(
        columns: List,
        col_merge: List[List[str]],
        one_way_dep: List[Tuple[str, str]],
    ) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        """Convert merge groups to contiguous blocks and dependencies to block edges."""
        ncols = len(columns)
        parent = list(range(ncols))

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left, right):
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        def locate(name):
            exact = [i for i, col in enumerate(columns) if col == name]
            if len(exact) == 1:
                return exact[0]

            token = str(name)
            partial = [i for i, col in enumerate(columns) if token in str(col)]
            return partial[0] if len(partial) == 1 else None

        for group in col_merge or []:
            positions = [locate(name) for name in group]
            positions = [position for position in positions if position is not None]
            if len(positions) > 1:
                first = positions[0]
                for position in positions[1:]:
                    union(first, position)

        grouped = {}
        for column_id in range(ncols):
            grouped.setdefault(find(column_id), []).append(column_id)

        blocks = sorted(grouped.values(), key=lambda block: min(block))
        block_of = {}
        for block_id, block in enumerate(blocks):
            for column_id in block:
                block_of[column_id] = block_id

        edges = set()
        for dependency in one_way_dep or []:
            if not isinstance(dependency, (tuple, list)) or len(dependency) != 2:
                continue

            before = locate(dependency[0])
            after = locate(dependency[1])
            if before is None or after is None:
                continue

            source = block_of[before]
            target = block_of[after]
            if source != target:
                edges.add((source, target))

        return blocks, sorted(edges)

    @staticmethod
    def _topological_blocks(
        blocks: List[List[int]],
        edges: List[Tuple[int, int]],
        priority: List[int],
    ) -> List[int]:
        """Return a deterministic priority-aware topological ordering of blocks."""
        nblocks = len(blocks)
        children = [[] for _ in range(nblocks)]
        indegree = [0] * nblocks

        for source, target in edges:
            children[source].append(target)
            indegree[target] += 1

        def block_key(block_id):
            return (
                min(priority[column_id] for column_id in blocks[block_id]),
                min(blocks[block_id]),
            )

        available = [block_id for block_id in range(nblocks) if indegree[block_id] == 0]
        output = []

        while available:
            available.sort(key=block_key)
            current = available.pop(0)
            output.append(current)

            for target in children[current]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    available.append(target)

        if len(output) < nblocks:
            used = set(output)
            remaining = [block_id for block_id in range(nblocks) if block_id not in used]
            remaining.sort(key=block_key)
            output.extend(remaining)

        return output

    @staticmethod
    def _flatten(block_order: List[int], blocks: List[List[int]]) -> List[int]:
        """Expand ordered blocks into one complete source-column permutation."""
        return [column_id for block_id in block_order for column_id in blocks[block_id]]

    @staticmethod
    def _factorize_columns(text_cells: List[List[str]], nrows: int, ncols: int):
        """Factorize serialized column values once for inexpensive repeated-value scoring."""
        codes = []
        lengths = []

        for column_id in range(ncols):
            mapping = {}
            column_codes = np.empty(nrows, dtype=np.int32)
            value_lengths = []

            for row_id in range(nrows):
                value = text_cells[row_id][column_id]
                code = mapping.get(value)
                if code is None:
                    code = len(mapping)
                    mapping[value] = code
                    value_lengths.append(len(value))
                column_codes[row_id] = code

            codes.append(column_codes)
            lengths.append(np.asarray(value_lengths, dtype=np.int64))

        return codes, lengths

    @classmethod
    def _column_prefix_weight(
        cls,
        text_cells: List[List[str]],
        column_id: int,
    ) -> int:
        """Measure exact single-column character-prefix reuse across all rows."""
        values = sorted(row[column_id] for row in text_cells)
        if len(values) < 2:
            return 0

        total = 0
        previous = values[0]
        for current in values[1:]:
            total += cls._lcp(previous, current)
            previous = current
        return total

    @staticmethod
    def _group_pair_weight(
        row_ids: np.ndarray,
        column_codes: np.ndarray,
        value_lengths: np.ndarray,
    ) -> int:
        """Score repeated character mass for a field restricted to one row group."""
        if len(row_ids) < 2:
            return 0

        counts = np.bincount(column_codes[row_ids], minlength=len(value_lengths))
        return int(np.dot(value_lengths, counts * (counts - 1)))

    def _conditional_orders(
        self,
        blocks: List[List[int]],
        edges: List[Tuple[int, int]],
        base_block_order: List[int],
        candidate_blocks: List[int],
        codes: List[np.ndarray],
        lengths: List[np.ndarray],
        nrows: int,
    ) -> List[List[int]]:
        """Build a bounded conditional prefix partition tree with deterministic tails."""
        nblocks = len(blocks)
        predecessors = [set() for _ in range(nblocks)]
        for source, target in edges:
            predecessors[target].add(source)

        output_orders = [None] * nrows
        max_depth = min(7, nblocks)
        max_nodes = 112
        node_count = 0

        def complete(prefix, remaining):
            tail = [block_id for block_id in base_block_order if block_id in remaining]
            return self._flatten(prefix + tail, blocks)

        def assign(row_ids, order):
            for row_id in row_ids:
                output_orders[int(row_id)] = order

        def visit(row_ids, prefix, remaining, depth):
            nonlocal node_count

            if (
                len(row_ids) < 2
                or not remaining
                or depth >= max_depth
                or node_count >= max_nodes
            ):
                assign(row_ids, complete(prefix, remaining))
                return

            available = [
                block_id
                for block_id in remaining
                if predecessors[block_id].issubset(prefix)
            ]
            preferred = [block_id for block_id in candidate_blocks if block_id in available]
            tested = preferred if preferred else available

            selected = None
            best_score = 0

            for block_id in tested:
                score = 0
                for column_id in blocks[block_id]:
                    score += self._group_pair_weight(
                        row_ids,
                        codes[column_id],
                        lengths[column_id],
                    )

                if score > best_score or (
                    score == best_score
                    and selected is not None
                    and block_id < selected
                ):
                    selected = block_id
                    best_score = score

            if selected is None or best_score <= 0:
                assign(row_ids, complete(prefix, remaining))
                return

            node_count += 1
            next_prefix = prefix + [selected]
            next_remaining = set(remaining)
            next_remaining.remove(selected)
            selected_columns = blocks[selected]

            buckets = {}
            if len(selected_columns) == 1:
                selected_codes = codes[selected_columns[0]][row_ids]
                for position, code in enumerate(selected_codes):
                    buckets.setdefault(int(code), []).append(int(row_ids[position]))
            else:
                for row_id in row_ids:
                    token = tuple(
                        int(codes[column_id][row_id])
                        for column_id in selected_columns
                    )
                    buckets.setdefault(token, []).append(int(row_id))

            if len(buckets) <= 1:
                visit(row_ids, next_prefix, next_remaining, depth + 1)
                return

            for bucket in buckets.values():
                visit(
                    np.asarray(bucket, dtype=np.int64),
                    next_prefix,
                    next_remaining,
                    depth + 1,
                )

        visit(
            np.arange(nrows, dtype=np.int64),
            [],
            set(range(nblocks)),
            0,
        )

        fallback = self._flatten(base_block_order, blocks)
        return [order if order is not None else fallback for order in output_orders]

    @staticmethod
    def _materialize(
        source: np.ndarray,
        index,
        columns,
        orders: List[List[int]],
    ) -> pd.DataFrame:
        """Materialize row-specific source permutations without converting cell values."""
        nrows, ncols = source.shape
        output = np.empty((nrows, ncols), dtype=object)

        for row_id, order in enumerate(orders):
            output[row_id, :] = source[row_id, order]

        return pd.DataFrame(output, index=index, columns=columns, dtype=object)

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
        """
        Return the best bounded prefix-sharing candidate while preserving all cells.

        The compatibility parameters are accepted unchanged.  They are not used
        as stopping controls because candidate count and conditional tree size
        are explicitly bounded internally.
        """
        nrows, ncols = df.shape
        columns = list(df.columns)

        if nrows == 0 or ncols == 0:
            return df.copy(), [[] for _ in range(nrows)]

        source = df.to_numpy(dtype=object, copy=True)
        text_cells = [
            [self._serial_value(source[row_id, column_id]) for column_id in range(ncols)]
            for row_id in range(nrows)
        ]

        blocks, edges = self._resolve_constraints(columns, col_merge, one_way_dep)
        codes, lengths = self._factorize_columns(text_cells, nrows, ncols)

        pair_weight = [0] * ncols
        for column_id in range(ncols):
            counts = np.bincount(
                codes[column_id],
                minlength=len(lengths[column_id]),
            )
            pair_weight[column_id] = int(
                np.dot(lengths[column_id], counts * (counts - 1))
            )

        pair_priority = [-weight for weight in pair_weight]
        pair_block_order = self._topological_blocks(
            blocks,
            edges,
            pair_priority,
        )
        pair_order = self._flatten(pair_block_order, blocks)

        candidates = [[pair_order] * nrows]
        scores = [self._trie_score(self._serialize(text_cells, candidates[0]))]

        # Character-prefix ranking captures useful reuse in columns such as
        # timestamps, IDs, URLs, and numeric strings whose values differ but
        # have long shared prefixes.  Bound its sorting cost on giant matrices.
        allow_prefix_global = nrows * ncols <= 900_000 and ncols <= 192
        if allow_prefix_global and len(blocks) > 1:
            prefix_weight = [
                self._column_prefix_weight(text_cells, column_id)
                for column_id in range(ncols)
            ]
            prefix_order = self._flatten(
                self._topological_blocks(
                    blocks,
                    edges,
                    [-weight for weight in prefix_weight],
                ),
                blocks,
            )
            prefix_orders = [prefix_order] * nrows
            candidates.append(prefix_orders)
            scores.append(self._trie_score(self._serialize(text_cells, prefix_orders)))

        # Keep one conditional proposal.  It is intentionally bounded more
        # tightly than exhaustive recursive search to protect runtime.
        allow_conditional = (
            nrows >= 2
            and ncols <= 256
            and nrows * ncols <= 1_500_000
            and len(blocks) > 1
        )

        if allow_conditional:
            ranked_blocks = sorted(
                range(len(blocks)),
                key=lambda block_id: (
                    -sum(pair_weight[column_id] for column_id in blocks[block_id]),
                    block_id,
                ),
            )

            conditional_orders = self._conditional_orders(
                blocks=blocks,
                edges=edges,
                base_block_order=pair_block_order,
                candidate_blocks=ranked_blocks[:min(36, len(ranked_blocks))],
                codes=codes,
                lengths=lengths,
                nrows=nrows,
            )
            candidates.append(conditional_orders)
            scores.append(
                self._trie_score(self._serialize(text_cells, conditional_orders))
            )

        best_id = max(range(len(candidates)), key=lambda i: (scores[i], -i))
        best_orders = candidates[best_id]

        result = self._materialize(source, df.index, columns, best_orders)
        column_orderings = [
            [columns[column_id] for column_id in order]
            for order in best_orders
        ]

        return result, column_orderings


# EVOLVE-BLOCK-END