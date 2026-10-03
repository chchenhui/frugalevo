import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List


class Evolved(Algorithm):
    """Reorder row fields to maximize serialized-prefix reuse without changing cells."""

    def __init__(self, df: pd.DataFrame = None):
        """Store the optional input dataframe for compatibility with Algorithm."""
        self.df = df

    @staticmethod
    def _string_matrix(df: pd.DataFrame) -> np.ndarray:
        """Create evaluator-compatible strings while leaving the source dataframe unchanged."""
        # This intentionally affects only the scoring representation.  The returned
        # dataframe later uses the original objects, including missing values.
        normalized = df.astype(object).where(pd.notna(df), "")
        return normalized.astype(str).to_numpy(dtype=object, copy=False)

    @staticmethod
    def _column_scores(strings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute two cheap repetition-based scores for every source column."""
        rows, cols = strings.shape
        strong = np.zeros(cols, dtype=np.int64)
        light = np.zeros(cols, dtype=np.int64)

        for col in range(cols):
            counts = {}
            lengths = {}
            for value in strings[:, col]:
                counts[value] = counts.get(value, 0) + 1
                if value not in lengths:
                    lengths[value] = len(value)

            # Required frequency ranking: len(v) * count(v) * (count(v)-1).
            strong[col] = sum(
                lengths[value] * count * (count - 1)
                for value, count in counts.items()
                if count > 1
            )

            # A deliberately different alternative which favors useful repeated
            # prefixes but does not over-emphasize extremely frequent short values.
            light[col] = sum(
                lengths[value] * (count - 1)
                for value, count in counts.items()
                if count > 1
            )

        return strong, light

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Return a string LCP using C-level slice comparisons and binary search."""
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

    def _serial_score(
        self, strings: np.ndarray, orders
    ) -> int:
        """Measure exact ideal Trie reuse via lexicographically adjacent row strings."""
        n_rows = strings.shape[0]
        if n_rows < 2:
            return 0

        if isinstance(orders, tuple):
            serial = ["".join(strings[row, orders]) for row in range(n_rows)]
        else:
            serial = [
                "".join(strings[row, orders[row]])
                for row in range(n_rows)
            ]

        serial.sort()
        return sum(
            self._lcp(serial[index - 1], serial[index])
            for index in range(1, n_rows)
        )

    @staticmethod
    def _constraint_order(
        base_order: List[int],
        columns: List,
        col_merge: List[List[str]],
        one_way_dep: List[Tuple[str, str]],
    ) -> List[int]:
        """Apply merge adjacency and one-way precedence constraints to a global order."""
        if not col_merge and not one_way_dep:
            return base_order

        name_to_positions = {}
        for position, name in enumerate(columns):
            name_to_positions.setdefault(name, []).append(position)

        # Build non-overlapping merge blocks.  A block remains consecutive in output.
        blocks = []
        consumed = set()
        for group in col_merge or []:
            members = []
            for name in group:
                for position in name_to_positions.get(name, []):
                    if position not in consumed:
                        members.append(position)
                        consumed.add(position)
            if members:
                members.sort(key=lambda p: base_order.index(p))
                blocks.append(members)

        for position in base_order:
            if position not in consumed:
                blocks.append([position])

        block_of = {}
        for block_index, block in enumerate(blocks):
            for position in block:
                block_of[position] = block_index

        # Preserve rank unless a dependency requires a source block to precede target.
        edges = set()
        for source, target in one_way_dep or []:
            for src_pos in name_to_positions.get(source, []):
                for dst_pos in name_to_positions.get(target, []):
                    src_block = block_of[src_pos]
                    dst_block = block_of[dst_pos]
                    if src_block != dst_block:
                        edges.add((src_block, dst_block))

        ranked_blocks = sorted(
            range(len(blocks)),
            key=lambda b: min(base_order.index(p) for p in blocks[b]),
        )
        rank = {block: i for i, block in enumerate(ranked_blocks)}
        outgoing = {block: [] for block in ranked_blocks}
        indegree = {block: 0 for block in ranked_blocks}

        for source, target in edges:
            outgoing[source].append(target)
            indegree[target] += 1

        ready = sorted(
            [block for block in ranked_blocks if indegree[block] == 0],
            key=lambda block: rank[block],
        )
        ordered_blocks = []
        while ready:
            block = ready.pop(0)
            ordered_blocks.append(block)
            for target in outgoing[block]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort(key=lambda candidate: rank[candidate])

        # Cycles retain their original rank rather than failing or dropping fields.
        if len(ordered_blocks) != len(blocks):
            seen = set(ordered_blocks)
            ordered_blocks.extend(
                block for block in ranked_blocks if block not in seen
            )

        return [position for block in ordered_blocks for position in blocks[block]]

    def _conditional_orders(
        self,
        strings: np.ndarray,
        default_order: Tuple[int, ...],
        max_depth: int,
    ) -> List[Tuple[int, ...]]:
        """Build a bounded conditional prefix partition tree with per-row suffix orders."""
        n_rows, n_cols = strings.shape
        orders = [None] * n_rows
        candidate_cols = list(default_order[: min(n_cols, 32)])
        max_nodes = 384
        nodes_seen = 0

        stack = [(np.arange(n_rows, dtype=np.int64), tuple(), tuple(candidate_cols), 0)]
        while stack:
            rows, prefix, remaining, depth = stack.pop()
            nodes_seen += 1

            if (
                len(rows) < 2
                or not remaining
                or depth >= max_depth
                or nodes_seen > max_nodes
            ):
                tail = tuple(col for col in default_order if col not in prefix)
                final_order = prefix + tail
                for row in rows:
                    orders[int(row)] = final_order
                continue

            best_col = None
            best_score = 0
            best_groups = None

            for col in remaining:
                groups = {}
                for row in rows:
                    value = strings[int(row), col]
                    groups.setdefault(value, []).append(int(row))

                score = sum(
                    len(value) * len(group) * (len(group) - 1)
                    for value, group in groups.items()
                    if len(group) > 1
                )
                if score > best_score or (
                    score == best_score and best_col is not None and col < best_col
                ):
                    best_col = col
                    best_score = score
                    best_groups = groups

            if best_col is None or best_score <= 0:
                tail = tuple(col for col in default_order if col not in prefix)
                final_order = prefix + tail
                for row in rows:
                    orders[int(row)] = final_order
                continue

            next_remaining = tuple(col for col in remaining if col != best_col)
            for group in best_groups.values():
                stack.append(
                    (
                        np.asarray(group, dtype=np.int64),
                        prefix + (best_col,),
                        next_remaining,
                        depth + 1,
                    )
                )

        fallback = tuple(default_order)
        return [order if order is not None else fallback for order in orders]

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
        """Select among bounded full-row constructions using exact serialized Trie reuse."""
        n_rows, n_cols = df.shape
        columns = list(df.columns)

        if n_rows == 0 or n_cols == 0:
            return df.copy(), [[] for _ in range(n_rows)]

        strings = self._string_matrix(df)
        strong, light = self._column_scores(strings)

        original = list(range(n_cols))
        global_a = sorted(original, key=lambda c: (-int(strong[c]), c))
        global_b = sorted(
            original,
            key=lambda c: (-int(light[c]), -int(strong[c]), c),
        )

        global_a = self._constraint_order(
            global_a, columns, col_merge, one_way_dep
        )
        global_b = self._constraint_order(
            global_b, columns, col_merge, one_way_dep
        )

        candidates = [tuple(global_a)]
        if tuple(global_b) != tuple(global_a):
            candidates.append(tuple(global_b))

        # Per-row conditional orders are useful on moderate data, but global orders
        # are safer for very wide/large frames and for explicit ordering constraints.
        cell_budget = n_rows * n_cols
        if (
            not col_merge
            and not one_way_dep
            and cell_budget <= 1_250_000
            and n_cols <= 128
        ):
            depth_limit = min(
                8,
                n_cols,
                col_stop if col_stop is not None and col_stop > 0 else 8,
            )
            conditional = self._conditional_orders(
                strings, tuple(global_a), depth_limit
            )
            candidates.append(conditional)

        # Exact scoring is deliberately bounded.  The serialized strings are also
        # necessary to evaluate partial-field and cross-field prefix matches.
        total_chars = sum(len(value) for value in strings.ravel())
        if total_chars <= 30_000_000:
            best = max(candidates, key=lambda candidate: self._serial_score(strings, candidate))
        else:
            best = candidates[0]

        source_values = df.to_numpy(dtype=object, copy=True)
        if isinstance(best, tuple):
            output_values = source_values[:, list(best)]
            order_positions = [best] * n_rows
        else:
            output_values = np.empty_like(source_values, dtype=object)
            for row, order in enumerate(best):
                output_values[row, :] = source_values[row, list(order)]
            order_positions = best

        # Object dtype avoids coercing mixed values while preserving every source cell.
        reordered = pd.DataFrame(
            output_values,
            index=df.index.copy(),
            columns=df.columns.copy(),
            dtype=object,
        )
        column_orderings = [
            [columns[position] for position in order]
            for order in order_positions
        ]
        return reordered, column_orderings