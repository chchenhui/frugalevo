# EVOLVE-BLOCK-START
import numpy as np
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from collections import Counter


class Evolved(Algorithm):
    """
    Safe prompt-prefix-cache reordering.

    Values are never modified.  A returned row is a permutation of the input
    row, and column_orderings describes which original source column occupies
    every returned position.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _serialized_value(value) -> str:
        """Match evaluator normalization without changing stored values."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and missing:
                return ""
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """LCP without a Python per-character scan."""
        if left == right:
            return len(left)
        hi = min(len(left), len(right))
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if left[:mid] == right[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _trie_score(self, strings: List[str]) -> int:
        """Exact serial Trie reuse for a fixed multiset of strings."""
        if len(strings) < 2:
            return 0
        ordered = sorted(strings)
        return sum(
            self._lcp(ordered[i - 1], ordered[i])
            for i in range(1, len(ordered))
        )

    @staticmethod
    def _column_repetition_scores(serialized: List[List[str]]) -> List[int]:
        scores = []
        if not serialized:
            return scores
        rows = len(serialized)
        cols = len(serialized[0])
        for col in range(cols):
            counts = Counter(serialized[row][col] for row in range(rows))
            scores.append(
                sum(
                    len(value) * count * (count - 1)
                    for value, count in counts.items()
                )
            )
        return scores

    @staticmethod
    def _resolve_column(columns: List[str], requested: str):
        """Retain the old API's convenient exact-or-unique-substring behavior."""
        if requested in columns:
            return requested
        matches = [column for column in columns if requested in str(column)]
        return matches[0] if len(matches) == 1 else None

    def _legal_global_order(
        self,
        ranking: List[int],
        columns: List[str],
        col_merge: List[List[str]],
        one_way_dep: List[Tuple[str, str]],
    ) -> List[int]:
        """
        Build a deterministic legal global order.

        Merge groups are kept contiguous.  Dependencies between groups are
        respected with a small stable topological ordering where possible.
        """
        ncols = len(columns)
        position = {column: idx for idx, column in enumerate(columns)}

        blocks = []
        used = set()
        for merge_group in col_merge or []:
            members = []
            for requested in merge_group:
                resolved = self._resolve_column(columns, requested)
                if resolved is not None and position[resolved] not in used:
                    members.append(position[resolved])
                    used.add(position[resolved])
            if members:
                blocks.append(members)

        for idx in range(ncols):
            if idx not in used:
                blocks.append([idx])

        block_of = {}
        for block_id, block in enumerate(blocks):
            for idx in block:
                block_of[idx] = block_id

        edges = {idx: set() for idx in range(len(blocks))}
        indegree = [0] * len(blocks)
        for before, after in one_way_dep or []:
            left = self._resolve_column(columns, before)
            right = self._resolve_column(columns, after)
            if left is None or right is None:
                continue
            source = block_of[position[left]]
            target = block_of[position[right]]
            if source != target and target not in edges[source]:
                edges[source].add(target)
                indegree[target] += 1

        block_score = [
            max((ranking[idx] for idx in block), default=0)
            for block in blocks
        ]
        available = [idx for idx, degree in enumerate(indegree) if degree == 0]
        block_order = []

        while available:
            available.sort(key=lambda idx: (-block_score[idx], idx))
            current = available.pop(0)
            block_order.append(current)
            for target in sorted(edges[current]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    available.append(target)

        # Cyclic dependency input cannot be perfectly topologically sorted.
        # Preserve all columns deterministically rather than dropping data.
        if len(block_order) != len(blocks):
            remaining = [idx for idx in range(len(blocks)) if idx not in block_order]
            remaining.sort(key=lambda idx: (-block_score[idx], idx))
            block_order.extend(remaining)

        result = []
        for block_id in block_order:
            block = blocks[block_id]
            # Explicit merge order is retained; singleton blocks are ranked.
            if len(block) == 1:
                result.extend(block)
            else:
                result.extend(block)
        return result

    def _conditional_orders(
        self,
        serialized: List[List[str]],
        fallback: List[int],
        candidate_columns: List[int],
    ) -> List[List[int]]:
        """
        Construct a bounded conditional prefix partition tree.

        Each node chooses the field with the greatest length-weighted repeated
        value score within that node.  The method only changes ordering; it
        never groups, removes, or changes rows.
        """
        nrows = len(serialized)
        ncols = len(fallback)
        orders = [None] * nrows
        if nrows == 0 or ncols == 0:
            return [list(fallback) for _ in range(nrows)]

        max_nodes = 128
        max_depth = min(ncols, 12)
        nodes_seen = 0
        stack = [(list(range(nrows)), [], set(), 0)]

        while stack:
            rows, prefix, chosen, depth = stack.pop()
            if (
                len(rows) <= 1
                or depth >= max_depth
                or nodes_seen >= max_nodes
            ):
                tail = [col for col in fallback if col not in chosen]
                order = prefix + tail
                for row in rows:
                    orders[row] = order
                continue

            nodes_seen += 1
            best_col = None
            best_score = 0
            for col in candidate_columns:
                if col in chosen:
                    continue
                counts = Counter(serialized[row][col] for row in rows)
                score = sum(
                    len(value) * count * (count - 1)
                    for value, count in counts.items()
                )
                if score > best_score or (
                    score == best_score and score > 0 and
                    (best_col is None or col < best_col)
                ):
                    best_col = col
                    best_score = score

            if best_col is None or best_score <= 0:
                tail = [col for col in fallback if col not in chosen]
                order = prefix + tail
                for row in rows:
                    orders[row] = order
                continue

            partitions = {}
            for row in rows:
                partitions.setdefault(serialized[row][best_col], []).append(row)

            # A constant field can still be a useful shared prefix.  Continue
            # with the remaining fields, but do not branch unnecessarily.
            next_prefix = prefix + [best_col]
            next_chosen = set(chosen)
            next_chosen.add(best_col)
            for value_rows in partitions.values():
                stack.append((value_rows, next_prefix, next_chosen, depth + 1))

        for row in range(nrows):
            if orders[row] is None:
                orders[row] = list(fallback)
        return orders

    def _score_orders(
        self,
        serialized: List[List[str]],
        orders: List[List[int]],
    ) -> int:
        strings = [
            "".join(serialized[row][col] for col in orders[row])
            for row in range(len(serialized))
        ]
        return self._trie_score(strings)

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
        Return a data-preserving row-wise column permutation.

        Arguments retained for API compatibility are deliberately bounded:
        no recursive dataframe reconstruction, synthetic index column, merge,
        row loss, or value conversion is performed.
        """
        nrows, ncols = df.shape
        columns = list(df.columns)

        if nrows == 0 or ncols == 0:
            return df.copy(), [list(columns) for _ in range(nrows)]

        raw_values = df.to_numpy(dtype=object, copy=True)
        serialized = [
            [self._serialized_value(raw_values[row, col]) for col in range(ncols)]
            for row in range(nrows)
        ]

        repetition = self._column_repetition_scores(serialized)
        length_totals = [
            sum(len(serialized[row][col]) for row in range(nrows))
            for col in range(ncols)
        ]

        original_ranking = list(range(ncols))
        frequency_ranking = sorted(
            range(ncols), key=lambda col: (-repetition[col], col)
        )
        length_ranking = sorted(
            range(ncols), key=lambda col: (-length_totals[col], col)
        )

        # With structural constraints, use only globally legal candidates.
        global_original = self._legal_global_order(
            original_ranking, columns, col_merge, one_way_dep
        )
        global_frequency = self._legal_global_order(
            frequency_ranking, columns, col_merge, one_way_dep
        )
        global_length = self._legal_global_order(
            length_ranking, columns, col_merge, one_way_dep
        )

        candidates = []
        seen = set()
        for order in (global_original, global_frequency, global_length):
            key = tuple(order)
            if key not in seen:
                seen.add(key)
                candidates.append([list(order) for _ in range(nrows)])

        # A row-specific order cannot safely preserve explicit global merge or
        # dependency constraints, so only use it when no such constraints exist.
        if not col_merge and not one_way_dep and len(candidates) < 3:
            bounded_columns = frequency_ranking[:min(ncols, 48)]
            conditional = self._conditional_orders(
                serialized, global_frequency, bounded_columns
            )
            candidates.append(conditional)

        best_orders = candidates[0]
        best_score = self._score_orders(serialized, best_orders)
        for candidate in candidates[1:3]:
            score = self._score_orders(serialized, candidate)
            if score > best_score:
                best_score = score
                best_orders = candidate

        output = np.empty((nrows, ncols), dtype=object)
        for row, order in enumerate(best_orders):
            output[row, :] = raw_values[row, order]

        # Keep the original index and visible dataframe shape.  The returned
        # ordering maps each output position to its original source column.
        reordered = pd.DataFrame(output, index=df.index.copy(), columns=df.columns.copy())
        column_orderings = [
            [columns[col] for col in order]
            for order in best_orders
        ]
        return reordered, column_orderings


# EVOLVE-BLOCK-END