"""
Prefix-cache-aware dataframe reordering.

Rows are scored as concatenated serialized cell strings.  This implementation
constructs at most three data-preserving column-order candidates:

1. Original column order.
2. Global order ranked by length-weighted repeated-value mass.
3. Row-specific conditional prefix orders using factorized value partitions.

The final selection uses the exact ideal serial-Trie objective: the sum of LCP
lengths between lexicographically adjacent serialized rows.  Source cell values
are never normalized or modified; missing-value handling is used only in the
temporary scoring representation.
"""

import heapq
from typing import List, Tuple

import numpy as np
import pandas as pd

from solver import Algorithm


class Evolved(Algorithm):
    """Construct bounded column-permutation candidates for prefix-cache reuse."""

    def __init__(self, df: pd.DataFrame = None):
        """Store an optional dataframe reference without changing it."""
        self.df = df

    @staticmethod
    def _string_matrix(df: pd.DataFrame) -> np.ndarray:
        """
        Build evaluator-compatible temporary cell strings.

        The returned array mirrors fillna("").astype(str) where supported.
        Original dataframe values remain untouched and are materialized later.
        """
        rows, cols = df.shape
        result = np.empty((rows, cols), dtype=object)

        for col in range(cols):
            series = df.iloc[:, col]

            try:
                result[:, col] = series.fillna("").astype(str).to_numpy(
                    dtype=object,
                    copy=False,
                )
                continue
            except (TypeError, ValueError):
                pass

            values = series.to_numpy(dtype=object, copy=False)
            converted = []

            for value in values:
                missing = False

                try:
                    marker = pd.isna(value)
                    missing = isinstance(marker, (bool, np.bool_)) and bool(marker)
                except (TypeError, ValueError):
                    missing = False

                converted.append("" if missing else str(value))

            result[:, col] = converted

        return result

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """
        Compute one character LCP using binary search and C-level slice equality.

        This avoids Python character-by-character loops on long serialized rows.
        """
        limit = min(len(left), len(right))

        if limit == 0:
            return 0

        if left == right:
            return limit

        low = 0
        high = limit

        while low < high:
            middle = (low + high + 1) // 2

            if left[:middle] == right[:middle]:
                low = middle
            else:
                high = middle - 1

        return low

    @classmethod
    def _reuse_score(cls, row_strings: List[str]) -> int:
        """
        Return exact ideal serial-Trie reuse for a multiset of row strings.

        For fixed output strings, reuse is the sum of LCP values of adjacent
        strings after lexical sorting.
        """
        if len(row_strings) < 2:
            return 0

        ordered = sorted(row_strings)

        return sum(
            cls._lcp(ordered[index - 1], ordered[index])
            for index in range(1, len(ordered))
        )

    @staticmethod
    def _serialize(
        strings: np.ndarray,
        orders: List[List[int]],
    ) -> List[str]:
        """Serialize every source row according to its candidate permutation."""
        return [
            "".join(strings[row_id, order])
            for row_id, order in enumerate(orders)
        ]

    @staticmethod
    def _dependency_edges(columns, one_way_dep) -> List[Tuple[int, int]]:
        """
        Resolve unambiguous dependency declarations into source-position edges.

        Matching follows the established substring convention: (a, b) requires
        the uniquely matching column for a to appear before that for b.
        """
        edges = []

        for dependency in one_way_dep or []:
            if not isinstance(dependency, (tuple, list)) or len(dependency) != 2:
                continue

            left_name, right_name = dependency

            left_positions = [
                position
                for position, column in enumerate(columns)
                if str(left_name) in str(column)
            ]
            right_positions = [
                position
                for position, column in enumerate(columns)
                if str(right_name) in str(column)
            ]

            if (
                len(left_positions) == 1
                and len(right_positions) == 1
                and left_positions[0] != right_positions[0]
            ):
                edges.append((left_positions[0], right_positions[0]))

        return edges

    @staticmethod
    def _apply_dependencies(
        order: List[int],
        edges: List[Tuple[int, int]],
    ) -> List[int]:
        """
        Stably topologically order a proposal while preserving proposal rank.

        If constraints contain a cycle, the valid original permutation is kept.
        """
        if not edges:
            return list(order)

        rank = {column: position for position, column in enumerate(order)}
        children = {column: [] for column in order}
        indegree = {column: 0 for column in order}

        for left, right in edges:
            if left in indegree and right in indegree:
                children[left].append(right)
                indegree[right] += 1

        ready = [
            (rank[column], column)
            for column in order
            if indegree[column] == 0
        ]
        heapq.heapify(ready)

        result = []

        while ready:
            _, current = heapq.heappop(ready)
            result.append(current)

            for child in children[current]:
                indegree[child] -= 1

                if indegree[child] == 0:
                    heapq.heappush(ready, (rank[child], child))

        if len(result) != len(order):
            return list(order)

        return result

    @staticmethod
    def _merge_units(columns, col_merge) -> List[List[int]]:
        """
        Build non-overlapping adjacency units for requested merged columns.

        Members of an accepted merge group retain their supplied group order.
        Every source column occurs exactly once in the returned units.
        """
        used = set()
        units = []

        for group in col_merge or []:
            if not isinstance(group, (tuple, list)):
                continue

            positions = []

            for name in group:
                matches = [
                    position
                    for position, column in enumerate(columns)
                    if column == name and position not in used
                ]

                if len(matches) == 1:
                    positions.append(matches[0])

            if len(positions) > 1:
                units.append(positions)
                used.update(positions)

        for position in range(len(columns)):
            if position not in used:
                units.append([position])

        return units

    @staticmethod
    def _factorize(
        strings: np.ndarray,
        retain_codes: bool,
    ):
        """
        Factorize each serialized field once and compute repetition statistics.

        The required global heuristic score is:
        sum(len(value) * count(value) * (count(value) - 1)).
        """
        rows, cols = strings.shape
        codes = np.empty((rows, cols), dtype=np.int32) if retain_codes else None
        lengths = []
        scores = np.zeros(cols, dtype=np.int64)

        for col in range(cols):
            column_codes, unique_values = pd.factorize(
                strings[:, col],
                sort=False,
            )
            column_codes = column_codes.astype(np.int32, copy=False)

            value_lengths = np.fromiter(
                (len(value) for value in unique_values),
                dtype=np.int64,
                count=len(unique_values),
            )

            counts = np.bincount(
                column_codes,
                minlength=len(unique_values),
            ).astype(np.int64, copy=False)

            scores[col] = int(
                np.sum(value_lengths * counts * (counts - 1))
            )
            lengths.append(value_lengths)

            if retain_codes:
                codes[:, col] = column_codes

        return codes, lengths, scores

    def _global_order(
        self,
        scores: np.ndarray,
        units: List[List[int]],
        dependencies: List[Tuple[int, int]],
    ) -> List[int]:
        """
        Rank merge-safe units using global length-weighted pair repetition.

        Unit ordering preserves merge adjacency.  Individual fields inside each
        unit are ranked deterministically by their own repetition score.
        """
        ranked_units = sorted(
            units,
            key=lambda unit: (
                -sum(int(scores[column]) for column in unit),
                min(unit),
            ),
        )

        proposal = []

        for unit in ranked_units:
            proposal.extend(
                sorted(
                    unit,
                    key=lambda column: (-int(scores[column]), column),
                )
            )

        return self._apply_dependencies(proposal, dependencies)

    @staticmethod
    def _local_score(
        row_ids: np.ndarray,
        column: int,
        codes: np.ndarray,
        lengths: List[np.ndarray],
    ) -> int:
        """
        Score one field inside a conditional row partition.

        This is the local form of length-weighted count*(count-1) repetition.
        """
        selected = codes[row_ids, column]

        if len(selected) < 2:
            return 0

        counts = np.bincount(selected).astype(np.int64, copy=False)

        return int(
            np.sum(lengths[column][:len(counts)] * counts * (counts - 1))
        )

    def _conditional_orders(
        self,
        codes: np.ndarray,
        lengths: List[np.ndarray],
        global_order: List[int],
        dependencies: List[Tuple[int, int]],
        row_stop,
        col_stop,
    ) -> List[List[int]]:
        """
        Build bounded row-specific conditional prefix permutations.

        Each active row group selects its strongest remaining repeated field,
        partitions rows by its factorized value, and recursively specializes
        duplicate child partitions.  Remaining suffix fields use global order.
        """
        rows, cols = codes.shape
        orders = [list(global_order) for _ in range(rows)]

        if rows < 2 or cols < 2:
            return orders

        depth_limit = min(cols, 10)

        for limit in (row_stop, col_stop):
            if limit is not None:
                try:
                    depth_limit = min(depth_limit, max(1, int(limit)))
                except (TypeError, ValueError):
                    pass

        candidate_cap = min(cols, 28)
        max_nodes = 160
        rank = {
            column: position
            for position, column in enumerate(global_order)
        }

        stack = [(np.arange(rows, dtype=np.int64), [], 0)]
        nodes = 0

        while stack and nodes < max_nodes:
            row_ids, prefix, depth = stack.pop()
            nodes += 1

            if len(row_ids) < 2 or depth >= depth_limit:
                continue

            used = set(prefix)
            remaining = [
                column
                for column in global_order
                if column not in used
            ]

            if not remaining:
                continue

            best_column = None
            best_score = 0

            for column in remaining[:candidate_cap]:
                score = self._local_score(
                    row_ids,
                    column,
                    codes,
                    lengths,
                )

                if (
                    score > best_score
                    or (
                        score == best_score
                        and best_column is not None
                        and rank[column] < rank[best_column]
                    )
                ):
                    best_column = column
                    best_score = score

            if best_column is None or best_score <= 0:
                continue

            next_prefix = prefix + [best_column]
            prefix_set = set(next_prefix)

            local_order = next_prefix + [
                column
                for column in global_order
                if column not in prefix_set
            ]
            local_order = self._apply_dependencies(
                local_order,
                dependencies,
            )

            for row_id in row_ids:
                orders[int(row_id)] = local_order

            selected_codes = codes[row_ids, best_column]
            sorting = np.argsort(selected_codes, kind="stable")
            grouped_rows = row_ids[sorting]
            grouped_codes = selected_codes[sorting]

            start = 0

            while start < len(grouped_rows):
                end = start + 1

                while (
                    end < len(grouped_rows)
                    and grouped_codes[end] == grouped_codes[start]
                ):
                    end += 1

                if end - start > 1:
                    stack.append(
                        (
                            grouped_rows[start:end],
                            next_prefix,
                            depth + 1,
                        )
                    )

                start = end

        return orders

    @staticmethod
    def _materialize(
        df: pd.DataFrame,
        orders: List[List[int]],
    ) -> pd.DataFrame:
        """
        Apply per-row permutations while preserving original source objects.

        Object dtype prevents coercion when different rows move heterogeneous
        source values into the same output dataframe column.
        """
        source = df.to_numpy(dtype=object, copy=True)
        rows, cols = source.shape
        output = np.empty((rows, cols), dtype=object)

        for row_id, order in enumerate(orders):
            output[row_id, :] = source[row_id, order]

        return pd.DataFrame(
            output,
            index=df.index.copy(),
            columns=df.columns.copy(),
        )

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
        Return the best exact-Trie-score candidate under supplied constraints.

        Candidate construction is deliberately bounded: identity, global
        repetition ranking, and conditional grouping are each scored once.
        """
        rows, cols = df.shape
        columns = list(df.columns)

        if rows == 0 or cols == 0:
            return df.copy(), [[] for _ in range(rows)]

        strings = self._string_matrix(df)
        units = self._merge_units(columns, col_merge)
        dependencies = self._dependency_edges(columns, one_way_dep)

        has_merge_constraint = any(len(unit) > 1 for unit in units)

        retain_codes = (
            not has_merge_constraint
            and (rows * cols) <= 20_000_000
        )

        codes, lengths, frequency_scores = self._factorize(
            strings,
            retain_codes,
        )

        global_order = self._global_order(
            frequency_scores,
            units,
            dependencies,
        )

        candidates = []

        if not dependencies and not has_merge_constraint:
            identity = list(range(cols))
            candidates.append([list(identity) for _ in range(rows)])

        candidates.append([list(global_order) for _ in range(rows)])

        if retain_codes and codes is not None:
            candidates.append(
                self._conditional_orders(
                    codes,
                    lengths,
                    global_order,
                    dependencies,
                    row_stop,
                    col_stop,
                )
            )

        best_orders = candidates[0]
        best_score = -1

        for candidate in candidates:
            score = self._reuse_score(
                self._serialize(strings, candidate)
            )

            if score > best_score:
                best_score = score
                best_orders = candidate

        reordered = self._materialize(df, best_orders)

        column_orderings = [
            [columns[position] for position in order]
            for order in best_orders
        ]

        return reordered, column_orderings