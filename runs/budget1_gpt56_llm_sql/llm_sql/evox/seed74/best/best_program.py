# EVOLVE-BLOCK-START
"""
Prefix-cache-oriented dataframe reordering.

The implementation preserves every source cell, row, index, column label, and
shape. It only permutes source cells within an individual row. A bounded set of
column-order candidates is evaluated with the serial Trie objective equivalent:
the sum of LCP lengths between lexicographically adjacent serialized rows.
"""

from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd

from solver import Algorithm


class Evolved(Algorithm):
    """Construct and select bounded prefix-sharing column layouts."""

    def __init__(self, df: pd.DataFrame = None):
        """Store an optional dataframe for compatibility with base callers."""
        self.df = df

    @staticmethod
    def _cell_string(value) -> str:
        """Return evaluator-compatible text without modifying the source value."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and bool(missing):
                return ""
        except Exception:
            pass
        return str(value)

    def _string_matrix(self, df: pd.DataFrame) -> List[List[str]]:
        """Serialize every source cell once for reuse across all candidates."""
        values = df.to_numpy(dtype=object, copy=False)
        return [
            [self._cell_string(value) for value in row]
            for row in values
        ]

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Compute character LCP using binary-search slice equality in C."""
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

    def _trie_score(self, row_strings: List[str]) -> int:
        """Score fixed serialized rows by sorted-adjacent LCP total."""
        if len(row_strings) < 2:
            return 0

        ordered = sorted(row_strings)
        total = 0
        previous = ordered[0]

        for current in ordered[1:]:
            total += self._lcp(previous, current)
            previous = current

        return total

    @staticmethod
    def _serialize_rows(
        cells: List[List[str]],
        orders: List[List[int]],
    ) -> List[str]:
        """Serialize each row according to its row-specific source order."""
        return [
            "".join(cells[row_id][column] for column in order)
            for row_id, order in enumerate(orders)
        ]

    @staticmethod
    def _resolve_column(name, columns: List) -> int:
        """Resolve a column first exactly, then by unique textual containment."""
        for index, column in enumerate(columns):
            if column == name:
                return index

        text = str(name)
        matches = [
            index
            for index, column in enumerate(columns)
            if text in str(column)
        ]
        return matches[0] if len(matches) == 1 else -1

    def _column_scores(
        self,
        cells: List[List[str]],
    ) -> Tuple[List[int], List[int]]:
        """Rank columns with strong and light length-weighted repetition metrics."""
        if not cells:
            return [], []

        ncols = len(cells[0])
        pair_scores = [0] * ncols
        light_scores = [0] * ncols

        for column in range(ncols):
            counts = Counter(row[column] for row in cells)

            pair_scores[column] = sum(
                len(value) * count * (count - 1)
                for value, count in counts.items()
            )
            light_scores[column] = sum(
                len(value) * (count - 1)
                for value, count in counts.items()
            )

        pair_rank = sorted(
            range(ncols),
            key=lambda column: (-pair_scores[column], column),
        )
        light_rank = sorted(
            range(ncols),
            key=lambda column: (
                -light_scores[column],
                -pair_scores[column],
                column,
            ),
        )
        return pair_rank, light_rank

    def _respect_constraints(
        self,
        desired: List[int],
        columns: List,
        col_merge,
        one_way_dep,
    ) -> List[int]:
        """Apply contiguous merge units and before-to-after dependency edges."""
        if len(desired) <= 1:
            return list(desired)

        rank = {column: position for position, column in enumerate(desired)}
        used = set()
        units = []

        for group in col_merge or []:
            unit = []
            for name in group:
                column = self._resolve_column(name, columns)
                if column >= 0 and column not in used:
                    unit.append(column)

            if unit:
                unit.sort(key=lambda column: rank[column])
                units.append(unit)
                used.update(unit)

        for column in desired:
            if column not in used:
                units.append([column])

        unit_for = {}
        for unit_id, unit in enumerate(units):
            for column in unit:
                unit_for[column] = unit_id

        edges = defaultdict(set)
        indegree = [0] * len(units)

        for dependency in one_way_dep or []:
            if not isinstance(dependency, (list, tuple)) or len(dependency) < 2:
                continue

            before = self._resolve_column(dependency[0], columns)
            after = self._resolve_column(dependency[1], columns)

            if before < 0 or after < 0:
                continue

            source = unit_for[before]
            target = unit_for[after]

            if source != target and target not in edges[source]:
                edges[source].add(target)
                indegree[target] += 1

        priority = [
            min(rank[column] for column in unit)
            for unit in units
        ]

        available = [
            unit_id
            for unit_id, degree in enumerate(indegree)
            if degree == 0
        ]
        selected = []

        while available:
            available.sort(key=lambda unit_id: (priority[unit_id], unit_id))
            current = available.pop(0)
            selected.append(current)

            for successor in edges[current]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    available.append(successor)

        if len(selected) < len(units):
            selected_set = set(selected)
            remaining = [
                unit_id
                for unit_id in range(len(units))
                if unit_id not in selected_set
            ]
            selected.extend(
                sorted(
                    remaining,
                    key=lambda unit_id: (priority[unit_id], unit_id),
                )
            )

        return [
            column
            for unit_id in selected
            for column in units[unit_id]
        ]

    def _conditional_orders(
        self,
        cells: List[List[str]],
        global_rank: List[int],
        max_depth: int,
        max_nodes: int,
    ) -> List[List[int]]:
        """Build a bounded conditional prefix partition tree by pair repetition."""
        nrows = len(cells)
        ncols = len(global_rank)

        if nrows == 0:
            return []

        orders = [None] * nrows
        fallback = list(global_rank)
        candidate_cap = min(ncols, 24)
        expanded = 0

        stack = [(list(range(nrows)), [], list(global_rank), 0)]

        while stack:
            row_ids, prefix, remaining, depth = stack.pop()

            if (
                len(row_ids) <= 1
                or not remaining
                or depth >= max_depth
                or expanded >= max_nodes
            ):
                final_order = prefix + [
                    column for column in global_rank
                    if column in remaining
                ]
                for row_id in row_ids:
                    orders[row_id] = final_order
                continue

            best_column = -1
            best_score = 0
            best_groups = None

            for column in remaining[:candidate_cap]:
                groups = defaultdict(list)

                for row_id in row_ids:
                    groups[cells[row_id][column]].append(row_id)

                score = sum(
                    len(value) * len(group) * (len(group) - 1)
                    for value, group in groups.items()
                )

                if (
                    score > best_score
                    or (
                        score == best_score
                        and score > 0
                        and (best_column < 0 or column < best_column)
                    )
                ):
                    best_column = column
                    best_score = score
                    best_groups = groups

            if best_column < 0 or best_score <= 0 or best_groups is None:
                final_order = prefix + [
                    column for column in global_rank
                    if column in remaining
                ]
                for row_id in row_ids:
                    orders[row_id] = final_order
                continue

            expanded += 1
            next_prefix = prefix + [best_column]
            next_remaining = [
                column for column in remaining
                if column != best_column
            ]

            grouped_rows = sorted(
                best_groups.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )

            for _, group_rows in reversed(grouped_rows):
                stack.append(
                    (group_rows, next_prefix, next_remaining, depth + 1)
                )

        return [
            order if order is not None else fallback
            for order in orders
        ]

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
        """Select the best of two global layouts and one bounded conditional layout."""
        nrows, ncols = df.shape

        if ncols == 0:
            return df.copy(), [[] for _ in range(nrows)]

        if nrows == 0:
            return df.copy(), []

        columns = list(df.columns)
        cells = self._string_matrix(df)

        pair_rank, light_rank = self._column_scores(cells)

        global_pair = self._respect_constraints(
            pair_rank,
            columns,
            col_merge,
            one_way_dep,
        )
        global_light = self._respect_constraints(
            light_rank,
            columns,
            col_merge,
            one_way_dep,
        )

        depth_limit = min(ncols, 8)

        if col_stop is not None and col_stop > 0:
            depth_limit = min(depth_limit, int(col_stop))

        if row_stop is not None and row_stop > 0:
            depth_limit = min(depth_limit, int(row_stop))

        depth_limit = max(1, depth_limit)

        conditional_raw = self._conditional_orders(
            cells=cells,
            global_rank=pair_rank,
            max_depth=depth_limit,
            max_nodes=128,
        )

        constrained_cache = {}
        conditional = []

        for order in conditional_raw:
            key = tuple(order)
            resolved = constrained_cache.get(key)

            if resolved is None:
                resolved = self._respect_constraints(
                    order,
                    columns,
                    col_merge,
                    one_way_dep,
                )
                constrained_cache[key] = resolved

            conditional.append(resolved)

        candidates = [
            [global_pair] * nrows,
            [global_light] * nrows,
            conditional,
        ]

        best_orders = candidates[0]
        best_score = -1

        for orders in candidates:
            score = self._trie_score(self._serialize_rows(cells, orders))
            if score > best_score:
                best_score = score
                best_orders = orders

        source_values = df.to_numpy(dtype=object, copy=False)

        global_layout = all(order == best_orders[0] for order in best_orders)

        if global_layout:
            output_values = source_values[:, best_orders[0]].copy()
        else:
            output_values = np.empty((nrows, ncols), dtype=object)

            for row_id, order in enumerate(best_orders):
                output_values[row_id, :] = [
                    source_values[row_id, column]
                    for column in order
                ]

        result = pd.DataFrame(
            output_values,
            index=df.index.copy(),
            columns=df.columns.copy(),
        )

        column_orderings = [
            [columns[column] for column in order]
            for order in best_orders
        ]

        return result, column_orderings


# EVOLVE-BLOCK-END