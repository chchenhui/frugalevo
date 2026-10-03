# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict, Any
from collections import Counter


class Evolved(Algorithm):
    """
    Safe bounded prefix-cache optimizer.

    The dataframe is never modified in place.  Global candidates reorder normal
    dataframe columns.  The conditional candidate may use a different source
    column permutation per row; in that case output columns retain the original
    labels and column_orderings describes the source column represented at each
    output position.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _string_value(value: Any) -> str:
        """Match the evaluator's fillna('').astype(str) representation safely."""
        if value is None:
            return ""
        try:
            missing = pd.isna(value)
            if isinstance(missing, bool) and missing:
                return ""
            # numpy scalar booleans are intentionally handled without numpy.
            if hasattr(missing, "item") and bool(missing.item()):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        if left == right:
            return len(left)
        limit = min(len(left), len(right))
        low, high = 0, limit
        # Slice comparison is implemented in C and avoids Python character loops.
        while low < high:
            middle = (low + high + 1) // 2
            if left[:middle] == right[:middle]:
                low = middle
            else:
                high = middle - 1
        return low

    def _trie_score(self, serial_rows: List[str]) -> int:
        if len(serial_rows) < 2:
            return 0
        ordered = sorted(serial_rows)
        return sum(self._lcp(ordered[i - 1], ordered[i])
                   for i in range(1, len(ordered)))

    @staticmethod
    def _serial_rows(values: List[List[str]], orders: List[List[int]]) -> List[str]:
        return [
            "".join(values[row][column] for column in orders[row])
            for row in range(len(values))
        ]

    def _constraint_order(
        self,
        proposed: List[int],
        columns: List[Any],
        col_merge: List[List[str]],
        one_way_dep: List[Tuple[str, str]],
    ) -> List[int]:
        """
        Make requested merge groups contiguous and apply dependency precedence.
        Constraints are resolved deterministically and never add or remove a
        source column.
        """
        count = len(columns)
        rank = {column: position for position, column in enumerate(proposed)}
        name_to_positions: Dict[str, List[int]] = {}
        for position, name in enumerate(columns):
            name_to_positions.setdefault(str(name), []).append(position)

        used = set()
        units: List[List[int]] = []
        for requested_group in col_merge or []:
            group = []
            for requested_name in requested_group:
                positions = name_to_positions.get(str(requested_name), [])
                for position in positions:
                    if position not in used:
                        group.append(position)
                        used.add(position)
            if group:
                group.sort(key=lambda position: rank.get(position, position))
                units.append(group)

        for position in proposed:
            if position not in used:
                units.append([position])
                used.add(position)

        unit_for = {}
        for unit_number, unit in enumerate(units):
            for position in unit:
                unit_for[position] = unit_number

        edges = {number: set() for number in range(len(units))}
        indegree = [0] * len(units)

        def resolve(name: str) -> List[int]:
            exact = name_to_positions.get(str(name), [])
            if exact:
                return exact
            # Retain compatibility with the former substring dependency API.
            matches = [
                position for position, column in enumerate(columns)
                if str(name) in str(column)
            ]
            return matches[:1]

        for before_name, after_name in one_way_dep or []:
            before_positions = resolve(before_name)
            after_positions = resolve(after_name)
            if not before_positions or not after_positions:
                continue
            source = unit_for[before_positions[0]]
            target = unit_for[after_positions[0]]
            if source != target and target not in edges[source]:
                edges[source].add(target)
                indegree[target] += 1

        unit_rank = [
            min(rank.get(position, position) for position in unit)
            for unit in units
        ]
        available = [i for i in range(len(units)) if indegree[i] == 0]
        result_units = []

        while available:
            available.sort(key=lambda i: (unit_rank[i], i))
            current = available.pop(0)
            result_units.append(current)
            for child in sorted(edges[current]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    available.append(child)

        # Cycles cannot satisfy every dependency.  Keep all columns safely and
        # deterministically rather than dropping data.
        if len(result_units) != len(units):
            remaining = [i for i in range(len(units)) if i not in set(result_units)]
            remaining.sort(key=lambda i: (unit_rank[i], i))
            result_units.extend(remaining)

        return [position for unit in result_units for position in units[unit]]

    def _conditional_orders(
        self,
        values: List[List[str]],
        global_order: List[int],
        max_depth: int,
        candidate_limit: int,
        columns: List[Any],
        col_merge: List[List[str]],
        one_way_dep: List[Tuple[str, str]],
    ) -> List[List[int]]:
        rows = len(values)
        cols = len(global_order)
        orders: List[List[int]] = [None] * rows
        candidates = global_order[:candidate_limit]
        node_budget = 256
        nodes_used = [0]

        def finish(row_ids: List[int], prefix: List[int], remaining: List[int]) -> None:
            tail = [column for column in global_order if column in remaining]
            order = self._constraint_order(
                prefix + tail, columns, col_merge, one_way_dep
            )
            for row_id in row_ids:
                orders[row_id] = order

        def visit(row_ids: List[int], remaining: List[int], prefix: List[int]) -> None:
            if (len(row_ids) < 2 or not remaining or len(prefix) >= max_depth
                    or nodes_used[0] >= node_budget):
                finish(row_ids, prefix, remaining)
                return

            nodes_used[0] += 1
            best_column = None
            best_weight = 0
            for column in candidates:
                if column not in remaining:
                    continue
                counts = Counter(values[row_id][column] for row_id in row_ids)
                weight = sum(
                    len(value) * frequency * (frequency - 1)
                    for value, frequency in counts.items()
                    if frequency > 1
                )
                if weight > best_weight or (
                    weight == best_weight and best_column is not None
                    and global_order.index(column) < global_order.index(best_column)
                ):
                    best_column = column
                    best_weight = weight

            if best_column is None or best_weight <= 0:
                finish(row_ids, prefix, remaining)
                return

            partitions: Dict[str, List[int]] = {}
            for row_id in row_ids:
                partitions.setdefault(values[row_id][best_column], []).append(row_id)

            next_remaining = [c for c in remaining if c != best_column]
            for partition in partitions.values():
                visit(partition, next_remaining, prefix + [best_column])

        visit(list(range(rows)), list(global_order), [])
        fallback = self._constraint_order(
            global_order, columns, col_merge, one_way_dep
        )
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
        # The arguments remain part of the public API.  parallel is deliberately
        # not used: deterministic local construction avoids row/index races.
        del early_stop, distinct_value_threshold, parallel

        rows, column_count = df.shape
        columns = list(df.columns)

        if column_count == 0:
            return df.copy(), [[] for _ in range(rows)]
        if rows == 0:
            return df.copy(), []

        values = [
            [self._string_value(df.iat[row, column])
             for column in range(column_count)]
            for row in range(rows)
        ]

        original = list(range(column_count))
        original = self._constraint_order(
            original, columns, col_merge, one_way_dep
        )

        frequency_scores = []
        for column in range(column_count):
            counts = Counter(values[row][column] for row in range(rows))
            score = sum(
                len(value) * frequency * (frequency - 1)
                for value, frequency in counts.items()
                if frequency > 1
            )
            frequency_scores.append(score)

        frequency_order = sorted(
            range(column_count),
            key=lambda column: (-frequency_scores[column], column),
        )
        frequency_order = self._constraint_order(
            frequency_order, columns, col_merge, one_way_dep
        )

        # Candidate one: existing order constrained safely.
        candidates = [original, frequency_order]

        # Candidate two: bounded conditional partitioning.  Do not construct a
        # large tree on exceptionally wide frames.
        depth = min(column_count, 8 if col_stop is None else max(0, col_stop))
        if depth > 0 and column_count > 1:
            conditional = self._conditional_orders(
                values=values,
                global_order=frequency_order,
                max_depth=depth,
                candidate_limit=min(column_count, 48),
                columns=columns,
                col_merge=col_merge,
                one_way_dep=one_way_dep,
            )
            candidates.append(conditional)

        best_score = -1
        best_orders = None
        best_is_global = False

        for candidate in candidates:
            if candidate and isinstance(candidate[0], int):
                row_orders = [candidate] * rows
                is_global = True
            else:
                row_orders = candidate
                is_global = False

            score = self._trie_score(self._serial_rows(values, row_orders))
            if score > best_score:
                best_score = score
                best_orders = row_orders
                best_is_global = is_global

        assert best_orders is not None

        if best_is_global:
            order = best_orders[0]
            reordered = df.iloc[:, order].copy()
            order_names = [columns[position] for position in order]
            return reordered, [list(order_names) for _ in range(rows)]

        # Row-specific output is positional.  The accompanying ordering records
        # exactly which original source column occupies every output position.
        output_rows = [
            [df.iat[row, column] for column in best_orders[row]]
            for row in range(rows)
        ]
        reordered = pd.DataFrame(
            output_rows, index=df.index.copy(), columns=columns, dtype=object
        )
        ordering_names = [
            [columns[position] for position in order]
            for order in best_orders
        ]
        return reordered, ordering_names

# EVOLVE-BLOCK-END