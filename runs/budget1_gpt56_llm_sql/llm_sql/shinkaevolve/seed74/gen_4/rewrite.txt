# EVOLVE-BLOCK-START
import math
from collections import Counter, defaultdict
from typing import Tuple, List

import pandas as pd
from solver import Algorithm


class Evolved(Algorithm):
    """
    Prefix-cache-oriented row/column reorderer.

    The returned dataframe contains the same rows and values as the input.
    For a global ordering, dataframe columns are reordered normally.  For a
    row-specific ordering, values are placed in output positions according to
    the corresponding entry in column_orderings.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _serial_value(value) -> str:
        """Match evaluator serialization without changing stored dataframe values."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, bool) and missing:
                return ""
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        if left == right:
            return len(left)
        limit = min(len(left), len(right))
        if limit == 0:
            return 0

        # Binary-searching slices keeps comparisons in optimized C code and
        # avoids a Python loop over long text fields.
        lo, hi = 0, limit
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
        return sum(self._lcp(ordered[i - 1], ordered[i]) for i in range(1, len(ordered)))

    @staticmethod
    def _resolve_column(reference, columns):
        if reference in columns:
            return reference
        matches = [column for column in columns if str(reference) in str(column)]
        return matches[0] if len(matches) == 1 else None

    def _constrained_order(
        self,
        preferred: List,
        columns: List,
        col_merge: List[List],
        one_way_dep: List[Tuple],
    ) -> List:
        """
        Keep merge groups contiguous and apply dependency precedence using a
        stable topological ordering.  Invalid constraints are ignored rather
        than risking loss or duplication of columns.
        """
        rank = {column: i for i, column in enumerate(preferred)}
        used = set()
        units = []

        for requested_group in col_merge or []:
            members = []
            for name in requested_group:
                resolved = self._resolve_column(name, columns)
                if resolved is not None and resolved not in used:
                    members.append(resolved)
                    used.add(resolved)
            if members:
                members.sort(key=lambda value: rank.get(value, len(columns)))
                units.append(members)

        for column in columns:
            if column not in used:
                units.append([column])

        unit_for = {}
        for unit_index, unit in enumerate(units):
            for column in unit:
                unit_for[column] = unit_index

        edges = defaultdict(set)
        indegree = [0] * len(units)
        for before, after in one_way_dep or []:
            left = self._resolve_column(before, columns)
            right = self._resolve_column(after, columns)
            if left is None or right is None:
                continue
            source, target = unit_for[left], unit_for[right]
            if source != target and target not in edges[source]:
                edges[source].add(target)
                indegree[target] += 1

        unit_rank = [
            min(rank.get(column, len(columns)) for column in unit)
            for unit in units
        ]
        available = [index for index, value in enumerate(indegree) if value == 0]
        available.sort(key=lambda index: unit_rank[index])
        ordered_units = []

        while available:
            current = available.pop(0)
            ordered_units.append(current)
            for target in sorted(edges[current], key=lambda index: unit_rank[index]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    available.append(target)
                    available.sort(key=lambda index: unit_rank[index])

        # Cycles cannot satisfy every dependency; preserve all columns with a
        # deterministic preferred-order fallback.
        if len(ordered_units) != len(units):
            missing = [i for i in range(len(units)) if i not in set(ordered_units)]
            ordered_units.extend(sorted(missing, key=lambda index: unit_rank[index]))

        return [column for unit_index in ordered_units for column in units[unit_index]]

    def _global_frequency_order(self, strings, columns) -> List:
        scores = []
        for col_index, column in enumerate(columns):
            counts = Counter(strings[row][col_index] for row in range(len(strings)))
            score = sum(
                len(value) * count * (count - 1)
                for value, count in counts.items()
                if count > 1
            )
            scores.append((score, -col_index, column))
        scores.sort(reverse=True)
        return [column for _, _, column in scores]

    def _conditional_orders(self, strings, columns, global_order) -> List[List]:
        """
        Build a bounded conditional prefix partition tree.  It uses
        length-weighted pair repetition inside each group and then gives each
        leaf a cheap deterministic global-frequency tail.
        """
        row_count = len(strings)
        col_count = len(columns)
        if row_count == 0 or col_count == 0:
            return [[] for _ in range(row_count)]

        # Factorized serialized values are reused at every node.
        codes = []
        for col_index in range(col_count):
            mapping = {}
            next_code = 0
            current = []
            for row in range(row_count):
                value = strings[row][col_index]
                if value not in mapping:
                    mapping[value] = next_code
                    next_code += 1
                current.append(mapping[value])
            codes.append(current)

        global_indices = [columns.index(column) for column in global_order]
        paths = [[] for _ in range(row_count)]
        candidate_columns = global_indices[: min(col_count, 48)]
        max_depth = min(col_count, 10)
        max_nodes = 256
        node_count = 0

        def visit(rows, remaining, depth):
            nonlocal node_count
            if (
                len(rows) < 2
                or not remaining
                or depth >= max_depth
                or node_count >= max_nodes
            ):
                return
            node_count += 1

            best_column = None
            best_score = 0
            for col_index in remaining:
                counts = Counter(codes[col_index][row] for row in rows)
                score = 0
                for code, count in counts.items():
                    if count > 1:
                        representative = strings[rows[0]][col_index]
                        # Find the representative only when the code occurs;
                        # this keeps string storage shared with `strings`.
                        for row in rows:
                            if codes[col_index][row] == code:
                                representative = strings[row][col_index]
                                break
                        score += len(representative) * count * (count - 1)
                if score > best_score or (
                    score == best_score
                    and best_column is not None
                    and col_index < best_column
                ):
                    best_score = score
                    best_column = col_index

            if best_column is None or best_score <= 0:
                return

            for row in rows:
                paths[row].append(best_column)

            buckets = defaultdict(list)
            for row in rows:
                buckets[codes[best_column][row]].append(row)

            next_remaining = [value for value in remaining if value != best_column]
            for bucket in buckets.values():
                if len(bucket) > 1:
                    visit(bucket, next_remaining, depth + 1)

        visit(list(range(row_count)), candidate_columns, 0)

        result = []
        for row in range(row_count):
            selected = paths[row]
            selected_set = set(selected)
            result.append(selected + [index for index in global_indices if index not in selected_set])
        return result

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
        # Never mutate caller-owned data.
        source = df.copy(deep=True)
        columns = list(source.columns)
        row_count, col_count = source.shape

        if row_count == 0:
            return source, []
        if col_count == 0:
            return source, [[] for _ in range(row_count)]

        # Cache evaluator-equivalent cell serialization once.
        values = source.to_numpy(dtype=object, copy=True)
        serialized_cells = [
            [self._serial_value(values[row, col]) for col in range(col_count)]
            for row in range(row_count)
        ]

        natural = self._constrained_order(
            list(columns), columns, col_merge, one_way_dep
        )
        frequency = self._constrained_order(
            self._global_frequency_order(serialized_cells, columns),
            columns,
            col_merge,
            one_way_dep,
        )

        candidates = []
        for order in (natural, frequency):
            index_order = [columns.index(column) for column in order]
            strings = [
                "".join(serialized_cells[row][col] for col in index_order)
                for row in range(row_count)
            ]
            candidates.append((self._trie_score(strings), "global", order))

        # The conditional candidate is bounded and is skipped for very wide
        # tables where a global order is more reliable for runtime.
        if col_count <= 128:
            raw_orders = self._conditional_orders(serialized_cells, columns, frequency)
            conditional_orders = [
                self._constrained_order(
                    [columns[index] for index in order],
                    columns,
                    col_merge,
                    one_way_dep,
                )
                for order in raw_orders
            ]
            strings = [
                "".join(
                    serialized_cells[row][columns.index(column)]
                    for column in conditional_orders[row]
                )
                for row in range(row_count)
            ]
            candidates.append(
                (self._trie_score(strings), "conditional", conditional_orders)
            )

        # Stable tie-breaking favors conventional global dataframe columns.
        best_score, kind, payload = max(
            candidates,
            key=lambda item: (item[0], 1 if item[1] == "global" else 0),
        )

        if kind == "global":
            order = payload
            reordered = source.loc[:, order].copy()
            return reordered, [list(order) for _ in range(row_count)]

        row_orders = payload
        output = []
        for row in range(row_count):
            output.append([
                values[row, columns.index(column)]
                for column in row_orders[row]
            ])

        # Output labels are positional for row-specific layouts; each matching
        # entry in column_orderings identifies the original source column.
        reordered = pd.DataFrame(output, index=source.index, columns=columns, dtype=object)
        return reordered, row_orders


# EVOLVE-BLOCK-END