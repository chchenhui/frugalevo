# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List


class Evolved(Algorithm):
    """
    Bounded prompt-prefix optimizer.

    Values in the returned frame are always original values.  For row-specific
    orders, output position j contains the source value from
    column_orderings[row][j].
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _score_string(value) -> str:
        """Match evaluator normalization without modifying the stored value."""
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
        high = min(len(left), len(right))
        low = 0
        # Slice comparison delegates the character comparison to CPython/C
        # rather than performing a Python loop over every character.
        while low < high:
            mid = (low + high + 1) // 2
            if left[:mid] == right[:mid]:
                low = mid
            else:
                high = mid - 1
        return low

    def _trie_score(self, strings: List[str]) -> int:
        if len(strings) < 2:
            return 0
        ordered = sorted(strings)
        return sum(self._lcp(ordered[i - 1], ordered[i])
                   for i in range(1, len(ordered)))

    @staticmethod
    def _resolve_column(columns, requested):
        """Support the historical API's exact-or-unique-substring convention."""
        if requested in columns:
            return requested
        matches = [column for column in columns if str(requested) in str(column)]
        return matches[0] if len(matches) == 1 else None

    def _apply_constraints(self, order, columns, col_merge, one_way_dep):
        """
        Keep merge sets adjacent and dependency sources before destinations.
        These constraints affect only permutations, never values.
        """
        rank = {column: i for i, column in enumerate(order)}
        merge_groups = []
        for group in col_merge or []:
            resolved = [self._resolve_column(columns, item) for item in group]
            resolved = [item for item in resolved if item is not None]
            if len(resolved) > 1:
                merge_groups.append(resolved)

        dependencies = []
        for source, destination in one_way_dep or []:
            source = self._resolve_column(columns, source)
            destination = self._resolve_column(columns, destination)
            if source is not None and destination is not None and source != destination:
                dependencies.append((source, destination))

        # A few stable passes are sufficient for normal API constraints and
        # avoid unbounded work on malformed cyclic dependency inputs.
        result = list(order)
        for _ in range(3):
            for group in merge_groups:
                members = [column for column in result if column in group]
                if len(members) < 2:
                    continue
                first = min(result.index(column) for column in members)
                result = [column for column in result if column not in group]
                members.sort(key=lambda column: rank.get(column, len(rank)))
                result[first:first] = members

            for source, destination in dependencies:
                source_pos = result.index(source)
                destination_pos = result.index(destination)
                if source_pos > destination_pos:
                    result.pop(source_pos)
                    destination_pos = result.index(destination)
                    result.insert(destination_pos, source)
        return result

    def _global_order(self, cell_strings, columns, col_merge, one_way_dep):
        """Length-weighted pair-repetition ordering requested by the API."""
        rows = len(cell_strings)
        scores = []
        for column_index, column in enumerate(columns):
            counts = {}
            lengths = {}
            for row_index in range(rows):
                value = cell_strings[row_index][column_index]
                counts[value] = counts.get(value, 0) + 1
                lengths[value] = len(value)
            score = sum(lengths[value] * count * (count - 1)
                        for value, count in counts.items())
            scores.append((score, column_index, column))

        scores.sort(key=lambda item: (-item[0], item[1]))
        order = [item[2] for item in scores]
        return self._apply_constraints(order, columns, col_merge, one_way_dep)

    def _conditional_orders(self, cell_strings, columns, base_order,
                            col_merge, one_way_dep, row_stop, col_stop):
        """
        A bounded conditional partition tree.  It chooses leading fields inside
        a group using actual length-weighted repeated serialized values.
        """
        row_count = len(cell_strings)
        col_count = len(columns)
        position = {column: index for index, column in enumerate(columns)}
        base_positions = [position[column] for column in base_order]
        orders = [None] * row_count

        max_depth = col_count if col_stop is None or col_stop <= 0 else min(col_count, col_stop)
        max_depth = min(max_depth, 12)
        # Bound wide-table and high-cardinality work.
        candidate_limit = min(col_count, 48)
        preferred = base_positions[:candidate_limit]
        node_budget = 128
        nodes_used = [0]

        def finish(indices, prefix, remaining):
            tail = [p for p in base_positions if p in remaining]
            raw = prefix + tail
            labels = [columns[p] for p in raw]
            labels = self._apply_constraints(labels, columns, col_merge, one_way_dep)
            final_positions = [position[label] for label in labels]
            for index in indices:
                orders[index] = final_positions

        def visit(indices, prefix, remaining, depth):
            if (len(indices) <= 1 or not remaining or depth >= max_depth or
                    nodes_used[0] >= node_budget):
                finish(indices, prefix, remaining)
                return

            candidates = [p for p in preferred if p in remaining]
            if not candidates:
                finish(indices, prefix, remaining)
                return

            best_column = None
            best_score = 0
            for column in candidates:
                counts = {}
                lengths = {}
                for row in indices:
                    value = cell_strings[row][column]
                    counts[value] = counts.get(value, 0) + 1
                    lengths[value] = len(value)
                score = sum(lengths[value] * count * (count - 1)
                            for value, count in counts.items())
                if score > best_score or (score == best_score and
                                          best_column is not None and column < best_column):
                    best_score = score
                    best_column = column

            if best_column is None or best_score <= 0:
                finish(indices, prefix, remaining)
                return

            buckets = {}
            for row in indices:
                buckets.setdefault(cell_strings[row][best_column], []).append(row)

            nodes_used[0] += 1
            new_remaining = [p for p in remaining if p != best_column]
            new_prefix = prefix + [best_column]
            for value in sorted(buckets):
                visit(buckets[value], new_prefix, new_remaining, depth + 1)

        visit(list(range(row_count)), [], list(range(col_count)), 0)
        for row in range(row_count):
            if orders[row] is None:
                finish([row], [], list(range(col_count)))
        return orders

    def _materialize(self, df, source_values, cell_strings, orders,
                     output_columns, output_positions):
        serialized = [
            "".join(cell_strings[row][column] for column in order)
            for row, order in enumerate(orders)
        ]
        row_indices = sorted(range(len(orders)), key=lambda row: (serialized[row], row))

        values = []
        returned_orders = []
        for row in row_indices:
            order = orders[row]
            values.append([source_values[row][column] for column in order])
            returned_orders.append([df.columns[column] for column in order])

        result = pd.DataFrame(
            values,
            index=df.index.take(row_indices),
            columns=output_columns,
            dtype=object,
        )
        return result, returned_orders, self._trie_score(serialized)

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
        # Never mutate the caller's frame or create bookkeeping columns.
        if df.shape[0] == 0:
            return df.copy().astype(object), []

        columns = list(df.columns)
        source_values = df.astype(object).to_numpy(copy=True)
        cell_strings = [
            [self._score_string(source_values[row][column])
             for column in range(len(columns))]
            for row in range(len(source_values))
        ]

        global_labels = self._global_order(
            cell_strings, columns, col_merge, one_way_dep
        )
        position = {column: index for index, column in enumerate(columns)}
        global_positions = [position[column] for column in global_labels]
        global_orders = [list(global_positions) for _ in range(len(df))]

        # Candidate one: reliable global frequency order.
        best_df, best_orders, best_score = self._materialize(
            df, source_values, cell_strings, global_orders,
            global_labels, global_positions
        )

        # Candidate two: conditional leading-field partitions.  It is bounded
        # and is selected only if exact full-row Trie reuse improves.
        conditional_orders = self._conditional_orders(
            cell_strings, columns, global_labels, col_merge, one_way_dep,
            row_stop, col_stop
        )
        conditional_df, conditional_orderings, conditional_score = self._materialize(
            df, source_values, cell_strings, conditional_orders,
            columns, list(range(len(columns)))
        )

        if conditional_score > best_score:
            return conditional_df, conditional_orderings
        return best_df, best_orders


# EVOLVE-BLOCK-END