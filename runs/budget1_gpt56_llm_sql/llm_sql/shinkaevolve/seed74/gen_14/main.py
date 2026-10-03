# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from collections import Counter, defaultdict


class Evolved(Algorithm):
    """
    Bounded, data-preserving row/column reordering for serial prompt caching.

    The returned dataframe contains the original cells only.  For a global
    ordering, dataframe labels and column_orderings agree directly.  The
    conditional candidate may use row-specific orders; in that case the
    corresponding entry in column_orderings identifies the source column of
    every positional output cell.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _serialized_cells(df: pd.DataFrame):
        """Match evaluator normalization without modifying the returned data."""
        normalized = df.fillna("").astype(str)
        return normalized.to_numpy(dtype=object, copy=False)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        if left == right:
            return len(left)
        upper = min(len(left), len(right))
        low = 0
        # Slice comparisons are implemented in C and avoid Python character loops.
        while low < upper:
            mid = (low + upper + 1) // 2
            if left[:mid] == right[:mid]:
                low = mid
            else:
                upper = mid - 1
        return low

    def _trie_score(self, strings: List[str]) -> int:
        if len(strings) < 2:
            return 0
        ordered = sorted(strings)
        return sum(self._lcp(ordered[i - 1], ordered[i])
                   for i in range(1, len(ordered)))

    @staticmethod
    def _resolve_column(name, columns):
        """Resolve exact names first, then the legacy unique-substring form."""
        if name in columns:
            return name
        matches = [column for column in columns if str(name) in str(column)]
        return matches[0] if len(matches) == 1 else None

    def _apply_constraints(
        self,
        proposed: List[int],
        columns: List,
        col_merge: List[List[str]],
        one_way_dep: List[Tuple[str, str]],
    ) -> List[int]:
        """
        Keep requested merge groups contiguous and satisfy dependency order.
        Constraints only move columns; they never merge or modify cell values.
        """
        rank = {column: pos for pos, column in enumerate(proposed)}
        used = set()
        units = []

        for requested_group in col_merge or []:
            members = []
            for requested in requested_group:
                resolved = self._resolve_column(requested, columns)
                if resolved is not None:
                    index = columns.index(resolved)
                    if index not in used:
                        members.append(index)
                        used.add(index)
            if members:
                members.sort(key=lambda x: rank[x])
                units.append(members)

        for index in proposed:
            if index not in used:
                units.append([index])
                used.add(index)

        unit_of = {}
        for unit_index, unit in enumerate(units):
            for column_index in unit:
                unit_of[column_index] = unit_index

        edges = defaultdict(set)
        indegree = [0] * len(units)
        for before_name, after_name in one_way_dep or []:
            before = self._resolve_column(before_name, columns)
            after = self._resolve_column(after_name, columns)
            if before is None or after is None:
                continue
            source, target = unit_of[columns.index(before)], unit_of[columns.index(after)]
            if source != target and target not in edges[source]:
                edges[source].add(target)
                indegree[target] += 1

        # Stable Kahn ordering: retain heuristic order whenever constraints allow.
        unit_rank = {
            unit_index: min(rank[column] for column in unit)
            for unit_index, unit in enumerate(units)
        }
        available = sorted(
            [i for i, value in enumerate(indegree) if value == 0],
            key=lambda i: unit_rank[i],
        )
        ordered_units = []
        while available:
            current = available.pop(0)
            ordered_units.append(current)
            for target in sorted(edges[current], key=lambda i: unit_rank[i]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    available.append(target)
                    available.sort(key=lambda i: unit_rank[i])

        # A cyclic dependency has no valid topological arrangement.  Preserve
        # deterministic heuristic order for the remaining units rather than
        # dropping any source columns.
        if len(ordered_units) != len(units):
            remaining = [i for i in range(len(units)) if i not in set(ordered_units)]
            ordered_units.extend(sorted(remaining, key=lambda i: unit_rank[i]))

        return [column for unit in ordered_units for column in units[unit]]

    def _global_orders(self, text, columns, col_merge, one_way_dep):
        rows, width = text.shape
        pair_scores = []
        repeat_scores = []
        for column in range(width):
            counts = Counter(text[:, column])
            pair_scores.append(sum(
                len(value) * count * (count - 1)
                for value, count in counts.items()
            ))
            repeat_scores.append(sum(
                len(value) * max(0, count - 1)
                for value, count in counts.items()
            ))

        first = sorted(range(width), key=lambda c: (-pair_scores[c], c))
        second = sorted(range(width), key=lambda c: (-repeat_scores[c], c))
        first = self._apply_constraints(first, columns, col_merge, one_way_dep)
        second = self._apply_constraints(second, columns, col_merge, one_way_dep)
        return [first, second], pair_scores

    def _conditional_order(
        self,
        text,
        base_order,
        pair_scores,
        columns,
        col_merge,
        one_way_dep,
        col_stop,
    ):
        """
        Build a small conditional prefix partition tree.  It is deliberately
        bounded: at most 64 nodes, six levels, and 32 candidate fields/node.
        """
        row_count, width = text.shape
        prefixes = [[] for _ in range(row_count)]
        max_depth = min(width, 6 if col_stop is None else max(1, min(6, col_stop)))
        candidate_cap = min(width, 32)
        candidate_columns = sorted(range(width), key=lambda c: (-pair_scores[c], c))[:candidate_cap]
        node_budget = [64]

        def finish(indices, prefix, remaining):
            tail = [c for c in base_order if c in remaining]
            order = self._apply_constraints(prefix + tail, columns, col_merge, one_way_dep)
            for row in indices:
                prefixes[row] = order

        def visit(indices, remaining, prefix, depth):
            if (len(indices) < 2 or not remaining or depth >= max_depth
                    or node_budget[0] <= 0):
                finish(indices, prefix, remaining)
                return
            node_budget[0] -= 1

            choices = [c for c in candidate_columns if c in remaining]
            if not choices:
                choices = sorted(remaining)[:candidate_cap]

            best_column = None
            best_score = 0
            best_groups = None
            for column in choices:
                groups = defaultdict(list)
                for row in indices:
                    groups[text[row, column]].append(row)
                score = sum(
                    len(value) * len(group) * (len(group) - 1)
                    for value, group in groups.items()
                    if len(group) > 1
                )
                if score > best_score or (
                    score == best_score and best_column is not None and column < best_column
                ):
                    best_column, best_score, best_groups = column, score, groups

            if best_column is None or best_score <= 0:
                finish(indices, prefix, remaining)
                return

            next_remaining = set(remaining)
            next_remaining.remove(best_column)
            for value in sorted(best_groups, key=lambda value: str(value)):
                visit(best_groups[value], next_remaining, prefix + [best_column], depth + 1)

        visit(list(range(row_count)), set(range(width)), [], 0)
        return prefixes

    def _materialize(self, source, text, orders, columns, global_order=False):
        strings = [
            "".join(text[row, column] for column in orders[row])
            for row in range(len(orders))
        ]
        row_order = sorted(range(len(orders)), key=lambda row: (strings[row], row))
        values = source.to_numpy(dtype=object, copy=False)

        output_values = [
            [values[row, column] for column in orders[row]]
            for row in row_order
        ]
        output_index = source.index.take(row_order)
        output_columns = [columns[column] for column in orders[0]] if global_order and orders else columns
        output = pd.DataFrame(output_values, index=output_index, columns=output_columns, dtype=object)
        output_orders = [[columns[column] for column in orders[row]] for row in row_order]
        return output, output_orders, self._trie_score(strings)

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True):
        columns = list(df.columns)
        text = self._serialized_cells(df)
        orders, _ = self._global_orders(text, columns, [], [])
        order = orders[0]
        per_row = [order[:] for _ in range(len(df))]
        result, ordering, _ = self._materialize(df, text, per_row, columns, global_order=True)
        if not row_sort:
            result = result.reindex(df.index)
            ordering = [[columns[column] for column in order] for _ in range(len(df))]
        return result, ordering

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
        # Copying preserves caller-owned data while retaining exact source cells.
        source = df.copy(deep=True)
        rows, width = source.shape
        if rows == 0 or width == 0:
            return source, [[] for _ in range(rows)]

        columns = list(source.columns)
        text = self._serialized_cells(source)
        global_orders, pair_scores = self._global_orders(
            text, columns, col_merge, one_way_dep
        )

        candidates = []
        for order in global_orders:
            per_row = [order[:] for _ in range(rows)]
            candidates.append(
                self._materialize(source, text, per_row, columns, global_order=True)
            )

        # Third bounded alternative: conditional order choices inside shared groups.
        conditional = self._conditional_order(
            text, global_orders[0], pair_scores, columns, col_merge, one_way_dep, col_stop
        )
        candidates.append(
            self._materialize(source, text, conditional, columns, global_order=False)
        )

        # Deterministic first-candidate tie breaking avoids needless instability.
        best = max(enumerate(candidates), key=lambda item: (item[1][2], -item[0]))[1]
        result, orderings, _ = best

        if result.shape != source.shape or len(orderings) != len(source):
            raise RuntimeError("reordering must preserve dataframe shape and row metadata")
        return result, orderings
# EVOLVE-BLOCK-END