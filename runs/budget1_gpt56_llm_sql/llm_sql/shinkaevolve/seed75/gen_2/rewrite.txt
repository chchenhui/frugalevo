# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict
from collections import Counter


class Evolved(Algorithm):
    """
    Prefix-cache-oriented dataframe reordering.

    The returned dataframe stores each row's values in the order described by
    the corresponding entry in column_orderings.  Values themselves are never
    converted, merged, fabricated, or discarded.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _score_value(value) -> str:
        """Match evaluator serialization without modifying the stored value."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and missing:
                return ""
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Bounded binary-search LCP; string slice comparisons execute in C."""
        limit = min(len(left), len(right))
        if limit == 0:
            return 0
        if left == right:
            return limit

        low, high = 0, limit
        while low < high:
            mid = (low + high + 1) // 2
            if left[:mid] == right[:mid]:
                low = mid
            else:
                high = mid - 1
        return low

    def _trie_score(self, serialized_rows: List[str]) -> int:
        """Exact ideal serial Trie reuse for a fixed collection of strings."""
        if len(serialized_rows) < 2:
            return 0
        ordered = sorted(serialized_rows)
        return sum(
            self._lcp(ordered[i - 1], ordered[i])
            for i in range(1, len(ordered))
        )

    @staticmethod
    def _resolve_column(name, columns: List) -> int:
        """Resolve the API's historical exact-or-unique-substring column rule."""
        for index, column in enumerate(columns):
            if column == name:
                return index

        matches = [
            index for index, column in enumerate(columns)
            if str(name) in str(column)
        ]
        return matches[0] if len(matches) == 1 else -1

    def _make_units(self, columns: List, col_merge: List[List[str]]) -> List[List[int]]:
        """
        Merge constraints mean the listed source columns are kept adjacent.
        They are not physically merged: all cells and all original columns
        remain present in the result.
        """
        used = set()
        units = []

        for group in col_merge or []:
            unit = []
            for requested in group:
                position = self._resolve_column(requested, columns)
                if position >= 0 and position not in used:
                    unit.append(position)
                    used.add(position)
            if unit:
                units.append(unit)

        for position in range(len(columns)):
            if position not in used:
                units.append([position])

        return units

    def _dependency_edges(
        self,
        columns: List,
        units: List[List[int]],
        one_way_dep: List[Tuple[str, str]],
    ) -> List[Tuple[int, int]]:
        column_to_unit = {}
        for unit_index, unit in enumerate(units):
            for column_index in unit:
                column_to_unit[column_index] = unit_index

        edges = set()
        for before_name, after_name in one_way_dep or []:
            before = self._resolve_column(before_name, columns)
            after = self._resolve_column(after_name, columns)
            if before < 0 or after < 0:
                continue
            source = column_to_unit[before]
            target = column_to_unit[after]
            if source != target:
                edges.add((source, target))
        return sorted(edges)

    @staticmethod
    def _topological_units(
        proposed: List[int],
        unit_count: int,
        edges: List[Tuple[int, int]],
    ) -> List[int]:
        """
        Stable topological repair: proposed order is retained wherever it does
        not conflict with a declared one-way dependency.
        """
        rank = {unit: index for index, unit in enumerate(proposed)}
        incoming = [0] * unit_count
        outgoing = [[] for _ in range(unit_count)]

        for source, target in edges:
            outgoing[source].append(target)
            incoming[target] += 1

        available = [unit for unit in range(unit_count) if incoming[unit] == 0]
        available.sort(key=lambda unit: rank.get(unit, unit))
        result = []

        while available:
            unit = available.pop(0)
            result.append(unit)
            for target in outgoing[unit]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    available.append(target)
            available.sort(key=lambda candidate: rank.get(candidate, candidate))

        # Cyclic dependencies cannot all be honored.  Keep deterministic,
        # valid permutations rather than losing data or raising unexpectedly.
        if len(result) != unit_count:
            seen = set(result)
            result.extend(unit for unit in proposed if unit not in seen)

        return result

    def _candidate_strings(
        self,
        cell_strings: List[List[str]],
        row_orders: List[List[int]],
    ) -> List[str]:
        return [
            "".join(cell_strings[row][column] for column in row_orders[row])
            for row in range(len(row_orders))
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
        # Do not mutate the caller's dataframe or normalize its stored values.
        row_count, column_count = df.shape
        columns = list(df.columns)

        if row_count == 0 or column_count == 0:
            return df.copy(), [columns[:] for _ in range(row_count)]

        # Materialize evaluator-compatible text once.  Object access preserves
        # mixed source values and avoids pandas coercion of the returned frame.
        source_values = df.to_numpy(dtype=object, copy=False)
        cell_strings = [
            [self._score_value(source_values[row, col]) for col in range(column_count)]
            for row in range(row_count)
        ]

        units = self._make_units(columns, col_merge)
        unit_count = len(units)
        edges = self._dependency_edges(columns, units, one_way_dep)

        # Unit strings and integer factorization are shared by all candidates.
        unit_strings = []
        unit_codes = []
        unit_value_lengths = []
        unit_repetition_scores = []

        for unit in units:
            values = ["".join(cell_strings[row][col] for col in unit) for row in range(row_count)]
            codes, uniques = pd.factorize(values, sort=False)
            lengths = np.asarray([len(value) for value in uniques], dtype=np.int64)
            counts = np.bincount(codes, minlength=len(uniques)).astype(np.int64)

            unit_strings.append(values)
            unit_codes.append(codes)
            unit_value_lengths.append(lengths)
            unit_repetition_scores.append(
                int(np.sum(lengths * counts * np.maximum(counts - 1, 0)))
            )

        original_units = self._topological_units(
            list(range(unit_count)), unit_count, edges
        )

        # Required global alternative: length-weighted pair repetition.
        global_units = self._topological_units(
            sorted(
                range(unit_count),
                key=lambda unit: (-unit_repetition_scores[unit], unit),
            ),
            unit_count,
            edges,
        )

        def expand(unit_order: List[int]) -> List[int]:
            return [column for unit in unit_order for column in units[unit]]

        candidates = []
        candidates.append([expand(original_units) for _ in range(row_count)])
        candidates.append([expand(global_units) for _ in range(row_count)])

        # Conditional partition candidate.  Work is deliberately bounded: only
        # the most promising global fields participate in the recursive prefix.
        candidate_limit = min(unit_count, 32)
        depth_limit = min(
            candidate_limit,
            col_stop if col_stop is not None and col_stop > 0 else 8,
            8,
        )
        branch_limit = row_stop if row_stop is not None and row_stop > 0 else 256
        selectable = set(
            sorted(
                range(unit_count),
                key=lambda unit: (-unit_repetition_scores[unit], unit),
            )[:candidate_limit]
        )
        conditional_orders = [None] * row_count
        nodes_used = [0]

        def build_group(rows: List[int], remaining: List[int], prefix: List[int], depth: int):
            if (
                len(rows) <= 1
                or not remaining
                or depth >= depth_limit
                or nodes_used[0] >= branch_limit
            ):
                tail = self._topological_units(
                    [unit for unit in global_units if unit in remaining],
                    unit_count,
                    [(a, b) for a, b in edges if a in remaining and b in remaining],
                )
                order = prefix + tail
                flat = expand(order)
                for row in rows:
                    conditional_orders[row] = flat
                return

            choices = [unit for unit in remaining if unit in selectable]
            if not choices:
                build_group(rows, [], prefix + remaining, depth_limit)
                return

            best_unit = None
            best_score = 0
            row_array = np.asarray(rows, dtype=np.int64)

            for unit in choices:
                local_codes = unit_codes[unit][row_array]
                counts = np.bincount(
                    local_codes,
                    minlength=len(unit_value_lengths[unit]),
                ).astype(np.int64)
                score = int(np.sum(
                    unit_value_lengths[unit] * counts * np.maximum(counts - 1, 0)
                ))
                if score > best_score or (
                    score == best_score and best_unit is not None and unit < best_unit
                ):
                    best_score = score
                    best_unit = unit

            if best_unit is None or best_score <= 0:
                build_group(rows, [], prefix + remaining, depth_limit)
                return

            nodes_used[0] += 1
            next_remaining = [unit for unit in remaining if unit != best_unit]
            partitions = {}
            for row in rows:
                code = int(unit_codes[best_unit][row])
                partitions.setdefault(code, []).append(row)

            for code in sorted(partitions):
                build_group(
                    partitions[code],
                    next_remaining,
                    prefix + [best_unit],
                    depth + 1,
                )

        build_group(
            list(range(row_count)),
            global_units[:],
            [],
            0,
        )

        for row in range(row_count):
            if conditional_orders[row] is None:
                conditional_orders[row] = expand(global_units)
        candidates.append(conditional_orders)

        # Select using actual serialized strings and the exact sorted-LCP score.
        best_orders = candidates[0]
        best_strings = self._candidate_strings(cell_strings, best_orders)
        best_score = self._trie_score(best_strings)

        for candidate in candidates[1:]:
            serialized = self._candidate_strings(cell_strings, candidate)
            score = self._trie_score(serialized)
            if score > best_score:
                best_orders = candidate
                best_strings = serialized
                best_score = score

        # Trie reuse is insertion-order independent, but lexical order is a
        # deterministic cache-friendly output order.  Preserve the original
        # dataframe index so row identity remains available to callers.
        row_order = sorted(range(row_count), key=lambda row: (best_strings[row], row))
        output_values = [
            [source_values[row, column] for column in best_orders[row]]
            for row in row_order
        ]
        output_index = [df.index[row] for row in row_order]

        reordered = pd.DataFrame(
            output_values,
            index=output_index,
            columns=df.columns,
            dtype=object,
        )
        column_orderings = [
            [columns[column] for column in best_orders[row]]
            for row in row_order
        ]

        assert reordered.shape == df.shape
        assert len(column_orderings) == len(reordered)
        return reordered, column_orderings


# EVOLVE-BLOCK-END