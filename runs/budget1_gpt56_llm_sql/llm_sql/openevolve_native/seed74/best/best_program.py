# EVOLVE-BLOCK-START
"""
Fast bounded dataframe column-permutation optimizer for LLM prefix caching.

The algorithm preserves every row, column, index, and cell value.  It considers
two bounded ordering strategies:

1. A global dependency-aware column-unit ordering ranked by length-weighted
   repeated serialized values.
2. A shallow conditional prefix partition tree that may select different leading
   units for different row groups.

The two candidates are compared using the exact ideal serial-Trie objective:
the sum of longest-common-prefix lengths between lexicographically adjacent
serialized rows.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from solver import Algorithm


class Evolved(Algorithm):
    """Bounded prefix-cache-oriented dataframe reordering."""

    def __init__(self, df: pd.DataFrame = None):
        """Store an optional dataframe reference without mutating it."""
        self.df = df

    @staticmethod
    def _cell_string(value) -> str:
        """Serialize one scalar using evaluator-compatible missing-value rules."""
        if value is None:
            return ""

        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and bool(missing):
                return ""
        except Exception:
            pass

        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Compute the character LCP using binary-searched slice comparisons."""
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

    def _trie_score(self, strings: List[str]) -> int:
        """Return exact ideal serial-Trie reuse from sorted adjacent row strings."""
        if len(strings) < 2:
            return 0

        strings.sort()
        score = 0
        previous = strings[0]

        for current in strings[1:]:
            score += self._lcp(previous, current)
            previous = current

        return score

    @staticmethod
    def _normalize_groups(col_merge) -> List[List]:
        """Normalize merge declarations while tolerating scalar or malformed input."""
        if not col_merge:
            return []

        result = []

        for group in col_merge:
            if isinstance(group, (list, tuple)):
                if group:
                    result.append(list(group))
            else:
                result.append([group])

        return result

    @staticmethod
    def _build_units(
        columns: List,
        col_merge,
        dependencies,
    ) -> Tuple[List[List[int]], Dict[int, set]]:
        """
        Build contiguous source-column units and unit-level predecessor edges.

        Merge groups are retained as indivisible contiguous units.  Dependency
        edges are mapped to units without altering any dataframe value.
        """
        positions_by_name = defaultdict(list)
        for position, name in enumerate(columns):
            positions_by_name[name].append(position)

        units = []
        used = set()

        for merge_group in Evolved._normalize_groups(col_merge):
            positions = []

            for name in merge_group:
                for position in positions_by_name.get(name, []):
                    if position not in used:
                        positions.append(position)
                        used.add(position)

            if positions:
                units.append(positions)

        for position in range(len(columns)):
            if position not in used:
                units.append([position])

        position_to_unit = {}
        for unit_id, unit in enumerate(units):
            for position in unit:
                position_to_unit[position] = unit_id

        predecessors = {unit_id: set() for unit_id in range(len(units))}

        for pair in dependencies or []:
            if not isinstance(pair, (tuple, list)) or len(pair) < 2:
                continue

            before, after = pair[0], pair[1]

            for before_position in positions_by_name.get(before, []):
                for after_position in positions_by_name.get(after, []):
                    left = position_to_unit[before_position]
                    right = position_to_unit[after_position]

                    if left != right:
                        predecessors[right].add(left)

        return units, predecessors

    @staticmethod
    def _priority_topological_order(
        unit_count: int,
        predecessors: Dict[int, set],
        priority: List[int],
    ) -> List[int]:
        """
        Produce a deterministic priority-biased topological order.

        Cyclic dependency declarations cannot be fully satisfied, so they are
        broken deterministically rather than causing data loss or failure.
        """
        rank = {unit: index for index, unit in enumerate(priority)}
        remaining = set(range(unit_count))
        completed = set()
        result = []

        while remaining:
            available = [
                unit
                for unit in remaining
                if predecessors.get(unit, set()).issubset(completed)
            ]

            if not available:
                available = list(remaining)

            chosen = min(
                available,
                key=lambda unit: (rank.get(unit, unit_count), unit),
            )

            result.append(chosen)
            remaining.remove(chosen)
            completed.add(chosen)

        return result

    @staticmethod
    def _factorize(values: List[str]) -> Tuple[List[int], List[int], List[int]]:
        """
        Factorize serialized values into row codes, code lengths, and frequencies.
        """
        mapping = {}
        codes = []
        lengths = []
        counts = []

        for value in values:
            code = mapping.get(value)

            if code is None:
                code = len(lengths)
                mapping[value] = code
                lengths.append(len(value))
                counts.append(0)

            codes.append(code)
            counts[code] += 1

        return codes, lengths, counts

    @staticmethod
    def _pair_score(lengths: List[int], counts: List[int]) -> int:
        """Compute length-weighted repeated-value pair contribution for one unit."""
        return sum(
            length * count * (count - 1)
            for length, count in zip(lengths, counts)
            if length and count > 1
        )

    @staticmethod
    def _serialize_global(
        unit_values: List[List[str]],
        order: List[int],
        n_rows: int,
    ) -> List[str]:
        """Serialize all rows under one common unit order."""
        return [
            "".join(unit_values[unit][row] for unit in order)
            for row in range(n_rows)
        ]

    @staticmethod
    def _serialize_orders(
        unit_values: List[List[str]],
        orders: List[List[int]],
    ) -> List[str]:
        """Serialize rows under potentially row-specific unit orders."""
        return [
            "".join(unit_values[unit][row] for unit in orders[row])
            for row in range(len(orders))
        ]

    def _conditional_candidate(
        self,
        codes_by_unit: List[List[int]],
        lengths_by_unit: List[List[int]],
        global_order: List[int],
        predecessors: Dict[int, set],
        col_stop,
        early_stop,
    ) -> List[List[int]]:
        """
        Build a shallow conditional prefix partition candidate.

        At each tree level, each active row group chooses the legal remaining
        candidate unit with the greatest local length-weighted pair repetition.
        The group is partitioned by its pre-factorized integer code.  Every row
        receives a deterministic global-order suffix afterward.
        """
        unit_count = len(codes_by_unit)
        n_rows = len(codes_by_unit[0]) if unit_count else 0
        orders = [[] for _ in range(n_rows)]

        if not n_rows or not unit_count:
            return orders

        candidate_limit = min(12, unit_count)
        candidate_set = set(global_order[:candidate_limit])

        changed = True
        while changed:
            changed = False
            for unit in tuple(candidate_set):
                for predecessor in predecessors.get(unit, set()):
                    if predecessor not in candidate_set:
                        candidate_set.add(predecessor)
                        changed = True

        candidates = [unit for unit in global_order if unit in candidate_set]

        max_depth = min(4, len(candidates))
        if col_stop is not None:
            try:
                if int(col_stop) > 0:
                    max_depth = min(max_depth, int(col_stop))
            except Exception:
                pass

        try:
            minimum_score = max(0, int(early_stop or 0))
        except Exception:
            minimum_score = 0

        groups = [(list(range(n_rows)), frozenset())]

        for _ in range(max_depth):
            next_groups = []
            progressed = False

            for rows, completed in groups:
                if len(rows) < 2:
                    next_groups.append((rows, completed))
                    continue

                available = [
                    unit
                    for unit in candidates
                    if unit not in completed
                    and predecessors.get(unit, set()).issubset(completed)
                ]

                if not available:
                    next_groups.append((rows, completed))
                    continue

                best_unit = None
                best_score = 0

                for unit in available:
                    local_counts = Counter(
                        codes_by_unit[unit][row]
                        for row in rows
                    )
                    lengths = lengths_by_unit[unit]

                    score = sum(
                        lengths[code] * count * (count - 1)
                        for code, count in local_counts.items()
                        if count > 1 and lengths[code]
                    )

                    if (
                        score > best_score
                        or (
                            score == best_score
                            and best_unit is not None
                            and unit < best_unit
                        )
                    ):
                        best_score = score
                        best_unit = unit

                if best_unit is None or best_score <= minimum_score:
                    next_groups.append((rows, completed))
                    continue

                progressed = True
                unit_codes = codes_by_unit[best_unit]

                for row in rows:
                    orders[row].append(best_unit)

                partitions = defaultdict(list)
                for row in rows:
                    partitions[unit_codes[row]].append(row)

                updated = frozenset(set(completed) | {best_unit})

                for partition_rows in partitions.values():
                    next_groups.append((partition_rows, updated))

            groups = next_groups

            if not progressed:
                break

        for row, prefix in enumerate(orders):
            used = set(prefix)
            prefix.extend(unit for unit in global_order if unit not in used)

        return orders

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
        Return the best bounded global or conditional prefix-cache ordering.

        Cells are only permuted within their original row.  The returned
        dataframe has the same shape, index, and column labels as the input;
        each entry of column_orderings identifies the original source columns
        used to form that returned row.
        """
        n_rows, n_cols = df.shape
        columns = list(df.columns)

        if n_rows == 0 or n_cols == 0:
            return df.copy(), [columns[:] for _ in range(n_rows)]

        raw_values = df.to_numpy(dtype=object, copy=False)
        units, predecessors = self._build_units(
            columns,
            col_merge,
            one_way_dep,
        )

        cell_strings = [
            [
                self._cell_string(raw_values[row, column])
                for row in range(n_rows)
            ]
            for column in range(n_cols)
        ]

        unit_values = []
        codes_by_unit = []
        lengths_by_unit = []
        global_scores = []

        for unit in units:
            if len(unit) == 1:
                values = cell_strings[unit[0]]
            else:
                values = [
                    "".join(cell_strings[column][row] for column in unit)
                    for row in range(n_rows)
                ]

            codes, lengths, counts = self._factorize(values)

            unit_values.append(values)
            codes_by_unit.append(codes)
            lengths_by_unit.append(lengths)
            global_scores.append(self._pair_score(lengths, counts))

        priority = sorted(
            range(len(units)),
            key=lambda unit: (-global_scores[unit], unit),
        )

        global_order = self._priority_topological_order(
            len(units),
            predecessors,
            priority,
        )

        use_conditional = (
            n_rows >= 2
            and len(units) >= 2
            and n_rows <= 30000
            and n_rows * len(units) <= 180000
        )

        best_orders = None
        use_global = True

        if use_conditional:
            conditional_orders = self._conditional_candidate(
                codes_by_unit,
                lengths_by_unit,
                global_order,
                predecessors,
                col_stop,
                early_stop,
            )

            global_score = self._trie_score(
                self._serialize_global(unit_values, global_order, n_rows)
            )

            conditional_score = self._trie_score(
                self._serialize_orders(unit_values, conditional_orders)
            )

            if conditional_score > global_score:
                best_orders = conditional_orders
                use_global = False

        global_positions = [
            position
            for unit in global_order
            for position in units[unit]
        ]

        if use_global:
            output = raw_values[:, global_positions].copy()
            ordering = [columns[position] for position in global_positions]
            column_orderings = [ordering[:] for _ in range(n_rows)]
        else:
            output = np.empty((n_rows, n_cols), dtype=object)
            column_orderings = []

            for row in range(n_rows):
                positions = [
                    position
                    for unit in best_orders[row]
                    for position in units[unit]
                ]

                output[row, :] = raw_values[row, positions]
                column_orderings.append(
                    [columns[position] for position in positions]
                )

        reordered = pd.DataFrame(
            output,
            index=df.index.copy(),
            columns=df.columns.copy(),
            dtype=object,
        )

        return reordered, column_orderings


# EVOLVE-BLOCK-END