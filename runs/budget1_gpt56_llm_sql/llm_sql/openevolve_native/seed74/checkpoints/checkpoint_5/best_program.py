# EVOLVE-BLOCK-START
"""
Fast prefix-cache-oriented dataframe reordering.

This implementation preserves every source cell, row, index, shape, and column
label.  It primarily uses a global column-unit order ranked by length-weighted
value repetition, which is a cheap and reliable prefix-sharing heuristic.

For smaller inputs, it additionally constructs one bounded conditional prefix
partition candidate and selects between the global and conditional candidates
using the exact ideal character-Trie reuse objective: the adjacent LCP sum after
sorting serialized rows.  Large inputs deliberately avoid expensive per-row
candidate construction and repeated full-string sorting, improving runtime.

`column_orderings[i]` records the original source-column order used to create
returned row `i`.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from solver import Algorithm


class Evolved(Algorithm):
    """Bounded dataframe column permutation optimizer for serial prompt reuse."""

    def __init__(self, df: pd.DataFrame = None):
        """Store an optional dataframe reference without modifying it."""
        self.df = df

    @staticmethod
    def _cell_string(value) -> str:
        """Serialize one scalar with evaluator-compatible missing-value handling."""
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
        """Return character LCP using binary searched slice comparisons."""
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
        """Compute exact ideal Trie reuse from sorted adjacent-string LCPs."""
        if len(strings) < 2:
            return 0

        strings.sort()
        total = 0
        previous = strings[0]
        for current in strings[1:]:
            total += self._lcp(previous, current)
            previous = current
        return total

    @staticmethod
    def _build_units(
        columns: List,
        col_merge: List[List],
        dependencies: List[Tuple],
    ) -> Tuple[List[List[int]], Dict[int, set]]:
        """Create contiguous merge units and unit-level dependency predecessors."""
        name_to_position = {}
        for position, name in enumerate(columns):
            if name not in name_to_position:
                name_to_position[name] = position

        units = []
        used = set()

        for merge_group in col_merge or []:
            positions = []
            for name in merge_group:
                position = name_to_position.get(name)
                if position is not None and position not in used:
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
        for before, after in dependencies or []:
            before_position = name_to_position.get(before)
            after_position = name_to_position.get(after)
            if before_position is None or after_position is None:
                continue

            before_unit = position_to_unit[before_position]
            after_unit = position_to_unit[after_position]
            if before_unit != after_unit:
                predecessors[after_unit].add(before_unit)

        return units, predecessors

    @staticmethod
    def _priority_topological_order(
        unit_count: int,
        predecessors: Dict[int, set],
        priority: List[int],
    ) -> List[int]:
        """Produce a deterministic priority-biased topological unit order."""
        rank = {unit: position for position, unit in enumerate(priority)}
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
                # Cyclic dependency input cannot be fully honored.  Retain a
                # deterministic stable order rather than failing or dropping data.
                available = list(remaining)

            chosen = min(available, key=lambda unit: (rank[unit], unit))
            remaining.remove(chosen)
            completed.add(chosen)
            result.append(chosen)

        return result

    @staticmethod
    def _factorize(values: List[str]) -> Tuple[List[int], List[int], List[int]]:
        """Factorize strings into row codes, unique lengths, and code frequencies."""
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
        """Score a field by length-weighted repeated-value pairs."""
        return sum(
            length * count * (count - 1)
            for length, count in zip(lengths, counts)
            if count > 1 and length
        )

    @staticmethod
    def _serialize_global(
        unit_values: List[List[str]],
        order: List[int],
        n_rows: int,
    ) -> List[str]:
        """Serialize every row under one shared unit order."""
        return [
            "".join(unit_values[unit][row] for unit in order)
            for row in range(n_rows)
        ]

    @staticmethod
    def _serialize_orders(
        unit_values: List[List[str]],
        orders: List[List[int]],
    ) -> List[str]:
        """Serialize rows under their individually supplied unit orders."""
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
    ) -> List[List[int]]:
        """
        Build one shallow conditional prefix partition candidate.

        Groups choose a legal leading unit by local length-weighted pair
        repetition, partition by its pre-factorized value code, and stop after
        four levels.  Every remaining suffix follows the global order with
        already-selected units removed.
        """
        unit_count = len(codes_by_unit)
        n_rows = len(codes_by_unit[0]) if unit_count else 0
        orders = [[] for _ in range(n_rows)]

        if not n_rows or not unit_count:
            return orders

        candidate_limit = min(12, unit_count)
        candidate_set = set(global_order[:candidate_limit])

        # Include prerequisites so selected leading fields remain legal.
        changed = True
        while changed:
            changed = False
            for unit in tuple(candidate_set):
                for predecessor in predecessors.get(unit, set()):
                    if predecessor not in candidate_set:
                        candidate_set.add(predecessor)
                        changed = True

        candidates = [unit for unit in global_order if unit in candidate_set]
        groups = [(list(range(n_rows)), frozenset())]
        max_depth = min(4, len(candidates))

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
                    counts = Counter(codes_by_unit[unit][row] for row in rows)
                    lengths = lengths_by_unit[unit]
                    score = sum(
                        lengths[code] * count * (count - 1)
                        for code, count in counts.items()
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

                if best_unit is None or best_score <= 0:
                    next_groups.append((rows, completed))
                    continue

                progressed = True
                for row in rows:
                    orders[row].append(best_unit)

                partitions = defaultdict(list)
                unit_codes = codes_by_unit[best_unit]
                for row in rows:
                    partitions[unit_codes[row]].append(row)

                updated = frozenset(set(completed) | {best_unit})
                for partition_rows in partitions.values():
                    next_groups.append((partition_rows, updated))

            groups = next_groups
            if not progressed:
                break

        # A global topological order remains valid after removing already-used
        # prefix units because every conditional prefix was dependency legal.
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
        Return a data-preserving ordering optimized for serialized prefix reuse.

        A global repeated-value ordering is always constructed.  A second,
        shallow conditional candidate is only evaluated when its bounded work
        is cheap relative to the dataframe size.  This avoids the previous
        expensive per-row full ordering and repeated Trie scoring on large data.
        """
        n_rows, n_cols = df.shape
        columns = list(df.columns)

        if n_rows == 0 or n_cols == 0:
            return df.copy(), [columns[:] for _ in range(n_rows)]

        raw_values = df.to_numpy(dtype=object, copy=False)
        units, predecessors = self._build_units(columns, col_merge, one_way_dep)

        cell_strings = [
            [self._cell_string(raw_values[row, column]) for row in range(n_rows)]
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

        # Conditional grouping is valuable on modest inputs but can cost more
        # than its expected gain on wide or very large datasets.
        use_conditional = (
            n_rows >= 2
            and len(units) >= 2
            and n_rows * len(units) <= 180000
            and n_rows <= 30000
        )

        best_orders = None
        best_is_global = True

        if use_conditional:
            conditional_orders = self._conditional_candidate(
                codes_by_unit,
                lengths_by_unit,
                global_order,
                predecessors,
            )

            global_score = self._trie_score(
                self._serialize_global(unit_values, global_order, n_rows)
            )
            conditional_score = self._trie_score(
                self._serialize_orders(unit_values, conditional_orders)
            )

            if conditional_score > global_score:
                best_orders = conditional_orders
                best_is_global = False

        global_positions = [
            position
            for unit in global_order
            for position in units[unit]
        ]

        if best_is_global:
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
                column_orderings.append([columns[position] for position in positions])

        reordered = pd.DataFrame(
            output,
            index=df.index.copy(),
            columns=df.columns.copy(),
        )
        return reordered, column_orderings


# EVOLVE-BLOCK-END