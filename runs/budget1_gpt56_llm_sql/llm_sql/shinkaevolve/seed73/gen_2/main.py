# EVOLVE-BLOCK-START
import heapq
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from solver import Algorithm


class Evolved(Algorithm):
    """
    Prefix-cache-aware row/column reordering.

    Values are never modified.  A returned row may contain its original cells in
    a row-specific order; column_orderings describes the source column for every
    returned cell position.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _serialized_value(value) -> str:
        """Match the evaluator's fillna('').astype(str) representation safely."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and bool(missing):
                return ""
        except Exception:
            pass
        try:
            return str(value)
        except Exception:
            return repr(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        if left == right:
            return len(left)
        limit = min(len(left), len(right))
        lo, hi = 0, limit
        # Slice comparisons are performed in C and avoid Python character loops.
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
    def _topological_units(
        units: List[int],
        predecessors: Dict[int, set],
        priority: Dict[int, Tuple],
    ) -> List[int]:
        """Deterministic topological ordering, retaining all units on cycles."""
        remaining = set(units)
        result = []
        while remaining:
            ready = [
                unit for unit in remaining
                if not (predecessors.get(unit, set()) & remaining)
            ]
            if not ready:
                # Invalid/cyclic dependency input must not lose any source field.
                ready = list(remaining)
            ready.sort(key=lambda unit: priority.get(unit, (unit,)))
            chosen = ready[0]
            result.append(chosen)
            remaining.remove(chosen)
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
        # Work with positional columns so duplicate or non-identifier labels work.
        nrows, ncols = df.shape
        labels = list(df.columns)

        if nrows == 0 or ncols == 0:
            return df.copy(), [[] for _ in range(nrows)]

        source_values = df.to_numpy(dtype=object, copy=True)
        text = [
            [self._serialized_value(source_values[r, c]) for c in range(ncols)]
            for r in range(nrows)
        ]

        # col_merge is treated as an adjacency constraint: members of a supplied
        # group form one output unit, without combining or changing their cells.
        used = set()
        units = []
        for group in col_merge or []:
            positions = []
            for requested_name in group:
                for pos, label in enumerate(labels):
                    if pos not in used and label == requested_name:
                        positions.append(pos)
                        used.add(pos)
            if positions:
                units.append(positions)
        for pos in range(ncols):
            if pos not in used:
                units.append([pos])

        unit_of_column = {}
        for unit_id, members in enumerate(units):
            for member in members:
                unit_of_column[member] = unit_id

        # Preserve the historical API's substring matching behavior for
        # dependency names, but do not fabricate or remove any columns.
        predecessors = {unit_id: set() for unit_id in range(len(units))}
        for before_name, after_name in one_way_dep or []:
            before_positions = [
                i for i, label in enumerate(labels)
                if str(before_name) in str(label)
            ]
            after_positions = [
                i for i, label in enumerate(labels)
                if str(after_name) in str(label)
            ]
            for before in before_positions:
                for after in after_positions:
                    left, right = unit_of_column[before], unit_of_column[after]
                    if left != right:
                        predecessors[right].add(left)

        unit_values = []
        unit_scores = []
        for members in units:
            values = ["".join(text[row][col] for col in members) for row in range(nrows)]
            counts = Counter(values)
            score = sum(len(value) * count * (count - 1) for value, count in counts.items())
            unit_values.append(values)
            unit_scores.append(score)

        unit_ids = list(range(len(units)))
        original_priority = {unit: (unit,) for unit in unit_ids}
        frequency_priority = {
            unit: (-unit_scores[unit], unit)
            for unit in unit_ids
        }

        # Candidate 1: stable, dependency-safe original ordering.
        original_units = self._topological_units(unit_ids, predecessors, original_priority)

        # Candidate 2: global length-weighted repeated-value ranking.
        frequency_units = self._topological_units(unit_ids, predecessors, frequency_priority)

        def expand(unit_order: List[int]) -> List[int]:
            return [column for unit in unit_order for column in units[unit]]

        candidates = [
            [expand(original_units) for _ in range(nrows)],
            [expand(frequency_units) for _ in range(nrows)],
        ]

        # Candidate 3: bounded conditional prefix partition tree.  The leading
        # unit is chosen independently within each partition using actual
        # length-weighted pair repetition.  Remaining columns use a cheap,
        # deterministic dependency-safe tail.
        conditional_orders = [None] * nrows
        max_depth = min(len(units), 12)
        max_candidates = min(len(units), 48)

        ranked_units = sorted(
            unit_ids,
            key=lambda unit: (-unit_scores[unit], unit),
        )
        allowed_units = set(ranked_units[:max_candidates])

        def fill_group(rows: List[int], remaining: List[int], prefix: List[int], depth: int):
            if not rows:
                return
            if len(rows) == 1 or not remaining or depth >= max_depth:
                tail_priority = {
                    unit: (-unit_scores[unit], unit)
                    for unit in remaining
                }
                tail = self._topological_units(remaining, predecessors, tail_priority)
                order = expand(prefix + tail)
                for row in rows:
                    conditional_orders[row] = order
                return

            remaining_set = set(remaining)
            eligible = [
                unit for unit in remaining
                if not (predecessors.get(unit, set()) & remaining_set)
                and unit in allowed_units
            ]
            best_unit = None
            best_score = 0
            for unit in eligible:
                counts = Counter(unit_values[unit][row] for row in rows)
                score = sum(
                    len(value) * count * (count - 1)
                    for value, count in counts.items()
                )
                if score > best_score or (score == best_score and best_unit is not None and unit < best_unit):
                    best_score = score
                    best_unit = unit

            if best_unit is None or best_score <= 0:
                tail_priority = {
                    unit: (-unit_scores[unit], unit)
                    for unit in remaining
                }
                tail = self._topological_units(remaining, predecessors, tail_priority)
                order = expand(prefix + tail)
                for row in rows:
                    conditional_orders[row] = order
                return

            partitions = {}
            for row in rows:
                partitions.setdefault(unit_values[best_unit][row], []).append(row)
            next_remaining = [unit for unit in remaining if unit != best_unit]
            for _, child_rows in sorted(partitions.items(), key=lambda item: item[0]):
                fill_group(child_rows, next_remaining, prefix + [best_unit], depth + 1)

        fill_group(list(range(nrows)), unit_ids, [], 0)
        candidates.append(conditional_orders)

        best_orders = None
        best_score = -1
        best_strings = None
        for orders in candidates:
            strings = [
                "".join(text[row][column] for column in orders[row])
                for row in range(nrows)
            ]
            score = self._trie_score(strings)
            if score > best_score:
                best_score = score
                best_orders = orders
                best_strings = strings

        # Lexical row order is deterministic and is compatible with a serial
        # Trie, while preserving source row identity through the original index.
        row_order = sorted(range(nrows), key=lambda row: (best_strings[row], row))
        output = np.empty((nrows, ncols), dtype=object)
        output_orders = []
        output_index = []
        for destination, source_row in enumerate(row_order):
            order = best_orders[source_row]
            output[destination, :] = [source_values[source_row, column] for column in order]
            output_orders.append([labels[column] for column in order])
            output_index.append(df.index[source_row])

        result = pd.DataFrame(output, columns=df.columns, index=output_index)
        result.index.name = df.index.name
        return result, output_orders


# EVOLVE-BLOCK-END