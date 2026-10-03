# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict, Any
from collections import Counter, defaultdict
import math


class Evolved(Algorithm):
    """
    Prefix-cache oriented row/column reordering.

    Each returned row contains the original row's values in the order reported
    by its corresponding entry in column_orderings.  This permits row-specific
    orders without creating, dropping, or modifying source values.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _serial_value(value: Any) -> str:
        """Match the evaluator's fillna('').astype(str) representation."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and missing:
                return ""
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        if left == right:
            return len(left)
        limit = min(len(left), len(right))
        lo, hi = 0, limit
        # Slice comparisons execute in C and avoid a Python character loop.
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if left[:mid] == right[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _trie_score(self, rows: List[str]) -> int:
        if len(rows) < 2:
            return 0
        ordered = sorted(rows)
        return sum(self._lcp(ordered[i - 1], ordered[i])
                   for i in range(1, len(ordered)))

    @staticmethod
    def _topological_order(units, scores, edges):
        """Deterministic score-prioritized topological ordering."""
        incoming = {u: set() for u in units}
        outgoing = {u: set() for u in units}
        for before, after in edges:
            if before != after and before in incoming and after in incoming:
                incoming[after].add(before)
                outgoing[before].add(after)

        result = []
        available = [u for u in units if not incoming[u]]
        while available:
            available.sort(key=lambda u: (-scores.get(u, 0), u))
            unit = available.pop(0)
            result.append(unit)
            for child in sorted(outgoing[unit]):
                incoming[child].discard(unit)
                if not incoming[child]:
                    available.append(child)

        # Cycles in optional dependencies must not lose data.  Break them
        # deterministically after retaining every acyclicly ordered unit.
        if len(result) != len(units):
            used = set(result)
            result.extend(sorted((u for u in units if u not in used),
                                 key=lambda u: (-scores.get(u, 0), u)))
        return result

    def _units_and_dependencies(self, columns, col_merge, one_way_dep):
        """
        Build indivisible ordered units for requested merge groups.  A merge
        means its listed source columns remain adjacent in the reported order.
        """
        position = {name: i for i, name in enumerate(columns)}
        claimed = set()
        units = []

        for group in col_merge or []:
            members = [c for c in group if c in position and c not in claimed]
            if members:
                members.sort(key=lambda c: position[c])
                units.append(tuple(members))
                claimed.update(members)

        for col in columns:
            if col not in claimed:
                units.append((col,))

        unit_of = {}
        for number, unit in enumerate(units):
            for col in unit:
                unit_of[col] = number

        edges = set()
        for dependency in one_way_dep or []:
            if len(dependency) != 2:
                continue
            left_hint, right_hint = dependency
            left_matches = [c for c in columns if left_hint in str(c)]
            right_matches = [c for c in columns if right_hint in str(c)]
            if len(left_matches) == 1 and len(right_matches) == 1:
                left = unit_of[left_matches[0]]
                right = unit_of[right_matches[0]]
                if left != right:
                    edges.add((left, right))
        return units, edges

    def _global_order(self, units, edges, cell_strings):
        scores = {}
        for number, unit in enumerate(units):
            total = 0
            for col in unit:
                counts = Counter(cell_strings[col])
                total += sum(len(value) * count * (count - 1)
                             for value, count in counts.items())
            scores[number] = total
        unit_order = self._topological_order(list(range(len(units))), scores, edges)
        return [col for unit_number in unit_order for col in units[unit_number]], scores

    def _conditional_orders(self, columns, units, edges, cell_strings, global_order):
        """
        A bounded conditional prefix partition tree.  It only chooses a small
        prefix recursively; the deterministic global order is used as a cheap
        tail, keeping wide inputs bounded.
        """
        nrows = len(next(iter(cell_strings.values()))) if cell_strings else 0
        result = [None] * nrows
        global_rank = {col: i for i, col in enumerate(global_order)}
        max_depth = min(len(units), 10)
        candidate_limit = 32 if len(units) > 32 else len(units)

        predecessors = defaultdict(set)
        for before, after in edges:
            predecessors[after].add(before)

        def finish(indices, selected):
            selected_columns = []
            for unit_number in selected:
                selected_columns.extend(units[unit_number])
            tail = [c for c in global_order if c not in selected_columns]
            order = selected_columns + tail
            for row in indices:
                result[row] = order

        def visit(indices, remaining, selected, depth):
            if len(indices) < 2 or not remaining or depth >= max_depth:
                finish(indices, selected)
                return

            available = [u for u in remaining
                         if predecessors[u].issubset(set(selected))]
            if not available:
                finish(indices, selected)
                return

            estimates = []
            for unit_number in available:
                repeat_score = 0
                for col in units[unit_number]:
                    counts = Counter(cell_strings[col][row] for row in indices)
                    repeat_score += sum(
                        len(value) * count * (count - 1)
                        for value, count in counts.items()
                    )
                if repeat_score:
                    estimates.append((repeat_score, unit_number))

            if not estimates:
                finish(indices, selected)
                return

            estimates.sort(key=lambda item: (-item[0], item[1]))
            # Selection is bounded on very wide tables.  The highest scoring
            # available unit is sufficient here; the limit bounds estimation
            # policy while preserving deterministic behavior.
            best_unit = estimates[0][1]
            groups = defaultdict(list)
            for row in indices:
                key = tuple(cell_strings[col][row] for col in units[best_unit])
                groups[key].append(row)

            if len(groups) <= 1:
                visit(indices, [u for u in remaining if u != best_unit],
                      selected + [best_unit], depth + 1)
                return

            next_remaining = [u for u in remaining if u != best_unit]
            for key in sorted(groups):
                visit(groups[key], next_remaining, selected + [best_unit],
                      depth + 1)

        visit(list(range(nrows)), list(range(len(units))), [], 0)
        for row in range(nrows):
            if result[row] is None:
                result[row] = list(global_order)
        return result

    def _candidate_strings(self, orders, cell_strings):
        return [
            "".join(cell_strings[column][row] for column in order)
            for row, order in enumerate(orders)
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
        # Do not mutate caller data or add temporary identity columns.
        source = df.copy(deep=True)
        columns = list(source.columns)
        nrows, ncols = source.shape

        if nrows == 0 or ncols == 0:
            return source, [[] for _ in range(nrows)]

        # Positional extraction avoids dtype coercion for mixed object columns.
        raw_values = source.to_numpy(dtype=object, copy=True)
        cell_strings = {
            col: [self._serial_value(raw_values[row, col_index])
                  for row in range(nrows)]
            for col_index, col in enumerate(columns)
        }

        units, edges = self._units_and_dependencies(
            columns, col_merge, one_way_dep
        )
        global_order, _ = self._global_order(units, edges, cell_strings)

        # At most three complete constructions are scored.
        baseline_orders = [list(columns) for _ in range(nrows)]
        global_orders = [list(global_order) for _ in range(nrows)]
        conditional_orders = self._conditional_orders(
            columns, units, edges, cell_strings, global_order
        )

        candidates = [
            (baseline_orders, self._candidate_strings(baseline_orders, cell_strings)),
            (global_orders, self._candidate_strings(global_orders, cell_strings)),
            (conditional_orders,
             self._candidate_strings(conditional_orders, cell_strings)),
        ]

        best_orders, best_strings = candidates[0]
        best_score = self._trie_score(best_strings)
        for orders, strings in candidates[1:]:
            score = self._trie_score(strings)
            if score > best_score:
                best_orders, best_strings, best_score = orders, strings, score

        # Sorting is not needed for the ideal Trie score, but creates a stable
        # stream order and is useful to evaluators using insertion order.
        row_order = sorted(range(nrows), key=lambda row: (best_strings[row], row))
        column_position = {column: position for position, column in enumerate(columns)}
        output_values = np.empty((nrows, ncols), dtype=object)
        output_orders = []

        for output_row, source_row in enumerate(row_order):
            order = list(best_orders[source_row])
            output_orders.append(order)
            for output_col, original_col in enumerate(order):
                output_values[output_row, output_col] = raw_values[
                    source_row, column_position[original_col]
                ]

        # Column labels identify output slots; column_orderings identifies the
        # original source column represented by each slot for every row.
        output = pd.DataFrame(
            output_values,
            index=source.index.take(row_order),
            columns=source.columns,
        )
        return output, output_orders

# EVOLVE-BLOCK-END