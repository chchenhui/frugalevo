# EVOLVE-BLOCK-START
import math
from collections import defaultdict
from typing import Tuple, List

import pandas as pd
from solver import Algorithm


class Evolved(Algorithm):
    """
    Prefix-cache-oriented row and column reorderer.

    The dataframe values are never changed.  Only their row order and their
    position within each row may change, with ``column_orderings`` describing
    the source-column order used for every returned row.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _text(value) -> str:
        """Match evaluator normalization without modifying the stored value."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, bool) and missing:
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
        # Prefix slicing performs the character comparison in C and avoids a
        # Python loop for long text fields.
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
    def _constraint_units(columns, col_merge, one_way_dep):
        """
        Build indivisible column blocks.  A merge request is interpreted as a
        contiguous block constraint; it never combines or drops source cells.
        """
        names = list(columns)
        position = {name: i for i, name in enumerate(names)}
        used = set()
        units = []

        for group in col_merge or []:
            members = [name for name in group if name in position and name not in used]
            if members:
                members.sort(key=position.__getitem__)
                units.append(members)
                used.update(members)

        for name in names:
            if name not in used:
                units.append([name])

        unit_of = {}
        for unit_index, unit in enumerate(units):
            for name in unit:
                unit_of[name] = unit_index

        predecessors = defaultdict(set)
        for dep in one_way_dep or []:
            if len(dep) != 2:
                continue
            before_matches = [name for name in names if dep[0] == name or dep[0] in str(name)]
            after_matches = [name for name in names if dep[1] == name or dep[1] in str(name)]
            if len(before_matches) == 1 and len(after_matches) == 1:
                before = unit_of[before_matches[0]]
                after = unit_of[after_matches[0]]
                if before != after:
                    predecessors[after].add(before)

        return units, predecessors

    @staticmethod
    def _topological_rank(unit_scores, predecessors):
        """Highest score first, subject to supplied dependency precedence."""
        remaining = set(range(len(unit_scores)))
        result = []
        while remaining:
            available = [
                unit for unit in remaining
                if not (predecessors.get(unit, set()) & remaining)
            ]
            # Cyclic caller constraints cannot be fully satisfied.  Break a
            # cycle deterministically while preserving all columns.
            if not available:
                available = list(remaining)
            best = min(available, key=lambda unit: (-unit_scores[unit], unit))
            result.append(best)
            remaining.remove(best)
        return result

    @staticmethod
    def _serialize_row(text_values, unit_order, units):
        return "".join(
            text_values[col]
            for unit in unit_order
            for col in units[unit]
        )

    def _conditional_orders(self, text_rows, units, global_order, predecessors):
        """
        Bounded conditional prefix partitioning.  Each branch chooses a
        repeated leading unit, then lets descendants choose their own suffix
        order.  The work is capped for wide frames and deep trees.
        """
        row_count = len(text_rows)
        unit_count = len(units)
        orders = [None] * row_count
        candidate_units = sorted(
            range(unit_count),
            key=lambda u: (-sum(len(text_rows[r][c]) for r in range(row_count)
                                 for c in units[u]), u)
        )[: min(unit_count, 32)]

        def visit(rows, remaining, prefix, depth):
            if not rows:
                return
            if len(rows) == 1 or not remaining or depth >= min(8, unit_count):
                tail = [u for u in global_order if u in remaining]
                for row in rows:
                    orders[row] = prefix + tail
                return

            allowed = [
                unit for unit in remaining
                if unit in candidate_units
                and not (predecessors.get(unit, set()) & remaining)
            ]
            if not allowed:
                allowed = [u for u in remaining if not (predecessors.get(u, set()) & remaining)]

            best_unit = None
            best_score = 0
            best_groups = None

            for unit in allowed:
                groups = defaultdict(list)
                for row in rows:
                    key = tuple(text_rows[row][col] for col in units[unit])
                    groups[key].append(row)

                score = 0
                for key, members in groups.items():
                    if len(members) > 1:
                        score += sum(len(value) for value in key) * len(members) * (len(members) - 1)

                if score > best_score or (score == best_score and best_unit is not None and unit < best_unit):
                    best_unit = unit
                    best_score = score
                    best_groups = groups

            if best_unit is None or best_score <= 0:
                tail = [u for u in global_order if u in remaining]
                for row in rows:
                    orders[row] = prefix + tail
                return

            next_remaining = set(remaining)
            next_remaining.remove(best_unit)
            for members in best_groups.values():
                visit(members, next_remaining, prefix + [best_unit], depth + 1)

        visit(list(range(row_count)), set(range(unit_count)), [], 0)
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
        # Empty frames still need a valid ordering result.
        if df.shape[1] == 0:
            return df.copy(), [[] for _ in range(len(df))]

        columns = list(df.columns)
        source_values = df.to_numpy(dtype=object, copy=True)
        text_rows = [
            [self._text(value) for value in source_values[row]]
            for row in range(len(df))
        ]

        units, predecessors = self._constraint_units(columns, col_merge, one_way_dep)

        # Length-weighted repetition score requested by the objective guidance.
        unit_scores = []
        for unit in units:
            score = 0
            for col in unit:
                counts = defaultdict(int)
                lengths = {}
                for row in range(len(df)):
                    value = text_rows[row][columns.index(col)]
                    counts[value] += 1
                    lengths[value] = len(value)
                score += sum(lengths[value] * count * (count - 1)
                             for value, count in counts.items())
            unit_scores.append(score)

        global_units = self._topological_rank(unit_scores, predecessors)
        original_units = self._topological_rank(
            [-index for index in range(len(units))], predecessors
        )

        candidates = [original_units, global_units]
        if len(df) > 1 and len(units) > 1:
            candidates.append(
                self._conditional_orders(text_rows, units, global_units, predecessors)
            )

        best_orders = None
        best_score = -1
        best_strings = None

        for candidate in candidates:
            if candidate and isinstance(candidate[0], int):
                orders = [candidate] * len(df)
            else:
                orders = candidate

            strings = [
                self._serialize_row(text_rows[row], orders[row], units)
                for row in range(len(df))
            ]
            score = self._trie_score(strings)
            if score > best_score:
                best_score = score
                best_orders = orders
                best_strings = strings

        # Lexicographic output order is deterministic and gives the evaluator's
        # canonical adjacent-LCP ordering without changing row identity.
        row_order = sorted(range(len(df)), key=lambda row: (best_strings[row], row))

        output_rows = []
        column_orderings = []
        for row in row_order:
            unit_order = best_orders[row]
            source_positions = [
                columns.index(column)
                for unit in unit_order
                for column in units[unit]
            ]
            output_rows.append([source_values[row][pos] for pos in source_positions])
            column_orderings.append([
                column for unit in unit_order for column in units[unit]
            ])

        # For a global order, labels accurately describe the output.  For a
        # row-specific order there is no single truthful label permutation, so
        # retain the original positional labels and provide the exact per-row
        # order through column_orderings.
        if column_orderings and all(order == column_orderings[0] for order in column_orderings):
            output_columns = column_orderings[0]
        else:
            output_columns = columns

        result = pd.DataFrame(
            output_rows,
            columns=output_columns,
            index=[df.index[row] for row in row_order],
            dtype=object,
        )
        return result, column_orderings


# EVOLVE-BLOCK-END