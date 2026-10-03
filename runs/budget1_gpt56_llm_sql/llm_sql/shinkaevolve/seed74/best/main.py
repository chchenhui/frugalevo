# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict


class Evolved(Algorithm):
    """
    Prefix-cache-oriented dataframe reordering.

    The dataframe returned by this class always contains the original cells,
    merely permuted within rows.  ``column_orderings[i][j]`` identifies the
    source column for the value stored at output row i, output position j.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _score_strings(strings: List[str]) -> int:
        """Exact serial-Trie reuse for a fixed collection of strings."""
        if len(strings) < 2:
            return 0

        ordered = sorted(strings)
        score = 0
        previous = ordered[0]

        for current in ordered[1:]:
            if previous == current:
                score += len(current)
            else:
                limit = min(len(previous), len(current))
                low, high = 0, limit
                # Slice comparisons execute in C and avoid Python character loops.
                while low < high:
                    middle = (low + high + 1) // 2
                    if previous[:middle] == current[:middle]:
                        low = middle
                    else:
                        high = middle - 1
                score += low
            previous = current
        return score

    @staticmethod
    def _serial_values(df: pd.DataFrame):
        """
        Match evaluator serialization without mutating the dataframe that will
        be returned.  pandas fillna/astype is intentionally used only here.
        """
        if df.empty:
            return []
        return df.fillna("").astype(str).to_numpy(dtype=object).tolist()

    @staticmethod
    def _dependency_columns(columns, one_way_dep):
        """Resolve legacy substring dependency specifications deterministically."""
        position = {name: i for i, name in enumerate(columns)}
        dependencies = {}

        for pair in one_way_dep or []:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                continue
            left_hint, right_hint = pair
            left = [c for c in columns if c == left_hint]
            right = [c for c in columns if c == right_hint]

            if not left:
                left = [c for c in columns if str(left_hint) in str(c)]
            if not right:
                right = [c for c in columns if str(right_hint) in str(c)]

            if len(left) == 1 and len(right) == 1 and left[0] != right[0]:
                dependencies.setdefault(right[0], set()).add(left[0])

        return dependencies, position

    @staticmethod
    def _make_units(columns, col_merge):
        """
        A merged group is treated as an indivisible contiguous ordering unit.
        Values are not concatenated or changed; this only constrains ordering.
        """
        used = set()
        units = []

        for group in col_merge or []:
            members = [c for c in group if c in columns and c not in used]
            if members:
                units.append(members)
                used.update(members)

        for column in columns:
            if column not in used:
                units.append([column])

        return units

    @staticmethod
    def _topological_order(unit_count, rank, prerequisites):
        """
        Deterministic ranked topological ordering.  Invalid/cyclic dependency
        specifications cannot cause data loss: remaining units are appended by
        their heuristic rank.
        """
        remaining = set(range(unit_count))
        result = []

        while remaining:
            available = [
                unit for unit in remaining
                if prerequisites.get(unit, set()).isdisjoint(remaining)
            ]

            if not available:
                available = list(remaining)

            selected = min(available, key=lambda u: (-rank[u], u))
            result.append(selected)
            remaining.remove(selected)

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
        """
        Return an unchanged-shape dataframe with values only permuted within
        their original rows, plus the corresponding source-column permutation
        for every row.
        """
        del early_stop, distinct_value_threshold, parallel

        row_count, column_count = df.shape
        columns = list(df.columns)

        if row_count == 0 or column_count == 0:
            return df.copy(), [columns[:] for _ in range(row_count)]

        source_values = df.to_numpy(dtype=object, copy=True)
        text_values = self._serial_values(df)
        units = self._make_units(columns, col_merge)
        unit_count = len(units)

        column_index = {column: i for i, column in enumerate(columns)}
        unit_for_column = {}
        for unit_index, unit in enumerate(units):
            for column in unit:
                unit_for_column[column] = unit_index

        dependencies, _ = self._dependency_columns(columns, one_way_dep)
        prerequisites = {}
        for child, parents in dependencies.items():
            child_unit = unit_for_column.get(child)
            if child_unit is None:
                continue
            for parent in parents:
                parent_unit = unit_for_column.get(parent)
                if parent_unit is not None and parent_unit != child_unit:
                    prerequisites.setdefault(child_unit, set()).add(parent_unit)

        # Precompute each unit's actual serialized prefix contribution.
        unit_strings = []
        for unit in units:
            indexes = [column_index[c] for c in unit]
            unit_strings.append([
                "".join(text_values[row][idx] for idx in indexes)
                for row in range(row_count)
            ])

        # Required reliable global heuristic:
        # sum(len(v) * count(v) * (count(v)-1)).
        pair_rank = []
        alternate_rank = []
        for values in unit_strings:
            counts = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1

            pair_rank.append(sum(
                len(value) * count * (count - 1)
                for value, count in counts.items()
            ))
            alternate_rank.append(sum(
                len(value) * len(value) * count * (count - 1)
                for value, count in counts.items()
            ))

        global_units = self._topological_order(unit_count, pair_rank, prerequisites)
        alternate_units = self._topological_order(unit_count, alternate_rank, prerequisites)

        def expand(unit_order):
            return [column for unit in unit_order for column in units[unit]]

        global_columns = expand(global_units)
        alternate_columns = expand(alternate_units)
        global_orderings = [global_columns[:] for _ in range(row_count)]
        alternate_orderings = [alternate_columns[:] for _ in range(row_count)]

        # Conditional partition alternative.  Work is bounded on wide tables:
        # only the strongest leading units may be selected recursively.
        candidate_limit = min(unit_count, 24)
        selectable = set(sorted(
            range(unit_count),
            key=lambda u: (-pair_rank[u], u)
        )[:candidate_limit])

        max_depth = min(unit_count, 12)
        if col_stop is not None:
            max_depth = min(max_depth, max(0, int(col_stop)))
        if row_stop is not None and int(row_stop) <= 0:
            max_depth = 0

        conditional_units = [None] * row_count

        def tail_order(selected):
            remaining = set(range(unit_count)) - set(selected)
            result = list(selected)

            while remaining:
                available = [
                    unit for unit in remaining
                    if prerequisites.get(unit, set()).isdisjoint(remaining)
                ]
                if not available:
                    available = list(remaining)
                chosen = min(available, key=lambda u: (-pair_rank[u], u))
                result.append(chosen)
                remaining.remove(chosen)
            return result

        def assign_group(rows, prefix):
            order = tail_order(prefix)
            for row in rows:
                conditional_units[row] = order

        def split_group(rows, prefix, depth):
            if len(rows) < 2 or depth >= max_depth:
                assign_group(rows, prefix)
                return

            selected = set(prefix)
            available = [
                unit for unit in selectable
                if unit not in selected
                and prerequisites.get(unit, set()).issubset(selected)
            ]
            if not available:
                assign_group(rows, prefix)
                return

            best_unit = None
            best_score = 0
            for unit in available:
                counts = {}
                values = unit_strings[unit]
                for row in rows:
                    value = values[row]
                    counts[value] = counts.get(value, 0) + 1
                score = sum(
                    len(value) * count * (count - 1)
                    for value, count in counts.items()
                )
                if score > best_score or (
                    score == best_score and best_unit is not None and unit < best_unit
                ):
                    best_score = score
                    best_unit = unit

            if best_unit is None or best_score <= 0:
                assign_group(rows, prefix)
                return

            partitions = {}
            values = unit_strings[best_unit]
            for row in rows:
                partitions.setdefault(values[row], []).append(row)

            new_prefix = prefix + [best_unit]
            for group_rows in partitions.values():
                split_group(group_rows, new_prefix, depth + 1)

        split_group(list(range(row_count)), [], 0)

        conditional_orderings = []
        for unit_order in conditional_units:
            if unit_order is None:
                unit_order = global_units
            conditional_orderings.append(expand(unit_order))

        def serialized_rows(orderings):
            result = []
            for row, ordering in enumerate(orderings):
                result.append("".join(
                    text_values[row][column_index[column]]
                    for column in ordering
                ))
            return result

        # Select among exactly three bounded complete constructions using the
        # actual character-Trie objective, including partial-field prefixes.
        candidates = [
            (global_orderings, self._score_strings(serialized_rows(global_orderings))),
            (alternate_orderings, self._score_strings(serialized_rows(alternate_orderings))),
            (conditional_orderings, self._score_strings(serialized_rows(conditional_orderings))),
        ]
        best_orderings, _ = max(
            enumerate(candidates),
            key=lambda item: (item[1][1], -item[0])
        )[1]

        # Keep output labels and index stable.  The accompanying ordering maps
        # every output position back to its source column for that row.
        output = [[None] * column_count for _ in range(row_count)]
        for row, ordering in enumerate(best_orderings):
            for output_position, source_column in enumerate(ordering):
                output[row][output_position] = source_values[row][column_index[source_column]]

        reordered = pd.DataFrame(output, index=df.index.copy(), columns=df.columns, dtype=object)
        return reordered, best_orderings

# EVOLVE-BLOCK-END