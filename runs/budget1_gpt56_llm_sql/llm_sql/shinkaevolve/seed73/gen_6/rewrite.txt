# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict
from collections import Counter


class Evolved(Algorithm):
    """
    Prefix-cache-oriented row serializer.

    The returned frame contains the source values from each row in the order
    described by the corresponding entry in column_orderings.  Its displayed
    column labels remain the input labels; column_orderings is the authoritative
    mapping for row-specific layouts.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _string_value(value) -> str:
        """Match fillna('').astype(str) without changing stored source values."""
        if value is None or value is pd.NA:
            return ""
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and bool(missing):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Bounded Python-level LCP using C-level slice comparisons."""
        if left == right:
            return len(left)
        high = min(len(left), len(right))
        low = 0
        while low < high:
            middle = (low + high + 1) // 2
            if left[:middle] == right[:middle]:
                low = middle
            else:
                high = middle - 1
        return low

    def _score_orders(self, values, orders):
        """Exact ideal serial-Trie reuse and lexicographic row order."""
        serial = [
            "".join(values[row][column] for column in orders[row])
            for row in range(len(values))
        ]
        ranked = sorted(range(len(serial)), key=lambda row: (serial[row], row))
        score = 0
        for previous, current in zip(ranked, ranked[1:]):
            score += self._lcp(serial[previous], serial[current])
        return score, ranked

    @staticmethod
    def _resolve_column(name, columns):
        if name in columns:
            return name
        matches = [column for column in columns if str(name) in str(column)]
        return matches[0] if len(matches) == 1 else None

    def _apply_constraints(self, order, columns, col_merge, one_way_dep):
        """Apply API ordering constraints without changing any values."""
        result = list(order)
        position = {column: i for i, column in enumerate(columns)}

        # A one-way dependency means the source must be emitted before target.
        # Repeat a bounded number of times because several dependencies can chain.
        dependencies = []
        for source, target in one_way_dep or []:
            source = self._resolve_column(source, columns)
            target = self._resolve_column(target, columns)
            if source is not None and target is not None and source != target:
                dependencies.append((source, target))

        for _ in range(max(1, len(dependencies))):
            changed = False
            locations = {column: i for i, column in enumerate(result)}
            for source, target in dependencies:
                if locations[source] > locations[target]:
                    result.remove(target)
                    source_at = result.index(source)
                    result.insert(source_at + 1, target)
                    changed = True
            if not changed:
                break

        # Preserve each requested merge group as a contiguous stable block.
        for group in col_merge or []:
            members = [column for column in result if column in set(group)]
            if len(members) < 2:
                continue
            first = min(result.index(column) for column in members)
            result = [column for column in result if column not in members]
            result[first:first] = members

        # Defensive completion for malformed constraints.
        seen = set()
        result = [column for column in result
                  if column in position and not (column in seen or seen.add(column))]
        result.extend(column for column in columns if column not in seen)
        return result

    def _global_order(self, strings, columns, frequency_power=2):
        """Global frequency/length proposal requested by the experiment."""
        ranking = []
        row_count = len(strings)
        for column in range(len(columns)):
            counts = Counter(strings[row][column] for row in range(row_count))
            if frequency_power == 2:
                score = sum(
                    len(value) * count * (count - 1)
                    for value, count in counts.items()
                )
            else:
                score = sum(
                    len(value) * max(0, count - 1)
                    for value, count in counts.items()
                )
            ranking.append((-score, column))
        ranking.sort()
        return [column for _, column in ranking]

    def _conditional_orders(self, strings, global_order, max_depth=12,
                            max_candidates=32, max_nodes=128):
        """
        Cheap conditional prefix partition tree.  Factorization is performed
        once, then each bounded node only counts integer codes for candidate
        fields.
        """
        rows = len(strings)
        columns = len(global_order)
        if rows < 2 or columns < 2:
            return [list(global_order) for _ in range(rows)]

        codes = []
        code_lengths = []
        for column in range(columns):
            mapping = {}
            lengths = []
            encoded = np.empty(rows, dtype=np.int64)
            for row in range(rows):
                value = strings[row][column]
                code = mapping.get(value)
                if code is None:
                    code = len(mapping)
                    mapping[value] = code
                    lengths.append(len(value))
                encoded[row] = code
            codes.append(encoded)
            code_lengths.append(lengths)

        orders = [list(global_order) for _ in range(rows)]
        candidate_columns = global_order[:min(columns, max_candidates)]
        node_count = [0]

        def build(group, remaining, prefix, depth):
            if (len(group) < 2 or not remaining or depth >= max_depth or
                    node_count[0] >= max_nodes):
                tail = [column for column in global_order if column in remaining]
                final = prefix + tail
                for row in group:
                    orders[row] = final
                return

            node_count[0] += 1
            best_column = None
            best_score = 0
            for column in remaining:
                if column not in candidate_columns:
                    continue
                counts = Counter(int(codes[column][row]) for row in group)
                score = sum(
                    code_lengths[column][code] * count * (count - 1)
                    for code, count in counts.items()
                    if count > 1
                )
                if score > best_score or (
                    score == best_score and best_column is not None and
                    global_order.index(column) < global_order.index(best_column)
                ):
                    best_score = score
                    best_column = column

            if best_column is None or best_score <= 0:
                tail = [column for column in global_order if column in remaining]
                final = prefix + tail
                for row in group:
                    orders[row] = final
                return

            buckets = {}
            for row in group:
                buckets.setdefault(int(codes[best_column][row]), []).append(row)
            next_remaining = [column for column in remaining if column != best_column]
            for bucket in buckets.values():
                build(bucket, next_remaining, prefix + [best_column], depth + 1)

        build(list(range(rows)), list(global_order), [], 0)
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
        # Never add helper columns or coerce the stored source cells.
        source = df.copy(deep=True)
        columns = list(source.columns)
        row_count, column_count = source.shape

        if row_count == 0 or column_count == 0:
            return source, [[] for _ in range(row_count)]

        values = [
            [self._string_value(source.iat[row, column])
             for column in range(column_count)]
            for row in range(row_count)
        ]

        global_order = self._global_order(values, columns, frequency_power=2)
        alternate_order = self._global_order(values, columns, frequency_power=1)

        candidates = []
        for proposal in (global_order, alternate_order):
            constrained = self._apply_constraints(
                [columns[index] for index in proposal],
                columns,
                col_merge,
                one_way_dep,
            )
            index_order = [columns.index(column) for column in constrained]
            candidates.append([list(index_order) for _ in range(row_count)])

        conditional = self._conditional_orders(values, global_order)
        conditional = [
            [columns.index(column) for column in self._apply_constraints(
                [columns[index] for index in order],
                columns,
                col_merge,
                one_way_dep,
            )]
            for order in conditional
        ]
        candidates.append(conditional)

        best_orders = candidates[0]
        best_score, best_rows = self._score_orders(values, best_orders)
        for candidate in candidates[1:]:
            score, ranked_rows = self._score_orders(values, candidate)
            if score > best_score:
                best_score, best_orders, best_rows = score, candidate, ranked_rows

        # Sorting is stable and preserves each source index as its row identity.
        output_values = [
            [source.iat[row, column] for column in best_orders[row]]
            for row in best_rows
        ]
        output = pd.DataFrame(
            output_values,
            index=source.index.take(best_rows),
            columns=columns,
            dtype=object,
        )
        column_orderings = [
            [columns[column] for column in best_orders[row]]
            for row in best_rows
        ]

        if output.shape != source.shape:
            raise RuntimeError("reordering changed dataframe shape")
        return output, column_orderings


# EVOLVE-BLOCK-END