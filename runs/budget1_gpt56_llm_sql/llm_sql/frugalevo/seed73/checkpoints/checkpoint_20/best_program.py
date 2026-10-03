import pandas as pd
from collections import Counter
from typing import List, Tuple

from solver import Algorithm


class Evolved(Algorithm):
    """Bounded character-prefix optimizer with partial-prefix global layout."""

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.num_rows = 0
        self.num_cols = 0

    @staticmethod
    def _text(value) -> str:
        try:
            missing = pd.isna(value)
            if not hasattr(missing, "__len__") and bool(missing):
                return ""
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        limit = min(len(left), len(right))
        if limit == 0 or left[0] != right[0]:
            return 0
        low, high = 1, limit
        while low < high:
            middle = (low + high + 1) // 2
            if left[:middle] == right[:middle]:
                low = middle
            else:
                high = middle - 1
        return low

    @classmethod
    def _reuse(cls, strings: List[str]) -> int:
        if len(strings) < 2:
            return 0
        ordered = sorted(strings)
        total = 0
        previous = ordered[0]
        for current in ordered[1:]:
            if current == previous:
                total += len(current)
            else:
                total += cls._lcp(previous, current)
            previous = current
        return total

    @staticmethod
    def _merge(columns, order, col_merge):
        if not col_merge:
            return list(order)

        positions = {column: index for index, column in enumerate(order)}
        owners = {}
        claimed = set()

        for requested in col_merge:
            group = [
                column for column in order
                if column in requested and column not in claimed
            ]
            if not group:
                continue
            group.sort(key=positions.__getitem__)
            for column in group:
                owners[column] = group
                claimed.add(column)

        result = []
        emitted = set()
        for column in order:
            if column in emitted:
                continue
            group = owners.get(column)
            if group is None:
                result.append(column)
                emitted.add(column)
            else:
                result.extend(group)
                emitted.update(group)
        return result

    @classmethod
    def _normalize(cls, columns, integer_order, col_merge):
        proposed = [columns[index] for index in integer_order]
        merged = cls._merge(columns, proposed, col_merge)
        lookup = {column: index for index, column in enumerate(columns)}
        return [lookup[column] for column in merged]

    @staticmethod
    def _serialize(text, orders):
        return [
            "".join(text[row][column] for column in orders[row])
            for row in range(len(orders))
        ]

    def _global_candidates(self, text, columns, col_merge):
        rows = len(text)
        width = len(columns)
        whole_mass = []
        prefix_mass = []

        for column in range(width):
            values = Counter(text[row][column] for row in range(rows))
            mass = sum(
                len(value) * count * (count - 1)
                for value, count in values.items()
            )
            whole_mass.append((mass, len(values)))

            prefixes = Counter()
            for value, count in values.items():
                upto = min(8, len(value))
                for length in range(1, upto + 1):
                    prefixes[value[:length]] += count
            partial = sum(
                len(prefix) * count * (count - 1)
                for prefix, count in prefixes.items()
            )
            prefix_mass.append((partial, len(prefixes)))

        frequency_ranked = sorted(
            range(width),
            key=lambda column: (
                -whole_mass[column][0],
                whole_mass[column][1],
                column,
            ),
        )
        prefix_ranked = sorted(
            range(width),
            key=lambda column: (
                -prefix_mass[column][0],
                prefix_mass[column][1],
                -whole_mass[column][0],
                column,
            ),
        )

        return [
            self._normalize(columns, frequency_ranked, col_merge),
            self._normalize(columns, prefix_ranked, col_merge),
        ], frequency_ranked

    def _prototype_candidate(self, text, columns, col_merge, base_order):
        rows = len(text)
        width = len(columns)
        fallback = self._normalize(columns, base_order, col_merge)
        result = [fallback[:] for _ in range(rows)]

        if rows < 2 or width < 2:
            return result

        sample_rows = list(range(min(rows, 192)))
        active = base_order[:min(width, 28)]

        def make_string(row, order):
            return "".join(text[row][column] for column in order)

        sample_strings = [make_string(row, active) for row in sample_rows]
        anchors = [0]
        anchor_strings = [sample_strings[0]]

        while len(anchors) < min(4, len(sample_rows)):
            best_position = -1
            best_distance = -1
            for position, value in enumerate(sample_strings):
                if position in anchors:
                    continue
                nearest = max(self._lcp(value, anchor) for anchor in anchor_strings)
                distance = len(value) - nearest
                if distance > best_distance:
                    best_distance = distance
                    best_position = position
            if best_position < 0:
                break
            anchors.append(best_position)
            anchor_strings.append(sample_strings[best_position])

        def construct(prototypes):
            orders = [fallback[:] for _ in range(rows)]
            for position, row in enumerate(sample_rows):
                current = sample_strings[position]
                prototype = max(
                    prototypes,
                    key=lambda candidate: self._lcp(current, candidate),
                )
                remaining = list(active)
                chosen = []
                cursor = 0

                while remaining:
                    best_column = remaining[0]
                    best_score = -1
                    suffix = prototype[cursor:]
                    for column in remaining:
                        score = self._lcp(text[row][column], suffix)
                        if score > best_score:
                            best_score = score
                            best_column = column
                    chosen.append(best_column)
                    cursor += len(text[row][best_column])
                    remaining.remove(best_column)

                tail = [column for column in base_order if column not in chosen]
                orders[row] = self._normalize(columns, chosen + tail, col_merge)
            return orders

        first = construct(anchor_strings)
        rebuilt = [
            make_string(sample_rows[position], first[sample_rows[position]])
            for position in anchors
        ]
        return construct(rebuilt)

    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: int = None,
        col_stop: int = None,
        col_merge=[],
        one_way_dep=[],
        distinct_value_threshold=0.8,
        parallel=True,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        self.df = df
        self.num_rows, self.num_cols = df.shape

        rows, width = df.shape
        columns = list(df.columns)
        if rows == 0 or width == 0:
            return df.copy(), [[] for _ in range(rows)]

        raw = df.to_numpy(dtype=object, copy=True)
        text = [
            [self._text(raw[row, column]) for column in range(width)]
            for row in range(rows)
        ]

        global_orders, base_order = self._global_candidates(
            text, columns, col_merge
        )
        candidates = []

        for order in global_orders:
            orders = [order[:] for _ in range(rows)]
            serialized = self._serialize(text, orders)
            candidates.append((self._reuse(serialized), orders, serialized))

        prototype_orders = self._prototype_candidate(
            text, columns, col_merge, base_order
        )
        prototype_serialized = self._serialize(text, prototype_orders)
        candidates.append((
            self._reuse(prototype_serialized),
            prototype_orders,
            prototype_serialized,
        ))

        _, best_orders, best_serialized = max(
            candidates, key=lambda item: item[0]
        )

        row_order = sorted(range(rows), key=lambda row: best_serialized[row])
        output = [
            [raw[row, column] for column in best_orders[row]]
            for row in row_order
        ]
        orderings = [
            [columns[column] for column in best_orders[row]]
            for row in row_order
        ]

        result = pd.DataFrame(
            output,
            columns=columns,
            index=df.index.take(row_order),
            dtype=object,
        )
        return result, orderings