import pandas as pd
from collections import Counter, defaultdict
from typing import List, Tuple

from solver import Algorithm


class Evolved(Algorithm):
    """Optimize serialized character-prefix reuse with bounded global and conditional layouts."""

    def __init__(self, df: pd.DataFrame = None):
        """Store the optional dataframe and initialize shape metadata."""
        self.df = df
        self.num_rows = 0
        self.num_cols = 0

    @staticmethod
    def _text(value) -> str:
        """Convert a scalar to evaluator text, mapping scalar missing values to empty text."""
        try:
            missing = pd.isna(value)
            if not hasattr(missing, "__len__") and bool(missing):
                return ""
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Return the exact common-prefix length using binary-search slicing."""
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
        """Compute ideal character-Trie reuse from sorted adjacent serialized rows."""
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
        """Make requested merge groups contiguous while preserving proposed order."""
        if not col_merge:
            return list(order)

        position = {column: index for index, column in enumerate(order)}
        owners = {}
        claimed = set()

        for requested in col_merge:
            group = [
                column for column in order
                if column in requested and column not in claimed
            ]
            if not group:
                continue

            group.sort(key=position.__getitem__)
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
        """Convert integer positions into a valid column permutation respecting merges."""
        proposed = [columns[index] for index in integer_order]
        merged = cls._merge(columns, proposed, col_merge)
        lookup = {column: index for index, column in enumerate(columns)}
        return [lookup[column] for column in merged]

    @staticmethod
    def _serialize(text, orders):
        """Serialize every row according to its row-specific integer column order."""
        return [
            "".join(text[row][column] for column in orders[row])
            for row in range(len(orders))
        ]

    def _global_candidates(self, text, columns, col_merge):
        """Build frequency-ranked and partial-prefix-ranked global layouts."""
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
        """Build one anchor-value layout bank with group-specific suffix rankings.

        Up to six globally promising columns are examined as possible anchors.
        The best anchor is selected by length-weighted repeated-value mass.
        Rows are factorized into anchor-value groups once.  Repeated groups
        receive a suffix ordering ranked by their local pair-repetition mass;
        singleton groups use the global frequency ordering.  Only the first
        32 globally promising suffix columns are scored locally, with the
        remaining columns appended deterministically.
        """
        rows = len(text)
        width = len(columns)

        fallback = self._normalize(columns, base_order, col_merge)
        result = [fallback[:] for _ in range(rows)]

        if rows < 2 or width < 2:
            return result

        anchor_pool = base_order[:min(6, width)]
        anchor = anchor_pool[0]
        anchor_mass = -1

        # Select the anchor using the required length-weighted pair mass.
        for column in anchor_pool:
            counts = Counter(text[row][column] for row in range(rows))
            mass = sum(
                len(value) * count * (count - 1)
                for value, count in counts.items()
            )
            if mass > anchor_mass:
                anchor_mass = mass
                anchor = column

        # Factorize the chosen anchor exactly once.
        code_by_value = {}
        group_rows = []
        for row in range(rows):
            value = text[row][anchor]
            code = code_by_value.get(value)
            if code is None:
                code = len(group_rows)
                code_by_value[value] = code
                group_rows.append([])
            group_rows[code].append(row)

        rank_position = {
            column: position for position, column in enumerate(base_order)
        }

        suffix_candidates = [
            column for column in base_order
            if column != anchor
        ][:min(32, width - 1)]

        suffix_set = set(suffix_candidates)
        tail = [
            column for column in base_order
            if column != anchor and column not in suffix_set
        ]

        global_integer_order = [anchor] + suffix_candidates + tail
        global_order = self._normalize(
            columns,
            global_integer_order,
            col_merge,
        )

        for members in group_rows:
            if len(members) < 2:
                for row in members:
                    result[row] = global_order[:]
                continue

            ranked = []
            for column in suffix_candidates:
                counts = Counter(text[row][column] for row in members)
                mass = sum(
                    len(value) * count * (count - 1)
                    for value, count in counts.items()
                )
                ranked.append((
                    -mass,
                    rank_position[column],
                    column,
                ))

            ranked.sort()
            local_suffix = [entry[2] for entry in ranked] + tail
            local_order = self._normalize(
                columns,
                [anchor] + local_suffix,
                col_merge,
            )

            for row in members:
                result[row] = local_order[:]

        return result

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
        """Return the highest-scoring bounded layout while preserving all data and identity."""
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
            text,
            columns,
            col_merge,
        )

        candidates = []

        for order in global_orders:
            orders = [order[:] for _ in range(rows)]
            serialized = self._serialize(text, orders)
            candidates.append((
                self._reuse(serialized),
                orders,
                serialized,
            ))

        conditional_orders = self._prototype_candidate(
            text,
            columns,
            col_merge,
            base_order,
        )
        conditional_serialized = self._serialize(
            text,
            conditional_orders,
        )
        candidates.append((
            self._reuse(conditional_serialized),
            conditional_orders,
            conditional_serialized,
        ))

        _, best_orders, best_serialized = max(
            candidates,
            key=lambda candidate: candidate[0],
        )

        row_order = sorted(
            range(rows),
            key=lambda row: best_serialized[row],
        )

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