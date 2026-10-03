import pandas as pd
from collections import Counter
from typing import List, Tuple

from solver import Algorithm


class Evolved(Algorithm):
    """Bounded character-prefix optimizer using global and prototype-aligned layouts."""

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
            if isinstance(missing, bool) and missing:
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
        """Compute exact Trie reuse as LCP lengths of lexicographically adjacent strings."""
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
        """Make requested column groups contiguous without changing source values."""
        if not col_merge:
            return list(order)

        positions = {column: i for i, column in enumerate(order)}
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
        """Apply merge constraints and return a complete integer permutation."""
        proposed = [columns[i] for i in integer_order]
        merged = cls._merge(columns, proposed, col_merge)
        lookup = {column: i for i, column in enumerate(columns)}
        return [lookup[column] for column in merged]

    @staticmethod
    def _serialize(text, orders):
        """Serialize each row using its row-specific column permutation."""
        return [
            "".join(text[row][column] for column in orders[row])
            for row in range(len(orders))
        ]

    def _global_candidates(self, text, columns, col_merge):
        """Build frequency-ranked and repeated-pair global column orderings."""
        rows = len(text)
        width = len(columns)
        statistics = []

        for column in range(width):
            counts = Counter(text[row][column] for row in range(rows))
            mass = sum(
                len(value) * count * (count - 1)
                for value, count in counts.items()
            )
            statistics.append((mass, len(counts)))

        ranked = sorted(
            range(width),
            key=lambda column: (
                -statistics[column][0],
                statistics[column][1],
                column,
            ),
        )

        first = self._normalize(columns, ranked, col_merge)

        examined = ranked[:min(28, width)]
        pair_scores = []

        for left in examined:
            for right in examined:
                if left == right:
                    continue

                counts = Counter(
                    text[row][left] + text[row][right]
                    for row in range(rows)
                )
                score = sum(
                    len(value) * count * (count - 1)
                    for value, count in counts.items()
                )
                if score:
                    pair_scores.append((-score, left, right))

        pair_scores.sort()
        used = set()
        paired = []

        for _, left, right in pair_scores:
            if left in used or right in used:
                continue
            paired.extend((left, right))
            used.add(left)
            used.add(right)

        paired.extend(column for column in ranked if column not in used)
        second = self._normalize(columns, paired, col_merge)

        return [first, second], ranked

    def _prototype_candidate(self, text, columns, col_merge, base_order):
        """Align bounded row samples to four diverse anchors using greedy prefix matching."""
        rows = len(text)
        width = len(columns)
        fallback = self._normalize(columns, base_order, col_merge)
        result = [fallback[:] for _ in range(rows)]

        if rows < 2 or width < 2:
            return result

        sample_rows = list(range(min(rows, 256)))
        active = base_order[:min(width, 32)]

        def make_string(row, order):
            """Serialize one row through a bounded order."""
            return "".join(text[row][column] for column in order)

        initial = [make_string(row, active) for row in sample_rows]

        # Select anchors by farthest-first prefix distance from already selected rows.
        anchor_positions = [0]
        anchor_strings = [initial[0]]

        while len(anchor_positions) < min(4, len(sample_rows)):
            best_position = -1
            best_distance = -1

            for position, value in enumerate(initial):
                if position in anchor_positions:
                    continue

                nearest = max(
                    self._lcp(value, prototype)
                    for prototype in anchor_strings
                )
                distance = len(value) - nearest

                if distance > best_distance:
                    best_distance = distance
                    best_position = position

            if best_position < 0:
                break

            anchor_positions.append(best_position)
            anchor_strings.append(initial[best_position])

        def assign(strings, prototypes):
            """Assign each sampled row to the prototype with greatest character LCP."""
            assignments = []

            for value in strings:
                chosen = 0
                chosen_score = -1

                for index, prototype in enumerate(prototypes):
                    score = self._lcp(value, prototype)
                    if score > chosen_score:
                        chosen = index
                        chosen_score = score

                assignments.append(chosen)

            return assignments

        def construct(strings, prototypes):
            """Construct row orders by greedily matching fields to prototype suffixes."""
            assignments = assign(strings, prototypes)
            orders = [fallback[:] for _ in range(rows)]

            for position, row in enumerate(sample_rows):
                prototype = prototypes[assignments[position]]
                remaining = list(active)
                chosen = []
                cursor = 0

                while remaining:
                    best_column = remaining[0]
                    best_score = -1

                    for column in remaining:
                        score = self._lcp(
                            text[row][column],
                            prototype[cursor:],
                        )
                        if score > best_score:
                            best_score = score
                            best_column = column

                    chosen.append(best_column)
                    cursor += len(text[row][best_column])
                    remaining.remove(best_column)

                tail = [
                    column for column in base_order
                    if column not in chosen
                ]
                orders[row] = self._normalize(
                    columns,
                    chosen + tail,
                    col_merge,
                )

            return orders

        first_orders = construct(initial, anchor_strings)
        first_strings = [
            make_string(row, first_orders[row])
            for row in sample_rows
        ]

        # Rebuild prototypes from actual generated row strings for the second round.
        prototypes = [
            first_strings[position]
            for position in anchor_positions
        ]
        return construct(first_strings, prototypes)

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
        """Select the highest exact-reuse candidate and return its reordered dataframe."""
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

        global_candidates, base_order = self._global_candidates(
            text,
            columns,
            col_merge,
        )

        candidates = []

        for order in global_candidates:
            orders = [order[:] for _ in range(rows)]
            serialized = self._serialize(text, orders)
            candidates.append((
                self._reuse(serialized),
                orders,
                serialized,
            ))

        prototype_orders = self._prototype_candidate(
            text,
            columns,
            col_merge,
            base_order,
        )
        prototype_serialized = self._serialize(text, prototype_orders)
        candidates.append((
            self._reuse(prototype_serialized),
            prototype_orders,
            prototype_serialized,
        ))

        _, best_orders, best_serialized = max(
            candidates,
            key=lambda item: item[0],
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