import pandas as pd
from collections import Counter, defaultdict
from typing import List, Tuple

from solver import Algorithm


class Evolved(Algorithm):
    """Bounded column-order optimizer for character-level serialized Trie reuse."""

    def __init__(self, df: pd.DataFrame = None):
        """Store the optional dataframe and initialize shape metadata."""
        self.df = df
        self.num_rows = 0
        self.num_cols = 0

    @staticmethod
    def _text(value) -> str:
        """Convert one cell to evaluator-compatible text, mapping scalar missing values to empty text."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, bool) and missing:
                return ""
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Compute the common-prefix length using binary search and C-level string slicing."""
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
        """Compute exact character-Trie reuse as LCPs of lexicographically adjacent rows."""
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
        """Make requested column groups contiguous without changing their internal proposed order."""
        if not col_merge:
            return list(order)

        positions = {column: index for index, column in enumerate(order)}
        owner = {}
        claimed = set()

        for request in col_merge:
            group = [
                column for column in order
                if column in request and column not in claimed
            ]
            if not group:
                continue

            group.sort(key=positions.__getitem__)
            for column in group:
                owner[column] = group
                claimed.add(column)

        result = []
        emitted = set()

        for column in order:
            if column in emitted:
                continue

            group = owner.get(column)
            if group is None:
                result.append(column)
                emitted.add(column)
            else:
                result.extend(group)
                emitted.update(group)

        return result

    @staticmethod
    def _normalize(columns, integer_order, col_merge):
        """Apply merge constraints to an integer order and convert it back to integer positions."""
        proposed = [columns[index] for index in integer_order]
        merged = Evolved._merge(columns, proposed, col_merge)
        lookup = {column: index for index, column in enumerate(columns)}
        return [lookup[column] for column in merged]

    @staticmethod
    def _serialize(text, orders):
        """Serialize each source row according to its row-specific column permutation."""
        return [
            "".join(text[row][column] for column in orders[row])
            for row in range(len(orders))
        ]

    def _global_candidates(self, text, columns, col_merge):
        """Build a frequency-ranked order and a disjoint repeated ordered-pair-block order."""
        rows = len(text)
        width = len(columns)

        statistics = []
        for column in range(width):
            counts = Counter(text[row][column] for row in range(rows))
            pair_mass = sum(
                len(value) * count * (count - 1)
                for value, count in counts.items()
            )
            statistics.append((pair_mass, len(counts)))

        ranked = sorted(
            range(width),
            key=lambda column: (
                -statistics[column][0],
                statistics[column][1],
                column,
            ),
        )

        singleton = self._normalize(columns, ranked, col_merge)

        examined = ranked[:min(28, width)]
        pair_scores = []

        for first in examined:
            for second in examined:
                if first == second:
                    continue

                counts = Counter(
                    text[row][first] + text[row][second]
                    for row in range(rows)
                )
                score = sum(
                    len(value) * count * (count - 1)
                    for value, count in counts.items()
                )

                if score:
                    pair_scores.append((-score, first, second))

        pair_scores.sort()

        used = set()
        pair_order = []

        for negative_score, first, second in pair_scores:
            if first in used or second in used:
                continue
            pair_order.extend((first, second))
            used.add(first)
            used.add(second)

        pair_order.extend(column for column in ranked if column not in used)
        pair_order = self._normalize(columns, pair_order, col_merge)

        return [singleton, pair_order], ranked, statistics

    def _conditional(self, text, columns, col_merge, base_order, statistics):
        """Construct bounded row-specific orders by recursively partitioning on repeated field values."""
        rows = len(text)
        width = len(columns)
        result = [None] * rows

        useful = [
            column for column in base_order
            if statistics[column][0] > 0
        ][:min(20, width)]
        useful_set = set(useful)

        node_limit = 384
        max_depth = min(8, len(useful))
        visited = 0

        def finish(group, prefix, remaining):
            """Assign the normalized global tail to every row in a terminal group."""
            tail = [column for column in base_order if column in remaining]
            order = self._normalize(columns, prefix + tail, col_merge)
            for row in group:
                result[row] = order[:]

        def split(group, remaining, prefix, depth):
            """Select the best length-weighted repeated field and recurse into its value groups."""
            nonlocal visited
            visited += 1

            candidates = [
                column for column in remaining
                if column in useful_set
            ]

            if (
                len(group) < 2
                or not candidates
                or depth >= max_depth
                or visited > node_limit
            ):
                finish(group, prefix, remaining)
                return

            best_column = -1
            best_score = 0
            best_groups = None

            for column in candidates:
                groups = defaultdict(list)
                for row in group:
                    groups[text[row][column]].append(row)

                score = sum(
                    len(value) * len(members) * (len(members) - 1)
                    for value, members in groups.items()
                    if len(members) > 1
                )

                if score > best_score or (
                    score == best_score and score > 0
                    and (best_column < 0 or column < best_column)
                ):
                    best_column = column
                    best_score = score
                    best_groups = groups

            if best_column < 0 or best_score <= 0:
                finish(group, prefix, remaining)
                return

            rest = [column for column in remaining if column != best_column]
            children = sorted(
                best_groups.values(),
                key=lambda members: (-len(members), min(members)),
            )

            for child in children:
                split(child, rest, prefix + [best_column], depth + 1)

        split(list(range(rows)), list(range(width)), [], 0)

        fallback = self._normalize(columns, base_order, col_merge)
        for row in range(rows):
            if result[row] is None:
                result[row] = fallback[:]

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
        """Evaluate global and conditional layouts, returning the highest-reuse valid layout."""
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

        global_candidates, base_order, statistics = self._global_candidates(
            text, columns, col_merge
        )

        candidates = []

        for order in global_candidates:
            orders = [order[:] for _ in range(rows)]
            serialized = self._serialize(text, orders)
            candidates.append((self._reuse(serialized), orders, serialized))

        conditional = self._conditional(
            text,
            columns,
            col_merge,
            base_order,
            statistics,
        )
        conditional_serialized = self._serialize(text, conditional)
        candidates.append((
            self._reuse(conditional_serialized),
            conditional,
            conditional_serialized,
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