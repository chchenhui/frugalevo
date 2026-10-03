# EVOLVE-BLOCK-START
from typing import Tuple, List
import numpy as np
import pandas as pd
from solver import Algorithm


class Evolved(Algorithm):
    """Lossless bounded optimizer for character-prefix reuse in serialized rows."""

    def __init__(self, df: pd.DataFrame = None):
        """Store the optional source dataframe and lightweight shape metadata."""
        self.df = df
        self.num_rows = 0
        self.num_cols = 0

    @staticmethod
    def _cell_text(value) -> str:
        """Convert one cell to the evaluator's text form, mapping scalar missing values to ''."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and bool(missing):
                return ""
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Compute a common-prefix length with logarithmic C-level slice comparisons."""
        high = min(len(left), len(right))
        low = 0
        while low < high:
            middle = (low + high + 1) // 2
            if left[:middle] == right[:middle]:
                low = middle
            else:
                high = middle - 1
        return low

    @classmethod
    def _trie_score(cls, strings: List[str]) -> int:
        """Return ideal character-Trie reuse as sorted adjacent-string LCP totals."""
        if len(strings) < 2:
            return 0
        ordered = sorted(strings)
        return sum(
            cls._lcp(ordered[i - 1], ordered[i])
            for i in range(1, len(ordered))
        )

    @classmethod
    def _group_score(cls, cells, rows, order) -> int:
        """Score one leaf ordering using the exact serialized character objective."""
        values = [
            "".join(cells[row, column] for column in order)
            for row in rows
        ]
        if len(values) < 2:
            return 0
        values.sort()
        return sum(
            cls._lcp(values[i - 1], values[i])
            for i in range(1, len(values))
        )

    @staticmethod
    def _make_strings(cells, orders) -> List[str]:
        """Serialize every row according to its supplied column permutation."""
        return [
            "".join(cells[row, column] for column in orders[row])
            for row in range(cells.shape[0])
        ]

    def _global_order(self, cells, mode: int) -> List[int]:
        """Rank columns by deterministic length-weighted value repetition."""
        n, m = cells.shape
        ranked = []

        for column in range(m):
            counts = {}
            total_length = 0
            for value in cells[:, column]:
                total_length += len(value)
                counts[value] = counts.get(value, 0) + 1

            repetition = sum(
                count * (count - 1)
                for count in counts.values()
            )
            weighted = sum(
                len(value) * count * (count - 1)
                for value, count in counts.items()
            )

            if mode == 0:
                key = weighted
            elif mode == 1:
                key = repetition * max(1, total_length // max(1, n))
            else:
                key = weighted + repetition

            ranked.append((key, column))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [column for _, column in ranked]

    def _refine_leaf(self, cells, rows, order) -> List[int]:
        """Improve a leaf suffix with two bounded exact adjacent-swap passes."""
        if len(order) < 2 or len(rows) < 2:
            return list(order)

        current = list(order)
        score = self._group_score(cells, rows, current)

        # Only the first twelve adjacent boundaries are considered, keeping
        # refinement bounded on wide tables.
        boundary_count = min(12, len(current) - 1)

        for reverse in (False, True):
            positions = (
                range(boundary_count - 1, -1, -1)
                if reverse else range(boundary_count)
            )
            for position in positions:
                trial = current[:]
                trial[position], trial[position + 1] = (
                    trial[position + 1],
                    trial[position],
                )
                trial_score = self._group_score(cells, rows, trial)
                if trial_score > score:
                    current = trial
                    score = trial_score

        return current

    def _conditional_orders(self, cells) -> List[List[int]]:
        """Partition by one repeated leading field and refine each group's suffix."""
        n, m = cells.shape
        if m <= 1:
            return [list(range(m)) for _ in range(n)]

        lead_candidates = []
        for column in range(m):
            counts = {}
            for value in cells[:, column]:
                counts[value] = counts.get(value, 0) + 1

            score = sum(
                len(value) * count * (count - 1)
                for value, count in counts.items()
            )
            lead_candidates.append((score, column))

        lead = max(
            lead_candidates,
            key=lambda item: (item[0], -item[1])
        )[1]

        groups = {}
        for row in range(n):
            groups.setdefault(cells[row, lead], []).append(row)

        remaining = [column for column in range(m) if column != lead]
        result = [None] * n

        for rows in groups.values():
            ranked = []
            for column in remaining:
                counts = {}
                for row in rows:
                    value = cells[row, column]
                    counts[value] = counts.get(value, 0) + 1

                score = sum(
                    len(value) * count * (count - 1)
                    for value, count in counts.items()
                )
                ranked.append((score, column))

            ranked.sort(key=lambda item: (-item[0], item[1]))
            suffix = [column for _, column in ranked]
            suffix = self._refine_leaf(cells, rows, suffix)
            full_order = [lead] + suffix

            for row in rows:
                result[row] = full_order[:]

        return result

    def _candidate(self, cells, orders):
        """Serialize, exactly score, and lexicographically order one candidate."""
        strings = self._make_strings(cells, orders)
        score = self._trie_score(strings)
        row_order = sorted(
            range(len(strings)),
            key=lambda row: (strings[row], row)
        )
        return score, strings, row_order

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
        """Evaluate two global orders and one refined conditional order losslessly."""
        n, m = df.shape
        self.num_rows = n
        self.num_cols = m

        if n == 0 or m == 0:
            return df.copy(), [[] for _ in range(n)]

        raw = df.to_numpy(dtype=object, copy=True)
        cells = np.empty((n, m), dtype=object)

        for row in range(n):
            for column in range(m):
                cells[row, column] = self._cell_text(raw[row, column])

        global_a = self._global_order(cells, 0)
        global_b = self._global_order(cells, 1)
        conditional = self._conditional_orders(cells)

        candidates = [
            [global_a[:] for _ in range(n)],
            [global_b[:] for _ in range(n)],
            conditional,
        ]

        best = None
        for orders in candidates:
            score, strings, row_order = self._candidate(cells, orders)
            signature = tuple(orders[0]) if orders else ()
            tie_break = tuple(-column for column in signature)
            choice = (score, tie_break)

            if best is None or choice > best[0]:
                best = (choice, strings, row_order, orders)

        _, strings, row_order, orders = best

        output = np.empty((n, m), dtype=object)
        column_orders = []

        for destination, source in enumerate(row_order):
            source_order = orders[source]
            output[destination, :] = [
                raw[source, column] for column in source_order
            ]
            column_orders.append([
                df.columns[column] for column in source_order
            ])

        result = pd.DataFrame(output, columns=df.columns)
        return result, column_orders


# EVOLVE-BLOCK-END