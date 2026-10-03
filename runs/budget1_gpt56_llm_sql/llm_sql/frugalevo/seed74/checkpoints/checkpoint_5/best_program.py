"""Character-prefix-aware DataFrame column reordering algorithm."""

import heapq
from typing import List, Tuple

import numpy as np
import pandas as pd

from solver import Algorithm


class Evolved(Algorithm):
    """Bounded optimizer for serialized character-Trie prefix reuse."""

    def __init__(self, df: pd.DataFrame = None):
        """Store only the optional input reference and last returned orders."""
        self.df = df
        self._last_orders = None

    @staticmethod
    def _cell_string(value) -> str:
        """Convert one scalar using the evaluator's missing-value convention."""
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return ""
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Find the exact common-prefix length using binary-search slicing."""
        if left == right:
            return len(left)
        high = min(len(left), len(right))
        low = 0
        while low < high:
            mid = (low + high + 1) // 2
            if left[:mid] == right[:mid]:
                low = mid
            else:
                high = mid - 1
        return low

    @classmethod
    def _trie_score(cls, strings: List[str]) -> int:
        """Compute ideal character-Trie reuse from sorted adjacent strings."""
        if len(strings) < 2:
            return 0
        ordered = sorted(strings)
        total = 0
        previous = ordered[0]
        for current in ordered[1:]:
            total += len(current) if current == previous else cls._lcp(previous, current)
            previous = current
        return total

    @staticmethod
    def _dependency_edges(columns, requested):
        """Translate valid requested dependencies into deduplicated index edges."""
        positions = {}
        for i, column in enumerate(columns):
            positions.setdefault(str(column), []).append(i)

        edges = []
        seen = set()
        for left, right in requested or []:
            li = positions.get(str(left), [])
            ri = positions.get(str(right), [])
            if len(li) == 1 and len(ri) == 1 and li[0] != ri[0]:
                edge = (li[0], ri[0])
                if edge not in seen:
                    seen.add(edge)
                    edges.append(edge)
        return edges

    @classmethod
    def _constrained_order(cls, m, preference, columns, dependencies):
        """Return a preference-ranked topological ordering honoring dependencies."""
        edges = cls._dependency_edges(columns, dependencies)
        outgoing = [[] for _ in range(m)]
        indegree = [0] * m

        for left, right in edges:
            outgoing[left].append(right)
            indegree[right] += 1

        rank = {column: i for i, column in enumerate(preference)}
        heap = [(rank[i], i) for i in range(m) if indegree[i] == 0]
        heapq.heapify(heap)
        result = []

        while heap:
            _, node = heapq.heappop(heap)
            result.append(node)
            for nxt in outgoing[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    heapq.heappush(heap, (rank[nxt], nxt))

        if len(result) != m:
            return list(preference)
        return result

    @staticmethod
    def _column_statistics(text):
        """Compute repetition-weighted statistics once for every column."""
        n, m = text.shape
        stats = []
        for j in range(m):
            counts = {}
            lengths = {}
            for value in text[:, j]:
                counts[value] = counts.get(value, 0) + 1
                lengths[value] = len(value)

            pair_chars = 0
            repeat_chars = 0
            total_chars = 0
            for value, count in counts.items():
                length = lengths[value]
                pair_chars += length * count * (count - 1)
                repeat_chars += length * max(0, count - 1)
                total_chars += length * count

            stats.append((pair_chars, repeat_chars, total_chars))
        return stats

    @staticmethod
    def _global_preference(stats, mode):
        """Produce one of three deterministic whole-table column preferences."""
        if mode == 0:
            key = lambda item: (-item[1][0], -item[1][1], -item[1][2], item[0])
        elif mode == 1:
            key = lambda item: (-item[1][1], -item[1][0], -item[1][2], item[0])
        else:
            key = lambda item: (-item[1][2], -item[1][0], -item[1][1], item[0])
        return [j for j, _ in sorted(enumerate(stats), key=key)]

    @staticmethod
    def _conditional_orders(text, base_order, stats):
        """Choose a leading field and independently rank suffixes per group."""
        n, m = text.shape
        if n == 0 or m <= 1:
            return [list(base_order) for _ in range(n)]

        # Prefer a field whose repeated values carry many serialized characters.
        first = max(
            range(m),
            key=lambda j: (stats[j][0], stats[j][1], stats[j][2], -j),
        )
        suffix = [j for j in base_order if j != first]

        groups = {}
        for row in range(n):
            groups.setdefault(text[row, first], []).append(row)

        group_orders = {}
        for value, rows in groups.items():
            ranked = []
            for col in suffix:
                counts = {}
                lengths = {}
                for row in rows:
                    cell = text[row, col]
                    counts[cell] = counts.get(cell, 0) + 1
                    lengths[cell] = len(cell)

                pair_score = sum(
                    lengths[cell] * count * (count - 1)
                    for cell, count in counts.items()
                )
                total = sum(len(text[row, col]) for row in rows)
                ranked.append((-pair_score, -total, base_order.index(col), col))

            ranked.sort()
            group_orders[value] = [item[3] for item in ranked]

        return [[first] + group_orders[text[row, first]] for row in range(n)]

    @staticmethod
    def _serialize_rows(text, orders):
        """Serialize every row according to its selected column permutation."""
        return [
            "".join(text[row, col] for col in orders[row])
            for row in range(text.shape[0])
        ]

    @staticmethod
    def _materialize(df, source, orders, serialized):
        """Sort rows by serialized value and materialize untouched source cells."""
        n, m = source.shape
        row_order = sorted(range(n), key=lambda row: (serialized[row], row))
        output = np.empty((n, m), dtype=object)
        public_orders = []

        for output_row, source_row in enumerate(row_order):
            order = orders[source_row]
            output[output_row, :] = [source[source_row, col] for col in order]
            public_orders.append([df.columns[col] for col in order])

        result = pd.DataFrame(
            output,
            index=df.index.take(row_order),
            columns=list(df.columns),
        )
        return result, public_orders

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
    ) -> Tuple[pd.DataFrame, List[List]]:
        """Evaluate three bounded column-order hypotheses and return the best."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        n, m = df.shape
        if n == 0 or m == 0:
            result = df.copy()
            orders = [[] for _ in range(n)]
            self._last_orders = orders
            return result, orders

        source = df.to_numpy(dtype=object, copy=True)
        text = np.empty((n, m), dtype=object)

        for col in range(m):
            text[:, col] = [self._cell_string(value) for value in source[:, col]]

        stats = self._column_statistics(text)
        candidates = []

        # Three distinct global hypotheses, each constrained independently.
        for mode in (0, 1, 2):
            preference = self._global_preference(stats, mode)
            order = self._constrained_order(
                m, preference, list(df.columns), one_way_dep
            )
            candidates.append([order[:] for _ in range(n)])

        # Conditional partitioning hypothesis.
        conditional_base = self._global_preference(stats, 0)
        conditional_base = self._constrained_order(
            m, conditional_base, list(df.columns), one_way_dep
        )
        candidates.append(
            self._conditional_orders(text, conditional_base, stats)
        )

        best_score = None
        best_orders = None
        best_serialized = None

        for orders in candidates:
            serialized = self._serialize_rows(text, orders)
            score = self._trie_score(serialized)
            if best_score is None or score > best_score:
                best_score = score
                best_orders = orders
                best_serialized = serialized

        result, public_orders = self._materialize(
            df, source, best_orders, best_serialized
        )

        if result.shape != df.shape:
            raise RuntimeError("Reordering changed DataFrame shape")

        self._last_orders = public_orders
        return result, public_orders