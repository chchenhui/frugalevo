"""Bounded beam-search optimization for character-prefix Trie reuse."""

import heapq
from typing import List, Tuple

import numpy as np
import pandas as pd

from solver import Algorithm


class Evolved(Algorithm):
    """Optimize column permutations using bounded block-topology search."""

    def __init__(self, df: pd.DataFrame = None):
        """Store the optional frame and the most recently returned orders."""
        self.df = df
        self._last_orders = None

    @staticmethod
    def _cell_string(value) -> str:
        """Serialize a scalar, replacing scalar missing values with an empty string."""
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return ""
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Return the exact common-prefix length using binary-search slicing."""
        if left == right:
            return len(left)
        lo, hi = 0, min(len(left), len(right))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if left[:mid] == right[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    @classmethod
    def _trie_score(cls, strings: List[str]) -> int:
        """Score a fixed row-string multiset by adjacent sorted-string LCP reuse."""
        if len(strings) < 2:
            return 0
        values = sorted(strings)
        score = 0
        previous = values[0]
        for current in values[1:]:
            score += len(current) if current == previous else cls._lcp(
                previous, current
            )
            previous = current
        return score

    @staticmethod
    def _positions(columns):
        """Map stringified labels to their source positions."""
        result = {}
        for i, column in enumerate(columns):
            result.setdefault(str(column), []).append(i)
        return result

    @classmethod
    def _edges(cls, columns, dependencies, blocks):
        """Translate dependencies and merge adjacency into directed edges."""
        positions = cls._positions(columns)
        result = []
        seen = set()

        def add(a, b):
            if a != b and (a, b) not in seen:
                seen.add((a, b))
                result.append((a, b))

        for pair in dependencies or []:
            if len(pair) != 2:
                continue
            left = positions.get(str(pair[0]), [])
            right = positions.get(str(pair[1]), [])
            if len(left) == 1 and len(right) == 1:
                add(left[0], right[0])

        for block in blocks:
            for a, b in zip(block, block[1:]):
                add(a, b)
        return result

    @classmethod
    def _constrained_order(cls, m, preference, columns, dependencies, blocks):
        """Topologically repair a preference while retaining all requested constraints."""
        outgoing = [[] for _ in range(m)]
        indegree = [0] * m
        for a, b in cls._edges(columns, dependencies, blocks):
            outgoing[a].append(b)
            indegree[b] += 1

        rank = {value: i for i, value in enumerate(preference)}
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

        return result if len(result) == m else list(preference)

    @staticmethod
    def _stats(text):
        """Compute repeated-value character statistics for each column."""
        _, m = text.shape
        result = []
        for j in range(m):
            counts = {}
            for value in text[:, j]:
                counts[value] = counts.get(value, 0) + 1
            pairs = sum(len(v) * c * (c - 1) for v, c in counts.items())
            repeats = sum(len(v) * max(0, c - 1) for v, c in counts.items())
            total = sum(len(v) * c for v, c in counts.items())
            result.append((pairs, repeats, total))
        return result

    @staticmethod
    def _merge_blocks(m, columns, requested):
        """Build deterministic source-order blocks from valid merge requests."""
        parent = list(range(m))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        positions = {}
        for i, name in enumerate(columns):
            positions.setdefault(str(name), []).append(i)

        for pair in requested or []:
            if len(pair) != 2:
                continue
            left = positions.get(str(pair[0]), [])
            right = positions.get(str(pair[1]), [])
            if len(left) == 1 and len(right) == 1:
                a, b = find(left[0]), find(right[0])
                if a != b:
                    parent[b] = a

        groups = {}
        for i in range(m):
            groups.setdefault(find(i), []).append(i)
        return sorted((sorted(v) for v in groups.values()), key=lambda x: x[0])

    @staticmethod
    def _affinity(text, left, right):
        """Measure length-weighted repeated right values within equal-left groups."""
        groups = {}
        for row, value in enumerate(text[:, left]):
            groups.setdefault(value, []).append(row)

        score = 0
        for rows in groups.values():
            if len(rows) < 2:
                continue
            counts = {}
            for row in rows:
                value = text[row, right]
                counts[value] = counts.get(value, 0) + 1
            score += sum(len(v) * c * (c - 1) for v, c in counts.items())
        return score

    @classmethod
    def _block_candidates(cls, text, stats, blocks):
        """Construct two block orders with bounded four-way beam lookahead."""
        if len(blocks) <= 1:
            return [sum(blocks, [])]

        mass = [sum(stats[j][0] for j in block) for block in blocks]
        active = sorted(
            range(len(blocks)),
            key=lambda k: (-mass[k], blocks[k][0]),
        )[:32]
        active_set = set(active)

        reps = [blocks[k][:2] for k in range(len(blocks))]
        affinity = {}
        for a in active:
            for b in active:
                if a == b:
                    continue
                value = 0
                for left in reps[a]:
                    for right in reps[b]:
                        value += cls._affinity(text, left, right)
                affinity[a, b] = value

        # Each state is (path tuple, remaining frozenset, accumulated affinity).
        beams = []
        for seed in active:
            beams.append(((seed,), frozenset(active_set - {seed}), 0))

        for _ in range(max(0, len(active) - 1)):
            expanded = []
            for path, remaining, value in beams:
                if not remaining:
                    expanded.append((path, remaining, value))
                    continue
                previous = path[-1]
                for nxt in remaining:
                    expanded.append(
                        (
                            path + (nxt,),
                            remaining - {nxt},
                            value + affinity.get((previous, nxt), 0),
                        )
                    )

            if not expanded:
                break

            # Keep distinct topologies. Remaining mass is a cheap admissible
            # tie-break that favors paths retaining useful repeated blocks.
            expanded.sort(
                key=lambda state: (
                    -state[2],
                    -sum(mass[k] for k in state[1]),
                    state[0],
                )
            )
            beams = expanded[:4]

        completed = []
        for path, remaining, value in beams:
            tail = sorted(
                (k for k in range(len(blocks)) if k not in set(path)),
                key=lambda k: (-mass[k], blocks[k][0]),
            )
            completed.append((value, path + tuple(tail)))

        completed.sort(key=lambda item: (-item[0], item[1]))
        return [
            [column for k in path for column in blocks[k]]
            for _, path in completed[:2]
        ]

    @staticmethod
    def _serialize(text, orders):
        """Serialize each source row according to its row-specific column order."""
        return [
            "".join(text[row, column] for column in orders[row])
            for row in range(text.shape[0])
        ]

    @staticmethod
    def _materialize(df, source, orders, serialized):
        """Sort rows by serialized keys and copy original cells without modification."""
        n, m = source.shape
        row_order = sorted(range(n), key=lambda r: (serialized[r], r))
        values = np.empty((n, m), dtype=object)
        public_orders = []

        for output_row, source_row in enumerate(row_order):
            order = orders[source_row]
            values[output_row, :] = [source[source_row, c] for c in order]
            public_orders.append([df.columns[c] for c in order])

        result = pd.DataFrame(
            values,
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
        """Evaluate a baseline and beam-generated candidates, selecting exact best reuse."""
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
        for j in range(m):
            text[:, j] = [self._cell_string(v) for v in source[:, j]]

        stats = self._stats(text)
        columns = list(df.columns)
        blocks = self._merge_blocks(m, columns, col_merge)

        baseline = [
            j for j, _ in sorted(
                enumerate(stats),
                key=lambda item: (
                    -item[1][0],
                    -item[1][1],
                    -item[1][2],
                    item[0],
                ),
            )
        ]

        preferences = [baseline]
        preferences.extend(self._block_candidates(text, stats, blocks))

        best_score = -1
        best_orders = None
        best_serialized = None

        for preference in preferences[:3]:
            order = self._constrained_order(
                m, preference, columns, one_way_dep, blocks
            )
            orders = [order[:] for _ in range(n)]
            serialized = self._serialize(text, orders)
            score = self._trie_score(serialized)

            if score > best_score:
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