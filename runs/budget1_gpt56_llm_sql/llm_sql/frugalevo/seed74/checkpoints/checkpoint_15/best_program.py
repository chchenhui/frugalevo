"""Bounded beam search for character-prefix-aware DataFrame reordering."""

import heapq
from typing import List, Tuple

import numpy as np
import pandas as pd

from solver import Algorithm


class Evolved(Algorithm):
    """Optimize valid column permutations with bounded block-topology search."""

    def __init__(self, df: pd.DataFrame = None):
        """Store the optional frame and the latest public row orders."""
        self.df = df
        self._last_orders = None

    @staticmethod
    def _cell_string(value) -> str:
        """Serialize one scalar like the evaluator, converting scalar missing values to ''."""
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
        """Return the sorted-adjacent character Trie reuse score."""
        if len(strings) < 2:
            return 0
        ordered = sorted(strings)
        score = 0
        previous = ordered[0]
        for current in ordered[1:]:
            score += len(current) if current == previous else cls._lcp(
                previous, current
            )
            previous = current
        return score

    @staticmethod
    def _positions(columns):
        """Map stringified column labels to source positions."""
        result = {}
        for i, column in enumerate(columns):
            result.setdefault(str(column), []).append(i)
        return result

    @classmethod
    def _edges(cls, columns, dependencies, blocks):
        """Translate dependency and merge requests into directed index edges."""
        positions = cls._positions(columns)
        edges = []
        seen = set()

        def add(left, right):
            """Add one valid, nonduplicate directed edge."""
            if left != right and (left, right) not in seen:
                seen.add((left, right))
                edges.append((left, right))

        for pair in dependencies or []:
            if len(pair) != 2:
                continue
            left = positions.get(str(pair[0]), [])
            right = positions.get(str(pair[1]), [])
            if len(left) == 1 and len(right) == 1:
                add(left[0], right[0])

        for block in blocks:
            for left, right in zip(block, block[1:]):
                add(left, right)
        return edges

    @classmethod
    def _constrained_order(cls, width, preference, columns, dependencies, blocks):
        """Topologically repair a preference while honoring all valid constraints."""
        outgoing = [[] for _ in range(width)]
        indegree = [0] * width
        for left, right in cls._edges(columns, dependencies, blocks):
            outgoing[left].append(right)
            indegree[right] += 1

        rank = {value: i for i, value in enumerate(preference)}
        queue = [
            (rank.get(i, width + i), i)
            for i in range(width)
            if indegree[i] == 0
        ]
        heapq.heapify(queue)

        result = []
        while queue:
            _, node = heapq.heappop(queue)
            result.append(node)
            for successor in outgoing[node]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    heapq.heappush(
                        queue, (rank.get(successor, width + successor), successor)
                    )
        return result if len(result) == width else list(preference)

    @staticmethod
    def _stats(text):
        """Compute pair repetition, ordinary repetition, and character mass per column."""
        rows, width = text.shape
        result = []
        for column in range(width):
            counts = {}
            for row in range(rows):
                value = text[row, column]
                counts[value] = counts.get(value, 0) + 1

            pair_mass = repeat_mass = total_mass = 0
            for value, count in counts.items():
                length = len(value)
                pair_mass += length * count * (count - 1)
                repeat_mass += length * max(0, count - 1)
                total_mass += length * count
            result.append((pair_mass, repeat_mass, total_mass))
        return result

    @staticmethod
    def _merge_blocks(width, columns, requested):
        """Build atomic ordered blocks from valid merge requests."""
        parent = list(range(width))

        def find(value):
            """Find a disjoint-set root with path compression."""
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        positions = {}
        for i, column in enumerate(columns):
            positions.setdefault(str(column), []).append(i)

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
        for i in range(width):
            groups.setdefault(find(i), []).append(i)
        return sorted((sorted(group) for group in groups.values()), key=lambda x: x[0])

    @staticmethod
    def _affinity(text, left, right):
        """Measure length-weighted repeated right values conditioned on a left value."""
        grouped = {}
        for row in range(text.shape[0]):
            grouped.setdefault(text[row, left], []).append(row)

        score = 0
        for rows in grouped.values():
            if len(rows) < 2:
                continue
            counts = {}
            for row in rows:
                value = text[row, right]
                counts[value] = counts.get(value, 0) + 1
            for value, count in counts.items():
                score += len(value) * count * (count - 1)
        return score

    @classmethod
    def _block_candidates(cls, text, stats, blocks):
        """Construct at most two block orders with a bounded width-four lookahead beam."""
        if len(blocks) <= 1:
            return [sum(blocks, [])]

        mass = [
            sum(stats[column][0] for column in block)
            for block in blocks
        ]
        active = sorted(
            range(len(blocks)),
            key=lambda i: (-mass[i], blocks[i][0]),
        )[:32]
        active_set = frozenset(active)

        # At most two representatives per block keep pairwise affinity bounded.
        representatives = [block[:2] for block in blocks]
        affinity = {}

        for left_block in active:
            for right_block in active:
                if left_block == right_block:
                    continue
                total = 0
                for left in representatives[left_block]:
                    for right in representatives[right_block]:
                        total += cls._affinity(text, left, right)
                affinity[left_block, right_block] = total

        # Each state contains a distinct partial topology.  Keeping four states
        # avoids the incumbent's single greedy successor commitment.
        beam = [
            ((seed,), active_set - {seed}, 0)
            for seed in active
        ]

        for _ in range(max(0, len(active) - 1)):
            expanded = []
            for path, remaining, value in beam:
                if not remaining:
                    expanded.append((path, remaining, value))
                    continue

                previous = path[-1]
                for successor in remaining:
                    expanded.append(
                        (
                            path + (successor,),
                            remaining - {successor},
                            value + affinity.get((previous, successor), 0),
                        )
                    )

            if not expanded:
                break

            expanded.sort(
                key=lambda state: (
                    -state[2],
                    -sum(mass[i] for i in state[1]),
                    state[0],
                )
            )
            beam = expanded[:4]

        completed = []
        for path, remaining, value in beam:
            used = set(path)
            tail = sorted(
                (i for i in range(len(blocks)) if i not in used),
                key=lambda i: (-mass[i], blocks[i][0]),
            )
            completed.append((value, path + tuple(tail)))

        completed.sort(key=lambda item: (-item[0], item[1]))
        result = []
        seen = set()

        for _, path in completed:
            flattened = tuple(
                column
                for block_index in path
                for column in blocks[block_index]
            )
            if flattened not in seen:
                seen.add(flattened)
                result.append(list(flattened))
            if len(result) == 2:
                break

        return result or [sum(blocks, [])]

    @staticmethod
    def _serialize(text, orders):
        """Serialize each source row using its row-specific column permutation."""
        return [
            "".join(text[row, column] for column in orders[row])
            for row in range(text.shape[0])
        ]

    @staticmethod
    def _materialize(df, source, orders, serialized):
        """Sort rows by serialized keys and copy every original cell unchanged."""
        row_order = sorted(
            range(source.shape[0]),
            key=lambda row: (serialized[row], row),
        )
        values = np.empty(source.shape, dtype=object)
        public_orders = []

        for output_row, source_row in enumerate(row_order):
            order = orders[source_row]
            values[output_row, :] = [source[source_row, column] for column in order]
            public_orders.append([df.columns[column] for column in order])

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
        """Evaluate one global and two beam block orders using exact Trie scoring."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        rows, width = df.shape
        if rows == 0 or width == 0:
            result = df.copy()
            orders = [[] for _ in range(rows)]
            self._last_orders = orders
            return result, orders

        source = df.to_numpy(dtype=object, copy=True)
        text = np.empty((rows, width), dtype=object)
        for column in range(width):
            text[:, column] = [
                self._cell_string(value) for value in source[:, column]
            ]

        stats = self._stats(text)
        columns = list(df.columns)
        blocks = self._merge_blocks(width, columns, col_merge)

        baseline = [
            column
            for column, _ in sorted(
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
            repaired = self._constrained_order(
                width,
                preference,
                columns,
                one_way_dep,
                blocks,
            )
            orders = [repaired[:] for _ in range(rows)]
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