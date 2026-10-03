"""Bounded character-prefix-aware DataFrame column permutation search."""

import heapq
from typing import List, Tuple

import numpy as np
import pandas as pd

from solver import Algorithm


class Evolved(Algorithm):
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self._last_orders = None

    @staticmethod
    def _cell_string(value) -> str:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return ""
        return str(value)

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        if a == b:
            return len(a)
        lo, hi = 0, min(len(a), len(b))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    @classmethod
    def _score(cls, strings):
        if len(strings) < 2:
            return 0
        strings = sorted(strings)
        total = 0
        previous = strings[0]
        for current in strings[1:]:
            total += len(current) if current == previous else cls._lcp(previous, current)
            previous = current
        return total

    @staticmethod
    def _positions(columns):
        result = {}
        for i, name in enumerate(columns):
            result.setdefault(str(name), []).append(i)
        return result

    @classmethod
    def _blocks(cls, width, columns, requested):
        parent = list(range(width))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        pos = cls._positions(columns)
        for pair in requested or []:
            if len(pair) != 2:
                continue
            left, right = pos.get(str(pair[0]), []), pos.get(str(pair[1]), [])
            if len(left) == 1 and len(right) == 1:
                a, b = find(left[0]), find(right[0])
                if a != b:
                    parent[b] = a

        groups = {}
        for i in range(width):
            groups.setdefault(find(i), []).append(i)
        return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])

    @classmethod
    def _repair(cls, width, preference, columns, dependencies, blocks):
        pos = cls._positions(columns)
        edges, seen = [], set()

        def add(a, b):
            if a != b and (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))

        for pair in dependencies or []:
            if len(pair) != 2:
                continue
            left, right = pos.get(str(pair[0]), []), pos.get(str(pair[1]), [])
            if len(left) == 1 and len(right) == 1:
                add(left[0], right[0])
        for block in blocks:
            for a, b in zip(block, block[1:]):
                add(a, b)

        out = [[] for _ in range(width)]
        indegree = [0] * width
        for a, b in edges:
            out[a].append(b)
            indegree[b] += 1

        rank = {v: i for i, v in enumerate(preference)}
        heap = [(rank.get(i, width + i), i) for i in range(width) if indegree[i] == 0]
        heapq.heapify(heap)
        answer = []
        while heap:
            _, node = heapq.heappop(heap)
            answer.append(node)
            for nxt in out[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    heapq.heappush(heap, (rank.get(nxt, width + nxt), nxt))
        return answer if len(answer) == width else list(preference)

    @staticmethod
    def _stats(text):
        rows, width = text.shape
        result = []
        for col in range(width):
            counts = {}
            for value in text[:, col]:
                counts[value] = counts.get(value, 0) + 1
            pair = repeat = mass = 0
            for value, count in counts.items():
                length = len(value)
                pair += length * count * (count - 1)
                repeat += length * max(0, count - 1)
                mass += length * count
            result.append((pair, repeat, mass))
        return result

    @staticmethod
    def _affinity(text, left, right):
        groups = {}
        for row in range(text.shape[0]):
            groups.setdefault(text[row, left], []).append(row)
        score = 0
        for members in groups.values():
            if len(members) < 2:
                continue
            counts = {}
            for row in members:
                value = text[row, right]
                counts[value] = counts.get(value, 0) + 1
            for value, count in counts.items():
                score += len(value) * count * (count - 1)
        return score

    @classmethod
    def _beam_orders(cls, text, stats, blocks):
        if len(blocks) <= 1:
            return [sum(blocks, [])]

        mass = [sum(stats[c][0] for c in block) for block in blocks]
        active = sorted(range(len(blocks)), key=lambda i: (-mass[i], blocks[i][0]))[:24]
        active_set = frozenset(active)
        reps = [block[:2] for block in blocks]
        affinity = {}
        for a in active:
            for b in active:
                if a != b:
                    affinity[a, b] = sum(
                        cls._affinity(text, x, y) for x in reps[a] for y in reps[b]
                    )

        beam = [((seed,), active_set - {seed}, 0) for seed in active]
        for _ in range(max(0, len(active) - 1)):
            expanded = []
            for path, remaining, value in beam:
                if not remaining:
                    expanded.append((path, remaining, value))
                else:
                    last = path[-1]
                    for nxt in remaining:
                        expanded.append(
                            (path + (nxt,), remaining - {nxt},
                             value + affinity.get((last, nxt), 0))
                        )
            expanded.sort(key=lambda x: (-x[2], x[0]))
            beam = expanded[:4]

        answer, seen = [], set()
        for path, _, _ in sorted(beam, key=lambda x: (-x[2], x[0])):
            used = set(path)
            tail = sorted(
                (i for i in range(len(blocks)) if i not in used),
                key=lambda i: (-mass[i], blocks[i][0]),
            )
            flat = tuple(c for b in tuple(path) + tuple(tail) for c in blocks[b])
            if flat not in seen:
                seen.add(flat)
                answer.append(list(flat))
            if len(answer) == 2:
                break
        return answer or [sum(blocks, [])]

    @staticmethod
    def _serialize(text, order):
        return ["".join(text[row, col] for col in order) for row in range(text.shape[0])]

    @staticmethod
    def _materialize(df, source, order, serialized):
        row_order = sorted(range(len(serialized)), key=lambda r: (serialized[r], r))
        values = np.empty(source.shape, dtype=object)
        public_orders = []
        for out_row, src_row in enumerate(row_order):
            values[out_row, :] = [source[src_row, col] for col in order]
            public_orders.append([df.columns[col] for col in order])
        return (
            pd.DataFrame(values, index=df.index.take(row_order), columns=list(df.columns)),
            public_orders,
        )

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
        for col in range(width):
            text[:, col] = [self._cell_string(v) for v in source[:, col]]

        columns = list(df.columns)
        stats = self._stats(text)
        blocks = self._blocks(width, columns, col_merge)
        baseline = [
            col for col, _ in sorted(
                enumerate(stats),
                key=lambda item: (-item[1][0], -item[1][1], -item[1][2], item[0]),
            )
        ]

        preferences = [baseline] + self._beam_orders(text, stats, blocks)
        best_score, best_order, best_serialized = -1, None, None
        for preference in preferences[:3]:
            order = self._repair(width, preference, columns, one_way_dep, blocks)
            serialized = self._serialize(text, order)
            score = self._score(serialized)
            if score > best_score:
                best_score, best_order, best_serialized = score, order, serialized

        result, public_orders = self._materialize(
            df, source, best_order, best_serialized
        )
        if result.shape != df.shape:
            raise RuntimeError("Reordering changed DataFrame shape")
        self._last_orders = public_orders
        return result, public_orders