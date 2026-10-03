# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Any, List, Tuple

from solver import Algorithm


class Evolved(Algorithm):
    """
    Reorder rows and, when beneficial, reorder cells within each row to improve
    serialized character-prefix reuse while preserving every original object.
    """

    def __init__(self, df: pd.DataFrame = None):
        """Store the optional input frame for compatibility with Algorithm."""
        self.df = df

    @staticmethod
    def _string_value(value: Any) -> str:
        """Return evaluator-compatible serialization without changing stored data."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and bool(missing):
                return ""
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Compute character LCP using binary search and C-level slice comparison."""
        if a == b:
            return len(a)
        high = min(len(a), len(b))
        low = 0
        while low < high:
            mid = (low + high + 1) // 2
            if a[:mid] == b[:mid]:
                low = mid
            else:
                high = mid - 1
        return low

    @classmethod
    def _score(cls, rows: List[str]) -> int:
        """Compute exact ideal Trie reuse via lexicographically adjacent LCPs."""
        if len(rows) < 2:
            return 0
        ordered = sorted(rows)
        return sum(cls._lcp(ordered[i - 1], ordered[i])
                   for i in range(1, len(ordered)))

    @staticmethod
    def _resolve_columns(names: List[Any], requested: List[Any]) -> List[int]:
        """Resolve requested labels to distinct positional columns deterministically."""
        result = []
        used = set()
        for wanted in requested:
            for i, name in enumerate(names):
                if i not in used and name == wanted:
                    used.add(i)
                    result.append(i)
                    break
        return result

    def _blocks(self, names: List[Any], col_merge: List[List[Any]]) -> List[List[int]]:
        """Create indivisible merge blocks while retaining all original columns."""
        result = []
        used = set()

        for group in col_merge or []:
            positions = self._resolve_columns(names, list(group))
            positions = [p for p in positions if p not in used]
            if positions:
                positions.sort()
                result.append(positions)
                used.update(positions)

        for i in range(len(names)):
            if i not in used:
                result.append([i])

        return result

    @staticmethod
    def _block_lookup(blocks: List[List[int]]) -> dict:
        """Build a column-position to block-position mapping."""
        return {column: block_id
                for block_id, block in enumerate(blocks)
                for column in block}

    def _dependency_edges(
        self,
        names: List[Any],
        blocks: List[List[int]],
        dependencies: List[Tuple[Any, Any]],
    ) -> Tuple[List[set], List[set]]:
        """Convert named dependency pairs into block predecessor/successor sets."""
        lookup = self._block_lookup(blocks)
        positions = defaultdict(list)
        for i, name in enumerate(names):
            positions[name].append(i)

        before = [set() for _ in blocks]
        after = [set() for _ in blocks]

        for pair in dependencies or []:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                continue
            left, right = pair
            if left not in positions or right not in positions:
                continue

            a = lookup[positions[left][0]]
            b = lookup[positions[right][0]]
            if a != b:
                before[b].add(a)
                after[a].add(b)

        return before, after

    @staticmethod
    def _column_scores(values: List[List[str]], mode: str) -> List[float]:
        """Rank columns from length-weighted repeated serialized values."""
        nrows = len(values)
        ncols = len(values[0]) if nrows else 0
        scores = [0.0] * ncols

        for col in range(ncols):
            counts = Counter(values[row][col] for row in range(nrows))
            score = 0.0
            for value, count in counts.items():
                if count > 1:
                    score += len(value) * count * (count - 1)

            if mode == "prefix":
                initials = Counter(
                    values[row][col][0] if values[row][col] else ""
                    for row in range(nrows)
                )
                for _, count in initials.items():
                    if count > 1:
                        score += 0.20 * count * (count - 1)

            scores[col] = score

        return scores

    def _global_order(
        self,
        values: List[List[str]],
        blocks: List[List[int]],
        names: List[Any],
        dependencies: List[Tuple[Any, Any]],
        mode: str,
    ) -> List[int]:
        """Build a dependency-respecting global order from repetition statistics."""
        scores = self._column_scores(values, mode)
        predecessors, successors = self._dependency_edges(names, blocks, dependencies)

        block_score = [
            max(scores[c] for c in block) if block else 0.0
            for block in blocks
        ]
        indegree = [len(x) for x in predecessors]
        available = [i for i, degree in enumerate(indegree) if degree == 0]
        chosen = []

        while available:
            available.sort(key=lambda b: (-block_score[b], b))
            block = available.pop(0)
            chosen.append(block)

            for child in successors[block]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    available.append(child)

        # Cyclic dependencies cannot all be satisfied; preserve a deterministic
        # full permutation rather than losing or duplicating columns.
        if len(chosen) != len(blocks):
            remaining = [b for b in range(len(blocks)) if b not in set(chosen)]
            remaining.sort(key=lambda b: (-block_score[b], b))
            chosen.extend(remaining)

        return [column for block in chosen for column in blocks[block]]

    def _conditional_orders(
        self,
        values: List[List[str]],
        blocks: List[List[int]],
        global_order: List[int],
        max_depth: int,
    ) -> List[List[int]]:
        """
        Build bounded row-specific orders by recursively choosing repeated blocks.

        Each partition chooses the remaining block with largest local
        length*count*(count-1) reuse potential, then falls back to global order.
        """
        nrows = len(values)
        if not nrows:
            return []

        lookup = self._block_lookup(blocks)
        global_blocks = []
        seen = set()
        for column in global_order:
            block = lookup[column]
            if block not in seen:
                seen.add(block)
                global_blocks.append(block)

        output = [None] * nrows
        stack = [(list(range(nrows)), [], set(), 0)]
        node_limit = 128
        nodes = 0

        while stack:
            rows, prefix, used, depth = stack.pop()
            nodes += 1

            if (len(rows) < 2 or depth >= max_depth or nodes > node_limit
                    or len(used) == len(blocks)):
                tail = [b for b in global_blocks if b not in used]
                order = [c for b in prefix + tail for c in blocks[b]]
                for row in rows:
                    output[row] = order
                continue

            candidates = [b for b in global_blocks if b not in used][:20]
            best_block = None
            best_score = 0.0

            for block in candidates:
                counts = Counter(
                    tuple(values[row][column] for column in blocks[block])
                    for row in rows
                )
                score = 0.0
                for key, count in counts.items():
                    if count > 1:
                        score += sum(map(len, key)) * count * (count - 1)
                if score > best_score:
                    best_score = score
                    best_block = block

            if best_block is None or best_score <= 0:
                tail = [b for b in global_blocks if b not in used]
                order = [c for b in prefix + tail for c in blocks[b]]
                for row in rows:
                    output[row] = order
                continue

            parts = defaultdict(list)
            for row in rows:
                key = tuple(values[row][column] for column in blocks[best_block])
                parts[key].append(row)

            if len(parts) == 1:
                tail = [b for b in global_blocks if b not in used]
                order = [c for b in prefix + tail for c in blocks[b]]
                for row in rows:
                    output[row] = order
                continue

            new_used = set(used)
            new_used.add(best_block)
            new_prefix = prefix + [best_block]

            for key in sorted(parts.keys(), reverse=True):
                stack.append((parts[key], new_prefix, new_used, depth + 1))

        fallback = global_order[:]
        return [order if order is not None else fallback for order in output]

    @staticmethod
    def _serialize(values: List[List[str]], orders: List[List[int]]) -> List[str]:
        """Serialize all rows according to their candidate per-row permutations."""
        return [
            "".join(values[row][column] for column in orders[row])
            for row in range(len(values))
        ]

    def _materialize(
        self,
        source: np.ndarray,
        source_index: pd.Index,
        source_columns: pd.Index,
        orders: List[List[int]],
        strings: List[str],
    ) -> Tuple[pd.DataFrame, List[List[Any]]]:
        """Create object-typed output containing only original source cell objects."""
        row_order = sorted(range(len(strings)), key=lambda row: (strings[row], row))
        out = np.empty(source.shape, dtype=object)
        ordering_labels = []

        for target, original_row in enumerate(row_order):
            order = orders[original_row]
            out[target, :] = source[original_row, order]
            ordering_labels.append([source_columns[column] for column in order])

        result = pd.DataFrame(
            out,
            index=source_index.take(row_order),
            columns=source_columns,
        )
        return result, ordering_labels

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
        """
        Select the best of global repetition, prefix-aware, and conditional plans.

        Candidate selection uses exact sorted-string Trie reuse, but only three
        whole-dataset candidates are scored to keep runtime bounded.
        """
        nrows, ncols = df.shape
        if nrows == 0:
            return df.copy().astype(object), []
        if ncols == 0:
            return df.copy().astype(object), [[] for _ in range(nrows)]

        source = df.to_numpy(dtype=object, copy=True)
        names = list(df.columns)
        values = [
            [self._string_value(source[row, col]) for col in range(ncols)]
            for row in range(nrows)
        ]

        blocks = self._blocks(names, col_merge)
        pair_order = self._global_order(
            values, blocks, names, one_way_dep, "pair"
        )
        prefix_order = self._global_order(
            values, blocks, names, one_way_dep, "prefix"
        )

        if row_stop is None:
            depth = min(8, len(blocks))
        else:
            depth = max(1, min(int(row_stop), 8, len(blocks)))
        if col_stop is not None:
            depth = min(depth, max(1, int(col_stop)))

        pair_orders = [pair_order[:] for _ in range(nrows)]
        prefix_orders = [prefix_order[:] for _ in range(nrows)]
        conditional_orders = self._conditional_orders(
            values, blocks, prefix_order, depth
        )

        best_orders = pair_orders
        best_strings = self._serialize(values, pair_orders)
        best_score = self._score(best_strings)

        for candidate in (prefix_orders, conditional_orders):
            strings = self._serialize(values, candidate)
            score = self._score(strings)
            if score > best_score:
                best_orders = candidate
                best_strings = strings
                best_score = score

        return self._materialize(
            source,
            df.index,
            df.columns,
            best_orders,
            best_strings,
        )

# EVOLVE-BLOCK-END