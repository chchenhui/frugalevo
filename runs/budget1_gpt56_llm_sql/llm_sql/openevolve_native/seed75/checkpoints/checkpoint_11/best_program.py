# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Any, List, Tuple

from solver import Algorithm


class Evolved(Algorithm):
    """
    Build a small set of safe row/column permutation candidates and select the
    one with the greatest exact serialized-row Trie reuse.

    Stored values are never normalized or changed.  String conversion is used
    only for optimization and candidate scoring.
    """

    def __init__(self, df: pd.DataFrame = None):
        """Store the optional dataframe for compatibility with Algorithm."""
        self.df = df

    @staticmethod
    def _string_value(value: Any) -> str:
        """Return evaluator-compatible text without changing the stored object."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and bool(missing):
                return ""
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Return exact character LCP using binary searched slice comparisons."""
        if left == right:
            return len(left)

        low = 0
        high = min(len(left), len(right))
        while low < high:
            middle = (low + high + 1) // 2
            if left[:middle] == right[:middle]:
                low = middle
            else:
                high = middle - 1
        return low

    @classmethod
    def _score(cls, rows: List[str]) -> int:
        """Compute exact ideal serial-Trie reuse from lexicographically sorted rows."""
        if len(rows) < 2:
            return 0

        ordered = sorted(rows)
        total = 0
        previous = ordered[0]
        for current in ordered[1:]:
            total += cls._lcp(previous, current)
            previous = current
        return total

    @staticmethod
    def _resolve_columns(names: List[Any], requested: List[Any]) -> List[int]:
        """Resolve labels to distinct source positions in deterministic order."""
        result = []
        used = set()

        for wanted in requested:
            for position, name in enumerate(names):
                if position not in used and name == wanted:
                    used.add(position)
                    result.append(position)
                    break
        return result

    def _blocks(
        self,
        names: List[Any],
        col_merge: List[List[Any]],
    ) -> List[List[int]]:
        """Create indivisible merge blocks plus singleton blocks for other columns."""
        blocks = []
        used = set()

        for group in col_merge or []:
            positions = self._resolve_columns(names, list(group))
            positions = [position for position in positions if position not in used]
            if positions:
                # Existing API semantics treat a merge as an indivisible block.
                positions.sort()
                blocks.append(positions)
                used.update(positions)

        for position in range(len(names)):
            if position not in used:
                blocks.append([position])

        return blocks

    @staticmethod
    def _block_lookup(blocks: List[List[int]]) -> dict:
        """Map each source column position to its enclosing block id."""
        return {
            column: block_id
            for block_id, block in enumerate(blocks)
            for column in block
        }

    def _dependency_edges(
        self,
        names: List[Any],
        blocks: List[List[int]],
        dependencies: List[Tuple[Any, Any]],
    ) -> Tuple[List[set], List[set]]:
        """Translate named one-way dependencies into block predecessor relations."""
        lookup = self._block_lookup(blocks)
        positions = defaultdict(list)

        for position, name in enumerate(names):
            positions[name].append(position)

        before = [set() for _ in blocks]
        after = [set() for _ in blocks]

        for pair in dependencies or []:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                continue

            left, right = pair
            if left not in positions or right not in positions:
                continue

            left_block = lookup[positions[left][0]]
            right_block = lookup[positions[right][0]]

            if left_block != right_block:
                before[right_block].add(left_block)
                after[left_block].add(right_block)

        return before, after

    @staticmethod
    def _column_scores(
        values: List[List[str]],
        mode: str,
    ) -> List[float]:
        """
        Rank fields by exact repeated text and, for prefix mode, repeated starts.

        Pair mode uses the required len(value)*count*(count-1) statistic.
        Prefix mode adds a cheap first-character grouping signal rather than
        expensive field-level LCP sorting.
        """
        nrows = len(values)
        ncols = len(values[0]) if nrows else 0
        scores = [0.0] * ncols

        for column in range(ncols):
            counts = Counter(values[row][column] for row in range(nrows))
            score = 0.0

            for value, count in counts.items():
                if count > 1:
                    score += len(value) * count * (count - 1)

            if mode == "prefix":
                initials = Counter(
                    values[row][column][0] if values[row][column] else ""
                    for row in range(nrows)
                )
                for count in initials.values():
                    if count > 1:
                        score += 0.20 * count * (count - 1)

            scores[column] = score

        return scores

    def _global_order(
        self,
        values: List[List[str]],
        blocks: List[List[int]],
        names: List[Any],
        dependencies: List[Tuple[Any, Any]],
        mode: str,
    ) -> List[int]:
        """Build a dependency-respecting global field order from column scores."""
        column_scores = self._column_scores(values, mode)
        predecessors, successors = self._dependency_edges(
            names, blocks, dependencies
        )

        # Max preserves the strongest useful field signal for merge blocks.
        block_scores = [
            max(column_scores[column] for column in block) if block else 0.0
            for block in blocks
        ]

        indegree = [len(items) for items in predecessors]
        available = [
            block_id
            for block_id, degree in enumerate(indegree)
            if degree == 0
        ]
        chosen = []

        while available:
            available.sort(key=lambda block_id: (-block_scores[block_id], block_id))
            block_id = available.pop(0)
            chosen.append(block_id)

            for child in successors[block_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    available.append(child)

        # Preserve all blocks even when supplied dependencies contain a cycle.
        if len(chosen) != len(blocks):
            chosen_set = set(chosen)
            remaining = [
                block_id
                for block_id in range(len(blocks))
                if block_id not in chosen_set
            ]
            remaining.sort(key=lambda block_id: (-block_scores[block_id], block_id))
            chosen.extend(remaining)

        return [
            column
            for block_id in chosen
            for column in blocks[block_id]
        ]

    def _conditional_orders(
        self,
        values: List[List[str]],
        blocks: List[List[int]],
        global_order: List[int],
        max_depth: int,
    ) -> List[List[int]]:
        """
        Construct bounded conditional prefix plans using local repeat savings.

        Each node selects a remaining block with the greatest local
        length*count*(count-1) repetition opportunity, partitions rows by that
        block's text, and gives child groups independent suffix layouts.
        """
        nrows = len(values)
        if not nrows:
            return []

        lookup = self._block_lookup(blocks)
        global_blocks = []
        seen = set()

        for column in global_order:
            block_id = lookup[column]
            if block_id not in seen:
                seen.add(block_id)
                global_blocks.append(block_id)

        output = [None] * nrows
        stack = [(list(range(nrows)), [], frozenset(), 0)]

        node_limit = 128
        candidate_limit = 20
        nodes = 0

        while stack:
            rows, prefix, used, depth = stack.pop()
            nodes += 1

            if (
                len(rows) < 2
                or depth >= max_depth
                or nodes > node_limit
                or len(used) == len(blocks)
            ):
                tail = [block for block in global_blocks if block not in used]
                order = [
                    column
                    for block in prefix + tail
                    for column in blocks[block]
                ]
                for row in rows:
                    output[row] = order
                continue

            candidates = [
                block
                for block in global_blocks
                if block not in used
            ][:candidate_limit]

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
                        score += (
                            sum(len(piece) for piece in key)
                            * count
                            * (count - 1)
                        )

                if score > best_score:
                    best_score = score
                    best_block = block

            if best_block is None or best_score <= 0:
                tail = [block for block in global_blocks if block not in used]
                order = [
                    column
                    for block in prefix + tail
                    for column in blocks[block]
                ]
                for row in rows:
                    output[row] = order
                continue

            partitions = defaultdict(list)
            for row in rows:
                key = tuple(values[row][column] for column in blocks[best_block])
                partitions[key].append(row)

            if len(partitions) <= 1:
                tail = [block for block in global_blocks if block not in used]
                order = [
                    column
                    for block in prefix + tail
                    for column in blocks[block]
                ]
                for row in rows:
                    output[row] = order
                continue

            next_used = frozenset(set(used) | {best_block})
            next_prefix = prefix + [best_block]

            for key in sorted(partitions.keys(), reverse=True):
                stack.append((
                    partitions[key],
                    next_prefix,
                    next_used,
                    depth + 1,
                ))

        fallback = global_order[:]
        return [
            order if order is not None else fallback
            for order in output
        ]

    @staticmethod
    def _serialize(
        values: List[List[str]],
        orders: List[List[int]],
    ) -> List[str]:
        """Serialize each source row according to its candidate field permutation."""
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
        """
        Sort rows by final serialization and materialize only original objects.

        The returned ordering list describes the exact source-column sequence
        used to construct each corresponding returned dataframe row.
        """
        row_order = sorted(
            range(len(strings)),
            key=lambda row: (strings[row], row),
        )

        output = np.empty(source.shape, dtype=object)
        ordering_labels = []

        for target_row, source_row in enumerate(row_order):
            order = orders[source_row]
            output[target_row, :] = source[source_row, order]
            ordering_labels.append([
                source_columns[column]
                for column in order
            ])

        result = pd.DataFrame(
            output,
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
        Select among pair-ranked, prefix-ranked, and conditional layouts.

        Only three complete candidates are serialized and scored with the exact
        sorted-string LCP objective.  All data values, row identities, and
        dataframe shape are preserved.
        """
        nrows, ncols = df.shape

        if nrows == 0:
            return df.copy().astype(object), []

        if ncols == 0:
            return df.copy().astype(object), [[] for _ in range(nrows)]

        source = df.to_numpy(dtype=object, copy=True)
        names = list(df.columns)

        values = [
            [
                self._string_value(source[row, column])
                for column in range(ncols)
            ]
            for row in range(nrows)
        ]

        blocks = self._blocks(names, col_merge)

        pair_order = self._global_order(
            values,
            blocks,
            names,
            one_way_dep,
            "pair",
        )
        prefix_order = self._global_order(
            values,
            blocks,
            names,
            one_way_dep,
            "prefix",
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
            values,
            blocks,
            prefix_order,
            depth,
        )

        best_orders = pair_orders
        best_strings = self._serialize(values, pair_orders)
        best_score = self._score(best_strings)

        for candidate_orders in (prefix_orders, conditional_orders):
            strings = self._serialize(values, candidate_orders)
            score = self._score(strings)

            if score > best_score:
                best_orders = candidate_orders
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