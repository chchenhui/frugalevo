import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List


class Evolved(Algorithm):
    """Build a few bounded field-permutation candidates and choose the layout with
    the greatest exact serialized character-Trie reuse."""

    def __init__(self, df: pd.DataFrame = None):
        """Store the optional dataframe for compatibility with the base class."""
        self.df = df

    @staticmethod
    def _string_matrix(df: pd.DataFrame) -> np.ndarray:
        """Create evaluator-normalized planning strings without altering source cells."""
        safe = df.astype(object).where(pd.notna(df), "")
        return safe.astype(str).to_numpy(dtype=object, copy=False)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Return the character LCP using binary searched slice comparisons."""
        limit = min(len(left), len(right))
        if limit == 0:
            return 0
        if left == right:
            return limit

        low, high = 0, limit
        while low < high:
            middle = (low + high + 1) // 2
            if left[:middle] == right[:middle]:
                low = middle
            else:
                high = middle - 1
        return low

    def _column_statistics(self, strings: np.ndarray):
        """Factorize columns once and compute repetition and bounded prefix scores."""
        _, n_cols = strings.shape
        pair = np.zeros(n_cols, dtype=np.int64)
        light = np.zeros(n_cols, dtype=np.int64)
        average_length = np.zeros(n_cols, dtype=np.float64)
        codes = []
        lengths = []
        uniques = []
        counts_by_col = []

        for col in range(n_cols):
            code, unique = pd.factorize(strings[:, col], sort=False)
            code = np.asarray(code, dtype=np.int64)
            unique = np.asarray(unique, dtype=object)
            value_lengths = np.asarray([len(value) for value in unique], dtype=np.int64)
            counts = np.bincount(code, minlength=len(unique)).astype(np.int64)

            pair[col] = int(np.sum(value_lengths * counts * (counts - 1)))
            light[col] = int(np.sum(value_lengths * np.maximum(counts - 1, 0)))

            if len(counts):
                average_length[col] = float(np.sum(value_lengths * counts)) / max(1, len(code))

            codes.append(code)
            lengths.append(value_lengths)
            uniques.append(unique)
            counts_by_col.append(counts)

        # The second global layout receives an actual field-level Trie estimate for
        # a bounded deterministic set of promising columns.  This adds sharing from
        # common prefixes among distinct values, not merely exact equal values.
        prefix = light.copy()
        ranked_pair = sorted(range(n_cols), key=lambda c: (-int(pair[c]), -int(light[c]), c))
        ranked_length = sorted(range(n_cols), key=lambda c: (-average_length[c], c))
        selected = set(ranked_pair[: min(36, n_cols)])
        selected.update(ranked_length[: min(12, n_cols)])

        for col in selected:
            unique = uniques[col]
            counts = counts_by_col[col]
            value_lengths = lengths[col]

            # Avoid expensive sorting for very high-cardinality textual columns.
            if len(unique) < 2 or len(unique) > 12000:
                continue

            order = sorted(range(len(unique)), key=lambda i: unique[i])
            score = int(np.sum(value_lengths * np.maximum(counts - 1, 0)))

            previous = unique[order[0]]
            for position in order[1:]:
                current = unique[position]
                score += self._lcp(previous, current)
                previous = current

            prefix[col] = score

        return pair, light, prefix, codes, lengths

    def _serial_score(self, strings: np.ndarray, orders) -> int:
        """Compute exact ideal Trie reuse from sorted complete serialized rows."""
        n_rows = strings.shape[0]
        if n_rows < 2:
            return 0

        if isinstance(orders, tuple):
            arranged = strings[:, list(orders)]
            serial = ["".join(row) for row in arranged]
        else:
            serial = [
                "".join(strings[row, list(order)])
                for row, order in enumerate(orders)
            ]

        serial.sort()
        total = 0
        previous = serial[0]

        for current in serial[1:]:
            total += self._lcp(previous, current)
            previous = current

        return total

    @staticmethod
    def _constraint_order(
        base_order: List[int],
        columns: List,
        col_merge: List[List[str]],
        one_way_dep: List[Tuple[str, str]],
    ) -> List[int]:
        """Apply merge-contiguity and dependency precedence to a proposed order."""
        n_cols = len(columns)
        if n_cols < 2 or (not col_merge and not one_way_dep):
            return list(base_order)

        positions_by_name = {}
        for position, name in enumerate(columns):
            positions_by_name.setdefault(name, []).append(position)

        rank = {position: index for index, position in enumerate(base_order)}
        parent = list(range(n_cols))

        def find(item):
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        def union(left, right):
            left = find(left)
            right = find(right)
            if left != right:
                parent[right] = left

        for group in col_merge or []:
            members = []
            for name in group:
                members.extend(positions_by_name.get(name, []))
            if len(members) > 1:
                anchor = members[0]
                for member in members[1:]:
                    union(anchor, member)

        components = {}
        for position in range(n_cols):
            components.setdefault(find(position), []).append(position)

        blocks = list(components.values())
        for block in blocks:
            block.sort(key=lambda position: rank[position])

        block_of = {}
        block_rank = {}
        for block_id, block in enumerate(blocks):
            block_rank[block_id] = min(rank[position] for position in block)
            for position in block:
                block_of[position] = block_id

        outgoing = [[] for _ in blocks]
        indegree = [0] * len(blocks)
        edges = set()

        for source_name, target_name in one_way_dep or []:
            for source in positions_by_name.get(source_name, []):
                for target in positions_by_name.get(target_name, []):
                    left = block_of[source]
                    right = block_of[target]
                    if left != right:
                        edges.add((left, right))

        for left, right in edges:
            outgoing[left].append(right)
            indegree[right] += 1

        ready = sorted(
            [block for block in range(len(blocks)) if indegree[block] == 0],
            key=lambda block: block_rank[block],
        )

        selected = []
        while ready:
            block = ready.pop(0)
            selected.append(block)

            for target in outgoing[block]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort(key=lambda item: block_rank[item])

        if len(selected) != len(blocks):
            used = set(selected)
            selected.extend(
                block
                for block in sorted(range(len(blocks)), key=lambda b: block_rank[b])
                if block not in used
            )

        return [position for block in selected for position in blocks[block]]

    @staticmethod
    def _conditional_orders(
        codes,
        lengths,
        default_order: Tuple[int, ...],
        n_rows: int,
        max_depth: int,
    ) -> List[Tuple[int, ...]]:
        """Construct a bounded conditional repeated-value prefix partition tree."""
        n_cols = len(default_order)
        candidate_cols = tuple(default_order[: min(32, n_cols)])
        orders = [None] * n_rows
        stack = [(np.arange(n_rows, dtype=np.int64), tuple(), candidate_cols, 0)]
        max_nodes = 320
        nodes = 0

        while stack:
            rows, prefix, remaining, depth = stack.pop()
            nodes += 1

            if (
                len(rows) < 2
                or not remaining
                or depth >= max_depth
                or nodes > max_nodes
            ):
                tail = tuple(col for col in default_order if col not in prefix)
                final = prefix + tail
                for row in rows:
                    orders[int(row)] = final
                continue

            best_col = None
            best_score = 0

            for col in remaining:
                selected_codes = codes[col][rows]
                used, counts = np.unique(selected_codes, return_counts=True)
                counts = counts.astype(np.int64)
                score = int(np.sum(lengths[col][used] * counts * (counts - 1)))

                if (
                    score > best_score
                    or (score == best_score and best_col is not None and col < best_col)
                ):
                    best_col = col
                    best_score = score

            if best_col is None or best_score <= 0:
                tail = tuple(col for col in default_order if col not in prefix)
                final = prefix + tail
                for row in rows:
                    orders[int(row)] = final
                continue

            selected_codes = codes[best_col][rows]
            sorter = np.argsort(selected_codes, kind="stable")
            sorted_rows = rows[sorter]
            sorted_codes = selected_codes[sorter]
            next_remaining = tuple(col for col in remaining if col != best_col)

            start = 0
            while start < len(sorted_rows):
                end = start + 1
                while end < len(sorted_rows) and sorted_codes[end] == sorted_codes[start]:
                    end += 1

                stack.append(
                    (
                        sorted_rows[start:end],
                        prefix + (best_col,),
                        next_remaining,
                        depth + 1,
                    )
                )
                start = end

        fallback = tuple(default_order)
        return [order if order is not None else fallback for order in orders]

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
        """Select the best exact-score layout from two globals and one conditional tree."""
        n_rows, n_cols = df.shape
        columns = list(df.columns)

        if n_rows == 0 or n_cols == 0:
            return df.copy(), [[] for _ in range(n_rows)]

        strings = self._string_matrix(df)
        pair, light, prefix, codes, lengths = self._column_statistics(strings)
        positions = list(range(n_cols))

        frequency_order = sorted(
            positions,
            key=lambda col: (-int(pair[col]), -int(light[col]), col),
        )

        prefix_order = sorted(
            positions,
            key=lambda col: (-int(prefix[col]), -int(pair[col]), col),
        )

        frequency_order = self._constraint_order(
            frequency_order, columns, col_merge, one_way_dep
        )
        prefix_order = self._constraint_order(
            prefix_order, columns, col_merge, one_way_dep
        )

        candidates = [tuple(frequency_order)]
        if tuple(prefix_order) != tuple(frequency_order):
            candidates.append(tuple(prefix_order))

        if (
            int(np.max(pair)) > 0
            and not col_merge
            and not one_way_dep
            and n_rows * n_cols <= 1_250_000
            and n_cols <= 128
        ):
            requested_depth = col_stop if col_stop is not None and col_stop > 0 else 8
            depth_limit = min(8, n_cols, requested_depth)

            candidates.append(
                self._conditional_orders(
                    codes,
                    lengths,
                    tuple(frequency_order),
                    n_rows,
                    depth_limit,
                )
            )

        total_chars = sum(len(value) for value in strings.ravel())
        if total_chars <= 30_000_000:
            best = max(
                candidates,
                key=lambda candidate: self._serial_score(strings, candidate),
            )
        else:
            best = candidates[0]

        source = df.to_numpy(dtype=object, copy=True)

        if isinstance(best, tuple):
            output = source[:, list(best)]
            row_orders = [best] * n_rows
        else:
            output = np.empty(source.shape, dtype=object)
            for row, order in enumerate(best):
                output[row, :] = source[row, list(order)]
            row_orders = best

        reordered = pd.DataFrame(
            output,
            index=df.index.copy(),
            columns=df.columns.copy(),
            dtype=object,
        )

        column_orderings = [
            [columns[position] for position in order]
            for order in row_orders
        ]

        return reordered, column_orderings