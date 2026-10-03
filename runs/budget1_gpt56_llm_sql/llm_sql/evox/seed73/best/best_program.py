# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Tuple, List

from solver import Algorithm


class Evolved(Algorithm):
    """
    Select a bounded set of column-permutation candidates using the exact
    sorted-string LCP Trie objective while preserving every original cell.
    """

    def __init__(self, df: pd.DataFrame = None):
        """Store the optional source DataFrame for API compatibility."""
        self.df = df

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Return the character longest common prefix using slice comparisons."""
        limit = min(len(left), len(right))
        if limit == 0:
            return 0
        if left == right:
            return limit
        if left[0] != right[0]:
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
    def _trie_score(cls, strings: List[str]) -> int:
        """Compute ideal serial-Trie reuse from sorted adjacent row strings."""
        if len(strings) < 2:
            return 0

        ordered = sorted(strings)
        total = 0
        previous = ordered[0]

        for current in ordered[1:]:
            total += cls._lcp(previous, current)
            previous = current

        return total

    @staticmethod
    def _serialize(text_values: np.ndarray, orders: List[List[int]]) -> List[str]:
        """Serialize normalized cell strings according to each proposed order."""
        n_rows = len(orders)
        if n_rows == 0:
            return []

        first = orders[0]
        if all(order is first for order in orders):
            return ["".join(row[first]) for row in text_values]

        result = [""] * n_rows
        for row_id, order in enumerate(orders):
            result[row_id] = "".join(text_values[row_id, order])
        return result

    @staticmethod
    def _dependency_edges(columns, one_way_dep):
        """Resolve caller dependency names into integer precedence edges."""
        exact = {}
        for index, name in enumerate(columns):
            if name not in exact:
                exact[name] = index

        def resolve(value):
            if value in exact:
                return exact[value]

            matches = [
                index
                for index, name in enumerate(columns)
                if str(value) in str(name)
            ]
            return matches[0] if len(matches) == 1 else None

        edges = []
        for pair in one_way_dep or []:
            if len(pair) != 2:
                continue

            left = resolve(pair[0])
            right = resolve(pair[1])

            if left is not None and right is not None and left != right:
                edges.append((left, right))

        return edges

    @staticmethod
    def _apply_dependencies(order, edges):
        """Stably topologically repair an order without removing any columns."""
        if not edges:
            return list(order)

        position = {column: index for index, column in enumerate(order)}
        indegree = {column: 0 for column in order}
        children = defaultdict(list)

        for left, right in edges:
            if left in indegree and right in indegree:
                children[left].append(right)
                indegree[right] += 1

        ready = [column for column in order if indegree[column] == 0]
        ready.sort(key=position.get)

        result = []
        while ready:
            current = ready.pop(0)
            result.append(current)

            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)

            ready.sort(key=position.get)

        if len(result) != len(order):
            emitted = set(result)
            result.extend(column for column in order if column not in emitted)

        return result

    @staticmethod
    def _merge_blocks(order, col_merge, columns):
        """Keep requested column groups contiguous without changing any values."""
        if not col_merge:
            return list(order)

        name_to_index = {}
        for index, name in enumerate(columns):
            if name not in name_to_index:
                name_to_index[name] = index

        member_to_group = {}
        groups = []

        for group_id, group in enumerate(col_merge):
            members = []
            for name in group:
                column = name_to_index.get(name)
                if column is not None and column not in member_to_group:
                    member_to_group[column] = group_id
                    members.append(column)
            groups.append(members)

        if not any(groups):
            return list(order)

        rank = {column: index for index, column in enumerate(order)}
        emitted = set()
        result = []

        for column in order:
            group_id = member_to_group.get(column)

            if group_id is None:
                result.append(column)
            elif group_id not in emitted:
                emitted.add(group_id)
                result.extend(sorted(groups[group_id], key=rank.get))

        return result

    @staticmethod
    def _column_statistics(text_values: np.ndarray):
        """Factorize each serialized column once and score repeated values."""
        _, n_cols = text_values.shape
        codes = []
        lengths = []
        repetition_scores = np.zeros(n_cols, dtype=np.int64)

        for column in range(n_cols):
            code, uniques = pd.factorize(text_values[:, column], sort=False)
            code = np.asarray(code, dtype=np.int64)

            value_lengths = np.fromiter(
                (len(value) for value in uniques),
                dtype=np.int64,
                count=len(uniques),
            )
            counts = np.bincount(code, minlength=len(uniques)).astype(np.int64)

            repetition_scores[column] = int(
                np.sum(value_lengths * counts * (counts - 1))
            )

            codes.append(code)
            lengths.append(value_lengths)

        return codes, lengths, repetition_scores

    @staticmethod
    def _local_score(group_codes, value_lengths):
        """Compute local weighted repetition using dense or sparse counting."""
        n_values = len(value_lengths)
        group_size = len(group_codes)

        if n_values <= max(64, group_size * 3):
            counts = np.bincount(group_codes, minlength=n_values)
            return int(np.sum(value_lengths * counts * (counts - 1)))

        used_codes, counts = np.unique(group_codes, return_counts=True)
        return int(
            np.sum(
                value_lengths[used_codes]
                * counts
                * (counts - 1)
            )
        )

    def _conditional_orders(
        self,
        codes,
        lengths,
        global_order,
        n_rows,
        n_cols,
        max_depth,
        candidate_limit,
        dependency_edges,
    ):
        """Build bounded row-specific conditional prefix-partition orders."""
        candidate_columns = list(global_order[:candidate_limit])
        fallback = self._apply_dependencies(global_order, dependency_edges)

        if not candidate_columns or max_depth <= 0:
            return [fallback] * n_rows

        orders = [None] * n_rows
        terminal_cache = {}

        def terminal(prefix):
            key = tuple(prefix)
            cached = terminal_cache.get(key)
            if cached is not None:
                return cached

            if not prefix:
                result = fallback
            else:
                used = set(prefix)
                tail = [column for column in global_order if column not in used]
                result = self._apply_dependencies(prefix + tail, dependency_edges)

            terminal_cache[key] = result
            return result

        stack = [(np.arange(n_rows, dtype=np.int64), (), 0)]

        while stack:
            rows, prefix, depth = stack.pop()

            if (
                len(rows) <= 1
                or depth >= max_depth
                or len(prefix) >= n_cols
            ):
                final_order = terminal(prefix)
                for row_id in rows:
                    orders[int(row_id)] = final_order
                continue

            used = set(prefix)
            best_column = None
            best_score = 0

            for column in candidate_columns:
                if column in used:
                    continue

                score = self._local_score(
                    codes[column][rows],
                    lengths[column],
                )

                if (
                    score > best_score
                    or (
                        score == best_score
                        and best_column is not None
                        and column < best_column
                    )
                ):
                    best_column = column
                    best_score = score

            if best_column is None or best_score <= 0:
                final_order = terminal(prefix)
                for row_id in rows:
                    orders[int(row_id)] = final_order
                continue

            selected_codes = codes[best_column][rows]
            sort_index = np.argsort(selected_codes, kind="stable")
            sorted_rows = rows[sort_index]
            sorted_codes = selected_codes[sort_index]

            start = 0
            while start < len(sorted_rows):
                end = start + 1
                current_code = sorted_codes[start]

                while end < len(sorted_rows) and sorted_codes[end] == current_code:
                    end += 1

                stack.append(
                    (
                        sorted_rows[start:end],
                        prefix + (best_column,),
                        depth + 1,
                    )
                )
                start = end

        for row_id in range(n_rows):
            if orders[row_id] is None:
                orders[row_id] = fallback

        return orders

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
        """Return the best bounded serialization permutation without cell edits."""
        n_rows, n_cols = df.shape

        if n_rows == 0 or n_cols == 0:
            return df.copy(), [[] for _ in range(n_rows)]

        # Evaluator-equivalent representation used only for scoring.
        text_values = df.fillna("").astype(str).to_numpy(
            dtype=object,
            copy=True,
        )

        # Original stored values are retained separately and returned unchanged.
        raw_values = df.to_numpy(dtype=object, copy=True)
        columns = list(df.columns)

        codes, lengths, repetition_scores = self._column_statistics(text_values)
        dependency_edges = self._dependency_edges(columns, one_way_dep)

        original_order = list(range(n_cols))
        global_order = sorted(
            range(n_cols),
            key=lambda column: (-int(repetition_scores[column]), column),
        )

        original_order = self._merge_blocks(
            original_order,
            col_merge,
            columns,
        )
        global_order = self._merge_blocks(
            global_order,
            col_merge,
            columns,
        )

        original_order = self._apply_dependencies(
            original_order,
            dependency_edges,
        )
        global_order = self._apply_dependencies(
            global_order,
            dependency_edges,
        )

        original_orders = [original_order] * n_rows
        global_orders = [global_order] * n_rows

        depth_limit = min(n_cols, 8)
        if col_stop is not None and col_stop > 0:
            depth_limit = min(depth_limit, int(col_stop))

        # A conditional tree can only improve when there are multiple useful
        # repeated fields and no contiguity block would be split.
        useful_columns = int(np.count_nonzero(repetition_scores > 0))

        if col_merge or useful_columns <= 1 or depth_limit <= 0:
            conditional_orders = global_orders
        else:
            conditional_orders = self._conditional_orders(
                codes=codes,
                lengths=lengths,
                global_order=global_order,
                n_rows=n_rows,
                n_cols=n_cols,
                max_depth=depth_limit,
                candidate_limit=min(n_cols, 40),
                dependency_edges=dependency_edges,
            )

        candidates = [original_orders, global_orders]

        if not all(order is global_order for order in conditional_orders):
            candidates.append(conditional_orders)

        best_orders = original_orders
        best_score = -1

        for candidate in candidates:
            score = self._trie_score(self._serialize(text_values, candidate))
            if score > best_score:
                best_score = score
                best_orders = candidate

        first_order = best_orders[0]

        if all(order is first_order for order in best_orders):
            output = raw_values[:, first_order].copy()
        else:
            output = np.empty((n_rows, n_cols), dtype=object)
            for row_id, order in enumerate(best_orders):
                output[row_id, :] = raw_values[row_id, order]

        reordered_df = pd.DataFrame(
            output,
            index=df.index.copy(),
            columns=df.columns.copy(),
            dtype=object,
        )

        column_orderings = [
            [columns[column] for column in order]
            for order in best_orders
        ]

        return reordered_df, column_orderings


# EVOLVE-BLOCK-END