import heapq
from typing import List, Tuple

import numpy as np
import pandas as pd
from solver import Algorithm


class Evolved(Algorithm):
    """
    Prefix-cache-oriented cell permutation algorithm.

    Rows are never removed, duplicated, reordered, or modified.  Each output
    row consists only of the original row's cells, potentially in a different
    column order.  Candidate layouts are selected using exact sorted-string LCP
    reuse, equivalent to the ideal serial Trie reuse objective.
    """

    def __init__(self, df: pd.DataFrame = None):
        """Store the optional dataframe for compatibility with Algorithm."""
        self.df = df

    @staticmethod
    def _string_value(value) -> str:
        """Serialize one value exactly as evaluator scoring would serialize it."""
        if value is None:
            return ""
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and bool(missing):
                return ""
        except Exception:
            pass
        return str(value)

    def _prepare(self, df: pd.DataFrame):
        """Create original object cells and a separate evaluator-string matrix."""
        original = df.to_numpy(dtype=object, copy=True)
        rows, cols = original.shape
        strings = np.empty((rows, cols), dtype=object)

        for col in range(cols):
            strings[:, col] = [self._string_value(v) for v in original[:, col]]

        return original, strings

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Compute LCP with binary searched slice equality rather than char loops."""
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

    def _score(self, serialized_rows: List[str]) -> int:
        """Return exact ideal serial-Trie reuse via adjacent LCPs after sorting."""
        if len(serialized_rows) < 2:
            return 0

        ordered = sorted(serialized_rows)
        total = 0
        previous = ordered[0]

        for current in ordered[1:]:
            total += self._lcp(previous, current)
            previous = current

        return total

    @staticmethod
    def _serialize_fixed(strings: np.ndarray, order: List[int]) -> List[str]:
        """Serialize every row under one common column permutation."""
        rows = strings.shape[0]
        return [
            "".join(strings[row, col] for col in order)
            for row in range(rows)
        ]

    @staticmethod
    def _serialize_variable(
        strings: np.ndarray, orders: List[List[int]]
    ) -> List[str]:
        """Serialize rows under independently selected column permutations."""
        rows = strings.shape[0]
        return [
            "".join(strings[row, col] for col in orders[row])
            for row in range(rows)
        ]

    def _statistics(self, strings: np.ndarray):
        """
        Factorize each field once and calculate repetition-weighted heuristics.

        pair_score is sum(len(value) * count * (count - 1)), the required
        global ranking heuristic.  Factorized codes and value lengths are reused
        by the conditional partition construction.
        """
        rows, cols = strings.shape
        pair_scores = np.zeros(cols, dtype=np.int64)
        simple_scores = np.zeros(cols, dtype=np.int64)
        codes_by_column = []
        lengths_by_column = []

        for col in range(cols):
            mapping = {}
            codes = np.empty(rows, dtype=np.int32)
            lengths = []

            for row, value in enumerate(strings[:, col]):
                code = mapping.get(value)
                if code is None:
                    code = len(lengths)
                    mapping[value] = code
                    lengths.append(len(value))
                codes[row] = code

            code_lengths = np.asarray(lengths, dtype=np.int64)
            counts = np.bincount(codes, minlength=len(code_lengths)).astype(
                np.int64, copy=False
            )

            pair_scores[col] = int(
                np.sum(code_lengths * counts * (counts - 1))
            )
            simple_scores[col] = int(
                np.sum(code_lengths * np.maximum(counts - 1, 0))
            )

            codes_by_column.append(codes)
            lengths_by_column.append(code_lengths)

        return pair_scores, simple_scores, codes_by_column, lengths_by_column

    @staticmethod
    def _dependency_edges(columns, dependencies):
        """Resolve named dependency constraints to source-column index edges."""
        exact = {}
        for index, column in enumerate(columns):
            exact.setdefault(str(column), index)

        edges = []

        for dependency in dependencies or []:
            if len(dependency) != 2:
                continue

            before_key = str(dependency[0])
            after_key = str(dependency[1])

            before = exact.get(before_key)
            after = exact.get(after_key)

            if before is None:
                matches = [
                    i for i, column in enumerate(columns)
                    if before_key in str(column)
                ]
                before = matches[0] if len(matches) == 1 else None

            if after is None:
                matches = [
                    i for i, column in enumerate(columns)
                    if after_key in str(column)
                ]
                after = matches[0] if len(matches) == 1 else None

            if before is not None and after is not None and before != after:
                edges.append((before, after))

        return edges

    def _apply_constraints(self, base_order, columns, col_merge, one_way_dep):
        """
        Apply merge adjacency and dependency precedence to a preferred order.

        Merge groups are made contiguous.  Dependency edges are resolved with a
        stable topological order where possible.  Cycles fall back safely to the
        preferred deterministic order.
        """
        width = len(columns)
        rank = {column: position for position, column in enumerate(base_order)}

        assigned = set()
        units = []

        for group in col_merge or []:
            wanted = set(group)
            members = [
                col for col in range(width)
                if col not in assigned and columns[col] in wanted
            ]
            if members:
                members.sort(key=lambda col: rank[col])
                units.append(members)
                assigned.update(members)

        for col in range(width):
            if col not in assigned:
                units.append([col])

        unit_of = {}
        for unit_index, unit in enumerate(units):
            for col in unit:
                unit_of[col] = unit_index

        edges = self._dependency_edges(columns, one_way_dep)

        for unit in units:
            local_position = {col: pos for pos, col in enumerate(unit)}

            for before, after in edges:
                if before in local_position and after in local_position:
                    if local_position[before] > local_position[after]:
                        unit.remove(before)
                        unit.insert(unit.index(after), before)
                        local_position = {
                            col: pos for pos, col in enumerate(unit)
                        }

        outgoing = [[] for _ in units]
        indegree = [0] * len(units)
        seen = set()

        for before, after in edges:
            source = unit_of[before]
            target = unit_of[after]

            if source != target and (source, target) not in seen:
                seen.add((source, target))
                outgoing[source].append(target)
                indegree[target] += 1

        unit_rank = [min(rank[col] for col in unit) for unit in units]
        queue = [
            (unit_rank[unit], unit)
            for unit in range(len(units))
            if indegree[unit] == 0
        ]
        heapq.heapify(queue)

        ordered_units = []

        while queue:
            _, unit = heapq.heappop(queue)
            ordered_units.append(unit)

            for child in outgoing[unit]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(queue, (unit_rank[child], child))

        if len(ordered_units) != len(units):
            ordered_units = sorted(
                range(len(units)),
                key=lambda unit: unit_rank[unit],
            )

        result = []
        for unit in ordered_units:
            result.extend(units[unit])

        return result

    @staticmethod
    def _local_pair_score(local_codes, code_lengths) -> int:
        """
        Compute local length-weighted repetition without global-cardinality bins.

        Using unique local codes prevents repeated bincount allocations sized to
        the entire column cardinality for small recursive partitions.
        """
        unique_codes, counts = np.unique(local_codes, return_counts=True)

        repeated = counts > 1
        if not np.any(repeated):
            return 0

        repeated_codes = unique_codes[repeated]
        repeated_counts = counts[repeated].astype(np.int64, copy=False)

        return int(
            np.sum(
                code_lengths[repeated_codes]
                * repeated_counts
                * (repeated_counts - 1)
            )
        )

    def _conditional_orders(
        self,
        rows,
        global_order,
        codes,
        code_lengths,
        candidate_columns,
        depth_limit,
        early_stop,
        columns,
        col_merge,
        one_way_dep,
    ):
        """
        Build a bounded conditional prefix partition tree.

        Every node selects the remaining field with best local length-weighted
        repetition, partitions rows by its serialized value, and allows child
        partitions to select different suffix prefixes.
        """
        prefixes = [[] for _ in range(rows)]

        if rows < 2 or depth_limit <= 0:
            return [list(global_order) for _ in range(rows)]

        stack = [(np.arange(rows, dtype=np.int32), frozenset(), 0)]

        while stack:
            group_rows, used, depth = stack.pop()

            if len(group_rows) < 2 or depth >= depth_limit:
                continue

            best_column = None
            best_score = 0

            for col in candidate_columns:
                if col in used:
                    continue

                score = self._local_pair_score(
                    codes[col][group_rows],
                    code_lengths[col],
                )

                if score > best_score or (
                    score == best_score
                    and score > 0
                    and (best_column is None or col < best_column)
                ):
                    best_column = col
                    best_score = score

            if best_column is None or best_score <= early_stop:
                continue

            for row in group_rows:
                prefixes[int(row)].append(best_column)

            local_codes = codes[best_column][group_rows]
            permutation = np.argsort(local_codes, kind="stable")
            sorted_rows = group_rows[permutation]
            sorted_codes = local_codes[permutation]

            boundaries = np.flatnonzero(
                sorted_codes[1:] != sorted_codes[:-1]
            ) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [len(sorted_rows)]))
            next_used = used | {best_column}

            for start, end in zip(starts, ends):
                if end - start >= 2:
                    stack.append(
                        (sorted_rows[start:end], next_used, depth + 1)
                    )

        constrained = bool(col_merge) or bool(one_way_dep)
        cache = {}
        result = []

        for prefix in prefixes:
            prefix_key = tuple(prefix)
            prefix_set = set(prefix)

            if not constrained:
                result.append(
                    prefix + [
                        col for col in global_order
                        if col not in prefix_set
                    ]
                )
                continue

            order = cache.get(prefix_key)

            if order is None:
                preferred = prefix + [
                    col for col in global_order
                    if col not in prefix_set
                ]
                order = self._apply_constraints(
                    preferred,
                    columns,
                    col_merge,
                    one_way_dep,
                )
                cache[prefix_key] = order

            result.append(list(order))

        return result

    @staticmethod
    def _materialize_fixed(original, index, columns, order):
        """Materialize a common source-column permutation using object dtype."""
        output = original[:, order].copy()
        return pd.DataFrame(
            output,
            index=index.copy(),
            columns=columns.copy(),
            dtype=object,
        )

    @staticmethod
    def _materialize_variable(original, index, columns, orders):
        """Materialize row-specific source-column permutations using object dtype."""
        rows, cols = original.shape
        output = np.empty((rows, cols), dtype=object)

        for row, order in enumerate(orders):
            output[row, :] = original[row, order]

        return pd.DataFrame(
            output,
            index=index.copy(),
            columns=columns.copy(),
            dtype=object,
        )

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
        Select the best of source, global repetition, and conditional layouts.

        Candidate evaluation uses complete serialized rows and exact sorted-LCP
        Trie reuse.  The returned dataframe preserves shape, index, cell values,
        row identity, and source-cell multiplicity.
        """
        del distinct_value_threshold, parallel

        rows, cols = df.shape
        columns = list(df.columns)

        if rows == 0 or cols == 0:
            return df.copy(), [[] for _ in range(rows)]

        original, strings = self._prepare(df)
        pair_scores, simple_scores, codes, code_lengths = self._statistics(
            strings
        )

        heuristic_order = sorted(
            range(cols),
            key=lambda col: (
                -int(pair_scores[col]),
                -int(simple_scores[col]),
                col,
            ),
        )

        source_order = self._apply_constraints(
            list(range(cols)),
            columns,
            col_merge,
            one_way_dep,
        )
        global_order = self._apply_constraints(
            heuristic_order,
            columns,
            col_merge,
            one_way_dep,
        )

        source_score = self._score(
            self._serialize_fixed(strings, source_order)
        )
        global_score = self._score(
            self._serialize_fixed(strings, global_order)
        )

        best_kind = "source"
        best_score = source_score
        best_order = source_order
        best_orders = None

        if global_score > best_score:
            best_kind = "global"
            best_score = global_score
            best_order = global_order

        depth_limit = min(cols, 8)

        if col_stop is not None and col_stop > 0:
            depth_limit = min(depth_limit, int(col_stop))

        if row_stop is not None and row_stop > 0:
            depth_limit = min(depth_limit, int(row_stop))

        if rows >= 2 and depth_limit > 0:
            tree_columns = heuristic_order[:min(cols, 24)]

            conditional_orders = self._conditional_orders(
                rows=rows,
                global_order=global_order,
                codes=codes,
                code_lengths=code_lengths,
                candidate_columns=tree_columns,
                depth_limit=depth_limit,
                early_stop=max(0, int(early_stop)),
                columns=columns,
                col_merge=col_merge,
                one_way_dep=one_way_dep,
            )

            conditional_score = self._score(
                self._serialize_variable(strings, conditional_orders)
            )

            if conditional_score > best_score:
                best_kind = "conditional"
                best_score = conditional_score
                best_orders = conditional_orders

        if best_kind == "conditional":
            reordered = self._materialize_variable(
                original,
                df.index,
                columns,
                best_orders,
            )
            returned_orders = best_orders
        else:
            reordered = self._materialize_fixed(
                original,
                df.index,
                columns,
                best_order,
            )
            returned_orders = [list(best_order) for _ in range(rows)]

        column_orderings = [
            [columns[col] for col in order]
            for order in returned_orders
        ]

        return reordered, column_orderings