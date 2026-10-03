# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List
from collections import Counter, defaultdict


class Evolved(Algorithm):
    """
    Prefix-cache-oriented row column reordering.

    Values are never changed.  The returned dataframe stores each row in the
    order described by its corresponding column_orderings entry.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _serialized_value(value) -> str:
        """Match the evaluator's fillna('').astype(str) representation."""
        try:
            missing = pd.isna(value)
            if bool(missing):
                return ""
        except (TypeError, ValueError):
            # Non-scalar objects are not normal dataframe cells in normal use,
            # but str(value) is the safest non-destructive fallback.
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        if left == right:
            return len(left)
        upper = min(len(left), len(right))
        lower = 0
        # Slice comparisons are implemented in C and avoid Python
        # character-by-character scans for long values.
        while lower < upper:
            middle = (lower + upper + 1) // 2
            if left[:middle] == right[:middle]:
                lower = middle
            else:
                upper = middle - 1
        return lower

    def _trie_score(self, strings: List[str]) -> int:
        if len(strings) < 2:
            return 0
        ordered = sorted(strings)
        return sum(self._lcp(ordered[i - 1], ordered[i])
                   for i in range(1, len(ordered)))

    @staticmethod
    def _column_match(columns, requested):
        """Use exact names first, then retain the legacy unique-substring rule."""
        exact = [i for i, name in enumerate(columns) if name == requested]
        if exact:
            return exact[0]
        partial = [i for i, name in enumerate(columns) if str(requested) in str(name)]
        return partial[0] if len(partial) == 1 else None

    def _constraint_repair(self, desired, columns, col_merge, one_way_dep):
        """
        Convert a preferred order to a valid order respecting merge blocks and
        directed dependencies where they can be resolved unambiguously.
        """
        ncols = len(columns)
        rank = {column: i for i, column in enumerate(desired)}

        # A merge list is represented as one indivisible unit.  Overlapping
        # declarations are handled deterministically by accepting first use.
        used = set()
        units = []
        for merge in col_merge or []:
            members = []
            for name in merge:
                index = self._column_match(columns, name)
                if index is not None and index not in used:
                    members.append(index)
                    used.add(index)
            if members:
                units.append(members)
        for index in range(ncols):
            if index not in used:
                units.append([index])

        unit_for_column = {}
        for unit_index, unit in enumerate(units):
            for column in unit:
                unit_for_column[column] = unit_index

        edges = defaultdict(set)
        indegree = [0] * len(units)
        for dependency in one_way_dep or []:
            if len(dependency) != 2:
                continue
            before = self._column_match(columns, dependency[0])
            after = self._column_match(columns, dependency[1])
            if before is None or after is None:
                continue
            source = unit_for_column[before]
            target = unit_for_column[after]
            if source != target and target not in edges[source]:
                edges[source].add(target)
                indegree[target] += 1

        def unit_rank(unit_index):
            return min(rank.get(column, ncols) for column in units[unit_index])

        available = [i for i in range(len(units)) if indegree[i] == 0]
        output = []
        while available:
            current = min(available, key=unit_rank)
            available.remove(current)
            output.extend(units[current])
            for target in edges[current]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    available.append(target)

        # Cyclic dependency declarations cannot all be honored.  Preserve data
        # and resolve the remaining units deterministically.
        if len(output) != ncols:
            emitted = set(output)
            remaining = [i for i in range(len(units))
                         if not any(column in emitted for column in units[i])]
            for unit_index in sorted(remaining, key=unit_rank):
                output.extend(units[unit_index])

        return output

    def _global_orders(self, strings, columns, col_merge, one_way_dep):
        nrows = len(strings)
        ncols = len(columns)
        pair_scores = []
        simple_scores = []

        for column in range(ncols):
            counts = Counter(strings[row][column] for row in range(nrows))
            pair = sum(len(value) * count * (count - 1)
                       for value, count in counts.items())
            simple = sum(len(value) * (count - 1)
                         for value, count in counts.items())
            pair_scores.append(pair)
            simple_scores.append(simple)

        by_pairs = sorted(range(ncols), key=lambda c: (-pair_scores[c], c))
        by_simple = sorted(
            range(ncols),
            key=lambda c: (-(simple_scores[c] * 2 + pair_scores[c]), c),
        )
        return (
            self._constraint_repair(by_pairs, columns, col_merge, one_way_dep),
            self._constraint_repair(by_simple, columns, col_merge, one_way_dep),
            pair_scores,
        )

    def _conditional_orders(
        self,
        strings,
        global_order,
        pair_scores,
        columns,
        col_merge,
        one_way_dep,
        early_stop,
        row_stop,
        col_stop,
    ):
        """
        Build a bounded conditional prefix partition tree.  Frequency work is
        performed on cached strings, not pandas Series, and only the strongest
        globally reusable columns are considered at each node.
        """
        nrows = len(strings)
        ncols = len(columns)
        if nrows == 0:
            return []

        candidate_limit = min(ncols, 32)
        candidate_columns = sorted(range(ncols),
                                   key=lambda c: (-pair_scores[c], c))[:candidate_limit]
        max_depth = min(ncols, 8)
        if col_stop is not None and col_stop > 0:
            max_depth = min(max_depth, int(col_stop))
        if row_stop is not None and row_stop > 0:
            max_depth = min(max_depth, int(row_stop))

        result = [None] * nrows
        queue = [(list(range(nrows)), [], tuple(range(ncols)), 0)]
        processed_nodes = 0
        max_nodes = 256

        while queue:
            rows, prefix, remaining, depth = queue.pop()
            processed_nodes += 1
            if (depth >= max_depth or not remaining or
                    processed_nodes > max_nodes):
                for row in rows:
                    desired = prefix + [c for c in global_order if c in remaining]
                    result[row] = self._constraint_repair(
                        desired, columns, col_merge, one_way_dep
                    )
                continue

            possible = [c for c in candidate_columns if c in remaining]
            best_column = None
            best_gain = 0
            best_groups = None

            for column in possible:
                groups = defaultdict(list)
                for row in rows:
                    groups[strings[row][column]].append(row)
                gain = sum(len(value) * len(group) * (len(group) - 1)
                           for value, group in groups.items())
                if gain > best_gain or (
                    gain == best_gain and best_column is not None and column < best_column
                ):
                    best_column = column
                    best_gain = gain
                    best_groups = groups

            if best_column is None or best_gain <= max(0, early_stop):
                for row in rows:
                    desired = prefix + [c for c in global_order if c in remaining]
                    result[row] = self._constraint_repair(
                        desired, columns, col_merge, one_way_dep
                    )
                continue

            next_remaining = tuple(c for c in remaining if c != best_column)
            # Deterministic order avoids depending on dictionary insertion
            # details when output rows are later inspected.
            for value in sorted(best_groups):
                group_rows = best_groups[value]
                queue.append((group_rows, prefix + [best_column],
                              next_remaining, depth + 1))

        for row in range(nrows):
            if result[row] is None:
                result[row] = list(global_order)
        return result

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
        # parallel and distinct_value_threshold remain accepted API parameters.
        # This bounded deterministic implementation intentionally does not
        # alter data based on either parameter.
        del parallel, distinct_value_threshold

        nrows, ncols = df.shape
        columns = list(df.columns)

        if nrows == 0 or ncols == 0:
            return df.copy(), [[] for _ in range(nrows)]

        values = df.to_numpy(dtype=object, copy=True)
        strings = [
            [self._serialized_value(values[row][column])
             for column in range(ncols)]
            for row in range(nrows)
        ]

        global_pair, global_simple, pair_scores = self._global_orders(
            strings, columns, col_merge, one_way_dep
        )
        conditional = self._conditional_orders(
            strings, global_pair, pair_scores, columns, col_merge, one_way_dep,
            early_stop, row_stop, col_stop
        )

        candidates = [
            [global_pair] * nrows,
            [global_simple] * nrows,
            conditional,
        ]

        best_orders = candidates[0]
        best_score = -1
        for orders in candidates:
            serialized_rows = [
                "".join(strings[row][column] for column in orders[row])
                for row in range(nrows)
            ]
            score = self._trie_score(serialized_rows)
            if score > best_score:
                best_score = score
                best_orders = orders

        # Output position j contains the source value named by
        # column_orderings[row][j].  Object dtype prevents pandas from
        # coercing mixed values or missing values during reconstruction.
        output_rows = [
            [values[row][column] for column in best_orders[row]]
            for row in range(nrows)
        ]
        reordered = pd.DataFrame(
            output_rows, columns=columns, index=df.index.copy(), dtype=object
        )
        column_orderings = [
            [columns[column] for column in order] for order in best_orders
        ]

        assert reordered.shape == df.shape
        assert len(column_orderings) == len(reordered)
        return reordered, column_orderings


# EVOLVE-BLOCK-END