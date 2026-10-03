# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict


class Evolved(Algorithm):
    """
    Prefix-cache-oriented row and column reorderer.

    The returned DataFrame contains the original cells, arranged positionally
    according to the corresponding entry in column_orderings.  No source value
    is changed while computing serialization strings.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _serialized_value(value) -> str:
        """Match evaluator normalization without changing the stored value."""
        try:
            missing = pd.isna(value)
            if not hasattr(missing, "__len__") and bool(missing):
                return ""
        except Exception:
            pass
        try:
            return str(value)
        except Exception:
            return repr(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Bounded Python-level LCP using C slice comparisons."""
        limit = min(len(left), len(right))
        if limit == 0:
            return 0
        if left == right:
            return limit

        low, high = 0, limit
        while low < high:
            mid = (low + high + 1) // 2
            if left[:mid] == right[:mid]:
                low = mid
            else:
                high = mid - 1
        return low

    def _trie_score(self, strings: List[str]) -> int:
        if len(strings) < 2:
            return 0
        ordered = sorted(strings)
        return sum(self._lcp(ordered[i - 1], ordered[i])
                   for i in range(1, len(ordered)))

    @staticmethod
    def _matching_position(columns, requested):
        matches = [i for i, name in enumerate(columns) if name == requested]
        if len(matches) == 1:
            return matches[0]
        matches = [i for i, name in enumerate(columns) if str(requested) in str(name)]
        return matches[0] if len(matches) == 1 else None

    def _constraint_units(self, columns, col_merge, one_way_dep, ranks):
        """Build contiguous merge units and a deterministic dependency order."""
        count = len(columns)
        parent = list(range(count))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a

        for merge_group in col_merge or []:
            positions = [self._matching_position(columns, name) for name in merge_group]
            positions = [p for p in positions if p is not None]
            for pos in positions[1:]:
                union(positions[0], pos)

        groups = {}
        for position in range(count):
            groups.setdefault(find(position), []).append(position)

        units = list(groups.values())
        # Preserve a useful internal order in a required merged block.
        for unit in units:
            unit.sort(key=lambda p: (-ranks[p], p))

        pos_to_unit = {}
        for unit_id, unit in enumerate(units):
            for pos in unit:
                pos_to_unit[pos] = unit_id

        edges = {i: set() for i in range(len(units))}
        indegree = [0] * len(units)
        for before_name, after_name in one_way_dep or []:
            before = self._matching_position(columns, before_name)
            after = self._matching_position(columns, after_name)
            if before is None or after is None:
                continue
            source, target = pos_to_unit[before], pos_to_unit[after]
            if source != target and target not in edges[source]:
                edges[source].add(target)
                indegree[target] += 1

        unit_rank = [max(ranks[p] for p in unit) for unit in units]
        available = [u for u in range(len(units)) if indegree[u] == 0]
        ordered_units = []
        while available:
            available.sort(key=lambda u: (-unit_rank[u], min(units[u])))
            current = available.pop(0)
            ordered_units.append(current)
            for target in edges[current]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    available.append(target)

        # Cycles cannot satisfy all directed requirements; retain every column
        # deterministically rather than dropping or duplicating any cell.
        remaining = [u for u in range(len(units)) if u not in set(ordered_units)]
        remaining.sort(key=lambda u: (-unit_rank[u], min(units[u])))
        ordered_units.extend(remaining)

        return [position for unit_id in ordered_units for position in units[unit_id]]

    def _conditional_orders(self, strings_by_column, global_order, max_depth):
        """
        Build row-specific prefix orders by repeatedly partitioning on the
        length-weighted repeated value that has the most pair reuse.
        """
        row_count = len(strings_by_column[0]) if strings_by_column else 0
        col_count = len(strings_by_column)
        orders = [[] for _ in range(row_count)]
        if row_count == 0 or col_count == 0:
            return [list(global_order) for _ in range(row_count)]

        candidate_columns = list(global_order[:min(48, col_count)])
        groups = [(list(range(row_count)), candidate_columns, 0)]
        processed = 0
        node_limit = 256
        depth_limit = min(max_depth, col_count, 12)

        while groups and processed < node_limit:
            rows, remaining, depth = groups.pop()
            processed += 1
            if depth >= depth_limit or len(rows) < 2 or not remaining:
                continue

            best_col = None
            best_score = 0
            for col in remaining:
                counts = {}
                for row in rows:
                    value = strings_by_column[col][row]
                    counts[value] = counts.get(value, 0) + 1
                score = sum(len(value) * amount * (amount - 1)
                            for value, amount in counts.items() if amount > 1)
                if score > best_score or (score == best_score and
                                          best_col is not None and col < best_col):
                    best_col, best_score = col, score

            if best_col is None or best_score <= 0:
                continue

            partitions = {}
            values = strings_by_column[best_col]
            for row in rows:
                partitions.setdefault(values[row], []).append(row)

            if len(partitions) <= 1:
                continue

            next_remaining = [col for col in remaining if col != best_col]
            for partition_rows in partitions.values():
                for row in partition_rows:
                    orders[row].append(best_col)
                if len(partition_rows) > 1:
                    groups.append((partition_rows, next_remaining, depth + 1))

        global_set = set(global_order)
        for row in range(row_count):
            used = set(orders[row])
            orders[row].extend(col for col in global_order if col not in used)
            # Defensive completion if a future constraint implementation omits
            # a column from global_order.
            orders[row].extend(col for col in range(col_count)
                               if col not in global_set and col not in used)
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
        del early_stop, row_stop, distinct_value_threshold, parallel

        row_count, col_count = df.shape
        columns = list(df.columns)

        if row_count == 0:
            return df.copy(), []
        if col_count == 0:
            return df.copy(), [[] for _ in range(row_count)]

        # String and frequency information is separate from the stored cells.
        strings_by_column = []
        frequency_ranks = []
        alternate_ranks = []
        for position in range(col_count):
            values = [self._serialized_value(value) for value in df.iloc[:, position].tolist()]
            strings_by_column.append(values)
            counts = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            frequency_ranks.append(sum(len(value) * amount * (amount - 1)
                                       for value, amount in counts.items()))
            alternate_ranks.append(sum(len(value) * amount * amount
                                      for value, amount in counts.items()))

        # Required merge/dependency constraints are represented as column units.
        global_frequency = self._constraint_units(
            columns, col_merge, one_way_dep, frequency_ranks
        )
        global_alternate = self._constraint_units(
            columns, col_merge, one_way_dep, alternate_ranks
        )

        candidates = [
            [list(global_frequency) for _ in range(row_count)],
            [list(global_alternate) for _ in range(row_count)],
        ]

        # Conditional orders are only used where there are no explicit global
        # ordering constraints that could be violated by row-specific prefixes.
        if not col_merge and not one_way_dep:
            depth = col_stop if col_stop is not None else col_count
            candidates.append(self._conditional_orders(
                strings_by_column, global_frequency, max(1, depth)
            ))

        best_score = -1
        best_orders = candidates[0]
        best_strings = None
        for orders in candidates:
            serialized = [
                "".join(strings_by_column[column][row] for column in orders[row])
                for row in range(row_count)
            ]
            score = self._trie_score(serialized)
            if score > best_score:
                best_score = score
                best_orders = orders
                best_strings = serialized

        # Sorting by the final serialization is deterministic and is compatible
        # with the serial Trie objective.  Stable ties preserve source order.
        row_order = sorted(range(row_count), key=lambda row: (best_strings[row], row))
        source_values = df.astype(object).values.tolist()
        output_values = [
            [source_values[row][column] for column in best_orders[row]]
            for row in row_order
        ]
        result = pd.DataFrame(
            output_values,
            columns=columns,
            index=df.index.take(row_order),
            dtype=object,
        )
        column_orderings = [
            [columns[column] for column in best_orders[row]]
            for row in row_order
        ]

        assert result.shape == df.shape
        assert len(column_orderings) == len(result)
        assert all(len(order) == col_count for order in column_orderings)
        return result, column_orderings

# EVOLVE-BLOCK-END