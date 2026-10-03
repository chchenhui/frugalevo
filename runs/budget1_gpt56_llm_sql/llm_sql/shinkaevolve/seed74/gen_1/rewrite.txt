# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List
from collections import Counter, defaultdict
import heapq


class Evolved(Algorithm):
    """
    Safe bounded column-permutation optimizer.

    The returned DataFrame always contains the original values.  A row-specific
    ordering describes which source column supplied every positional output cell.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _score_value(value) -> str:
        """Match evaluator normalization without changing the stored value."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, bool) and missing:
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Bounded Python-level LCP using C-level slice comparisons."""
        if left == right:
            return len(left)
        high = min(len(left), len(right))
        low = 0
        while low < high:
            middle = (low + high + 1) // 2
            if left[:middle] == right[:middle]:
                low = middle
            else:
                high = middle - 1
        return low

    def _trie_score(self, rows, strings) -> int:
        serialized = ["".join(strings[row][col] for col in order) for row, order in enumerate(rows)]
        serialized.sort()
        return sum(self._lcp(serialized[i - 1], serialized[i]) for i in range(1, len(serialized)))

    @staticmethod
    def _matching_columns(columns, requested):
        exact = [i for i, name in enumerate(columns) if name == requested]
        if exact:
            return exact
        return [i for i, name in enumerate(columns) if requested in str(name)]

    def _make_units(self, columns, col_merge):
        """Create contiguous atomic units for requested column merge groups."""
        width = len(columns)
        parent = list(range(width))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a

        for group in col_merge or []:
            positions = []
            for requested in group:
                positions.extend(self._matching_columns(columns, requested))
            positions = sorted(set(positions))
            for pos in positions[1:]:
                union(positions[0], pos)

        grouped = defaultdict(list)
        for position in range(width):
            grouped[find(position)].append(position)
        return sorted(grouped.values(), key=lambda group: min(group))

    def _dependency_edges(self, columns, units, one_way_dep):
        unit_for_column = {}
        for unit_id, unit in enumerate(units):
            for col in unit:
                unit_for_column[col] = unit_id

        edges = set()
        for before, after in one_way_dep or []:
            before_positions = self._matching_columns(columns, before)
            after_positions = self._matching_columns(columns, after)
            for left in before_positions:
                for right in after_positions:
                    source = unit_for_column[left]
                    target = unit_for_column[right]
                    if source != target:
                        edges.add((source, target))
        return edges

    @staticmethod
    def _topological_order(unit_count, edges, priority):
        """Deterministic topological order; cycles fall back to priority order."""
        outgoing = [[] for _ in range(unit_count)]
        indegree = [0] * unit_count
        for source, target in edges:
            outgoing[source].append(target)
            indegree[target] += 1

        ready = [(priority[unit], unit) for unit in range(unit_count) if indegree[unit] == 0]
        heapq.heapify(ready)
        result = []
        while ready:
            _, unit = heapq.heappop(ready)
            result.append(unit)
            for target in outgoing[unit]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    heapq.heappush(ready, (priority[target], target))

        if len(result) != unit_count:
            used = set(result)
            result.extend(sorted((u for u in range(unit_count) if u not in used), key=lambda u: priority[u]))
        return result

    def _unit_members(self, units, edges):
        """Order fields inside merge units while honoring internal dependencies."""
        result = []
        for unit_id, unit in enumerate(units):
            local = list(unit)
            local_index = {column: index for index, column in enumerate(local)}
            local_edges = []
            for source_unit, target_unit in edges:
                if source_unit == unit_id and target_unit == unit_id:
                    local_edges.append((local_index[source_unit], local_index[target_unit]))
            # Unit-level edges cannot normally be internal. Preserve source order.
            result.append(local)
        return result

    def _conditional_orders(self, strings, units, unit_columns, edges, global_units, unit_scores):
        """
        Build a bounded conditional prefix partition tree.  Each node chooses a
        currently eligible field group using length-weighted pair repetition.
        """
        row_count = len(strings)
        unit_count = len(units)
        if row_count == 0:
            return []

        predecessors = [set() for _ in range(unit_count)]
        for source, target in edges:
            predecessors[target].add(source)

        global_rank = {unit: rank for rank, unit in enumerate(global_units)}
        output = [None] * row_count
        max_depth = min(unit_count, 12)
        max_nodes = 256
        nodes_seen = 0

        def finish(rows, prefix):
            remaining = [u for u in global_units if u not in prefix]
            full = prefix + remaining
            order = [column for unit in full for column in unit_columns[unit]]
            for row in rows:
                output[row] = order

        def visit(rows, prefix, remaining, depth):
            nonlocal nodes_seen
            nodes_seen += 1
            if (len(rows) < 2 or not remaining or depth >= max_depth or
                    nodes_seen > max_nodes):
                finish(rows, prefix)
                return

            prefix_set = set(prefix)
            eligible = [u for u in remaining if predecessors[u].issubset(prefix_set)]
            if not eligible:
                finish(rows, prefix)
                return

            best_unit = None
            best_score = 0
            for unit in eligible:
                score = 0
                for column in unit_columns[unit]:
                    counts = Counter(strings[row][column] for row in rows)
                    score += sum(
                        len(value) * count * (count - 1)
                        for value, count in counts.items()
                        if count > 1
                    )
                if (score > best_score or
                        (score == best_score and best_unit is not None and
                         global_rank[unit] < global_rank[best_unit])):
                    best_score = score
                    best_unit = unit

            if best_unit is None or best_score <= 0:
                finish(rows, prefix)
                return

            partitions = defaultdict(list)
            for row in rows:
                key = tuple(strings[row][column] for column in unit_columns[best_unit])
                partitions[key].append(row)

            if len(partitions) <= 1:
                finish(rows, prefix + [best_unit])
                return

            new_remaining = [u for u in remaining if u != best_unit]
            for key in sorted(partitions):
                visit(partitions[key], prefix + [best_unit], new_remaining, depth + 1)

        visit(list(range(row_count)), [], list(range(unit_count)), 0)
        fallback = [column for unit in global_units for column in unit_columns[unit]]
        return [order if order is not None else fallback for order in output]

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
        del early_stop, row_stop, col_stop, distinct_value_threshold, parallel

        row_count, column_count = df.shape
        columns = list(df.columns)
        if row_count == 0 or column_count == 0:
            return df.copy(), [[] for _ in range(row_count)]

        # Object values preserve mixed types and avoid pandas coercion.
        values = df.to_numpy(dtype=object, copy=True)
        strings = [
            [self._score_value(values[row, col]) for col in range(column_count)]
            for row in range(row_count)
        ]

        units = self._make_units(columns, col_merge)
        edges = self._dependency_edges(columns, units, one_way_dep)
        unit_columns = [list(unit) for unit in units]

        # Required global length/frequency ranking:
        # sum(len(v) * count(v) * (count(v)-1)).
        unit_scores = []
        for unit in unit_columns:
            score = 0
            for column in unit:
                counts = Counter(strings[row][column] for row in range(row_count))
                score += sum(
                    len(value) * count * (count - 1)
                    for value, count in counts.items()
                    if count > 1
                )
            unit_scores.append(score)

        natural_units = self._topological_order(
            len(units), edges, [(min(unit), min(unit)) for unit in units]
        )
        global_units = self._topological_order(
            len(units),
            edges,
            [(-unit_scores[unit], min(units[unit])) for unit in range(len(units))],
        )

        natural_order = [column for unit in natural_units for column in unit_columns[unit]]
        global_order = [column for unit in global_units for column in unit_columns[unit]]
        candidates = [
            [list(natural_order) for _ in range(row_count)],
            [list(global_order) for _ in range(row_count)],
        ]

        # The conditional candidate is intentionally bounded and is evaluated
        # exactly only once alongside the two reliable global alternatives.
        if len(units) > 1 and row_count > 1:
            candidates.append(
                self._conditional_orders(
                    strings, units, unit_columns, edges, global_units, unit_scores
                )
            )

        best_orders = candidates[0]
        best_score = self._trie_score(best_orders, strings)
        for candidate in candidates[1:]:
            score = self._trie_score(candidate, strings)
            if score > best_score:
                best_score = score
                best_orders = candidate

        serialized = [
            "".join(strings[row][column] for column in best_orders[row])
            for row in range(row_count)
        ]
        row_order = sorted(range(row_count), key=lambda row: (serialized[row], row))

        output_values = [
            [values[row, column] for column in best_orders[row]]
            for row in row_order
        ]
        reordered_df = pd.DataFrame(output_values, index=df.index.take(row_order), columns=columns, dtype=object)
        column_orderings = [[columns[column] for column in best_orders[row]] for row in row_order]

        assert reordered_df.shape == df.shape
        assert len(column_orderings) == len(reordered_df)
        return reordered_df, column_orderings


# EVOLVE-BLOCK-END