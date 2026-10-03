# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from collections import defaultdict
import heapq


class Evolved(Algorithm):
    """
    Bounded prompt-prefix optimization.

    The returned dataframe always contains the original values.  For a
    row-specific ordering, dataframe columns represent output positions and
    column_orderings[row] identifies the source column placed at each position.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _score_string(value) -> str:
        """Match evaluator normalization without changing stored values."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, bool) and missing:
                return ""
        except Exception:
            pass
        try:
            return str(value)
        except Exception:
            return repr(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        if left == right:
            return len(left)
        limit = min(len(left), len(right))
        lo, hi = 0, limit
        # Slice equality is implemented in C and avoids Python character loops.
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if left[:mid] == right[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _trie_score(self, strings: List[str]) -> int:
        if len(strings) < 2:
            return 0
        ordered = sorted(strings)
        return sum(self._lcp(ordered[i - 1], ordered[i])
                   for i in range(1, len(ordered)))

    @staticmethod
    def _resolved_dependencies(columns, dependencies):
        """Resolve legacy dependency specifications deterministically."""
        position = {name: i for i, name in enumerate(columns)}
        result = []
        for before, after in dependencies or []:
            if before in position:
                left = before
            else:
                matches = [c for c in columns if str(before) in str(c)]
                left = matches[0] if len(matches) == 1 else None
            if after in position:
                right = after
            else:
                matches = [c for c in columns if str(after) in str(c)]
                right = matches[0] if len(matches) == 1 else None
            if left is not None and right is not None and left != right:
                result.append((left, right))
        return result

    def _apply_constraints(self, order, columns, col_merge, dependencies):
        """
        Keep requested merge groups contiguous and honor acyclic one-way
        dependencies.  A merge group is treated as one ordered unit.
        """
        pos = {c: i for i, c in enumerate(order)}
        used = set()
        units = []

        for group in col_merge or []:
            members = [c for c in group if c in pos and c not in used]
            if members:
                members.sort(key=lambda c: pos[c])
                units.append(members)
                used.update(members)

        for col in order:
            if col not in used:
                units.append([col])
                used.add(col)

        unit_of = {}
        for unit_id, unit in enumerate(units):
            for col in unit:
                unit_of[col] = unit_id

        edges = defaultdict(set)
        indegree = [0] * len(units)
        for left, right in dependencies:
            a, b = unit_of.get(left), unit_of.get(right)
            if a is not None and b is not None and a != b and b not in edges[a]:
                edges[a].add(b)
                indegree[b] += 1

        # Stable topological ordering; cycles retain original preference.
        priority = [min(pos[c] for c in unit) for unit in units]
        available = [(priority[i], i) for i in range(len(units)) if indegree[i] == 0]
        heapq.heapify(available)
        result_units = []
        while available:
            _, unit_id = heapq.heappop(available)
            result_units.append(unit_id)
            for child in sorted(edges[unit_id], key=lambda x: priority[x]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(available, (priority[child], child))

        if len(result_units) != len(units):
            present = set(result_units)
            result_units.extend(sorted(
                (i for i in range(len(units)) if i not in present),
                key=lambda i: priority[i],
            ))

        return [col for unit_id in result_units for col in units[unit_id]]

    def _conditional_orders(self, strings_by_col, global_order, max_columns=32,
                            max_nodes=128, max_depth=12):
        """
        Build a bounded conditional prefix partition tree.  Each leaf receives
        a cheap deterministic global tail order.
        """
        nrows = len(next(iter(strings_by_col.values()))) if strings_by_col else 0
        columns = list(strings_by_col)
        candidates = global_order[:min(max_columns, len(global_order))]
        orders = [None] * nrows
        nodes = [0]

        def repeated_weight(col, rows):
            counts = defaultdict(int)
            for row in rows:
                counts[strings_by_col[col][row]] += 1
            lengths = strings_by_col[col]
            return sum(len(value) * count * (count - 1)
                       for value, count in counts.items() if count > 1)

        def visit(rows, prefix, remaining, depth):
            if not rows:
                return
            if (len(rows) < 2 or not remaining or depth >= max_depth
                    or nodes[0] >= max_nodes):
                tail = [c for c in global_order if c not in prefix]
                final = prefix + tail
                for row in rows:
                    orders[row] = final
                return

            choices = remaining[:min(16, len(remaining))]
            best = max(choices, key=lambda c: (repeated_weight(c, rows), -global_order.index(c)))
            if repeated_weight(best, rows) <= 0:
                tail = [c for c in global_order if c not in prefix]
                final = prefix + tail
                for row in rows:
                    orders[row] = final
                return

            nodes[0] += 1
            partitions = defaultdict(list)
            for row in rows:
                partitions[strings_by_col[best][row]].append(row)

            next_remaining = [c for c in remaining if c != best]
            next_prefix = prefix + [best]
            for key in sorted(partitions):
                visit(partitions[key], next_prefix, next_remaining, depth + 1)

        visit(list(range(nrows)), [], candidates, 0)
        fallback = list(global_order)
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
        # No auxiliary index column is inserted: index and source-row identity
        # remain intact, including duplicate index labels.
        if df is None:
            return df, []

        columns = list(df.columns)
        nrows, ncols = df.shape
        if ncols == 0:
            return df.copy(), [[] for _ in range(nrows)]
        if nrows == 0:
            return df.copy(), []

        raw_values = df.to_numpy(dtype=object, copy=True)
        strings_by_col = {
            col: [self._score_string(raw_values[row, col_index])
                  for row in range(nrows)]
            for col_index, col in enumerate(columns)
        }

        # Required global frequency proposal:
        # sum(len(v) * count(v) * (count(v)-1)).
        frequency_weight = {}
        for col in columns:
            counts = defaultdict(int)
            for value in strings_by_col[col]:
                counts[value] += 1
            frequency_weight[col] = sum(
                len(value) * count * (count - 1)
                for value, count in counts.items() if count > 1
            )

        original_order = list(columns)
        global_order = sorted(
            columns,
            key=lambda c: (-frequency_weight[c], columns.index(c)),
        )
        dependencies = self._resolved_dependencies(columns, one_way_dep)

        original_order = self._apply_constraints(
            original_order, columns, col_merge, dependencies
        )
        global_order = self._apply_constraints(
            global_order, columns, col_merge, dependencies
        )

        # At most three complete constructions are scored.
        candidates = [
            [original_order[:] for _ in range(nrows)],
            [global_order[:] for _ in range(nrows)],
        ]

        conditional = self._conditional_orders(strings_by_col, global_order)
        conditional = [
            self._apply_constraints(order, columns, col_merge, dependencies)
            for order in conditional
        ]
        candidates.append(conditional)

        column_position = {col: i for i, col in enumerate(columns)}

        def candidate_strings(orders):
            return [
                "".join(strings_by_col[col][row] for col in orders[row])
                for row in range(nrows)
            ]

        best_orders = candidates[0]
        best_strings = candidate_strings(best_orders)
        best_score = self._trie_score(best_strings)

        for candidate in candidates[1:]:
            serial = candidate_strings(candidate)
            score = self._trie_score(serial)
            if score > best_score:
                best_orders = candidate
                best_strings = serial
                best_score = score

        # Lexicographic row order is deterministic and compatible with the
        # serial Trie, while original index labels remain attached to their row.
        row_order = sorted(range(nrows), key=lambda row: (best_strings[row], row))
        output = []
        reported_orders = []
        for row in row_order:
            order = best_orders[row]
            output.append([raw_values[row, column_position[col]] for col in order])
            reported_orders.append(list(order))

        # Object dtype preserves heterogeneous input values without coercion.
        result = pd.DataFrame(output, index=df.index.take(row_order),
                              columns=columns, dtype=object)
        return result, reported_orders


# EVOLVE-BLOCK-END