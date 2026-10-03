# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List
from collections import Counter, defaultdict


class Evolved(Algorithm):
    """Bounded column-permutation optimizer for serialized Trie prefix reuse."""

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _score_value(value) -> str:
        """Match evaluator-style missing-value normalization without changing data."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, bool) and missing:
                return ""
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        if left == right:
            return len(left)
        hi = min(len(left), len(right))
        lo = 0
        # Slice comparison is performed in optimized C code and avoids Python
        # character-by-character loops for long values.
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
        return sum(self._lcp(ordered[i - 1], ordered[i]) for i in range(1, len(ordered)))

    @staticmethod
    def _column_scores(text, nrows: int, ncols: int):
        """Length-weighted repeated-value statistics for each source field."""
        scores = [0] * ncols
        alternate = [0] * ncols
        for col in range(ncols):
            counts = Counter(text[row][col] for row in range(nrows))
            for value, count in counts.items():
                if count > 1:
                    length = len(value)
                    scores[col] += length * count * (count - 1)
                    alternate[col] += length * (count - 1)
        return scores, alternate

    @staticmethod
    def _name_positions(columns):
        result = defaultdict(list)
        for pos, name in enumerate(columns):
            result[name].append(pos)
        return result

    def _normalizer(self, columns, col_merge, one_way_dep):
        """
        Produce a stable ordering normalizer.  Merge lists are treated as
        contiguous ordering blocks; dependencies are stable precedence edges.
        """
        ncols = len(columns)
        names = self._name_positions(columns)
        used = set()
        units = []

        for merge in col_merge or []:
            positions = []
            for name in merge:
                for pos in names.get(name, []):
                    if pos not in used:
                        positions.append(pos)
                        used.add(pos)
            if positions:
                units.append(positions)

        for pos in range(ncols):
            if pos not in used:
                units.append([pos])

        unit_of = {}
        for unit_id, unit in enumerate(units):
            for pos in unit:
                unit_of[pos] = unit_id

        edges = set()
        for source, target in one_way_dep or []:
            source_positions = names.get(source, [])
            target_positions = names.get(target, [])
            # Existing API commonly supplies exact names.  Retain a conservative
            # substring fallback for its historical one-way dependency behavior.
            if not source_positions:
                source_positions = [i for i, name in enumerate(columns) if str(source) in str(name)]
            if not target_positions:
                target_positions = [i for i, name in enumerate(columns) if str(target) in str(name)]
            for left in source_positions[:1]:
                for right in target_positions[:1]:
                    if unit_of[left] != unit_of[right]:
                        edges.add((unit_of[left], unit_of[right]))

        def normalize(order):
            rank = {pos: i for i, pos in enumerate(order)}
            unit_rank = {
                unit_id: min(rank.get(pos, ncols + pos) for pos in unit)
                for unit_id, unit in enumerate(units)
            }
            outgoing = defaultdict(set)
            indegree = [0] * len(units)
            for left, right in edges:
                if right not in outgoing[left]:
                    outgoing[left].add(right)
                    indegree[right] += 1

            available = [u for u in range(len(units)) if indegree[u] == 0]
            chosen_units = []
            while available:
                available.sort(key=lambda u: (unit_rank[u], u))
                current = available.pop(0)
                chosen_units.append(current)
                for child in sorted(outgoing[current]):
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        available.append(child)

            # Cycles cannot satisfy every precedence relation.  Preserve all
            # columns deterministically rather than dropping or duplicating one.
            if len(chosen_units) != len(units):
                remaining = [u for u in range(len(units)) if u not in set(chosen_units)]
                chosen_units.extend(sorted(remaining, key=lambda u: (unit_rank[u], u)))

            result = []
            for unit_id in chosen_units:
                unit = units[unit_id]
                result.extend(sorted(unit, key=lambda pos: rank.get(pos, ncols + pos)))
            return result

        return normalize

    def _conditional_orders(self, text, global_order, scores, max_depth, max_nodes):
        """Construct a bounded conditional prefix partition tree."""
        nrows = len(text)
        ncols = len(global_order)
        orders = [[] for _ in range(nrows)]
        if nrows == 0 or ncols == 0:
            return orders

        candidate_limit = min(ncols, 32)
        useful = sorted(range(ncols), key=lambda c: (-scores[c], c))[:candidate_limit]
        nodes_seen = 0

        def visit(rows, remaining, depth):
            nonlocal nodes_seen
            nodes_seen += 1
            if (len(rows) < 2 or not remaining or depth >= max_depth or
                    nodes_seen >= max_nodes):
                tail = [c for c in global_order if c in remaining]
                for row in rows:
                    orders[row].extend(tail)
                return

            candidates = [c for c in useful if c in remaining]
            if not candidates:
                candidates = list(remaining)[:candidate_limit]

            best_col = None
            best_gain = 0
            for col in candidates:
                counts = Counter(text[row][col] for row in rows)
                gain = sum(len(value) * count * (count - 1)
                           for value, count in counts.items() if count > 1)
                if gain > best_gain or (gain == best_gain and best_col is not None and col < best_col):
                    best_gain = gain
                    best_col = col

            if best_col is None or best_gain <= 0:
                tail = [c for c in global_order if c in remaining]
                for row in rows:
                    orders[row].extend(tail)
                return

            for row in rows:
                orders[row].append(best_col)
            next_remaining = set(remaining)
            next_remaining.remove(best_col)
            groups = defaultdict(list)
            for row in rows:
                groups[text[row][best_col]].append(row)

            # Deterministic group traversal also prevents dependence on input hash order.
            for key in sorted(groups):
                visit(groups[key], next_remaining, depth + 1)

        visit(list(range(nrows)), set(range(ncols)), 0)
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
        del early_stop, distinct_value_threshold, parallel

        nrows, ncols = df.shape
        columns = list(df.columns)
        if ncols == 0:
            return df.copy(), [[] for _ in range(nrows)]
        if nrows == 0:
            return df.copy(), []

        # Read values by position so duplicate column labels and mixed dtypes are safe.
        source = [[df.iat[row, col] for col in range(ncols)] for row in range(nrows)]
        text = [[self._score_value(value) for value in row] for row in source]
        repetition, alternate = self._column_scores(text, nrows, ncols)
        normalize = self._normalizer(columns, col_merge, one_way_dep)

        global_frequency = normalize(
            sorted(range(ncols), key=lambda col: (-repetition[col], -alternate[col], col))
        )
        global_alternate = normalize(
            sorted(range(ncols), key=lambda col: (-alternate[col], -repetition[col], col))
        )

        candidates = [global_frequency, global_alternate]
        depth = min(ncols, col_stop if col_stop is not None and col_stop > 0 else 12)
        node_limit = row_stop if row_stop is not None and row_stop > 0 else 512
        conditional = self._conditional_orders(
            text, global_frequency, repetition, max_depth=depth, max_nodes=min(1024, node_limit)
        )
        conditional = [normalize(order) for order in conditional]
        candidates.append(conditional)

        best_orders = None
        best_score = -1
        for candidate in candidates:
            if candidate and isinstance(candidate[0], int):
                orders = [candidate] * nrows
            else:
                orders = candidate
            strings = ["".join(text[row][col] for col in orders[row]) for row in range(nrows)]
            score = self._trie_score(strings)
            if score > best_score:
                best_score = score
                best_orders = [list(order) for order in orders]

        # The physical dataframe is positional: output position k contains the
        # source cell named by column_orderings[row][k].  Object dtype preserves
        # mixed source values and avoids coercing missing values for scoring.
        output_values = [
            [source[row][col] for col in best_orders[row]]
            for row in range(nrows)
        ]
        result = pd.DataFrame(output_values, index=df.index, columns=columns, dtype=object)
        column_orderings = [[columns[col] for col in order] for order in best_orders]
        return result, column_orderings

# EVOLVE-BLOCK-END