# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List
from collections import Counter, defaultdict


class Evolved(Algorithm):
    """
    Prefix-cache-oriented row-local column ordering.

    The returned dataframe keeps the original labels and row index.  Values in
    each output row are placed in the positional order described by the matching
    entry in column_orderings.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _text(value) -> str:
        """Match evaluator normalization without changing the stored value."""
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
        limit = min(len(left), len(right))
        lo, hi = 0, limit
        # Slice comparisons are implemented in C and avoid Python character loops.
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
    def _resolve_column(name, columns):
        """Resolve API dependency names with exact match preferred."""
        if name in columns:
            return columns.index(name)
        matches = [i for i, col in enumerate(columns) if str(name) in str(col)]
        return matches[0] if len(matches) == 1 else None

    def _make_units(self, columns, col_merge):
        """
        Build indivisible contiguous units.  Overlapping merge requests are
        coalesced, while unspecified columns remain individual units.
        """
        ncols = len(columns)
        parent = list(range(ncols))

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
            positions = [self._resolve_column(col, columns) for col in group]
            positions = [p for p in positions if p is not None]
            for pos in positions[1:]:
                union(positions[0], pos)

        grouped = defaultdict(list)
        for pos in range(ncols):
            grouped[find(pos)].append(pos)

        units = sorted((sorted(values) for values in grouped.values()),
                       key=lambda values: values[0])
        col_to_unit = {}
        for unit_id, unit in enumerate(units):
            for col in unit:
                col_to_unit[col] = unit_id
        return units, col_to_unit

    @staticmethod
    def _topological_units(units, predecessors, preferred):
        """Stable topological order with a deterministic cycle fallback."""
        remaining = set(range(len(units)))
        done = set()
        result = []
        rank = {unit: i for i, unit in enumerate(preferred)}

        while remaining:
            ready = [u for u in remaining if predecessors[u].issubset(done)]
            if not ready:
                # Invalid/cyclic dependency input cannot be fully satisfied.
                # Keep the ordering deterministic and preserve all columns.
                ready = list(remaining)
            ready.sort(key=lambda u: (rank.get(u, len(units) + u), units[u][0]))
            chosen = ready[0]
            result.append(chosen)
            remaining.remove(chosen)
            done.add(chosen)
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
        # Never modify df: validators may reuse it after this method returns.
        nrows, ncols = df.shape
        columns = list(df.columns)
        if nrows == 0 or ncols == 0:
            return df.copy(), [[] for _ in range(nrows)]

        source = df.to_numpy(dtype=object, copy=True)
        text = [[self._text(source[r, c]) for c in range(ncols)]
                for r in range(nrows)]

        # Factorized string codes and lengths are reused by all constructions.
        codes = [[0] * nrows for _ in range(ncols)]
        code_lengths = []
        column_scores = []
        for col in range(ncols):
            mapping = {}
            lengths = []
            next_code = 0
            for row in range(nrows):
                value = text[row][col]
                if value not in mapping:
                    mapping[value] = next_code
                    lengths.append(len(value))
                    next_code += 1
                codes[col][row] = mapping[value]
            counts = Counter(codes[col])
            column_scores.append(
                sum(lengths[code] * count * (count - 1)
                    for code, count in counts.items())
            )
            code_lengths.append(lengths)

        units, col_to_unit = self._make_units(columns, col_merge)
        nunits = len(units)
        predecessors = [set() for _ in range(nunits)]

        # Existing API semantics treat (a, b) as "a precedes b".
        for dependency in one_way_dep or []:
            if len(dependency) != 2:
                continue
            left = self._resolve_column(dependency[0], columns)
            right = self._resolve_column(dependency[1], columns)
            if left is not None and right is not None:
                a, b = col_to_unit[left], col_to_unit[right]
                if a != b:
                    predecessors[b].add(a)

        unit_scores = [sum(column_scores[col] for col in unit) for unit in units]
        global_preferred = sorted(
            range(nunits), key=lambda u: (-unit_scores[u], units[u][0])
        )
        original_preferred = list(range(nunits))
        global_units = self._topological_units(units, predecessors, global_preferred)
        original_units = self._topological_units(units, predecessors, original_preferred)

        def expand(unit_order):
            return [col for unit in unit_order for col in units[unit]]

        def complete_order(prefix_units, preference):
            selected = set(prefix_units)
            result = list(prefix_units)
            remaining = set(range(nunits)) - selected
            rank = {unit: i for i, unit in enumerate(preference)}
            while remaining:
                ready = [u for u in remaining
                         if predecessors[u].issubset(selected)]
                if not ready:
                    ready = list(remaining)
                ready.sort(key=lambda u: (rank.get(u, nunits + u), units[u][0]))
                chosen = ready[0]
                result.append(chosen)
                selected.add(chosen)
                remaining.remove(chosen)
            return expand(result)

        original_order = expand(original_units)
        global_order = expand(global_units)

        # Conditional partition candidate.  Work is deliberately bounded:
        # at most 64 tree nodes and at most 32 promising fields per node.
        conditional_orders = [None] * nrows
        candidate_units = global_preferred[:min(32, nunits)]
        max_depth = min(nunits, col_stop if col_stop is not None and col_stop > 0 else 12)
        stack = [(list(range(nrows)), [])]
        nodes = 0

        while stack:
            rows, prefix = stack.pop()
            selected = set(prefix)
            if (len(rows) < 2 or len(prefix) >= max_depth or nodes >= 64):
                order = complete_order(prefix, global_preferred)
                for row in rows:
                    conditional_orders[row] = order
                continue

            eligible = [u for u in candidate_units
                        if u not in selected and predecessors[u].issubset(selected)]
            if not eligible:
                order = complete_order(prefix, global_preferred)
                for row in rows:
                    conditional_orders[row] = order
                continue

            best_unit = None
            best_score = 0
            for unit in eligible:
                score = 0
                for col in units[unit]:
                    counts = Counter(codes[col][row] for row in rows)
                    score += sum(
                        code_lengths[col][code] * count * (count - 1)
                        for code, count in counts.items()
                    )
                if score > best_score or (
                    score == best_score and best_unit is not None and
                    units[unit][0] < units[best_unit][0]
                ):
                    best_score, best_unit = score, unit

            if best_unit is None or best_score <= 0:
                order = complete_order(prefix, global_preferred)
                for row in rows:
                    conditional_orders[row] = order
                continue

            nodes += 1
            partitions = defaultdict(list)
            for row in rows:
                key = tuple(codes[col][row] for col in units[best_unit])
                partitions[key].append(row)

            # A non-partitioning field still improves prefix placement once,
            # but recursing would not make progress.
            new_prefix = prefix + [best_unit]
            if len(partitions) <= 1:
                order = complete_order(new_prefix, global_preferred)
                for row in rows:
                    conditional_orders[row] = order
            else:
                for group in partitions.values():
                    stack.append((group, new_prefix))

        for row in range(nrows):
            if conditional_orders[row] is None:
                conditional_orders[row] = global_order

        candidates = [
            [original_order for _ in range(nrows)],
            [global_order for _ in range(nrows)],
            conditional_orders,
        ]

        best_orders = candidates[0]
        best_score = -1
        for orders in candidates:
            serialized = [
                "".join(text[row][col] for col in orders[row])
                for row in range(nrows)
            ]
            score = self._trie_score(serialized)
            if score > best_score:
                best_score = score
                best_orders = orders

        output = []
        for row in range(nrows):
            output.append([source[row, col] for col in best_orders[row]])

        # Object dtype prevents coercion of mixed source values.
        reordered = pd.DataFrame(
            output, index=df.index.copy(), columns=df.columns.copy(), dtype=object
        )
        column_orderings = [
            [columns[col] for col in best_orders[row]]
            for row in range(nrows)
        ]
        return reordered, column_orderings

# EVOLVE-BLOCK-END