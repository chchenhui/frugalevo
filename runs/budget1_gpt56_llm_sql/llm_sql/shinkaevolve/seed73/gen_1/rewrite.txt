# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict, Any
from collections import Counter, defaultdict
import heapq


class Evolved(Algorithm):
    """
    Prefix-cache aware row-wise column reordering.

    Values are never changed or fabricated.  For a row-specific ordering, the
    returned dataframe stores that row's values in positional output slots and
    column_orderings identifies the source column for every output slot.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _text(value: Any) -> str:
        """Match the evaluator's fillna('').astype(str) representation."""
        if value is None:
            return ""
        try:
            missing = pd.isna(value)
            if isinstance(missing, bool) and missing:
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        if left == right:
            return len(left)
        limit = min(len(left), len(right))
        lo, hi = 0, limit
        # Slice comparisons execute the character comparison in C and avoid a
        # Python loop over every character in long common prefixes.
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if left[:mid] == right[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _trie_score(self, rows: List[str]) -> int:
        if len(rows) < 2:
            return 0
        ordered = sorted(rows)
        return sum(self._lcp(ordered[i - 1], ordered[i])
                   for i in range(1, len(ordered)))

    @staticmethod
    def _resolve_column(columns: List[Any], requested: Any) -> Any:
        if requested in columns:
            return requested
        matches = [column for column in columns if str(requested) in str(column)]
        return matches[0] if len(matches) == 1 else None

    def _units(
        self,
        columns: List[Any],
        col_merge: List[List[str]],
        one_way_dep: List[Tuple[str, str]],
    ) -> Tuple[List[List[Any]], Dict[int, set]]:
        """
        Make merge groups indivisible.  A merge constraint is honored as
        adjacency while retaining original order inside the requested group.
        """
        used = set()
        units: List[List[Any]] = []

        for requested_group in col_merge or []:
            group = []
            for requested in requested_group:
                column = self._resolve_column(columns, requested)
                if column is not None and column not in used:
                    group.append(column)
                    used.add(column)
            if group:
                units.append(group)

        for column in columns:
            if column not in used:
                units.append([column])
                used.add(column)

        unit_of = {}
        for index, unit in enumerate(units):
            for column in unit:
                unit_of[column] = index

        predecessors = {i: set() for i in range(len(units))}
        for dependency in one_way_dep or []:
            if len(dependency) != 2:
                continue
            before = self._resolve_column(columns, dependency[0])
            after = self._resolve_column(columns, dependency[1])
            if before is None or after is None:
                continue
            source, target = unit_of[before], unit_of[after]
            if source != target:
                predecessors[target].add(source)
        return units, predecessors

    @staticmethod
    def _legal_order(preferred: List[int], predecessors: Dict[int, set]) -> List[int]:
        """Stable topological ordering, falling back safely for cycles."""
        rank = {unit: position for position, unit in enumerate(preferred)}
        remaining = set(preferred)
        result = []
        while remaining:
            ready = [u for u in remaining if not (predecessors.get(u, set()) & remaining)]
            if not ready:
                # Cyclic user constraints cannot all be satisfied.  Preserve a
                # deterministic preferred order rather than losing any column.
                ready = list(remaining)
            ready.sort(key=lambda u: rank[u])
            chosen = ready[0]
            result.append(chosen)
            remaining.remove(chosen)
        return result

    def _global_order(
        self,
        units: List[List[Any]],
        unit_text: List[List[str]],
        predecessors: Dict[int, set],
        alternate: bool = False,
    ) -> List[int]:
        scores = []
        for unit_index, values in enumerate(unit_text):
            counts = Counter(values)
            if alternate:
                score = sum(len(value) * (count - 1)
                            for value, count in counts.items())
            else:
                score = sum(len(value) * count * (count - 1)
                            for value, count in counts.items())
            scores.append(score)
        preferred = sorted(range(len(units)), key=lambda u: (-scores[u], u))
        return self._legal_order(preferred, predecessors)

    def _conditional_orders(
        self,
        units: List[List[Any]],
        unit_text: List[List[str]],
        global_tail: List[int],
        predecessors: Dict[int, set],
        depth_limit: int,
    ) -> List[List[int]]:
        nrows = len(unit_text[0]) if unit_text else 0
        result = [None] * nrows
        candidate_cap = min(len(units), 48)
        branch_budget = 2048

        def finish(rows, prefix, remaining):
            preferred = [u for u in global_tail if u in remaining]
            preferred += [u for u in remaining if u not in preferred]
            tail = self._legal_order(prefix + preferred, predecessors)
            for row in rows:
                result[row] = tail

        def visit(rows, prefix, remaining, depth):
            nonlocal branch_budget
            if not rows:
                return
            if (depth >= depth_limit or not remaining or len(rows) < 2
                    or branch_budget <= 0):
                finish(rows, prefix, remaining)
                return

            candidates = [u for u in global_tail if u in remaining][:candidate_cap]
            best_unit = None
            best_score = 0
            best_groups = None

            for unit in candidates:
                groups = defaultdict(list)
                values = unit_text[unit]
                for row in rows:
                    groups[values[row]].append(row)
                score = sum(len(value) * len(group) * (len(group) - 1)
                            for value, group in groups.items())
                if score > best_score or (score == best_score and
                                          best_unit is not None and unit < best_unit):
                    best_unit, best_score, best_groups = unit, score, groups

            if best_unit is None or best_score <= 0 or len(best_groups) <= 1:
                finish(rows, prefix, remaining)
                return

            branch_budget -= len(best_groups)
            next_remaining = [u for u in remaining if u != best_unit]
            for value in sorted(best_groups):
                visit(best_groups[value], prefix + [best_unit],
                      next_remaining, depth + 1)

        visit(list(range(nrows)), [], list(range(len(units))), 0)
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
        # Do not mutate df: validation requires preservation of the caller's
        # data, columns, index, and row identities.
        columns = list(df.columns)
        nrows, ncols = df.shape
        if nrows == 0 or ncols == 0 or len(set(columns)) != len(columns):
            return df.copy(), [columns[:] for _ in range(nrows)]

        units, predecessors = self._units(columns, col_merge, one_way_dep)
        positions = {column: i for i, column in enumerate(columns)}
        raw_values = df.to_numpy(dtype=object, copy=True)

        # Text and unit strings are shared by all candidate constructions.
        cell_text = [[self._text(raw_values[row, col])
                      for row in range(nrows)]
                     for col in range(ncols)]
        unit_text = []
        for unit in units:
            member_positions = [positions[column] for column in unit]
            unit_text.append([
                "".join(cell_text[position][row] for position in member_positions)
                for row in range(nrows)
            ])

        global_order = self._global_order(units, unit_text, predecessors, False)
        alternate_order = self._global_order(units, unit_text, predecessors, True)
        candidates = [global_order]
        if alternate_order != global_order:
            candidates.append(alternate_order)

        # Conditional construction is deliberately bounded on very wide/large
        # frames.  The global candidate remains a reliable safe alternative.
        if nrows * max(1, len(units)) <= 2_000_000 and len(units) > 1:
            limit = col_stop if col_stop is not None and col_stop > 0 else 12
            conditional = self._conditional_orders(
                units, unit_text, global_order, predecessors, min(limit, 16)
            )
            candidates.append(conditional)

        best_orders = None
        best_score = -1
        for candidate in candidates[:3]:
            if candidate and isinstance(candidate[0], list):
                row_unit_orders = candidate
            else:
                row_unit_orders = [candidate] * nrows

            row_orders = [
                [column for unit in row_unit_orders[row] for column in units[unit]]
                for row in range(nrows)
            ]
            serialized = [
                "".join(cell_text[positions[column]][row] for column in row_orders[row])
                for row in range(nrows)
            ]
            score = self._trie_score(serialized)
            if score > best_score:
                best_score = score
                best_orders = row_orders

        # Store each ordered row positionally.  column_orderings supplies the
        # source column for each slot, including when orders differ by row.
        output = []
        for row, order in enumerate(best_orders):
            output.append([raw_values[row, positions[column]] for column in order])

        reordered = pd.DataFrame(output, index=df.index, columns=columns, dtype=object)
        return reordered, best_orders

# EVOLVE-BLOCK-END