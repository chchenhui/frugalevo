# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from collections import Counter


class Evolved(Algorithm):
    """
    Prefix-cache-oriented dataframe reordering.

    The dataframe values are never changed.  A row-specific ordering describes
    which source column supplied each output position when row-specific layouts
    are used.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _text(value) -> str:
        """Match evaluator normalization without modifying stored dataframe data."""
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
        high = min(len(left), len(right))
        low = 0
        # Slice comparisons run in C and avoid a Python loop over characters.
        while low < high:
            mid = (low + high + 1) // 2
            if left[:mid] == right[:mid]:
                low = mid
            else:
                high = mid - 1
        return low

    def _trie_score(self, rows: List[str]) -> int:
        if len(rows) < 2:
            return 0
        ordered = sorted(rows)
        return sum(self._lcp(ordered[i - 1], ordered[i]) for i in range(1, len(ordered)))

    @staticmethod
    def _constraint_maps(columns, col_merge, one_way_dep):
        """Resolve API column names to positional constraints, including duplicates."""
        positions = {}
        for i, name in enumerate(columns):
            positions.setdefault(name, []).append(i)

        merge_groups = []
        for group in col_merge or []:
            found = []
            for requested in group:
                # Existing API historically allowed substring dependency names.
                for name, indices in positions.items():
                    if name == requested or str(requested) in str(name):
                        found.extend(indices)
            unique = []
            for index in found:
                if index not in unique:
                    unique.append(index)
            if len(unique) > 1:
                merge_groups.append(unique)

        dependencies = []
        for before_name, after_name in one_way_dep or []:
            before = []
            after = []
            for name, indices in positions.items():
                if before_name == name or str(before_name) in str(name):
                    before.extend(indices)
                if after_name == name or str(after_name) in str(name):
                    after.extend(indices)
            if before and after:
                dependencies.append((before[0], after[0]))
        return merge_groups, dependencies

    @staticmethod
    def _apply_constraints(order, merge_groups, dependencies):
        """
        Keep merged columns contiguous and place declared dependency predecessors
        before successors.  This only permutes positions; no dataframe values are
        changed.
        """
        result = list(order)

        for group in merge_groups:
            members = [item for item in result if item in group]
            if len(members) < 2:
                continue
            first = min(result.index(item) for item in members)
            result = [item for item in result if item not in group]
            result[first:first] = members

        # A bounded relaxation handles ordinary acyclic dependency declarations.
        for _ in range(max(1, len(dependencies))):
            changed = False
            for before, after in dependencies:
                try:
                    before_pos = result.index(before)
                    after_pos = result.index(after)
                except ValueError:
                    continue
                if before_pos > after_pos:
                    result.pop(before_pos)
                    after_pos = result.index(after)
                    result.insert(after_pos, before)
                    changed = True
            if not changed:
                break
        return result

    def _global_orders(self, cell_text, mode, merge_groups, dependencies):
        rows = len(cell_text)
        cols = len(cell_text[0]) if rows else 0
        scores = []
        for column in range(cols):
            counts = Counter(cell_text[row][column] for row in range(rows))
            if mode == 0:
                # Requested length-weighted pair repetition heuristic.
                score = sum(len(value) * count * (count - 1) for value, count in counts.items())
            else:
                # A distinct, cheaper tradeoff retaining useful repeated prefixes.
                score = sum(len(value) * (count - 1) for value, count in counts.items())
            scores.append(score)
        order = sorted(range(cols), key=lambda c: (-scores[c], c))
        order = self._apply_constraints(order, merge_groups, dependencies)
        return [order[:] for _ in range(rows)]

    def _conditional_orders(
        self,
        cell_text,
        merge_groups,
        dependencies,
        max_depth,
    ):
        """
        Conditional prefix partition tree.  Value codes are represented by the
        already-normalized strings and no pandas work is performed at tree nodes.
        """
        row_count = len(cell_text)
        col_count = len(cell_text[0]) if row_count else 0
        orders = [[] for _ in range(row_count)]
        global_order = self._global_orders(cell_text, 0, [], [])[0] if row_count else []

        def finish(indices, remaining):
            tail = [column for column in global_order if column in remaining]
            for row in indices:
                orders[row].extend(tail)

        def visit(indices, remaining, depth):
            if not indices:
                return
            if not remaining or depth >= max_depth or len(indices) < 2:
                finish(indices, remaining)
                return

            # Limit wide-table node work, while global_order gives deterministic
            # candidates most likely to contain useful repeated material.
            candidates = [c for c in global_order if c in remaining][:24]
            best_column = None
            best_score = 0
            best_parts = None

            for column in candidates:
                parts = {}
                for row in indices:
                    parts.setdefault(cell_text[row][column], []).append(row)
                score = sum(
                    len(value) * len(group) * (len(group) - 1)
                    for value, group in parts.items()
                    if len(group) > 1
                )
                if score > best_score or (score == best_score and best_column is not None and column < best_column):
                    best_column = column
                    best_score = score
                    best_parts = parts

            if best_column is None or best_score <= 0:
                finish(indices, remaining)
                return

            for row in indices:
                orders[row].append(best_column)
            next_remaining = [column for column in remaining if column != best_column]

            # Large fan-outs with no repetition cannot yield another useful
            # conditional prefix, so finish those leaves immediately.
            for value in sorted(best_parts):
                group = best_parts[value]
                if len(group) < 2:
                    finish(group, next_remaining)
                else:
                    visit(group, next_remaining, depth + 1)

        visit(list(range(row_count)), list(range(col_count)), 0)
        return [
            self._apply_constraints(order, merge_groups, dependencies)
            for order in orders
        ]

    def _candidate_score(self, cell_text, orders):
        serialized = [
            "".join(cell_text[row][column] for column in orders[row])
            for row in range(len(orders))
        ]
        return self._trie_score(serialized), serialized

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True):
        """Compatibility helper retaining the original public method."""
        order = list(range(df.shape[1]))
        text = [[self._text(v) for v in row] for row in df.to_numpy(dtype=object)]
        serialized = ["".join(row) for row in text]
        indices = sorted(range(len(df)), key=serialized.__getitem__) if row_sort else list(range(len(df)))
        output = df.iloc[indices, :].copy()
        return output, [list(df.columns) for _ in indices]

    def calculate_length(self, value):
        return len(self._text(value))

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

        row_count, col_count = df.shape
        if row_count == 0 or col_count == 0:
            return df.copy(), [[] for _ in range(row_count)]

        values = df.to_numpy(dtype=object, copy=True)
        cell_text = [
            [self._text(values[row, column]) for column in range(col_count)]
            for row in range(row_count)
        ]
        merge_groups, dependencies = self._constraint_maps(
            list(df.columns), col_merge, one_way_dep
        )

        # At most three whole-data candidates are scored.
        candidates = [
            self._global_orders(cell_text, 0, merge_groups, dependencies),
            self._global_orders(cell_text, 1, merge_groups, dependencies),
        ]
        depth_limit = col_count
        if col_stop is not None and col_stop > 0:
            depth_limit = min(depth_limit, col_stop)
        if row_stop is not None and row_stop > 0:
            # row_stop was historically recursion-related, not a row deletion
            # request; use it only as a harmless tree-depth work bound.
            depth_limit = min(depth_limit, max(1, row_stop))
        candidates.append(
            self._conditional_orders(
                cell_text, merge_groups, dependencies, max_depth=min(depth_limit, 12)
            )
        )

        best_orders = candidates[0]
        best_score, best_serialized = self._candidate_score(cell_text, best_orders)
        for candidate in candidates[1:]:
            score, serialized = self._candidate_score(cell_text, candidate)
            if score > best_score:
                best_orders = candidate
                best_score = score
                best_serialized = serialized

        # Sorting is deterministic and does not alter ideal Trie reuse.  Preserve
        # original dataframe index so each output row retains its source identity.
        row_order = sorted(range(row_count), key=best_serialized.__getitem__)
        output_values = [
            [values[row, source_column] for source_column in best_orders[row]]
            for row in row_order
        ]

        # Row-specific layouts cannot have one universally meaningful set of
        # column labels.  The ordering list explicitly maps each output position
        # to its original source column, while object dtype protects mixed values.
        output = pd.DataFrame(
            output_values,
            index=df.index.take(row_order),
            columns=df.columns,
            dtype=object,
        )
        column_orderings = [
            [df.columns[source_column] for source_column in best_orders[row]]
            for row in row_order
        ]
        return output, column_orderings


# EVOLVE-BLOCK-END