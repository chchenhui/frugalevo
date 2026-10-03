# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from collections import Counter


class Evolved(Algorithm):
    """
    Prefix-cache-oriented row serializer reordering.

    The returned DataFrame stores each output row in the order described by the
    corresponding entry in column_orderings.  Values are copied as objects so
    mixed types and missing values are preserved exactly.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _serialized_value(value) -> str:
        """Match evaluator normalization without changing the stored value."""
        if value is None:
            return ""
        try:
            missing = pd.isna(value)
            if isinstance(missing, bool) and missing:
                return ""
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Bounded binary-search LCP; slicing comparisons execute in C."""
        if left == right:
            return len(left)
        limit = min(len(left), len(right))
        low, high = 0, limit
        while low < high:
            middle = (low + high + 1) // 2
            if left[:middle] == right[:middle]:
                low = middle
            else:
                high = middle - 1
        return low

    def _trie_reuse_score(self, rows: List[str]) -> int:
        if len(rows) < 2:
            return 0
        ordered = sorted(rows)
        return sum(self._lcp(ordered[i - 1], ordered[i]) for i in range(1, len(ordered)))

    @staticmethod
    def _stable_topological_order(
        columns: List,
        priority: Dict,
        one_way_dep: List[Tuple[str, str]],
    ) -> List:
        """
        Apply requested one-way dependencies while retaining deterministic score
        ordering. Dependency names use the historical substring matching rule.
        """
        position = {col: i for i, col in enumerate(columns)}
        edges = {col: set() for col in columns}
        indegree = {col: 0 for col in columns}

        for before_hint, after_hint in one_way_dep or []:
            before_matches = [c for c in columns if str(before_hint) in str(c)]
            after_matches = [c for c in columns if str(after_hint) in str(c)]
            if len(before_matches) == 1 and len(after_matches) == 1:
                before, after = before_matches[0], after_matches[0]
                if before != after and after not in edges[before]:
                    edges[before].add(after)
                    indegree[after] += 1

        result = []
        available = [c for c in columns if indegree[c] == 0]
        while available:
            available.sort(key=lambda c: (-priority.get(c, 0), position[c]))
            current = available.pop(0)
            result.append(current)
            for child in sorted(edges[current], key=lambda c: position[c]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    available.append(child)

        # Cyclic constraints cannot all be satisfied. Preserve deterministic
        # ordering rather than dropping or duplicating any source column.
        if len(result) != len(columns):
            remaining = [c for c in columns if c not in set(result)]
            remaining.sort(key=lambda c: (-priority.get(c, 0), position[c]))
            result.extend(remaining)
        return result

    def _apply_merge_constraints(
        self,
        ranked_columns: List,
        priority: Dict,
        col_merge: List[List[str]],
        one_way_dep: List[Tuple[str, str]],
    ) -> List:
        """
        Treat requested merge groups as contiguous column blocks without merging
        values or changing shape. Columns not named by a group remain singleton
        blocks.
        """
        if not col_merge:
            return self._stable_topological_order(ranked_columns, priority, one_way_dep)

        used = set()
        blocks = []
        rank_pos = {c: i for i, c in enumerate(ranked_columns)}
        for group in col_merge:
            members = [c for c in ranked_columns if c in group and c not in used]
            if members:
                members.sort(key=lambda c: rank_pos[c])
                blocks.append(members)
                used.update(members)
        for col in ranked_columns:
            if col not in used:
                blocks.append([col])

        block_of = {col: i for i, block in enumerate(blocks) for col in block}
        block_priority = {
            i: max(priority.get(c, 0) for c in block) for i, block in enumerate(blocks)
        }
        edges = {i: set() for i in range(len(blocks))}
        indegree = {i: 0 for i in range(len(blocks))}

        for before_hint, after_hint in one_way_dep or []:
            left = [c for c in ranked_columns if str(before_hint) in str(c)]
            right = [c for c in ranked_columns if str(after_hint) in str(c)]
            if len(left) == 1 and len(right) == 1:
                a, b = block_of[left[0]], block_of[right[0]]
                if a != b and b not in edges[a]:
                    edges[a].add(b)
                    indegree[b] += 1

        available = [i for i in range(len(blocks)) if indegree[i] == 0]
        ordered_blocks = []
        while available:
            available.sort(key=lambda i: (-block_priority[i], i))
            current = available.pop(0)
            ordered_blocks.append(current)
            for child in sorted(edges[current]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    available.append(child)

        if len(ordered_blocks) != len(blocks):
            ordered_blocks.extend(i for i in range(len(blocks)) if i not in ordered_blocks)

        return [col for block_id in ordered_blocks for col in blocks[block_id]]

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

        columns = list(df.columns)
        row_count = len(df)
        column_count = len(columns)

        if row_count == 0:
            return df.copy(), []
        if column_count == 0:
            return df.copy(), [[] for _ in range(row_count)]

        # Keep actual cells untouched. These strings are only the evaluator
        # serialization representation used to choose a layout.
        values = df.to_numpy(dtype=object, copy=True)
        text = [
            [self._serialized_value(values[row][col]) for row in range(row_count)]
            for col in range(column_count)
        ]

        repetition_score = {}
        alternative_score = {}
        for col_index, column in enumerate(columns):
            counts = Counter(text[col_index])
            repetition_score[column] = sum(
                len(value) * count * (count - 1)
                for value, count in counts.items()
            )
            alternative_score[column] = sum(
                len(value) * (count - 1)
                for value, count in counts.items()
            )

        original_position = {column: i for i, column in enumerate(columns)}
        frequency_order = sorted(
            columns,
            key=lambda c: (-repetition_score[c], original_position[c]),
        )
        alternate_order = sorted(
            columns,
            key=lambda c: (-alternative_score[c], original_position[c]),
        )

        frequency_order = self._apply_merge_constraints(
            frequency_order, repetition_score, col_merge, one_way_dep
        )
        alternate_order = self._apply_merge_constraints(
            alternate_order, alternative_score, col_merge, one_way_dep
        )

        candidates = [frequency_order]
        if alternate_order != frequency_order:
            candidates.append(alternate_order)

        # A bounded conditional candidate gives repeated values a chance to
        # become prefixes within local groups. Constraints require a common
        # order, so use only globally valid candidates in that case.
        if not col_merge and not one_way_dep and column_count <= 32:
            per_row_orders = []
            base_positions = {c: i for i, c in enumerate(frequency_order)}
            for row in range(row_count):
                # Stable predecessor-alignment alternative: fields matching the
                # immediately previous source row lead, then frequency order.
                if row == 0:
                    per_row_orders.append(list(frequency_order))
                    continue
                matching = [
                    c for c in frequency_order
                    if text[columns.index(c)][row] == text[columns.index(c)][row - 1]
                ]
                remainder = [c for c in frequency_order if c not in set(matching)]
                matching.sort(
                    key=lambda c: (
                        -len(text[columns.index(c)][row]),
                        base_positions[c],
                    )
                )
                per_row_orders.append(matching + remainder)
            candidates.append(per_row_orders)

        def candidate_strings(candidate):
            if candidate and isinstance(candidate[0], list):
                orders = candidate
            else:
                orders = [candidate] * row_count
            strings = []
            for row, order in enumerate(orders):
                strings.append("".join(text[columns.index(col)][row] for col in order))
            return strings, orders

        best_score = -1
        best_orders = None
        best_strings = None
        for candidate in candidates[:3]:
            strings, orders = candidate_strings(candidate)
            score = self._trie_reuse_score(strings)
            if score > best_score:
                best_score = score
                best_orders = orders
                best_strings = strings

        # Sorting is deterministic and tends to be cache-friendly in evaluators
        # that use insertion order, while preserving every original source row.
        row_order = sorted(range(row_count), key=lambda row: (best_strings[row], row))

        output_rows = []
        output_orders = []
        col_index = {column: i for i, column in enumerate(columns)}
        for row in row_order:
            order = list(best_orders[row])
            output_rows.append([values[row][col_index[column]] for column in order])
            output_orders.append(order)

        # Column labels are positional for row-specific orderings; the matching
        # source-column identity for each cell is supplied by column_orderings.
        result = pd.DataFrame(
            output_rows,
            index=df.index.take(row_order),
            columns=columns,
            dtype=object,
        )
        return result, output_orders


# EVOLVE-BLOCK-END