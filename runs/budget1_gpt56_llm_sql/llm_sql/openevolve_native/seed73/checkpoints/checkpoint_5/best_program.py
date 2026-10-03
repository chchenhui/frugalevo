"""
Data-preserving prefix-cache-oriented dataframe reordering.

The evaluator scores concatenated row strings using serial Trie reuse.  This
implementation constructs a small bounded set of column-order candidates and
selects the candidate with the highest exact sorted-row LCP score.

Candidates:
1. Identity order, preserving the original representation as a baseline.
2. Global order ranked by repeated full-value character mass.
3. Global order ranked by actual per-column Trie-prefix reuse.
4. Bounded row-specific conditional prefix partitioning.

All output cells remain original source objects.  Only temporary scoring strings
normalize missing values to empty strings.
"""

from collections import Counter
import heapq
from typing import List, Tuple

import numpy as np
import pandas as pd

from solver import Algorithm


class Evolved(Algorithm):
    """Build bounded data-preserving column permutation candidates."""

    def __init__(self, df: pd.DataFrame = None):
        """Store an optional dataframe reference without mutating it."""
        self.df = df

    @staticmethod
    def _cell_strings(df: pd.DataFrame) -> np.ndarray:
        """Return evaluator-compatible string values while retaining source data."""
        rows, cols = df.shape
        out = np.empty((rows, cols), dtype=object)

        for col in range(cols):
            values = df.iloc[:, col].to_numpy(dtype=object, copy=False)
            converted = []

            for value in values:
                try:
                    missing = pd.isna(value)
                    if isinstance(missing, (bool, np.bool_)) and missing:
                        converted.append("")
                        continue
                except Exception:
                    pass

                converted.append(str(value))

            out[:, col] = converted

        return out

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Compute a string LCP with logarithmically many C-level slice checks."""
        limit = min(len(a), len(b))

        if limit == 0:
            return 0
        if a == b:
            return limit

        low, high = 0, limit
        while low < high:
            middle = (low + high + 1) // 2
            if a[:middle] == b[:middle]:
                low = middle
            else:
                high = middle - 1

        return low

    @classmethod
    def _reuse_score(cls, rows: List[str]) -> int:
        """Compute exact ideal serial-Trie reuse from sorted adjacent row LCPs."""
        if len(rows) < 2:
            return 0

        ordered = sorted(rows)
        return sum(cls._lcp(ordered[i - 1], ordered[i]) for i in range(1, len(ordered)))

    @staticmethod
    def _serialize(strings: np.ndarray, orders: List[List[int]]) -> List[str]:
        """Serialize each row according to its row-specific source-column order."""
        return [
            "".join(strings[row_id, order])
            for row_id, order in enumerate(orders)
        ]

    @staticmethod
    def _factorize(strings: np.ndarray):
        """Factorize each scoring-string column once for repeated-value statistics."""
        rows, cols = strings.shape
        codes = np.empty((rows, cols), dtype=np.int32)
        lengths = []

        for col in range(cols):
            code, uniques = pd.factorize(strings[:, col], sort=False)
            codes[:, col] = code.astype(np.int32, copy=False)
            lengths.append(np.asarray([len(value) for value in uniques], dtype=np.int64))

        return codes, lengths

    @staticmethod
    def _frequency_scores(codes: np.ndarray, lengths: List[np.ndarray]) -> np.ndarray:
        """Score fields by len(value) * count(value) * (count(value)-1)."""
        _, cols = codes.shape
        result = np.zeros(cols, dtype=np.int64)

        for col in range(cols):
            counts = np.bincount(codes[:, col], minlength=len(lengths[col]))
            result[col] = int(np.sum(lengths[col] * counts * (counts - 1)))

        return result

    @classmethod
    def _prefix_scores(cls, strings: np.ndarray, frequency: np.ndarray) -> np.ndarray:
        """
        Score leading fields by their actual standalone character-Trie reuse.

        Full sorting is bounded on very wide frames: highly repeated fields are
        the most plausible useful prefix fields, while remaining fields retain
        their frequency-rank fallback order.
        """
        rows, cols = strings.shape
        result = np.zeros(cols, dtype=np.int64)

        if rows < 2:
            return result

        if cols <= 48:
            candidates = range(cols)
        else:
            candidates = np.argsort(-frequency, kind="stable")[:48]

        for col in candidates:
            values = sorted(strings[:, int(col)].tolist())
            result[int(col)] = sum(
                cls._lcp(values[i - 1], values[i])
                for i in range(1, len(values))
            )

        return result

    @staticmethod
    def _dependency_edges(columns, one_way_dep) -> List[Tuple[int, int]]:
        """Resolve dependency names into unique source-column position pairs."""
        edges = []

        for edge in one_way_dep or []:
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                continue

            left_name, right_name = edge
            left = [i for i, col in enumerate(columns) if str(left_name) in str(col)]
            right = [i for i, col in enumerate(columns) if str(right_name) in str(col)]

            if len(left) == 1 and len(right) == 1 and left[0] != right[0]:
                edges.append((left[0], right[0]))

        return edges

    @staticmethod
    def _apply_dependencies(order: List[int], edges: List[Tuple[int, int]]) -> List[int]:
        """Return a stable topological ordering, preserving order on cycles."""
        if not edges:
            return list(order)

        rank = {col: i for i, col in enumerate(order)}
        children = {col: [] for col in order}
        indegree = {col: 0 for col in order}

        for left, right in edges:
            if left in indegree and right in indegree:
                children[left].append(right)
                indegree[right] += 1

        ready = [(rank[col], col) for col in order if indegree[col] == 0]
        heapq.heapify(ready)
        result = []

        while ready:
            _, current = heapq.heappop(ready)
            result.append(current)

            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, (rank[child], child))

        return result if len(result) == len(order) else list(order)

    @staticmethod
    def _merge_units(columns, col_merge) -> List[List[int]]:
        """Construct non-overlapping merged source-column units."""
        used = set()
        units = []

        for group in col_merge or []:
            if not isinstance(group, (list, tuple)):
                continue

            positions = []
            for name in group:
                matches = [
                    i for i, col in enumerate(columns)
                    if col == name and i not in used
                ]
                if len(matches) == 1:
                    positions.append(matches[0])

            if len(positions) > 1:
                units.append(positions)
                used.update(positions)

        for col in range(len(columns)):
            if col not in used:
                units.append([col])

        return units

    def _global_order(
        self,
        scores: np.ndarray,
        units: List[List[int]],
        edges: List[Tuple[int, int]],
    ) -> List[int]:
        """Rank merge-safe units by score and apply dependency constraints."""
        ranked_units = sorted(
            units,
            key=lambda unit: (-sum(int(scores[col]) for col in unit), min(unit)),
        )

        order = []
        for unit in ranked_units:
            order.extend(sorted(unit, key=lambda col: (-int(scores[col]), col)))

        return self._apply_dependencies(order, edges)

    @staticmethod
    def _local_score(
        row_ids: np.ndarray,
        column: int,
        codes: np.ndarray,
        lengths: List[np.ndarray],
    ) -> int:
        """Measure repeated full-value character mass inside one row partition."""
        selected = codes[row_ids, column]

        if len(selected) < 2:
            return 0

        counts = np.bincount(selected, minlength=len(lengths[column]))
        return int(np.sum(lengths[column] * counts * (counts - 1)))

    def _conditional_orders(
        self,
        codes: np.ndarray,
        lengths: List[np.ndarray],
        global_order: List[int],
        edges: List[Tuple[int, int]],
        row_stop,
        col_stop,
    ) -> List[List[int]]:
        """
        Build bounded row-specific prefix partitions using local repetition mass.

        Each partition chooses one remaining field with maximal repeated character
        value, then recursively specializes only duplicate-value child groups.
        """
        rows, cols = codes.shape
        orders = [list(global_order) for _ in range(rows)]

        if rows < 2 or cols < 2:
            return orders

        depth_limit = min(cols, 10)

        for value in (row_stop, col_stop):
            if value is not None:
                try:
                    depth_limit = min(depth_limit, max(1, int(value)))
                except Exception:
                    pass

        rank = {col: pos for pos, col in enumerate(global_order)}
        stack = [(np.arange(rows, dtype=np.int64), [], 0)]
        nodes = 0
        max_nodes = 160
        candidate_cap = min(cols, 28)

        while stack and nodes < max_nodes:
            row_ids, prefix, depth = stack.pop()
            nodes += 1

            if len(row_ids) < 2 or depth >= depth_limit:
                continue

            used = set(prefix)
            remaining = [col for col in global_order if col not in used]

            if not remaining:
                continue

            best_col = None
            best_score = 0

            for col in remaining[:candidate_cap]:
                score = self._local_score(row_ids, col, codes, lengths)

                if (
                    score > best_score
                    or (
                        score == best_score
                        and best_col is not None
                        and rank[col] < rank[best_col]
                    )
                ):
                    best_col = col
                    best_score = score

            if best_col is None or best_score <= 0:
                continue

            next_prefix = prefix + [best_col]
            prefix_set = set(next_prefix)
            local = next_prefix + [col for col in global_order if col not in prefix_set]
            local = self._apply_dependencies(local, edges)

            for row_id in row_ids:
                orders[int(row_id)] = local

            selected = codes[row_ids, best_col]
            sort_order = np.argsort(selected, kind="stable")
            grouped_rows = row_ids[sort_order]
            grouped_codes = selected[sort_order]

            start = 0
            while start < len(grouped_rows):
                end = start + 1
                while end < len(grouped_rows) and grouped_codes[end] == grouped_codes[start]:
                    end += 1

                if end - start > 1:
                    stack.append((grouped_rows[start:end], next_prefix, depth + 1))

                start = end

        return orders

    @staticmethod
    def _materialize(df: pd.DataFrame, orders: List[List[int]]) -> pd.DataFrame:
        """Create an object dataframe by applying each row's source permutation."""
        source = df.to_numpy(dtype=object, copy=True)
        rows, cols = source.shape
        output = np.empty((rows, cols), dtype=object)

        for row in range(rows):
            output[row, :] = source[row, orders[row]]

        return pd.DataFrame(output, index=df.index.copy(), columns=df.columns.copy())

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
        Return the highest exact-Trie-score candidate under supplied constraints.

        The implementation never changes dataframe values, row count, column
        count, index, or source row identity.  Missing-value normalization occurs
        solely in temporary evaluator-compatible scoring strings.
        """
        rows, cols = df.shape
        columns = list(df.columns)

        if rows == 0 or cols == 0:
            return df.copy(), [[] for _ in range(rows)]

        strings = self._cell_strings(df)
        codes, lengths = self._factorize(strings)

        edges = self._dependency_edges(columns, one_way_dep)
        units = self._merge_units(columns, col_merge)
        has_merges = any(len(unit) > 1 for unit in units)

        frequency_scores = self._frequency_scores(codes, lengths)
        prefix_scores = self._prefix_scores(strings, frequency_scores)

        identity = list(range(cols))
        frequency_order = self._global_order(frequency_scores, units, edges)
        prefix_order = self._global_order(prefix_scores, units, edges)

        candidates = []

        if not edges and not has_merges:
            candidates.append([list(identity) for _ in range(rows)])

        candidates.append([list(frequency_order) for _ in range(rows)])
        candidates.append([list(prefix_order) for _ in range(rows)])

        if not has_merges:
            candidates.append(
                self._conditional_orders(
                    codes,
                    lengths,
                    frequency_order,
                    edges,
                    row_stop,
                    col_stop,
                )
            )

        best_orders = candidates[0]
        best_score = -1

        for candidate in candidates:
            score = self._reuse_score(self._serialize(strings, candidate))
            if score > best_score:
                best_score = score
                best_orders = candidate

        reordered = self._materialize(df, best_orders)
        column_orderings = [
            [columns[position] for position in order]
            for order in best_orders
        ]

        return reordered, column_orderings