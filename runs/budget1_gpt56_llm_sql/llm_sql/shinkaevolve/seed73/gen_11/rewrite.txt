# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List
from collections import defaultdict
import numpy as np


class Evolved(Algorithm):
    """
    Prefix-cache-oriented row and per-row column reordering.

    A row ordering is represented by a list of source-column positions for
    every output row.  Output position k contains the value from source column
    ordering[k], and the matching returned ordering describes that permutation.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _score_string(value) -> str:
        """Match pandas fillna('').astype(str) for scalar cell values."""
        if value is None:
            return ""
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and missing:
                return ""
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """Bounded Python-level LCP using C slice comparisons."""
        limit = min(len(left), len(right))
        if limit == 0:
            return 0
        if left == right:
            return limit
        lo, hi = 0, limit
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
    def _frequency_order(strings: np.ndarray, pair_weight: bool) -> List[int]:
        """Global deterministic column ranking."""
        nrows, ncols = strings.shape
        ranked = []
        for col in range(ncols):
            counts = defaultdict(int)
            lengths = {}
            for value in strings[:, col]:
                counts[value] += 1
                lengths[value] = len(value)
            if pair_weight:
                score = sum(lengths[v] * count * (count - 1)
                            for v, count in counts.items())
            else:
                score = sum(lengths[v] * max(0, count - 1)
                            for v, count in counts.items())
            ranked.append((-score, col))
        ranked.sort()
        return [col for _, col in ranked]

    def _column_groups(self, columns, col_merge):
        """Create non-overlapping contiguous merge groups by source position."""
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

        positions = defaultdict(list)
        for i, name in enumerate(columns):
            positions[name].append(i)

        for requested_group in col_merge or []:
            members = []
            for name in requested_group:
                members.extend(positions.get(name, []))
            if len(members) > 1:
                first = members[0]
                for member in members[1:]:
                    union(first, member)

        grouped = defaultdict(list)
        for pos in range(ncols):
            grouped[find(pos)].append(pos)
        return list(grouped.values())

    def _apply_constraints(self, order, columns, col_merge, one_way_dep):
        """
        Keep requested merge groups contiguous and enforce resolvable directed
        dependencies without changing any values.
        """
        ncols = len(order)
        groups = self._column_groups(columns, col_merge)
        group_for_pos = {}
        for group_id, members in enumerate(groups):
            for pos in members:
                group_for_pos[pos] = group_id

        rank = {pos: i for i, pos in enumerate(order)}
        groups = [sorted(group, key=lambda p: rank[p]) for group in groups]
        group_rank = {
            group_id: min(rank[p] for p in members)
            for group_id, members in enumerate(groups)
        }

        by_name = defaultdict(list)
        for pos, name in enumerate(columns):
            by_name[str(name)].append(pos)

        edges = defaultdict(set)
        indegree = [0] * len(groups)
        for dependency in one_way_dep or []:
            if len(dependency) != 2:
                continue
            left_matches = [
                pos for name, positions in by_name.items()
                if str(dependency[0]) in name for pos in positions
            ]
            right_matches = [
                pos for name, positions in by_name.items()
                if str(dependency[1]) in name for pos in positions
            ]
            if len(left_matches) != 1 or len(right_matches) != 1:
                continue
            source = group_for_pos[left_matches[0]]
            target = group_for_pos[right_matches[0]]
            if source != target and target not in edges[source]:
                edges[source].add(target)
                indegree[target] += 1

        available = [g for g in range(len(groups)) if indegree[g] == 0]
        result_groups = []
        while available:
            available.sort(key=lambda g: (group_rank[g], g))
            current = available.pop(0)
            result_groups.append(current)
            for target in sorted(edges[current]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    available.append(target)

        # A cyclic dependency cannot be satisfied; preserve deterministic order.
        if len(result_groups) != len(groups):
            used = set(result_groups)
            result_groups.extend(
                g for g in sorted(range(len(groups)),
                                  key=lambda x: (group_rank[x], x))
                if g not in used
            )

        return [pos for group_id in result_groups for pos in groups[group_id]]

    def _conditional_orders(self, strings, global_order, depth_limit, early_stop):
        """
        Conditional prefix partition tree.  It uses factorized values once and
        selects fields by actual length-weighted pair repetition within a node.
        """
        nrows, ncols = strings.shape
        if nrows == 0 or ncols == 0:
            return [list(global_order) for _ in range(nrows)]

        codes = np.empty((nrows, ncols), dtype=np.int32)
        for col in range(ncols):
            codes[:, col] = pd.factorize(strings[:, col], sort=False)[0]

        shortlist = global_order[:min(ncols, 32)]
        result = [None] * nrows
        max_nodes = 256
        nodes_used = 0

        def finish(rows, prefix):
            used = set(prefix)
            tail = [col for col in global_order if col not in used]
            final = prefix + tail
            for row in rows:
                result[int(row)] = final

        def visit(rows, remaining, prefix, depth):
            nonlocal nodes_used
            if (len(rows) <= 1 or not remaining or depth >= depth_limit or
                    nodes_used >= max_nodes):
                finish(rows, prefix)
                return

            candidates = [c for c in shortlist if c in remaining]
            if not candidates:
                candidates = remaining[:min(8, len(remaining))]

            best_col = None
            best_score = 0
            for col in candidates:
                local_codes = codes[rows, col]
                counts = np.bincount(local_codes)
                present = np.flatnonzero(counts > 1)
                if len(present) == 0:
                    continue
                lengths = np.array(
                    [len(strings[rows[np.where(local_codes == code)[0][0]], col])
                     for code in present],
                    dtype=np.int64,
                )
                repeated = counts[present].astype(np.int64)
                score = int(np.sum(lengths * repeated * (repeated - 1)))
                if score > best_score or (
                    score == best_score and best_col is not None and col < best_col
                ):
                    best_score, best_col = score, col

            if best_col is None or best_score <= early_stop:
                finish(rows, prefix)
                return

            nodes_used += 1
            ordered_rows = rows[np.argsort(codes[rows, best_col], kind="stable")]
            ordered_codes = codes[ordered_rows, best_col]
            start = 0
            next_remaining = [c for c in remaining if c != best_col]
            while start < len(ordered_rows):
                end = start + 1
                while end < len(ordered_rows) and ordered_codes[end] == ordered_codes[start]:
                    end += 1
                visit(ordered_rows[start:end], next_remaining,
                      prefix + [best_col], depth + 1)
                start = end

        visit(np.arange(nrows, dtype=np.int64), list(global_order), [], 0)
        for row in range(nrows):
            if result[row] is None:
                result[row] = list(global_order)
        return result

    def _candidate(self, values, strings, orders):
        serialized = [
            "".join(strings[row, col] for col in orders[row])
            for row in range(len(orders))
        ]
        score = self._trie_score(serialized)
        row_order = sorted(range(len(orders)), key=lambda row: (serialized[row], row))
        return score, row_order, orders

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
        self.df = df
        nrows, ncols = df.shape
        columns = list(df.columns)

        if nrows == 0:
            return df.copy().astype(object), []
        if ncols == 0:
            return df.copy().astype(object), [[] for _ in range(nrows)]

        values = df.to_numpy(dtype=object, copy=True)
        strings = np.empty((nrows, ncols), dtype=object)
        for row in range(nrows):
            for col in range(ncols):
                strings[row, col] = self._score_string(values[row, col])

        global_pair = self._frequency_order(strings, pair_weight=True)
        global_repeat = self._frequency_order(strings, pair_weight=False)
        global_pair = self._apply_constraints(
            global_pair, columns, col_merge, one_way_dep
        )
        global_repeat = self._apply_constraints(
            global_repeat, columns, col_merge, one_way_dep
        )

        # At most three complete constructions are measured.
        candidates = []
        candidates.append(
            self._candidate(values, strings, [global_pair[:] for _ in range(nrows)])
        )
        candidates.append(
            self._candidate(values, strings, [global_repeat[:] for _ in range(nrows)])
        )

        requested_depth = col_stop if col_stop is not None and col_stop > 0 else 12
        depth_limit = max(1, min(ncols, int(requested_depth), 12))
        conditional = self._conditional_orders(
            strings, global_pair, depth_limit, max(0, int(early_stop or 0))
        )
        conditional = [
            self._apply_constraints(order, columns, col_merge, one_way_dep)
            for order in conditional
        ]
        candidates.append(self._candidate(values, strings, conditional))

        # Stable tie-breaking retains the cheapest first global construction.
        best_score = max(candidate[0] for candidate in candidates)
        best = next(candidate for candidate in candidates
                    if candidate[0] == best_score)
        _, rows, orders = best

        output = np.empty((nrows, ncols), dtype=object)
        output_orders = []
        output_index = []
        for out_row, source_row in enumerate(rows):
            order = orders[source_row]
            output[out_row, :] = values[source_row, order]
            output_orders.append([columns[col] for col in order])
            output_index.append(df.index[source_row])

        reordered = pd.DataFrame(output, columns=df.columns,
                                 index=pd.Index(output_index, name=df.index.name))
        return reordered, output_orders


# EVOLVE-BLOCK-END