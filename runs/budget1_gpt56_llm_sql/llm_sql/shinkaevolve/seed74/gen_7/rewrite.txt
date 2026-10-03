# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List


class Evolved(Algorithm):
    """
    Prefix-cache-oriented row and column reorderer.

    Returned data contains exactly one copy of every input row and value.
    `column_orderings[i]` identifies the source-column order used to construct
    returned row i.  This permits row-specific physical field orderings.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _score_value(value) -> str:
        """Match evaluator normalization without changing the stored value."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and missing:
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
        # Slice comparison is implemented in C and avoids character Python loops.
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if left[:mid] == right[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _trie_score(self, serialized: List[str]) -> int:
        if len(serialized) < 2:
            return 0
        ordered = sorted(serialized)
        return sum(self._lcp(ordered[i - 1], ordered[i])
                   for i in range(1, len(ordered)))

    @staticmethod
    def _column_names(columns, order):
        return [columns[i] for i in order]

    def _constraint_blocks(self, columns, col_merge, one_way_dep):
        """
        Make merged fields contiguous.  Dependencies are represented as
        predecessor constraints between block representatives.
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

        name_to_positions = {}
        for i, name in enumerate(columns):
            name_to_positions.setdefault(name, []).append(i)

        for group in col_merge or []:
            positions = []
            for name in group:
                positions.extend(name_to_positions.get(name, []))
            for pos in positions[1:]:
                union(positions[0], pos)

        groups = {}
        for i in range(ncols):
            groups.setdefault(find(i), []).append(i)
        blocks = list(groups.values())

        block_of = {}
        for b, block in enumerate(blocks):
            for i in block:
                block_of[i] = b

        edges = set()
        for left_name, right_name in one_way_dep or []:
            # Preserve historical API behavior: exact match first, then one
            # unambiguous substring match.
            left = name_to_positions.get(left_name, [])
            right = name_to_positions.get(right_name, [])
            if len(left) != 1:
                left = [i for i, c in enumerate(columns) if left_name in str(c)]
            if len(right) != 1:
                right = [i for i, c in enumerate(columns) if right_name in str(c)]
            if len(left) == 1 and len(right) == 1:
                a, b = block_of[left[0]], block_of[right[0]]
                if a != b:
                    edges.add((a, b))
        return blocks, edges

    def _legal_global_order(self, field_scores, blocks, edges):
        """Stable topological order of contiguous blocks."""
        block_score = [sum(field_scores[i] for i in block) for block in blocks]
        incoming = [0] * len(blocks)
        outgoing = [[] for _ in blocks]
        for a, b in edges:
            outgoing[a].append(b)
            incoming[b] += 1

        available = [i for i in range(len(blocks)) if incoming[i] == 0]
        chosen = []
        while available:
            available.sort(key=lambda b: (-block_score[b], min(blocks[b])))
            block = available.pop(0)
            chosen.append(block)
            for nxt in outgoing[block]:
                incoming[nxt] -= 1
                if incoming[nxt] == 0:
                    available.append(nxt)

        # Cyclic dependency input cannot satisfy every edge.  Preserve all
        # fields deterministically rather than dropping or fabricating cells.
        if len(chosen) != len(blocks):
            chosen.extend(b for b in range(len(blocks)) if b not in chosen)

        result = []
        for block in chosen:
            result.extend(sorted(blocks[block], key=lambda i: (-field_scores[i], i)))
        return result

    def _conditional_orders(self, codes, lengths, global_order, depth_limit):
        """
        Bounded conditional prefix partitioning.  Every leaf gets the selected
        prefix followed by a deterministic global tail.
        """
        nrows, ncols = codes.shape
        orders = [None] * nrows
        candidate_cap = min(ncols, 40)
        candidate_columns = global_order[:candidate_cap]

        def visit(rows, remaining, prefix, depth):
            if len(rows) == 0:
                return
            if depth >= depth_limit or not remaining or len(rows) < 2:
                tail = [c for c in global_order if c not in prefix]
                final = prefix + tail
                for r in rows:
                    orders[int(r)] = final
                return

            best_col = None
            best_score = 0
            for col in remaining:
                vals = codes[rows, col]
                counts = np.bincount(vals)
                repeated = counts * (counts - 1)
                score = int(np.dot(repeated, lengths[col]))
                if score > best_score or (score == best_score and best_col is not None and col < best_col):
                    best_col, best_score = col, score

            if best_col is None or best_score <= 0:
                tail = [c for c in global_order if c not in prefix]
                final = prefix + tail
                for r in rows:
                    orders[int(r)] = final
                return

            next_prefix = prefix + [best_col]
            next_remaining = [c for c in remaining if c != best_col]
            vals = codes[rows, best_col]
            for code in np.unique(vals):
                visit(rows[vals == code], next_remaining, next_prefix, depth + 1)

        visit(np.arange(nrows, dtype=np.int64), candidate_columns, [], 0)
        return [order if order is not None else list(global_order) for order in orders]

    def _materialize(self, values, columns, row_order, per_row_orders):
        nrows, ncols = values.shape
        output = np.empty((nrows, ncols), dtype=object)
        output_orders = []
        for out_i, source_i in enumerate(row_order):
            field_order = per_row_orders[source_i]
            output[out_i, :] = values[source_i, field_order]
            output_orders.append(self._column_names(columns, field_order))
        # Original labels are position labels for row-specific layouts.
        return pd.DataFrame(output, columns=columns), output_orders

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
        if nrows == 0:
            return df.copy(), []
        if ncols == 0:
            return df.copy(), [[] for _ in range(nrows)]

        # Object arrays preserve heterogeneous source values without coercion.
        values = df.to_numpy(dtype=object, copy=True)
        text = [[self._score_value(values[r, c]) for c in range(ncols)]
                for r in range(nrows)]

        codes = np.empty((nrows, ncols), dtype=np.int64)
        length_tables = []
        field_scores = []
        for c in range(ncols):
            mapping = {}
            code_lengths = []
            next_code = 0
            for r in range(nrows):
                value = text[r][c]
                code = mapping.get(value)
                if code is None:
                    code = next_code
                    mapping[value] = code
                    code_lengths.append(len(value))
                    next_code += 1
                codes[r, c] = code
            lengths = np.asarray(code_lengths, dtype=np.int64)
            counts = np.bincount(codes[:, c], minlength=len(lengths))
            field_scores.append(int(np.dot(lengths, counts * (counts - 1))))
            length_tables.append(lengths)

        blocks, edges = self._constraint_blocks(columns, col_merge, one_way_dep)
        global_order = self._legal_global_order(field_scores, blocks, edges)

        # Candidate 1: original column order, except that explicit constraints
        # require a legal order.
        identity = list(range(ncols))
        constrained = bool(col_merge or one_way_dep)
        candidate_orders = []
        if not constrained:
            candidate_orders.append([identity] * nrows)
        candidate_orders.append([global_order] * nrows)

        # Candidate 3: conditional order only when it can safely honor all
        # constraints.  Merge/dependency APIs require global consistency.
        if not constrained and ncols > 1:
            max_depth = min(ncols, col_stop if col_stop is not None else 12)
            if row_stop is not None:
                max_depth = min(max_depth, max(1, row_stop))
            candidate_orders.append(
                self._conditional_orders(codes, length_tables, global_order, max_depth)
            )

        best_score = None
        best_orders = None
        best_strings = None
        for orders in candidate_orders[:3]:
            serialized = [
                "".join(text[r][c] for c in orders[r])
                for r in range(nrows)
            ]
            score = self._trie_score(serialized)
            if best_score is None or score > best_score:
                best_score, best_orders, best_strings = score, orders, serialized

        # Row insertion order does not affect the ideal Trie score.  Sorting is
        # deterministic and tends to improve practical prefix-cache locality.
        row_order = sorted(range(nrows), key=lambda r: (best_strings[r], r))
        result, orderings = self._materialize(values, columns, row_order, best_orders)

        # A fresh RangeIndex avoids retaining misleading source row positions;
        # row identity is preserved by the one-to-one row permutation.
        result.index = pd.RangeIndex(len(result))
        return result, orderings


# EVOLVE-BLOCK-END