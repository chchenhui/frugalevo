# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List
from collections import Counter
import heapq


class Evolved(Algorithm):
    """
    Prefix-cache oriented column reordering.

    The returned dataframe keeps its original visible column labels.  A row's
    values may be permuted across those positions, and column_orderings[row]
    records which original column supplied each output position.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _text(value) -> str:
        """Match fillna('').astype(str) for ordinary pandas scalar values."""
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
        # Slice comparisons execute the character comparison in C and avoid a
        # Python loop for long common prefixes.
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
    def _find_column(columns, requested):
        """Keep compatibility with the original substring dependency syntax."""
        matches = [i for i, col in enumerate(columns)
                   if str(col) == str(requested)]
        if len(matches) == 1:
            return matches[0]
        matches = [i for i, col in enumerate(columns)
                   if str(requested) in str(col)]
        return matches[0] if len(matches) == 1 else None

    def _blocks_and_dependencies(self, columns, scores, col_merge, one_way_dep):
        """
        Convert requested merge groups into indivisible ordering blocks and
        convert dependencies into block precedence constraints.  This preserves
        dataframe shape; a merge means adjacency rather than concatenating or
        deleting cells.
        """
        ncols = len(columns)
        used = set()
        blocks = []

        for group in col_merge or []:
            positions = []
            for name in group:
                pos = self._find_column(columns, name)
                if pos is not None and pos not in used:
                    positions.append(pos)
                    used.add(pos)
            if positions:
                blocks.append(positions)

        for pos in range(ncols):
            if pos not in used:
                blocks.append([pos])

        col_to_block = {}
        for block_id, block in enumerate(blocks):
            for pos in block:
                col_to_block[pos] = block_id

        edges = {i: set() for i in range(len(blocks))}
        indegree = [0] * len(blocks)
        for before, after in one_way_dep or []:
            a = self._find_column(columns, before)
            b = self._find_column(columns, after)
            if a is None or b is None:
                continue
            ba, bb = col_to_block[a], col_to_block[b]
            if ba != bb and bb not in edges[ba]:
                edges[ba].add(bb)
                indegree[bb] += 1

        block_score = [sum(scores[p] for p in block) for block in blocks]
        return blocks, edges, indegree, block_score

    @staticmethod
    def _topological_order(blocks, edges, indegree, block_score, reverse=False):
        """Highest-value available block first, with deterministic ties."""
        degree = list(indegree)
        ready = []
        for block_id, value in enumerate(block_score):
            if degree[block_id] == 0:
                key = value if reverse else -value
                heapq.heappush(ready, (key, block_id))

        result = []
        while ready:
            _, block_id = heapq.heappop(ready)
            result.extend(blocks[block_id])
            for child in sorted(edges[block_id]):
                degree[child] -= 1
                if degree[child] == 0:
                    key = block_score[child] if reverse else -block_score[child]
                    heapq.heappush(ready, (key, child))

        # Cyclic caller constraints cannot all be satisfied.  Retain every
        # column deterministically rather than dropping any data.
        if len(result) != sum(len(block) for block in blocks):
            seen = set(result)
            for block in blocks:
                for pos in block:
                    if pos not in seen:
                        result.append(pos)
        return result

    def _score_global_order(self, order, texts):
        return self._trie_score(["".join(row[col] for col in order)
                                 for row in texts])

    def _conditional_orders(self, texts, codes, code_lengths, tail_order):
        """
        Bounded conditional prefix partitioning.  It considers only a modest
        number of promising fields at every node and never mutates dataframe
        values.
        """
        nrows = len(texts)
        ncols = len(tail_order)
        if nrows == 0 or ncols == 0:
            return [list(tail_order) for _ in range(nrows)]

        # Wide tables need a bounded node cost.  The global tail order places
        # the most promising fields first, so it is also a useful candidate set.
        candidates = list(tail_order[:min(ncols, 48)])
        max_depth = min(len(candidates), 12)
        answer = [None] * nrows

        def visit(rows, prefix, available, depth):
            if len(rows) < 2 or depth >= max_depth or not available:
                prefix_set = set(prefix)
                order = prefix + [c for c in tail_order if c not in prefix_set]
                for row in rows:
                    answer[row] = order
                return

            best_col = None
            best_value = 0
            # length * count * (count - 1) measures shared leading characters
            # of repeated complete serialized cells.
            for col in available:
                counts = Counter(codes[col][row] for row in rows)
                value = 0
                for code, count in counts.items():
                    if count > 1:
                        value += code_lengths[col][code] * count * (count - 1)
                if value > best_value or (value == best_value and
                                          best_col is not None and col < best_col):
                    best_value = value
                    best_col = col

            if best_col is None or best_value <= 0:
                prefix_set = set(prefix)
                order = prefix + [c for c in tail_order if c not in prefix_set]
                for row in rows:
                    answer[row] = order
                return

            partitions = {}
            for row in rows:
                partitions.setdefault(codes[best_col][row], []).append(row)
            next_available = [c for c in available if c != best_col]
            for key in sorted(partitions):
                visit(partitions[key], prefix + [best_col],
                      next_available, depth + 1)

        visit(list(range(nrows)), [], candidates, 0)
        return answer

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
        # This implementation intentionally does not use early_stop/parallel:
        # deterministic bounded work is safer than recursively changing data.
        nrows, ncols = df.shape
        columns = list(df.columns)

        if ncols == 0:
            return df.copy(), [[] for _ in range(nrows)]
        if nrows == 0:
            return df.copy(), []

        # Materialize scoring text once.  Values themselves remain in df and
        # are copied directly to the final object dataframe.
        values = [[df.iloc[row, col] for col in range(ncols)]
                  for row in range(nrows)]
        texts = [[self._text(values[row][col]) for col in range(ncols)]
                 for row in range(nrows)]

        codes = []
        code_lengths = []
        frequency_scores = []
        length_scores = []
        for col in range(ncols):
            mapping = {}
            col_codes = []
            lengths = []
            for row in range(nrows):
                text = texts[row][col]
                code = mapping.get(text)
                if code is None:
                    code = len(mapping)
                    mapping[text] = code
                    lengths.append(len(text))
                col_codes.append(code)
            counts = Counter(col_codes)
            frequency_scores.append(sum(
                lengths[code] * count * (count - 1)
                for code, count in counts.items()
            ))
            length_scores.append(sum(
                lengths[code] * count
                for code, count in counts.items()
            ))
            codes.append(col_codes)
            code_lengths.append(lengths)

        blocks, edges, indegree, block_scores = self._blocks_and_dependencies(
            columns, frequency_scores, col_merge, one_way_dep
        )
        global_order = self._topological_order(
            blocks, edges, indegree, block_scores, reverse=False
        )

        # A second inexpensive global hypothesis emphasizes long fields rather
        # than pair repetition.  Both are scored using real serialized strings.
        length_block_scores = [
            sum(length_scores[pos] for pos in block) for block in blocks
        ]
        length_order = self._topological_order(
            blocks, edges, indegree, length_block_scores, reverse=False
        )

        best_kind = "global"
        best_order = global_order
        best_score = self._score_global_order(global_order, texts)

        alternate_score = self._score_global_order(length_order, texts)
        if alternate_score > best_score:
            best_kind = "global"
            best_order = length_order
            best_score = alternate_score

        # Conditional orders cannot safely preserve indivisible merge blocks or
        # dependency precedence, so use them only when no such constraints were
        # supplied.
        if not col_merge and not one_way_dep:
            conditional = self._conditional_orders(
                texts, codes, code_lengths, global_order
            )
            conditional_score = self._trie_score([
                "".join(texts[row][col] for col in conditional[row])
                for row in range(nrows)
            ])
            if conditional_score > best_score:
                best_kind = "conditional"
                best_order = conditional
                best_score = conditional_score

        if best_kind == "global":
            numeric_orders = [list(best_order) for _ in range(nrows)]
        else:
            numeric_orders = best_order

        # Object dtype prevents mixed input values from being coerced while
        # retaining exact scalar objects and all original row indexes.
        output_values = [
            [values[row][source_col] for source_col in numeric_orders[row]]
            for row in range(nrows)
        ]
        reordered = pd.DataFrame(output_values, index=df.index,
                                 columns=df.columns, dtype=object)
        column_orderings = [
            [columns[source_col] for source_col in order]
            for order in numeric_orders
        ]
        return reordered, column_orderings

# EVOLVE-BLOCK-END