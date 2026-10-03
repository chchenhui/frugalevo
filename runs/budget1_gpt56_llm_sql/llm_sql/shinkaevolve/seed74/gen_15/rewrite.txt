# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from functools import lru_cache
from collections import Counter, defaultdict


class Evolved(Algorithm):
    """
    Prefix-cache oriented reordering algorithm.

    The returned frame contains the source values in the physical order
    described by the corresponding entry in column_orderings.  This permits
    row-specific field orders while preserving every source value and row.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.dep_graph = None
        self.num_rows = 0
        self.num_cols = 0
        self.column_stats = None
        self.val_len = None
        self.row_stop = None
        self.col_stop = None
        self.base = 2000

    @staticmethod
    def _text_value(value) -> str:
        """Evaluator-compatible scoring text without changing stored values."""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    def find_max_group_value(self, df: pd.DataFrame, value_counts: Dict,
                             early_stop: int = 0) -> str:
        if not value_counts:
            return None
        best_value = None
        best_score = early_stop - 1
        for value, count in value_counts.items():
            length = len(self._text_value(value))
            score = length * max(0, count - 1)
            if score > best_score:
                best_value = value
                best_score = score
        return best_value

    def reorder_columns_for_value(self, row, value, column_names,
                                  grouped_rows_len: int = 1):
        order = []
        rest = []
        for pos, col in enumerate(column_names):
            try:
                cell = row[pos]
            except (IndexError, KeyError, TypeError):
                cell = getattr(row, str(col), None)
            if cell == value:
                order.append(col)
            else:
                rest.append(col)
        return list(row), order + rest

    def get_dependent_columns(self, col: str) -> List[str]:
        if not self.dep_graph:
            return []
        return list(self.dep_graph.get(col, ()))

    @lru_cache(maxsize=None)
    def get_cached_dependent_columns(self, col: str) -> List[str]:
        return self.get_dependent_columns(col)

    def fixed_reorder(self, df: pd.DataFrame,
                      row_sort: bool = True) -> Tuple[pd.DataFrame, List[List[str]]]:
        # A deterministic, safe global order for compatibility with callers of
        # the old helper.  It never adds helper columns or changes values.
        columns = list(df.columns)
        texts = [[self._text_value(df.iloc[r, c]) for r in range(len(df))]
                 for c in range(len(columns))]
        scores = []
        for c, values in enumerate(texts):
            counts = Counter(values)
            scores.append(sum(len(v) * n * (n - 1) for v, n in counts.items()))
        order = sorted(range(len(columns)), key=lambda c: (-scores[c], c))
        result = df.iloc[:, order].copy().astype(object)
        names = [columns[c] for c in order]
        return result, [names[:] for _ in range(len(result))]

    # Compatibility helpers retained with their historical public signatures.
    # The safe implementation does not use the old lossy recursive machinery.
    def column_recursion(self, result_df, max_value, grouped_rows,
                         row_stop, col_stop, early_stop):
        return result_df, Counter()

    def recursive_reorder(self, df: pd.DataFrame, value_counts: Dict,
                          early_stop: int = 0, original_columns: List[str] = None,
                          row_stop: int = 0, col_stop: int = 0):
        return self.fixed_reorder(df, row_sort=False)

    def recursive_split_and_reorder(self, df: pd.DataFrame,
                                    original_columns: List[str] = None,
                                    early_stop: int = 0):
        return self.fixed_reorder(df, row_sort=False)[0]

    @lru_cache(maxsize=None)
    def calculate_length(self, value):
        return len(self._text_value(value))

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        if left == right:
            return len(left)
        limit = min(len(left), len(right))
        low, high = 0, limit
        while low < high:
            mid = (low + high + 1) // 2
            if left[:mid] == right[:mid]:
                low = mid
            else:
                high = mid - 1
        return low

    def _trie_score(self, strings: List[str]) -> int:
        if len(strings) < 2:
            return 0
        ordered = sorted(strings)
        return sum(self._lcp(ordered[i - 1], ordered[i])
                   for i in range(1, len(ordered)))

    @staticmethod
    def _factorize(values: List[str]):
        lookup = {}
        codes = []
        counts = []
        for value in values:
            code = lookup.get(value)
            if code is None:
                code = len(counts)
                lookup[value] = code
                counts.append(0)
            codes.append(code)
            counts[code] += 1
        return codes, counts

    def _apply_constraints(self, order, columns, col_merge, one_way_dep):
        """Keep requested merge groups contiguous and dependencies ordered."""
        if not order:
            return order

        # Merge groups are represented as contiguous blocks, preserving the
        # candidate's relative ranking inside each block.
        position = {c: i for i, c in enumerate(order)}
        blocks = []
        used = set()
        for group in col_merge or []:
            members = [columns.index(name) for name in group if name in columns]
            members = [c for c in members if c not in used]
            if len(members) > 1:
                members.sort(key=lambda c: position[c])
                blocks.append(set(members))
                used.update(members)

        if blocks:
            member_to_block = {}
            for block_id, block in enumerate(blocks):
                for member in block:
                    member_to_block[member] = block_id
            rebuilt = []
            emitted = set()
            for col in order:
                block_id = member_to_block.get(col)
                if block_id is None:
                    rebuilt.append(col)
                elif block_id not in emitted:
                    rebuilt.extend(sorted(blocks[block_id], key=lambda c: position[c]))
                    emitted.add(block_id)
            order = rebuilt

        # The historical API resolves dependency names by substring matching.
        # Move a dependency predecessor before its successor deterministically.
        for source_name, target_name in one_way_dep or []:
            sources = [i for i, name in enumerate(columns) if source_name in str(name)]
            targets = [i for i, name in enumerate(columns) if target_name in str(name)]
            if len(sources) != 1 or len(targets) != 1:
                continue
            source, target = sources[0], targets[0]
            source_pos, target_pos = order.index(source), order.index(target)
            if source_pos > target_pos:
                order.pop(source_pos)
                target_pos = order.index(target)
                order.insert(target_pos, source)
        return order

    def _conditional_orders(self, codes, texts, global_order, depth_limit,
                            early_stop):
        rows = len(texts[0]) if texts else 0
        cols = len(texts)
        orders = [[] for _ in range(rows)]
        node_budget = 128
        nodes_seen = [0]
        candidate_columns = global_order[:min(cols, 48)]

        def visit(indices, remaining, depth):
            if not indices:
                return
            if (depth >= depth_limit or len(remaining) <= 1 or
                    len(indices) < 2 or nodes_seen[0] >= node_budget):
                tail = [c for c in global_order if c in remaining]
                for r in indices:
                    orders[r].extend(tail)
                return

            nodes_seen[0] += 1
            best_col = None
            best_gain = early_stop - 1
            for col in candidate_columns:
                if col not in remaining:
                    continue
                local_counts = Counter(codes[col][r] for r in indices)
                gain = 0
                for code, count in local_counts.items():
                    if count > 1:
                        value = texts[col][indices[0]]
                        # Find one representative for this code only when it
                        # repeats; this bounded lookup avoids pandas work.
                        for r in indices:
                            if codes[col][r] == code:
                                value = texts[col][r]
                                break
                        gain += len(value) * count * (count - 1)
                if gain > best_gain or (gain == best_gain and
                                        best_col is not None and col < best_col):
                    best_col, best_gain = col, gain

            if best_col is None or best_gain <= 0:
                tail = [c for c in global_order if c in remaining]
                for r in indices:
                    orders[r].extend(tail)
                return

            buckets = defaultdict(list)
            for r in indices:
                orders[r].append(best_col)
                buckets[codes[best_col][r]].append(r)
            next_remaining = [c for c in remaining if c != best_col]
            for bucket in buckets.values():
                visit(bucket, next_remaining, depth + 1)

        visit(list(range(rows)), list(range(cols)), 0)
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
        # Scoring representations are separate from the stored source data.
        nrows, ncols = df.shape
        columns = list(df.columns)
        self.num_rows, self.num_cols = nrows, ncols

        if ncols == 0:
            return df.copy().astype(object), [[] for _ in range(nrows)]
        if nrows == 0:
            return df.copy().astype(object), []

        texts = []
        codes = []
        field_scores = []
        for col in range(ncols):
            values = [self._text_value(df.iloc[row, col]) for row in range(nrows)]
            col_codes, counts = self._factorize(values)
            texts.append(values)
            codes.append(col_codes)
            field_scores.append(
                sum(len(value) * count * (count - 1)
                    for value, count in Counter(values).items())
            )

        global_order = sorted(range(ncols), key=lambda c: (-field_scores[c], c))
        global_order = self._apply_constraints(
            global_order, columns, col_merge, one_way_dep
        )

        # A second global hypothesis favors long repeated prefixes but uses
        # distinctness as a deterministic tie-breaking tradeoff.
        diversity = [len(set(codes[c])) for c in range(ncols)]
        length_order = sorted(
            range(ncols),
            key=lambda c: (-(field_scores[c] * 2 + sum(map(len, texts[c]))),
                           diversity[c], c),
        )
        length_order = self._apply_constraints(
            length_order, columns, col_merge, one_way_dep
        )

        candidate_orders = [
            [global_order[:] for _ in range(nrows)],
            [length_order[:] for _ in range(nrows)],
        ]

        depth_limit = min(ncols, col_stop if col_stop is not None else 10)
        if nrows > 1 and depth_limit > 0 and (row_stop is None or row_stop > 1):
            conditional = self._conditional_orders(
                codes, texts, global_order, depth_limit, max(0, early_stop)
            )
            conditional = [
                self._apply_constraints(order, columns, col_merge, one_way_dep)
                for order in conditional
            ]
            candidate_orders.append(conditional)

        best_orders = candidate_orders[0]
        best_score = -1
        for orders in candidate_orders:
            serialized = [
                "".join(texts[col][row] for col in orders[row])
                for row in range(nrows)
            ]
            score = self._trie_score(serialized)
            if score > best_score:
                best_score = score
                best_orders = orders

        # Output slot j for a row contains the source value for the column
        # named by column_orderings[row][j].  Object dtype prevents mixed-type
        # coercion and leaves all source values unchanged.
        output_values = [
            [df.iloc[row, col] for col in best_orders[row]]
            for row in range(nrows)
        ]
        result = pd.DataFrame(output_values, columns=columns, index=df.index,
                              dtype=object)
        column_orderings = [
            [columns[col] for col in order] for order in best_orders
        ]
        return result, column_orderings

# EVOLVE-BLOCK-END