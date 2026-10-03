# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from collections import Counter, defaultdict
from functools import lru_cache


class Evolved(Algorithm):
    """
    Prefix-cache-oriented row field reorderer.

    The returned dataframe stores each row's values in the order described by
    the corresponding entry in column_orderings.  Values are never changed,
    fabricated, dropped, or merged.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _serialized_value(value) -> str:
        """Match the evaluator's fillna('').astype(str) representation."""
        if value is None or value is pd.NA or value is pd.NaT:
            return ""

        try:
            missing = pd.isna(value)
            # pd.isna on scalar values returns a scalar; containers return an
            # array/Series and should retain their normal string conversion.
            if not hasattr(missing, "__len__") and bool(missing):
                return ""
        except (TypeError, ValueError):
            pass

        return str(value)

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """C-backed slice comparisons avoid Python character-by-character work."""
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

    def _trie_reuse_score(self, strings: List[str]) -> int:
        if len(strings) < 2:
            return 0

        ordered = sorted(strings)
        return sum(self._lcp(ordered[i - 1], ordered[i]) for i in range(1, len(ordered)))

    @staticmethod
    def _resolve_column(name, columns: List) -> List[int]:
        """Resolve exact names first, then preserve the old substring behavior."""
        exact = [i for i, col in enumerate(columns) if col == name]
        if exact:
            return exact
        return [i for i, col in enumerate(columns) if str(name) in str(col)]

    def _constraint_blocks(
        self,
        num_cols: int,
        columns: List,
        col_merge: List[List[str]],
        one_way_dep: List[Tuple[str, str]],
        weights: List[int],
    ) -> List[int]:
        """
        Produce one deterministic global order.  Merge groups are represented
        as contiguous blocks; dependencies are respected where resolvable.
        """
        parent = list(range(num_cols))

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
            positions = []
            for name in group:
                positions.extend(self._resolve_column(name, columns))
            positions = sorted(set(positions))
            for pos in positions[1:]:
                union(positions[0], pos)

        grouped = defaultdict(list)
        for pos in range(num_cols):
            grouped[find(pos)].append(pos)

        blocks = list(grouped.values())
        for block in blocks:
            block.sort(key=lambda pos: (-weights[pos], pos))

        block_of = {}
        for block_id, block in enumerate(blocks):
            for pos in block:
                block_of[pos] = block_id

        edges = set()
        for source_name, target_name in one_way_dep or []:
            sources = self._resolve_column(source_name, columns)
            targets = self._resolve_column(target_name, columns)
            if len(sources) == 1 and len(targets) == 1:
                source_block = block_of[sources[0]]
                target_block = block_of[targets[0]]
                if source_block != target_block:
                    edges.add((source_block, target_block))

        incoming = [0] * len(blocks)
        outgoing = defaultdict(list)
        for source, target in edges:
            outgoing[source].append(target)
            incoming[target] += 1

        block_weight = [sum(weights[pos] for pos in block) for block in blocks]
        available = [i for i in range(len(blocks)) if incoming[i] == 0]
        block_order = []

        while available:
            available.sort(key=lambda block_id: (-block_weight[block_id], min(blocks[block_id])))
            current = available.pop(0)
            block_order.append(current)
            for target in outgoing[current]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    available.append(target)

        # Cycles cannot be simultaneously satisfied.  Retain a stable order
        # for the unresolved portion instead of dropping columns.
        if len(block_order) != len(blocks):
            remaining = [i for i in range(len(blocks)) if i not in set(block_order)]
            remaining.sort(key=lambda block_id: (-block_weight[block_id], min(blocks[block_id])))
            block_order.extend(remaining)

        result = []
        for block_id in block_order:
            result.extend(blocks[block_id])
        return result

    def _conditional_orders(
        self,
        codes: List[List[int]],
        code_lengths: List[List[int]],
        global_order: List[int],
        max_depth: int,
    ) -> List[List[int]]:
        """
        Build a bounded conditional prefix partition tree.  This is only used
        when there are no explicit cross-column constraints, because a single
        global constrained order is safer for required merge/dependency rules.
        """
        row_count = len(codes[0]) if codes else 0
        col_count = len(codes)
        orders = [[] for _ in range(row_count)]

        if row_count == 0 or col_count == 0:
            return orders

        queue = [(list(range(row_count)), tuple(range(col_count)), 0)]
        processed_nodes = 0
        max_nodes = 256

        while queue and processed_nodes < max_nodes:
            rows, remaining, depth = queue.pop(0)
            processed_nodes += 1

            if len(rows) <= 1 or not remaining or depth >= max_depth:
                tail = [col for col in global_order if col in remaining]
                for row in rows:
                    orders[row].extend(tail)
                continue

            candidate_columns = [col for col in global_order if col in remaining][:24]
            best_col = None
            best_score = 0

            for col in candidate_columns:
                counts = Counter(codes[col][row] for row in rows)
                score = 0
                for code, count in counts.items():
                    if count > 1:
                        score += code_lengths[col][code] * count * (count - 1)

                if score > best_score or (score == best_score and best_col is not None and col < best_col):
                    best_col = col
                    best_score = score

            if best_col is None or best_score <= 0:
                tail = [col for col in global_order if col in remaining]
                for row in rows:
                    orders[row].extend(tail)
                continue

            partitions = defaultdict(list)
            for row in rows:
                orders[row].append(best_col)
                partitions[codes[best_col][row]].append(row)

            next_remaining = tuple(col for col in remaining if col != best_col)
            for partition_rows in partitions.values():
                queue.append((partition_rows, next_remaining, depth + 1))

        # If the node limit was reached, finish every incomplete row cheaply.
        for row in range(row_count):
            if len(orders[row]) < col_count:
                used = set(orders[row])
                orders[row].extend(col for col in global_order if col not in used)

        return orders

    def _strings_for_orders(self, cell_strings: List[List[str]], orders: List[List[int]]) -> List[str]:
        return [
            "".join(cell_strings[row][col] for col in order)
            for row, order in enumerate(orders)
        ]

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True) -> Tuple[pd.DataFrame, List[List[str]]]:
        """Compatibility method: retain all values using a deterministic order."""
        order = list(range(df.shape[1]))
        output = pd.DataFrame(df.astype(object).to_numpy(copy=True), index=df.index, columns=df.columns, dtype=object)
        names = list(df.columns)
        return output, [names[:] for _ in range(len(df))]

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
        # No helper columns are inserted: row index remains the caller's row
        # identity and all source values remain present exactly once.
        row_count, col_count = df.shape
        columns = list(df.columns)

        if row_count == 0 or col_count == 0:
            output = pd.DataFrame(df.astype(object).to_numpy(copy=True), index=df.index, columns=df.columns, dtype=object)
            return output, [[] for _ in range(row_count)]

        raw_values = df.astype(object).to_numpy(copy=True)
        cell_strings = [
            [self._serialized_value(raw_values[row][col]) for col in range(col_count)]
            for row in range(row_count)
        ]

        # Factorize each field once and calculate the requested global
        # length-weighted pair repetition heuristic.
        codes = []
        code_lengths = []
        weights = []

        for col in range(col_count):
            value_to_code = {}
            lengths = []
            col_codes = []
            counts = []

            for row in range(row_count):
                value = cell_strings[row][col]
                code = value_to_code.get(value)
                if code is None:
                    code = len(lengths)
                    value_to_code[value] = code
                    lengths.append(len(value))
                    counts.append(0)
                col_codes.append(code)
                counts[code] += 1

            codes.append(col_codes)
            code_lengths.append(lengths)
            weights.append(sum(lengths[code] * count * (count - 1) for code, count in enumerate(counts)))

        original_order = list(range(col_count))
        global_order = self._constraint_blocks(
            col_count, columns, col_merge or [], one_way_dep or [], weights
        )

        candidates = [
            [original_order[:] for _ in range(row_count)],
            [global_order[:] for _ in range(row_count)],
        ]

        # A conditional candidate can safely use row-specific order only when
        # no explicit ordering constraints were supplied.
        if not col_merge and not one_way_dep and col_count > 1:
            depth_limit = col_count
            if col_stop is not None and col_stop > 0:
                depth_limit = min(depth_limit, col_stop)
            depth_limit = min(depth_limit, 16)
            candidates.append(self._conditional_orders(codes, code_lengths, global_order, depth_limit))

        best_orders = candidates[0]
        best_score = -1

        # At most three whole-program candidates are scored.
        for candidate in candidates:
            strings = self._strings_for_orders(cell_strings, candidate)
            score = self._trie_reuse_score(strings)
            if score > best_score:
                best_score = score
                best_orders = candidate

        output_rows = [
            [raw_values[row][col] for col in best_orders[row]]
            for row in range(row_count)
        ]

        # Positional output columns are retained; column_orderings states which
        # original field occupies each position for every returned row.
        output = pd.DataFrame(output_rows, index=df.index, columns=df.columns, dtype=object)
        column_orderings = [
            [columns[col] for col in order]
            for order in best_orders
        ]

        assert output.shape == df.shape
        assert len(column_orderings) == len(output)
        return output, column_orderings

# EVOLVE-BLOCK-END