# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict
from functools import lru_cache
from collections import Counter, defaultdict


class Evolved(Algorithm):
    """
    Prefix-cache-oriented row serialization reorderer.

    The dataframe values are never changed or fabricated.  A column ordering
    describes which source column occupies each physical output position for a
    row.  For a global ordering the dataframe also retains the corresponding
    reordered column labels.  For row-specific orderings, positional columns
    retain the original labels and ``column_orderings`` provides the source
    column for every position.
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
    def _score_string(value) -> str:
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
        # Slice comparisons execute in C and avoid a Python character loop.
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

    @lru_cache(maxsize=8192)
    def calculate_length(self, value):
        return len(self._score_string(value))

    def find_max_group_value(self, df: pd.DataFrame, value_counts: Dict,
                             early_stop: int = 0):
        best_value = None
        best_score = early_stop
        for value, count in value_counts.items():
            score = len(self._score_string(value)) * max(0, count - 1)
            if score > best_score:
                best_value, best_score = value, score
        return best_value

    def get_dependent_columns(self, col: str) -> List[str]:
        return []

    @lru_cache(maxsize=1024)
    def get_cached_dependent_columns(self, col: str) -> List[str]:
        return self.get_dependent_columns(col)

    def reorder_columns_for_value(self, row, value, column_names,
                                  grouped_rows_len: int = 1):
        order = [c for c in column_names if row[c] == value]
        order.extend(c for c in column_names if row[c] != value)
        return [row[c] for c in order], order

    def calculate_col_stats(self, df: pd.DataFrame, enable_index: bool = True):
        stats = []
        for col in df.columns:
            vals = [self._score_string(v) for v in df[col].tolist()]
            counts = Counter(vals)
            groups = len(counts)
            avg_len = (sum(map(len, vals)) / len(vals)) if vals else 0.0
            score = sum(len(v) * n * max(0, n - 1)
                        for v, n in counts.items())
            stats.append((col, groups, avg_len, score))
        stats.sort(key=lambda x: (-x[3], str(x[0])))
        return len(df), stats

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True):
        _, stats = self.calculate_col_stats(df)
        order = [x[0] for x in stats]
        out = df.loc[:, order].copy()
        if row_sort and len(out):
            serial = ["".join(self._score_string(v) for v in row)
                      for row in out.to_numpy(dtype=object)]
            positions = sorted(range(len(out)), key=lambda i: serial[i])
            out = out.iloc[positions].copy()
        return out, [order[:] for _ in range(len(out))]

    # Compatibility entry points retained for callers of the previous class.
    def column_recursion(self, result_df, max_value, grouped_rows, row_stop,
                         col_stop, early_stop):
        out, _ = self.fixed_reorder(grouped_rows, row_sort=False)
        return out, Counter()

    def recursive_reorder(self, df, value_counts=None, early_stop=0,
                          original_columns=None, row_stop=0, col_stop=0):
        return self.fixed_reorder(df)

    def recursive_split_and_reorder(self, df, original_columns=None,
                                    early_stop=0):
        return self.fixed_reorder(df)[0]

    def _units_and_dependencies(self, columns, col_merge, one_way_dep):
        """Build contiguous merge units and conservative dependency edges."""
        used = set()
        units = []
        for group in col_merge or []:
            members = [c for c in group if c in columns and c not in used]
            if members:
                units.append(members)
                used.update(members)
        for col in columns:
            if col not in used:
                units.append([col])
                used.add(col)

        col_unit = {}
        for unit_id, unit in enumerate(units):
            for col in unit:
                col_unit[col] = unit_id

        prerequisites = [set() for _ in units]
        for before_hint, after_hint in one_way_dep or []:
            before = [c for c in columns if before_hint in str(c)]
            after = [c for c in columns if after_hint in str(c)]
            if len(before) == 1 and len(after) == 1:
                a, b = col_unit[before[0]], col_unit[after[0]]
                if a != b:
                    prerequisites[b].add(a)
        return units, prerequisites

    def _global_unit_order(self, units, prerequisites, strings):
        nrows = len(strings)
        unit_score = []
        for unit in units:
            score = 0
            for col in unit:
                counts = Counter(strings[r][col] for r in range(nrows))
                score += sum(len(value) * count * (count - 1)
                             for value, count in counts.items())
            unit_score.append(score)

        remaining = set(range(len(units)))
        chosen = []
        while remaining:
            ready = [u for u in remaining if prerequisites[u].issubset(chosen)]
            # Cyclic constraints are handled deterministically rather than
            # dropping a column.
            if not ready:
                ready = list(remaining)
            pick = min(ready, key=lambda u: (-unit_score[u], u))
            chosen.append(pick)
            remaining.remove(pick)
        return chosen

    def _conditional_orders(self, strings, units, prerequisites, global_units):
        nrows = len(strings)
        orders = [None] * nrows
        max_depth = min(8, len(units))
        candidate_limit = min(24, len(units))

        def finish(rows, prefix, remaining):
            tail = [u for u in global_units if u in remaining]
            flat = prefix[:]
            for unit_id in tail:
                flat.extend(units[unit_id])
            for row in rows:
                orders[row] = flat[:]

        def visit(rows, prefix, remaining, depth):
            if (len(rows) < 2 or not remaining or depth >= max_depth or
                    len(rows) > 50000):
                finish(rows, prefix, remaining)
                return

            completed = set()
            for unit_id, unit in enumerate(units):
                if unit and unit[0] in prefix:
                    completed.add(unit_id)
            ready = [u for u in remaining
                     if prerequisites[u].issubset(completed)]
            if not ready:
                finish(rows, prefix, remaining)
                return

            # Global ranking bounds work on very wide inputs.
            ready = [u for u in global_units if u in ready][:candidate_limit]
            best_unit = None
            best_score = 0
            best_groups = None
            for unit_id in ready:
                groups = defaultdict(list)
                unit = units[unit_id]
                for row in rows:
                    groups[tuple(strings[row][c] for c in unit)].append(row)
                score = sum(
                    sum(len(part) for part in key) * len(group) * (len(group) - 1)
                    for key, group in groups.items() if len(group) > 1
                )
                if score > best_score:
                    best_unit, best_score, best_groups = unit_id, score, groups

            if best_unit is None or best_score <= 0:
                finish(rows, prefix, remaining)
                return

            next_prefix = prefix + units[best_unit]
            next_remaining = set(remaining)
            next_remaining.remove(best_unit)
            for group in best_groups.values():
                visit(group, next_prefix, next_remaining, depth + 1)

        visit(list(range(nrows)), [], set(range(len(units))), 0)
        fallback = [c for unit_id in global_units for c in units[unit_id]]
        return [order if order is not None else fallback[:] for order in orders]

    def _serialized_for_orders(self, strings, orders):
        return ["".join(strings[row][col] for col in order)
                for row, order in enumerate(orders)]

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
        self.num_rows, self.num_cols = df.shape
        columns = list(df.columns)
        nrows = len(df)

        if not columns:
            return df.copy(), [[] for _ in range(nrows)]

        values = df.to_numpy(dtype=object, copy=True)
        strings = [[self._score_string(values[r, c])
                    for c in range(len(columns))]
                   for r in range(nrows)]

        units, prerequisites = self._units_and_dependencies(
            columns, col_merge, one_way_dep
        )
        global_units = self._global_unit_order(units, prerequisites, strings)
        global_order = [c for unit_id in global_units for c in units[unit_id]]
        identity_order = columns[:]

        candidates = [
            [identity_order[:] for _ in range(nrows)],
            [global_order[:] for _ in range(nrows)],
        ]
        if nrows > 1 and len(units) > 1:
            candidates.append(self._conditional_orders(
                strings, units, prerequisites, global_units
            ))

        best_orders = candidates[0]
        best_score = self._trie_score(
            self._serialized_for_orders(strings, best_orders)
        )
        for candidate in candidates[1:]:
            score = self._trie_score(self._serialized_for_orders(strings, candidate))
            if score > best_score:
                best_score, best_orders = score, candidate

        # A uniform order can preserve meaningful dataframe column labels.
        if all(order == best_orders[0] for order in best_orders):
            order = best_orders[0]
            out = df.loc[:, order].copy()
            return out, [order[:] for _ in range(nrows)]

        # Row-specific layout: values remain exact source objects.  The
        # accompanying order list maps each positional output cell to source
        # column identity.
        positions = {column: index for index, column in enumerate(columns)}
        output_values = [
            [values[row, positions[column]] for column in order]
            for row, order in enumerate(best_orders)
        ]
        out = pd.DataFrame(output_values, index=df.index, columns=columns,
                           dtype=object)
        return out, best_orders


# EVOLVE-BLOCK-END