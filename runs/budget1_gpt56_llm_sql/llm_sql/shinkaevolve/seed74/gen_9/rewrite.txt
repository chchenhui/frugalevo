# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List
from collections import Counter, defaultdict
import heapq


class Evolved(Algorithm):
    """
    Prefix-cache-oriented row/column layout optimizer.

    Cells are never transformed: only their positions within a row may change.
    ``column_orderings[r]`` describes which original source column supplied
    each physical value in output row ``r``.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @staticmethod
    def _text(value) -> str:
        """Match evaluator-style serialization without changing stored values."""
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
        if left == right:
            return len(left)
        upper = min(len(left), len(right))
        low = 0
        # Slice comparisons run in C and avoid an expensive Python character loop.
        while low < upper:
            middle = (low + upper + 1) // 2
            if left[:middle] == right[:middle]:
                low = middle
            else:
                upper = middle - 1
        return low

    def _trie_score(self, strings: List[str]) -> int:
        if len(strings) < 2:
            return 0
        strings = sorted(strings)
        return sum(self._lcp(strings[i - 1], strings[i])
                   for i in range(1, len(strings)))

    @staticmethod
    def _resolve_column(name, columns):
        """Use exact names first, then the legacy unique-substring convention."""
        for pos, col in enumerate(columns):
            if col == name:
                return pos
        matches = [i for i, col in enumerate(columns) if str(name) in str(col)]
        return matches[0] if len(matches) == 1 else None

    def _constraint_units(self, candidate, columns, col_merge, one_way_dep):
        """
        Make requested merge groups contiguous and satisfy dependency edges by
        stable topological sorting of those groups.  Invalid/ambiguous requests
        are ignored rather than risking data loss.
        """
        m = len(columns)
        groups = []
        used = set()

        for requested in col_merge or []:
            members = []
            for item in requested:
                pos = self._resolve_column(item, columns)
                if pos is not None and pos not in members:
                    members.append(pos)
            # Overlapping groups have no unambiguous block representation.
            if len(members) > 1 and not any(x in used for x in members):
                groups.append(set(members))
                used.update(members)

        units = [set(g) for g in groups]
        units.extend([{i} for i in range(m) if i not in used])
        unit_of = {}
        for unit_id, unit in enumerate(units):
            for col in unit:
                unit_of[col] = unit_id

        # Candidate rank is used for deterministic stable topological ties.
        rank = {col: i for i, col in enumerate(candidate)}
        unit_rank = {
            u: min(rank.get(col, m + col) for col in unit)
            for u, unit in enumerate(units)
        }

        edges = defaultdict(set)
        indegree = [0] * len(units)
        for dep in one_way_dep or []:
            if len(dep) != 2:
                continue
            before = self._resolve_column(dep[0], columns)
            after = self._resolve_column(dep[1], columns)
            if before is None or after is None:
                continue
            src, dst = unit_of[before], unit_of[after]
            if src != dst and dst not in edges[src]:
                edges[src].add(dst)
                indegree[dst] += 1

        available = [(unit_rank[i], i) for i in range(len(units))
                     if indegree[i] == 0]
        heapq.heapify(available)
        ordered_units = []
        while available:
            _, unit = heapq.heappop(available)
            ordered_units.append(unit)
            for nxt in sorted(edges[unit], key=lambda x: unit_rank[x]):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    heapq.heappush(available, (unit_rank[nxt], nxt))

        # Cycles cannot satisfy every requested dependency. Keep their original
        # candidate-relative order while retaining all columns exactly once.
        if len(ordered_units) != len(units):
            remaining = [u for u in range(len(units)) if u not in set(ordered_units)]
            ordered_units.extend(sorted(remaining, key=lambda u: unit_rank[u]))

        output = []
        for unit in ordered_units:
            # Preserve the candidate's preferred order within a merge block.
            output.extend(sorted(units[unit], key=lambda x: rank.get(x, m + x)))
        return output

    def _global_order(self, texts, m):
        scores = []
        for col in range(m):
            counts = Counter(row[col] for row in texts)
            score = sum(len(value) * count * (count - 1)
                        for value, count in counts.items() if count > 1)
            scores.append(score)
        return sorted(range(m), key=lambda c: (-scores[c], c))

    def _conditional_orders(self, texts, global_order):
        """
        Bounded conditional prefix partition tree.  It chooses leading fields
        by length-weighted pair repetition and permits a different suffix order
        in each value partition.
        """
        n = len(texts)
        m = len(global_order)
        result = [None] * n
        candidate_limit = min(24, m)
        depth_limit = min(12, m)

        def finish(rows, prefix, remaining):
            tail = [c for c in global_order if c in remaining]
            for row in rows:
                result[row] = prefix + tail

        def visit(rows, prefix, remaining, depth):
            if not rows or not remaining:
                finish(rows, prefix, remaining)
                return
            if depth >= depth_limit or len(rows) < 2:
                finish(rows, prefix, remaining)
                return

            possible = [c for c in global_order if c in remaining][:candidate_limit]
            best_col = None
            best_score = 0
            for col in possible:
                counts = Counter(texts[row][col] for row in rows)
                score = sum(len(value) * count * (count - 1)
                            for value, count in counts.items() if count > 1)
                if score > best_score:
                    best_score, best_col = score, col

            if best_col is None or best_score <= 0:
                finish(rows, prefix, remaining)
                return

            partitions = defaultdict(list)
            for row in rows:
                partitions[texts[row][best_col]].append(row)
            new_remaining = set(remaining)
            new_remaining.remove(best_col)
            for key in sorted(partitions):
                visit(partitions[key], prefix + [best_col],
                      new_remaining, depth + 1)

        visit(list(range(n)), [], set(range(m)), 0)
        return result

    def _strings_for_orders(self, texts, orders):
        return [
            "".join(texts[row][col] for col in orders[row])
            for row in range(len(texts))
        ]

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
        # Do not add bookkeeping columns: they are values too and can corrupt
        # user data when duplicate indexes or mixed dtypes are present.
        n, m = df.shape
        columns = list(df.columns)
        if n == 0 or m == 0:
            return df.copy(), [[] for _ in range(n)]

        values = df.to_numpy(dtype=object, copy=True)
        texts = [[self._text(values[r, c]) for c in range(m)] for r in range(n)]

        global_order = self._global_order(texts, m)
        identity_order = list(range(m))
        conditional = self._conditional_orders(texts, global_order)

        raw_candidates = [
            [global_order[:] for _ in range(n)],
            conditional,
            [identity_order[:] for _ in range(n)],
        ]

        best_orders = None
        best_score = -1
        for candidate in raw_candidates:
            constrained = [
                self._constraint_units(order, columns, col_merge, one_way_dep)
                for order in candidate
            ]
            # Defensive guarantee for unusual duplicate labels/constraints.
            if any(sorted(order) != list(range(m)) for order in constrained):
                continue
            score = self._trie_score(self._strings_for_orders(texts, constrained))
            if score > best_score:
                best_score = score
                best_orders = constrained

        if best_orders is None:
            best_orders = [identity_order[:] for _ in range(n)]

        output = [[values[row, col] for col in best_orders[row]]
                  for row in range(n)]
        reordered = pd.DataFrame(output, index=df.index.copy(), columns=df.columns.copy(),
                                 dtype=object)
        column_orderings = [
            [columns[col] for col in order] for order in best_orders
        ]
        return reordered, column_orderings

# EVOLVE-BLOCK-END