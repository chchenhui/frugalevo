import pandas as pd
from collections import Counter, defaultdict
from typing import List, Tuple

from solver import Algorithm


class Evolved(Algorithm):
    """Bounded, validity-preserving optimizer for character Trie reuse."""

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.num_rows = 0
        self.num_cols = 0

    @staticmethod
    def _text(value) -> str:
        try:
            missing = pd.isna(value)
            if bool(missing):
                return ""
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        limit = min(len(a), len(b))
        if not limit or a[0] != b[0]:
            return 0
        lo, hi = 1, limit
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    @classmethod
    def _reuse(cls, strings: List[str]) -> int:
        if len(strings) < 2:
            return 0
        strings = sorted(strings)
        total = 0
        previous = strings[0]
        for current in strings[1:]:
            total += len(current) if current == previous else cls._lcp(previous, current)
            previous = current
        return total

    @staticmethod
    def _merge(columns, order, col_merge):
        """Return a full order with requested source-column groups contiguous."""
        if not col_merge:
            return list(order)
        pos = {c: i for i, c in enumerate(order)}
        member_group = {}
        claimed = set()
        for request in col_merge:
            group = [c for c in order if c in request and c not in claimed]
            if group:
                group.sort(key=pos.__getitem__)
                for c in group:
                    member_group[c] = group
                    claimed.add(c)

        out, emitted = [], set()
        for c in order:
            if c in emitted:
                continue
            group = member_group.get(c)
            if group is None:
                out.append(c)
                emitted.add(c)
            else:
                out.extend(group)
                emitted.update(group)
        return out

    @staticmethod
    def _strings(text, orders):
        return ["".join(text[r][j] for j in orders[r]) for r in range(len(orders))]

    def _global_candidates(self, text, columns, col_merge):
        n, m = len(text), len(columns)
        stats = []
        for j in range(m):
            counts = Counter(text[r][j] for r in range(n))
            pair_mass = sum(len(v) * c * (c - 1) for v, c in counts.items())
            repeat_mass = sum(len(v) * (c - 1) for v, c in counts.items())
            stats.append((pair_mass, repeat_mass, len(counts)))

        rankings = [
            sorted(range(m), key=lambda j: (-stats[j][0], -stats[j][1], stats[j][2], j)),
            sorted(range(m), key=lambda j: (-stats[j][1], -stats[j][0], stats[j][2], j)),
        ]
        lookup = {c: j for j, c in enumerate(columns)}
        result = []
        for ranking in rankings:
            names = self._merge(columns, [columns[j] for j in ranking], col_merge)
            result.append([lookup[c] for c in names])
        return result, rankings[0], stats

    def _conditional(self, text, columns, col_merge, global_order, stats):
        """
        Conditional prefix tree.  Crucially, every finish includes every
        remaining column; candidate limits control branching, not output shape.
        """
        n = len(text)
        m = len(columns)
        result = [None] * n
        useful = [j for j in global_order if stats[j][0] > 0][:min(20, m)]
        candidate_set = set(useful)
        max_depth = min(8, len(useful))
        node_limit = 384
        nodes = 0
        lookup = {c: j for j, c in enumerate(columns)}

        def complete(rows, prefix, remaining):
            tail = [j for j in global_order if j in remaining]
            base = prefix + tail
            names = self._merge(columns, [columns[j] for j in base], col_merge)
            order = [lookup[c] for c in names]
            for r in rows:
                result[r] = order[:]

        def visit(rows, remaining, prefix, depth):
            nonlocal nodes
            nodes += 1
            branchable = [j for j in remaining if j in candidate_set]
            if (len(rows) < 2 or not branchable or depth >= max_depth
                    or nodes > node_limit):
                complete(rows, prefix, remaining)
                return

            best_j, best_score, best_groups = -1, 0, None
            for j in branchable:
                groups = defaultdict(list)
                for r in rows:
                    groups[text[r][j]].append(r)
                score = sum(
                    len(value) * len(members) * (len(members) - 1)
                    for value, members in groups.items()
                    if len(members) > 1
                )
                if score > best_score or (score == best_score and score > 0 and
                                          (best_j < 0 or j < best_j)):
                    best_j, best_score, best_groups = j, score, groups

            if best_j < 0 or best_score <= 0:
                complete(rows, prefix, remaining)
                return

            rest = [j for j in remaining if j != best_j]
            groups = sorted(best_groups.values(), key=lambda g: (-len(g), min(g)))
            for group in groups:
                visit(group, rest, prefix + [best_j], depth + 1)

        visit(list(range(n)), list(range(m)), [], 0)
        fallback = list(global_order)
        for r in range(n):
            if result[r] is None:
                result[r] = fallback[:]
        return result

    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: int = None,
        col_stop: int = None,
        col_merge=[],
        one_way_dep=[],
        distinct_value_threshold=0.8,
        parallel=True,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        self.df = df
        self.num_rows, self.num_cols = df.shape
        n, m = df.shape
        columns = list(df.columns)

        if n == 0 or m == 0:
            return df.copy(), [[] for _ in range(n)]

        raw = df.to_numpy(dtype=object, copy=True)
        text = [[self._text(raw[r, j]) for j in range(m)] for r in range(n)]

        global_orders, unmerged_global, stats = self._global_candidates(
            text, columns, col_merge
        )
        candidates = []
        for order in global_orders:
            orders = [order[:] for _ in range(n)]
            serial = self._strings(text, orders)
            candidates.append((self._reuse(serial), orders, serial))

        conditional = self._conditional(
            text, columns, col_merge, unmerged_global, stats
        )
        serial = self._strings(text, conditional)
        candidates.append((self._reuse(serial), conditional, serial))

        _, best_orders, best_serial = max(candidates, key=lambda item: item[0])
        row_order = sorted(range(n), key=lambda r: best_serial[r])

        output = [[raw[r, j] for j in best_orders[r]] for r in row_order]
        orderings = [[columns[j] for j in best_orders[r]] for r in row_order]
        result = pd.DataFrame(
            output,
            columns=columns,
            index=df.index.take(row_order),
            dtype=object,
        )
        return result, orderings