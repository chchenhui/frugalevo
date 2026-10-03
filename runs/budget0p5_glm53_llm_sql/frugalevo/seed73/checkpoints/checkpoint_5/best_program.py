# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List


def _lcp(a: str, b: str) -> int:
    """Length of longest common prefix via binary search on C-level slice equality."""
    if a == b:
        return len(a)
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _trie_score(row_strings: List[str]) -> int:
    """Exact serial-Trie reuse: sum of adjacent LCPs over lexicographically sorted
    unique strings. For a fixed multiset of row strings this is insertion-order
    independent, so it is a valid deterministic internal objective."""
    if len(row_strings) < 2:
        return 0
    uniq = sorted(set(row_strings))
    total = 0
    for i in range(1, len(uniq)):
        total += _lcp(uniq[i - 1], uniq[i])
    return total


class Evolved(Algorithm):
    """
    Bounded construction selection: three cheap candidate row-field orderings —
    (A) global frequency ranking by sum len(v)*cnt(v)*(cnt(v)-1),
    (B) an alternative length/frequency tradeoff, and
    (C) a bounded recursive conditional prefix-partition tree producing
    per-row column orderings — each scored exactly once with the exact
    sorted-string LCP (serial-Trie) objective; the best is returned.
    Serialization is scoring-only normalization; stored cell values, rows and
    columns are never altered.
    """

    MAX_DEPTH = 3   # bounded tree depth for the conditional partition
    MAX_CAND = 6    # candidate columns examined per tree node

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ---------- serialization ----------

    def _serialize(self, df: pd.DataFrame) -> np.ndarray:
        """Serialize each cell exactly as the evaluator does (fillna('') then str).
        Scoring-only normalization; stored values are never modified."""
        n, m = df.shape
        S = np.empty((n, m), dtype=object)
        for j, c in enumerate(df.columns):
            col = df[c]
            if col.dtype == object:
                S[:, j] = [
                    ("" if v is None or v is pd.NA or (isinstance(v, float) and v != v) else str(v))
                    for v in col.values
                ]
            else:
                S[:, j] = col.fillna("").astype(str).values
        return S

    # ---------- factorization ----------

    def _factorize(self, S: np.ndarray):
        """Factorize each serialized column once. Returns (codes, lens, gscores):
        codes[i, j] is an int code for the serialized value in cell (i, j);
        lens[j] holds string lengths of each unique value of column j;
        gscores[j] = sum over values of len(v)*cnt(v)*(cnt(v)-1)."""
        n, m = S.shape
        codes = np.empty((n, m), dtype=np.int64)
        lens = []
        gscores = []
        for j in range(m):
            cj, uniq = pd.factorize(S[:, j], sort=False)
            codes[:, j] = cj
            lj = np.array([len(u) for u in uniq], dtype=np.int64)
            lens.append(lj)
            cnt = np.bincount(cj, minlength=len(uniq)).astype(np.int64)
            gscores.append(int(np.sum(lj * cnt * (cnt - 1))))
        return codes, lens, gscores

    # ---------- candidate constructions ----------

    def _global_order(self, gscores, m) -> List[int]:
        """Columns ranked descending by length-weighted pair repetition with
        deterministic tie-break on column index."""
        return sorted(range(m), key=lambda j: (-gscores[j], j))

    def _alt_order(self, gscores, lens, m) -> List[int]:
        """Alternative tradeoff: pair-repetition normalized by minimum value
        length, so columns whose repetitions are long per distinct value lead."""
        return sorted(
            range(m),
            key=lambda j: (-(gscores[j] // max(1, int(lens[j].min()))) if gscores[j] else 0, j),
        )

    def _partition_orderings(self, codes, lens, gorder, n, m) -> List[List[int]]:
        """Bounded recursive conditional prefix partition. Within each row group
        pick the remaining column with the largest length-weighted pair repetition
        (bincount on precomputed integer codes — no pandas calls per node), place
        it first, partition rows on its serialized value, and choose the suffix
        ordering separately inside each partition. Depth and candidate columns are
        bounded; every row receives a full permutation of all m columns."""
        orderings: list = [None] * n
        top = gorder[: self.MAX_CAND]

        def full_order(prefix: List[int]) -> List[int]:
            """prefix followed by remaining columns in global-frequency order."""
            return list(prefix) + [c for c in gorder if c not in prefix]

        def pick(rows: np.ndarray, cand: List[int]):
            """Return (best_col, score) maximizing len-weighted pair repetition."""
            best_j, best_s = -1, 0
            for j in cand:
                cj = codes[rows, j]
                cnt = np.bincount(cj, minlength=int(cj.max()) + 1).astype(np.int64)
                s = int(np.sum(lens[j][: len(cnt)] * cnt * (cnt - 1)))
                if s > best_s:
                    best_s, best_j = s, j
            return best_j, best_s

        def node(rows: np.ndarray, depth: int, prefix: List[int], rest: List[int]):
            """Assign per-row orderings for a group: either branch on a leading
            column or emit prefix + global-order tail. Every emitted order is a
            full permutation of range(m)."""
            if rows.size == 0:
                return
            if rows.size < 2 or depth >= self.MAX_DEPTH or not rest:
                order = full_order(prefix)
                for i in rows:
                    orderings[i] = order
                return
            cand = [c for c in top if c in rest]
            best_j, best_s = pick(rows, cand) if cand else (-1, 0)
            if best_j < 0 or best_s <= 0:
                order = full_order(prefix)
                for i in rows:
                    orderings[i] = order
                return
            cj = codes[rows, best_j]
            order = np.argsort(cj, kind="stable")
            sorted_rows = rows[order]
            boundaries = np.flatnonzero(np.diff(cj[order])) + 1
            new_prefix = prefix + [best_j]
            new_rest = [c for c in rest if c != best_j]
            start = 0
            for b in list(boundaries) + [len(rows)]:
                node(sorted_rows[start:b], depth + 1, new_prefix, new_rest)
                start = b

        node(np.arange(n), 0, [], list(range(m)))
        # Safety net: no row may keep a None or incomplete ordering.
        for i in range(n):
            if not isinstance(orderings[i], list) or len(orderings[i]) != m:
                orderings[i] = full_order([])
        return orderings

    # ---------- row strings ----------

    @staticmethod
    def _row_strings(S: np.ndarray, orderings) -> List[str]:
        """Build serialized row strings under per-row column orderings."""
        n = S.shape[0]
        uniform = None
        if orderings and not isinstance(orderings[0], (list, np.ndarray)):
            uniform = orderings
        out = []
        if uniform is not None:
            for i in range(n):
                out.append("".join(S[i, uniform]))
        else:
            for i in range(n):
                out.append("".join(S[i, orderings[i]]))
        return out

    # ---------- main ----------

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
        """Apply col_merge via the existing API, serialize and factorize once,
        build the bounded candidate set (two global orderings + one conditional
        partition tree), score each candidate exactly once with the exact
        sorted-string LCP objective, and return the best construction with
        valid per-row column orderings consistent with the returned frame."""
        df = df.copy()

        # Honor col_merge through the existing API before any construction.
        if col_merge:
            for group in col_merge:
                group_cols = [c for c in group if c in df.columns]
                if len(group_cols) > 1:
                    try:
                        df = self.merging_columns(df, group_cols, prepended=False)
                    except Exception:
                        pass

        n, m = df.shape
        if n == 0 or m == 0:
            return df, [[] for _ in range(n)]

        cols = list(df.columns)
        S = self._serialize(df)
        codes, lens, gscores = self._factorize(S)

        # Candidate A: global frequency ranking.
        order_a = self._global_order(gscores, m)
        # Candidate B: alternative length/frequency tradeoff.
        order_b = self._alt_order(gscores, lens, m)
        # Candidate C: bounded conditional prefix-partition tree (per-row orders).
        try:
            orderings_c = self._partition_orderings(codes, lens, order_a, n, m)
        except Exception:
            orderings_c = None

        # (uniform_order_or_None, per_row_orderings) pairs; one LCP score each.
        candidates = [(order_a, [order_a] * n), (order_b, [order_b] * n)]
        if orderings_c is not None:
            candidates.append((None, orderings_c))

        best_score = -1
        best = candidates[0]
        for uniform, ords in candidates:
            sc = _trie_score(self._row_strings(S, ords))
            if sc > best_score:
                best_score = sc
                best = (uniform, ords)

        uniform, ords = best
        if uniform is not None:
            col_names = [cols[j] for j in uniform]
            out = df.iloc[:, list(uniform)].copy()
            out.columns = col_names
            out = out.astype(object)
            column_orderings = [list(col_names)] * n
        else:
            first = list(ords[0])
            out = df.iloc[:, first].copy()
            out.columns = [cols[j] for j in first]
            out = out.astype(object)
            column_orderings = [[cols[j] for j in o] for o in ords]

        self._last_trie_score = best_score
        return out, column_orderings
# EVOLVE-BLOCK-END