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
    prev = uniq[0]
    for i in range(1, len(uniq)):
        cur = uniq[i]
        total += _lcp(prev, cur)
        prev = cur
    return total


class Evolved(Algorithm):
    """
    Bounded construction selection, each candidate scored exactly once with the
    exact sorted-string LCP (serial-Trie) objective:
      (A) global frequency ranking by sum len(v)*cnt(v)*(cnt(v)-1),
      (B) alternative length/frequency tradeoff,
      (C) flat two-level row-cluster specialization: partition rows once on the
          factorized codes of a lead column (trying each of the top-3 gscore
          columns as lead and keeping the best total), then per cluster pick the
          best uniform order among {global A, cluster-local gscore ranking,
          cluster-local tradeoff, lead-first local ranking}, each verified with
          the exact LCP objective on that cluster's rows only. Largest clusters
          are specialized individually; the remainder merges into one bucket.
      (D) bounded recursive conditional prefix-partition tree (retained floor).
    Refinement: ONE bounded adjacent-swap local search on the winning uniform
    order (or the per-row winner's modal ordering as a uniform seed), 2 sweeps,
    hard cap min(90, max(60, 2*m)) swap scorings on a <=1000-row deterministic
    sample; strict improvements only. Serialization is scoring-only
    normalization; stored cell values, rows and columns are never altered.
    """

    MAX_CLUSTERS = 16   # largest clusters that get individual specialization
    MAX_LEADS = 3       # lead columns tried for the cluster partition
    MAX_DEPTH = 3       # depth of retained tree candidate D
    MAX_CAND = 6        # candidate columns examined per tree node
    MAX_SWEEPS = 2      # bounded local-search sweeps
    SAMPLE_ROWS = 1000  # rows used to score swaps during refinement
    MAX_SWAP_EVALS = 90 # hard cap on swap scorings per refinement call

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

    def _cluster_specialized(self, S, codes, lens, gorder, border, n, m):
        """Flat two-level row-cluster order specialization. Partition rows once on
        the factorized codes of the given lead column. Keep the MAX_CLUSTERS
        largest value-groups as separate clusters; merge the rest into one
        bucket. For each cluster build up to four candidate uniform orders —
        global A, cluster-local gscore ranking, cluster-local tradeoff, and a
        lead-first local ranking (lead value is shared inside the cluster so its
        serialized prefix is created once per cluster) — and score each with the
        exact sorted-string LCP objective on the cluster's row strings only.
        Assign the winning order to every row of the cluster. Returns
        (per_row_orderings, total_score) or None on degenerate input."""
        if m == 0 or n < 2:
            return None
        cj = codes[:, gorder[0]] if False else codes[:, self._lead_col]
        k = int(cj.max()) + 1
        cnt = np.bincount(cj, minlength=k)
        if k < 2:
            return None
        order_groups = np.argsort(-cnt, kind="stable")
        keep = [int(g) for g in order_groups[: self.MAX_CLUSTERS] if cnt[g] > 0]
        member = {}
        rem_mask = np.ones(n, dtype=bool)
        for g in keep:
            idx = np.flatnonzero(cj == g)
            member[g] = idx
            rem_mask[idx] = False
        rem_idx = np.flatnonzero(rem_mask)
        buckets = [member[g] for g in keep]
        if rem_idx.size > 0:
            buckets.append(rem_idx)

        orderings = [None] * n
        total = 0
        gkey = tuple(gorder)
        for rows in buckets:
            if rows.size < 2:
                orderings[rows[0]] = list(gorder)
                continue
            sub = S[rows]
            # One bincount per column on the cluster's precomputed codes.
            local_scores = []
            for j in range(m):
                cj2 = codes[rows, j]
                c2 = np.bincount(cj2, minlength=int(cj2.max()) + 1).astype(np.int64)
                local_scores.append(int(np.sum(lens[j][: len(c2)] * c2 * (c2 - 1))))
            local_order = sorted(range(m), key=lambda j: (-local_scores[j], j))
            min_len = [max(1, int(lens[j].min())) for j in range(m)]
            local_alt = sorted(
                range(m),
                key=lambda j: (-(local_scores[j] // min_len[j]) if local_scores[j] else 0, j),
            )
            # Lead-first variant: shared lead value prefix is trie-cheap inside
            # the cluster, then remaining columns by local savings.
            lead = self._lead_col
            lead_first = [lead] + [c for c in local_order if c != lead]
            cands = []
            for o in (gkey, tuple(local_order), tuple(local_alt), tuple(lead_first)):
                if o not in cands:
                    cands.append(o)
            best_o, best_sc = None, -1
            for key in cands:
                sc = _trie_score(["".join(r[key]) for r in sub])
                if sc > best_sc:
                    best_sc, best_o = sc, list(key)
            for i in rows:
                orderings[i] = best_o
            total += best_sc
        return orderings, total

    def _partition_orderings(self, codes, lens, gorder, n, m,
                             max_depth=None, local_suffix=False) -> List[List[int]]:
        """Bounded recursive conditional prefix partition (retained floor
        candidate). Within each row group pick the remaining column with the
        largest length-weighted pair repetition (bincount on precomputed integer
        codes — no pandas calls per node), place it first, partition rows on its
        serialized value, and recurse. Depth and candidate columns are bounded;
        every row receives a full permutation of all m columns."""
        max_depth = max_depth if max_depth is not None else self.MAX_DEPTH
        orderings: list = [None] * n
        top = gorder[: self.MAX_CAND]

        def local_savings(rows: np.ndarray, j: int) -> int:
            """Group-local length-weighted pair repetition for column j."""
            cj = codes[rows, j]
            cnt = np.bincount(cj, minlength=int(cj.max()) + 1).astype(np.int64)
            return int(np.sum(lens[j][: len(cnt)] * cnt * (cnt - 1)))

        def full_order(prefix: List[int], rows=None) -> List[int]:
            """prefix followed by remaining columns in global-frequency order."""
            return list(prefix) + [c for c in gorder if c not in prefix]

        def node(rows: np.ndarray, depth: int, prefix: List[int], rest: List[int]):
            """Assign per-row orderings for a group: either branch on a leading
            column or emit prefix + tail. Every emitted order is a full
            permutation of range(m)."""
            if rows.size == 0:
                return
            if rows.size < 2 or depth >= max_depth or not rest:
                order = full_order(prefix)
                for i in rows:
                    orderings[i] = order
                return
            cand = [c for c in top if c in rest]
            best_j, best_s = -1, 0
            for j in cand:
                s = local_savings(rows, j)
                if s > best_s:
                    best_s, best_j = s, j
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

    # ---------- local-search refinement ----------

    def _refine_uniform(self, S: np.ndarray, order: List[int]) -> List[int]:
        """Bounded first-improvement adjacent-swap local search on a uniform
        column order. At most MAX_SWEEPS sweeps over the m-1 adjacent
        transpositions, with a hard cap of min(90, max(60, 2*m)) swap scorings
        total. Each swap is scored once with the exact sorted-string LCP
        objective on a deterministic bounded row sample (<= SAMPLE_ROWS rows,
        evenly spaced) so per-swap cost is bounded; only strict improvements
        are accepted and the incumbent order is the fallback. Pure permutation:
        no data is touched."""
        m = len(order)
        if m < 2:
            return list(order)
        n = S.shape[0]
        if n > self.SAMPLE_ROWS:
            idx = np.linspace(0, n - 1, self.SAMPLE_ROWS).astype(np.int64)
            Ss = S[idx]
        else:
            Ss = S
        best_order = list(order)
        best_score = _trie_score(self._row_strings(Ss, best_order))
        evals = 0
        cap = min(self.MAX_SWAP_EVALS, max(60, 2 * m))
        for _ in range(self.MAX_SWEEPS):
            improved = False
            for k in range(m - 1):
                if evals >= cap:
                    return best_order
                cand = list(best_order)
                cand[k], cand[k + 1] = cand[k + 1], cand[k]
                sc = _trie_score(self._row_strings(Ss, cand))
                evals += 1
                if sc > best_score:
                    best_score = sc
                    best_order = cand
                    improved = True
            if not improved:
                break
        return best_order

    @staticmethod
    def _modal_order(orderings) -> List[int]:
        """Most frequent per-row ordering (deterministic tie-break: first seen),
        used as a uniform seed for local-search refinement of a per-row
        candidate."""
        counts = {}
        for o in orderings:
            key = tuple(o)
            counts[key] = counts.get(key, 0) + 1
        best_key = max(counts, key=lambda k: (counts[k], -len(k)))
        return list(best_key)

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
        build the bounded candidate set — two global orderings, the flat
        row-cluster specialization tried over the top-3 lead columns (each
        specialization scored once as a sum of per-cluster exact LCP scores),
        and the retained conditional-partition tree — score whole-table
        candidates once with the exact LCP objective, run the bounded
        adjacent-swap local search ONCE on the winning uniform order (or the
        per-row winner's modal order as a uniform seed), and return the best
        construction with valid per-row column orderings consistent with the
        returned frame."""
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

        # Candidate C: flat cluster specialization, tried over the top-3 lead
        # columns; keep the best-scoring specialization (sum of per-cluster
        # exact scores). Bounded: <= 3 * (MAX_CLUSTERS + 1) * 4 scorings.
        spec_best = None
        spec_best_score = -1
        for lead in order_a[: self.MAX_LEADS]:
            self._lead_col = lead
            try:
                res = self._cluster_specialized(S, codes, lens, order_a, order_b, n, m)
            except Exception:
                res = None
            if res is not None and res[1] > spec_best_score:
                spec_best_score = res[1]
                spec_best = res[0]

        # Candidate D: bounded conditional prefix-partition tree (floor).
        try:
            orderings_d = self._partition_orderings(codes, lens, order_a, n, m,
                                                    max_depth=self.MAX_DEPTH)
        except Exception:
            orderings_d = None

        # (uniform_order_or_None, per_row_orderings) pairs; one LCP score each
        # for whole-table candidates; specialization score already computed.
        candidates = [(order_a, [order_a] * n), (order_b, [order_b] * n)]
        if orderings_d is not None:
            candidates.append((None, orderings_d))

        best_score = -1
        best = candidates[0]
        for uniform, ords in candidates:
            sc = _trie_score(self._row_strings(S, ords))
            if sc > best_score:
                best_score = sc
                best = (uniform, ords)
        if spec_best is not None and spec_best_score > best_score:
            best_score = spec_best_score
            best = (None, spec_best)

        uniform, ords = best

        # Refinement: ONE bounded adjacent-swap local search on the winner's
        # uniform order (or the per-row winner's modal ordering as a uniform
        # seed). Replaces the incumbent only on strict improvement of the full
        # exact score, so a winning per-row candidate is never degraded.
        if uniform is not None:
            refined = self._refine_uniform(S, uniform)
            if refined != uniform:
                rsc = _trie_score(self._row_strings(S, refined))
                if rsc > best_score:
                    best_score = rsc
                    uniform = refined
        else:
            seed = self._modal_order(ords)
            refined = self._refine_uniform(S, seed)
            rsc = _trie_score(self._row_strings(S, refined))
            if rsc > best_score:
                best_score = rsc
                uniform = refined

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