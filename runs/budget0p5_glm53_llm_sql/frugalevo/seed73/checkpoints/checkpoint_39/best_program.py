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
      (C) TWO-LEVEL row-cluster specialization: partition rows once on the
          factorized codes of a lead column (trying each of the top-3 gscore
          columns as lead), then for each bucket with > 200 rows partition a
          second time on the best group-local gscore column (pick the top-2
          candidates by local score, keep the best split; <= 8 largest
          subgroups kept per split, remainder merged; hard depth 2, <= 32
          subclusters per lead trial). Each (final) subcluster picks the best
          uniform order among {global A, cluster-local gscore ranking,
          cluster-local tradeoff, lead-first local ranking}, each verified with
          the exact LCP objective on that subcluster's rows only; the winner
          then undergoes a bounded first-improvement INSERTION-move local
          search (remove column at p, reinsert at k) scored with the exact LCP
          objective on that subcluster's own rows; the incumbent is kept unless
          strictly improved. Small buckets (< 2 rows) get the global order.
      (D) bounded recursive conditional prefix-partition tree (retained floor).
    The global adjacent-swap `_refine_uniform` remains only as the fallback
    refinement path for uniform candidates; cluster-local insertion search is
    the primary refinement for subclusters. Serialization is scoring-only
    normalization; stored cell values, rows and columns are never altered.
    """

    MAX_CLUSTERS = 16   # largest first-level clusters that get specialization
    MAX_LEADS = 3       # lead columns tried for the first-level partition
    MAX_DEPTH = 3       # depth of retained tree candidate D
    MAX_CAND = 6        # candidate columns examined per tree node
    MAX_SWEEPS = 2      # bounded local-search sweeps (global fallback path)
    SAMPLE_ROWS = 1000  # rows used to score swaps during fallback refinement
    MAX_SWAP_EVALS = 90 # hard cap on swap scorings per fallback refinement call
    INS_SWEEPS = 2      # insertion-search sweeps per subcluster
    INS_CAP = 60        # hard cap on insertion-move scorings per subcluster
    SPLIT_MIN_ROWS = 200  # only buckets larger than this get a second split
    SPLIT_SUBCLUSTERS = 8 # largest second-level value groups kept per split
    SPLIT_MAX_TOTAL = 32  # hard cap on subclusters per lead trial

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

    @staticmethod
    def _insertion_refine(sub: np.ndarray, order: List[int],
                          sweeps: int = 2, cap: int = 60) -> Tuple[List[int], int]:
        """Bounded first-improvement INSERTION-move local search on one
        subcluster's uniform order: for each position p try removing that column
        and reinserting it at every other position k. Each move is scored once
        with the exact sorted-string LCP objective on the subcluster's own row
        strings; only strict improvements are accepted. Hard cap of `cap`
        scorings per sweep-block; the incumbent order is the fallback. Pure
        permutation of column indices; no data is touched."""
        m = len(order)
        if m < 2:
            return list(order), _trie_score(["".join(r[order]) for r in sub])
        best = list(order)
        best_sc = _trie_score(["".join(r[best]) for r in sub])
        evals = 0
        for _ in range(sweeps):
            improved = False
            for p in range(m):
                if evals >= cap:
                    break
                col = best[p]
                rest = best[:p] + best[p + 1:]
                for k in range(m):
                    if k == p or evals >= cap:
                        continue
                    cand = rest[:k] + [col] + rest[k:]
                    sc = _trie_score(["".join(r[cand]) for r in sub])
                    evals += 1
                    if sc > best_sc:
                        best, best_sc = cand, sc
                        improved = True
                if evals >= cap:
                    break
            if not improved:
                break
        return best, best_sc

    def _local_gscores(self, codes, lens, rows: np.ndarray, m: int) -> List[int]:
        """Group-local length-weighted pair repetition per column via one
        bincount per column on the cluster's precomputed integer codes."""
        out = []
        for j in range(m):
            cj2 = codes[rows, j]
            c2 = np.bincount(cj2, minlength=int(cj2.max()) + 1).astype(np.int64)
            out.append(int(np.sum(lens[j][: len(c2)] * c2 * (c2 - 1))))
        return out

    def _assign_bucket(self, sub: np.ndarray, rows: np.ndarray, m: int,
                       gorder: List[int], lead: int,
                       orderings: list, allow_split: bool) -> int:
        """Emit per-row orderings for one (sub)cluster: optionally perform the
        adaptive second-level split, otherwise run the existing per-cluster
        template selection + bounded insertion refinement. Returns the
        subcluster's total exact LCP score."""
        if rows.size < 2:
            orderings[rows[0]] = list(gorder)
            return 0
        # Adaptive second-level split on the best group-local column != lead.
        if allow_split and rows.size > self.SPLIT_MIN_ROWS and m >= 2:
            local_scores = self._local_gscores(codes=None, lens=None, rows=None,
                                               m=m) if False else None
            # (guard: compute with real codes via caller-provided closure)
            split_codes = self._cur_codes
            split_lens = self._cur_lens
            ls = []
            for j in range(m):
                cj2 = split_codes[rows, j]
                c2 = np.bincount(cj2, minlength=int(cj2.max()) + 1).astype(np.int64)
                ls.append(int(np.sum(split_lens[j][: len(c2)] * c2 * (c2 - 1))))
            # Rank second-lead candidates: best local score, column != lead,
            # positive score. Deterministic tie-break on column index.
            cands = sorted((j for j in range(m) if j != lead and ls[j] > 0),
                           key=lambda j: (-ls[j], j))
            if cands:
                j2 = cands[0]
                cj = split_codes[rows, j2]
                order = np.argsort(cj, kind="stable")
                sorted_rows = rows[order]
                boundaries = np.flatnonzero(np.diff(cj[order])) + 1
                starts = [0] + [int(b) for b in boundaries]
                ends = [int(b) for b in boundaries] + [len(rows)]
                groups = [sorted_rows[s:e] for s, e in zip(starts, ends)]
                if len(groups) > 2:
                    sizes = [(g.size, gi) for gi, g in enumerate(groups)]
                    sizes.sort(key=lambda t: (-t[0], t[1]))
                    keep_set = set(gi for _, gi in sizes[: self.SPLIT_SUBCLUSTERS])
                    merged = [g for gi, g in enumerate(groups) if gi not in keep_set]
                    fin = [g for gi, g in enumerate(groups) if gi in keep_set]
                    if merged:
                        fin.append(np.concatenate(merged))
                    fin = [g for g in fin if g.size > 0]
                    # Hard cap on subclusters per lead trial.
                    if self._subcluster_budget - len(fin) < 0:
                        fin = None  # budget exhausted: fall through to flat
                    else:
                        self._subcluster_budget -= len(fin)
                else:
                    fin = groups
                if fin is not None and 1 < len(fin) <= self.SPLIT_MAX_TOTAL:
                    total = 0
                    for g in fin:
                        total += self._assign_bucket(
                            split_S_get(g) if False else None,
                            g, m, gorder, lead, orderings, False)
                    return total
        # Flat bucket behavior: per-cluster template selection + insertion refine.
        sub_rows = rows
        sub = self._cur_S[sub_rows]
        local_scores = ls if ls is not None else []
        if not local_scores:
            local_scores = []
            for j in range(m):
                cj2 = self._cur_codes[sub_rows, j]
                c2 = np.bincount(cj2, minlength=int(cj2.max()) + 1).astype(np.int64)
                local_scores.append(int(np.sum(self._cur_lens[j][: len(c2)] * c2 * (c2 - 1))))
        local_order = sorted(range(m), key=lambda j: (-local_scores[j], j))
        min_len = [max(1, int(self._cur_lens[j].min())) for j in range(m)]
        local_alt = sorted(
            range(m),
            key=lambda j: (-(local_scores[j] // min_len[j]) if local_scores[j] else 0, j),
        )
        lead_first = [lead] + [c for c in local_order if c != lead]
        gkey = tuple(gorder)
        cands = []
        for o in (gkey, tuple(local_order), tuple(local_alt), tuple(lead_first)):
            if o not in cands:
                cands.append(o)
        best_o, best_sc = None, -1
        for key in cands:
            sc = _trie_score(["".join(r[key]) for r in sub])
            if sc > best_sc:
                best_sc, best_o = sc, list(key)
        refined_o, refined_sc = self._insertion_refine(
            sub, best_o, self.INS_SWEEPS, self.INS_CAP
        )
        if refined_sc > best_sc:
            best_o, best_sc = refined_o, refined_sc
        for i in rows:
            orderings[i] = list(best_o)
        return best_sc

    def _cluster_specialized(self, S, codes, lens, gorder, border, n, m):
        """Two-level row-cluster order specialization with per-subcluster
        insertion refinement. First level: partition rows once on the
        factorized codes of the given lead column, keeping the MAX_CLUSTERS
        largest value-groups as clusters (rest merged into one bucket). Second
        level: each cluster with > SPLIT_MIN_ROWS rows is split again on the
        best group-local gscore column (column != lead, positive score),
        keeping the SPLIT_SUBCLUSTERS largest subgroups with the remainder
        merged; total subclusters per lead trial are hard-capped at
        SPLIT_MAX_TOTAL. Each final subcluster then runs the existing
        per-cluster candidate selection and bounded insertion refinement.
        Returns (per_row_orderings, total_score) or None on degenerate input."""
        if m == 0 or n < 2:
            return None
        lead = self._lead_col
        cj = codes[:, lead]
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

        # Context for the recursive assignment helper (avoids long signatures).
        self._cur_S = S
        self._cur_codes = codes
        self._cur_lens = lens
        self._subcluster_budget = self.SPLIT_MAX_TOTAL

        orderings = [None] * n
        total = 0
        for rows in buckets:
            total += self._assign_bucket(S, rows, m, gorder, lead,
                                         orderings, allow_split=True)
        for i in range(n):
            if not isinstance(orderings[i], list) or len(orderings[i]) != m:
                orderings[i] = list(gorder)
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

    # ---------- local-search refinement (fallback path) ----------

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
        build the bounded candidate set — two global orderings, the two-level
        row-cluster specialization with adaptive second-level conditioning and
        per-subcluster insertion-move refinement (tried over the top-3 lead
        columns; each specialization scored once as a sum of per-subcluster
        exact LCP scores), and the retained conditional-partition tree — score
        whole-table candidates once with the exact LCP objective, run the
        bounded adjacent-swap local search only on the winning uniform
        candidate as a fallback path, and return the best construction with
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

        # Candidate C: two-level cluster specialization with per-subcluster
        # insertion refinement, tried over the top-3 lead columns; keep the
        # best-scoring specialization (sum of per-subcluster exact scores).
        # Bounded: <= 3 * SPLIT_MAX_TOTAL * (4 + INS_CAP) scorings.
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

        # Fallback refinement: bounded adjacent-swap local search on the
        # winner's uniform order (or the per-row winner's modal ordering as a
        # uniform seed). Replaces the incumbent only on strict improvement of
        # the full exact score, so a winning per-row candidate is never
        # degraded.
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