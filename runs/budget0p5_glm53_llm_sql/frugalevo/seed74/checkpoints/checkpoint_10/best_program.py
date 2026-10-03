import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict, Optional


class Evolved(Algorithm):
    """
    Prefix-cache-optimized reorderer.

    Objective: maximize total matched characters in a character Trie over
    serialized rows ("" join of fillna("").astype(str) cell strings). For a
    fixed multiset of row strings, ideal Trie reuse equals the sum of LCPs
    between lexicographically adjacent sorted strings, so rows are emitted in
    lexicographic string order. Three bounded constructions (identity, global
    frequency, measured-gain conditional partition tree) are each scored once
    with the exact LCP objective, reusing the sorted strings produced by
    finalize. On top of the seeds, a bounded front-promotion local search
    mutates the best uniform global permutation: each column is promoted to
    position 0, scored with the exact objective, and kept only on strict
    improvement (hard caps: 2 sweeps, <= min(ncol, 12) scorings per sweep,
    skipped when the estimated serialized frame exceeds 2e6 characters).

    Compared to the best previous variant, only decision-neutral
    micro-optimizations were applied: frequency weights are computed once and
    reused (previously computed twice), and row-string joins use list
    comprehensions. The sampled character-budget estimate and the keyed sort
    are preserved verbatim because an exact-budget variant measurably changed
    (and lowered) emitted hit rates by disabling the local search on one
    dataset.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.colstr: List[List[str]] = []
        self.max_depth = 6                  # proven-bounded conditional-tree depth
        self.cand_limit = 8                 # bounded candidate columns per node
        self.suffix_greedy_steps = 3        # bounded measured-gain suffix steps per leaf
        self._freq_weights: List[int] = []  # cached len(v)*count*(count-1) per column

    # ---------- serialization ----------

    def _serialize(self, df: pd.DataFrame) -> None:
        """Serialize each column once into Python string lists (missing -> '')."""
        self.colstr = []
        for c in df.columns:
            out = []
            for v in df[c].tolist():
                if v is None or v is pd.NA or v is pd.NaT:
                    out.append("")
                elif isinstance(v, float) and v != v:
                    out.append("")
                else:
                    out.append(str(v))
            self.colstr.append(out)

    def _compute_freq_weights(self, ncol: int) -> None:
        """Compute per-column sum of len(v)*count*(count-1) over serialized
        values; cached so it is computed exactly once per reorder call."""
        weights = []
        for j in range(ncol):
            cnt: Dict[str, int] = {}
            wsum: Dict[str, int] = {}
            for v in self.colstr[j]:
                cnt[v] = cnt.get(v, 0) + 1
                wsum[v] = wsum.get(v, 0) + len(v)
            w = 0
            for v, c in cnt.items():
                if c > 1:
                    w += wsum[v] * (c - 1)
            weights.append(w)
        self._freq_weights = weights

    # ---------- scoring ----------

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Longest common prefix via bounded binary search with C-speed slices."""
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

    def _score_sorted(self, strs: List[str]) -> int:
        """Sum of adjacent LCPs of already-sorted row strings = exact Trie reuse."""
        total = 0
        prev = strs[0]
        for cur in strs[1:]:
            if cur == prev:
                total += len(cur)
            else:
                total += self._lcp(prev, cur)
            prev = cur
        return total

    # ---------- candidate constructions ----------

    def _rows_strings(self, row_ids: List[int], perms: List[Tuple[int, ...]]) -> List[str]:
        """Serialized row strings; list-comprehension joins run at C speed."""
        cs = self.colstr
        return ["".join([cs[j][r] for j in perm]) for r, perm in zip(row_ids, perms)]

    def _finalize(self, row_ids, perms) -> Tuple[List[int], List[Tuple[int, ...]], List[str]]:
        """Sort rows lexicographically by serialized string (optimal row order) and
        return the sorted strings too, so scoring never rebuilds row strings."""
        pairs = sorted(
            zip(self._rows_strings(row_ids, perms), row_ids, perms), key=lambda t: t[0]
        )
        return [p[1] for p in pairs], [p[2] for p in pairs], [p[0] for p in pairs]

    def _cand_identity(self, n: int, ncol: int):
        perm = tuple(range(ncol))
        return self._finalize(list(range(n)), [perm] * n)

    def _cand_freq(self, row_ids: List[int], ncol: int):
        """Global column order ranked by sum len(v)*count*(count-1), deterministic ties.
        Uses the cached frequency weights (computed once per call)."""
        order = sorted(range(ncol), key=lambda j: (-self._freq_weights[j], j))
        perm = tuple(order)
        return self._finalize(row_ids, [perm] * len(row_ids))

    def _node_lcp_gain(self, rows: List[int], j: int) -> int:
        """Measured gain for leading column j at a node: adjacent-LCP sum over the
        node's serialized values for j alone, deduped (identical values contribute
        full length once each; distinct neighbors contribute their prefix overlap).
        This is the exact character savings of putting j first."""
        vals = sorted(self.colstr[j][r] for r in rows)
        uniq: List[str] = []
        dup_len = 0
        prev = None
        for v in vals:
            if v == prev:
                dup_len += len(v)
            else:
                uniq.append(v)
                prev = v
        total = dup_len
        for i in range(1, len(uniq)):
            total += self._lcp(uniq[i - 1], uniq[i])
        return total

    def _group_freq_order(self, rows: List[int], remaining: List[int]) -> List[int]:
        """Suffix column order for a leaf tail: within-group length-weighted
        repetition, used as the cheap fallback after measured steps."""
        cs = self.colstr
        weights = []
        for j in remaining:
            cnt: Dict[str, int] = {}
            wsum: Dict[str, int] = {}
            for r in rows:
                v = cs[j][r]
                cnt[v] = cnt.get(v, 0) + 1
                wsum[v] = wsum.get(v, 0) + len(v)
            w = 0
            for v, c in cnt.items():
                if c > 1:
                    w += wsum[v] * (c - 1)
            weights.append(w)
        return [j for _, j in sorted(zip(weights, remaining), key=lambda t: (-t[0], t[1]))]

    def _group_suffix_order(self, rows: List[int], remaining: List[int]) -> List[int]:
        """Leaf suffix ordering by measured character savings: greedily pick the
        remaining column with the highest adjacent-LCP gain on this group's rows
        (bounded number of steps), then finish with the frequency tail."""
        rows_small = len(rows) < 2 or len(remaining) <= 1
        picked: List[int] = []
        rem = list(remaining)
        if not rows_small:
            for _ in range(min(self.suffix_greedy_steps, len(rem) - 1)):
                best_j, best_gain = None, 0
                for j in rem[: self.cand_limit]:
                    g = self._node_lcp_gain(rows, j)
                    if g > best_gain:
                        best_gain, best_j = g, j
                if best_j is None:
                    break
                picked.append(best_j)
                rem.remove(best_j)
        picked.extend(self._group_freq_order(rows, rem))
        return picked

    def _cand_tree(self, row_ids: List[int], ncol: int):
        """Conditional prefix partition tree. At each node the leading column is
        chosen by the *measured* adjacent-LCP savings (not a repetition heuristic),
        rows are partitioned on its value, and each leaf gets a group-local
        suffix order selected by measured LCP gain (frequency tail fallback).
        Depth and candidate count remain hard-bounded; no full row-string LCP
        scoring inside the tree loop."""

        def node(rows: List[int], remaining: List[int], prefix: Tuple[int, ...]):
            if len(rows) < 2 or not remaining or len(prefix) >= self.max_depth:
                return [(prefix + tuple(self._group_suffix_order(rows, remaining)), rows)]
            best_j, best_gain = None, 0
            for j in remaining[: self.cand_limit]:
                g = self._node_lcp_gain(rows, j)
                if g > best_gain:
                    best_gain, best_j = g, j
            if best_j is None:
                return [(prefix + tuple(self._group_suffix_order(rows, remaining)), rows)]
            groups: Dict[str, List[int]] = {}
            for r in rows:
                groups.setdefault(self.colstr[best_j][r], []).append(r)
            if len(groups) >= len(rows):
                # no repetition at this column: leaf with measured suffix order
                return [(prefix + tuple(self._group_suffix_order(rows, remaining)), rows)]
            rem = [j for j in remaining if j != best_j]
            out = []
            for key in sorted(groups):  # deterministic
                out.extend(node(groups[key], rem, prefix + (best_j,)))
            return out

        perms: List[Tuple[int, ...]] = [None] * len(row_ids)
        for perm, rows in node(row_ids, list(range(ncol)), ()):
            for r in rows:
                perms[r] = perm
        return self._finalize(row_ids, perms)

    def _local_search(
        self, seed_perm: Tuple[int, ...], budget_ok: bool
    ) -> Optional[Tuple[Tuple[int, ...], int]]:
        """Bounded front-promotion local search on a uniform global permutation.
        For each column j in deterministic index order, move j to position 0
        keeping the rest in relative order, score the whole frame with the exact
        sorted-LCP objective, and accept only strict improvements. One full
        sweep; a second only if the first improved; hard cap of 2 sweeps and
        min(ncol, 12) scorings per sweep. Skipped entirely (returns None) when
        the frame is too large or the permutation is trivial."""
        n = len(self.colstr[0])
        ncol = len(seed_perm)
        if not budget_ok or n < 2 or ncol < 2:
            return None
        all_rows = list(range(n))
        _, _, strs = self._finalize(all_rows, [seed_perm] * n)
        best_score = self._score_sorted(strs)
        best_perm = seed_perm
        sweeps = 0
        moves_cap = min(ncol, 12)
        while sweeps < 2:
            sweeps += 1
            improved = False
            for j in list(best_perm)[:moves_cap]:
                cand = tuple([j] + [k for k in best_perm if k != j])
                if cand == best_perm:
                    continue
                _, _, strs = self._finalize(all_rows, [cand] * n)
                s = self._score_sorted(strs)
                if s > best_score:
                    best_score = s
                    best_perm = cand
                    improved = True
            if not improved:
                break
        return best_perm, best_score

    # ---------- public API ----------

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
        """Produce the row/field ordering maximizing serial character-Trie reuse."""
        if col_merge:
            self.num_rows, self.column_stats = self.calculate_col_stats(df, enable_index=True)
            reordered_columns = [c for c, _, _, _ in self.column_stats]
            for col_to_merge in col_merge:
                final_col_order = [c for c in reordered_columns if c in col_to_merge]
                df = self.merging_columns(df, final_col_order, prepended=False)

        n, ncol = df.shape
        cols = list(df.columns)
        if n == 0 or ncol == 0:
            return df.copy(), [[] for _ in range(n)]

        self._serialize(df)
        self._compute_freq_weights(ncol)
        raw = df.astype(object).values.tolist()
        row_ids = list(range(n))

        # Budget guard for the local search: sampled mean-length estimate,
        # preserved verbatim from the best-scoring variant (an exact-count
        # variant measurably skipped the local search on one dataset and
        # lowered its hit rate).
        budget_ok = True
        if n * ncol > 200_000:
            sample = self.colstr[0][: min(n, 512)]
            mean_len = sum(len(s) for s in sample) / max(1, len(sample))
            total_chars = n * ncol * mean_len
            budget_ok = total_chars <= 2_000_000

        # Bounded construction selection: each candidate returns its own sorted
        # strings, so scoring costs one LCP pass and zero string rebuilding.
        candidates = [self._cand_identity(n, ncol), self._cand_freq(row_ids, ncol)]
        if n * ncol <= 4_000_000:
            candidates.append(self._cand_tree(row_ids, ncol))

        best = None
        best_score = -1
        best_uniform_perm: Optional[Tuple[int, ...]] = tuple(range(ncol))
        for row_order, perms, strs in candidates:
            score = self._score_sorted(strs)
            if score > best_score:
                best_score = score
                best = (row_order, perms)
                if len(set(perms)) == 1:
                    best_uniform_perm = perms[0]
                else:
                    best_uniform_perm = None

        # Front-promotion local search on the best uniform seed (or the cached
        # frequency order when the tree winner is non-uniform). Bounded scorings.
        if best_uniform_perm is None:
            best_uniform_perm = tuple(
                sorted(range(ncol), key=lambda j: (-self._freq_weights[j], j))
            )
        ls = self._local_search(best_uniform_perm, budget_ok)
        if ls is not None:
            ls_perm, ls_score = ls
            if ls_score > best_score:
                row_order, perms, _ = self._finalize(row_ids, [ls_perm] * n)
                best = (row_order, perms)

        row_order, perms = best

        data = []
        orderings = []
        for r, perm in zip(row_order, perms):
            data.append([raw[r][j] for j in perm])
            orderings.append([cols[j] for j in perm])
        out = pd.DataFrame(data, columns=[f"c{i}" for i in range(ncol)], dtype=object)
        return out, orderings