# EVOLVE-BLOCK-START
import time
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict, Optional
from collections import Counter


class Evolved(Algorithm):
    """
    Prefix-caching reorderer (refined c-lcp-scorer construction selection).

    The evaluator serializes each row as "".join(fillna('').astype(str)) and shares
    characters through a serial character Trie. For a fixed multiset of row strings,
    ideal reuse is insertion-order independent and equals the sum of LCP lengths
    between lexicographically adjacent sorted strings. We build the original column
    order plus at most three cheap alternatives, score each exactly once with a fast
    exact LCP scorer (O(log L) binary search using C-speed slice equality; identical
    adjacent rows contribute len(row) directly), and emit the best (reordered
    DataFrame, per-row column orderings). Merge groups from col_merge are atomic
    contiguous blocks in every candidate, so constraint validity never depends on a
    fallback path.

    Refinements over the incumbent parent (constant-factor only):
      * No redundant whole-frame copy when col_merge is empty; the final iloc
        reindex already materializes a fresh frame with all original values.
      * The merge-block expansion closure is skipped entirely on the common
        no-merge path.
      * Scoring loop binds hot callables to locals and skips re-joining rows
        for a candidate whose expanded order matches the current best.
      * Exact objective, candidate count, and public API are unchanged.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ---------- serialization ----------

    @staticmethod
    def _serialize_columns(df: pd.DataFrame) -> List[List[str]]:
        """Serialize every cell once, exactly matching "".join(fillna('').astype(str))."""
        norm = df.fillna("").astype(str)
        return [norm[c].tolist() for c in df.columns]

    # ---------- scoring ----------

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """LCP length via O(log L) binary search with C-speed slice equality."""
        n = min(len(a), len(b))
        if n == 0:
            return 0
        if a[:n] == b[:n]:
            return n
        lo, hi = 0, n  # invariant: a[:lo]==b[:lo], a[:hi]!=b[:hi]
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid
        return lo

    @classmethod
    def _trie_reuse(cls, sorted_strings: List[str]) -> int:
        """Ideal serial-Trie reuse = sum of adjacent LCPs of pre-sorted strings.

        Identical adjacent strings contribute len(s) directly (no LCP call), so
        duplicate-row runs are handled exactly and cheaply.
        """
        if len(sorted_strings) < 2:
            return 0
        total = 0
        prev = sorted_strings[0]
        for cur in sorted_strings[1:]:
            if prev == cur:
                total += len(prev)
            else:
                total += cls._lcp(prev, cur)
            prev = cur
        return total

    # ---------- candidate orderings ----------

    def _candidate_orders(
        self, col_names: List[str], col_strings: List[List[str]]
    ) -> List[List[str]]:
        """Build the original order plus at most three cheap alternatives.

        Alt 1: global ranking by length-weighted pair repetition
               sum(len(v)*cnt*(cnt-1)) with deterministic name tie-breaks.
        Alt 2: force the single best lead column by that statistic, rank the
               rest by the same global statistic.
        Alt 3: force the same lead column, rank the rest by pair repetition
               *conditioned on the lead value's groups* (one O(n) dict pass per
               suffix column, sharing the lead's group codes; skipped when the
               table is too wide/long so total cost stays bounded).
        Returns deduplicated column orders in construction order.
        """
        m = len(col_names)
        cands: List[List[str]] = [list(col_names)]
        if m < 2:
            return cands
        n = len(col_strings[0]) if m else 0

        counts_all: List[Dict[str, int]] = []
        weighted: List[float] = []
        best_i, best_w = 0, -1.0
        for i, vals in enumerate(col_strings):
            counts = Counter(vals)
            counts_all.append(counts)
            w = sum(len(v) * c * (c - 1) for v, c in counts.items())
            weighted.append(w)
            if w > best_w:
                best_w, best_i = w, i

        names = col_names
        by_rep = sorted(range(m), key=lambda i: (-weighted[i], names[i]))
        cands.append([names[i] for i in by_rep])

        lead = best_i
        rest = [j for j in by_rep if j != lead]
        cands.append([names[lead]] + [names[j] for j in rest])

        # Conditional candidate: bounded O(n) work per suffix column.
        if 0 < n * m <= 5_000_000:
            lead_vals = col_strings[lead]
            gid: Dict[str, int] = {}
            groups = [gid.setdefault(v, len(gid)) for v in lead_vals]
            cond_w: List[float] = []
            for j in rest:
                pair: Dict[Tuple[str, int], int] = {}
                vs = col_strings[j]
                for t in range(n):
                    key = (vs[t], groups[t])
                    pair[key] = pair.get(key, 0) + 1
                cond_w.append(
                    sum(len(v) * c * (c - 1) for (v, _g), c in pair.items() if c > 1)
                )
            order_idx = sorted(
                range(len(rest)), key=lambda k: (-cond_w[k], names[rest[k]])
            )
            cands.append([names[lead]] + [names[rest[k]] for k in order_idx])

        seen, uniq = set(), []
        for c in cands:
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        return uniq

    # ---------- public API ----------

    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: int = None,
        col_stop: int = None,
        col_merge=[],
        one_way_dep=[],
        distinct_value_threshold: float = 0.8,
        parallel: bool = True,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        """Select a global column ordering + lexicographic row order maximizing exact Trie reuse.

        Merge groups from col_merge are positioned as atomic contiguous blocks in
        every candidate (anchored at the parent's column_stats rank). Each
        candidate is scored exactly once with the direct adjacent-LCP objective;
        the best wins, and its row permutation is recovered with a single sort.
        """
        start = time.time()
        work = df

        atomic_groups: List[List[str]] = []
        if col_merge:
            work = df.copy()
            try:
                self.num_rows, self.column_stats = self.calculate_col_stats(
                    work, enable_index=True
                )
                parent_order = [c for c, _, _, _ in self.column_stats]
                merged = set()
                for group in col_merge:
                    members = [c for c in work.columns if c in group]
                    if members:
                        atomic_groups.append(members)
                        merged.update(members)
                anchors = {g[0]: g for g in atomic_groups}
                base_cols = []
                for c in parent_order:
                    if c in anchors:
                        base_cols.extend(anchors[c])
                    elif c not in merged:
                        base_cols.append(c)
                base_cols = [c for c in base_cols if c in work.columns]
                for c in work.columns:  # safety: never drop a column
                    if c not in base_cols:
                        base_cols.append(c)
                work = work[base_cols]
                atomic_groups = [
                    g for g in atomic_groups if all(mn in base_cols for mn in g)
                ]
            except Exception:
                atomic_groups = []

        n = len(work)
        if n == 0 or work.shape[1] == 0:
            return work, [[] for _ in range(n)]

        col_names = list(work.columns)
        col_strings = self._serialize_columns(work)
        col_strings_by_name: Dict[str, List[str]] = dict(zip(col_names, col_strings))

        if atomic_groups:
            def expand(order: List[str]) -> List[str]:
                """Expand atomic merge groups into contiguous blocks at the group anchor."""
                out, placed = [], set()
                for c in order:
                    if c in placed:
                        continue
                    hit = False
                    for g in atomic_groups:
                        if c in g:
                            out.extend(x for x in g if x in col_strings_by_name)
                            placed.update(g)
                            hit = True
                            break
                    if not hit:
                        out.append(c)
                        placed.add(c)
                return out
        else:
            expand = list

        trie_reuse = self._trie_reuse
        best: Optional[Tuple[int, List[str], List[str]]] = None
        best_key: Optional[Tuple[str, ...]] = None
        for order in self._candidate_orders(col_names, col_strings):
            full_order = expand(order)
            key = tuple(full_order)
            if key == best_key:
                continue
            col_lists = [col_strings_by_name[c] for c in full_order]
            rows = ["".join(t) for t in zip(*col_lists)]
            score = trie_reuse(sorted(rows))
            if best is None or score > best[0]:
                best = (score, full_order, rows)
                best_key = key
            if time.time() - start > 240:
                break

        _, order, best_rows = best
        idx = sorted(range(n), key=best_rows.__getitem__)
        out = work.iloc[idx][order].reset_index(drop=True)
        orderings = [list(order) for _ in range(len(out))]
        return out, orderings
# EVOLVE-BLOCK-END