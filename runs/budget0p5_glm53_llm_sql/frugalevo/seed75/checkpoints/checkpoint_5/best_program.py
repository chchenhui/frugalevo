# EVOLVE-BLOCK-START
import time
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict, Optional
from collections import Counter


class Evolved(Algorithm):
    """
    Prefix-caching reorderer (refined incumbent, c-lcp-scorer family).

    The evaluator serializes each row as "".join(fillna('').astype(str)) and shares
    characters through a serial character Trie. For a fixed multiset of row strings,
    ideal reuse is insertion-order independent and equals the sum of LCP lengths
    between lexicographically adjacent sorted strings. We build at most three cheap
    column-order candidates, score each exactly once with a fast exact LCP scorer
    (binary search using C-speed slice equality), and emit the best
    (reordered DataFrame, per-row column orderings). Merge groups are atomic
    contiguous blocks in every candidate so constraint validity never falls back.

    Refinements over the parent:
      * The third candidate is a conditional-partition hypothesis (best leading
        column by length-weighted pair repetition, base order thereafter) instead
        of a redundant length-ranked permutation.
      * Scoring uses sorted(rows) directly; the winning sort index is recovered
        with numpy argsort only once, for the final DataFrame construction.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ---------- serialization ----------

    def _serialize_columns(self, df: pd.DataFrame) -> List[List[str]]:
        """Serialize each column once, exactly matching "".join(fillna('').astype(str))."""
        return [df[c].fillna("").astype(str).tolist() for c in df.columns]

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
        """Ideal serial-Trie reuse = sum of adjacent LCPs of pre-sorted strings."""
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
    ) -> List[Tuple[List[str], Optional[int]]]:
        """Build at most three cheap candidates, each (order, forced_lead_col).

        Candidate 1 is the original order. Candidate 2 ranks columns by
        length-weighted pair repetition sum(len(v)*cnt*(cnt-1)) so frequently
        repeated, long serialized values lead. Candidate 3 is conditional: it
        forces the single best leading column by that same weighted statistic
        and keeps remaining columns in base order, letting row sorting group
        rows by the lead value for cross-group prefix sharing.
        """
        cands: List[Tuple[List[str], Optional[int]]] = [(list(col_names), None)]
        weighted = []
        best_i, best_w = 0, -1.0
        for i, vals in enumerate(col_strings):
            counts: Dict[str, int] = Counter(vals)
            w = sum(len(v) * c * (c - 1) for v, c in counts.items())
            weighted.append(w)
            if w > best_w:
                best_w, best_i = w, i

        by_rep = sorted(range(len(col_names)), key=lambda i: (-weighted[i], col_names[i]))
        cands.append(([col_names[i] for i in by_rep], None))
        # Conditional candidate: lead = argmax weighted repetition, rest in base order.
        lead = col_names[best_i]
        rest = [c for c in col_names if c != lead]
        cands.append(([lead] + rest, best_i))

        seen, uniq = set(), []
        for c in cands:
            key = tuple(c[0])
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
        every candidate (using the parent's column_stats rank as the anchor), so
        constraint validity never depends on a silent fallback. Each candidate is
        scored exactly once with the LCP-sum objective; the best wins.
        """
        start = time.time()
        work = df.copy()

        atomic_groups: List[List[str]] = []
        if col_merge:
            try:
                self.num_rows, self.column_stats = self.calculate_col_stats(work, enable_index=True)
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
                atomic_groups = [g for g in atomic_groups if all(m in base_cols for m in g)]
            except Exception:
                atomic_groups = []

        n = len(work)
        if n == 0 or work.shape[1] == 0:
            return work, [[] for _ in range(n)]

        col_names = list(work.columns)
        col_strings = self._serialize_columns(work)
        col_strings_by_name: Dict[str, List[str]] = dict(zip(col_names, col_strings))
        name_to_idx = {c: i for i, c in enumerate(col_names)}

        def expand(order: List[str]) -> List[str]:
            """Expand atomic merge groups into contiguous blocks at the group anchor."""
            if not atomic_groups:
                return list(order)
            out, placed = [], set()
            for c in order:
                if c in placed:
                    continue
                hit = False
                for g in atomic_groups:
                    if c in g:
                        out.extend(x for x in g if x in name_to_idx)
                        placed.update(g)
                        hit = True
                        break
                if not hit:
                    out.append(c)
                    placed.add(c)
            return out

        best: Optional[Tuple[int, List[str], np.ndarray]] = None
        for order, forced_lead in self._candidate_orders(col_names, col_strings):
            full_order = expand(order)
            col_lists = [col_strings_by_name[c] for c in full_order]
            rows = ["".join(t) for t in zip(*col_lists)]
            if forced_lead is not None and atomic_groups:
                # With merge groups, keeping base order after the lead preserves blocks;
                # rows already grouped by the forced lead via sorting.
                pass
            srt = sorted(rows)
            score = self._trie_reuse(srt)
            if best is None or score > best[0]:
                idx = np.argsort(np.array(rows, dtype=object), kind="stable")
                best = (score, full_order, idx)
            if time.time() - start > 240:
                break

        _, order, idx = best
        out = work.iloc[idx][order].reset_index(drop=True)
        orderings = [list(order) for _ in range(len(out))]
        return out, orderings
# EVOLVE-BLOCK-END