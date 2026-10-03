# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List


class Evolved(Algorithm):
    """
    Character-Trie prefix-caching optimizer.

    Serialize each cell once (missing -> ''), build at most three cheap
    candidate per-row column orderings:
      A) global frequency-ranked: sum over values v of len(v)*cnt*(cnt-1), desc
      B) original column order (as merge blocks)
      C) bounded conditional partition tree (narrow tables only)
    Score each candidate exactly with the ideal serial-Trie objective
    (sum of LCPs of lexicographically adjacent serialized rows) and return
    the best. Cell values are never modified; only per-row permutations.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.dep_graph = None

    # ---------------- serialization helpers ----------------

    def _serialize_columns(self, df: pd.DataFrame) -> dict:
        """Per-column list of scoring strings (missing -> ''), computed once."""
        out = {}
        for c in df.columns:
            s = df[c]
            if s.isna().any():
                s = s.astype(object).where(s.notna(), "")
            out[c] = [str(v) for v in s.tolist()]
        return out

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Longest common prefix length via binary search (C-speed slices)."""
        if a == b:
            return len(a)
        hi = min(len(a), len(b))
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _trie_score(self, strings: List[str]) -> int:
        """Exact ideal Trie reuse = sum of adjacent LCPs after sorting."""
        if len(strings) < 2:
            return 0
        ss = sorted(strings)
        tot = 0
        prev = ss[0]
        lcp = self._lcp
        for cur in ss[1:]:
            tot += lcp(prev, cur)
            prev = cur
        return tot

    # ---------------- candidate constructions ----------------

    @staticmethod
    def _rank_blocks(blocks, ser, rows):
        """Order blocks by sum over values v of len(v)*cnt*(cnt-1), desc."""
        scored = []
        for bi, b in enumerate(blocks):
            s = 0
            cnt = {}
            for c in b:
                col = ser[c]
                for r in rows:
                    v = col[r]
                    cnt[v] = cnt.get(v, 0) + 1
            for v, k in cnt.items():
                if k > 1:
                    s += len(v) * k * (k - 1)
            scored.append((-s, bi, b))
        scored.sort(key=lambda t: (t[0], t[1]))
        return [b for _, _, b in scored]

    def _cond_group(self, blocks, ser, rows, depth, max_depth, fallback):
        """Bounded conditional prefix partition tree.

        Pick the block with the largest length-weighted pair repetition
        (count*(count-1)*len) among current rows, partition rows by its
        serialized values, recurse inside each partition. Returns
        {row_index: ordered_block_list}.
        """
        if len(rows) < 3 or not blocks or depth >= max_depth:
            return {r: fallback for r in rows}
        best = None
        for bi, b in enumerate(blocks):
            groups = {}
            for r in rows:
                key = tuple(ser[c][r] for c in b)
                groups.setdefault(key, []).append(r)
            if len(groups) >= len(rows):
                continue  # no repetition at all
            sav = 0
            for key, rs in groups.items():
                k = len(rs)
                if k > 1:
                    sav += k * (k - 1) * sum(len(x) for x in key)
            if best is None or sav > best[0]:
                best = (sav, bi, groups)
        if best is None or best[0] <= 0:
            return {r: fallback for r in rows}
        _, bi, groups = best
        b = blocks[bi]
        rem = blocks[:bi] + blocks[bi + 1:]
        result = {}
        for rs in groups.values():
            sub_fallback = self._rank_blocks(rem, ser, rs)
            sub = self._cond_group(rem, ser, rs, depth + 1, max_depth, sub_fallback)
            for r in rs:
                result[r] = [b] + sub[r]
        return result

    # ---------------- main API ----------------

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
        n_rows, n_cols = df.shape
        if n_rows == 0 or n_cols == 0:
            return df.copy(), [[] for _ in range(n_rows)]

        # Build blocks: col_merge groups stay contiguous; each column in
        # exactly one block (dedupe defensively to avoid index errors).
        merged = set()
        blocks = []
        for group in col_merge:
            g = []
            for c in group:
                if c in df.columns and c not in merged and c not in g:
                    g.append(c)
            if g:
                blocks.append(g)
                merged.update(g)
        blocks.extend([c] for c in df.columns if c not in merged)

        ser = self._serialize_columns(df)
        all_rows = list(range(n_rows))

        # Candidate A: global frequency-ranked block order.
        global_order = self._rank_blocks(blocks, ser, all_rows)
        # Candidate B: original column order (as blocks).
        orig_order = blocks[:]

        # Candidate C: bounded conditional partition tree (narrow tables only).
        if len(blocks) <= 60:
            cond = self._cond_group(
                blocks, ser, all_rows, 0, min(len(blocks), 10), global_order
            )
            cond_orders = [cond[r] for r in all_rows]
        else:
            cond_orders = [global_order] * n_rows

        candidates = [
            [global_order] * n_rows,
            [orig_order] * n_rows,
            cond_orders,
        ]

        best_orders = None
        best_score = -1
        for cand in candidates:
            strings = [
                "".join(ser[c][i] for b in cand[i] for c in b)
                for i in all_rows
            ]
            sc = self._trie_score(strings)
            if sc > best_score:
                best_score = sc
                best_orders = cand

        # Expand blocks to per-row column orderings and permute values.
        pos = {c: j for j, c in enumerate(df.columns)}
        mat = df.values.astype(object)
        out = np.empty((n_rows, n_cols), dtype=object)
        orderings = []
        for i in all_rows:
            order = [c for b in best_orders[i] for c in b]
            orderings.append(order)
            for j, c in enumerate(order):
                out[i, j] = mat[i, pos[c]]

        out_df = pd.DataFrame(out, index=df.index, columns=df.columns)
        return out_df, orderings


# EVOLVE-BLOCK-END