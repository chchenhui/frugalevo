# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List


class Evolved(Algorithm):
    """
    Character-Trie prefix-caching optimizer.

    Every cell is serialized once; each merge-block's concatenated string per
    row is precomputed once and shared by all constructions and by the exact
    Trie scorer. A small bounded set of candidate per-row column orderings is
    built:
      A) global frequency-ranked order (len*cnt*(cnt-1) per block),
      B) original column order,
      C) conditional partition tree with frequency-ranked leaf fallback,
      D) conditional partition tree with self-similarity leaf fallback,
      E) greedy append-order chosen by actual Trie score on a row sample,
      F) blocks ordered by internal self-similarity (sorted-value LCP sum).
    Each candidate is scored with the exact ideal serial-Trie objective
    (sum of LCPs of lexicographically adjacent serialized rows) and the best
    is returned. Cell values are never modified; only per-row permutations.
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
            if cur != prev:
                tot += lcp(prev, cur)
                prev = cur
            else:
                tot += len(cur)
        return tot

    # ---------------- candidate constructions ----------------
    # All constructions work on block indices; block strings bstrs[bi][r]
    # are the concatenated serialized values of block bi for row r.

    @staticmethod
    def _pair_score(vals: List[str]) -> int:
        """Sum over distinct values v of len(v)*cnt*(cnt-1)."""
        cnt = {}
        for v in vals:
            cnt[v] = cnt.get(v, 0) + 1
        s = 0
        for v, k in cnt.items():
            if k > 1:
                s += len(v) * k * (k - 1)
        return s

    def _rank_blocks(self, ids: List[int], bstrs, rows) -> List[int]:
        """Order block indices by pair-repetition score, desc (deterministic)."""
        scored = []
        for bi in ids:
            vals = bstrs[bi]
            scored.append((-self._pair_score([vals[r] for r in rows]), bi))
        scored.sort()
        return [bi for _, bi in scored]

    def _selfsim_rank(self, ids: List[int], bstrs, rows) -> List[int]:
        """Order blocks by internal self-similarity (sorted-value LCP sum)."""
        scored = []
        for bi in ids:
            vals = bstrs[bi]
            sub = [vals[r] for r in rows]
            scored.append((-self._trie_score(sub), bi))
        scored.sort()
        return [bi for _, bi in scored]

    def _cond_group(self, ids: List[int], bstrs, rows, depth,
                    max_depth, fallback):
        """Bounded conditional prefix partition tree.

        Pick the block with the largest length-weighted pair repetition among
        the current rows, partition rows by its serialized block string, and
        recurse inside each partition. Returns {row_index: ordered id list}.
        """
        if len(rows) < 3 or not ids or depth >= max_depth:
            return {r: fallback for r in rows}
        best = None
        for bi in ids:
            vals = bstrs[bi]
            groups = {}
            for r in rows:
                groups.setdefault(vals[r], []).append(r)
            if len(groups) >= len(rows):
                continue
            sav = 0
            for key, rs in groups.items():
                k = len(rs)
                if k > 1:
                    sav += k * (k - 1) * len(key)
            if best is None or sav > best[0]:
                best = (sav, bi, groups)
        if best is None or best[0] <= 0:
            return {r: fallback for r in rows}
        _, bi, groups = best
        rem = [x for x in ids if x != bi]
        result = {}
        for rs in groups.values():
            sub_fallback = self._rank_blocks(rem, bstrs, rs)
            sub = self._cond_group(rem, bstrs, rs, depth + 1,
                                   max_depth, sub_fallback)
            for r in rs:
                result[r] = [bi] + sub[r]
        return result

    def _greedy_order(self, ids: List[int], bstrs, rows) -> List[int]:
        """Greedy block append order chosen by real Trie score on a sample."""
        k = len(ids)
        if k > 12:
            return self._rank_blocks(ids, bstrs, rows)
        if len(rows) > 1200:
            step = len(rows) / 1200.0
            rows = [rows[int(i * step)] for i in range(1200)]
        trie = self._trie_score
        remaining = self._rank_blocks(ids, bstrs, rows)
        order = [remaining.pop(0)]
        cur = [bstrs[order[0]][r] for r in rows]
        while remaining:
            best_b = None
            best_sc = -1
            best_cur = None
            for b in remaining:
                vals = bstrs[b]
                cand = [cur[i] + vals[r] for i, r in enumerate(rows)]
                sc = trie(cand)
                if sc > best_sc:
                    best_sc = sc
                    best_b = b
                    best_cur = cand
            order.append(best_b)
            cur = best_cur
            remaining.remove(best_b)
        return order

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

        # Build blocks: col_merge groups stay contiguous in given order.
        merged = set()
        blocks = []
        for group in col_merge:
            g = [c for c in group if c in df.columns]
            if g:
                blocks.append(g)
                merged.update(g)
        blocks.extend([c] for c in df.columns if c not in merged)

        ser = self._serialize_columns(df)
        # Precompute concatenated block strings per row, once.
        bstrs = []
        for b in blocks:
            cols = [ser[c] for c in b]
            if len(cols) == 1:
                col = cols[0]
                bstrs.append([col[r] for r in range(n_rows)])
            else:
                bstrs.append(["".join(col[r] for col in cols)
                              for r in range(n_rows)])
        all_rows = list(range(n_rows))
        ids = list(range(len(blocks)))

        # Candidate A: global frequency-ranked block order.
        global_order = self._rank_blocks(ids, bstrs, all_rows)
        # Candidate B: original column order (as blocks).
        orig_order = ids[:]
        # Candidate F: self-similarity ranked order.
        ss_order = self._selfsim_rank(ids, bstrs, all_rows)

        candidates = [
            [global_order] * n_rows,
            [orig_order] * n_rows,
            [ss_order] * n_rows,
        ]

        # Candidates C/D: bounded conditional partition trees.
        if len(blocks) <= 60:
            max_depth = min(len(blocks), 10)
            cond_f = self._cond_group(ids, bstrs, all_rows, 0, max_depth,
                                      global_order)
            cond_s = self._cond_group(ids, bstrs, all_rows, 0, max_depth,
                                      ss_order)
            candidates.append([cond_f[r] for r in all_rows])
            candidates.append([cond_s[r] for r in all_rows])

        # Candidate E: greedy append order by real Trie score (bounded).
        greedy = self._greedy_order(ids, bstrs, all_rows)
        candidates.append([greedy] * n_rows)

        best_orders = None
        best_score = -1
        for cand in candidates:
            strings = []
            for i in all_rows:
                co = cand[i]
                strings.append("".join(bstrs[bi][i] for bi in co))
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
            order = [c for bi in best_orders[i] for c in blocks[bi]]
            orderings.append(order)
            row_mat = mat[i]
            for j, c in enumerate(order):
                out[i, j] = row_mat[pos[c]]

        out_df = pd.DataFrame(out, index=df.index, columns=df.columns)
        return out_df, orderings


# EVOLVE-BLOCK-END