# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List


class Evolved(Algorithm):
    """
    Character-Trie prefix-caching optimizer.

    Serializes every cell once, then builds a small bounded set of candidate
    per-row column orderings:
      A) global frequency-ranked order (len*count*(count-1) per block),
      B) original column order,
      C) bounded conditional partition tree,
      D) greedy append-order chosen by actual Trie score on a row sample,
      E) blocks ordered by internal self-similarity (sorted-value LCP sum).
    Each candidate is scored exactly with the ideal serial-Trie objective
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
            tot += lcp(prev, cur)
            prev = cur
        return tot

    # ---------------- candidate constructions ----------------

    @staticmethod
    def _block_pair_score(b, ser, rows):
        """Sum over values v of len(v)*cnt*(cnt-1) for one block."""
        s = 0
        cnt = {}
        cols = [ser[c] for c in b]
        if len(cols) == 1:
            col = cols[0]
            for r in rows:
                v = col[r]
                cnt[v] = cnt.get(v, 0) + 1
            for v, k in cnt.items():
                if k > 1:
                    s += len(v) * k * (k - 1)
        else:
            for r in rows:
                key = tuple(col[r] for col in cols)
                cnt[key] = cnt.get(key, 0) + 1
            for key, k in cnt.items():
                if k > 1:
                    s += k * (k - 1) * sum(len(x) for x in key)
        return s

    @classmethod
    def _rank_blocks(cls, blocks, ser, rows):
        """Order blocks by pair-repetition score, descending (deterministic)."""
        scored = [(-cls._block_pair_score(b, ser, rows), bi, b)
                  for bi, b in enumerate(blocks)]
        scored.sort(key=lambda t: (t[0], t[1]))
        return [b for _, _, b in scored]

    def _cond_group(self, blocks, ser, rows, depth, max_depth, fallback):
        """Bounded conditional prefix partition tree.

        Pick the block with the largest length-weighted pair repetition among
        the current rows, partition rows by its serialized values, and recurse
        inside each partition. Returns {row_index: ordered_block_list}.
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
                continue
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

    def _greedy_order(self, blocks, ser, rows):
        """Greedy block append order chosen by real Trie score on a sample.

        Start from the block with the highest pair-repetition score, then
        repeatedly append the block that maximizes the exact sorted-LCP Trie
        score of the serialized rows built so far. Bounded: only runs when
        the block count and sampled row count are small enough.
        """
        k = len(blocks)
        if k > 12:
            return None
        if len(rows) > 1200:
            step = len(rows) / 1200.0
            rows = [rows[int(i * step)] for i in range(1200)]
        trie = self._trie_score
        remaining = list(blocks)
        remaining.sort(key=lambda b: (-self._block_pair_score(b, ser, rows),
                                      blocks.index(b)))
        order = [remaining.pop(0)]
        cur = ["".join(ser[c][r] for c in order[0]) for r in rows]
        while remaining:
            best_b = None
            best_sc = -1
            best_cur = None
            for b in remaining:
                vs = ["".join(ser[c][r] for c in b) for r in rows]
                cand = [cur[i] + vs[i] for i in range(len(rows))]
                sc = trie(cand)
                if sc > best_sc:
                    best_sc = sc
                    best_b = b
                    best_cur = cand
            order.append(best_b)
            cur = best_cur
            remaining.remove(best_b)
        return order

    def _selfsim_order(self, blocks, ser, rows):
        """Order blocks by internal self-similarity (sorted-value LCP sum)."""
        if len(rows) > 2000:
            step = len(rows) / 2000.0
            rows = [rows[int(i * step)] for i in range(2000)]
        scored = []
        for bi, b in enumerate(blocks):
            vals = ["".join(ser[c][r] for c in b) for r in rows]
            scored.append((-self._trie_score(vals),
                           -self._block_pair_score(b, ser, rows), bi, b))
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        return [b for _, _, _, b in scored]

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
        all_rows = list(range(n_rows))

        # Candidate A: global frequency-ranked block order.
        global_order = self._rank_blocks(blocks, ser, all_rows)
        # Candidate B: original column order (as blocks).
        orig_order = blocks[:]
        # Candidate C: bounded conditional partition tree.
        if len(blocks) <= 60:
            max_depth = min(len(blocks), 10)
            cond = self._cond_group(
                blocks, ser, all_rows, 0, max_depth, global_order
            )
            cond_orders = [cond[r] for r in all_rows]
        else:
            cond_orders = [global_order] * n_rows

        candidates = [
            [global_order] * n_rows,
            [orig_order] * n_rows,
            cond_orders,
        ]

        # Candidate D: greedy append order by real Trie score (bounded).
        greedy = self._greedy_order(blocks, ser, all_rows)
        if greedy is not None:
            candidates.append([greedy] * n_rows)

        # Candidate E: self-similarity ranked order (bounded, cheap).
        if len(blocks) <= 40:
            ss_order = self._selfsim_order(blocks, ser, all_rows)
            candidates.append([ss_order] * n_rows)

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