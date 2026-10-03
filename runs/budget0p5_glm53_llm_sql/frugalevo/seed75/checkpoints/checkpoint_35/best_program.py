# EVOLVE-BLOCK-START
import math
import time
import itertools
import pandas as pd
from solver import Algorithm
from typing import Tuple, List, Dict, Optional
from collections import Counter


class Evolved(Algorithm):
    """
    Prefix-caching reorderer: heuristic construction selection, bounded
    exhaustive top-k block permutation enumeration, then exact-objective
    swap/reinsertion local search over atomic blocks.

    The evaluator serializes each row as "".join(fillna('').astype(str)) and shares
    characters through a serial character Trie. For a fixed multiset of row strings,
    ideal reuse is insertion-order independent and equals the sum of LCP lengths
    between lexicographically adjacent sorted strings.

    Pipeline:
      1. Serialize cells once (exact evaluator semantics).
      2. Score a few cheap heuristic column orders exactly once each; keep best.
      3. Bounded exhaustive construction (subset-permutation-enumeration): rank
         atomic blocks by sum(len(v)*cnt*(cnt-1)); pick the largest k <= 7 with
         k! <= 5040 and k!*avg_row_len*n <= 20M scored characters; enumerate all
         k! orderings of the top-k blocks with the tail fixed in incumbent order;
         every full ordering is scored exactly once and joins the strict-
         improvement pool (deterministic k-shrink on budget breach, 5s cap).
      4. Local search on the exact objective over atomic blocks: up to 2 passes
         of adjacent swaps plus single-block reinsertions, strict improvements.
      5. Emit the best (reordered DataFrame, per-row column orderings).
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

        Identical adjacent strings contribute len(s) directly, so duplicate-row
        runs are handled exactly and cheaply.
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
               *conditioned on the lead value's groups* (bounded O(n*m) pass).
        Returns deduplicated column orders in construction order.
        """
        m = len(col_names)
        cands: List[List[str]] = [list(col_names)]
        if m < 2:
            return cands
        n = len(col_strings[0]) if m else 0

        weighted: List[float] = []
        best_i, best_w = 0, -1.0
        for i, vals in enumerate(col_strings):
            counts = Counter(vals)
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

    # ---------- scoring of a full column order ----------

    @staticmethod
    def _score_order(
        order: List[str], col_lists_by_name: Dict[str, List[str]], trie_reuse
    ) -> Tuple[int, List[str]]:
        """Score one full column order: join rows column-wise (C-level map),
        sort, sum adjacent LCPs. Returns (score, row_strings) so the winning
        order's row strings can be reused for the row permutation."""
        col_lists = [col_lists_by_name[c] for c in order]
        rows = list(map("".join, zip(*col_lists)))
        return trie_reuse(sorted(rows)), rows

    # ---------- bounded exhaustive top-k block permutation ----------

    @staticmethod
    def _exhaustive_permutations(
        blocks: List[List[str]],
        block_weight: List[float],
        col_strings_by_name: Dict[str, List[str]],
        score,
        trie_reuse,
        deadline: float,
        max_chars: int = 20_000_000,
        max_perms: int = 5040,
    ) -> Tuple[Optional[List[str]], int, Optional[List[str]]]:
        """Enumerate all orderings of the k most influential blocks, rest fixed.

        Blocks are ranked by the length-weighted pair-repetition statistic; the
        largest k <= 7 with k! <= max_perms and k!*avg_row_len*n <= max_chars is
        fully permuted while the remaining blocks keep the incumbent's relative
        order. avg_row_len is total serialized characters / n (average row
        length), matching the parent's semantics so k selection is identical.
        Each full ordering is scored exactly once; a wall-clock deadline
        truncates deterministically. On budget breach k shrinks by one and the
        enumeration restarts deterministically.
        Returns (best_perm_order, best_score, best_rows) or (None, -1, None).
        """
        b = len(blocks)
        if b < 2:
            return None, -1, None
        n = len(next(iter(col_strings_by_name.values())))
        total_chars = sum(len(v) for lst in col_strings_by_name.values() for v in lst)
        avg_len = total_chars / max(n, 1)
        # Deterministic block ranking: weight desc, then first-column name asc.
        idx_rank = sorted(range(b), key=lambda i: (-block_weight[i], blocks[i][0]))
        k_max = min(b, 7)
        while k_max >= 2:
            if math.factorial(k_max) <= max_perms and (
                math.factorial(k_max) * avg_len * n <= max_chars
            ):
                break
            k_max -= 1
        if k_max < 2:
            return None, -1, None

        top_idx = idx_rank[:k_max]
        top_blocks = [blocks[i] for i in top_idx]
        top_set = set(top_idx)
        tail: List[str] = []
        for i in range(b):
            if i not in top_set:
                tail.extend(blocks[i])

        best_score = -1
        best_order: Optional[List[str]] = None
        best_rows: Optional[List[str]] = None
        for perm in itertools.permutations(range(k_max)):
            if time.time() > deadline:
                break
            trial: List[str] = []
            for p in perm:
                trial.extend(top_blocks[p])
            trial.extend(tail)
            s, rows = score(trial, col_strings_by_name, trie_reuse)
            if s > best_score:
                best_score, best_order, best_rows = s, trial, rows
        return best_order, best_score, best_rows

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

        Heuristic candidates are scored exactly once; the best incumbent is then
        improved by (a) bounded exhaustive enumeration of the k! orderings of the
        most influential atomic blocks (tail fixed in incumbent order, strict
        improvement pool, deterministic k-shrink on budget breach, 5-second
        wall-clock cap), and (b) local search over adjacent block swaps and
        single-block reinsertions on the exact sorted-string LCP objective. Only
        strict improvements are accepted, so the final result is >= the
        incumbent. Merge groups stay atomic (blocks are super-columns and only
        whole blocks move).
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
        score = self._score_order

        # Phase 1: score each heuristic candidate once, keep the best incumbent.
        best_score = -1
        best_order: Optional[List[str]] = None
        best_rows: Optional[List[str]] = None
        for order in self._candidate_orders(col_names, col_strings):
            full_order = expand(order)
            s, rows = score(full_order, col_strings_by_name, trie_reuse)
            if s > best_score:
                best_score, best_order, best_rows = s, full_order, rows
            if time.time() - start > 60:
                break

        # Build atomic blocks (super-columns) aligned with the incumbent order.
        blocks: List[List[str]] = []
        placed: set = set()
        for g in atomic_groups:
            if all(mn in best_order for mn in g):
                blocks.append(list(g))
                placed.update(g)
        for c in best_order:
            if c not in placed:
                blocks.append([c])
                placed.add(c)

        b = len(blocks)
        total_chars = sum(len(v) for lst in col_strings for v in lst)

        # Phase 1.5: bounded exhaustive enumeration of top-k block permutations.
        # Parent-faithful budgets: k! <= 5040, k!*avg_row_len*n <= 20M chars,
        # 5-second wall-clock cap anchored at program start; k shrinks
        # deterministically on breach. Every full ordering is scored exactly
        # once and joins the strict-improvement pool.
        perm_order, p_score, p_rows = None, -1, None
        if b >= 2 and n >= 2 and total_chars * b <= 20_000_000:
            block_weight: List[float] = []
            for blk in blocks:
                w = 0.0
                for c in blk:
                    counts = Counter(col_strings_by_name[c])
                    w += sum(len(v) * cc * (cc - 1) for v, cc in counts.items())
                block_weight.append(w)
            perm_deadline = start + 5.0
            perm_order, p_score, p_rows = self._exhaustive_permutations(
                blocks, block_weight, col_strings_by_name, score, trie_reuse,
                perm_deadline,
            )
            if perm_order is not None and p_score > best_score:
                best_score, best_order, best_rows = p_score, perm_order, p_rows

        # Phase 2: bounded local search on the exact objective over atomic
        # blocks. Moves: adjacent block swaps and single-block reinsertions.
        # Strict improvements only; caps keep the runtime term intact.
        if b >= 2 and n >= 2 and total_chars * b <= 20_000_000:
            search_start = time.time()

            def flatten(blks: List[List[str]]) -> List[str]:
                out: List[str] = []
                for blk in blks:
                    out.extend(blk)
                return out

            for _pass in range(2):
                improved = False
                evals = 0
                max_evals = 3 * b

                # Adjacent swaps.
                for i in range(b - 1):
                    if evals >= max_evals or time.time() - search_start > 4.0:
                        break
                    blocks[i], blocks[i + 1] = blocks[i + 1], blocks[i]
                    trial = flatten(blocks)
                    s2, r2 = score(trial, col_strings_by_name, trie_reuse)
                    evals += 1
                    if s2 > best_score:
                        best_score, best_order, best_rows = s2, trial, r2
                        improved = True
                    else:
                        blocks[i], blocks[i + 1] = blocks[i + 1], blocks[i]

                # Single-block reinsertions: remove block i, try each position.
                for i in range(b):
                    if evals >= max_evals or time.time() - search_start > 4.0:
                        break
                    blk = blocks.pop(i)
                    best_j = -1
                    best_s = best_score
                    best_r: Optional[List[str]] = None
                    for j in range(b):
                        if j == i:
                            continue
                        blocks.insert(j, blk)
                        trial = flatten(blocks)
                        s2, r2 = score(trial, col_strings_by_name, trie_reuse)
                        evals += 1
                        blocks.pop(j)
                        if s2 > best_s:
                            best_s, best_j, best_r = s2, j, r2
                        if evals >= max_evals or time.time() - search_start > 4.0:
                            break
                    if best_j >= 0:
                        blocks.insert(best_j, blk)
                        best_score, best_order, best_rows = best_s, flatten(blocks), best_r
                        improved = True
                    else:
                        blocks.insert(i, blk)

                if not improved or evals >= max_evals or time.time() - search_start > 4.0:
                    break

        idx = sorted(range(n), key=best_rows.__getitem__)
        out = work.iloc[idx][best_order].reset_index(drop=True)
        orderings = [list(best_order) for _ in range(len(out))]
        return out, orderings
# EVOLVE-BLOCK-END