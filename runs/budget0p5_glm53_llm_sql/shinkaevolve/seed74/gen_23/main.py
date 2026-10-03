# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List


class Evolved(Algorithm):
    """
    Reorders columns (globally, consistently for every row) to maximize
    exact character-Trie prefix reuse under the evaluator's serialization
    "".join(row.fillna("").astype(str).values).

    Strategy:
      1. Serialize every cell once (missing -> "").
      2. Build a small bounded set of candidate column orderings:
         - original column order,
         - frequency-ranked order using sum(len(v) * count * (count-1)),
         - a length-weighted repetition variant,
         - col_merge groups are kept contiguous in every candidate.
      3. Score each candidate with the exact ideal Trie reuse:
         sum of adjacent LCP lengths of the sorted serialized rows.
      4. Return the best candidate applied to the DataFrame, with a
         per-row column_orderings list consistent with the returned data.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------ #
    # serialization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _serialize_cells(values):
        """Serialize a 2D array of raw values exactly like the evaluator."""
        out = np.empty(values.shape, dtype=object)
        for i in range(values.shape[0]):
            row = values[i]
            ser = ["" if (v is None or (isinstance(v, float) and np.isnan(v))) or v is pd.NA
                   else str(v) for v in row]
            out[i] = ser
        return out

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Longest common prefix length using C-speed slice comparisons."""
        n = min(len(a), len(b))
        if n == 0:
            return 0
        if a == b:
            return n
        l = 0
        step = 1 << 16
        while step > 0:
            while l + step <= n and a[l:l + step] == b[l:l + step]:
                l += step
            step >>= 1
        # final char check
        while l < n and a[l] == b[l]:
            l += 1
        return l

    @classmethod
    def _trie_reuse(cls, row_strings) -> int:
        """Exact ideal Trie reuse: sorted adjacent LCP sum."""
        if not row_strings:
            return 0
        strings = sorted(row_strings)
        total = 0
        prev = strings[0]
        for s in strings[1:]:
            if s != prev:
                total += cls._lcp(prev, s)
                prev = s
        return total

    # ------------------------------------------------------------------ #
    # candidate constructions
    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge_aware_order(columns, order, col_merge):
        """Reorder `order` so that each col_merge group (present in df)
        appears contiguously, keeping the relative internal group order."""
        groups = []
        for grp in col_merge:
            g = [c for c in order if c in set(grp)]
            if len(g) > 1:
                groups.append(g)
        if not groups:
            return list(order)
        grouped_cols = set()
        for g in groups:
            grouped_cols.update(g)
        free = [c for c in order if c not in grouped_cols]
        # interleave: place groups where their first member appeared
        result = []
        used = set()
        for c in order:
            if c in used:
                continue
            placed = False
            for g in groups:
                if c in g:
                    result.extend(g)
                    used.update(g)
                    placed = True
                    break
            if not placed:
                result.append(c)
                used.add(c)
        return result

    @staticmethod
    def _factorize_cols(ser, n_rows, n_cols):
        """Factorize each column's serialized strings to integer codes once."""
        codes = []
        uniq_lens = []
        for j in range(n_cols):
            c, uniques = pd.factorize(ser[:, j], sort=False)
            codes.append(np.asarray(c, dtype=np.int64))
            uniq_lens.append(np.fromiter((len(u) for u in uniques),
                                         dtype=np.int64, count=len(uniques)))
        return codes, uniq_lens

    def _tree_row_orders(self, codes, uniq_lens, freq_rank, n_rows, n_cols,
                         max_depth):
        """Conditional prefix partition tree -> per-row column index orders.

        Within each row group, choose the remaining column maximizing
        sum(len(v) * cnt * (cnt-1)) (vectorized bincount over integer codes),
        place it next, partition rows by its value code, and recurse on the
        suffix inside each group. Cheap deterministic freq-ranked tail when
        branching stops.
        """
        orders = [None] * n_rows
        freq_pos = {c: p for p, c in enumerate(freq_rank)}

        def tail_order(remaining):
            return sorted(remaining, key=lambda j: freq_pos[j])

        def rec(rows, prefix, remaining, depth):
            if rows.shape[0] < 2 or depth == 0 or not remaining:
                t = list(prefix) + tail_order(remaining)
                for i in rows:
                    orders[i] = t
                return
            if len(remaining) <= 10:
                cands = remaining
            else:
                cands = sorted(remaining, key=lambda j: freq_pos[j])[:10]
            best_j = -1
            best_gain = 0
            for j in cands:
                c = codes[j][rows]
                cnt = np.bincount(c)
                lens = uniq_lens[j][:cnt.shape[0]]
                gain = int(np.sum(lens * cnt * (cnt - 1)))
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j < 0:
                t = list(prefix) + tail_order(remaining)
                for i in rows:
                    orders[i] = t
                return
            new_rem = [j for j in remaining if j != best_j]
            new_pref = prefix + [best_j]
            c = codes[best_j][rows]
            for v in np.unique(c):
                rec(rows[c == v], new_pref, new_rem, depth - 1)

        rec(np.arange(n_rows, dtype=np.int64), [], list(range(n_cols)),
            max_depth)
        return orders

    def _build_candidates(self, columns, col_stats, col_merge):
        """col_stats: dict col -> (freq_score, len_weighted_score, avg_len)."""
        orig = list(columns)

        # Candidate A: original order (merge-aware)
        cands = [self._merge_aware_order(columns, orig, col_merge)]

        # Candidate B: frequency-ranked, sum(len(v) * count * (count-1)) desc
        by_freq = sorted(columns, key=lambda c: (-col_stats[c][0], str(c)))
        cands.append(self._merge_aware_order(columns, by_freq, col_merge))

        # Candidate C: length-weighted repetition variant (count*(count-1)) desc
        by_rep = sorted(columns, key=lambda c: (-col_stats[c][1], str(c)))
        cands.append(self._merge_aware_order(columns, by_rep, col_merge))

        # Candidate D: short fields first (cheap deterministic tail)
        by_len = sorted(columns, key=lambda c: (col_stats[c][2], str(c)))
        cands.append(self._merge_aware_order(columns, by_len, col_merge))

        # dedupe while preserving order
        seen = set()
        unique = []
        for c in cands:
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
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
        # ---- preserve the input exactly ----
        original_columns = list(df.columns)
        n_rows, n_cols = df.shape

        if n_rows == 0 or n_cols == 0:
            return df.copy(), [list(original_columns) for _ in range(n_rows)]

        work = df.copy()
        work = work.astype(object)

        col_index = {c: i for i, c in enumerate(original_columns)}
        values = work.values  # object ndarray, shape (n_rows, n_cols)

        # ---- serialize once per column (cell strings) ----
        ser = np.empty((n_rows, n_cols), dtype=object)
        for j, c in enumerate(original_columns):
            col_vals = values[:, j]
            ser[:, j] = ["" if (v is None or (isinstance(v, float) and np.isnan(v)))
                         else str(v) for v in col_vals]

        total_chars = int(sum(len(s) for s in ser.ravel() if isinstance(s, str)))

        # ---- per-column statistics (factorize once) ----
        col_stats = {}
        for j, c in enumerate(original_columns):
            col_ser = ser[:, j]
            counts = {}
            for s in col_ser:
                counts[s] = counts.get(s, 0) + 1
            freq_score = 0
            rep_score = 0
            avg_len = 0.0
            for s, cnt in counts.items():
                L = len(s)
                avg_len += L * cnt
                if cnt > 1:
                    freq_score += L * cnt * (cnt - 1)
                    rep_score += cnt * (cnt - 1)
            avg_len /= max(1, n_rows)
            col_stats[c] = (freq_score, rep_score, avg_len)

        # ---- candidate column orderings ----
        candidates = self._build_candidates(original_columns, col_stats, col_merge)

        # ---- score candidates with exact Trie reuse (bounded) ----
        best_order = candidates[0]
        best_score = -1
        best_row_orders = None  # per-row index orders if the tree wins
        # bound the scoring work: skip full LCP scoring if dataset is huge
        do_score = total_chars <= 30_000_000 and len(candidates) <= 6

        if do_score:
            for cand in candidates:
                idxs = [col_index[c] for c in cand]
                # build serialized rows for this ordering
                # (join per row using precomputed cell strings)
                row_strings = ["".join([ser[i, j] for j in idxs])
                               for i in range(n_rows)]
                score = self._trie_reuse(row_strings)
                if score > best_score:
                    best_score = score
                    best_order = cand

            # ---- conditional partition tree candidate (per-row orders) ----
            # only when no col_merge constraints must be honored contiguously
            if not col_merge and n_cols <= 512 and n_rows <= 500_000:
                codes, uniq_lens = self._factorize_cols(ser, n_rows, n_cols)
                freq_rank = [c for c in sorted(
                    original_columns, key=lambda c: (-col_stats[c][0], str(c)))]
                max_depth = 12 if (n_cols >= 8 and n_rows <= 200_000) else 8
                tree_orders = self._tree_row_orders(
                    codes, uniq_lens, freq_rank, n_rows, n_cols, max_depth)
                tree_strings = [
                    "".join([ser[i, j] for j in tree_orders[i]])
                    for i in range(n_rows)
                ]
                tree_score = self._trie_reuse(tree_strings)
                if tree_score > best_score:
                    best_score = tree_score
                    best_row_orders = tree_orders
        else:
            # fall back to the frequency-ranked heuristic
            best_order = candidates[1] if len(candidates) > 1 else candidates[0]

        if best_row_orders is not None:
            # ---- per-row orders won: permute each row's values directly ----
            out_vals = np.empty((n_rows, n_cols), dtype=object)
            for i in range(n_rows):
                o = best_row_orders[i]
                rv = values[i]
                out_vals[i] = [rv[j] for j in o]
            first_names = [original_columns[j] for j in best_row_orders[0]]
            out = pd.DataFrame(out_vals, columns=first_names, index=df.index)
            assert out.shape == df.shape, "Shape mismatch after reorder"
            column_orderings = [
                [original_columns[j] for j in o] for o in best_row_orders
            ]
            return out, column_orderings

        # ---- apply the winning order (same order for every row) ----
        out = work[best_order]

        # ensure identical result: verify shape and that values are preserved
        assert out.shape == df.shape, "Shape mismatch after reorder"

        column_orderings = [list(best_order) for _ in range(n_rows)]

        # If col_merge merging semantics from parent are needed, the columns
        # are kept contiguous above; data and shape are fully preserved.
        return out, column_orderings

# EVOLVE-BLOCK-END