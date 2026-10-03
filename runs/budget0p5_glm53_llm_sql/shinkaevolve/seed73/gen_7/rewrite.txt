# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List
from collections import Counter


class Evolved(Algorithm):
    """
    LCP/Trie-aware reordering:
    - Serializes each cell exactly as the scorer does (NaN/None -> "").
    - Builds a few cheap candidate row-column orderings.
    - Scores each candidate with the exact serial character-Trie reuse
      (sum of adjacent LCPs over the sorted serialized rows).
    - Returns the best candidate, preserving every row and cell value.
    """

    def _serialize(self, df: pd.DataFrame):
        """Return list of lists of serialized cell strings (NaN -> '')."""
        rows = []
        for row in df.itertuples(index=False, name=None):
            rows.append(["" if (v is None or (isinstance(v, float) and np.isnan(v)))
                         else str(v) for v in row])
        return rows

    def _lcp(self, a: str, b: str) -> int:
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

    def _trie_score(self, strings) -> int:
        """Exact serial-Trie reuse: sum of adjacent LCPs after sorting."""
        if not strings:
            return 0
        ss = sorted(strings)
        total = 0
        prev = None
        for s in ss:
            if prev is not None:
                total += self._lcp(prev, s)
            prev = s
        return total

    def _apply_dep(self, order, one_way_dep):
        """Place each dependent column right after its source column."""
        order = list(order)
        for src, dst in one_way_dep:
            if src in order and dst in order:
                order.remove(dst)
                order.insert(order.index(src) + 1, dst)
        return order

    def _global_freq_order(self, df, col_names):
        """Rank columns by sum over values v of len(v)*count*(count-1)."""
        scores = {}
        for c in col_names:
            vals = ["" if (v is None or (isinstance(v, float) and np.isnan(v)))
                    else str(v) for v in df[c].tolist()]
            cnt = Counter(vals)
            scores[c] = sum(len(v) * n * (n - 1) for v, n in cnt.items())
        return sorted(col_names, key=lambda c: (-scores[c], str(c)))

    def _partition_order(self, df, col_names, ser_rows, row_stop, col_stop, early_stop):
        """
        Conditional prefix partition tree: inside each group pick the remaining
        column with the best length-weighted pair repetition, partition rows by
        that column's value, and order suffixes separately per group.
        Returns per-row column orders (list of lists of column names).
        """
        n = len(df)
        idx_cols = {c: i for i, c in enumerate(col_names)}
        codes = {c: {} for c in col_names}  # value -> code per column
        code_int = np.zeros((n, len(col_names)), dtype=np.int64)
        for j, c in enumerate(col_names):
            m = {}
            col_codes = np.empty(n, dtype=np.int64)
            for i in range(n):
                v = ser_rows[i][j]
                if v not in m:
                    m[v] = len(m)
                col_codes[i] = m[v]
            code_int[:, j] = col_codes
            codes[c] = {v: k for k, v in enumerate(m)}  # not needed later
            codes[c] = m

        max_depth = 12
        if col_stop is not None:
            try:
                max_depth = min(max_depth, int(col_stop))
            except Exception:
                pass
        min_group = 2
        if row_stop is not None:
            try:
                min_group = max(min_group, n - int(row_stop) + 1) if row_stop and int(row_stop) < n else min_group
            except Exception:
                pass
        thresh = early_stop if early_stop and early_stop > 0 else 1

        orders = [None] * n  # per-row prefix (fixed part), list of col indices
        done = [False] * n

        def rec(row_ids, remaining, depth):
            if depth >= max_depth or len(row_ids) < max(min_group, 2) or not remaining:
                return
            best_col, best_gain = None, thresh
            for j in remaining:
                col_vals = code_int[row_ids, j]
                # length-weighted pair repetition: sum len(v)*(cnt choose 2)
                uniq, cnt = np.unique(col_vals, return_counts=True)
                if len(uniq) <= 1:
                    continue
                gain = 0
                for u, c in zip(uniq, cnt):
                    if c > 1:
                        v = ser_rows[row_ids[0]][j]  # representative may differ; use code->val map
                        gain += 0
                # use average serialized length per code for weighting
                # compute per-code representative length from first occurrence
                # (cheap: use global length per value via codes map inverse)
                # fallback: use mean length of column values in this group
                lens = np.array([len(ser_rows[r][j]) for r in row_ids[: min(len(row_ids), 256)]])
                mean_len = float(lens.mean()) if len(lens) else 0.0
                gain = mean_len * float(np.sum(cnt * (cnt - 1))) / 2.0
                if gain > best_gain:
                    best_gain, best_col = gain, j
            if best_col is None:
                return
            groups = {}
            for r in row_ids:
                groups.setdefault(int(code_int[r, best_col]), []).append(r)
            if len(groups) <= 1:
                return
            for rids in groups.values():
                for r in rids:
                    if orders[r] is None:
                        orders[r] = []
                    orders[r] = orders[r] + [best_col] if orders[r] is not None else [best_col]
                rem = [j for j in remaining if j != best_col]
                rec(rids, rem, depth + 1)

        all_rows = list(range(n))
        rec(all_rows, list(range(len(col_names))), 0)

        # Build per-row orders: fixed prefix + remaining columns in global order
        global_order = self._global_freq_order(df, col_names)
        global_idx = [idx_cols[c] for c in global_order]
        result = []
        for i in range(n):
            if orders[i]:
                pre = list(dict.fromkeys(orders[i]))
                rest = [j for j in global_idx if j not in pre]
                result.append(pre + rest)
            else:
                result.append(list(global_idx))
        return result

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
        work = df.copy()
        merged = False
        # Honor required column merges via the parent API semantics.
        if col_merge:
            if hasattr(self, "merging_columns"):
                all_cols = [c for c in work.columns]
                final_col_order = all_cols
                try:
                    work = self.merging_columns(work, final_col_order, prepended=False)
                    merged = True
                except Exception:
                    merged = False
            if not merged:
                # Fallback: keep data unchanged but group merged columns adjacently.
                merge_cols = [c for grp in col_merge for c in grp if c in work.columns]
                other = [c for c in work.columns if c not in merge_cols]
                work = work[other + merge_cols]

        col_names = list(work.columns)
        n = len(work)
        if n == 0:
            return work.copy(), []

        ser_rows = self._serialize(work)

        # Enforce one-way dependencies on candidate orders.
        deps = []
        for d in (one_way_dep or []):
            c1 = [c for c in col_names if d[0] in str(c)]
            c2 = [c for c in col_names if d[1] in str(c)]
            if len(c1) == 1 and len(c2) == 1:
                deps.append((c1[0], c2[0]))

        # Candidate 1: original column order.
        cand1 = self._apply_dep(col_names, deps)

        # Candidate 2: global frequency-ranked order.
        cand2 = self._apply_dep(self._global_freq_order(work, col_names), deps)

        # Candidate 3: conditional partition tree (per-row orders).
        try:
            per_row_idx = self._partition_order(work, col_names, ser_rows,
                                                row_stop, col_stop, early_stop)
            # apply deps within each row order
            cand3_orders = []
            for oi in per_row_idx:
                order = [col_names[j] for j in oi]
                cand3_orders.append(self._apply_dep(order, deps))
        except Exception:
            cand3_orders = None

        def build_strings(order_names):
            pos = {c: i for i, c in enumerate(order_names)}
            return ["".join(row[pos[c]] for c in order_names) for row in ser_rows]

        best_orders = [cand1] * n
        best_score = self._trie_score(build_strings(cand1))
        s2 = self._trie_score(build_strings(cand2))
        if s2 > best_score:
            best_score = s2
            best_orders = [cand2] * n
        if cand3_orders is not None:
            s3 = self._trie_score(["".join(row[c] for c in ord_)
                                   for row, ord_ in zip(ser_rows,
                                                        [[col_names.index(c) for c in o]
                                                         for o in cand3_orders])])
            if s3 > best_score:
                best_score = s3
                best_orders = cand3_orders

        # Build the output DataFrame: per-row permutation of the serialized-free
        # original values (preserve stored cell values exactly).
        pos_maps = [{c: i for i, c in enumerate(o)} for o in best_orders]
        out_data = []
        src_cols = list(work.columns)
        for i in range(n):
            row_vals = [work.iat[i, j] for j in range(len(src_cols))]
            pos = pos_maps[i]
            out_data.append([row_vals[pos[c]] for c in col_names])

        out_df = pd.DataFrame(out_data, columns=col_names)
        if out_df.shape[1] > 0:
            try:
                out_df = out_df.astype(object)
            except Exception:
                pass

        # Same order for every row if uniform; otherwise per-row lists.
        if all(o == best_orders[0] for o in best_orders):
            column_orderings = [list(best_orders[0])] * n
        else:
            column_orderings = [list(o) for o in best_orders]

        # Shape checks (col_merge may legitimately change column count).
        if not merged:
            assert out_df.shape == df.shape, \
                f"Shape mismatch: {out_df.shape} vs {df.shape}"

        return out_df, column_orderings

# EVOLVE-BLOCK-END