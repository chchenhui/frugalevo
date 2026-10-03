# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List


class Evolved(Algorithm):
    """
    Character-Trie prefix-reuse reordering.

    Objective: maximize total matched prefix characters when serialized rows
    ("".join of str cells, missing -> "") are inserted into a shared Trie.
    For a fixed set of strings this equals the sum of adjacent LCPs of the
    sorted strings. We build a few bounded candidate row/column orderings,
    score each with that exact objective, and return the best.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lcp(a: str, b: str) -> int:
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

    def _trie_score(self, strings: List[str]) -> int:
        """Exact serial-Trie reuse = sum of adjacent LCPs of sorted strings."""
        if len(strings) <= 1:
            return 0
        s = sorted(strings)
        total = 0
        prev = s[0]
        for cur in s[1:]:
            total += self._lcp(prev, cur)
            prev = cur
        return total

    @staticmethod
    def _serialize_series(s: pd.Series) -> np.ndarray:
        """Cell -> '' when missing, else str(value). Scoring representation only."""
        obj = s.astype(object)
        mask = s.isna().values
        out = np.empty(len(s), dtype=object)
        vals = obj.values
        for i in range(len(s)):
            v = vals[i]
            if mask[i]:
                out[i] = ""
            else:
                out[i] = "" if v is None else str(v)
        return out

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

        # ---------- trivial cases ----------
        if df is None or df.shape[0] == 0 or df.shape[1] == 0:
            return df.copy(), [list(df.columns)] * len(df) if df is not None and len(df) else []

        # ---------- honor col_merge: concatenate groups into one column ----------
        work = df.copy()
        merged_sources = {}  # merged col name -> list of source cols
        for group in col_merge:
            group = [c for c in group if c in work.columns]
            if len(group) >= 2:
                name = "|".join(group)
                if name in work.columns:
                    name = name + "|merged"
                cells = work[group].astype(object)
                new_vals = []
                ser = {c: self._serialize_series(work[c]) for c in group}
                for i in range(len(work)):
                    new_vals.append("".join(ser[c][i] for c in group))
                pos = work.columns.get_loc(group[0])
                work = work.drop(columns=group)
                work.insert(pos, name, pd.Series(new_vals, index=work.index))
                merged_sources[name] = group

        columns = list(work.columns)
        n_rows, n_cols = work.shape
        raw_values = [list(work.iloc[r]) for r in range(n_rows)]

        # ---------- serialize / factorize each column once ----------
        cells = np.empty((n_rows, n_cols), dtype=object)
        codes = []
        uniques = []
        counts = []
        vlengths = []
        for j, c in enumerate(columns):
            s = work[c]
            ser = self._serialize_series(s)
            cells[:, j] = ser
            code, uniq = pd.factorize(ser, sort=True)
            codes.append(code.astype(np.int64))
            k = len(uniq)
            uniques.append(uniq)
            cnt = np.bincount(codes[j], minlength=k) if k else np.zeros(0, dtype=np.int64)
            counts.append(cnt)
            # length of each unique serialized value
            vl = np.empty(k, dtype=np.int64)
            for i in range(k):
                v = uniq[i]
                vl[i] = len(v)
            vlengths.append(vl)

        # global column score: sum over values of len(v)*cnt*(cnt-1)
        gscore = np.zeros(n_cols, dtype=np.int64)
        avg_len = np.zeros(n_cols, dtype=float)
        for j in range(n_cols):
            cnt = counts[j]
            if len(cnt):
                gscore[j] = int(np.sum(vlengths[j] * cnt * np.maximum(cnt - 1, 0)))
                avg_len[j] = float(np.sum(vlengths[j] * cnt)) / n_rows

        def order_by_score() -> List[int]:
            return sorted(range(n_cols), key=lambda j: (-gscore[j], columns[j]))

        def order_by_length() -> List[int]:
            return sorted(range(n_cols), key=lambda j: (-avg_len[j], -gscore[j], columns[j]))

        def rows_of(order: List[int]) -> List[List[int]]:
            return [list(order) for _ in range(n_rows)]

        def serialize_row(r: int, order: List[int]) -> str:
            return "".join(cells[r, j] for j in order)

        # ---------- candidate A: global repetition-ranked order ----------
        order_a = order_by_score()
        cand_a = rows_of(order_a)

        # ---------- candidate B: length-then-repetition order ----------
        order_b = order_by_length()
        cand_b = rows_of(order_b)

        # ---------- candidate C: conditional per-row grouping tree ----------
        max_depth = 5
        if row_stop is not None:
            try:
                max_depth = min(max_depth, int(row_stop))
            except Exception:
                pass
        if col_stop is not None:
            try:
                max_depth = min(max_depth, int(col_stop))
            except Exception:
                pass
        max_depth = max(1, max_depth)
        cand_c = rows_of(order_a)  # default; filled by tree below
        try:
            cand_c = self._conditional_tree(
                n_rows, n_cols, codes, counts, vlengths, gscore,
                order_a, max_depth,
            )
        except Exception:
            cand_c = rows_of(order_a)

        # ---------- score candidates exactly, pick best ----------
        def score(rows_orders: List[List[int]]) -> Tuple[int, List[str]]:
            strs = [serialize_row(r, rows_orders[r]) for r in range(n_rows)]
            return self._trie_score(strs), strs

        best_orders = cand_a
        best_score = -1
        for cand in (cand_a, cand_b, cand_c):
            try:
                sc, _ = score(cand)
            except Exception:
                continue
            if sc > best_score:
                best_score = sc
                best_orders = cand

        # ---------- build output with per-row permutations ----------
        data = []
        for r in range(n_rows):
            data.append([raw_values[r][j] for j in best_orders[r]])
        out = pd.DataFrame(data, index=df.index, columns=columns)
        out = out.astype(object)

        orderings = [[columns[j] for j in best_orders[r]] for r in range(n_rows)]
        return out, orderings

    # ------------------------------------------------------------------ #
    # conditional prefix partition tree (bounded)
    # ------------------------------------------------------------------ #
    def _conditional_tree(
        self, n_rows, n_cols, codes, counts, vlengths, gscore,
        tail_order: List[int], max_depth: int,
    ) -> List[List[int]]:
        """Per-row orderings: within each group pick the remaining column with
        the largest length-weighted pair repetition, partition on its value,
        recurse. Deterministic cheap tail when branching stops."""
        per_row = [None] * n_rows

        # limit candidate columns per node on very wide tables
        cand_limit = 64
        ranked = sorted(range(n_cols), key=lambda j: (-gscore[j], j))

        def build(rows: List[int], used: List[int], depth: int):
            remaining = [j for j in ranked[:cand_limit] if j not in used]
            tail = list(used) + [j for j in tail_order if j not in used]
            if depth >= max_depth or len(rows) <= 3 or not remaining:
                for r in rows:
                    per_row[r] = list(tail)
                return
            # choose column with best gain inside this group
            best_j, best_gain = -1, 0
            for j in remaining:
                cc = codes[j][rows]
                cnt = np.bincount(cc, minlength=len(counts[j]))
                gain = int(np.sum(vlengths[j] * cnt * np.maximum(cnt - 1, 0)))
                if gain > best_gain:
                    best_gain, best_j = gain, j
            if best_j < 0 or best_gain <= 0:
                for r in rows:
                    per_row[r] = list(tail)
                return
            new_used = used + [best_j]
            cc = codes[best_j][rows]
            order_vals = sorted(set(cc.tolist()))
            for v in order_vals:
                sub = [rows[i] for i in range(len(rows)) if cc[i] == v]
                build(sub, new_used, depth + 1)

        build(list(range(n_rows)), [], 0)
        for r in range(n_rows):
            if per_row[r] is None:
                per_row[r] = list(tail_order)
        return per_row
# EVOLVE-BLOCK-END