# EVOLVE-BLOCK-START
import pandas as pd
from solver import Algorithm
from typing import Tuple, List
from collections import Counter


class Evolved(Algorithm):
    """
    Simple, correct reordering aimed at character-Trie prefix reuse.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    # ---------------- serialization helpers ----------------

    @staticmethod
    def _serialize_column(s: pd.Series) -> List[str]:
        """Serialized string per cell: missing -> '', else str(value).
        Does not modify stored values."""
        vals = s.tolist()
        mask = s.isna().tolist()
        out = []
        for v, m in zip(vals, mask):
            if m:
                out.append("")
            else:
                if isinstance(v, float) and v == int(v) and not pd.isna(v):
                    # keep str() semantics identical to evaluator: str(v)
                    out.append(str(v))
                else:
                    out.append(str(v))
        return out

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Longest common prefix via binary search on slice equality (C-speed)."""
        lo, hi = 0, min(len(a), len(b))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    @classmethod
    def _ideal_reuse(cls, row_strings: List[str]) -> int:
        """Ideal Trie reuse = sum of adjacent LCPs of sorted strings."""
        if len(row_strings) < 2:
            return 0
        rs = sorted(row_strings)
        total = 0
        prev = rs[0]
        for cur in rs[1:]:
            if cur == prev:
                total += len(cur)
            else:
                total += cls._lcp(prev, cur)
            prev = cur
        return total

    # ---------------- main API ----------------

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
        # Honor column merges with existing API semantics.
        work = df
        if col_merge and hasattr(self, "merging_columns"):
            for group in col_merge:
                cols = [c for c in work.columns if c in group]
                if len(cols) > 1:
                    try:
                        work = self.merging_columns(work, cols, prepended=False)
                    except Exception:
                        pass

        ncols = len(work.columns)
        nrows = len(work)
        if ncols == 0 or nrows == 0:
            order = list(work.columns)
            out = work.copy()
            if ncols and hasattr(out, "reset_index"):
                try:
                    out = out.reset_index(drop=True)
                except Exception:
                    pass
            return out, [list(order) for _ in range(nrows)]

        columns = list(work.columns)
        col_idx = {c: i for i, c in enumerate(columns)}

        # Serialize every cell once (scoring representation only).
        ser = {c: self._serialize_column(work[c]) for c in columns}

        # Column scores: sum over distinct values of len(v)*count*(count-1)
        score_len = {}
        score_cnt = {}
        total_chars = 0
        for c in columns:
            cnt = Counter(ser[c])
            s_len = 0
            s_cnt = 0
            for v, k in cnt.items():
                s_len += len(v) * k * (k - 1)
                s_cnt += k * (k - 1)
            score_len[c] = s_len
            score_cnt[c] = s_cnt
            total_chars += sum(len(x) for x in ser[c])

        # Candidate orders (deterministic tie-break by original position).
        cand_len = sorted(columns, key=lambda c: (-score_len[c], col_idx[c]))
        cand_cnt = sorted(columns, key=lambda c: (-score_cnt[c], -score_len[c], col_idx[c]))
        candidates = []
        for cand in (cand_len, cand_cnt, columns):
            if cand not in candidates:
                candidates.append(cand)

        # Bounded selection: only run full LCP scoring when affordable.
        budget_chars = 8_000_000
        est = total_chars * len(candidates)
        if est <= budget_chars and nrows > 0:
            best_order, best_score = None, -1
            for cand in candidates:
                row_strings = ["".join(ser[c][i] for c in cand) for i in range(nrows)]
                sc = self._ideal_reuse(row_strings)
                if sc > best_score:
                    best_score, best_order = sc, cand
        else:
            best_order = candidates[0]

        # Build output: same rows, same values, columns permuted globally.
        reordered = work.loc[:, best_order].copy()
        try:
            reordered = reordered.reset_index(drop=True)
        except Exception:
            pass

        # Per-row column orderings consistent with returned data.
        order_list = list(best_order)
        column_orderings = [list(order_list) for _ in range(nrows)]

        return reordered, column_orderings

# EVOLVE-BLOCK-END