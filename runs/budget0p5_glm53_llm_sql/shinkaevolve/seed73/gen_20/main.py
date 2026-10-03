# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
from solver import Algorithm
from typing import Tuple, List, Dict, Optional


class Evolved(Algorithm):
    """
    Row-wise column reordering optimized for character-level prefix reuse
    (prompt prefix caching). Candidates are constructed cheaply, then the
    exact serial Trie objective (sum of adjacent LCPs over sorted serialized
    rows) selects the best one. Data is never modified, only permuted
    within each row.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.dep_graph = None
        self.num_rows = 0
        self.num_cols = 0

    # ------------------------------------------------------------------
    # Serialization helpers (matches evaluator: fillna("") then astype(str))
    # ------------------------------------------------------------------
    @staticmethod
    def _serialize_column(s: pd.Series) -> List[str]:
        obj = s.astype(object)
        try:
            mask = pd.isna(obj)
            if isinstance(mask, pd.Series):
                obj = obj.where(~mask, "")
        except (TypeError, ValueError):
            pass
        out = []
        for v in obj.tolist():
            if v is None:
                out.append("")
            elif isinstance(v, str):
                out.append(v)
            else:
                out.append(str(v))
        return out

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        """Longest common prefix using C-speed slice equality (binary search)."""
        if a == b:
            return len(a)
        n = min(len(a), len(b))
        if n == 0:
            return 0
        if a[:n] == b[:n]:
            return n
        lo, hi = 0, n - 1  # a[:lo]==b[:lo]; a[:hi+1]!=b[:hi+1]
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _score_strings(self, strings: List[str]) -> int:
        """Exact serial-Trie reuse: sum of adjacent LCPs of sorted strings."""
        n = len(strings)
        if n <= 1:
            return 0
        ss = sorted(strings)
        total = 0
        prev = ss[0]
        for s in ss[1:]:
            total += self._lcp(prev, s)
            prev = s
        return total

    # ------------------------------------------------------------------
    # Candidate 2: global frequency-ranked column order
    # ------------------------------------------------------------------
    def _global_order(
        self,
        col_names: List[str],
        global_score: Dict[str, int],
    ) -> List[str]:
        return sorted(col_names, key=lambda c: (-global_score.get(c, 0), str(c)))

    # ------------------------------------------------------------------
    # Candidate 3: conditional partition tree -> per-row orderings
    # ------------------------------------------------------------------
    def _tree_orderings(
        self,
        n: int,
        col_names: List[str],
        col_codes: Dict[str, np.ndarray],
        code_len: Dict[str, List[int]],
        global_score: Dict[str, int],
        early_stop: int,
        max_depth: int,
        top_k: int = 32,
        max_nodes: int = 4096,
    ) -> List[List[str]]:
        orderings: List[Optional[List[str]]] = [None] * n
        node_budget = [max_nodes]

        def finish(rows: List[int], prefixes: List[List[str]], remaining: List[str]):
            for k, i in enumerate(rows):
                orderings[i] = prefixes[k] + list(remaining)

        def recurse(rows: List[int], prefixes: List[List[str]], remaining: List[str], depth: int):
            if node_budget[0] <= 0:
                finish(rows, prefixes, remaining)
                return
            if not remaining or depth >= max_depth or len(rows) <= 1:
                finish(rows, prefixes, remaining)
                return

            cands = remaining
            if len(cands) > top_k:
                cands = sorted(
                    remaining, key=lambda c: (-global_score.get(c, 0), str(c))
                )[:top_k]

            best_col = None
            best_gain = 0
            best_groups = None
            for c in cands:
                codes = col_codes[c]
                cnt: Dict[int, int] = {}
                for i in rows:
                    code = int(codes[i])
                    cnt[code] = cnt.get(code, 0) + 1
                gain = 0
                for code, k in cnt.items():
                    if k > 1:
                        gain += code_len[c][code] * k * (k - 1)
                if gain > best_gain:
                    groups: Dict[int, List[int]] = {}
                    for i in rows:
                        groups.setdefault(int(codes[i]), []).append(i)
                    best_gain = gain
                    best_col = c
                    best_groups = groups

            if (
                best_col is None
                or best_gain <= max(early_stop, 0)
            ):
                finish(rows, prefixes, remaining)
                return

            node_budget[0] -= 1
            rem = [c for c in remaining if c != best_col]
            if len(best_groups) == 1:
                # Constant column within this group: pin it at the front.
                new_prefixes = [prefixes[k] + [best_col] for k in range(len(rows))]
                recurse(rows, new_prefixes, rem, depth + 1)
            else:
                pos = {i: k for k, i in enumerate(rows)}
                for grp in best_groups.values():
                    grp_rows = grp
                    grp_prefixes = [prefixes[pos[i]] + [best_col] for i in grp_rows]
                    recurse(grp_rows, grp_prefixes, rem, depth + 1)

        recurse(list(range(n)), [[] for _ in range(n)], list(col_names), 0)

        result: List[List[str]] = []
        for o in orderings:
            result.append(o if o is not None else list(col_names))
        return result

    # ------------------------------------------------------------------
    # Row strings for a candidate (per-row orderings)
    # ------------------------------------------------------------------
    @staticmethod
    def _row_strings(
        orderings: List[List[str]], ser_cols: Dict[str, List[str]]
    ) -> List[str]:
        out = []
        for i, order in enumerate(orderings):
            parts = [ser_cols[c][i] for c in order]
            out.append("".join(parts))
        return out

    # ------------------------------------------------------------------
    # One-way dependency: place dependent column right after its source
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_dep(order: List[str], dep_pairs: List[Tuple[str, str]]) -> List[str]:
        for a, b in dep_pairs:
            if a in order and b in order:
                order = [c for c in order if c != b]
                idx = order.index(a)
                order = order[: idx + 1] + [b] + order[idx + 1:]
        return order

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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
        # ---- honor column merges (existing API semantics) ----
        work = df.copy()
        if col_merge:
            for group in col_merge:
                cols = [c for c in work.columns if c in group]
                if len(cols) >= 2:
                    work = self.merging_columns(work, cols, prepended=False)

        n, m = work.shape
        col_names = [str(c) for c in work.columns]
        if n == 0 or m == 0:
            return work.copy(), [[] for _ in range(n)]

        # ---- one-way dependency pairs (resolved on final column names) ----
        dep_pairs: List[Tuple[str, str]] = []
        for dep in (one_way_dep or []):
            try:
                a, b = dep[0], dep[1]
            except (TypeError, IndexError):
                continue
            ca = [c for c in col_names if a in c]
            cb = [c for c in col_names if b in c]
            if len(ca) == 1 and len(cb) == 1:
                dep_pairs.append((ca[0], cb[0]))

        # ---- precompute per-column serialized strings and integer codes ----
        ser_cols: Dict[str, List[str]] = {}
        col_codes: Dict[str, np.ndarray] = {}
        code_len: Dict[str, List[int]] = {}
        global_score: Dict[str, int] = {}

        for c in col_names:
            sers = self._serialize_column(work[c])
            ser_cols[c] = sers
            codes, uniques = pd.factorize(work[c].astype(object), sort=False)
            codes = np.asarray(codes, dtype=np.int64)
            # NaN / missing maps to code -1; remap to a valid code index.
            remap = np.where(codes < 0, len(uniques), codes)
            col_codes[c] = remap
            lens = [len(s) for s in sers]  # placeholder, replaced below
            # length per code
            clens = [0] * (len(uniques) + 1)
            for i in range(n):
                clens[int(remap[i])] = len(sers[i])
            code_len[c] = clens
            # global score: sum over values of len(v) * count * (count - 1)
            counts = np.bincount(remap, minlength=len(uniques) + 1)
            score = 0
            for code in range(len(uniques) + 1):
                k = int(counts[code])
                if k > 1:
                    score += clens[code] * k * (k - 1)
            global_score[c] = score

        # ---- candidate 1: original column order ----
        cand_orderings: List[List[List[str]]] = [ [list(col_names) for _ in range(n)] ]

        # ---- candidate 2: global frequency-ranked order ----
        g_order = self._global_order(col_names, global_score)
        cand_orderings.append([list(g_order) for _ in range(n)])

        # ---- candidate 3: conditional partition tree ----
        if m >= 2 and n >= 2:
            max_depth = col_stop if col_stop is not None else m
            max_depth = min(max_depth, m, 24)
            tree = self._tree_orderings(
                n,
                col_names,
                col_codes,
                code_len,
                global_score,
                early_stop=int(early_stop),
                max_depth=max_depth,
            )
            cand_orderings.append(tree)

        # ---- select best candidate by the exact serial-Trie objective ----
        total_chars = sum(len(s) for s in ser_cols[col_names[0]]) * max(1, m)
        score_cap = 40_000_000
        best_orderings = cand_orderings[0]
        best_score = -1
        for orderings in cand_orderings:
            if len(cand_orderings) > 1 and total_chars <= score_cap:
                strings = self._row_strings(orderings, ser_cols)
                sc = self._score_strings(strings)
            else:
                sc = 0
            if sc > best_score:
                best_score = sc
                best_orderings = orderings

        # ---- honor one-way dependencies ----
        if dep_pairs:
            best_orderings = [
                self._apply_dep(list(o), dep_pairs) for o in best_orderings
            ]

        # ---- build output: permute each row's values per its ordering ----
        col_pos = {c: i for i, c in enumerate(col_names)}
        vals = work.to_numpy(dtype=object)
        out = np.empty((n, m), dtype=object)
        for i in range(n):
            perm = [col_pos[c] for c in best_orderings[i]]
            if len(perm) != m:
                # safety: fall back to original order
                perm = list(range(m))
                best_orderings[i] = list(col_names)
            out[i] = vals[i][perm]

        final = pd.DataFrame(out, columns=list(work.columns))
        final = final.astype(object)
        final.index = work.index

        # sanity checks (never alter data)
        assert final.shape == work.shape
        self.num_rows = n
        self.num_cols = m
        return final, [list(o) for o in best_orderings]

# EVOLVE-BLOCK-END