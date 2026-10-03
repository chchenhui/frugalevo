import numpy as np
import pandas as pd
from solver import Algorithm
from typing import Tuple, List


class Evolved(Algorithm):
    """
    Prefix-cache-oriented dataframe reorderer.

    The implementation preserves every input value, row, index, column, and
    dataframe shape. It evaluates a bounded number of column-layout candidates
    using exact serialized character-Trie reuse:
    the sum of LCP lengths between lexicographically adjacent row strings.
    """

    def __init__(self, df: pd.DataFrame = None):
        """Store an optional dataframe for compatibility with Algorithm."""
        self.df = df

    @staticmethod
    def _is_missing(value) -> bool:
        """Return whether a scalar serializes as an evaluator empty string."""
        if value is None:
            return True
        try:
            missing = pd.isna(value)
            return bool(missing) if np.isscalar(missing) else False
        except Exception:
            return False

    @classmethod
    def _text_matrix(cls, df: pd.DataFrame) -> np.ndarray:
        """
        Build evaluator-compatible cell strings without modifying source data.

        Missing scalar values are represented by empty strings only in this
        planning matrix. The returned dataframe always keeps original values.
        """
        try:
            return df.fillna("").astype(str).to_numpy(dtype=object, copy=False)
        except Exception:
            raw = df.to_numpy(dtype=object, copy=False)
            result = np.empty(raw.shape, dtype=object)

            for row in range(raw.shape[0]):
                for col in range(raw.shape[1]):
                    value = raw[row, col]
                    result[row, col] = "" if cls._is_missing(value) else str(value)

            return result

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        """
        Compute longest common prefix with binary-search slice comparisons.

        Slice equality occurs in optimized string code and avoids Python-level
        character loops for long serialized rows.
        """
        limit = min(len(left), len(right))
        if limit == 0:
            return 0
        if left == right:
            return limit

        low, high = 0, limit
        while low < high:
            middle = (low + high + 1) // 2
            if left[:middle] == right[:middle]:
                low = middle
            else:
                high = middle - 1
        return low

    @classmethod
    def _trie_score(cls, strings: List[str]) -> int:
        """
        Compute exact ideal character-Trie reuse for a set of row strings.

        For a fixed multiset of strings, Trie reuse equals the sum of LCP
        lengths between adjacent lexicographically sorted strings.
        """
        if len(strings) < 2:
            return 0

        ordered = sorted(strings)
        score = 0
        previous = ordered[0]

        for current in ordered[1:]:
            score += cls._lcp(previous, current)
            previous = current

        return score

    @staticmethod
    def _serialize(text: np.ndarray, orders: List[List[int]]) -> List[str]:
        """Serialize each row from the text matrix under its column permutation."""
        return [
            "".join(text[row, col] for col in order)
            for row, order in enumerate(orders)
        ]

    @staticmethod
    def _resolve_column(requested, columns: List) -> int:
        """
        Resolve a requested constraint column to a unique positional column.

        Exact name matching is preferred. A unique substring match is retained
        for compatibility with callers that provide abbreviated names.
        """
        exact = [i for i, name in enumerate(columns) if name == requested]
        if len(exact) == 1:
            return exact[0]

        wanted = str(requested)
        partial = [
            i for i, name in enumerate(columns)
            if wanted in str(name)
        ]
        return partial[0] if len(partial) == 1 else None

    @classmethod
    def _apply_constraints(
        cls,
        order: List[int],
        columns: List,
        col_merge: List[List[str]],
        one_way_dep: List[Tuple[str, str]],
    ) -> List[int]:
        """
        Apply precedence and contiguous-group constraints by permutation only.

        No values are merged or rewritten. All positions remain present exactly
        once in the returned order.
        """
        result = list(order)

        if not result:
            return result

        for dependency in one_way_dep or []:
            if len(dependency) != 2:
                continue

            before = cls._resolve_column(dependency[0], columns)
            after = cls._resolve_column(dependency[1], columns)

            if before is None or after is None or before == after:
                continue

            before_pos = result.index(before)
            after_pos = result.index(after)

            if before_pos > after_pos:
                result.pop(before_pos)
                after_pos = result.index(after)
                result.insert(after_pos, before)

        for group in col_merge or []:
            members = []

            for requested in group:
                position = cls._resolve_column(requested, columns)
                if position is not None and position not in members:
                    members.append(position)

            if len(members) < 2:
                continue

            insertion = min(result.index(position) for position in members)
            block = [position for position in result if position in members]
            remaining = [position for position in result if position not in members]
            result = remaining[:insertion] + block + remaining[insertion:]

        return result

    @staticmethod
    def _column_metadata(text: np.ndarray):
        """
        Factorize serialized columns once and calculate repetition statistics.

        The global score for a field is:
        sum(len(value) * count(value) * (count(value) - 1)).
        """
        n_rows, n_cols = text.shape
        codes = []
        lengths = np.empty((n_rows, n_cols), dtype=np.int32)
        scores = np.zeros(n_cols, dtype=np.int64)

        for col in range(n_cols):
            values = text[:, col]
            code, _ = pd.factorize(values, sort=False)
            code = code.astype(np.int64, copy=False)
            codes.append(code)

            cell_lengths = np.fromiter(
                (len(value) for value in values),
                dtype=np.int32,
                count=n_rows,
            )
            lengths[:, col] = cell_lengths

            counts = np.bincount(code)
            if len(counts):
                masses = np.bincount(
                    code,
                    weights=cell_lengths,
                    minlength=len(counts),
                )
                scores[col] = int(np.sum(masses * (counts - 1)))

        return codes, lengths, scores

    @staticmethod
    def _frequency_order(scores: np.ndarray) -> List[int]:
        """Return global field order ranked by repeated length-weighted mass."""
        return sorted(
            range(len(scores)),
            key=lambda col: (-int(scores[col]), col),
        )

    def _conditional_orders(
        self,
        codes: List[np.ndarray],
        lengths: np.ndarray,
        tail_order: List[int],
        columns: List,
        col_merge: List[List[str]],
        one_way_dep: List[Tuple[str, str]],
        max_depth: int,
        max_groups: int,
        early_stop: int,
    ) -> List[List[int]]:
        """
        Construct bounded row-specific conditional prefix orders.

        Each active row group chooses the remaining field with highest local
        repeated-pair mass, partitions on its factorized values, and gives each
        child group its own suffix order. Work is capped by depth and group count.
        """
        n_rows, n_cols = lengths.shape

        if n_rows == 0:
            return []
        if n_cols == 0:
            return [[] for _ in range(n_rows)]

        candidate_columns = tail_order[:min(n_cols, 32)]
        orders = [None] * n_rows
        stack = [(np.arange(n_rows, dtype=np.int64), [], 0)]
        groups_processed = 0

        while stack and groups_processed < max_groups:
            rows, prefix, depth = stack.pop()
            groups_processed += 1

            used = set(prefix)
            available = [
                col for col in candidate_columns
                if col not in used
            ]

            if len(rows) <= 1 or depth >= max_depth or not available:
                final = prefix + [
                    col for col in tail_order
                    if col not in used
                ]
                final = self._apply_constraints(
                    final, columns, col_merge, one_way_dep
                )

                for row in rows:
                    orders[int(row)] = final
                continue

            best_col = None
            best_score = 0
            best_inverse = None

            for col in available:
                local_codes = codes[col][rows]

                _, inverse, counts = np.unique(
                    local_codes,
                    return_inverse=True,
                    return_counts=True,
                )

                if len(counts) == 0 or int(counts.max()) < 2:
                    continue

                masses = np.bincount(
                    inverse,
                    weights=lengths[rows, col],
                    minlength=len(counts),
                )

                score = int(np.sum(masses * (counts - 1)))

                if (
                    score > best_score
                    or (
                        score == best_score
                        and best_col is not None
                        and col < best_col
                    )
                ):
                    best_col = col
                    best_score = score
                    best_inverse = inverse

            if (
                best_col is None
                or best_score <= 0
                or (early_stop > 0 and best_score < early_stop)
            ):
                final = prefix + [
                    col for col in tail_order
                    if col not in used
                ]
                final = self._apply_constraints(
                    final, columns, col_merge, one_way_dep
                )

                for row in rows:
                    orders[int(row)] = final
                continue

            next_prefix = prefix + [best_col]

            for code in np.unique(best_inverse):
                child_rows = rows[best_inverse == code]
                stack.append((child_rows, next_prefix, depth + 1))

        while stack:
            rows, prefix, _ = stack.pop()
            used = set(prefix)

            final = prefix + [
                col for col in tail_order
                if col not in used
            ]
            final = self._apply_constraints(
                final, columns, col_merge, one_way_dep
            )

            for row in rows:
                orders[int(row)] = final

        return orders

    def _anchor_orders(
        self,
        text: np.ndarray,
        tail_order: List[int],
        columns: List,
        col_merge: List[List[str]],
        one_way_dep: List[Tuple[str, str]],
    ) -> List[List[int]]:
        """
        Construct orders that lead each row with its strongest repeated cell.

        This captures useful shared values that occur in different source
        columns across different rows, which field-only frequency ranking misses.
        """
        n_rows, n_cols = text.shape
        counts = {}

        for value in text.ravel():
            counts[value] = counts.get(value, 0) + 1

        value_scores = {
            value: len(value) * count * (count - 1)
            for value, count in counts.items()
            if value and count > 1
        }

        default_order = self._apply_constraints(
            tail_order,
            columns,
            col_merge,
            one_way_dep,
        )

        leading_orders = {}

        for col in range(n_cols):
            raw_order = [col] + [
                other for other in tail_order
                if other != col
            ]
            leading_orders[col] = self._apply_constraints(
                raw_order,
                columns,
                col_merge,
                one_way_dep,
            )

        orders = []

        for row in range(n_rows):
            best_col = None
            best_score = 0

            for col in range(n_cols):
                score = value_scores.get(text[row, col], 0)

                if (
                    score > best_score
                    or (
                        score == best_score
                        and score > 0
                        and (best_col is None or col < best_col)
                    )
                ):
                    best_score = score
                    best_col = col

            orders.append(
                default_order
                if best_col is None
                else leading_orders[best_col]
            )

        return orders

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
        """
        Return the best bounded prefix-cache layout and valid per-row metadata.

        Candidate selection uses actual serialized Trie reuse rather than
        adjacent input-row equality. The dataframe is only permuted; no source
        values, source rows, or source columns are removed or altered.
        """
        n_rows, n_cols = df.shape

        if n_rows == 0:
            return df.copy(), []

        if n_cols == 0:
            return df.copy(), [[] for _ in range(n_rows)]

        columns = list(df.columns)
        text = self._text_matrix(df)

        codes, lengths, pair_scores = self._column_metadata(text)

        global_order = self._apply_constraints(
            self._frequency_order(pair_scores),
            columns,
            col_merge,
            one_way_dep,
        )

        global_orders = [global_order] * n_rows
        global_strings = self._serialize(text, global_orders)

        candidates = [
            (
                self._trie_score(global_strings),
                global_orders,
                global_strings,
                False,
            )
        ]

        cell_count = n_rows * n_cols

        if (
            cell_count <= 2_000_000
            and n_rows > 1
            and n_cols > 1
        ):
            depth = min(n_cols, 8)

            if col_stop is not None and col_stop > 0:
                depth = min(depth, int(col_stop))

            group_limit = 160

            if row_stop is not None and row_stop > 0:
                group_limit = min(group_limit, max(8, int(row_stop)))

            conditional_orders = self._conditional_orders(
                codes=codes,
                lengths=lengths,
                tail_order=global_order,
                columns=columns,
                col_merge=col_merge,
                one_way_dep=one_way_dep,
                max_depth=max(1, depth),
                max_groups=group_limit,
                early_stop=early_stop,
            )

            conditional_strings = self._serialize(
                text,
                conditional_orders,
            )

            candidates.append((
                self._trie_score(conditional_strings),
                conditional_orders,
                conditional_strings,
                True,
            ))

            anchor_orders = self._anchor_orders(
                text=text,
                tail_order=global_order,
                columns=columns,
                col_merge=col_merge,
                one_way_dep=one_way_dep,
            )

            anchor_strings = self._serialize(text, anchor_orders)

            candidates.append((
                self._trie_score(anchor_strings),
                anchor_orders,
                anchor_strings,
                True,
            ))

        best_score, best_orders, best_strings, row_specific = max(
            candidates,
            key=lambda item: (item[0], not item[3]),
        )

        row_order = sorted(
            range(n_rows),
            key=lambda row: (best_strings[row], row),
        )

        output_orders = [best_orders[row] for row in row_order]

        if not row_specific:
            result = df.iloc[row_order, best_orders[0]].copy()
        else:
            source = df.to_numpy(dtype=object, copy=False)
            output = np.empty((n_rows, n_cols), dtype=object)

            for output_row, source_row in enumerate(row_order):
                output[output_row, :] = source[
                    source_row,
                    output_orders[output_row],
                ]

            result = pd.DataFrame(
                output,
                index=df.index.take(row_order),
                columns=columns,
            )

        column_orderings = [
            [columns[col] for col in order]
            for order in output_orders
        ]

        return result, column_orderings