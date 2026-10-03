"""Budget-Aware AUC over a fixed cost window.

A search run traces a curve on the (cost, performance) plane: after spending
``c``, the best solution found so far scores ``s(c)``.  ``s`` is a
non-decreasing step function -- the score you would report if the run were
stopped right there. Integrating it over ``[0, B]`` gives **BA-AUC(B)**: the
area between the best-so-far curve and the cost axis. Finding the same score
earlier produces a larger area.

The module has two halves:

``budget_aware_auc`` / ``normalized_budget_aware_auc``
    Pure functions over ``(cost, score)`` samples.  Use them to score runs
    that already finished -- the samples can come from a ``budget_curve.jsonl``
    written by a live run, or be reconstructed from any other log.

``BudgetCurve``
    Live recorder wired to the global :class:`~skydiscover.llm.pricing.CostTracker`.
    A controller calls :meth:`BudgetCurve.observe` once per iteration; the
    curve reads how much has been spent, tracks the incumbent, and can say
    whether the budget is exhausted.
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from skydiscover.llm.pricing import get_cost_tracker

logger = logging.getLogger(__name__)

#: Budget measured in US dollars, priced from ``models.yaml``.
USD = "usd"
#: Budget measured in total tokens -- the fallback when a model has no price.
TOKENS = "tokens"

CurveSample = Tuple[float, float]


@dataclass(frozen=True)
class CurvePoint:
    """One observation: after ``cost`` had been spent, a candidate scored ``score``.

    ``incumbent`` is the best-so-far score at that moment, i.e. the height of
    the step function.  It is redundant (recoverable from the sample
    sequence) but worth persisting so a plot can be drawn without replaying
    the running maximum.
    """

    cost: float
    score: float
    incumbent: float
    iteration: Optional[int] = None
    candidates: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cost": self.cost,
            "score": self.score,
            "incumbent": self.incumbent,
            "iteration": self.iteration,
            "candidates": self.candidates,
        }


def budget_aware_auc(
    samples: Iterable[CurveSample],
    budget: float,
    *,
    floor: float = 0.0,
) -> float:
    """Area under the best-so-far curve over the cost window ``[0, budget]``.

    Given samples ``(cost_i, score_i)`` -- cost being *cumulative* spend at
    the moment ``score_i`` became known -- the incumbent curve is

        s(c) = max({score_i : cost_i <= c} | {floor})

    and this returns ``∫₀^budget s(c) dc``.

    A run that stops before ``budget`` keeps its last incumbent to the right
    edge. A run that spends past ``budget`` is truncated, so improvements
    outside the window do not count.

    Args:
        samples: ``(cumulative_cost, score)`` pairs, in any order.
        budget: right edge ``B`` of the window, in the same unit as the costs.
        floor: incumbent before anything has been scored.  Scores below it are
            clamped, so ``BA-AUC`` stays well defined for evaluators that
            emit negative scores.

    Returns:
        The area, in ``score x cost`` units. Divide by ``budget`` to obtain
        the mean best-so-far score over the window.
    """
    if budget <= 0:
        raise ValueError(f"budget must be > 0, got {budget}")

    ordered = sorted((max(0.0, float(cost)), float(score)) for cost, score in samples)

    area = 0.0
    incumbent = float(floor)
    edge = 0.0
    for cost, score in ordered:
        if cost >= budget:
            break
        if cost > edge:
            area += incumbent * (cost - edge)
            edge = cost
        incumbent = max(incumbent, score)
    return area + incumbent * (budget - edge)


def normalized_budget_aware_auc(
    samples: Iterable[CurveSample],
    budget: float,
    *,
    floor: float = 0.0,
) -> float:
    """Mean best-so-far score over the budget window: ``BA-AUC(B) / B``.

    Same units as the score itself, so for the usual ``combined_score in
    [0, 1]`` convention this lands in ``[0, 1]``: 1.0 would mean a perfect
    solution was already in hand at zero cost.
    """
    return budget_aware_auc(samples, budget, floor=floor) / budget


class BudgetCurve:
    """Live (cost, incumbent) recorder that prices BA-AUC as a run proceeds.

    Spend is read from the process-global cost tracker, snapshotted at
    construction so that several runs in one process do not inherit each
    other's totals.  When the model in use has no price in ``models.yaml``
    the recorder falls back to counting tokens, because a budget expressed
    in dollars that silently reads as zero would make an expensive run look
    free.
    """

    def __init__(
        self,
        budget: Optional[float] = None,
        unit: str = USD,
        *,
        floor: float = 0.0,
        cost_fn: Optional[Callable[[], float]] = None,
    ):
        if budget is not None and budget <= 0:
            raise ValueError(f"budget must be > 0 when set, got {budget}")
        if unit not in (USD, TOKENS):
            raise ValueError(f"unit must be '{USD}' or '{TOKENS}', got {unit!r}")

        self.budget = budget
        self.unit = unit
        self.floor = float(floor)
        self._cost_fn = cost_fn
        self._cost_offset = 0.0
        # Mark the tracker unconditionally: cheap, and it keeps the baseline
        # non-optional for the ``since()`` call in spent().
        self._baseline = get_cost_tracker().snapshot()
        self._lock = threading.Lock()
        self._points: List[CurvePoint] = []
        self._incumbent = float(floor)
        self._warned_unpriced = False

    # -- cost axis ------------------------------------------------------

    def spent(self) -> float:
        """Cumulative cost since this curve was created, in ``self.unit``."""
        if self._cost_fn is not None:
            return self._cost_offset + max(0.0, float(self._cost_fn()))

        tracker = get_cost_tracker().since(self._baseline)
        if self.unit == TOKENS:
            return self._cost_offset + float(tracker.total_usage.total_tokens)

        cost = tracker.total_cost_usd
        if cost is None:
            # Unpriced model: keep measuring, but say so once and switch the
            # axis to tokens rather than reporting a run as free.
            if tracker.total_usage.calls and not self._warned_unpriced:
                self._warned_unpriced = True
                logger.warning(
                    "No USD price for %s -- budget curve switches to the token axis. "
                    "Add the model to models.yaml to get a dollar-denominated BA-AUC.",
                    ", ".join(tracker.unpriced_models) or "the model in use",
                )
            if self._warned_unpriced:
                self.unit = TOKENS
                return self._cost_offset + float(tracker.total_usage.total_tokens)
            return self._cost_offset
        return self._cost_offset + float(cost)

    def remaining(self) -> float:
        """Budget left, or ``inf`` when no budget was set."""
        if self.budget is None:
            return float("inf")
        return max(0.0, self.budget - self.spent())

    def exhausted(self) -> bool:
        """True once spend has reached the budget window's right edge."""
        return self.budget is not None and self.spent() >= self.budget

    # -- recording ------------------------------------------------------

    def observe(
        self,
        score: Optional[float],
        *,
        iteration: Optional[int] = None,
        candidates: Optional[int] = None,
    ) -> CurvePoint:
        """Record ``score`` against the spend so far and return the new point.

        ``score`` is the best score among whatever this step produced; pass
        ``None`` (or a low score) for a step that produced nothing usable --
        the step still cost money, and recording it is what makes wasted
        spend visible on the curve.
        """
        value = self.floor if score is None else max(self.floor, float(score))
        with self._lock:
            self._incumbent = max(self._incumbent, value)
            point = CurvePoint(
                cost=self.spent(),
                score=value,
                incumbent=self._incumbent,
                iteration=iteration,
                candidates=candidates,
            )
            self._points.append(point)
            return point

    @property
    def points(self) -> List[CurvePoint]:
        with self._lock:
            return list(self._points)

    @property
    def incumbent(self) -> float:
        with self._lock:
            return self._incumbent

    def restore(self, state: Dict[str, Any]) -> None:
        """Restore persisted points and continue the cost axis from their spend.

        The live tracker is snapshotted when this object is constructed, so
        ``spent`` from the previous process becomes an offset and only new
        calls are added to it. The current config keeps control of ``budget``;
        a checkpoint cannot silently change the new run's evaluation window.
        """
        records = state.get("points") or []
        points: List[CurvePoint] = []
        incumbent = self.floor
        for record in records:
            score = max(self.floor, float(record.get("score", self.floor)))
            incumbent = max(incumbent, float(record.get("incumbent", score)), score)
            points.append(
                CurvePoint(
                    cost=max(0.0, float(record.get("cost", 0.0))),
                    score=score,
                    incumbent=incumbent,
                    iteration=record.get("iteration"),
                    candidates=record.get("candidates"),
                )
            )

        saved_unit = state.get("unit")
        if saved_unit in (USD, TOKENS):
            self.unit = saved_unit
        last_cost = max((point.cost for point in points), default=0.0)
        offset = max(last_cost, float(state.get("spent", last_cost)))
        with self._lock:
            self._points = points
            self._incumbent = incumbent
            self._cost_offset = max(0.0, offset)

    # -- scoring --------------------------------------------------------

    def auc(self, budget: Optional[float] = None) -> Optional[float]:
        """BA-AUC over ``[0, budget]``; ``None`` if no budget is known."""
        window = budget if budget is not None else self.budget
        if window is None:
            return None
        samples = [(p.cost, p.score) for p in self.points]
        return budget_aware_auc(samples, window, floor=self.floor)

    def normalized_auc(self, budget: Optional[float] = None) -> Optional[float]:
        """BA-AUC divided by the window width -- the headline number."""
        window = budget if budget is not None else self.budget
        if window is None:
            return None
        area = self.auc(window)
        return None if area is None else area / window

    def to_dict(self, budget: Optional[float] = None) -> Dict[str, Any]:
        window = budget if budget is not None else self.budget
        return {
            "unit": self.unit,
            "budget": window,
            "spent": self.spent(),
            "final_incumbent": self.incumbent,
            "ba_auc": self.auc(window),
            "ba_auc_normalized": self.normalized_auc(window),
            "num_points": len(self._points),
            "points": [p.to_dict() for p in self.points],
        }

    def summary_line(self) -> str:
        """One-line progress string for the run log."""
        spent = self.spent()
        unit = "$" if self.unit == USD else ""
        suffix = "" if self.unit == USD else " tokens"
        parts = [f"spent={unit}{spent:,.4f}{suffix}", f"incumbent={self.incumbent:.6f}"]
        # The raw area, not the area divided by the window. Both are kept in
        # ``to_dict`` for downstream analysis; the log reports the integral so
        # a line can be read without also knowing the window width.
        area = self.auc()
        if area is not None:
            parts.append(f"BA-AUC@{unit}{self.budget:,.4f}{suffix}={area:.6f}")
        return ", ".join(parts)

    # -- persistence ----------------------------------------------------

    def write(self, output_dir: Optional[str]) -> None:
        """Write ``ba_auc.json`` and ``budget_curve.jsonl`` into *output_dir*.

        Accounting must never take a run down, so failures here are logged
        and swallowed.
        """
        if not output_dir:
            return
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "ba_auc.json"), "w") as handle:
                json.dump(self.to_dict(), handle, indent=2)
            with open(os.path.join(output_dir, "budget_curve.jsonl"), "w") as handle:
                for point in self.points:
                    handle.write(json.dumps(point.to_dict()) + "\n")
        except Exception:
            logger.debug("Failed to write budget curve files", exc_info=True)


def load_curve_samples(path: str) -> List[CurveSample]:
    """Read ``(cost, score)`` samples back from a ``budget_curve.jsonl``.

    Lets a finished run be re-scored at a different budget without re-running
    it: ``normalized_budget_aware_auc(load_curve_samples(p), B)``.
    """
    samples: List[CurveSample] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append((float(record["cost"]), float(record["score"])))
    return samples


def auc_table(
    runs: Dict[str, Sequence[CurveSample]],
    budgets: Sequence[float],
    *,
    floor: float = 0.0,
) -> Dict[str, Dict[float, float]]:
    """Normalized BA-AUC for several runs at one or more fixed budgets."""
    return {
        name: {
            budget: normalized_budget_aware_auc(samples, budget, floor=floor) for budget in budgets
        }
        for name, samples in runs.items()
    }


def welch_mean_difference_ci(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    confidence: float = 0.95,
) -> Tuple[float, float, float, float]:
    """Independent-sample Welch interval for ``mean(candidate) - mean(baseline)``.

    This is an offline repeated-run diagnostic, not part of the BA-AUC
    definition. It deliberately does not assume equal variances: search
    policies often have a mixture of ordinary and breakthrough runs while a
    baseline can converge to the same ceiling on every seed.

    Returns ``(difference, lower, upper, degrees_of_freedom)``. At least two
    observations are required in each group. The usual Student-t interval is
    only an approximation for small or strongly non-normal samples, so callers
    should always report sample counts and raw variability beside it.
    """
    if len(candidate) < 2 or len(baseline) < 2:
        raise ValueError("Welch confidence interval requires at least two values per group")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")

    candidate_values = [float(value) for value in candidate]
    baseline_values = [float(value) for value in baseline]
    difference = statistics.mean(candidate_values) - statistics.mean(baseline_values)
    candidate_term = statistics.variance(candidate_values) / len(candidate_values)
    baseline_term = statistics.variance(baseline_values) / len(baseline_values)
    standard_error = math.sqrt(candidate_term + baseline_term)
    if standard_error == 0.0:
        return difference, difference, difference, math.inf

    denominator = candidate_term**2 / (len(candidate_values) - 1) + baseline_term**2 / (
        len(baseline_values) - 1
    )
    degrees_of_freedom = (candidate_term + baseline_term) ** 2 / denominator

    # scipy is already a project dependency, but importing its stats package
    # lazily keeps the live search controller's startup path lightweight.
    from scipy.stats import t as student_t

    critical = float(student_t.ppf((1.0 + confidence) / 2.0, degrees_of_freedom))
    margin = critical * standard_error
    return difference, difference - margin, difference + margin, degrees_of_freedom


def mean_incumbent_curve(
    runs: Sequence[Sequence[CurveSample]],
    *,
    floor: float = 0.0,
) -> List[CurveSample]:
    """Pointwise mean best-so-far curve for repeated runs.

    The union of all improvement costs is sufficient: between adjacent costs
    every member curve is constant. A finished member keeps its final
    incumbent, matching BA-AUC's right-extension rule. By linearity, the area
    under this returned curve equals the mean BA-AUC of the member runs.
    """
    if not runs:
        raise ValueError("cannot average an empty run group")

    member_steps = [incumbent_steps(samples, floor=floor) for samples in runs]
    costs = sorted({0.0, *(cost for steps in member_steps for cost, _ in steps)})
    positions = [0] * len(member_steps)
    incumbents = [float(floor)] * len(member_steps)
    mean_curve: List[CurveSample] = []

    for cost in costs:
        for index, steps in enumerate(member_steps):
            while positions[index] < len(steps) and steps[positions[index]][0] <= cost:
                incumbents[index] = max(incumbents[index], steps[positions[index]][1])
                positions[index] += 1
        mean_curve.append((cost, sum(incumbents) / len(incumbents)))
    return mean_curve


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

# Validated categorical slots (see the data-viz reference palette). Line charts
# use the adjacent pairlist, on which this order clears every CVD gate in both
# modes. Light-mode aqua sits below 3:1 on the surface, so direct labels are
# mandatory relief -- which is why this plot always draws them.
_THEMES: Dict[str, Dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
    },
}


def incumbent_steps(samples: Iterable[CurveSample], *, floor: float = 0.0) -> List[CurveSample]:
    """The best-so-far step function as ``(cost, incumbent)`` breakpoints.

    Only the points where the incumbent actually rises are kept -- they fully
    determine the curve, and they are what a step plot needs.
    """
    steps: List[CurveSample] = []
    incumbent = float(floor)
    for cost, score in sorted((max(0.0, float(c)), float(s)) for c, s in samples):
        value = max(incumbent, score)
        if not steps:
            steps.append((cost, value))
        elif value > incumbent:
            steps.append((cost, value))
        incumbent = value
    return steps


def _place_labels(
    entries: List[Tuple[float, str, str]],
    min_gap: float,
    limits: Tuple[float, float],
) -> List[Tuple[float, float, str, str]]:
    """Nudge colliding end-labels apart, returning ``(y_true, y_label, text, color)``.

    Converging curves put their end-labels on top of each other. Pushing them
    apart and drawing a leader line keeps each label attached to its own curve;
    stacking them silently would not. The whole group is then slid back inside
    ``limits`` -- a label pushed off the top of the axes is worse than a
    collision, because it simply disappears.
    """
    low, high = limits
    ordered = sorted(entries, key=lambda item: item[0])
    labels: List[float] = []
    previous: Optional[float] = None
    for y_true, _, _ in ordered:
        y_label = y_true if previous is None else max(y_true, previous + min_gap)
        labels.append(y_label)
        previous = y_label

    if labels:
        overflow = labels[-1] - high
        if overflow > 0:
            labels = [value - overflow for value in labels]
        underflow = low - labels[0]
        if underflow > 0:
            labels = [value + underflow for value in labels]

    return [(entry[0], label, entry[1], entry[2]) for entry, label in zip(ordered, labels)]


def plot_budget_curves(
    runs: Dict[str, Sequence[CurveSample]],
    path: str,
    *,
    budget: Optional[float] = None,
    theme: str = "light",
    unit: str = USD,
    floor: float = 0.0,
    title: str = "Performance per unit of spend",
    subtitle: Optional[str] = None,
    fill: Optional[bool] = None,
) -> str:
    """Draw the incumbent curves of several runs on one cost/performance plot.

    One step line per run: ``x`` is cumulative spend, ``y`` is the best score
    found by then. The y-axis always starts at zero, so heights stay
    comparable and the first jump off the seed program reads at true scale.

    A run that stopped before the right edge keeps its last incumbent, drawn
    faded past the end marker: the score is still held, but nothing more was
    measured.

    The area under a curve *is* its BA-AUC, so a single run is shaded by
    default. Several overlapping washes stack into an unreadable block whose
    area belongs to no one run, so multi-run plots are drawn unshaded unless
    ``fill=True`` is forced.

    Args:
        runs: ``{run name: (cost, score) samples}``. Samples need not be
            sorted or already reduced to best-so-far.
        path: output image path; the format follows the extension.
        budget: draw a marker at ``B`` and clip the x-axis there. Defaults to
            the largest observed cost across runs.
        theme: ``"light"`` or ``"dark"`` -- each has its own validated steps.
        unit: ``USD`` or ``TOKENS``, for axis labels.
        floor: incumbent before anything is scored.
        fill: shade the area under each curve (the BA-AUC itself). Defaults
            to shading only when a single run is plotted.

    Returns:
        ``path``, for convenience.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not runs:
        raise ValueError("nothing to plot: runs is empty")
    if theme not in _THEMES:
        raise ValueError(f"theme must be one of {sorted(_THEMES)}, got {theme!r}")
    palette = _THEMES[theme]

    curves = {name: incumbent_steps(samples, floor=floor) for name, samples in runs.items()}
    spent = {
        name: max((cost for cost, _ in samples), default=0.0) for name, samples in runs.items()
    }
    right = budget if budget is not None else max(spent.values())
    if right <= 0:
        raise ValueError("cannot plot: no run spent anything and no budget was given")
    shade = len(runs) == 1 if fill is None else fill

    figure, axes = plt.subplots(figsize=(9.6, 5.4), dpi=200)
    figure.patch.set_facecolor(palette["surface"])
    axes.set_facecolor(palette["surface"])

    label_entries: List[Tuple[float, str, str]] = []
    for index, (name, steps) in enumerate(curves.items()):
        color = palette["series"][index % len(palette["series"])]
        end = min(spent[name], right)

        # Breakpoints inside the window, held to the run's own stopping point.
        xs = [0.0]
        ys = [floor]
        for cost, incumbent in steps:
            if cost > end:
                break
            xs.append(cost)
            ys.append(incumbent)
        xs.append(end)
        ys.append(ys[-1])

        axes.step(
            xs,
            ys,
            where="post",
            linewidth=2,
            color=color,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
            label=name,
        )
        if shade:
            axes.fill_between(xs, floor, ys, step="post", color=color, alpha=0.10, zorder=1)

        # The flat tail past the end marker is held, not measured.
        if end < right:
            axes.plot([end, right], [ys[-1], ys[-1]], color=color, lw=2, alpha=0.35, zorder=2)
            if shade:
                axes.fill_between(
                    [end, right], floor, [ys[-1], ys[-1]], color=color, alpha=0.04, zorder=1
                )

        # End marker with a 2px surface ring so overlaps stay legible.
        axes.plot(
            [end],
            [ys[-1]],
            marker="o",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor=palette["surface"],
            markeredgewidth=2,
            zorder=4,
        )
        label_entries.append((ys[-1], f"{ys[-1]:.4f}", color))

    axes.set_xlim(0, right * 1.16)
    axes.set_ylim(0, 1.0)

    # Direct labels: mandatory relief for the light-mode aqua slot, de-collided
    # with leader lines rather than stacked.
    for y_true, y_label, text, color in _place_labels(
        label_entries, min_gap=0.05, limits=(0.02, 0.98)
    ):
        if abs(y_label - y_true) > 1e-9:
            axes.plot(
                [right, right * 1.035],
                [y_true, y_label],
                color=color,
                lw=1,
                alpha=0.6,
                zorder=2,
            )
        axes.annotate(
            text,
            xy=(right * 1.045, y_label),
            va="center",
            ha="left",
            fontsize=9,
            color=palette["ink_secondary"],
        )

    if budget is not None:
        axes.axvline(budget, color=palette["axis"], lw=1, zorder=2)
        # Bottom of the rule: the top-right corner belongs to the end labels.
        axes.annotate(
            f"B = {_format_cost(budget, unit)}",
            xy=(budget, 0.0),
            xytext=(-5, 6),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            color=palette["muted"],
        )

    axes.grid(axis="y", color=palette["grid"], linewidth=1, linestyle="-", zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(palette["axis"])
        axes.spines[side].set_linewidth(1)
    axes.tick_params(colors=palette["muted"], labelsize=9, length=0)

    axis_unit = "cumulative cost (USD)" if unit == USD else "cumulative cost (tokens)"
    axes.set_xlabel(axis_unit, fontsize=10, color=palette["ink_secondary"], labelpad=8)
    axes.set_ylabel(
        "combined_score (best so far)", fontsize=10, color=palette["ink_secondary"], labelpad=8
    )
    # Title and subtitle are stacked by hand: set_title's pad reserves room for
    # one line only, so an annotated subtitle lands on top of it.
    axes.set_title(title, fontsize=13, color=palette["ink"], loc="left", pad=26 if subtitle else 10)
    if subtitle:
        axes.annotate(
            subtitle,
            xy=(0, 1.0),
            xytext=(0, 8),
            xycoords="axes fraction",
            textcoords="offset points",
            fontsize=9.5,
            color=palette["muted"],
        )

    # Run names are long, so the legend goes below the axes rather than eating
    # the plot area. It stays present regardless: identity is never colour-alone.
    legend = axes.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.13),
        frameon=False,
        fontsize=9.5,
        handlelength=1.8,
        borderaxespad=0,
    )
    for legend_text in legend.get_texts():
        legend_text.set_color(palette["ink_secondary"])

    figure.tight_layout()
    figure.savefig(path, facecolor=palette["surface"], bbox_inches="tight")
    plt.close(figure)
    logger.info("Wrote budget curve plot to %s", path)
    return path


def _format_cost(value: float, unit: str) -> str:
    return f"${value:,.4f}" if unit == USD else f"{value:,.0f} tokens"
