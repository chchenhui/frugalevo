"""Shared optional cost guard for sequential discovery controllers.

The guard is deliberately small: it observes the process-wide LLM cost
tracker through :class:`BudgetCurve`, records the incumbent once per search
step, and avoids starting a step whose historical mean cost is larger than
the remaining budget.  Controllers that do not configure ``budget_usd`` keep
their original iteration-bounded behaviour.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from skydiscover.search.utils.budget_curve import BudgetCurve
from skydiscover.utils.metrics import get_score


class IterationBudgetGuard:
    """Apply an optional predictive cost cap to a sequential controller."""

    def __init__(
        self,
        database_config: Any,
        output_dir: Optional[str],
        *,
        label: str,
        logger: logging.Logger,
    ) -> None:
        self.budget = getattr(database_config, "budget_usd", None)
        self.output_dir = output_dir
        self.label = label
        self.logger = logger
        self.curve = BudgetCurve(
            budget=self.budget,
            unit=str(getattr(database_config, "budget_unit", "usd")),
        )
        self._step_costs: List[float] = []

    @property
    def enabled(self) -> bool:
        return self.budget is not None

    def seed(self, database: Any) -> None:
        """Record the initial incumbent at zero model spend."""
        if not self.enabled or self.curve.points:
            return
        best = database.get_best_program() if database.programs else None
        if best is not None:
            self.curve.observe(get_score(best.metrics), iteration=0, candidates=0)

    def should_stop(self, controller: Any, iteration: int) -> bool:
        """Return whether the next step should be withheld."""
        if not self.enabled:
            return False

        remaining = self.curve.remaining()
        reason: Optional[str] = None
        if remaining <= 0:
            reason = "budget is spent"
        elif self._step_costs:
            forecast = sum(self._step_costs) / len(self._step_costs)
            if forecast > remaining:
                reason = (
                    f"forecast {forecast:.6f} {self.curve.unit} exceeds "
                    f"remaining {remaining:.6f}"
                )

        if reason is None:
            return False

        controller.early_stopping_triggered = True
        self.logger.info(
            "Stopping %s at iteration %d: %s (%s)",
            self.label,
            iteration,
            reason,
            self.curve.summary_line(),
        )
        return True

    def spent(self) -> float:
        """Return current spend for measuring one step's delta."""
        return self.curve.spent()

    def observe(
        self,
        database: Any,
        *,
        iteration: int,
        cost_before: float,
        candidates: int = 1,
        include_in_forecast: bool = True,
    ) -> float:
        """Record incumbent and cost after a step; return its cost delta."""
        if not self.enabled:
            return 0.0

        best = database.get_best_program() if database.programs else None
        point = self.curve.observe(
            get_score(best.metrics) if best is not None else None,
            iteration=iteration,
            candidates=candidates,
        )
        step_cost = max(0.0, point.cost - cost_before)
        if include_in_forecast and step_cost > 0:
            self._step_costs.append(step_cost)
        self.logger.info(
            "%s iteration %d: delta_cost=%.6f, %s",
            self.label,
            iteration,
            step_cost,
            self.curve.summary_line(),
        )
        return step_cost

    def write(self) -> None:
        """Persist the curve and BA-AUC summary when budgeting is enabled."""
        if self.enabled:
            self.curve.write(self.output_dir)
