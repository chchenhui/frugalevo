"""Budget-aware controller for the native OpenEvolve baseline.

Without ``database.budget_usd`` this delegates to the ordinary discovery
controller and is behaviourally identical to the previous implementation.
With a budget it records the same fixed-window BA-AUC curve as
FrugalEvo and stops when the window is spent, or when the historical
mean price of the next iteration no longer fits in the remaining window.
"""

from __future__ import annotations

import logging
import json
import os
from typing import Callable, List, Optional, Union

from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.utils.budget_curve import BudgetCurve
from skydiscover.search.utils.discovery_utils import SerializableResult
from skydiscover.utils.metrics import get_score

logger = logging.getLogger(__name__)

_CHECKPOINT_STATE_FILE = "openevolve_native_controller_state.json"


class OpenEvolveNativeController(DiscoveryController):
    """Run OpenEvolve under an optional dollar-denominated BA-AUC window."""

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)
        database_config = self.config.search.database
        self.budget = getattr(database_config, "budget_usd", None)
        self.curve = BudgetCurve(
            budget=self.budget,
            unit=str(getattr(database_config, "budget_unit", "usd")),
        )
        self._iteration_costs: List[float] = []

        if self.checkpoint_path:
            self._load_checkpoint_state(self.checkpoint_path)

    def save_checkpoint_state(self, checkpoint_path: str) -> None:
        """Persist budget accounting so a resumed run keeps the same window."""
        os.makedirs(checkpoint_path, exist_ok=True)
        state = {
            "curve": self.curve.to_dict(),
            "iteration_costs": self._iteration_costs,
        }
        with open(os.path.join(checkpoint_path, _CHECKPOINT_STATE_FILE), "w") as output:
            json.dump(state, output, indent=2)

    def _load_checkpoint_state(self, checkpoint_path: str) -> None:
        """Restore controller state when present; old checkpoints remain valid."""
        path = os.path.join(checkpoint_path, _CHECKPOINT_STATE_FILE)
        if not os.path.exists(path):
            logger.warning(
                "No %s found in checkpoint; budget accounting starts at zero",
                _CHECKPOINT_STATE_FILE,
            )
            return
        with open(path, "r") as source:
            state = json.load(source)
        self.curve.restore(state.get("curve") or {})
        self._iteration_costs = [
            max(0.0, float(value)) for value in state.get("iteration_costs", [])
        ]
        logger.info(
            "Restored OpenEvolve budget state: %s", self.curve.summary_line()
        )

    def _seed_curve(self) -> None:
        if self.curve.points:
            return
        best = self.database.get_best_program() if self.database.programs else None
        if best is not None:
            self.curve.observe(get_score(best.metrics), iteration=0, candidates=0)

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
        post_process_result: Optional[bool] = True,
        retry_times: Optional[int] = 3,
    ) -> Optional[Union[Program, SerializableResult]]:
        if self.budget is None:
            return await super().run_discovery(
                start_iteration,
                max_iterations,
                checkpoint_callback,
                post_process_result,
                retry_times,
            )

        if self.config.max_parallel_iterations != 1:
            raise ValueError(
                "Budgeted OpenEvolve requires max_parallel_iterations=1 so cost and "
                "best-so-far observations have a deterministic order"
            )

        result: Optional[SerializableResult] = None
        self._seed_curve()
        try:
            for iteration in range(start_iteration, start_iteration + max_iterations):
                if self.shutdown_event.is_set():
                    logger.info("Shutdown requested, stopping budgeted OpenEvolve")
                    break

                remaining = self.curve.remaining()
                if remaining <= 0:
                    self.early_stopping_triggered = True
                    logger.info(
                        "Stopping OpenEvolve at iteration %d: budget is spent (%s)",
                        iteration,
                        self.curve.summary_line(),
                    )
                    break

                if self._iteration_costs:
                    forecast = sum(self._iteration_costs) / len(self._iteration_costs)
                    if forecast > remaining:
                        self.early_stopping_triggered = True
                        logger.info(
                            "Stopping OpenEvolve at iteration %d: forecast %.6f %s "
                            "exceeds remaining %.6f (%s)",
                            iteration,
                            forecast,
                            self.curve.unit,
                            remaining,
                            self.curve.summary_line(),
                        )
                        break

                cost_before = self.curve.spent()
                try:
                    result = await self._run_iteration(iteration, retry_times=retry_times)
                    if result.error:
                        logger.warning("Iteration %d failed: %s", iteration, result.error)
                    elif post_process_result:
                        self._process_iteration_result(
                            result,
                            iteration,
                            checkpoint_callback,
                        )
                except Exception as exc:
                    logger.exception("Error in OpenEvolve iteration %d: %s", iteration, exc)

                best = self.database.get_best_program() if self.database.programs else None
                point = self.curve.observe(
                    get_score(best.metrics) if best is not None else None,
                    iteration=iteration,
                    candidates=1,
                )
                iteration_cost = max(0.0, point.cost - cost_before)
                if iteration_cost > 0:
                    self._iteration_costs.append(iteration_cost)
                logger.info(
                    "OpenEvolve iteration %d: delta_cost=%.6f, %s",
                    iteration,
                    iteration_cost,
                    self.curve.summary_line(),
                )
        finally:
            self.curve.write(self.output_dir)

        if not post_process_result:
            return result
        logger.info("Budgeted OpenEvolve completed: %s", self.curve.summary_line())
        return self.database.get_best_program()
