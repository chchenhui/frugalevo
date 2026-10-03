# EVOLVE-BLOCK-START
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive elite-and-plateau search strategy."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))

        # All adaptive state is updated in add(), so it is reconstructed when
        # programs are restored from a checkpoint.
        self.seen_program_ids: Dict[str, bool] = {}
        self.parent_uses: Dict[str, int] = {}
        self.parent_child_count: Dict[str, int] = {}
        self.parent_positive_count: Dict[str, int] = {}
        self.best_numeric_score: Optional[float] = None
        self.last_meaningful_improvement_iteration: int = 0
        self.latest_iteration: int = 0
        self.stagnation_steps: int = 0
        self.label_uses: Dict[str, int] = {"diverge": 0, "refine": 0}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        """Add a program and record outcome signals for later parent selection."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        is_new = program.id not in self.seen_program_ids
        self.programs[program.id] = program

        actual_iteration = iteration
        if actual_iteration is None:
            actual_iteration = program.iteration_found
        if isinstance(actual_iteration, int):
            self.latest_iteration = max(self.latest_iteration, actual_iteration)
            self.last_iteration = max(self.last_iteration, actual_iteration)

        if is_new:
            self.seen_program_ids[program.id] = True
            score = self._score(program)

            if program.parent_id:
                self.parent_uses[program.parent_id] = (
                    self.parent_uses.get(program.parent_id, 0) + 1
                )
                self.parent_child_count[program.parent_id] = (
                    self.parent_child_count.get(program.parent_id, 0) + 1
                )

                parent = self.get(program.parent_id)
                parent_score = self._score(parent) if parent is not None else None
                if (
                    score is not None
                    and parent_score is not None
                    and score > parent_score + 0.0005
                ):
                    self.parent_positive_count[program.parent_id] = (
                        self.parent_positive_count.get(program.parent_id, 0) + 1
                    )

            label = program.parent_info[0] if program.parent_info else ""
            if label == self.DIVERGE_LABEL:
                self.label_uses["diverge"] += 1
            elif label == self.REFINE_LABEL:
                self.label_uses["refine"] += 1

            if score is not None:
                if self.best_numeric_score is None:
                    self.best_numeric_score = score
                else:
                    threshold = max(0.01, abs(self.best_numeric_score) * 0.01)
                    if score > self.best_numeric_score + threshold:
                        self.best_numeric_score = score
                        self.last_meaningful_improvement_iteration = self.latest_iteration
                        self.stagnation_steps = 0
                    else:
                        self.best_numeric_score = max(self.best_numeric_score, score)
                        self.stagnation_steps = max(
                            0,
                            self.latest_iteration
                            - self.last_meaningful_improvement_iteration,
                        )

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug("Added program %s to the evolve database", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(p, self._score(p)) for p in candidates]
        numeric = [(p, s) for p, s in scored if s is not None]

        if not numeric:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        numeric.sort(key=lambda item: item[1], reverse=True)
        best_score = numeric[0][1]

        # The observed successful mutations came from the dense 0.1276 tier,
        # rather than repeatedly mutating the current best.  Keep selection
        # near the frontier, but reward underused and previously productive
        # parents to obtain new combinations.
        frontier = [
            p for p, score in numeric
            if score >= best_score - 0.0035
        ]
        if not frontier:
            frontier = [p for p, _ in numeric]

        weights: List[float] = []
        for p in frontier:
            score = self._score(p)
            closeness = 1.0 + max(0.0, (score - (best_score - 0.0035)) / 0.0035)
            uses = self.parent_uses.get(p.id, 0)
            children = self.parent_child_count.get(p.id, 0)
            positives = self.parent_positive_count.get(p.id, 0)

            # Strong novelty pressure prevents repeatedly selecting the same
            # elite solution, while successful parent lineages remain useful.
            novelty = 1.0 / (1.0 + uses)
            outcome_bonus = 1.0 + (positives / max(1, children))
            weights.append(closeness * (0.45 + novelty) * outcome_bonus)

        parent = self.random_state.choices(frontier, weights=weights, k=1)[0]

        # Long stagnation warrants occasional explicit direction changes.
        # Labels are deliberately limited and target different useful regions.
        label = ""
        if self.stagnation_steps >= 12:
            elite = [p for p, s in numeric if s >= best_score - 0.0015]
            mid_frontier = [
                p for p, s in numeric
                if best_score - 0.0035 <= s < best_score - 0.0010
            ]

            if (
                mid_frontier
                and self.label_uses["diverge"] <= self.label_uses["refine"] + 1
                and self.random_state.random() < 0.35
            ):
                parent = min(
                    mid_frontier,
                    key=lambda p: self.parent_uses.get(p.id, 0),
                )
                label = self.DIVERGE_LABEL
            elif elite and self.random_state.random() < 0.25:
                parent = min(
                    elite,
                    key=lambda p: self.parent_uses.get(p.id, 0),
                )
                label = self.REFINE_LABEL

        if label:
            return {label: parent}, {}

        context_count = max(0, num_context_programs or 0)
        available = [p for p in candidates if p.id != parent.id]
        context: List[EvolvedProgram] = []

        # Mix complementary examples: one elite, one close alternative, and
        # one broader frontier candidate instead of showing only near-duplicates.
        elite_pool = [p for p, s in numeric if p.id != parent.id and s >= best_score - 0.0015]
        nearby_pool = [
            p for p, s in numeric
            if p.id != parent.id and best_score - 0.004 <= s < best_score - 0.0010
        ]
        broader_pool = [p for p in available if p.id not in {x.id for x in elite_pool + nearby_pool}]

        for pool in (elite_pool, nearby_pool, broader_pool):
            choices = [p for p in pool if p.id not in {x.id for x in context}]
            if choices and len(context) < context_count:
                context.append(self.random_state.choice(choices))

        remaining = [p for p in available if p.id not in {x.id for x in context}]
        self.random_state.shuffle(remaining)
        context.extend(remaining[: max(0, context_count - len(context))])

        return {"": parent}, {"": context[:context_count]}


# EVOLVE-BLOCK-END