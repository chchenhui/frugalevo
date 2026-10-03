# EVOLVE-BLOCK-START
import logging
import math
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
    """Adaptive elite search with occasional focused refinement."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_observed_score = float("-inf")
        self.stagnation_count = 0
        self.parent_use_count: Dict[str, int] = {}
        self.context_use_count: Dict[str, int] = {}
        self.label_use_count: Dict[str, int] = {}
        self.score_history: List[float] = []

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        score = self._score(program)
        if score is not None:
            self.score_history.append(score)
            meaningful_gain = (
                self.best_observed_score == float("-inf")
                or score > self.best_observed_score
                + min(0.01, max(0.0, self.best_observed_score) * 0.01)
            )
            if meaningful_gain:
                self.best_observed_score = score
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

        if program.parent_id:
            self.parent_use_count[program.parent_id] = (
                self.parent_use_count.get(program.parent_id, 0) + 1
            )
        for context_id in program.other_context_ids or []:
            self.context_use_count[context_id] = (
                self.context_use_count.get(context_id, 0) + 1
            )

        if program.parent_info and len(program.parent_info) >= 1:
            label = program.parent_info[0]
            if label:
                self.label_use_count[label] = self.label_use_count.get(label, 0) + 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug("Added program %s", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        context_count = max(0, num_context_programs or 0)
        scored = [(self._score(p), p) for p in candidates]
        valid = [(s, p) for s, p in scored if s is not None]
        valid.sort(key=lambda item: item[0], reverse=True)

        if not valid:
            parent = self.random_state.choice(candidates)
            pool = [p for p in candidates if p.id != parent.id]
            self.random_state.shuffle(pool)
            return {"": parent}, {"": pool[:context_count]}

        # Use a broad elite set rather than always repeating the single best.
        elite_size = max(3, int(math.ceil(len(valid) * 0.30)))
        elite = valid[:min(elite_size, len(valid))]

        # Deep stagnation calls for a few concentrated attempts to improve a
        # promising solution. Previous divergence attempts did not help here.
        refinements = self.label_use_count.get(self.REFINE_LABEL, 0)
        if self.stagnation_count >= 10 and refinements < 3:
            top_choices = [p for _, p in elite[:min(4, len(elite))]]
            weights = [
                1.0 / (1 + self.parent_use_count.get(p.id, 0))
                for p in top_choices
            ]
            parent = self.random_state.choices(top_choices, weights=weights, k=1)[0]
            return {self.REFINE_LABEL: parent}, {}

        # Prefer strong programs, while reducing repeated mutations of one
        # lineage. Rank weighting keeps this stochastic.
        elite_programs = [p for _, p in elite]
        weights = [
            (len(elite_programs) - i) / (1 + self.parent_use_count.get(p.id, 0))
            for i, p in enumerate(elite_programs)
        ]
        parent = self.random_state.choices(elite_programs, weights=weights, k=1)[0]

        # Contexts deliberately mix excellent examples with alternatives,
        # including occasional lower-score attempts that may contain a useful
        # different geometric construction.
        remaining = [p for _, p in valid if p.id != parent.id]
        chosen: List[EvolvedProgram] = []

        def pick_from(pool: List[EvolvedProgram]) -> None:
            pool = [p for p in pool if p.id not in {x.id for x in chosen}]
            if pool and len(chosen) < context_count:
                weights = [
                    1.0 / (1 + self.context_use_count.get(p.id, 0)) for p in pool
                ]
                chosen.append(self.random_state.choices(pool, weights=weights, k=1)[0])

        pick_from([p for _, p in elite if p.id != parent.id])
        if remaining:
            middle_start = min(elite_size, len(remaining))
            middle_end = max(middle_start + 1, int(len(valid) * 0.75))
            pick_from(remaining[middle_start:middle_end])
            pick_from(remaining[middle_end:])
            pick_from(remaining)

        return {"": parent}, {"": chosen[:context_count]}


# EVOLVE-BLOCK-END