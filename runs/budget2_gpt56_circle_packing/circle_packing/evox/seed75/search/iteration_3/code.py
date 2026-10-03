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
    """Adaptive elite search with parent-use balancing and mixed contexts."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_uses: Dict[str, int] = {}
        self.context_uses: Dict[str, int] = {}
        self.best_score = float("-inf")
        self.meaningful_best = float("-inf")
        self.last_meaningful_iteration = 0
        self.last_seen_iteration = 0

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
        self.programs[program.id] = program

        found_at = iteration if isinstance(iteration, int) else program.iteration_found
        if not isinstance(found_at, int):
            found_at = 0
        self.last_seen_iteration = max(self.last_seen_iteration, found_at)

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
        for context_id in program.other_context_ids or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        score = self._score(program)
        if score is not None and score > self.best_score:
            self.best_score = score

        # Meaningful progress is deliberately stricter than small evaluator noise.
        if score is not None and (
            self.meaningful_best == float("-inf")
            or score > self.meaningful_best
            + max(0.01, abs(self.meaningful_best) * 0.01)
        ):
            self.meaningful_best = score
            self.last_meaningful_iteration = found_at

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(p, self._score(p)) for p in candidates]
        numeric = [(p, s) for p, s in scored if s is not None]
        numeric.sort(key=lambda item: item[1], reverse=True)

        if numeric:
            best = numeric[0][1]
            # At this near-ceiling stage, retain several distinct near-best ideas.
            elite = [p for p, s in numeric if s >= best - 0.0025]
            pool = elite if self.random_state.random() < 0.82 else [p for p, _ in numeric]
            weights = [
                (1.0 / (1 + self.parent_uses.get(p.id, 0))) * (1.0 + 0.15 * (len(pool) - i))
                for i, p in enumerate(pool)
            ]
            parent = self.random_state.choices(pool, weights=weights, k=1)[0]
        else:
            parent = self.random_state.choice(candidates)

        stalled = self.last_seen_iteration - self.last_meaningful_iteration >= 10
        label = ""
        if stalled and self.random_state.random() < 0.28:
            # Mostly polish strong layouts; occasionally request a genuinely new layout.
            label = self.REFINE_LABEL if self.random_state.random() < 0.65 else self.DIVERGE_LABEL
            return {label: parent}, {}

        count = max(0, num_context_programs or 0)
        available = [p for p in candidates if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Contexts intentionally mix competing elite layouts and less-converged ideas.
        if numeric:
            best = numeric[0][1]
            elite_other = [p for p, s in numeric if p.id != parent.id and s >= best - 0.003]
            mid = [p for p, s in numeric if p.id != parent.id and s < best - 0.003]
            self.random_state.shuffle(elite_other)
            self.random_state.shuffle(mid)
            ordered = elite_other[:2] + mid[:2] + elite_other[2:] + mid[2:]
        else:
            ordered = available[:]
            self.random_state.shuffle(ordered)

        for program in ordered:
            if len(contexts) >= count:
                break
            if program.id not in {p.id for p in contexts}:
                contexts.append(program)

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END