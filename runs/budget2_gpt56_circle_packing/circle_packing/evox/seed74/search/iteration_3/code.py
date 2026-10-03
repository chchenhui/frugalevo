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
    """Adaptive elite search with novelty-aware parent and context selection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_iteration = 0
        self.last_seen_iteration = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_successes: Dict[str, int] = {}

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
        score = self._score(program)
        found_iteration = iteration
        if found_iteration is None:
            found_iteration = program.iteration_found
        if isinstance(found_iteration, int):
            self.last_seen_iteration = max(self.last_seen_iteration, found_iteration)
            self.last_iteration = max(self.last_iteration, found_iteration)

        # Record whether a selected parent generated a genuinely better child.
        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            parent = self.get(program.parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None:
                threshold = max(0.01, 0.01 * abs(parent_score))
                if score > parent_score + threshold:
                    self.parent_successes[program.parent_id] = (
                        self.parent_successes.get(program.parent_id, 0) + 1
                    )

        if score is not None:
            threshold = max(0.01, 0.01 * abs(self.best_seen)) if math.isfinite(self.best_seen) else 0.0
            if score > self.best_seen + threshold:
                self.last_meaningful_iteration = self.last_seen_iteration
            self.best_seen = max(self.best_seen, score)

        self.programs[program.id] = program

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)

        logger.debug("Added program %s to the evolve database", program.id)
        return program.id

    def _weighted_choice(
        self, programs: List[EvolvedProgram], weights: List[float]
    ) -> EvolvedProgram:
        total = sum(weights)
        if total <= 0:
            return self.random_state.choice(programs)
        point = self.random_state.random() * total
        running = 0.0
        for program, weight in zip(programs, weights):
            running += weight
            if running >= point:
                return program
        return programs[-1]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        scored = [(p, self._score(p)) for p in self.programs.values()]
        scored = [(p, s) for p, s in scored if s is not None]
        if not scored:
            raise ValueError("No scored candidates available for sampling")

        scored.sort(key=lambda item: item[1], reverse=True)
        best = scored[0][1]
        context_count = max(0, num_context_programs or 0)

        # Keep search near the strong packing frontier, including the slightly
        # lower 1.0003-style candidates that have previously produced the best.
        elite = [p for p, s in scored if s >= best - 0.004]
        if not elite:
            elite = [p for p, _ in scored[: min(6, len(scored))]]

        # One representative per exact solution reduces repeated mutations of
        # cloned high-scoring programs.
        representatives: List[EvolvedProgram] = []
        seen_solutions = set()
        for program in elite:
            signature = program.solution
            if signature not in seen_solutions:
                representatives.append(program)
                seen_solutions.add(signature)
        parent_pool = representatives if representatives else elite

        weights: List[float] = []
        for program in parent_pool:
            score = self._score(program) or 0.0
            uses = self.parent_uses.get(program.id, 0)
            successes = self.parent_successes.get(program.id, 0)
            score_weight = 1.0 + max(0.0, score - (best - 0.01)) * 40.0
            novelty_weight = 1.0 / (1.0 + uses)
            evidence_weight = 1.0 + 2.0 * successes
            weights.append(score_weight * novelty_weight * evidence_weight)

        parent = self._weighted_choice(parent_pool, weights)

        stagnating = (
            self.last_seen_iteration - self.last_meaningful_iteration >= 12
        )

        # Labels are rare tools for a real plateau; ordinary contextual mutation
        # remains the default because it has been the productive mode here.
        if stagnating and self.random_state.random() < 0.12:
            if self.random_state.random() < 0.5 and len(scored) > len(elite):
                alternatives = [p for p, s in scored if best - 0.03 <= s < best - 0.004]
                if alternatives:
                    parent = self.random_state.choice(alternatives)
                    return {self.DIVERGE_LABEL: parent}, {}
            return {self.REFINE_LABEL: parent}, {}

        # Context combines alternative elite implementations with one nearby
        # contrast, while never repeating the selected parent or exact clones.
        remaining = [p for p, _ in scored if p.id != parent.id and p.solution != parent.solution]
        elite_context = [p for p in remaining if p in elite]
        near_context = [p for p, s in scored if p in remaining and best - 0.03 <= s < best - 0.004]

        self.random_state.shuffle(elite_context)
        self.random_state.shuffle(near_context)
        contexts: List[EvolvedProgram] = []

        for pool in (elite_context, near_context, remaining):
            for program in pool:
                if program.id not in {p.id for p in contexts}:
                    contexts.append(program)
                if len(contexts) >= context_count:
                    break
            if len(contexts) >= context_count:
                break

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END