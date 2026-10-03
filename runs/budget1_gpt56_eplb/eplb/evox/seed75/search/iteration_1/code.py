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
    """Adaptive exploit/explore search database."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score_seen: Optional[float] = None
        self.best_program_id: Optional[str] = None
        self.last_meaningful_improvement_iteration = 0
        self.parent_usage: Dict[str, int] = {}
        self.context_usage: Dict[str, int] = {}
        self.last_iteration = getattr(self, "last_iteration", 0)

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
        current_iteration = (
            iteration if iteration is not None else getattr(program, "iteration_found", 0)
        )
        if isinstance(current_iteration, int):
            self.last_iteration = max(self.last_iteration, current_iteration)

        # Record actual prior selection usage here so it survives normal add/resume flow.
        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1
        for context_id in program.other_context_ids or []:
            self.context_usage[context_id] = self.context_usage.get(context_id, 0) + 1

        score = self._score(program)
        if score is not None:
            if self.best_score_seen is None:
                self.best_score_seen = score
                self.best_program_id = program.id
                self.last_meaningful_improvement_iteration = self.last_iteration
            elif score > self.best_score_seen:
                delta = score - self.best_score_seen
                relative = delta / max(abs(self.best_score_seen), 1e-12)
                self.best_score_seen = score
                self.best_program_id = program.id
                # Meaningful means >1% relative OR >0.01 absolute.
                if delta > 0.01 or relative > 0.01:
                    self.last_meaningful_improvement_iteration = self.last_iteration

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

        scored = [(self._score(p), p) for p in candidates]
        numeric = [(s, p) for s, p in scored if s is not None]
        numeric.sort(key=lambda item: item[0], reverse=True)

        if not numeric:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        # Select mainly from the elite band, while penalizing repeatedly used parents.
        elite_count = max(2, int(math.ceil(len(numeric) * 0.4)))
        elite = numeric[:elite_count]
        low_score = elite[-1][0]
        high_score = elite[0][0]
        weights = []
        for score, program in elite:
            quality = 1.0 + (score - low_score) / max(high_score - low_score, 1e-9)
            reuse_penalty = 1.0 + self.parent_usage.get(program.id, 0)
            weights.append(quality / reuse_penalty)

        parent = self.random_state.choices(
            [program for _, program in elite], weights=weights, k=1
        )[0]

        stagnation = self.last_iteration - self.last_meaningful_improvement_iteration
        # Labels are reserved for sustained stagnation, not normal search.
        if stagnation >= 8 and len(numeric) >= 5:
            label = (
                self.DIVERGE_LABEL
                if self.random_state.random() < 0.55
                else self.REFINE_LABEL
            )
            target = parent if label == self.REFINE_LABEL else self.random_state.choice(
                [p for _, p in numeric[:elite_count]]
            )
            return {label: target}, {}

        context_count = max(0, num_context_programs or 0)
        available = [p for _, p in numeric if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Give the model complementary evidence: best alternative, middle solution,
        # and a lower-scoring but potentially different approach.
        if available and context_count:
            anchors = [
                available[0],
                available[len(available) // 2],
                available[-1],
            ]
            for candidate in anchors:
                if candidate.id not in {p.id for p in contexts}:
                    contexts.append(candidate)
                if len(contexts) >= context_count:
                    break

        remaining = [p for p in available if p.id not in {c.id for c in contexts}]
        while len(contexts) < context_count and remaining:
            weights = [1.0 / (1.0 + self.context_usage.get(p.id, 0)) for p in remaining]
            chosen = self.random_state.choices(remaining, weights=weights, k=1)[0]
            contexts.append(chosen)
            remaining = [p for p in remaining if p.id != chosen.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END