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
    """Score-aware, diversity-preserving program search database."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.initial_program = None

        # Rebuilt through add(), including when a database is restored.
        self.best_score_seen = float("-inf")
        self.best_iteration = 0
        self.last_meaningful_improvement = 0
        self.parent_usage: Dict[str, int] = {}
        self.context_usage: Dict[str, int] = {}
        self.parent_success: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

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

        # Account for actual parent/context use here rather than in sample(),
        # so state remains correct after checkpoint restore.
        if program.parent_id:
            self.parent_usage[program.parent_id] = (
                self.parent_usage.get(program.parent_id, 0) + 1
            )
            parent = self.get(program.parent_id)
            child_score = self._score(program)
            parent_score = self._score(parent) if parent is not None else None
            if (
                child_score is not None
                and parent_score is not None
                and (
                    child_score - parent_score > 0.01
                    or child_score > parent_score * 1.01
                )
            ):
                self.parent_success[program.parent_id] = (
                    self.parent_success.get(program.parent_id, 0) + 1
                )

        for context_id in program.other_context_ids or []:
            self.context_usage[context_id] = self.context_usage.get(context_id, 0) + 1

        score = self._score(program)
        if score is not None:
            if score > self.best_score_seen:
                previous_best = self.best_score_seen
                self.best_score_seen = score
                self.best_iteration = current_iteration
                if (
                    previous_best == float("-inf")
                    or score - previous_best > 0.01
                    or score > previous_best * 1.01
                ):
                    self.last_meaningful_improvement = current_iteration

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
        numeric = [(s, p) for s, p in scored if s is not None]

        # Invalid/unscored candidates are retained as a small exploration option,
        # but valid high scoring programs remain the normal mutation base.
        if not numeric:
            parent = self.random_state.choice(candidates)
            pool = [p for p in candidates if p.id != parent.id]
            self.random_state.shuffle(pool)
            return {"": parent}, {"": pool[:context_count]}

        numeric.sort(key=lambda item: item[0], reverse=True)
        population_size = len(numeric)
        current_iteration = getattr(self, "last_iteration", 0)
        stalled = current_iteration - self.last_meaningful_improvement >= 8

        # During a plateau broaden from the elite rather than repeatedly applying
        # an explicit divergence instruction that has already been unproductive.
        pool_size = max(4, int(math.ceil(population_size * (0.65 if stalled else 0.4))))
        parent_pool = [p for _, p in numeric[:pool_size]]

        # Rank rewards quality; low reuse prevents the many tied near-best
        # candidates from collapsing to one repeatedly sampled parent.
        weights: List[float] = []
        for rank, candidate in enumerate(parent_pool):
            quality = 1.0 + (pool_size - rank) / float(pool_size)
            reuse_penalty = 1.0 / (1.0 + self.parent_usage.get(candidate.id, 0))
            success_bonus = 1.0 + 0.35 * self.parent_success.get(candidate.id, 0)
            weights.append(quality * (0.45 + reuse_penalty) * success_bonus)

        parent = self.random_state.choices(parent_pool, weights=weights, k=1)[0]

        # Context deliberately combines strong examples with a score-distinct
        # bridge candidate.  This gives the model alternative constructions
        # without continually reusing the same low-score context.
        remaining = [p for _, p in numeric if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        if remaining and context_count:
            # One elite reference, selected randomly among tied/near-tied leaders.
            elite_count = min(len(remaining), max(2, int(math.ceil(len(remaining) * 0.2))))
            contexts.append(self.random_state.choice(remaining[:elite_count]))

        while len(contexts) < context_count:
            available = [p for p in remaining if p.id not in {x.id for x in contexts}]
            if not available:
                break

            # Prefer underused context and, after the first example, a different
            # score tier from contexts already supplied.
            chosen_scores = [self._score(p) for p in contexts]
            weights = []
            for candidate in available:
                score = self._score(candidate)
                distance = min(
                    [abs(score - s) for s in chosen_scores if s is not None] or [0.0]
                )
                novelty = 1.0 + min(distance * 80.0, 1.5)
                reuse = 1.0 / (1.0 + self.context_usage.get(candidate.id, 0))
                weights.append((0.4 + reuse) * novelty)

            contexts.append(self.random_state.choices(available, weights=weights, k=1)[0])

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END