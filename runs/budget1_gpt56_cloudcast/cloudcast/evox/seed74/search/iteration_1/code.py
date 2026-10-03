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
    """Adaptive database balancing strong parents, underused ideas, and recovery."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))

        self.initial_program = None
        self.best_score_seen: Optional[float] = None
        self.initial_score_seen: Optional[float] = None
        self.stagnation_steps = 0

        self.parent_uses: Dict[str, int] = {}
        self.parent_reward_sum: Dict[str, float] = {}
        self.parent_reward_count: Dict[str, int] = {}
        self.parent_last_used: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}

        self.last_iteration = getattr(self, "last_iteration", -1)

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    def _meaningful_improvement(self, old: float, new: float) -> bool:
        return (new - old) > max(0.01, abs(old) * 0.01)

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        """Store a program and learn which parent choices produced gains."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        score = self._score(program)
        if score is not None:
            if self.initial_score_seen is None:
                self.initial_score_seen = score

            if self.best_score_seen is None:
                self.best_score_seen = score
            elif self._meaningful_improvement(self.best_score_seen, score):
                self.best_score_seen = score
                self.stagnation_steps = 0
            else:
                self.best_score_seen = max(self.best_score_seen, score)
                self.stagnation_steps += 1

        parent_id = program.parent_id
        if parent_id:
            self.parent_uses[parent_id] = self.parent_uses.get(parent_id, 0) + 1
            used_iteration = iteration if iteration is not None else program.iteration_found
            if isinstance(used_iteration, int):
                self.parent_last_used[parent_id] = used_iteration

            parent = self.get(parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None:
                delta = score - parent_score
                self.parent_reward_sum[parent_id] = (
                    self.parent_reward_sum.get(parent_id, 0.0) + delta
                )
                self.parent_reward_count[parent_id] = (
                    self.parent_reward_count.get(parent_id, 0) + 1
                )

        if isinstance(program.parent_info, tuple) and len(program.parent_info) >= 1:
            label = program.parent_info[0]
            if label in (self.DIVERGE_LABEL, self.REFINE_LABEL):
                self.label_uses[label] = self.label_uses.get(label, 0) + 1

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug("Added program %s", program.id)
        return program.id

    def _weighted_choice(
        self, candidates: List[EvolvedProgram], weights: List[float]
    ) -> EvolvedProgram:
        total = sum(max(0.0, weight) for weight in weights)
        if total <= 0:
            return self.random_state.choice(candidates)

        target = self.random_state.random() * total
        running = 0.0
        for candidate, weight in zip(candidates, weights):
            running += max(0.0, weight)
            if running >= target:
                return candidate
        return candidates[-1]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Choose a good but not overused parent and contrasting examples."""
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        context_count = max(0, num_context_programs or 0)
        scored = [(program, self._score(program)) for program in candidates]
        numeric = [(program, score) for program, score in scored if score is not None]

        if numeric:
            numeric.sort(key=lambda item: item[1], reverse=True)
            elite_size = max(2, min(len(numeric), max(4, len(numeric) // 3)))
            elite = [program for program, _ in numeric[:elite_size]]

            # During a plateau, deliberately give lower-ranked approaches more
            # chances; otherwise mostly consolidate the current best family.
            explore_probability = 0.45 if self.stagnation_steps >= 5 else 0.20
            parent_pool = candidates if self.random_state.random() < explore_probability else elite

            ranks = {program.id: index for index, (program, _) in enumerate(numeric)}
            pool_weights: List[float] = []
            for program in parent_pool:
                rank = ranks.get(program.id, len(numeric))
                quality = 1.0 + (len(numeric) - rank) / max(1, len(numeric))
                uses = self.parent_uses.get(program.id, 0)
                novelty = 1.0 / (1.0 + 0.45 * uses)

                reward_count = self.parent_reward_count.get(program.id, 0)
                reward = 0.0
                if reward_count:
                    reward = self.parent_reward_sum[program.id] / reward_count
                reward_bonus = 1.25 if reward > 0 else 1.0

                pool_weights.append(quality * novelty * reward_bonus)

            parent = self._weighted_choice(parent_pool, pool_weights)
        else:
            # If the evaluator has not exposed a numeric score yet, avoid
            # repeatedly asking the model to mutate the same ancestor.
            weights = [
                1.0 / (1.0 + self.parent_uses.get(program.id, 0))
                for program in candidates
            ]
            parent = self._weighted_choice(candidates, weights)
            numeric = []

        label = ""
        if self.stagnation_steps >= 8:
            # A long plateau around nearly identical scores is stronger evidence
            # for a different approach than for another ordinary mutation.
            if self.label_uses.get(self.DIVERGE_LABEL, 0) <= self.label_uses.get(
                self.REFINE_LABEL, 0
            ) and self.random_state.random() < 0.40:
                label = self.DIVERGE_LABEL
        elif self.stagnation_steps >= 4 and self.random_state.random() < 0.20:
            label = self.REFINE_LABEL

        if label:
            return {label: parent}, {}

        remaining = [program for program in candidates if program.id != parent.id]
        self.random_state.shuffle(remaining)

        contexts: List[EvolvedProgram] = []
        used_ids = set()

        def take_from(pool: List[EvolvedProgram], limit: int) -> None:
            for program in pool:
                if len(contexts) >= context_count or len(contexts) >= limit:
                    break
                if program.id not in used_ids:
                    contexts.append(program)
                    used_ids.add(program.id)

        if numeric and context_count:
            ranked = [program for program, _ in numeric if program.id != parent.id]
            top_band = ranked[:max(1, min(4, len(ranked)))]
            low_band = ranked[max(0, len(ranked) // 2):]

            self.random_state.shuffle(top_band)
            self.random_state.shuffle(low_band)

            # One strong example preserves useful details; one contrasting
            # example helps escape the population's current convergence.
            take_from(top_band, 1)
            take_from(low_band, 2)

        self.random_state.shuffle(remaining)
        take_from(remaining, context_count)

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END