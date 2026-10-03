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
    """Adaptive score-aware search with parent and context diversity."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))

        # Persisted/rebuilt through add(), including during checkpoint loading.
        self.observed_best_score: Optional[float] = None
        self.meaningful_best_score: Optional[float] = None
        self.stagnant_adds = 0
        self.score_history: List[float] = []
        self.parent_uses: Dict[str, int] = {}
        self.parent_successes: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}
        self.recent_parent_ids: List[str] = []

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _meaningful_improvement(self, old: float, new: float) -> bool:
        return (new - old) > max(0.01, abs(old) * 0.01)

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        """Store a program and update search-outcome statistics."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        score = self._score(program)
        previous_best = self.observed_best_score

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            parent = self.get(program.parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if (
                score is not None
                and parent_score is not None
                and self._meaningful_improvement(parent_score, score)
            ):
                self.parent_successes[program.parent_id] = (
                    self.parent_successes.get(program.parent_id, 0) + 1
                )

        if program.parent_info and program.parent_info[0]:
            label = program.parent_info[0]
            self.label_uses[label] = self.label_uses.get(label, 0) + 1

        if score is not None:
            self.score_history.append(score)
            if self.observed_best_score is None or score > self.observed_best_score:
                self.observed_best_score = score

            if self.meaningful_best_score is None:
                self.meaningful_best_score = score
                self.stagnant_adds = 0
            elif self._meaningful_improvement(self.meaningful_best_score, score):
                self.meaningful_best_score = score
                self.stagnant_adds = 0
            else:
                self.stagnant_adds += 1

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

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

        context_count = max(0, num_context_programs or 0)
        scored = [(program, self._score(program)) for program in candidates]
        numeric = [(program, score) for program, score in scored if score is not None]

        # Retain broad exploration, but make high-quality candidates more likely.
        if numeric:
            numeric.sort(key=lambda item: item[1], reverse=True)
            broad_count = max(1, int(len(numeric) * 0.75))
            broad_pool = numeric[:broad_count]
            pool = broad_pool if self.random_state.random() < 0.82 else numeric

            low_score = numeric[-1][1]
            high_score = numeric[0][1]
            span = max(high_score - low_score, 1e-12)

            weights: List[float] = []
            programs: List[EvolvedProgram] = []
            for program, score in pool:
                normalized_score = (score - low_score) / span
                uses = self.parent_uses.get(program.id, 0)
                successes = self.parent_successes.get(program.id, 0)
                success_rate = successes / max(1, uses)

                # Score matters, but underused and historically productive parents
                # receive substantial opportunity in this narrow-score population.
                weight = 1.0 + normalized_score + 2.0 / (1.0 + uses) + 1.5 * success_rate
                if program.id in self.recent_parent_ids:
                    weight *= 0.25
                programs.append(program)
                weights.append(max(weight, 0.01))

            parent = self.random_state.choices(programs, weights=weights, k=1)[0]
        else:
            parent = self.random_state.choice(candidates)

        # This is selection state only; durable outcome/usage state is updated in add().
        self.recent_parent_ids.append(parent.id)
        self.recent_parent_ids = self.recent_parent_ids[-3:]

        remaining = [p for p in candidates if p.id != parent.id]
        selected: List[EvolvedProgram] = []
        used_ids = set()
        prior_context_ids = set(parent.other_context_ids or [])

        # Provide contrasting score levels: an elite reference, a middle approach,
        # and a lower/radically different approach. Avoid repeating parent history.
        ranked = sorted(
            remaining,
            key=lambda p: self._score(p) if self._score(p) is not None else float("-inf"),
            reverse=True,
        )
        groups = [
            ranked[:max(1, len(ranked) // 4)],
            ranked[len(ranked) // 3:max(1, 2 * len(ranked) // 3)],
            ranked[max(0, 3 * len(ranked) // 4):],
        ]

        for group in groups:
            if len(selected) >= context_count:
                break
            fresh = [p for p in group if p.id not in used_ids and p.id not in prior_context_ids]
            choices = fresh or [p for p in group if p.id not in used_ids]
            if choices:
                choice = self.random_state.choice(choices)
                selected.append(choice)
                used_ids.add(choice.id)

        fill_pool = [p for p in remaining if p.id not in used_ids and p.id not in prior_context_ids]
        if len(fill_pool) < context_count - len(selected):
            fill_pool = [p for p in remaining if p.id not in used_ids]

        self.random_state.shuffle(fill_pool)
        selected.extend(fill_pool[:max(0, context_count - len(selected))])

        # Keep labels empty: previous labeled attempts were unproductive, while
        # diverse unlabeled mutations have produced the observed improvements.
        return {"": parent}, {"": selected}


# EVOLVE-BLOCK-END