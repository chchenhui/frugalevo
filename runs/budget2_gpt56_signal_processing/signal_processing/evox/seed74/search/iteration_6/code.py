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
    """Adaptive frontier search with occasional deliberate escape attempts."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_iteration = -1
        self.last_iteration_seen = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_best_gain: Dict[str, float] = {}
        self.label_uses: Dict[str, int] = {}
        self.last_label_iteration = -1000

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        step = iteration if isinstance(iteration, int) else program.iteration_found
        if isinstance(step, int):
            self.last_iteration_seen = max(self.last_iteration_seen, step)

        score = self._score(program)
        old_best = self.best_seen
        if score is not None and score > self.best_seen:
            self.best_seen = score
            absolute_gain = score - old_best
            relative_gain = (
                absolute_gain / abs(old_best)
                if old_best != float("-inf") and old_best != 0
                else float("inf")
            )
            if old_best == float("-inf") or absolute_gain > 0.01 or relative_gain > 0.01:
                self.last_meaningful_iteration = (
                    step if isinstance(step, int) else self.last_iteration_seen
                )

        if program.parent_id:
            parent_id = program.parent_id
            self.parent_uses[parent_id] = self.parent_uses.get(parent_id, 0) + 1

            parent = self.get(parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None:
                gain = score - parent_score
                self.parent_best_gain[parent_id] = max(
                    gain, self.parent_best_gain.get(parent_id, float("-inf"))
                )

        if isinstance(program.parent_info, tuple) and program.parent_info:
            label = program.parent_info[0]
            if label:
                self.label_uses[label] = self.label_uses.get(label, 0) + 1
                if isinstance(step, int):
                    self.last_label_iteration = max(self.last_label_iteration, step)

        if isinstance(iteration, int):
            self.last_iteration = max(getattr(self, "last_iteration", 0), iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        scored: List[Tuple[EvolvedProgram, float]] = []
        for program in self.programs.values():
            score = self._score(program)
            if score is not None:
                scored.append((program, score))

        if not scored:
            raise ValueError("No scored candidates available for sampling")

        scored.sort(key=lambda item: item[1], reverse=True)
        total = len(scored)
        current_iteration = max(
            self.last_iteration_seen,
            getattr(self, "last_iteration", 0),
            max(
                p.iteration_found if isinstance(p.iteration_found, int) else 0
                for p, _ in scored
            ),
        )
        stalled = current_iteration - self.last_meaningful_iteration
        wanted = max(0, num_context_programs or 0)

        elite_n = min(total, max(6, int(math.ceil(total * 0.25))))
        upper_n = min(total, max(elite_n + 4, int(math.ceil(total * 0.50))))
        middle_start = min(total - 1, max(elite_n, int(total * 0.35)))
        middle_end = min(total, max(middle_start + 3, int(total * 0.70)))

        elites = scored[:elite_n]
        upper = scored[:upper_n]
        middle = scored[middle_start:middle_end]

        # A long plateau means minor variants of the repeated frontier are less
        # useful. Pick a less-exhausted alternate parent, while retaining elite
        # parents often enough to consolidate any new useful idea.
        if stalled >= 10 and middle and self.random_state.random() < 0.60:
            parent_pool = middle
        else:
            parent_pool = upper

        weights: List[float] = []
        for rank, (program, score) in enumerate(parent_pool):
            rank_bonus = 1.0 + 2.0 * (len(parent_pool) - rank) / len(parent_pool)
            gain_bonus = 1.0 + min(
                1.5, max(0.0, self.parent_best_gain.get(program.id, 0.0)) * 30.0
            )
            reuse_penalty = 1.0 + 0.55 * self.parent_uses.get(program.id, 0)
            weights.append(rank_bonus * gain_bonus / reuse_penalty)

        parent = self.random_state.choices(
            [program for program, _ in parent_pool], weights=weights, k=1
        )[0]

        # Labels are rare escape tools, not a default mutation mode. A plateau
        # this long merits one targeted direction change before more normal
        # recombination is attempted.
        label_cooldown = current_iteration - self.last_label_iteration
        if stalled >= 15 and label_cooldown >= 7:
            if self.label_uses.get(self.DIVERGE_LABEL, 0) <= self.label_uses.get(
                self.REFINE_LABEL, 0
            ):
                diverge_pool = middle if middle else upper
                parent = self.random_state.choice([p for p, _ in diverge_pool])
                return {self.DIVERGE_LABEL: parent}, {"": []}
            parent = self.random_state.choice([p for p, _ in elites])
            return {self.REFINE_LABEL: parent}, {"": []}

        contexts: List[EvolvedProgram] = []
        used_ids = {parent.id}

        def choose_from(pool: List[Tuple[EvolvedProgram, float]]) -> None:
            available = [p for p, _ in pool if p.id not in used_ids]
            if available and len(contexts) < wanted:
                choice = self.random_state.choice(available)
                contexts.append(choice)
                used_ids.add(choice.id)

        # Each prompt receives a frontier reference plus contrasting alternatives.
        choose_from(elites[:min(4, len(elites))])
        choose_from(elites[4:] if len(elites) > 4 else upper[elite_n:upper_n])
        choose_from(middle)

        remaining = [p for p, _ in scored if p.id not in used_ids]
        while remaining and len(contexts) < wanted:
            choice = self.random_state.choice(remaining)
            contexts.append(choice)
            used_ids.add(choice.id)
            remaining = [p for p in remaining if p.id != choice.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END