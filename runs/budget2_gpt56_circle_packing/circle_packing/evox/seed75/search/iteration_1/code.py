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
    """Adaptive elite search with parent and context diversity."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))

        self.best_score = None
        self.add_count = 0
        self.last_meaningful_add = 0
        self.parent_trials: Dict[str, int] = {}
        self.parent_improvements: Dict[str, int] = {}
        self.context_uses: Dict[str, int] = {}
        self.label_attempts: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def _meaningful_improvement(self, old: Optional[float], new: float) -> bool:
        if old is None:
            return True
        return new - old > max(0.01, abs(old) * 0.01)

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        self.add_count += 1

        score = self._score(program)
        previous_best = self.best_score
        if score is not None:
            if self.best_score is None or score > self.best_score:
                self.best_score = score
            if self._meaningful_improvement(previous_best, score):
                self.last_meaningful_add = self.add_count

        if program.parent_id:
            self.parent_trials[program.parent_id] = (
                self.parent_trials.get(program.parent_id, 0) + 1
            )
            parent = self.get(program.parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if (
                score is not None
                and parent_score is not None
                and self._meaningful_improvement(parent_score, score)
            ):
                self.parent_improvements[program.parent_id] = (
                    self.parent_improvements.get(program.parent_id, 0) + 1
                )

        for context_id in program.other_context_ids or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        if program.parent_info and len(program.parent_info) >= 1:
            label = program.parent_info[0]
            if label:
                self.label_attempts[label] = self.label_attempts.get(label, 0) + 1

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

        scored = [(p, self._score(p)) for p in candidates]
        valid = [(p, s) for p, s in scored if s is not None]
        if not valid:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        valid.sort(key=lambda item: item[1], reverse=True)
        best = valid[0][1]
        stagnation = self.add_count - self.last_meaningful_add

        # During a real plateau, make one deliberate clean-slate attempt from
        # a strong but not necessarily most-reused program.
        if (
            stagnation >= 6
            and self.label_attempts.get(self.DIVERGE_LABEL, 0) < 2
            and len(valid) > 1
        ):
            elite = [p for p, s in valid if s >= best - 0.005]
            parent = min(
                elite,
                key=lambda p: (
                    self.parent_trials.get(p.id, 0),
                    self.context_uses.get(p.id, 0),
                    self.random_state.random(),
                ),
            )
            return {self.DIVERGE_LABEL: parent}, {}

        # Keep selection concentrated on viable solutions, while rewarding
        # parents that have produced gains and avoiding repeatedly used ones.
        elite = [p for p, s in valid if s >= best - 0.006]
        if not elite:
            elite = [p for p, _ in valid[: min(4, len(valid))]]

        weights = []
        for p in elite:
            trials = self.parent_trials.get(p.id, 0)
            gains = self.parent_improvements.get(p.id, 0)
            score = self._score(p) or best
            quality = 1.0 + max(0.0, score - (best - 0.01)) * 40.0
            weights.append(quality * (1.0 + gains) / (1.0 + 0.45 * trials))
        parent = self.random_state.choices(elite, weights=weights, k=1)[0]

        wanted = max(0, num_context_programs or 0)
        pool = [p for p, _ in valid if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # First show strong alternatives, then fill with less-reused programs
        # from distinct score tiers rather than near-identical elite repeats.
        top_pool = [p for p in pool if (self._score(p) or -float("inf")) >= best - 0.006]
        self.random_state.shuffle(top_pool)
        top_pool.sort(key=lambda p: self.context_uses.get(p.id, 0))
        if top_pool and wanted:
            contexts.append(top_pool[0])

        remaining = [p for p in pool if p.id not in {c.id for c in contexts}]
        while remaining and len(contexts) < wanted:
            min_use = min(self.context_uses.get(p.id, 0) for p in remaining)
            least_used = [
                p for p in remaining if self.context_uses.get(p.id, 0) == min_use
            ]
            choice = self.random_state.choice(least_used)
            contexts.append(choice)
            remaining = [p for p in remaining if p.id != choice.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END