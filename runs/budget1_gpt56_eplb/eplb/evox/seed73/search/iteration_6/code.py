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
    """Adaptive score-stratified search with parent-credit tracking."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score: Optional[float] = None
        self.last_meaningful_iteration = 0
        self.add_count = 0
        self.parent_trials: Dict[str, int] = {}
        self.parent_gain: Dict[str, float] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        score = self._score(program)
        event_iteration = iteration
        if event_iteration is None and isinstance(program.iteration_found, int):
            event_iteration = program.iteration_found
        if event_iteration is None:
            event_iteration = self.add_count

        # Credit parent choices only after their child has been evaluated.
        if program.parent_id:
            self.parent_trials[program.parent_id] = (
                self.parent_trials.get(program.parent_id, 0) + 1
            )
            parent = self.get(program.parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None:
                self.parent_gain[program.parent_id] = (
                    self.parent_gain.get(program.parent_id, 0.0)
                    + max(0.0, score - parent_score)
                )

        if score is not None:
            if self.best_score is None:
                self.best_score = score
                self.last_meaningful_iteration = event_iteration
            elif score > self.best_score:
                improvement = score - self.best_score
                meaningful = (
                    improvement > 0.01
                    or improvement > 0.01 * max(abs(self.best_score), 1e-9)
                )
                self.best_score = score
                if meaningful:
                    self.last_meaningful_iteration = event_iteration

        self.programs[program.id] = program
        self.add_count += 1

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
        scored = [(p, self._score(p)) for p in self.programs.values()]
        scored = [(p, s) for p, s in scored if s is not None]
        if not scored:
            candidates = list(self.programs.values())
            if not candidates:
                raise ValueError("No candidates available for sampling")
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        scored.sort(key=lambda item: item[1], reverse=True)
        programs = [item[0] for item in scored]
        scores = [item[1] for item in scored]
        n = len(programs)
        q25 = scores[min(n - 1, int(0.75 * (n - 1)))]
        q75 = scores[min(n - 1, int(0.25 * (n - 1)))]
        current_iteration = max(self.last_iteration, self.add_count)
        stalled = current_iteration - self.last_meaningful_iteration >= 8

        # A stalled population deliberately emphasizes the productive middle:
        # this is where prior runs produced rare jumps beyond the frontier.
        weights: List[float] = []
        for program, score in scored:
            trials = self.parent_trials.get(program.id, 0)
            gain = self.parent_gain.get(program.id, 0.0)
            if stalled and q25 <= score <= q75:
                base = 2.2
            elif score >= q75:
                base = 1.4
            elif score >= q25:
                base = 1.0
            else:
                base = 0.45

            productivity = 1.0 + min(2.0, gain * 120.0)
            reuse_penalty = 1.0 / (1.0 + 0.35 * trials)
            weights.append(max(0.05, base * productivity * reuse_penalty))

        parent = self.random_state.choices(programs, weights=weights, k=1)[0]
        context_count = max(0, num_context_programs or 0)
        available = [p for p in programs if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Use complementary score tiers: strong implementations provide useful
        # details while middle/lower alternatives preserve different strategies.
        tiers = [
            available[:max(1, len(available) // 5)],
            available[max(1, len(available) // 5):max(2, len(available) // 2)],
            available[max(2, len(available) // 2):],
        ]
        while len(contexts) < context_count and available:
            tier = tiers[len(contexts) % len(tiers)]
            choices = [p for p in tier if p.id not in {c.id for c in contexts}]
            if not choices:
                choices = [p for p in available if p.id not in {c.id for c in contexts}]
            if not choices:
                break
            contexts.append(self.random_state.choice(choices))

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END