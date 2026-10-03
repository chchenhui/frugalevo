# EVOLVE-BLOCK-START
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive search: exploit top scorers with REFINE, periodically DIVERGE
    from diverse parents when progress stalls. Context mixes top programs with
    diverse ones so the LLM sees contrasting approaches."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = -float("inf")
        self.stagnation = 0
        self.sample_calls = 0
        self.parent_usage: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score + 0.01:
            self.best_score = s
            self.stagnation = 0
        else:
            self.stagnation += 1

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4
        self.sample_calls += 1

        # Rank by score
        ranked = sorted(candidates, key=self._score, reverse=True)
        top_tier = [p for p in ranked[:10] if self._score(p) > 0]
        diverse_tier = [p for p in ranked if self._score(p) > 0][10:]

        # Choose parent: mostly top tier (exploit), rotate to avoid overuse
        pool = top_tier if top_tier else ranked
        # Prefer least-used good parents
        pool_sorted = sorted(pool, key=lambda p: (self.parent_usage.get(p.id, 0), -self._score(p)))
        parent = pool_sorted[0] if self.stagnation < 3 else self.random_state.choice(pool)

        label = ""
        if self.stagnation >= 3:
            # Stalled: alternate refine on best vs diverge from a different parent
            if self.sample_calls % 3 == 0 and diverse_tier:
                parent = self.random_state.choice(diverse_tier)
                label = self.DIVERGE_LABEL
            else:
                parent = ranked[0]
                label = self.REFINE_LABEL

        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Context: mix of top programs and diverse ones, excluding parent
        others = [p for p in ranked if p.id != parent.id]
        top_ctx = [p for p in others if p in top_tier][: n_ctx // 2]
        rest = [p for p in others if p not in top_ctx]
        self.random_state.shuffle(rest)
        examples = top_ctx + rest[: n_ctx - len(top_ctx)]

        if label:
            parent_dict = {label: parent}
        else:
            parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END