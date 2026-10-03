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
    """Exploit-first strategy: mutate top-tier programs, use diverse top
    programs as context, and signal REFINE/DIVERGE when stuck."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = float("-inf")
        self.stall_count = 0
        self.parent_usage: Dict[str, int] = {}

    @staticmethod
    def _score(p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return float("-inf")

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score + 0.01:  # meaningful improvement
            self.best_score = s
            self.stall_count = 0
        else:
            self.stall_count += 1

        self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4
        candidates.sort(key=self._score, reverse=True)
        top = candidates[: max(3, len(candidates) // 4)]

        # Pick parent from top tier, avoiding overused parents.
        pool = sorted(top, key=lambda p: self.parent_usage.get(p.id, 0))
        low_use = [p for p in pool if self.parent_usage.get(p.id, 0) <= 1]
        parent_pool = low_use if low_use else pool[:3]
        parent = self.random_state.choice(parent_pool)

        # Context: best program + diverse top programs (distinct scores preferred).
        context: List[EvolvedProgram] = []
        seen_scores = set()
        for p in candidates:
            if p.id == parent.id:
                continue
            s = self._score(p)
            if s in seen_scores:
                continue
            seen_scores.add(s)
            context.append(p)
            if len(context) >= n_ctx:
                break
        for p in candidates:  # fill if needed
            if len(context) >= n_ctx:
                break
            if p.id != parent.id and all(c.id != p.id for c in context):
                context.append(p)

        # Adaptive labeling on stagnation.
        if self.stall_count >= 8 and self.random_state.random() < 0.3:
            # Deeply stuck: diverge from a mid-tier program for a new direction.
            mid = candidates[len(candidates) // 2] if len(candidates) > 1 else parent
            return {self.DIVERGE_LABEL: mid}, {"": []}
        if self.stall_count >= 3 and self.random_state.random() < 0.5:
            # Plateaued near best: refine the best program.
            best = max(candidates, key=self._score)
            return {self.REFINE_LABEL: best}, {"": []}

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END