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
    """Search strategy for a converged, stagnating population.

    Key ideas:
    - Avoid reusing the same parent/context (track usage counts).
    - Rotate parents across score tiers (best, mid, low) instead of always
      picking the incumbent best, which has repeatedly failed to improve.
    - When stagnating, use DIVERGE on a low-usage parent with fresh context.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.context_usage: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.stagnation: int = 0
        self.sample_calls: int = 0

    @staticmethod
    def _score(p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        self.parent_usage.setdefault(program.id, 0)
        self.context_usage.setdefault(program.id, 0)

        s = self._score(program)
        if s > self.best_score + max(0.01, 0.01 * abs(self.best_score)):
            self.best_score = s
            self.stagnation = 0
        else:
            self.stagnation += 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def _pick(self, pool: List[EvolvedProgram], usage: Dict[str, int]) -> Optional[EvolvedProgram]:
        """Pick the least-used program from pool, breaking ties randomly."""
        if not pool:
            return None
        min_u = min(usage.get(p.id, 0) for p in pool)
        least = [p for p in pool if usage.get(p.id, 0) == min_u]
        return self.random_state.choice(least)

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = [p for p in self.programs.values() if self._score(p) > 0]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4
        self.sample_calls += 1

        # Tier the population.
        scored = sorted(candidates, key=self._score, reverse=True)
        top = scored[: max(1, len(scored) // 4)]
        low = scored[-max(1, len(scored) // 4):]
        mid = scored[len(top):len(scored) - len(low)] or scored

        # Rotate tiers round-robin; on stagnation prefer divergence from
        # low/mid-tier, under-used parents (incumbent refine has failed).
        if self.stagnation >= 3:
            parent = self._pick(low + mid, self.parent_usage)
            label = self.DIVERGE_LABEL
            examples: List[EvolvedProgram] = []
        else:
            tier = [top, mid, low][self.sample_calls % 3]
            parent = self._pick(tier, self.parent_usage)
            label = ""
            rest = [p for p in candidates if p.id != parent.id]
            self.random_state.shuffle(rest)
            # Mix: one top scorer + diverse least-used others.
            examples = []
            if top and top[0].id != parent.id:
                examples.append(top[0])
            for p in sorted(rest, key=lambda q: self.context_usage.get(q.id, 0)):
                if p.id not in [e.id for e in examples]:
                    examples.append(p)
                if len(examples) >= n_ctx:
                    break

        if parent is None:
            parent = self.random_state.choice(candidates)
            label = ""
            examples = []

        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
        for e in examples:
            self.context_usage[e.id] = self.context_usage.get(e.id, 0) + 1

        return {label: parent}, {"": examples}


# EVOLVE-BLOCK-END