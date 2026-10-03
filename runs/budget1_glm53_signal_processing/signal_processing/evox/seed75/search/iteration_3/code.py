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
    """Search strategy: exploit top-tier parents with usage penalty,
    provide diverse context (best + mid + low + random). Avoid labels
    (they regressed in past runs)."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.usage_counts: Dict[str, int] = {}
        self.best_score = -1.0
        self.stagnation = 0

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        self.usage_counts[program.id] = self.usage_counts.get(program.id, 0)

        s = self._score(program)
        if s > self.best_score:
            # meaningful improvement threshold
            if self.best_score >= 0 and (s - self.best_score) > max(0.01, 0.01 * abs(self.best_score)):
                self.stagnation = 0
            elif self.best_score < 0:
                self.stagnation = 0
            self.best_score = max(self.best_score, s)
        else:
            self.stagnation += 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        if not self.programs:
            raise ValueError("No candidates available for sampling")

        k = num_context_programs or 4
        programs = list(self.programs.values())
        scored = [(self._score(p), p) for p in programs]
        scored = [t for t in scored if t[0] >= 0]
        scored.sort(key=lambda t: t[0], reverse=True)
        n = len(scored)

        # Parent: weighted random from top 40%, penalized by usage count
        top = scored[: max(3, n // 3)]
        weights = [1.0 / (1 + 2 * self.usage_counts.get(p.id, 0)) for _, p in top]
        total = sum(weights)
        r = self.random_state.random() * total
        parent = top[-1][1]
        acc = 0.0
        for (s, p), w in zip(top, weights):
            acc += w
            if r <= acc:
                parent = p
                break
        self.usage_counts[parent.id] = self.usage_counts.get(parent.id, 0) + 1

        # Context: diverse tiers, exclude parent
        context: List[EvolvedProgram] = []
        tiers = []
        if scored:
            tiers.append(scored[0][1])  # best
            tiers.append(scored[n // 2][1])  # median
            tiers.append(scored[-1][1])  # worst (may hold useful insights)
        for p in tiers:
            if p.id != parent.id and all(c.id != p.id for c in context):
                context.append(p)
        # fill remainder with random programs
        rest = [p for p in programs if p.id != parent.id and all(c.id != p.id for c in context)]
        self.random_state.shuffle(rest)
        context.extend(rest[: k - len(context)])

        return {"": parent}, {"": context[:k]}


# EVOLVE-BLOCK-END