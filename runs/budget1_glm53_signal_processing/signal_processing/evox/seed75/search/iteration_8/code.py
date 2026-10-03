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


def _score(p) -> float:
    v = p.metrics.get("combined_score")
    if isinstance(v, (int, float)):
        return float(v)
    return -1.0


class EvolvedProgramDatabase(ProgramDatabase):
    """Usage-penalized mid-tier parent selection with best-in-context pairing.

    Evidence from prior runs: unlabeled sampling where a mid-tier parent is
    paired with the best program as context produced all top-tier children;
    labeled (REFINE/DIVERGE) generations consistently regressed. So we keep
    labels rare and only apply DIVERGE to a *top-scored* parent after long
    stagnation, with diverse context (not empty).
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.usage_count: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.stagnation: int = 0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = _score(program)
        if s > self.best_score + max(0.01, 0.01 * abs(self.best_score)):
            self.best_score = s
            self.stagnation = 0
        else:
            self.stagnation += 1

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        n_ctx = num_context_programs or 4
        progs = list(self.programs.values())
        if not progs:
            raise ValueError("No candidates available for sampling")

        scored = sorted(progs, key=_score, reverse=True)
        best = scored[0]

        # Track parent usage to avoid overuse
        for pid, c in self.usage_count.items():
            if pid not in self.programs:
                continue

        # Rare divergence: after deep stagnation, diverge from a TOP program
        # (prior runs wasted DIVERGE on weak parents) with diverse context.
        if self.stagnation >= 6 and self.random_state.random() < 0.25:
            top_pool = scored[:5]
            parent = self.random_state.choice(top_pool)
            others = [p for p in scored if p.id != parent.id]
            self.random_state.shuffle(others)
            ctx = others[:n_ctx]
            self.usage_count[parent.id] = self.usage_count.get(parent.id, 0) + 1
            self.stagnation = max(0, self.stagnation - 3)
            return {self.DIVERGE_LABEL: parent}, {"": ctx}

        # Default: mid-tier parent (0.68-0.70 band historically jumps to best-tier
        # when best is in context), penalized by usage.
        mid = scored[1:max(4, len(scored) // 2)]
        if not mid:
            mid = scored
        weights = []
        for p in mid:
            u = self.usage_count.get(p.id, 0)
            weights.append(1.0 / (1.0 + u))
        parent = self.random_state.choices(mid, weights=weights, k=1)[0]
        self.usage_count[parent.id] = self.usage_count.get(parent.id, 0) + 1

        # Context: always include best program (proven booster), plus diverse
        # samples from different score bands.
        ctx = [best]
        bands = [scored[1:10], scored[len(scored) // 2:][:5], scored[-5:]]
        for band in bands:
            band = [p for p in band if p.id != parent.id and p.id != best.id
                    and all(p.id != c.id for c in ctx)]
            if band:
                ctx.append(self.random_state.choice(band))
            if len(ctx) >= n_ctx:
                break
        # Fill remaining slots randomly
        rest = [p for p in scored if all(p.id != c.id for c in ctx) and p.id != parent.id]
        self.random_state.shuffle(rest)
        ctx.extend(rest[: n_ctx - len(ctx)])

        return {"": parent}, {"": ctx[:n_ctx]}


# EVOLVE-BLOCK-END