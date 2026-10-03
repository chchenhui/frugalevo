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
    v = p.metrics.get("combined_score", 0.0)
    return float(v) if isinstance(v, (int, float)) else 0.0


class EvolvedProgramDatabase(ProgramDatabase):
    """Plateau-breaking strategy: mostly REFINE the current best with fresh
    diverse context; occasionally DIVERGE from an underused mid-tier parent
    (the pattern that previously broke the 0.9928 plateau)."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.since_improvement = 0
        self.best_score = -1.0
        self.parent_use_count: Dict[str, int] = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = _score(program)
        # meaningful improvement tracking
        if s > self.best_score + 0.001:
            self.best_score = max(self.best_score, s)
            self.since_improvement = 0
        else:
            self.since_improvement += 1

        # track parent usage
        pid = program.parent_id
        if pid:
            self.parent_use_count[pid] = self.parent_use_count.get(pid, 0) + 1

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        logger.debug(f"Added program {program.id}")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        k = num_context_programs or 4
        best = max(candidates, key=_score)

        # Decide mode: stagnating -> occasionally diverge from a fresh mid-tier parent
        use_diverge = self.since_improvement >= 3 and self.random_state.random() < 0.4

        if use_diverge:
            # mid-tier = between 25th pct and 90th pct, least used as parent
            scores = sorted(_score(p) for p in candidates)
            lo = scores[len(scores) // 4] if scores else 0.0
            hi = scores[int(len(scores) * 0.9)] if scores else 1.0
            pool = [p for p in candidates if lo < _score(p) < hi]
            if pool:
                parent = min(pool, key=lambda p: self.parent_use_count.get(p.id, 0))
                return {self.DIVERGE_LABEL: parent}, {"": []}

        # Default: refine the best program
        parent = best
        # Context: diverse, avoid reusing same context sets — mix good & mid programs
        others = [p for p in candidates if p.id != parent.id]
        self.random_state.shuffle(others)
        # bias: take 2 from top tier, rest random (already shuffled)
        top = sorted(others, key=_score, reverse=True)[:6]
        ctx = []
        if top:
            ctx.append(self.random_state.choice(top))
        for p in others:
            if len(ctx) >= k:
                break
            if p not in ctx:
                ctx.append(p)

        # If deeply stuck on refining best, signal REFINE explicitly
        label = self.REFINE_LABEL if self.since_improvement >= 2 else ""
        return {label: parent}, {"": ctx}


# EVOLVE-BLOCK-START