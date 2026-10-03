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


def _score(p: Program) -> float:
    v = p.metrics.get("combined_score") if p.metrics else None
    if isinstance(v, (int, float)):
        return float(v)
    return -1e9


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive exploit/explore sampler.

    Principle: mostly mutate top-tier programs (weighted by score) with
    diverse context drawn from different score bands; when stagnating, fall
    back to targeted REFINE (on best) / DIVERGE (on mid-tier) labels.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = -1e9
        self.stagnation = 0
        self.usage_count: Dict[str, int] = {}
        self.sample_counter = 0

    def _record(self, program: EvolvedProgram, iteration: Optional[int]) -> None:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program
        s = _score(program)
        if s > self.best_score + 1e-9:
            if s > self.best_score + max(0.01, 0.01 * abs(self.best_score)):
                self.stagnation = 0
            else:
                self.stagnation += 1
            self.best_score = s
        else:
            self.stagnation += 1

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        self._record(program, iteration)
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def _sorted_programs(self) -> List[EvolvedProgram]:
        return sorted(self.programs.values(), key=_score, reverse=True)

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        n_ctx = num_context_programs or 4
        self.sample_counter += 1
        programs = self._sorted_programs()
        if not programs:
            raise ValueError("No candidates available for sampling")

        # Track usage to avoid overusing any single parent.
        for pid in self.usage_count:
            pass

        # Deep stagnation: alternate targeted labels.
        if self.stagnation >= 6:
            if self.sample_counter % 2 == 0:
                best = programs[0]
                self.usage_count[best.id] = self.usage_count.get(best.id, 0) + 1
                return {self.REFINE_LABEL: best}, {}
            else:
                # Diverge from a mid-tier program (different basin, not the exhausted best).
                mid = programs[min(len(programs) - 1, len(programs) // 2)]
                self.usage_count[mid.id] = self.usage_count.get(mid.id, 0) + 1
                return {self.DIVERGE_LABEL: mid}, {}

        # Default: score-weighted choice over top half (exploit) with occasional exploration.
        top = programs[: max(1, len(programs) // 2)]
        if self.sample_counter % 5 == 0:
            pool = programs  # exploration round: full population
        else:
            pool = top
        weights = []
        for p in pool:
            s = _score(p)
            w = max(1e-3, s - (self.best_score - 15.0)) ** 2
            w /= 1 + self.usage_count.get(p.id, 0)  # discourage reuse
            weights.append(w)
        parent = self.random_state.choices(pool, weights=weights, k=1)[0]
        self.usage_count[parent.id] = self.usage_count.get(parent.id, 0) + 1

        # Context: best program + diverse picks from different score bands.
        ctx: List[EvolvedProgram] = []
        seen = {parent.id}
        best = programs[0]
        if best.id not in seen:
            ctx.append(best)
            seen.add(best.id)
        bands = [programs, programs[len(programs) // 2:]]
        for band in bands:
            avail = [p for p in band if p.id not in seen]
            if avail:
                pick = self.random_state.choice(avail)
                ctx.append(pick)
                seen.add(pick.id)
        while len(ctx) < n_ctx:
            avail = [p for p in programs if p.id not in seen]
            if not avail:
                break
            pick = self.random_state.choice(avail)
            ctx.append(pick)
            seen.add(pick.id)

        return {"": parent}, {"": ctx[:n_ctx]}


# EVOLVE-BLOCK-END