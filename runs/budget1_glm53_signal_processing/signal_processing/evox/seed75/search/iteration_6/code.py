# EVOLVE-BLOCK-START
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive search: usage-penalized parent selection from the strong tier,
    paired with deliberately diverse context (best / median / worst / random).

    Labels are avoided by default (they historically regressed); only when
    deeply stagnated do we occasionally REFINE the best program.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}   # program_id -> times used as parent
        self.best_score: float = -1.0
        self.stagnation: int = 0                 # adds since meaningful improvement
        self.sample_calls: int = 0

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
        # meaningful improvement threshold: 1% relative or 0.01 absolute
        if s > self.best_score and (s - self.best_score) > max(0.01, 0.01 * abs(self.best_score)):
            self.best_score = s
            self.stagnation = 0
        elif s > self.best_score:
            self.best_score = s
            self.stagnation += 1
        else:
            self.stagnation += 1

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def _pick_parent(self, sorted_progs: List[EvolvedProgram]) -> EvolvedProgram:
        """Tournament among top-half programs, penalized by past usage."""
        top = sorted_progs[: max(2, len(sorted_programs := sorted_progs) // 2)]
        def weight(p):
            return 1.0 / (1 + 3 * self.parent_usage.get(p.id, 0))
        weights = [weight(p) for p in top]
        return self.random_state.choices(top, weights=weights, k=1)[0]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = sorted(candidates, key=self._score, reverse=True)
        self.sample_calls += 1

        # Rare REFINE on the best program when deeply stagnated (labels usually
        # regress, so keep this to ~1 in 6 calls at most, and never back-to-back).
        if self.stagnation >= 12 and self.sample_calls % 6 == 0:
            parent = scored[0]
            self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
            return {self.REFINE_LABEL: parent}, {"": []}

        parent = self._pick_parent(scored)
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Diverse context: best, median, a weak/failure program, plus random fill.
        ctx: List[EvolvedProgram] = []
        seen = {parent.id}
        for pick in (scored[0], scored[len(scored) // 2], scored[-1]):
            if pick.id not in seen:
                ctx.append(pick)
                seen.add(pick.id)
        pool = [p for p in scored if p.id not in seen]
        if pool:
            self.random_state.shuffle(pool)
            ctx.extend(pool[: max(0, (num_context_programs or 4) - len(ctx))])

        return {"": parent}, {"": ctx[: num_context_programs or 4]}


# EVOLVE-BLOCK-END