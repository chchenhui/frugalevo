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
    """Adaptive search strategy.

    Simple ideas:
    1. Track how often each program is used as parent/context; rotate usage to
       avoid over-exploiting one program.
    2. When no meaningful improvement happens for several iterations, switch to
       DIVERGE (new direction) or REFINE (polish best) labels.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_use_count: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.iters_since_improvement: int = 0
        self.last_parent_id: Optional[str] = None
        self.sample_calls: int = 0

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        s = self._score(program)
        # Meaningful improvement threshold: 1% relative or 0.01 absolute.
        if s > self.best_score + 0.01 or (self.best_score > 0 and s > self.best_score * 1.01):
            self.best_score = s
            self.iters_since_improvement = 0
        else:
            self.iters_since_improvement += 1
            if s > self.best_score:
                self.best_score = s

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

        # Weighted parent choice: prefer higher scores but penalize heavy reuse.
        scored = []
        for p in candidates:
            s = self._score(p)
            uses = self.parent_use_count.get(p.id, 0)
            w = (s + 0.001) / (1.0 + uses * uses)
            if p.id == self.last_parent_id:
                w *= 0.3
            scored.append((w, p))
        total = sum(w for w, _ in scored)
        if total <= 0:
            parent = self.random_state.choice(candidates)
        else:
            r = self.random_state.uniform(0, total)
            acc = 0.0
            parent = scored[-1][1]
            for w, p in scored:
                acc += w
                if acc >= r:
                    parent = p
                    break

        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1
        self.last_parent_id = parent.id

        # Context: diverse pool — mix of top scorers and low/mid scorers,
        # excluding the parent, avoiding repeats.
        others = [p for p in candidates if p.id != parent.id]
        self.random_state.shuffle(others)
        by_score = sorted(others, key=self._score, reverse=True)
        ctx: List[EvolvedProgram] = []
        if by_score:
            ctx.append(by_score[0])  # best other
        if len(by_score) > 1:
            ctx.append(by_score[-1])  # worst other (failure insight)
        for p in others:
            if p not in ctx:
                ctx.append(p)
            if len(ctx) >= n_ctx:
                break
        ctx = ctx[:n_ctx]

        # Label logic based on stagnation.
        label = ""
        if self.iters_since_improvement >= 4:
            # Alternate between diverging and refining to escape plateau.
            if self.sample_calls % 2 == 0:
                label = self.DIVERGE_LABEL
                ctx = []  # targeted divergence
            else:
                label = self.REFINE_LABEL
                best = max(candidates, key=self._score)
                parent = best
                ctx = []
                self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1
                self.last_parent_id = parent.id

        return {label: parent}, {"": ctx}


# EVOLVE-BLOCK-END