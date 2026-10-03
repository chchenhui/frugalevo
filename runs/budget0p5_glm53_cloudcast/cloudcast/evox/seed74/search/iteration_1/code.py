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
    """Adaptive search: rotate parents weighted by score, diversify context,
    and trigger DIVERGE/REFINE when stuck."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.best_program_id: Optional[str] = None
        self.stall_count: int = 0
        self.last_sampled_ids: List[str] = []
        self.diverge_used_on: set = set()

    def _score(self, program: Program) -> float:
        v = program.metrics.get("combined_score") if program.metrics else None
        if isinstance(v, (int, float)) and v is not None:
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score + 1e-12:
            # meaningful improvement tracking
            if s > self.best_score * 1.01 or s > self.best_score + 0.01:
                self.stall_count = 0
            else:
                self.stall_count += 1
            self.best_score = s
            self.best_program_id = program.id
        else:
            self.stall_count += 1

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug(f"Added program {program.id} (score={s}) stall={self.stall_count}")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4
        n_ctx = min(n_ctx, max(len(candidates) - 1, 0))

        # --- Parent selection: least-used among top scorers, with randomization ---
        scored = [(self._score(p), p) for p in candidates]
        scored.sort(key=lambda t: t[0], reverse=True)
        top_tier = [p for s, p in scored if s >= scored[0][0] - 1e-9] or [scored[0][1]]
        # prefer least-used parents to spread exploration
        top_tier.sort(key=lambda p: self.parent_usage.get(p.id, 0))
        parent = self.random_state.choice(top_tier[:max(3, len(top_tier) // 2)])

        # Track usage
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # --- Context: diverse mix — best, worst, and random unseen ---
        others = [p for p in candidates if p.id != parent.id]
        ctx: List[EvolvedProgram] = []
        if others:
            by_score = sorted(others, key=lambda p: self._score(p), reverse=True)
            ctx.append(by_score[0])           # best other
            ctx.append(by_score[-1])          # worst other (contrast / failure info)
            rest = [p for p in by_score if p not in ctx]
            self.random_state.shuffle(rest)
            ctx.extend(rest)
        ctx = ctx[:n_ctx]

        # --- Label logic ---
        parent_dict: Dict[str, EvolvedProgram] = {"": parent}
        # Deep stagnation: try divergence on an underused good program
        if self.stall_count >= 4:
            diverge_pool = [p for p in top_tier if p.id not in self.diverge_used_on]
            if diverge_pool:
                target = self.random_state.choice(diverge_pool[:3])
                self.diverge_used_on.add(target.id)
                parent_dict = {self.DIVERGE_LABEL: target}
                ctx = []  # targeted divergence
            elif self.best_program_id and self.best_program_id in self.programs:
                # divergence exhausted -> refine best
                parent_dict = {self.REFINE_LABEL: self.programs[self.best_program_id]}
                ctx = []
        elif self.stall_count >= 2 and self.best_program_id and self.best_program_id in self.programs:
            # mild stall: refine the best
            parent_dict = {self.REFINE_LABEL: self.programs[self.best_program_id]}

        context_programs_dict: Dict[str, List[EvolvedProgram]] = {"": ctx}
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END