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

    Key ideas:
    - Never pick parents from the failure tier (score ~1.0).
    - Prefer parents from the top score band, weighted toward recent improvers,
      with round-robin usage tracking to avoid overuse.
    - When stagnating, alternate REFINE (on the current best) and DIVERGE labels.
    - Context = diverse high scorers + one mid-tier program for perspective.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = None
        self.stagnation_count = 0
        self.parent_usage: Dict[str, int] = {}
        self.sample_calls = 0

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return None

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s is not None:
            if self.best_score is None:
                self.best_score = s
            elif s > self.best_score + 0.01:
                self.best_score = s
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def _viable(self) -> List[EvolvedProgram]:
        """Programs that didn't collapse to the failure tier."""
        out = []
        for p in self.programs.values():
            s = self._score(p)
            if s is not None and s > 5.0:
                out.append(p)
        return out

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        if len(self.programs) == 0:
            raise ValueError("No candidates available for sampling")

        self.sample_calls += 1
        n_ctx = num_context_programs or 4

        viable = self._viable()
        if not viable:
            viable = list(self.programs.values())

        # Rank viable programs by score.
        scored = sorted(
            [(self._score(p) or 0.0, p) for p in viable],
            key=lambda t: t[0], reverse=True,
        )
        top_band = [p for _, p in scored[: max(4, len(scored) // 3)]]

        # Stagnation: alternate REFINE on best and DIVERGE from a top-band program.
        if self.stagnation_count >= 4 and len(scored) > 0:
            if self.sample_calls % 3 == 0:
                # DIVERGE from a good-but-not-best program.
                pool = [p for _, p in scored[: min(5, len(scored))]]
                parent = self.random_state.choice(pool)
                return {self.DIVERGE_LABEL: parent}, {"": []}
            else:
                # REFINE the current best.
                parent = scored[0][1]
                return {self.REFINE_LABEL: parent}, {"": []}

        # Normal selection: weighted random over top band, penalize overused parents.
        weights = []
        for p in top_band:
            s = self._score(p) or 1.0
            uses = self.parent_usage.get(p.id, 0)
            weights.append(max(0.01, s * (0.5 ** uses)))
        parent = self.random_state.choices(top_band, weights=weights, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Context: diverse top scorers + one mid-tier program.
        others = [p for p in scored if p[0] is not None and p[1].id != parent.id]
        ctx: List[EvolvedProgram] = []
        if others:
            top_ctx = [p for _, p in others[: max(2, n_ctx // 2)]]
            mid_start = len(others) // 2
            mid_ctx = [p for _, p in others[mid_start: mid_start + 2]]
            seen = set()
            for p in top_ctx + mid_ctx:
                if p.id not in seen:
                    ctx.append(p)
                    seen.add(p.id)
        ctx = ctx[:n_ctx]

        parent_dict = {"": parent}
        context_programs_dict = {"": ctx}
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END