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
    """Search strategy tuned for a converged, stagnating population.

    Key ideas:
    1. Score-weighted parent selection with a usage penalty, so no single
       top program gets over-mutated (REFINE on the same best never helped).
    2. On stagnation, use DIVERGE_LABEL on a mid/low-tier parent with
       diverse context (best + worst + random) — this previously produced
       the two 0.1298 breakthroughs.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -1e18
        self.stagnation_count: int = 0
        self.diverge_used_on: List[str] = []

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if program.parent_id in self.parent_usage:
            self.parent_usage[program.parent_id] += 1
        else:
            self.parent_usage[program.parent_id] = 1

        score = self._score(program)
        # Only count meaningful improvements (>0.01 absolute).
        if score > self.best_score + 0.01:
            self.best_score = score
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        programs = list(self.programs.values())
        if not programs:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4
        programs.sort(key=self._score, reverse=True)
        best = programs[0]

        # --- Parent selection: score-weighted with usage penalty ---
        weights = []
        for p in programs:
            s = self._score(p)
            usage = self.parent_usage.get(p.id, 0)
            weights.append(max(s, 0.001) / (1.0 + 2.0 * usage))

        label = ""
        parent = None

        # On stagnation, DIVERGE from a mid/low-tier, underused parent.
        if self.stagnation_count >= 3:
            lower = [p for p in programs if p.id not in self.diverge_used_on]
            if not lower:
                lower = programs[len(programs) // 2:]
            if lower:
                parent = self.random_state.choice(lower[len(lower) // 3:] or lower)
                label = self.DIVERGE_LABEL
                self.diverge_used_on.append(parent.id)
                self.stagnation_count = 0
                return {label: parent}, {}

        if parent is None:
            # Weighted choice, avoiding heavy reuse of the single best.
            chosen = self.random_state.choices(programs, weights=weights, k=1)[0]
            if chosen.id == best.id and len(programs) > 1 and self.random_state.random() < 0.6:
                chosen = self.random_state.choice(programs[1:3] or programs[1:])
            parent = chosen

        # --- Context: best program + worst + random diverse others ---
        context: List[EvolvedProgram] = []
        if parent.id != best.id:
            context.append(best)
        worst = programs[-1]
        if worst.id != parent.id and all(c.id != worst.id for c in context):
            context.append(worst)
        rest = [p for p in programs if p.id != parent.id and all(c.id != p.id for c in context)]
        self.random_state.shuffle(rest)
        context.extend(rest[: max(0, n_ctx - len(context))])
        context = context[:n_ctx]

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END