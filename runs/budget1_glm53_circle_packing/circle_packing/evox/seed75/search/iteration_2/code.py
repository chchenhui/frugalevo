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
    """Refinement-focused search near the top of the score distribution.

    Population is clustered just below the target (0.9916-0.9928), so the
    strategy concentrates on refining top-tier programs (which already
    produced the last breakthrough) while rotating parents to avoid overuse,
    with occasional divergence attempts when stuck.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_use_count: Dict[str, int] = {}
        self.sample_calls = 0
        self.recent_best_scores: List[float] = []

    def _score(self, program: EvolvedProgram) -> float:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None and hasattr(self, "last_iteration"):
            self.last_iteration = max(self.last_iteration, iteration)
        # Track usage of the parent that generated this program.
        if program.parent_id and program.parent_id in self.programs:
            self.parent_use_count[program.parent_id] = (
                self.parent_use_count.get(program.parent_id, 0) + 1
            )
        self.recent_best_scores.append(self._score(program))
        if len(self.recent_best_scores) > 12:
            self.recent_best_scores.pop(0)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        self.sample_calls += 1
        scored = sorted(
            candidates, key=lambda p: self._score(p), reverse=True
        )
        top_tier = [p for p in scored if self._score(p) >= 0.98] or scored[:3]

        # Rotate among top-tier parents, preferring least-used.
        top_tier.sort(key=lambda p: self.parent_use_count.get(p.id, 0))
        # Small random tie-break window among the least-used.
        window = top_tier[: max(1, min(3, len(top_tier)))]
        parent = self.random_state.choice(window)
        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1

        # Stagnation check: no meaningful gain in last ~8 adds.
        stagnant = False
        if len(self.recent_best_scores) >= 8:
            best_recent = max(self.recent_best_scores)
            older = self.recent_best_scores[: len(self.recent_best_scores) - 8]
            baseline = max(older) if older else best_recent
            stagnant = (best_recent - baseline) < 0.01

        label = ""
        context: List[EvolvedProgram] = []

        if stagnant and self.sample_calls % 4 == 0:
            # Occasionally try divergence on a lesser-used top parent.
            label = self.DIVERGE_LABEL
            parent = top_tier[0] if top_tier else parent
        else:
            # Refinement is the workhorse: it produced the 0.9928 breakthrough
            # when applied with empty context to a 0.9916 parent.
            if self.random_state.random() < 0.6:
                label = self.REFINE_LABEL
                context = []  # targeted refinement, no context
            else:
                # Diverse context: top-tier siblings + a mid/low-tier program
                # for contrast, avoiding heavy reuse.
                others = [p for p in scored if p.id != parent.id]
                high = [p for p in others if self._score(p) >= 0.9][:2]
                rest = [p for p in others if self._score(p) < 0.9]
                self.random_state.shuffle(rest)
                context = (high + rest[:2])[: num_context_programs or 4]

        parent_dict = {label: parent}
        return parent_dict, {"": context}


# EVOLVE-BLOCK-START