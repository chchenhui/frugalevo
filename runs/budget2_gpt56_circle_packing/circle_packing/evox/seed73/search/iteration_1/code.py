# EVOLVE-BLOCK-START
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive elite search with parent-use balancing and varied context."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score_seen = float("-inf")
        self.last_meaningful_iteration = 0
        self.parent_uses: Dict[str, int] = {}
        self.context_uses: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _refresh_state(self) -> None:
        self.parent_uses = {}
        self.context_uses = {}
        ordered = sorted(
            self.programs.values(),
            key=lambda p: (getattr(p, "iteration_found", 0), getattr(p, "timestamp", 0)),
        )

        best = float("-inf")
        last_gain = 0
        for program in ordered:
            if program.parent_id:
                self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            for context_id in program.other_context_ids or []:
                self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

            score = self._score(program)
            if score is not None and score > best:
                if best != float("-inf") and (
                    score - best > 0.01 or score > best * 1.01
                ):
                    last_gain = getattr(program, "iteration_found", 0)
                best = score

        self.best_score_seen = best
        self.last_meaningful_iteration = last_gain

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any) -> str:
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        self._refresh_state()
        logger.debug("Added program %s", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(p, self._score(p)) for p in candidates]
        valid = [(p, s) for p, s in scored if s is not None]
        if not valid:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        valid.sort(key=lambda item: item[1], reverse=True)
        best = valid[0][1]

        # Keep selection among near-best solutions, but favor branches that
        # have received fewer attempts.
        elite = [
            p for p, score in valid
            if score >= best - max(0.003, abs(best) * 0.003)
        ]
        weights = [
            1.0 / (1.0 + self.parent_uses.get(p.id, 0))
            for p in elite
        ]
        parent = self.random_state.choices(elite, weights=weights, k=1)[0]

        count = max(0, num_context_programs or 0)
        context: List[EvolvedProgram] = []
        seen_solutions = {parent.solution}
        pool = [p for p, _ in valid if p.id != parent.id]

        # Context is deliberately drawn from several strong but distinct
        # programs, rather than repeatedly showing one dominant solution.
        while pool and len(context) < count:
            choices = [p for p in pool if p.solution not in seen_solutions]
            if not choices:
                break
            weights = [
                (1.0 + max(0.0, self._score(p) or 0.0))
                / (1.0 + self.context_uses.get(p.id, 0))
                for p in choices
            ]
            chosen = self.random_state.choices(choices, weights=weights, k=1)[0]
            context.append(chosen)
            seen_solutions.add(chosen.solution)
            pool = [p for p in pool if p.id != chosen.id]

        current_iteration = max(
            [getattr(p, "iteration_found", 0) for p in candidates] + [0]
        )
        stalled = current_iteration - self.last_meaningful_iteration >= 8

        # A long plateau near the same score suggests changing construction
        # ideas instead of repeatedly making tiny edits to the current best.
        if stalled and len(elite) >= 3:
            exploratory = min(
                elite,
                key=lambda p: self.parent_uses.get(p.id, 0),
            )
            return {self.DIVERGE_LABEL: exploratory}, {"" : []}

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END