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
    """Adaptive elite search with parent/context reuse balancing."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.initial_program = None
        self.best_seen = float("-inf")
        self.stagnation = 0
        self.parent_uses: Dict[str, int] = {}
        self.context_uses: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}
        self.last_iteration = getattr(self, "last_iteration", -1)

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
        for context_id in program.other_context_ids or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        if program.parent_info and program.parent_info[0]:
            label = program.parent_info[0]
            self.label_uses[label] = self.label_uses.get(label, 0) + 1

        score = self._score(program)
        if score is not None:
            if self.best_seen == float("-inf"):
                self.best_seen = score
            else:
                improvement = score - self.best_seen
                meaningful = improvement > max(0.01, abs(self.best_seen) * 0.01)
                if meaningful:
                    self.best_seen = score
                    self.stagnation = 0
                else:
                    self.best_seen = max(self.best_seen, score)
                    self.stagnation += 1

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)

        logger.debug("Added program %s", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        scored = [(p, self._score(p)) for p in self.programs.values()]
        scored = [(p, s) for p, s in scored if s is not None]
        if not scored:
            raise ValueError("No scored candidates available for sampling")

        scored.sort(key=lambda item: item[1], reverse=True)
        best = scored[0][1]

        # At this late, near-optimal stage, mutate proven solutions, but spread
        # attention across tied elites rather than repeatedly selecting one.
        elite = [p for p, s in scored if s >= best - 0.003]
        strong = [p for p, s in scored if s >= best - 0.015]
        parent_pool = elite if self.random_state.random() < 0.82 else strong
        if not parent_pool:
            parent_pool = [p for p, _ in scored]

        weights = [
            1.0 / (1.0 + self.parent_uses.get(p.id, 0))
            for p in parent_pool
        ]
        parent = self.random_state.choices(parent_pool, weights=weights, k=1)[0]

        context_count = max(0, num_context_programs or 0)
        available = [p for p, _ in scored if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Context is mostly alternative successful constructions, with one
        # slightly weaker example when available to preserve approach diversity.
        elite_context = [p for p in elite if p.id != parent.id]
        strong_context = [p for p in strong if p.id != parent.id]

        def pick_from(pool: List[EvolvedProgram]) -> Optional[EvolvedProgram]:
            pool = [p for p in pool if p.id not in {x.id for x in contexts}]
            if not pool:
                return None
            weights = [1.0 / (1.0 + self.context_uses.get(p.id, 0)) for p in pool]
            return self.random_state.choices(pool, weights=weights, k=1)[0]

        while len(contexts) < context_count and elite_context:
            choice = pick_from(elite_context)
            if choice is None:
                break
            contexts.append(choice)

        while len(contexts) < context_count:
            choice = pick_from(strong_context) or pick_from(available)
            if choice is None:
                break
            contexts.append(choice)

        label = ""
        # Persistent plateau warrants occasional explicitly different attempts,
        # but labels remain exceptional rather than the normal control signal.
        if self.stagnation >= 12 and self.random_state.random() < 0.20:
            label = self.DIVERGE_LABEL if self.random_state.random() < 0.65 else self.REFINE_LABEL
            if label == self.DIVERGE_LABEL:
                contexts = []

        return {label: parent}, {"": contexts}


# EVOLVE-BLOCK-END