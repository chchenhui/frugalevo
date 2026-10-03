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
    """Adaptive elite/diversity search for compact geometric constructions."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = None
        self.meaningful_best = None
        self.stagnation = 0
        self.parent_uses: Dict[str, int] = {}
        self.context_uses: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}
        self.initial_program = None

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

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
        for context_id in program.other_context_ids or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1
        if program.parent_info and program.parent_info[0]:
            label = program.parent_info[0]
            self.label_uses[label] = self.label_uses.get(label, 0) + 1

        score = self._score(program)
        if score is not None:
            if self.best_seen is None or score > self.best_seen:
                self.best_seen = score

            if self.meaningful_best is None:
                self.meaningful_best = score
            else:
                improvement = score - self.meaningful_best
                threshold = min(0.01, abs(self.meaningful_best) * 0.01)
                if improvement > threshold:
                    self.meaningful_best = score
                    self.stagnation = 0
                else:
                    self.stagnation += 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(self._score(p), p) for p in candidates]
        scored.sort(key=lambda item: item[0] if item[0] is not None else float("-inf"),
                    reverse=True)

        # Most mutations exploit the strong plateau, but parent reuse is penalized.
        elite_count = min(len(scored), max(6, len(scored) // 3))
        elite = [p for _, p in scored[:elite_count]]
        broad_count = min(len(scored), max(elite_count, (2 * len(scored)) // 3))
        broad = [p for _, p in scored[:broad_count]]

        pool = elite
        if self.stagnation >= 8 and self.random_state.random() < 0.35:
            pool = broad

        weights = []
        for rank, parent in enumerate(pool):
            quality = len(pool) - rank
            reuse_penalty = 1.0 + self.parent_uses.get(parent.id, 0)
            weights.append(quality / reuse_penalty)
        parent = self.random_state.choices(pool, weights=weights, k=1)[0]

        label = ""
        # A long plateau warrants occasional deliberate changes of direction,
        # without making labels the default search mechanism.
        if self.stagnation >= 12 and self.random_state.random() < 0.25:
            if self.random_state.random() < 0.65:
                label = self.DIVERGE_LABEL
                parent = self.random.choice(broad)
            else:
                label = self.REFINE_LABEL
                parent = self.random.choice(elite)

            return {label: parent}, {}

        count = max(0, num_context_programs or 0)
        contexts: List[EvolvedProgram] = []
        seen_ids = {parent.id}
        seen_solutions = {parent.solution}

        # Mix strong examples with a broader, less-reused perspective.
        context_pool = elite + broad + candidates
        self.random_state.shuffle(context_pool)
        context_pool.sort(
            key=lambda p: (
                self.context_uses.get(p.id, 0),
                -((self._score(p) or float("-inf")))
            )
        )

        for program in context_pool:
            if len(contexts) >= count:
                break
            if program.id in seen_ids or program.solution in seen_solutions:
                continue
            contexts.append(program)
            seen_ids.add(program.id)
            seen_solutions.add(program.solution)

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END