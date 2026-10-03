# EVOLVE-BLOCK-START
import logging
import math
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
    """Adaptive elite search with occasional directed diversification."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.initial_program = None
        self.best_seen = -float("inf")
        self.stagnation = 0
        self.parent_uses: Dict[str, int] = {}
        self.score_by_id: Dict[str, float] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        score = self._score(program)
        if score is not None:
            self.score_by_id[program.id] = score
            improvement = score - self.best_seen
            relative = improvement / max(abs(self.best_seen), 1e-9)
            if improvement > 0.01 or (improvement > 0 and relative > 0.01):
                self.best_seen = score
                self.stagnation = 0
            else:
                self.best_seen = max(self.best_seen, score)
                self.stagnation += 1

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1

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
        valid = [(s, p) for s, p in scored if s is not None]
        if not valid:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        valid.sort(key=lambda item: item[0], reverse=True)
        best = valid[0][0]
        elite = [p for s, p in valid if s >= best - max(0.002, abs(best) * 0.003)]
        promising = [p for s, p in valid if s >= best - max(0.07, abs(best) * 0.08)]

        # Reward strong candidates, but avoid repeatedly asking for the same edit.
        parent_pool = promising if self.stagnation < 8 else valid[:max(6, len(valid) // 2)]
        weights = []
        for score, program in [(self._score(p), p) for p in parent_pool]:
            quality = max(0.05, (score or 0.0) / max(abs(best), 1e-9))
            novelty = 1.0 / (1.0 + self.parent_uses.get(program.id, 0))
            weights.append(quality + 0.8 * novelty)
        parent = self.random_state.choices(parent_pool, weights=weights, k=1)[0]

        # On a genuine plateau, occasionally explicitly request a new direction
        # or careful polishing; otherwise labels remain empty.
        if self.stagnation >= 10 and self.random_state.random() < 0.45:
            if self.random_state.random() < 0.55:
                alternatives = [p for p in promising if p.id != parent.id] or promising
                parent = self.random_state.choice(alternatives)
                return {self.DIVERGE_LABEL: parent}, {}
            parent = self.random_state.choice(elite)
            return {self.REFINE_LABEL: parent}, {}

        count = max(0, num_context_programs or 0)
        pool = [p for _, p in valid if p.id != parent.id]
        self.random_state.shuffle(pool)

        # Give the model both excellent examples and non-identical approaches.
        contexts: List[EvolvedProgram] = []
        for group in (elite, promising, pool):
            choices = [p for p in group if p.id != parent.id and p.id not in {x.id for x in contexts}]
            if choices and len(contexts) < count:
                contexts.append(self.random_state.choice(choices))
        for program in pool:
            if len(contexts) >= count:
                break
            if program.id not in {x.id for x in contexts}:
                contexts.append(program)

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END