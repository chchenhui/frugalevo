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
    """Small-population elite search with controlled exploitation and resets."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_iteration = 0
        self.last_added_iteration = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_gains: Dict[str, float] = {}
        self.label_uses: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        current_iteration = (
            iteration if isinstance(iteration, int)
            else getattr(program, "iteration_found", 0)
        )
        if isinstance(current_iteration, int):
            self.last_added_iteration = max(self.last_added_iteration, current_iteration)

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            parent = self.get(program.parent_id)
            parent_score = self._score(parent) if parent is not None else None
            child_score = self._score(program)
            if parent_score is not None and child_score is not None:
                gain = child_score - parent_score
                self.parent_gains[program.parent_id] = max(
                    self.parent_gains.get(program.parent_id, -1.0), gain
                )

        if program.parent_info and program.parent_info[0]:
            label = program.parent_info[0]
            self.label_uses[label] = self.label_uses.get(label, 0) + 1

        score = self._score(program)
        if score is not None:
            meaningful_margin = max(0.01, abs(self.best_seen) * 0.01) if math.isfinite(
                self.best_seen
            ) else 0.01
            if self.best_seen == float("-inf") or score > self.best_seen + meaningful_margin:
                self.best_seen = score
                self.last_meaningful_iteration = self.last_added_iteration
            else:
                self.best_seen = max(self.best_seen, score)

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        scored = [(self._score(program), program) for program in self.programs.values()]
        scored = [(score, program) for score, program in scored if score is not None]
        if not scored:
            raise ValueError("No scored candidates available for sampling")

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score = scored[0][0]
        pool_size = min(len(scored), max(6, min(16, len(scored) // 2)))
        pool = scored[:pool_size]

        # Favor elites which have either been underused or previously produced gains.
        weights: List[float] = []
        for score, program in pool:
            quality = math.exp((score - best_score) * 35.0)
            novelty = 1.0 / (1.0 + 0.45 * self.parent_uses.get(program.id, 0))
            gain = self.parent_gains.get(program.id, 0.0)
            evidence = 1.0 + max(0.0, gain) * 20.0
            weights.append(quality * novelty * evidence)

        parent = self.random_state.choices(
            [program for _, program in pool], weights=weights, k=1
        )[0]

        stalled = self.last_added_iteration - self.last_meaningful_iteration
        prior_targeted = sum(
            1
            for program in self.programs.values()
            if program.parent_info
            and program.parent_info[1] == parent.id
            and program.parent_info[0]
        )

        # The population has plateaued: occasionally request a deliberate new
        # direction, but do not repeatedly target the same solution.
        if stalled >= 8 and prior_targeted == 0 and self.random_state.random() < 0.30:
            return {self.DIVERGE_LABEL: parent}, {}
        if stalled >= 5 and prior_targeted == 0 and self.random_state.random() < 0.18:
            return {self.REFINE_LABEL: parent}, {}

        limit = max(0, num_context_programs or 0)
        available = [program for _, program in scored if program.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Context is deliberately high-quality: near-best alternatives are more
        # useful at this late-stage plateau than weak historical attempts.
        for program in available[:min(len(available), max(limit * 2, 6))]:
            if len(contexts) >= limit:
                break
            if self.random_state.random() < 0.70 or not contexts:
                contexts.append(program)

        remaining = [p for p in available if p.id not in {c.id for c in contexts}]
        self.random_state.shuffle(remaining)
        contexts.extend(remaining[:max(0, limit - len(contexts))])

        return {"": parent}, {"": contexts[:limit]}


# EVOLVE-BLOCK-END