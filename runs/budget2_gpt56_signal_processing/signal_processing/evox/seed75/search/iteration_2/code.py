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
    """Elite search that favors productive lineages while preserving diversity."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_iteration = 0
        self.last_added_iteration = 0
        self.parent_uses: Dict[str, int] = {}
        self.context_uses: Dict[str, int] = {}
        self.parent_gain_sum: Dict[str, float] = {}
        self.parent_gain_count: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        current_iteration = iteration
        if current_iteration is None:
            current_iteration = getattr(program, "iteration_found", 0)
        if isinstance(current_iteration, int):
            self.last_added_iteration = max(self.last_added_iteration, current_iteration)

        self.programs[program.id] = program

        if program.parent_id:
            self.parent_uses[program.parent_id] = (
                self.parent_uses.get(program.parent_id, 0) + 1
            )
            parent = self.get(program.parent_id)
            child_score = self._score(program)
            parent_score = self._score(parent) if parent is not None else None
            if child_score is not None and parent_score is not None:
                gain = child_score - parent_score
                self.parent_gain_sum[program.parent_id] = (
                    self.parent_gain_sum.get(program.parent_id, 0.0) + gain
                )
                self.parent_gain_count[program.parent_id] = (
                    self.parent_gain_count.get(program.parent_id, 0) + 1
                )

        for context_id in program.other_context_ids or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        score = self._score(program)
        if score is not None:
            meaningful_gap = max(0.01, abs(self.best_seen) * 0.01) if math.isfinite(
                self.best_seen
            ) else 0.01
            if not math.isfinite(self.best_seen) or score > self.best_seen + meaningful_gap:
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
        elite_count = min(len(scored), max(5, int(math.ceil(len(scored) * 0.35))))
        elite = scored[:elite_count]

        # Favor high-quality parents, especially lineages that have previously
        # generated gains, but decay selection probability after repeated use.
        weights: List[float] = []
        for score, program in elite:
            quality = math.exp((score - best_score) * 18.0)
            uses = self.parent_uses.get(program.id, 0)
            mean_gain = self.parent_gain_sum.get(program.id, 0.0) / max(
                1, self.parent_gain_count.get(program.id, 0)
            )
            productivity = 1.0 + max(-0.4, min(1.0, mean_gain * 20.0))
            novelty = 1.0 / (1.0 + 0.55 * uses)
            weights.append(max(0.01, quality * productivity * novelty))

        parent = self.random_state.choices(
            [program for _, program in elite], weights=weights, k=1
        )[0]

        stalled = self.last_added_iteration - self.last_meaningful_iteration
        prior_targeted = sum(
            1
            for program in self.programs.values()
            if program.parent_info
            and program.parent_info[1] == parent.id
            and program.parent_info[0]
        )

        # Reserve explicit instructions for a sustained plateau. Refinement is
        # more appropriate near a strong incumbent; divergence is a later escape.
        if stalled >= 8 and prior_targeted == 0:
            if self.random_state.random() < 0.30:
                return {self.REFINE_LABEL: parent}, {}
        if stalled >= 11 and prior_targeted < 2:
            if self.random_state.random() < 0.25:
                return {self.DIVERGE_LABEL: parent}, {}

        limit = max(0, num_context_programs or 0)
        if limit == 0:
            return {"": parent}, {"": []}

        available = [program for _, program in scored if program.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Include strong alternatives, then a useful contrasting score tier.
        top_band = available[:max(1, min(len(available), elite_count))]
        while top_band and len(contexts) < min(2, limit):
            least_used = min(self.context_uses.get(p.id, 0) for p in top_band)
            choices = [p for p in top_band if self.context_uses.get(p.id, 0) == least_used]
            chosen = self.random_state.choice(choices)
            contexts.append(chosen)
            top_band = [p for p in top_band if p.id != chosen.id]

        middle_start = max(1, len(available) // 3)
        middle_end = max(middle_start + 1, 2 * len(available) // 3)
        contrast_pool = [
            p for p in available[middle_start:middle_end]
            if p.id not in {c.id for c in contexts}
        ]
        if contrast_pool and len(contexts) < limit:
            least_used = min(self.context_uses.get(p.id, 0) for p in contrast_pool)
            choices = [p for p in contrast_pool if self.context_uses.get(p.id, 0) == least_used]
            contexts.append(self.random_state.choice(choices))

        remaining = [p for p in available if p.id not in {c.id for c in contexts}]
        self.random_state.shuffle(remaining)
        contexts.extend(remaining[: max(0, limit - len(contexts))])

        return {"": parent}, {"": contexts[:limit]}


# EVOLVE-BLOCK-END