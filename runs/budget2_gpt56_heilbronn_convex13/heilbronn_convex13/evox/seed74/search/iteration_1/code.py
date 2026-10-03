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
    """Adaptive score-aware search with parent and context diversity."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.initial_program = None
        self.best_score_seen = None
        self.last_meaningful_improvement = 0
        self.add_count = 0
        self.parent_use_count: Dict[str, int] = {}
        self.context_use_count: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        """Store programs while reconstructing durable search-progress signals."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        self.add_count += 1

        current_iteration = (
            iteration if isinstance(iteration, int) else program.iteration_found
        )
        if isinstance(current_iteration, int):
            self.last_iteration = max(self.last_iteration, current_iteration)

        if program.parent_id:
            self.parent_use_count[program.parent_id] = (
                self.parent_use_count.get(program.parent_id, 0) + 1
            )
        for context_id in program.other_context_ids or []:
            self.context_use_count[context_id] = (
                self.context_use_count.get(context_id, 0) + 1
            )

        score = self._score(program)
        if score is not None:
            if self.best_score_seen is None:
                self.best_score_seen = score
                self.last_meaningful_improvement = self.add_count
            else:
                improvement = score - self.best_score_seen
                relative = improvement / max(abs(self.best_score_seen), 1e-9)
                if improvement > 0.01 or relative > 0.01:
                    self.best_score_seen = score
                    self.last_meaningful_improvement = self.add_count
                elif score > self.best_score_seen:
                    self.best_score_seen = score

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        logger.debug("Added program %s", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(p, self._score(p)) for p in candidates]
        numeric = [(p, s) for p, s in scored if s is not None]
        context_count = max(0, num_context_programs or 0)

        if not numeric:
            parent = self.random_state.choice(candidates)
            pool = [p for p in candidates if p.id != parent.id]
            self.random_state.shuffle(pool)
            return {"": parent}, {"": pool[:context_count]}

        numeric.sort(key=lambda item: item[1], reverse=True)
        best = numeric[0][1]

        # Keep exploitation focused on the competitive tier, but permit a
        # modest chance for a genuinely different lower-ranked branch.
        tolerance = max(0.03, abs(best) * 0.08)
        elite = [p for p, s in numeric if s >= best - tolerance]
        if not elite:
            elite = [numeric[0][0]]

        stalled = self.add_count - self.last_meaningful_improvement
        explore_branch = stalled >= 8 and len(numeric) > len(elite)
        if explore_branch and self.random_state.random() < 0.22:
            lower = [p for p, s in numeric if p.id not in {x.id for x in elite}]
            parent_pool = lower or elite
        else:
            parent_pool = elite

        # Equal-scoring plateaus are common: favor under-used parents so that
        # multiple independent mutation paths get a chance.
        weights = [
            1.0 / (1.0 + self.parent_use_count.get(p.id, 0))
            for p in parent_pool
        ]
        parent = self.random_state.choices(parent_pool, weights=weights, k=1)[0]

        label = ""
        # A targeted divergence is reserved for a sustained plateau, rather
        # than being the normal sampling mode.
        if stalled >= 10 and self.random_state.random() < 0.25:
            label = self.DIVERGE_LABEL
            return {label: parent}, {}

        pool = [p for p in candidates if p.id != parent.id]
        selected: List[EvolvedProgram] = []
        used_ids = set()

        # First give the model a strong alternative, rather than repeatedly
        # showing the exact same top-scoring solution family.
        alternatives = [p for p in elite if p.id != parent.id]
        alternatives.sort(key=lambda p: self.context_use_count.get(p.id, 0))
        if alternatives and context_count:
            selected.append(alternatives[0])
            used_ids.add(alternatives[0].id)

        # Fill remaining context with score-diverse, lightly reused examples.
        while len(selected) < context_count:
            available = [p for p in pool if p.id not in used_ids]
            if not available:
                break
            weights = []
            for p in available:
                score = self._score(p)
                score_bonus = 1.0 if score is None else 1.0 + max(0.0, score) / max(abs(best), 1e-9)
                reuse_penalty = 1.0 + self.context_use_count.get(p.id, 0)
                weights.append(score_bonus / reuse_penalty)
            chosen = self.random_state.choices(available, weights=weights, k=1)[0]
            selected.append(chosen)
            used_ids.add(chosen.id)

        return {"": parent}, {"": selected}


# EVOLVE-BLOCK-END