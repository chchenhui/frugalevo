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
    """Adaptive elite search with parent-use balancing and mixed-score context."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_uses: Dict[str, int] = {}
        self.parent_gains: Dict[str, float] = {}
        self.best_score_seen = float("-inf")
        self.last_progress_iteration = 0

    def _score(self, program: EvolvedProgram) -> float:
        value = program.metrics.get("combined_score")
        return float(value) if isinstance(value, (int, float)) else float("-inf")

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        score = self._score(program)
        prior_best = self.best_score_seen

        # Reconstruct useful state during checkpoint/database restore as well.
        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            parent = self.get(program.parent_id)
            if parent is not None:
                parent_score = self._score(parent)
                if score != float("-inf") and parent_score != float("-inf"):
                    gain = score - parent_score
                    if gain > 0:
                        self.parent_gains[program.parent_id] = (
                            self.parent_gains.get(program.parent_id, 0.0) + gain
                        )

        # A progress event must clear the plateau only for a material gain.
        if score != float("-inf") and prior_best != float("-inf"):
            gain = score - prior_best
            relative_gain = gain / max(abs(prior_best), 1e-12)
            if gain > 0 and (gain > 0.01 or relative_gain > 0.01):
                self.last_progress_iteration = (
                    iteration if iteration is not None else program.iteration_found
                )
        elif score != float("-inf"):
            self.last_progress_iteration = iteration if iteration is not None else program.iteration_found

        if score > self.best_score_seen:
            self.best_score_seen = score

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug("Added program %s to the evolve database", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        ranked = sorted(candidates, key=self._score, reverse=True)
        context_count = max(0, num_context_programs or 0)

        # Search mostly among the strong solutions, but do not repeatedly mutate
        # one tied-best program.  Parent credit rewards lineages that produced gains.
        elite_size = min(len(ranked), max(8, len(ranked) // 3))
        parent_pool = ranked[:elite_size]

        weights: List[float] = []
        for rank, program in enumerate(parent_pool):
            quality = 1.0 + (elite_size - rank) / max(elite_size, 1)
            novelty = 1.0 / (1.0 + self.parent_uses.get(program.id, 0))
            evidence = 1.0 + min(3.0, self.parent_gains.get(program.id, 0.0) * 100.0)
            weights.append(quality * (0.45 + novelty) * evidence)

        parent = self.random_state.choices(parent_pool, weights=weights, k=1)[0]

        # Context deliberately spans nearby elite alternatives and lower-ranked
        # approaches, rather than repeatedly showing only the same best solution.
        remaining = [p for p in ranked if p.id != parent.id]
        contexts: List[EvolvedProgram] = []
        used_ids = set()

        if remaining and context_count:
            bands = [
                remaining[:min(10, len(remaining))],
                remaining[min(10, len(remaining)):min(25, len(remaining))],
                remaining[min(25, len(remaining)):min(45, len(remaining))],
                remaining[min(45, len(remaining)):],
            ]

            for band in bands:
                if len(contexts) >= context_count:
                    break
                choices = [p for p in band if p.id not in used_ids]
                if choices:
                    chosen = self.random_state.choice(choices)
                    contexts.append(chosen)
                    used_ids.add(chosen.id)

            leftovers = [p for p in remaining if p.id not in used_ids]
            self.random_state.shuffle(leftovers)
            contexts.extend(leftovers[: max(0, context_count - len(contexts))])

        return {"": parent}, {"": contexts[:context_count]}


# EVOLVE-BLOCK-END