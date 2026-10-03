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
    """Adaptive elite search with lightweight lineage-aware exploration."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))

        # All persistent search-state is updated in add(), so it is rebuilt
        # correctly when a saved database is loaded.
        self.best_numeric_score = float("-inf")
        self.last_meaningful_iteration = -1
        self.max_seen_iteration = -1
        self.parent_trials: Dict[str, int] = {}
        self.parent_wins: Dict[str, int] = {}
        self.context_uses: Dict[str, int] = {}
        self.refine_count = 0
        self.diverge_count = 0
        self.last_label_iteration = -1000

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        """Store a program and update lineage/progress statistics."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        seen_iteration = iteration
        if seen_iteration is None and isinstance(program.iteration_found, int):
            seen_iteration = program.iteration_found
        if isinstance(seen_iteration, int):
            self.max_seen_iteration = max(self.max_seen_iteration, seen_iteration)
            self.last_iteration = max(self.last_iteration, seen_iteration)

        score = self._score(program)
        parent = self.get(program.parent_id) if program.parent_id else None
        parent_score = self._score(parent) if parent is not None else None

        if parent is not None:
            self.parent_trials[parent.id] = self.parent_trials.get(parent.id, 0) + 1
            if score is not None and parent_score is not None:
                improvement = score - parent_score
                meaningful = improvement > max(0.01, abs(parent_score) * 0.01)
                if meaningful:
                    self.parent_wins[parent.id] = self.parent_wins.get(parent.id, 0) + 1

        if isinstance(program.other_context_ids, list):
            for context_id in program.other_context_ids:
                if isinstance(context_id, str):
                    self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        if isinstance(program.parent_info, tuple) and program.parent_info:
            label = program.parent_info[0]
            if label == self.REFINE_LABEL:
                self.refine_count += 1
                if isinstance(seen_iteration, int):
                    self.last_label_iteration = max(self.last_label_iteration, seen_iteration)
            elif label == self.DIVERGE_LABEL:
                self.diverge_count += 1
                if isinstance(seen_iteration, int):
                    self.last_label_iteration = max(self.last_label_iteration, seen_iteration)

        if score is not None:
            if self.best_numeric_score == float("-inf"):
                self.best_numeric_score = score
            elif score > self.best_numeric_score:
                improvement = score - self.best_numeric_score
                if improvement > max(0.01, abs(self.best_numeric_score) * 0.01):
                    if isinstance(seen_iteration, int):
                        self.last_meaningful_iteration = seen_iteration
                self.best_numeric_score = score

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)

        logger.debug("Added program %s to evolved database", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Choose a strong but underused parent and complementary examples."""
        scored = [(program, self._score(program)) for program in self.programs.values()]
        scored = [(program, score) for program, score in scored if score is not None]

        if not scored:
            raise ValueError("No programs with numeric combined_score available for sampling")

        scored.sort(key=lambda item: item[1], reverse=True)
        best_score = scored[0][1]
        current_iteration = max(self.max_seen_iteration, getattr(self, "last_iteration", -1))
        stagnation = current_iteration - self.last_meaningful_iteration

        # Focus on the high-quality basin: this task is already close to the
        # validity/score ceiling, so weak programs are mostly unhelpful.
        elite = [
            program for program, score in scored
            if score >= best_score - 0.008
        ]
        if len(elite) < 3:
            elite = [program for program, _ in scored[:min(6, len(scored))]]

        # Prefer good parents that have not already produced many children.
        weights: List[float] = []
        for program in elite:
            score = self._score(program) or 0.0
            trials = self.parent_trials.get(program.id, 0)
            wins = self.parent_wins.get(program.id, 0)
            quality = 1.0 + 8.0 * max(0.0, score - (best_score - 0.008)) / 0.008
            evidence = 1.0 + (wins / max(1, trials))
            novelty = 1.0 / math.sqrt(1.0 + trials)
            weights.append(quality * evidence * novelty)

        parent = self.random_state.choices(elite, weights=weights, k=1)[0]

        # A short, deliberate REFINE request is useful after a real plateau.
        # It is rate-limited and targeted at elite programs, not applied blindly.
        if (
            stagnation >= 6
            and self.refine_count < 2
            and current_iteration - self.last_label_iteration >= 3
        ):
            label = self.REFINE_LABEL
            return {label: parent}, {}

        requested = max(0, int(num_context_programs or 0))
        pool = [program for program, _ in scored if program.id != parent.id]

        # Context is drawn from nearby strong alternatives, with a small amount
        # of score-band diversity and a penalty for repeatedly shown examples.
        strong_pool = [
            program for program in pool
            if (self._score(program) or float("-inf")) >= best_score - 0.02
        ]
        if len(strong_pool) >= requested:
            pool = strong_pool

        contexts: List[EvolvedProgram] = []
        used_solutions = {parent.solution}
        while pool and len(contexts) < requested:
            context_weights = []
            for program in pool:
                score = self._score(program) or 0.0
                reuse_penalty = 1.0 / (1.0 + self.context_uses.get(program.id, 0))
                different_solution = 1.4 if program.solution not in used_solutions else 0.5
                score_weight = max(0.2, 1.0 + 20.0 * (score - (best_score - 0.02)))
                context_weights.append(score_weight * reuse_penalty * different_solution)

            chosen = self.random_state.choices(pool, weights=context_weights, k=1)[0]
            contexts.append(chosen)
            used_solutions.add(chosen.solution)
            pool = [program for program in pool if program.id != chosen.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END