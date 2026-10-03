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
    """Adaptive elite search with occasional focused refinement."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score_seen = float("-inf")
        self.best_program_id: Optional[str] = None
        self.last_meaningful_improvement = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_gain: Dict[str, float] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        """Store programs and update persistent progress/lineage statistics."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        current_iteration = (
            iteration if iteration is not None else getattr(program, "iteration_found", 0)
        )

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        # Record how often a parent has actually been used.  This is updated
        # here rather than sample(), so it survives checkpoint restoration.
        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            parent = self.get(program.parent_id)
            child_score = self._score(program)
            parent_score = self._score(parent) if parent is not None else None
            if child_score is not None and parent_score is not None:
                gain = child_score - parent_score
                self.parent_gain[program.parent_id] = (
                    self.parent_gain.get(program.parent_id, 0.0) + gain
                )

        score = self._score(program)
        if score is not None:
            previous_best = self.best_score_seen
            if score > self.best_score_seen:
                self.best_score_seen = score
                self.best_program_id = program.id

                # A tie or tiny numerical gain is not evidence that the search
                # direction changed meaningfully.
                threshold = max(0.01, abs(previous_best) * 0.01) if previous_best != float("-inf") else 0.0
                if previous_best == float("-inf") or score - previous_best > threshold:
                    self.last_meaningful_improvement = current_iteration

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)

        logger.debug("Added program %s to the evolve database", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Choose a strong but underused parent and complementary elite context."""
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(self._score(p), p) for p in candidates]
        numeric = [(s, p) for s, p in scored if s is not None]
        numeric.sort(key=lambda item: item[0], reverse=True)

        # Invalid/error candidates are rarely useful as mutation parents.
        ranked = [p for _, p in numeric] if numeric else candidates[:]
        elite_count = min(len(ranked), max(6, len(ranked) // 2))
        elite = ranked[:elite_count]

        # Prefer high scoring programs, while spreading mutations across tied
        # elites and rewarding parents whose children previously improved.
        weights: List[float] = []
        best = numeric[0][0] if numeric else 0.0
        for p in elite:
            score = self._score(p)
            quality = 1.0 + max(0.0, (score or 0.0) - best + 0.003) * 100.0
            novelty = 1.0 / (1.0 + self.parent_uses.get(p.id, 0))
            learned_gain = max(0.0, self.parent_gain.get(p.id, 0.0)) * 25.0
            weights.append(quality * (0.35 + novelty) + learned_gain)

        parent = self.random_state.choices(elite, weights=weights, k=1)[0]

        current_iteration = getattr(self, "last_iteration", 0)
        stalled = current_iteration - self.last_meaningful_improvement >= 8

        # Refinement is deliberately sparse and focused: this population is
        # already valid and tightly clustered, while repeated divergence has
        # produced worse candidates.
        if stalled and self.random_state.random() < 0.30:
            return {self.REFINE_LABEL: parent}, {}

        context_count = max(0, num_context_programs or 0)
        pool = [p for p in ranked if p.id != parent.id]
        context: List[EvolvedProgram] = []

        # Give the model several independently generated near-best layouts.
        for p in pool[: min(context_count, 3)]:
            context.append(p)

        # Add one contrasting, still-valid candidate when available, rather
        # than filling every context slot with nearly identical top ties.
        if len(context) < context_count:
            contrast = [
                p for p in pool[3:]
                if p.id not in {q.id for q in context}
                and self._score(p) is not None
                and self._score(p) >= best - 0.02
            ]
            if contrast:
                context.append(self.random_state.choice(contrast))

        remaining = [p for p in pool if p.id not in {q.id for q in context}]
        self.random_state.shuffle(remaining)
        context.extend(remaining[: max(0, context_count - len(context))])

        return {"": parent}, {"": context[:context_count]}


# EVOLVE-BLOCK-END