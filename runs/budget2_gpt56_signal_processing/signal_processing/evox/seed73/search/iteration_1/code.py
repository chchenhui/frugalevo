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
    """Adaptive score-guided search with diverse context selection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))

        # These are rebuilt naturally when checkpointed programs are re-added.
        self.best_observed_score = -float("inf")
        self.stagnation_steps = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_rewards: Dict[str, float] = {}
        self.context_uses: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            score = float(value)
            if math.isfinite(score):
                return score
        return None

    def _meaningful_improvement(self, score: float) -> bool:
        if not math.isfinite(self.best_observed_score):
            return True
        threshold = max(0.01, 0.01 * abs(self.best_observed_score))
        return score > self.best_observed_score + threshold

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        """Store the program and learn which ancestors produced improvements."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        score = self._score(program)

        # Learn from completed mutations. This belongs in add rather than sample
        # so it survives normal checkpoint reconstruction.
        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            parent = self.get(program.parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None:
                gain = score - parent_score
                self.parent_rewards[program.parent_id] = (
                    self.parent_rewards.get(program.parent_id, 0.0) + gain
                )

        for context_id in program.other_context_ids or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        if program.parent_info and program.parent_info[0]:
            label = program.parent_info[0]
            self.label_uses[label] = self.label_uses.get(label, 0) + 1

        if score is not None:
            if self._meaningful_improvement(score):
                self.best_observed_score = score
                self.stagnation_steps = 0
            else:
                if score > self.best_observed_score:
                    self.best_observed_score = score
                self.stagnation_steps += 1

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
        """Choose a strong but not repeatedly exhausted parent and diverse evidence."""
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(p, self._score(p)) for p in candidates]
        valid = [(p, s) for p, s in scored if s is not None]
        if not valid:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        valid.sort(key=lambda item: item[1], reverse=True)
        best_score = valid[0][1]
        worst_score = valid[-1][1]
        score_span = max(best_score - worst_score, 1e-9)

        # Restrict exploitation to the useful upper population, while giving
        # lightly used parents an explicit chance to prove themselves.
        elite_count = max(2, min(len(valid), max(4, (len(valid) + 2) // 2)))
        elite = valid[:elite_count]

        weights: List[float] = []
        for rank, (program, score) in enumerate(elite):
            normalized = (score - worst_score) / score_span
            uses = self.parent_uses.get(program.id, 0)
            reward = self.parent_rewards.get(program.id, 0.0) / max(1, uses)

            # Score is primary; novelty prevents repeatedly mutating one plateau.
            weight = 0.4 + 2.5 * normalized + 1.2 / (1 + uses)
            weight += max(-0.25, min(0.5, reward * 20.0))
            weight *= 1.0 / (1.0 + 0.12 * rank)
            weights.append(max(0.05, weight))

        parent = self.random_state.choices(
            [program for program, _ in elite], weights=weights, k=1
        )[0]

        label = ""
        # A stalled population benefits from an occasional explicit instruction,
        # but normal score/context selection remains the default mechanism.
        if self.stagnation_steps >= 8 and self.label_uses.get(self.DIVERGE_LABEL, 0) < 2:
            non_best = [p for p, _ in elite if p.id != valid[0][0].id]
            if non_best:
                parent = self.random_state.choice(non_best)
                label = self.DIVERGE_LABEL
        elif self.stagnation_steps >= 5 and self.label_uses.get(self.REFINE_LABEL, 0) < 2:
            parent = valid[0][0]
            label = self.REFINE_LABEL

        context_count = max(0, num_context_programs or 0)
        available = [p for p, _ in valid if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Include a high-quality reference, then choose underused alternatives
        # from distinct score regions instead of recycling weak old contexts.
        if available and context_count:
            contexts.append(available[0])

        remaining = [p for p in available if p.id not in {c.id for c in contexts}]
        while remaining and len(contexts) < context_count:
            choices: List[EvolvedProgram] = []
            choice_weights: List[float] = []
            for program in remaining:
                score = self._score(program)
                normalized = ((score or worst_score) - worst_score) / score_span
                use_bonus = 1.0 / (1 + self.context_uses.get(program.id, 0))
                # Favor upper-mid alternatives, with some lower-score contrast.
                diversity = 1.0 - abs(normalized - 0.65)
                choices.append(program)
                choice_weights.append(max(0.1, 0.4 + diversity + use_bonus))

            chosen = self.random_state.choices(choices, weights=choice_weights, k=1)[0]
            contexts.append(chosen)
            remaining = [p for p in remaining if p.id != chosen.id]

        return {label: parent}, {"": contexts}


# EVOLVE-BLOCK-END