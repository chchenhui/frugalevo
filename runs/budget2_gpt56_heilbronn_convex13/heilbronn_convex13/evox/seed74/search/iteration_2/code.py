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
    """Adaptive elite search with occasional targeted escapes from plateaus."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score_seen = float("-inf")
        self.stagnation_steps = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_rewards: Dict[str, float] = {}
        self.score_history: List[float] = []

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        score = self._score(program)
        previous_best = self.best_score_seen

        # Progress and parent-outcome accounting live here so they survive
        # normal add/checkpoint flows rather than being altered by sampling.
        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            parent = self.get(program.parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None:
                gain = score - parent_score
                old = self.parent_rewards.get(program.parent_id, 0.0)
                self.parent_rewards[program.parent_id] = 0.75 * old + 0.25 * gain

        if score is not None:
            self.score_history.append(score)
            if score > previous_best:
                absolute_gain = score - previous_best if math.isfinite(previous_best) else 0.0
                relative_gain = (
                    absolute_gain / max(abs(previous_best), 1e-9)
                    if math.isfinite(previous_best)
                    else 0.0
                )
                if not math.isfinite(previous_best) or (
                    absolute_gain >= 0.01 or relative_gain >= 0.01
                ):
                    self.stagnation_steps = 0
                else:
                    self.stagnation_steps += 1
                self.best_score_seen = score
            else:
                self.stagnation_steps += 1

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
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
        valid = [(p, s) for p, s in scored if s is not None]

        if not valid:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        valid.sort(key=lambda item: item[1], reverse=True)
        best = valid[0][1]

        # The population is strongly concentrated near one plateau.  Keep a
        # reasonably broad elite pool rather than always copying the single best.
        elite = [
            (p, s) for p, s in valid
            if s >= best - max(0.004, abs(best) * 0.006)
        ]
        if len(elite) < 4:
            elite = valid[:min(8, len(valid))]

        weights: List[float] = []
        for rank, (program, score) in enumerate(elite):
            quality = 1.0 + max(0.0, score - elite[-1][1]) * 80.0
            novelty = 1.0 / (1.0 + self.parent_uses.get(program.id, 0))
            reward = max(0.25, 1.0 + self.parent_rewards.get(program.id, 0.0) * 80.0)
            weights.append((quality * reward * novelty) + 0.15 / (rank + 1))

        parent = self.random_state.choices(
            [p for p, _ in elite], weights=weights, k=1
        )[0]

        # Long stagnation warrants a distinct attempt.  This is deliberately
        # intermittent: most iterations still exploit useful elite geometry.
        use_diverge = (
            self.stagnation_steps >= 8
            and self.random_state.random() < 0.40
        )
        if use_diverge:
            # A plateau representative is a better divergence target than an
            # isolated best point, and low-use parents diversify directions.
            plateau = elite[1:] if len(elite) > 1 else elite
            parent = min(
                (p for p, _ in plateau),
                key=lambda p: self.parent_uses.get(p.id, 0) + self.random_state.random(),
            )
            return {self.DIVERGE_LABEL: parent}, {}

        context_count = max(0, num_context_programs or 0)
        pool = [p for p, _ in valid if p.id != parent.id]

        # Context combines strong alternatives with one occasional contrasting
        # survivor, avoiding repeated identical parent/context combinations.
        top_pool = [p for p, s in valid if p.id != parent.id and s >= best - 0.01]
        self.random_state.shuffle(top_pool)
        context: List[EvolvedProgram] = top_pool[:min(context_count, 3)]

        remaining = [p for p in pool if p.id not in {x.id for x in context}]
        if len(context) < context_count and remaining:
            lower = [p for p in remaining if self._score(p) < best - 0.01]
            source = lower if lower and self.random_state.random() < 0.35 else remaining
            self.random_state.shuffle(source)
            context.extend(source[:context_count - len(context)])

        return {"": parent}, {"": context[:context_count]}


# EVOLVE-BLOCK-END