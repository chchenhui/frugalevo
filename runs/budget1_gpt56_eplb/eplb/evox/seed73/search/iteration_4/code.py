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
    """Adaptive score-aware population search."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))

        # Progress is updated in add(), so it survives normal database replay.
        self.best_numeric_score: Optional[float] = None
        self.initial_numeric_score: Optional[float] = None
        self.stagnation_steps = 0
        self.recent_parent_ids: List[str] = []
        self.parent_trials: Dict[str, int] = {}
        self.parent_gains: Dict[str, List[float]] = {}
        self.label_attempts = 0

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

        score = self._score(program)
        if score is not None:
            if self.initial_numeric_score is None:
                self.initial_numeric_score = score

            if self.best_numeric_score is None:
                self.best_numeric_score = score
            else:
                meaningful_gap = max(0.01, abs(self.best_numeric_score) * 0.01)
                if score > self.best_numeric_score + meaningful_gap:
                    self.best_numeric_score = score
                    self.stagnation_steps = 0
                else:
                    self.stagnation_steps += 1

        # Attribute child outcomes to the selected parent. This makes successful
        # parent families somewhat more likely without repeatedly fixing on one.
        if program.parent_id:
            self.parent_trials[program.parent_id] = (
                self.parent_trials.get(program.parent_id, 0) + 1
            )
            self.recent_parent_ids.append(program.parent_id)
            self.recent_parent_ids = self.recent_parent_ids[-8:]

            parent = self.get(program.parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None:
                gains = self.parent_gains.setdefault(program.parent_id, [])
                gains.append(score - parent_score)
                self.parent_gains[program.parent_id] = gains[-5:]

        if isinstance(program.parent_info, tuple) and program.parent_info:
            if program.parent_info[0] in (self.DIVERGE_LABEL, self.REFINE_LABEL):
                self.label_attempts += 1

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

        context_count = 4 if num_context_programs is None else max(0, num_context_programs)
        scored = [(self._score(p), p) for p in candidates]
        numeric = [(s, p) for s, p in scored if s is not None]

        # If evaluation failed for every member, retain robust random behavior.
        if not numeric:
            parent = self.random_state.choice(candidates)
            others = [p for p in candidates if p.id != parent.id]
            self.random_state.shuffle(others)
            return {"": parent}, {"": others[:context_count]}

        numeric.sort(key=lambda item: item[0], reverse=True)
        best = numeric[0][0]
        worst = numeric[-1][0]

        # The useful population is concentrated near the top.  Search within
        # that region, while retaining enough near-elite alternatives to escape
        # the repeated 0.1276-style plateau.
        pool_size = min(len(numeric), max(12, (len(numeric) * 3) // 5))
        parent_pool = numeric[:pool_size]

        weights: List[float] = []
        for rank, (score, program) in enumerate(parent_pool):
            span = max(best - worst, 1e-9)
            quality = (score - worst) / span
            trials = self.parent_trials.get(program.id, 0)
            gains = self.parent_gains.get(program.id, [])
            mean_gain = sum(gains) / len(gains) if gains else 0.0
            gain_bonus = max(-0.5, min(0.75, mean_gain * 80.0))
            novelty_bonus = 0.55 / (1.0 + trials)
            rank_bonus = 0.35 * (1.0 - rank / max(1, pool_size - 1))
            recent_penalty = 0.25 if program.id in self.recent_parent_ids[-3:] else 1.0
            weights.append(max(0.03, (0.7 + quality + novelty_bonus + rank_bonus + gain_bonus)
                               * recent_penalty))

        parent = self.random_state.choices(
            [p for _, p in parent_pool], weights=weights, k=1
        )[0]

        # Labels are deliberately rare: prior label attempts were not reliably
        # helpful, but a deeply stagnant run merits an occasional clean break.
        label = ""
        if (
            self.stagnation_steps >= 15
            and self.label_attempts < 2
            and self.random_state.random() < 0.12
        ):
            mid_start = min(6, len(parent_pool) - 1)
            alternatives = [p for _, p in parent_pool[mid_start:]]
            if alternatives:
                parent = self.random_state.choice(alternatives)
                label = self.DIVERGE_LABEL

        if label:
            return {label: parent}, {}

        # Context deliberately mixes elite examples with a different score band.
        # This gives the LLM strong templates plus a contrasting alternative,
        # rather than four near-identical plateau solutions.
        selected: List[EvolvedProgram] = []
        used_ids = {parent.id}

        def take_from(group: List[Tuple[float, EvolvedProgram]], count: int) -> None:
            available = [p for _, p in group if p.id not in used_ids]
            self.random_state.shuffle(available)
            for item in available[:count]:
                used_ids.add(item.id)
                selected.append(item)

        take_from(numeric[:8], min(2, context_count))
        if len(selected) < context_count:
            take_from(numeric[8:24], 1)
        if len(selected) < context_count:
            take_from(numeric[24:], 1)
        if len(selected) < context_count:
            take_from(numeric, context_count - len(selected))

        return {"": parent}, {"": selected[:context_count]}


# EVOLVE-BLOCK-END