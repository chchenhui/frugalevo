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
    """Adaptive elite search with novelty-aware parent and context selection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.initial_program = None
        self.best_seen_score: Optional[float] = None
        self.last_meaningful_add = 0
        self.add_count = 0
        self.parent_use_count: Dict[str, int] = {}
        self.context_use_count: Dict[str, int] = {}
        self.parent_gain_sum: Dict[str, float] = {}
        self.parent_gain_count: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.add_count += 1
        score = self._score(program)

        # Progress is deliberately tracked here so it survives normal database
        # checkpoint/replay behavior rather than depending on sampling calls.
        if score is not None:
            if self.best_seen_score is None:
                self.best_seen_score = score
                self.last_meaningful_add = self.add_count
            else:
                improvement = score - self.best_seen_score
                meaningful = (
                    improvement >= 0.01
                    or improvement >= 0.01 * max(abs(self.best_seen_score), 1e-12)
                )
                if score > self.best_seen_score:
                    self.best_seen_score = score
                if meaningful:
                    self.last_meaningful_add = self.add_count

        if program.parent_id:
            self.parent_use_count[program.parent_id] = (
                self.parent_use_count.get(program.parent_id, 0) + 1
            )
            parent = self.get(program.parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None:
                gain = score - parent_score
                self.parent_gain_sum[program.parent_id] = (
                    self.parent_gain_sum.get(program.parent_id, 0.0) + gain
                )
                self.parent_gain_count[program.parent_id] = (
                    self.parent_gain_count.get(program.parent_id, 0) + 1
                )

        for context_id in program.other_context_ids or []:
            self.context_use_count[context_id] = (
                self.context_use_count.get(context_id, 0) + 1
            )

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

        context_count = max(0, num_context_programs or 0)
        scored = [(p, self._score(p)) for p in candidates]
        valid = [(p, s) for p, s in scored if s is not None]

        if not valid:
            parent = self.random_state.choice(candidates)
            pool = [p for p in candidates if p.id != parent.id]
            self.random_state.shuffle(pool)
            return {"": parent}, {"": pool[:context_count]}

        valid.sort(key=lambda item: item[1], reverse=True)
        best_score = valid[0][1]
        elite_size = max(2, min(len(valid), int(math.ceil(len(valid) * 0.35))))
        elite = [p for p, _ in valid[:elite_size]]

        # Mostly exploit the strong near-optimal cluster, but preserve regular
        # access to alternative good lineages when the population is stuck.
        stalled = self.add_count - self.last_meaningful_add >= 7
        if stalled and len(valid) > elite_size and self.random_state.random() < 0.35:
            parent_pool = [p for p, _ in valid[:max(elite_size + 3, len(valid) // 2)]]
        else:
            parent_pool = elite

        weights: List[float] = []
        for program in parent_pool:
            score = self._score(program) or 0.0
            use = self.parent_use_count.get(program.id, 0)
            gains = self.parent_gain_count.get(program.id, 0)
            mean_gain = self.parent_gain_sum.get(program.id, 0.0) / max(gains, 1)

            # Score matters, while inverse reuse prevents repeatedly mutating
            # the same score-1.0 program.
            quality = 1.0 + max(0.0, score - best_score + 0.02) * 20.0
            novelty = 1.0 / (1.0 + use)
            evidence = 1.0 + max(-0.2, min(0.2, mean_gain)) * 3.0
            weights.append(max(0.05, quality * (0.45 + novelty) * evidence))

        parent = self.random_state.choices(parent_pool, weights=weights, k=1)[0]

        label = ""
        if stalled:
            labeled_attempts = sum(
                1 for p in candidates if p.parent_info and p.parent_info[0]
            )
            # Sparse directives are useful only after genuine stagnation.
            if labeled_attempts < max(3, len(candidates) // 6):
                if self.random_state.random() < 0.35:
                    label = self.REFINE_LABEL
                elif self.random_state.random() < 0.30:
                    label = self.DIVERGE_LABEL

        if label:
            return {label: parent}, {}

        # Context combines an elite reference, a nearby alternative, and
        # underused candidates from other score bands/lineages.
        remaining = [p for p in candidates if p.id != parent.id]
        selected: List[EvolvedProgram] = []

        elite_others = [p for p in elite if p.id != parent.id]
        if elite_others and context_count:
            selected.append(
                min(elite_others, key=lambda p: self.context_use_count.get(p.id, 0))
            )

        while len(selected) < context_count:
            pool = [p for p in remaining if p.id not in {x.id for x in selected}]
            if not pool:
                break

            def context_weight(p: EvolvedProgram) -> float:
                score = self._score(p) or 0.0
                distance = abs(score - (self._score(parent) or score))
                underused = 1.0 / (1.0 + self.context_use_count.get(p.id, 0))
                # Favor good examples, but retain contrasting near-good attempts.
                return max(0.05, (0.5 + score / max(best_score, 1e-9) + distance) * underused)

            choices = [context_weight(p) for p in pool]
            selected.append(self.random_state.choices(pool, weights=choices, k=1)[0])

        return {"": parent}, {"": selected}


# EVOLVE-BLOCK-END