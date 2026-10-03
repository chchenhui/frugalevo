# EVOLVE-BLOCK-START
import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive exploit-first search with stagnation-triggered refine/diverge.

    The population is tightly clustered near the best score, so the strategy is:
    1. EXPLOIT: mutate top-tier programs, rotating among them (usage penalties)
       so no single parent is overused.
    2. ADAPT: when progress stalls, first REFINE the best candidate; if still
       stuck, periodically DIVERGE from a fresh mid-tier program with the best
       program kept as contrast context.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = -math.inf
        self.stagnation = 0
        self.parent_usage: Dict[str, int] = {}
        self.sample_calls = 0

    # ---------- helpers ----------

    @staticmethod
    def _score(program: EvolvedProgram) -> Optional[float]:
        metrics = getattr(program, "metrics", None)
        if not metrics:
            return None
        value = metrics.get("combined_score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    # ---------- required API ----------

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)

        # Track which parents are being used (via children's parent_id).
        parent_id = getattr(program, "parent_id", None)
        if parent_id:
            self.parent_usage[parent_id] = self.parent_usage.get(parent_id, 0) + 1

        # Track progress state: stagnation since last meaningful improvement.
        score = self._score(program)
        if score is not None:
            if score > self.best_score:
                meaningful = (score - self.best_score) > max(
                    0.01, 0.01 * abs(self.best_score)
                ) if self.best_score != -math.inf else True
                self.best_score = score
                self.stagnation = 0 if meaningful else self.stagnation + 1
            else:
                self.stagnation += 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        num_ctx = num_context_programs or 4
        all_programs = list(self.programs.values())
        if not all_programs:
            raise ValueError("No candidates available for sampling")

        scored = [(p, self._score(p)) for p in all_programs]
        scored = [(p, s) for p, s in scored if s is not None]
        if not scored:
            # No numeric scores yet: random parent, random context.
            parent = self.random_state.choice(all_programs)
            ctx = [p for p in all_programs if p.id != parent.id]
            self.random_state.shuffle(ctx)
            return {"": parent}, {"": ctx[:num_ctx]}

        scored.sort(key=lambda x: -x[1])
        best_p, best_s = scored[0]

        # Context/parent pool: drop severe failures (e.g. error outputs ~0.02
        # when everything else is ~0.13) so they don't pollute generation.
        pool = [p for p, s in scored if s >= 0.5 * max(best_s, 1e-9)]
        if len(pool) < 2:
            pool = [p for p, _ in scored]

        self.sample_calls += 1
        top = pool[: max(2, len(pool) // 2)]  # upper half of viable pool
        bottom = pool[max(2, len(pool) // 2):]

        label = ""
        parent = None

        if self.stagnation >= 4 and self.sample_calls % 3 == 0:
            # Deeply stuck: periodically diverge from a fresh, least-used
            # mid/lower-tier program to break out of the local cluster.
            mid = bottom if bottom else pool
            parent = min(mid, key=lambda p: self.parent_usage.get(p.id, 0))
            label = self.DIVERGE_LABEL
        elif self.stagnation >= 2:
            # Stalled but promising: refine the best (rotate among top few to
            # avoid overusing a single program).
            top_few = sorted(top[:3], key=lambda p: self.parent_usage.get(p.id, 0))
            parent = top_few[0]
            label = self.REFINE_LABEL
        else:
            # Default: exploit top tier with usage penalty for diversity.
            weights = [
                max(s, 1e-6) / (1.0 + 2.0 * self.parent_usage.get(p.id, 0))
                for p, s in [(p, self._score(p)) for p in top]
            ]
            total = sum(weights)
            r = self.random_state.random() * total
            acc = 0.0
            parent = top[0]
            for p, w in zip(top, weights):
                acc += w
                if r <= acc:
                    parent = p
                    break

        # Context: best program (as target/contrast) + stratified diversity.
        context: List[EvolvedProgram] = []
        if best_p.id != parent.id:
            context.append(best_p)
        others = [p for p in pool if p.id != parent.id and p.id != best_p.id]
        self.random_state.shuffle(others)
        # Alternate picks from the strong half and weaker half for contrast.
        strong = [p for p in others if p in top]
        weak = [p for p in others if p not in top]
        i = j = 0
        while len(context) < num_ctx and (i < len(strong) or j < len(weak)):
            if i < len(strong):
                context.append(strong[i]); i += 1
            if len(context) < num_ctx and j < len(weak):
                context.append(weak[j]); j += 1

        return {label: parent}, {"": context}
# EVOLVE-BLOCK-END