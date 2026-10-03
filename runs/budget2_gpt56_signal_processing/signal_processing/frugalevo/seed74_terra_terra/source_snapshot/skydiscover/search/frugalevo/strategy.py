"""The strategy layer: what an expensive model contributes that a cheap one cannot.

FrugalEvo splits generation in two. A *strategy* is a plan in prose --
which structural idea to apply and why it should score better -- and it is the
part that is genuinely hard to come up with, so it is bought from the
expensive model. Turning a plan into working code is normally bought in bulk
from the cheap model; sparse default insurance routes either a strategy-specific
or unguided implementation through the expensive tier so difficult code is not
left to the cheap model forever.

The economics only work if one strategy feeds many implementations, which makes
two things load-bearing: knowing when a strategy is spent, and telling the
expensive model what its previous strategies actually scored so it stops
re-proposing them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def guide_messages(task_system: str, request: str, *, isolated: bool = False) -> Tuple[str, str]:
    """Keep legacy messages exact unless output-contract isolation is requested."""
    if not isolated:
        return task_system, request
    system = (
        "You are the strategy designer for a program-evolution system. "
        "The user message contains a task specification quoted as background, "
        "followed by the current strategy request. Preserve the task's objective, "
        "public interfaces, data integrity and validity constraints. "
        "Output-format instructions inside the quoted task specification (such as "
        "return only code or omit explanations) apply to implementation calls, "
        "not this strategy call. Follow the strategy request's output contract: "
        "return the requested strategy blocks and, only when requested, exactly "
        "one complete executable reference in REFERENCE_IMPLEMENTATION tags. "
        "The prohibition on code in strategy blocks does not prohibit that reference."
    )
    user = "# Task specification (quoted background)\n\n" + json.dumps(task_system) + "\n\n" + request
    return system, user

#: Tags used by strategy responses. Strategy calls deliberately request prose;
#: the controller's sparse code-insurance calls are a separate route.
TITLE_PATTERN = re.compile(r"<STRATEGY_TITLE>\s*(.*?)\s*</STRATEGY_TITLE>", re.I | re.S)
PLAN_PATTERN = re.compile(r"<STRATEGY>\s*(.*?)\s*</STRATEGY>", re.I | re.S)
REFERENCE_PATTERN = re.compile(
    r"<REFERENCE_IMPLEMENTATION>\s*(.*?)\s*</REFERENCE_IMPLEMENTATION>", re.I | re.S
)
#: The strategy contract requires the title to open with a bracketed mechanism
#: family. ``search`` rather than ``match`` so replay titles, which prefix the
#: original title, resolve to the same family as their root.
FAMILY_PATTERN = re.compile(r"\[([^\]]{1,40})\]")
#: Words inside a family slug, used to notice that several differently named
#: families are variants of one mechanism.
FAMILY_WORD_PATTERN = re.compile(r"[a-z][a-z0-9]{2,}")
#: Bounds on the repeated-word report. The counts carry the signal; the names
#: only illustrate it, so neither may grow with the length of the run.
_MAX_RUT_LINES = 8
_MAX_RUT_NAMES = 6


@dataclass
class Strategy:
    """One plan, plus what it went on to achieve."""

    index: int
    title: str
    plan: str
    cost: float = 0.0
    #: Best score reached by any implementation of this strategy.
    best_score: Optional[float] = None
    #: Score the incumbent stood at when this strategy was proposed.
    baseline_score: Optional[float] = None
    #: Minimum score gain that counts as a real improvement rather than jitter.
    improvement_epsilon: float = 1e-9
    rounds: int = 0
    #: Program candidates attempted across both model tiers, including
    #: parse/evaluation failures. Tier-specific counters below let the
    #: scheduler recover the Luna-only M_max usage.
    attempts: int = 0
    #: Implementation calls that produced a usable evaluated program.
    implementations: int = 0
    #: Strong-model code attempts already spent on this plan.
    guide_implementation_attempts: int = 0
    #: Strong-model code attempts that produced usable evaluated programs.
    guide_implementations: int = 0
    #: Replays are new ledger entries, but share credit with the original
    #: strategy through this root index. ``None`` means this is the root.
    source_strategy_index: Optional[int] = None
    #: Positive global-incumbent gain actually caused by this strategy.  This
    #: differs from ``best_score - baseline_score`` because it sums useful
    #: gains across multiple activations without crediting unrelated hedges.
    cumulative_gain: float = 0.0
    #: Legacy scale-free gain retained for old-checkpoint compatibility.
    cumulative_relative_gain: float = 0.0
    productive_rounds: int = 0
    barren_rounds: int = 0
    #: Measured generation spend after the prose strategy has been bought.
    implementation_spend: float = 0.0
    #: Guide-authored references are extra observations bundled with a
    #: strategy acquisition. They affect reward and validity statistics but
    #: do not consume the root's Luna M_max quota.
    reference_attempts: int = 0
    reference_implementations: int = 0
    #: The optional executable supplied for this strategy by schedulers that
    #: use a strategy-specific Terra reference. Ordered racing instead treats
    #: its reference as portfolio-level evidence and leaves these fields empty.
    reference_solution: Optional[str] = None
    reference_score: Optional[float] = None
    last_used_iteration: int = -1
    failure_feedback: List[str] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        if self.best_score is None or self.baseline_score is None:
            return False
        return self.best_score > self.baseline_score + self.improvement_epsilon

    @property
    def label(self) -> str:
        return f"#{self.index} {self.title}"

    @property
    def family(self) -> str:
        """Mechanism-family slug the guide emitted, or ``""`` when malformed.

        Derived from the title rather than stored, so replay entries and
        strategies restored from older checkpoints resolve it too.
        """
        match = FAMILY_PATTERN.search(self.title)
        return match.group(1).strip().lower() if match else ""

    @property
    def root_index(self) -> int:
        """Stable search-operator identity shared by replay entries."""
        return self.source_strategy_index or self.index

    def outcome(self) -> str:
        """One line for the next strategy prompt: what this plan actually did."""
        luna_attempts = max(
            0,
            self.attempts - self.reference_attempts - self.guide_implementation_attempts,
        )
        luna_implementations = max(
            0,
            self.implementations - self.reference_implementations - self.guide_implementations,
        )
        yield_text = (
            f"; {luna_implementations}/{luna_attempts} "
            "weak model implementations usable"
        )
        if self.guide_implementation_attempts:
            yield_text += (
                f"; {self.guide_implementations}/{self.guide_implementation_attempts} "
                "strong model implementation(s) usable"
            )
        if self.reference_attempts:
            yield_text += (
                f"; {self.reference_implementations}/{self.reference_attempts} "
                "strong model reference(s) usable"
            )
        failures = (
            "; failures: " + " | ".join(self.failure_feedback[-3:]) if self.failure_feedback else ""
        )
        if self.best_score is None:
            return (
                f"produced nothing that evaluated ({luna_implementations}/{luna_attempts} "
                "weak model implementations usable)"
                f"{failures}"
            )
        if self.improved:
            gain = self.best_score - (self.baseline_score or 0.0)
            return (
                f"best score {self.best_score:.6f} (+{gain:.6f} — an improvement)"
                f"{yield_text}{failures}"
            )
        return (
            f"best score {self.best_score:.6f} (no improvement on {self.baseline_score:.6f})"
            f"{yield_text}{failures}"
        )

    def record_failure(self, feedback: str) -> None:
        """Keep a few distinct, compact failure reasons for the next guide."""
        feedback = re.sub(r"\s+", " ", feedback).strip()[:400]
        if feedback and feedback not in self.failure_feedback:
            self.failure_feedback.append(feedback)
            self.failure_feedback = self.failure_feedback[-3:]


@dataclass
class StrategyLedger:
    """Every strategy tried, in order, with its outcome."""

    entries: List[Strategy] = field(default_factory=list)

    def add(self, strategy: Strategy) -> None:
        self.entries.append(strategy)

    def history_text(self, limit: int = 32) -> str:
        """Past strategies and their outcomes, for the strategy prompt.

        This is the whole reason to pay for an expensive model twice: without
        it the second call is a fresh draw from the same distribution and will
        propose the first idea again.
        """
        past = [s for s in self.entries if s.rounds > 0]
        if not past:
            return ""

        # A recent-only window eventually forgets the very operators that
        # produced the incumbent.  Keep a small hall of fame plus the most
        # recent failures.  This is deterministic and costs no extra guide
        # call; it only spends the bounded prompt slots more usefully.
        limit = max(1, int(limit))

        def gain(strategy: Strategy) -> float:
            if strategy.cumulative_gain > 0:
                return strategy.cumulative_gain
            if strategy.cumulative_relative_gain > 0:
                return strategy.cumulative_relative_gain
            if strategy.best_score is None or strategy.baseline_score is None:
                return 0.0
            return max(0.0, strategy.best_score - strategy.baseline_score)

        hall_size = max(1, limit // 3)
        hall = sorted(past, key=lambda strategy: (gain(strategy), strategy.index), reverse=True)
        hall = [strategy for strategy in hall if gain(strategy) > 0][:hall_size]
        selected = list(hall)
        selected_indices = {strategy.index for strategy in selected}
        for strategy in reversed(past):
            if strategy.index in selected_indices:
                continue
            selected.append(strategy)
            selected_indices.add(strategy.index)
            if len(selected) >= limit:
                break

        lines = []
        hall_indices = {strategy.index for strategy in hall}
        for strategy in selected:
            role = " [high-value operator]" if strategy.index in hall_indices else ""
            lines.append(f'- "{strategy.title}"{role} → {strategy.outcome()}')
        return (
            "\n\n# Strategies already tried\n\n"
            + "\n".join(lines)
            + "\n\nDo not restate any of these. If one of them improved the score, the next "
            "strategy may build on it, but it must add a new idea rather than repeat it. "
            "High-value operators are retained from the full run, not merely the recent window."
        )

    def _roots(self) -> List[Tuple[int, str, Optional[float], bool]]:
        """One ``(root, family, best score, ever improved)`` per operator.

        A root and its replays share ``root_index``; counting entries instead
        of roots would report one plan tried four times as four plans.
        """
        by_root: Dict[int, List[Strategy]] = {}
        for strategy in self.entries:
            if strategy.rounds > 0:
                by_root.setdefault(strategy.root_index, []).append(strategy)
        roots: List[Tuple[int, str, Optional[float], bool]] = []
        for root in sorted(by_root):
            group = by_root[root]
            scored = [s.best_score for s in group if s.best_score is not None]
            # Any entry of the group can carry the slug; a malformed title in
            # one replay must not erase the whole operator's family.
            family = next((s.family for s in group if s.family), "")
            roots.append(
                (
                    root,
                    family,
                    max(scored) if scored else None,
                    any(s.improved for s in group),
                )
            )
        return roots

    def exhausted_families_text(
        self,
        *,
        min_variants: int = 3,
        barren_tail: int = 6,
        max_families: int = 40,
    ) -> str:
        """Mechanism families the guide has already spent, plus a stall rule.

        ``history_text`` is a chronological record; drawing the conclusion from
        it is left to the reader, and a bounded window can drop the very
        repetition that matters. This block states the conclusion directly: a
        family that has been tried several times without a gain, and the words
        shared by differently named variants of one mechanism.
        """
        roots = self._roots()
        if not roots:
            return ""

        families: Dict[str, List[tuple]] = {}
        for _root, family, best, improved in roots:
            if family:
                families.setdefault(family, []).append((best, improved))

        # A long run accumulates more families than the prompt should carry.
        # Keep the ones that carry signal -- an operator that worked, or one
        # the guide has already drawn from repeatedly.
        ordered = sorted(
            families,
            key=lambda name: (
                not any(flag for _, flag in families[name]),
                -len(families[name]),
                name,
            ),
        )
        shown = sorted(ordered[: max(1, int(max_families))])
        omitted = len(families) - len(shown)

        lines = []
        for family in shown:
            records = families[family]
            scored = [best for best, _ in records if best is not None]
            if any(improved for _, improved in records):
                verdict = "improved the score at least once"
            else:
                verdict = "never improved"
            best_text = f"best {max(scored):.6f}" if scored else "nothing evaluated"
            lines.append(f'- "{family}" — {len(records)} attempt(s), {best_text}, {verdict}')

        # Differently named families that share a word are usually variants of
        # one mechanism. Counting words needs no curated list of what counts as
        # a generic modifier: a modifier repeated this often is itself a rut.
        word_families: Dict[str, set] = {}
        word_improved: Dict[str, bool] = {}
        for family, records in families.items():
            improved = any(flag for _, flag in records)
            for word in set(FAMILY_WORD_PATTERN.findall(family)):
                word_families.setdefault(word, set()).add(family)
                word_improved[word] = word_improved.get(word, False) or improved
        # Widest rut first, and both the number of ruts and the names quoted
        # inside each are bounded: a long run must not grow this block without
        # limit, and the count already carries the signal the names illustrate.
        ruts = sorted(
            (
                (word, names)
                for word, names in word_families.items()
                if len(names) >= max(2, int(min_variants)) and not word_improved[word]
            ),
            key=lambda item: (-len(item[1]), item[0]),
        )[:_MAX_RUT_LINES]
        rut_lines = []
        for word, names in ruts:
            listed = sorted(names)[:_MAX_RUT_NAMES]
            rest = len(names) - len(listed)
            shown = ", ".join(listed) + (f", +{rest} more" if rest > 0 else "")
            rut_lines.append(
                f'- "{word}" appears in {len(names)} distinct families '
                f"({shown}), none of which improved"
            )

        # Trailing operators that all failed: the portfolio is not merely
        # unlucky, the families it keeps drawing from are spent.
        barren = 0
        for _root, _family, _best, improved in reversed(roots):
            if improved:
                break
            barren += 1
        escalate = barren >= max(1, int(barren_tail))

        if omitted > 0:
            lines.append(f"- ... and {omitted} further family/families already tried")

        text = "\n\n# Mechanism families already spent\n\n" + "\n".join(lines)
        if rut_lines:
            text += "\n\nRepeated across differently named families:\n\n" + "\n".join(rut_lines)
        if escalate:
            text += (
                f"\n\nThe last {barren} search operators all failed to improve the score. "
                "Every strategy in this portfolio MUST come from a mechanism family absent "
                "from the list above. A variant, reparameterization, hybrid, or "
                "differently-windowed version of a spent family is not a new family."
            )
        else:
            text += (
                "\n\nDo not propose another variant of a family that never improved. "
                "A family that did improve may be refined further, but only with a new "
                "degree of freedom rather than a reparameterization."
            )
        return text

def parse_strategy(response: str, index: int) -> Optional[Strategy]:
    """Pull a titled plan out of a strategy response.

    Falls back to using the whole response as the plan when the model answers
    without tags: a usable plan in the wrong wrapper is still worth the call
    that produced it.
    """
    if not response or not response.strip():
        return None

    title_match = TITLE_PATTERN.search(response)
    plan_match = PLAN_PATTERN.search(response)

    plan = (plan_match.group(1) if plan_match else response).strip()
    if not plan:
        return None

    title = _strategy_title(title_match.group(1) if title_match else "", plan, index)

    return Strategy(index=index, title=title, plan=plan)


def _strategy_title(raw_title: str, plan: str, index: int) -> str:
    """Return a compact title even when a model emits malformed wrapper tags."""
    title = re.sub(r"</?STRATEGY(?:_TITLE)?[^>]*>", "", raw_title, flags=re.I).strip()
    if not title:
        title = re.split(r"(?<=[.!?])\s", plan.strip())[0]
        title = re.sub(r"</?STRATEGY(?:_TITLE)?[^>]*>", "", title, flags=re.I).strip()
    return title[:80].strip() or f"strategy {index}"


def parse_strategies(response: str, start_index: int, limit: int) -> List[Strategy]:
    """Parse up to ``limit`` repeated strategy blocks from one guide response.

    A single usable unwrapped plan remains valid for backward compatibility.
    """
    if not response or not response.strip() or limit < 1:
        return []

    titles = [value.strip() for value in TITLE_PATTERN.findall(response)]
    plans = [value.strip() for value in PLAN_PATTERN.findall(response) if value.strip()]
    if not plans:
        fallback = parse_strategy(response, start_index)
        return [fallback] if fallback is not None else []

    strategies: List[Strategy] = []
    for offset, plan in enumerate(plans[:limit]):
        index = start_index + offset
        raw_title = titles[offset] if offset < len(titles) else ""
        strategies.append(
            Strategy(index=index, title=_strategy_title(raw_title, plan, index), plan=plan)
        )
    return strategies


def parse_reference_implementation(response: str) -> Optional[str]:
    """Extract the optional guide-authored program kept outside prose plans."""
    match = REFERENCE_PATTERN.search(response or "")
    if not match or not match.group(1).strip():
        return None
    return match.group(1).strip()


def validate_guidance_profile(profile: str) -> None:
    if profile not in ("default", "txn_scheduling"):
        raise ValueError(f"Unknown EE guidance_profile: {profile!r}; expected default or txn_scheduling")


def strategy_request(
    program: str,
    score: Optional[float],
    ledger: StrategyLedger,
    *,
    metrics: Optional[Dict[str, Any]] = None,
    target_hint: str = "",
    count: int = 1,
    include_reference: bool = False,
    shared_reference: bool = False,
    history_limit: int = 32,
    language: str = "python",
    prompt_addendum: str = "",
    guidance_profile: str = "default",
) -> str:
    """User message asking the expensive model for a small portfolio of plans."""
    validate_guidance_profile(guidance_profile)
    txn_profile = guidance_profile == "txn_scheduling"
    if txn_profile:
        reference_scope = """a second end-to-end search or optimization pipeline. Keep the public interface,
but you may replace the entire editable algorithm, including its construction
phase. Describe strategies by their mechanism so implementers can adapt them
if this reference becomes their new parent. Bound every optimizer's starts and iterations, and avoid a material
"""
    else:
        reference_scope = """a second end-to-end search or optimization pipeline. Keep the public interface
and the replacement points named by the strategies intact so that each strategy
remains implementable if the controller promotes this program as their shared
parent. Bound every optimizer's starts and iterations, and avoid a material
"""
    if txn_profile:
        gain_check = """An untested heuristic need not prove improvement before evaluation. Distinguish
measured evidence from a hypothesis; do not invent predicted scores.
Compare like with like: a predicted raw objective
"""
    else:
        gain_check = """Before returning an idea, reject it if its calculable estimate or bound does not
strictly exceed the incumbent. Compare like with like: a predicted raw objective
"""
    if txn_profile:
        exploration_scope = """Within each requested portfolio role, identify the observed bottleneck, a concrete
change, and a falsifiable improvement hypothesis. Replacing the entire editable
algorithm is allowed; do not assume the incumbent's construction is optimal.
Treat the execution deadline as a hard constraint: produce a valid complete
solution first and return the best found before the total evaluation deadline.
Within that limit, choose the best balance of expected gain and implementation
reliability. Structural changes are useful
"""
    else:
        exploration_scope = """Within each requested portfolio role, choose the idea with the best balance of
calculable gain and implementation reliability. Structural changes are useful
"""
    score_line = f" It scores {score:.6f}." if score is not None else ""
    metric_lines = []
    for name, value in (metrics or {}).items():
        if isinstance(value, bool):
            rendered = str(value)
        elif isinstance(value, (int, float)):
            rendered = f"{float(value):.12g}"
        elif isinstance(value, str) and len(value) <= 120:
            rendered = value.replace("\n", " ")
        else:
            continue
        metric_lines.append(f"- {name}: {rendered}")
    metrics_text = (
        "\n\n# Current evaluated metrics\n\n"
        + "\n".join(metric_lines)
        + "\n\nThe controller ranks candidates by the scalar score shown above "
        "(`combined_score` when that metric is present). Metric names denote "
        "different quantities and may use different units or scales."
        if metric_lines
        else ""
    )
    count = max(1, int(count))
    quantity = "exactly ONE strategy" if count == 1 else f"exactly {count} distinct strategies"
    first_strategy_reference_role = (
        "The executable reference is portfolio-level evidence and is not assigned to "
        "this or any other strategy. "
        if shared_reference
        else "It may also receive the executable reference. "
    )
    diversity = (
        ""
        if count == 1
        else (
            "\nThe strategies must use genuinely different constructions or mechanisms, "
            "not parameter variations of one idea. Changing only RNG seeds, solver tolerances, "
            "iteration limits, active-set thresholds or sizes, acceptance thresholds, or small "
            "numerical perturbations of the same construction does not create a distinct strategy. "
            "A geometrically different deterministic seed, contact topology, subsystem "
            "decomposition, or continuation direction does count as a distinct mechanism. "
            "Strategies may reuse the same reliable low-level solver; judge diversity by the "
            "evaluator-visible gain mechanism, not by the solver name. At most one strategy may "
            "be a pure "
            "rerun or parameter-only polish of the incumbent optimizer; the others must change "
            "the construction, topology, representation, optimized subsystem, or escape "
            "direction. Rank the immediately executable, hard-constraint-"
            "preserving strategy with the highest predicted combined score FIRST; it is activated "
            f"first. {first_strategy_reference_role}Use implementation reliability "
            "to break near-ties, not to put a tiny safe gain ahead of a credible structural gain. "
            "Across a portfolio of three or more, cover "
            "an explicit local exploitation, a mesoscopic subsystem replacement, and a "
            "high-upside structural exploration, in whichever order their evidence warrants. "
            "For two strategies, cover local and structural roles. Do not fill the portfolio "
            "with only tiny perturbations or only speculative reconstructions."
        )
    )
    strategy_reference_contract = """

After all strategy blocks, also implement the FIRST strategy once as a complete,
standalone program. This is an executable reference for the cheap implementers,
not an additional strategy. Preserve the required public function/signature and
wrap exactly one complete program as follows:

Because this program is evaluated as a candidate, strategy 1 and its reference
must be the valid, immediately executable proposal with the highest predicted
combined score. Use reliability only to break near-ties. Do not make it a
conservative local proposal merely to satisfy portfolio diversity; the remaining
strategies can cover other scales.

<REFERENCE_IMPLEMENTATION>
```python
the complete program
```
</REFERENCE_IMPLEMENTATION>

The program must be immediately executable and satisfy every hard constraint.
Do not put code inside any <STRATEGY> block.
"""
    shared_reference_contract = f"""

After all strategy blocks, also produce exactly one complete, standalone
PORTFOLIO reference program. It is an independent direct candidate, not an
implementation of any one strategy, and it receives no strategy credit. Make it
the strongest compact direct candidate you can provide for the current parent;
do not weaken it into a lowest-common-denominator example merely to resemble all
of the prose strategies. Prefer replacing one existing mechanism over appending
{reference_scope}evaluation-time increase unless the code introduces a specific new source of
objective gain.

Preserve the required public function/signature and wrap exactly one complete
program as follows:

<REFERENCE_IMPLEMENTATION>
```python
the complete program
```
</REFERENCE_IMPLEMENTATION>

The program must be immediately executable and satisfy every hard constraint.
Do not put code inside any <STRATEGY> block.
"""
    reference_contract = (
        shared_reference_contract if shared_reference else strategy_reference_contract
    ) if include_reference else ""
    reference_contract = reference_contract.replace("```python", f"```{language}")
    addendum = prompt_addendum.strip() if txn_profile else ""
    addendum_section = f"\n\n# Task-specific strategy constraints\n\n{addendum}" if addendum else ""
    addendum_section = "\n" + addendum_section if txn_profile else ""
    return f"""# Current parent program{score_line}

```
{program}
```{metrics_text}
{ledger.history_text(history_limit)}{ledger.exhausted_families_text()}{target_hint}{addendum_section}

# Your task

Propose {quantity} for improving this program's score.{diversity}

A strategy is a **plan, not code**. Say which structural idea to apply and why it
should score better than what is there now. Be concrete enough that a competent
implementer needs to make no further design decisions — name the construction,
the arrangement, the invariant, whatever the idea turns on. Do not write the
program: someone else implements it.

Each strategy must explicitly state four labeled items: `Mechanism family` (a
short identifier used to detect duplicates), `Replacement scope` (the exact
existing function, phase, or block to replace rather than merely augment),
`Compute budget` (hard upper bounds on new optimizer starts and per-start
iterations, or "none", plus the expected evaluation-time effect), and `Gain
mechanism` (the specific new degree of freedom, contact/topology change, or
corrected bottleneck that the incumbent lacks). Treat evaluation time and
program complexity as secondary constraints and as tie-breakers between
similarly credible gains.

Every strategy must be self-contained and immediately implementable from the
program and context above. Do not depend on an offline solve, an unavailable
certificate or lookup table, fabricated constants/data, external tools, or a
future optimization step. State how the implementation preserves the problem's
hard validity constraints. Reject your own idea if you cannot provide all of the
information the implementer needs. An "intended", "designed", "about", or
hypothetical objective value is not a gain estimate unless it is derived from
the explicit constants or local transformation supplied in the strategy.

{gain_check}must be compared with the same named incumbent metric, never with
`combined_score`. End with a one-sentence expected-gain check that either (a)
derives a predicted `combined_score` from explicit constants and shows that it
strictly exceeds the current ranking score, or (b), when an exact prediction is
impossible, names the precise new degree of freedom or local transformation,
explains why the incumbent cannot realize it, and gives a falsifiable evaluator
threshold. An accept-only-if-better guard, incumbent fallback, feasibility test,
or promise to run an optimizer is not evidence of gain and cannot satisfy this
check. Preserve a valid incumbent fallback, but do not present the fallback as
the proposed improvement. Do not expand the check into a proof.

{exploration_scope}early, but once the incumbent is strong, the portfolio must preserve both a
local, verifiable refinement path and a distinct route that can escape the
incumbent's structural ceiling.

For continuous numerical problems, use an auxiliary objective variable,
deterministic multistart, active-set continuation, or analytic derivatives only
when that machinery is the strategy's actual gain mechanism. Do not fill the
portfolio with several variants of the same constrained optimizer. Prefer a
bounded replacement of the incumbent's weakest phase over another complete search
pipeline layered on top of it. Ignore this paragraph when the task is not a
continuous numerical optimization problem.

Respond with the following block once per strategy, for exactly {count} block(s):

<STRATEGY_TITLE>[mechanism-family] a short descriptive name</STRATEGY_TITLE>
<STRATEGY>
Mechanism family: ...
Replacement scope: ...
Compute budget: ...
Gain mechanism: ...
the self-contained plan, ending with its one-sentence expected-gain check
</STRATEGY>
{reference_contract}
"""


def implementation_instruction(
    strategy: Strategy,
    variant: int,
    total: int,
    *,
    scouting: bool = False,
    sequential: bool = False,
    prompt_addendum: str = "",
    guidance_profile: str = "default",
) -> str:
    """Appended to the normal generation prompt for the cheap model."""
    validate_guidance_profile(guidance_profile)
    txn_profile = guidance_profile == "txn_scheduling"
    if txn_profile:
        parent_adaptation = """
If a promoted reference changed the parent's internal structure, map the plan's
mechanism to the current code rather than searching for obsolete replacement
points. Preserve public interfaces and the allowed edit boundaries.
"""
    else:
        parent_adaptation = ""
    if scouting:
        variant_line = """

This is a scouting implementation used to rank this strategy against other
strategies from the same parent. Build the smallest executable version that
actually exercises the strategy's distinctive gain mechanism once. Do not build
the full expensive portfolio of variants, and do not return a fallback-only
implementation."""
    elif total <= 1:
        variant_line = ""
    elif sequential:
        variant_line = f"""

This is sequential implementation attempt {variant} of at most {total} from the
same parent. Use any evaluated-attempt feedback below to repair errors or resolve
an unsuccessful design choice. Keep the strategy's mechanism and bounded compute
contract; do not repeat an unsuccessful patch or merely change a random seed."""
    else:
        variant_line = f"""

This is independent implementation sample {variant} of {total}. Every sample
receives the same implementation brief: implement the strategy faithfully,
choose the strongest reliable resolution of choices the plan leaves open, and
keep all added work bounded. Do not manufacture diversity merely by changing
random seeds, tolerances, acceptance thresholds, or maximum iterations, and do
not substitute a different strategy."""
    reference_text = ""
    if strategy.reference_solution:
        score_text = (
            f" Its evaluated score is {strategy.reference_score:.12g}."
            if strategy.reference_score is not None
            else ""
        )
        reference_text = f"""

# Strong model reference implementation for this strategy

The strong model supplied the complete executable program below as an implementation of
this strategy.{score_text} Treat it as concrete implementation evidence and a
possible warm start, not as a score prediction or a requirement to copy it.
Return the strongest implementation you can after comparing it with the current
parent. If you build on this reference, return a complete program rather than
SEARCH/REPLACE edits, because edits are applied to the current parent, not to
the reference.

```
{strategy.reference_solution}
```
"""
    addendum = prompt_addendum.strip() if txn_profile else ""
    addendum_section = (
        f"\n\n# Task-specific implementation constraints\n\n{addendum}" if addendum else ""
    )
    addendum_section = "\n" + addendum_section if txn_profile else ""
    return f"""

# Strategy to implement: {strategy.title}

{strategy.plan}{addendum_section}
{reference_text}
{parent_adaptation}
Recent implementation failures for this strategy: {" | ".join(strategy.failure_feedback) or "none recorded"}.
Use this evidence to avoid repeating an invalid implementation. If the current
parent already implements the proposed mechanism, refine that working mechanism
instead of constructing another copy. Verify dimensions, mutability, public API
calls and the actual return path before submitting the program.
Use explicit dtypes and writable copies where local array mutation is required.

Implement exactly this strategy. Do not substitute your own idea, and do not fall
back to a small tweak of the current program: the plan above is what is being
tested. Identify the parent mechanism named by the plan, replace it, and remove
its superseded path when safe; do not append a second end-to-end optimizer or
search pipeline. Make sure the new mechanism executes on the normal return path.
Preserve a valid incumbent fallback, but a fallback or accept-only-if-better guard
does not count as implementing the strategy. For numerical optimization, do not
use more starts than the parent normally runs unless the plan explicitly requires
them, and always impose hard per-start iteration limits.

If you return SEARCH/REPLACE edits, copy every SEARCH block byte-for-byte from the
current parent program and verify that it occurs there exactly once; prefer
replacing one complete small function or block over reconstructing several
fragile line fragments. An unmatched SEARCH block produces no candidate. If the
plan is ambiguous, pick the reading that maximises the objective while respecting
the plan's mechanism and compute budget, then follow it through.{variant_line}
"""
