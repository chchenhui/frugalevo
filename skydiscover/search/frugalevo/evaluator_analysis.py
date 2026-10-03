"""Evaluator- and package-aware meta-analysis for FrugalEvo.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from skydiscover.llm.llm_pool import LLMPool

logger = logging.getLogger(__name__)


META_ANALYSIS_SYSTEM_PROMPT = """\
You are a domain expert designing search guidance for a program-evolution system.

Read the problem, evaluator, and available-package list before proposing methods.
Your guidance must:

1. Identify exactly what the evaluator rewards, its hard validity constraints,
   and anything it does not reward.
2. Read the evaluator's scoring function (for example ``combined_score``) and
   account for ALL components of the objective. When the objective combines
   multiple sub-scores, state their weights explicitly and the tradeoffs
   between them: which terms are already near their maximum, which are far
   from it, and which pairs move in opposite directions. Say so plainly when a
   weighted term rewards something other than the obvious domain goal -- the
   highest-scoring program is then not the one that best solves the nominal
   problem, and guidance that optimizes the nominal problem will stall.
3. Determine whether the evaluator gives the submitted program the complete
   input at once. If it does, explicitly consider offline/batch algorithms; do
   not assume the solution must be causal, streaming, or online.
4. Identify the standard toolboxes for the problem domain and name concrete,
   available library functions (including exact method names for wrappers).
5. Prefer mature library implementations over hand-written substitutes when
   they implement the required operation and comply with the evaluator.
6. Include complementary method families when applicable: causal/forward and
   bidirectional/zero-phase, online and batch, local and global, or deterministic
   and stochastic.
7. Judge ideas by evaluator-visible gain mechanisms. Do not suggest cosmetic
   changes or work that only improves an unscored property.
8. Separate genuinely different approaches from refinements of one approach.
   Do not fill the portfolio with variants of one textbook family; the
   weight/tradeoff analysis from item 2, not domain convention, decides which
   families are worth proposing.

Return exactly these three sections. The objective section is prose or bullets;
the other two are concise, actionable bullets:

### OBJECTIVE
- the scored terms, their weights, and what each one actually rewards
- which terms are near their maximum, which are far from it, and which pairs
  conflict; name the term that is the largest remaining lever
- any term that rewards something other than the obvious domain goal

### EXPLORATION (DIVERGE)
EXAMPLES OF DIFFERENT approaches:
- [mechanism family] approach; relevant library.module: function1, function2

### EXPLOITATION (REFINE)
EXAMPLES OF REFINEMENT approaches:
- refinement; concrete parameter, budget, tolerance, initialization, or polish

Do not output program code. Do not merely list packages: connect every named
function to an evaluator-visible mechanism and the constraints it preserves.
"""


def get_available_packages(problem_dir: Optional[str] = None) -> List[str]:
    """Return direct dependencies, prioritizing the benchmark requirements."""
    repo_root = Path(__file__).resolve().parents[3]
    candidates = []
    if problem_dir is not None:
        problem_path = Path(problem_dir)
        candidates.extend(
            [
                problem_path / "requirements.txt",
                problem_path / "evaluator" / "requirements.txt",
            ]
        )
    candidates.append(repo_root / "requirements.txt")

    for requirements_path in candidates:
        if not requirements_path.exists():
            continue
        try:
            packages = [
                line
                for raw_line in requirements_path.read_text().splitlines()
                if (line := raw_line.strip()) and not line.startswith(("#", "-e", "--"))
            ]
            if packages:
                logger.debug("Read %d packages from %s", len(packages), requirements_path)
                return packages
        except OSError as exc:
            logger.warning("Could not read %s: %s", requirements_path, exc)

    pyproject_path = repo_root / "pyproject.toml"
    try:
        import tomllib

        dependencies = (
            tomllib.loads(pyproject_path.read_text()).get("project", {}).get("dependencies", [])
        )
        if dependencies:
            return [str(dependency) for dependency in dependencies]
    except (OSError, ValueError) as exc:
        logger.warning("Could not read dependencies from %s: %s", pyproject_path, exc)

    try:
        result = subprocess.run(
            ["uv", "pip", "list", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return [f"{package['name']}=={package['version']}" for package in json.loads(result.stdout)]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        logger.warning("Could not inspect installed packages: %s", exc)
        return []


def _build_meta_analysis_prompt(
    system_message: str,
    evaluator_code: str,
    problem_dir: Optional[str],
) -> str:
    packages = get_available_packages(problem_dir)
    package_text = "\n".join(packages) if packages else "No packages discovered"
    return f"""Analyze this task and generate all three requested guidance sections.

## Problem description
```
{system_message}
```

## Available packages
```
{package_text}
```

## Evaluator code
```python
{evaluator_code}
```

Base every recommendation on the evaluator-visible objective and the packages
that are actually available. Explicitly report whether complete input is
available to the submitted program and what that permits.

Decompose the scoring function numerically before recommending anything: list
each scored term with its weight, and say which terms conflict. If the
weighting makes the best-scoring program differ from the best solution to the
nominal domain problem, state that outright -- it changes which method
families are worth proposing.
"""


def _extract_sections(response: str) -> Tuple[str, str, str]:
    """Split objective, exploration and refinement guidance; preserve pre-heading text as objective analysis."""
    objective: List[str] = []
    exploration: List[str] = []
    refinement: List[str] = []
    current: Optional[List[str]] = objective

    for line in response.splitlines():
        heading = line.upper().strip()
        if "### OBJECTIVE" in heading:
            current = objective
            continue
        if "### EXPLORATION" in heading or "EXPLORATION (DIVERGE" in heading:
            current = exploration
            continue
        if "### EXPLOITATION" in heading or "EXPLOITATION (REFINE" in heading:
            current = refinement
            continue
        if current is not None and line.strip() != "```":
            current.append(line)

    objective_text = "\n".join(objective).strip()
    explore_text = "\n".join(exploration).strip()
    refine_text = "\n".join(refinement).strip()
    if not explore_text and not refine_text:
        # Use an unstructured response as exploration guidance rather than discarding it.
        explore_text = response.strip()
        objective_text = ""
    return objective_text, explore_text, refine_text


async def generate_evaluator_package_guidance(
    system_message: str,
    evaluator_code: str,
    problem_dir: Optional[str] = None,
    llm_pool: Optional[LLMPool] = None,
) -> Tuple[str, str, str]:
    """Generate EE-native objective/exploration/refinement guidance."""
    if llm_pool is None:
        raise ValueError("llm_pool is required to generate evaluator/package guidance")

    result = await llm_pool.generate(
        system_message=META_ANALYSIS_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": _build_meta_analysis_prompt(
                    system_message,
                    evaluator_code,
                    problem_dir,
                ),
            }
        ],
    )
    return _extract_sections(result.text or "")
