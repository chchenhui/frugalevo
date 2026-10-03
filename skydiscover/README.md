# SkyDiscover

SkyDiscover is an iterative LLM-driven discovery engine. Each iteration runs a
four-step loop:

```
sample → prompt → generate → evaluate → add
  ↑                                       │
  └───────────────────────────────────────┘
```

1. **Sample** — the search algorithm (`search/`) picks a parent solution and
   any relevant context solutions from the database.
2. **Prompt** — the context builder (`context_builder/`) turns the parent solution,
   relevant context solutions (if any), and problem spec into system + user messages.
3. **Generate** — the LLM (`llm/`) produces a candidate solution (code, text,
   or image).
4. **Evaluate** — the evaluator (`evaluation/`) scores the candidate and
   returns metrics.
5. **Add** — the scored candidate is stored back in the database, closing the
   loop.

The `DiscoveryController` (`search/default_discovery_controller.py`) orchestrates
this loop. Search algorithms that need custom orchestration (e.g. co-evolution)
subclass it and override `run_discovery()`.

## Components

| Component | Subfolder | What it does | Extend by |
|:---|:---|:---|:---|
| **Context Builder** | `context_builder/` | Assembles LLM prompts from the problem spec, prior solutions, and feedback | Subclass `ContextBuilder` ([README](context_builder/README.md)) |
| **Solution Generator** | `llm/` | Produces candidates via LLM calls, with optional tool use | Subclass `LLMInterface` |
| **Evaluator** | `evaluation/` | Scores candidates and logs metadata back into the solution database | Provide an `evaluate.py` script |
| **Solution Selector** | `search/` | Maintains the solution database and picks parents for the next iteration | Subclass `ProgramDatabase` ([README](search/README.md)) |

## Additional subfolders

| Subfolder | What it does |
|:---|:---|
| `extras/` | External backends (OpenEvolve, GEPA, ShinkaEvolve) and the live monitor dashboard |
| `utils/` | Shared helpers — code parsing, metrics, formatting, async utilities, repo mapping |

## Entry points

| Entry point | Use case |
|:---|:---|
| `api.py` | Python API — `run_discovery()`, `discover_solution()` |
| `cli.py` | CLI — `skydiscover-run` |
| `runner.py` | Setup and run (used by both API and CLI) |
| `config.py` | Configuration loading and overrides |
