# FrugalEvo: Towards Cost-Aware LLM-Guided Program Evolution
[![Paper](https://img.shields.io/badge/Paper-Arxiv-darkred.svg)](https://arxiv.org/pdf/)
[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-2EA44F)](https://opensource.org/license/apache-2.0)


**FrugalEvo** is a cost-aware evolutionary framework for computational optimization. It uses a stronger, higher-cost LLM to explore solution strategies, and a cheaper LLM to implement them and iteratively refines the resulting code.

<br>
<div align="center">
  <img src="assets/frugalevo_overview.png" width="90%" ></img>
  <br>
  <em>
      Figure 1: The FrugalEvo framework.
  </em>
</div>
<br>



## 📦 Installation

This repository requires Python 3.10 to 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/chchenhui/frugalevo.git
cd frugalevo
uv sync
```
Install ```frugalevo``` as a package:
```shell
uv pip install -e .
```

Please set up your API keys before starting the experiments:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
export GOOGLE_API_KEY="your-google-api-key-here"
export OPENROUTER_API_KEY="your-openrouter-api-key-here"
```

### 🛠️ External backends
Install with `uv sync --extra external`, then use the corresponding flag:

| Backend | Flag | Source |
|:---|:---|:---|
| **OpenEvolve** | `--search openevolve` | [codelion/openevolve](https://github.com/codelion/openevolve) |
| **ShinkaEvolve** | `--search shinkaevolve` | [SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) (manual install) |

<details>
<summary>ShinkaEvolve manual install</summary>

```bash
git clone --depth 1 https://github.com/SakanaAI/ShinkaEvolve.git external_repos/ShinkaEvolve
uv pip install -e external_repos/ShinkaEvolve
```

</details>

## 🚀 Quick Start

If you want to try FrugalEvo, run the circle packing benchmark with the commands below:
```bash
# Try the circle packing benchmark
uv sync --extra math
uv run skydiscover-run benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  --config configs/budget2_gpt56_circle_packing/circle_packing/frugalevo.yaml \
  --search frugalevo
```

Alternatively, after installing the math dependencies and setting your API key, use the helper script:

```bash
bash scripts/run_frugalevo.sh

# Override the iteration limit and output directory
bash scripts/run_frugalevo.sh --iterations 100 --output runs/my_circle_packing

# Run your own optimization task
bash scripts/run_frugalevo.sh \
  --initial-program path/to/initial_program.py \
  --evaluator path/to/evaluator.py \
  --config path/to/frugalevo.yaml
```

```scripts/run_frugalevo.sh``` defaults to the circle packing configuration above. Customize the models, API endpoint, and budget in the YAML configuration for your provider and task. Run `bash scripts/run_frugalevo.sh --help` for usage details.


### 📖 Command-line Options

<details>
<summary><b>CLI flags</b></summary>

```
uv run skydiscover-run [INITIAL_PROGRAM] EVALUATOR [options]
```

| Flag | Description |
|:---|:---|
| `-c, --config FILE` | Config YAML |
| `-i, --iterations N` | Number of iterations |
| `-m, --model MODEL` | LLM model (overrides config) |
| `-s, --search TYPE` | Search algorithm |
| `-o, --output DIR` | Output directory |
| `--api-base URL` | Override LLM API endpoint |
| `--checkpoint DIR` | Resume from checkpoint |
| `--agentic` | Enable agentic mode (LLM can read your files) |
| `-l, --log-level LEVEL` | DEBUG, INFO, WARNING, or ERROR |

</details>

## 🔗 Acknowledgement
FrugalEvo is adapted from the [SkyDiscover](https://github.com/skydiscover-ai/skydiscover) framework, which is also inspired by [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) and incorporates useful code components from open-source efforts such as [OpenEvolve](https://github.com/codelion/openevolve). Its interface is compatible with the [optimize_anything](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/) API.



## 📬 Contact Us
If you have any questions or feedback, please reach out to:
[hui.chen@nus.edu.sg](mailto:hui.chen@nus.edu.sg)
