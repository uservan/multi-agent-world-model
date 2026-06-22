# Multi-Agent World Model

Data generation pipeline for multi-agent RL training.

## Quick Start

**1. Set up the environment**

```bash
uv venv        # create venv (skip if .venv already exists)
uv sync        # install dependencies
```

**2. Set API key and model**

Edit `export.sh` with your API key and model name, then source it:

```bash
source export.sh
```

**3. Adjust parameters (optional)**

Pipeline parameters have defaults in `core/config.py` (`PipelineConfig`). Commonly tuned:

| Parameter | Default | Description |
|---|---|---|
| `num_structures` | 50 | Number of task structures per budget |
| `tasks_per_structure` | 25 | Number of tasks per structure |
| `budget_list` | 5\~19 | Budget values to use |

Override via CLI without touching the code:

```bash
uv run python main.py --num_structures 2 --tasks_per_structure 1 --budget_list 5 6
```

**4. Run**

```bash
uv run python main.py
```

Outputs are written to `outputs/`. Files needed for RL training:

```
outputs/
├── tasks.jsonl     # training data
├── servers/        # platform FastAPI server code
└── databases/      # seed databases
```

The pipeline supports resuming from interruptions — re-running will skip already completed steps.
