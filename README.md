- ps -ef | grep "main.py --mode generate" | grep -v grep || echo "全部停止"
- source export.sh && nohup python main.py --mode generate --config multi-agent-world-model/core/config/gen_claude.yml > gen_$(date +%m%d_%H%M).log 2>&1 & echo "PID: $!"

# Multi-Agent World Model

A pipeline that **generates** multi-agent RL training data and **evaluates** agents on it.

The repo has two independent parts:

- **Part 1 — Data Production** (`main.py`): generate tasks, platform servers, seed databases, and verifiers.
- **Part 2 — Evaluation** (`eval_main.py`): run single- or multi-agent setups against the generated tasks and score them.

## Setup

```bash
uv venv        # create venv (skip if .venv already exists)
uv sync        # install dependencies
```

Edit `export.sh` with your API key and model name, then source it:

```bash
source export.sh
```

---

# Part 1 — Data Production

Generates the training data under `outputs/`.

## Parameters

Defaults live in `core/config.py` (`PipelineConfig`). Commonly tuned:

| Parameter | Default | Description |
|---|---|---|
| `num_structures` | 50 | Number of task structures per budget |
| `tasks_per_structure` | 25 | Number of tasks per structure |
| `budget_list` | 5\~19 | Budget values to use |

Override via CLI without touching the code:

```bash
uv run python main.py --mode generate --num_structures 2 --tasks_per_structure 1 --budget_list 5 6
```

## Modes

`main.py` runs in one of four `--mode` values (default `finalize`):

| Mode | Command | What it does |
|---|---|---|
| `generate` | `uv run python main.py --mode generate` | Full generation pipeline: scenario → schema → spec → env (FastAPI servers) → data (seed DBs) → verifier gen |
| `fix` | `uv run python main.py --mode fix` | Concurrent revision: verify tasks and auto-fix goals/data/verifiers/envs that fail |
| `debug` | `uv run python main.py --mode debug` | Same fix steps but sequential, one task at a time (easier to debug) |
| `finalize` | `uv run python main.py --mode finalize` | Consolidate fully-verified tasks into `task_final.jsonl` for training/eval |

Typical end-to-end run:

```bash
uv run python main.py --mode generate    # 1. produce raw tasks + envs + DBs + verifiers
uv run python main.py --mode fix         # 2. verify and repair until clean
uv run python main.py --mode finalize    # 3. emit task_final.jsonl
```

The pipeline supports resuming from interruptions — re-running skips already completed steps.

## Outputs

```
outputs/
├── generated/
│   ├── verified/
│   │   └── task_final.jsonl   # final training data (consumed by eval)
│   ├── envs.jsonl             # platform FastAPI server code
│   └── databases/             # seed databases
└── platforms.jsonl            # platform definitions
```

---

# Part 2 — Evaluation

Runs an agent against `task_final.jsonl` and scores it. Driven by `eval_main.py` and a config template.

## Eval modes

Three ready-made templates in `eval/config/`:

| Template | Mode | Description |
|---|---|---|
| `single.yml` | single | One model does everything |
| `multi_same.yml` | multi | Orchestrator + sub-agents, **same** model for both |
| `multi_diff.yml` | multi | Orchestrator + sub-agents, **different** models (e.g. strong planner + cheap workers) |

Key config fields (see the yml comments for the full list):

| Field | Description |
|---|---|
| `orch_model` / `sub_model` | Orchestrator / sub-agent model names |
| `*_api_key` / `*_base_url` | Optional; fall back to `OPENAI_API_KEY` / `OPENAI_BASE_URL` |
| `*_input_cost` / `*_output_cost` | Per-token cost (editable any time; `null` skips cost accounting) |
| `n` / `base_seed` / `temperature` | Repetition: `n` runs per task, each a different seed |
| `max_concurrent` / `max_queue` | Multi-agent concurrency (sub-agents running / total allowed) |
| `task_final`, `platforms_input`, `envs_input`, `databases_dir` | Data sources from Part 1 |

## Running an eval

**Start a new run** — creates a result folder, copies the yml in, and locks the frozen (model/run) fields:

```bash
uv run python eval_main.py --init eval/config/single.yml
uv run python eval_main.py --init eval/config/multi_same.yml
uv run python eval_main.py --init eval/config/multi_diff.yml
```

**Resume an existing run** — re-runs only missing tasks, then aggregates:

```bash
uv run python eval_main.py --config outputs/eval/<run_folder>
```

Completed trajectories are cached, so resuming is cheap. To refresh cost numbers, edit the `*_cost` fields in the run's `config.yml` and run `--config` again — finished tasks are skipped and only `eval.json` is recomputed.

> After `--init`, only the cost fields and data paths in the result's `config.yml` may be edited; model/run fields are locked to keep results reproducible.
