import os
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:

    # ── Directories ────────────────────────────────────────────────────────────
    generated_dir: str = "outputs/generated"
    verified_dir: str = "outputs/generated/verified"

    # ── Common LLM ─────────────────────────────────────────────────────────────
    # model: cheap model used for validation, schema, spec, etc.
    model: str = field(default_factory=lambda: os.environ.get("AWM_MODEL", "gpt-4o"))
    # gen_model: strong model used for creative generation (task.py, data.py); falls back to model
    gen_model: str = field(default_factory=lambda: os.environ.get("AWM_GEN_MODEL") or os.environ.get("AWM_MODEL", "gpt-4o"))
    api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY") or None)
    base_url: str | None = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL") or None)
    aws_region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    concurrency: int = 16
    max_retries: int = 3

    # ── Production: scenario.py ────────────────────────────────────────────────
    scenario_input: str = "outputs/scenario_platform.yml"

    # ── Production: task.py ────────────────────────────────────────────────────
    embed_model: str = field(default_factory=lambda: os.environ.get("AWM_EMBED_MODEL", "text-embedding-3-large"))
    embed_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY") or None)
    budget_list: list[int] = field(default_factory=lambda: [5, 13])  # list(range(5, 20))
    num_structures: int = 10  # 50
    tasks_per_structure: int = 4  # 20
    max_scenes: int = 4
    max_platforms_per_scene: int = 3   # max platform slots per scene
    max_agents_per_platform: int = 3   # max sub-agents per platform slot
    max_completion_tokens: int = 1024 * 64
    embed_threshold: float = 0.80
    validity_threshold: int = 3
    min_steps_per_subagent: int = 5
    max_steps_per_subagent: int = 20
    max_attempts_multiplier: int = 1
    max_task_attempts: int = 5
    max_structure_sample_attempts: int = 2000  # max random draws to find unique structures per budget

    # ── Production: spec.py ────────────────────────────────────────────────────
    max_results: int = 5

    # ── Production: env.py ─────────────────────────────────────────────────────
    env_max_retries: int = 4           # server startup needs more retries

    # ── Production: data.py ────────────────────────────────────────────────────
    num_distractors: int = 10          # kept for backward compat
    distractor_high: int = 30          # main search tables
    distractor_medium: int = 10        # secondary tables
    distractor_low: int = 3            # supporting tables (users, settings)

    # ── Fix pipeline ───────────────────────────────────────────────────────────
    agent_run_iterations_multiplier: int = 3  # max_iterations = len(sub_ops) * this
    fix_batch_size: int = 4
    fix_concurrency: int = 8           # parallel workers for env_fix server revision

    def __post_init__(self) -> None:
        self.agent_run_max_iterations: int = self.max_steps_per_subagent * self.agent_run_iterations_multiplier

        g = self.generated_dir
        v = self.verified_dir

        # ── Production paths ───────────────────────────────────────────────────
        # scenario.py
        self.scenario_output: str = f"outputs/platforms.jsonl"

        # task.py
        self.tasks_output: str = f"{g}/tasks.jsonl"
        self.task_plans_output: str = f"{g}/task_plans.jsonl"
        self.tasks_final_output: str = f"{g}/tasks_final.jsonl"
        self.embeddings_output: str = f"{g}/embeddings.jsonl"

        # schema.py
        self.schemas_output: str = f"{g}/schemas.jsonl"

        # spec.py
        self.specs_output: str = f"{g}/specs.jsonl"

        # env.py
        self.envs_output: str = f"{g}/envs.jsonl"
        self.servers_dir: str = f"{g}/servers"

        # data.py
        self.data_manifest: str = f"{g}/data_manifest.jsonl"
        self.databases_dir: str = f"{g}/databases"
        self.data_records: str = f"{g}/data_records.jsonl"
        self.task_repairs_output: str = f"{g}/task_repairs.jsonl"
        self.task_goals_output: str = f"{g}/task_goals.jsonl"

        # verifier_gen.py
        self.verifier_gen_output: str = f"{g}/verifiers_gen.jsonl"

        # verifier.py
        self.verifiers_output: str = f"{g}/verifiers.jsonl"

        # ── Fix paths ──────────────────────────────────────────────────────────
        self.verified_suggestions_dir: str = f"{v}/suggestions"
        self.verified_platforms_output: str = f"{v}/verified_platforms.jsonl"
        self.task_supplements_output: str = f"{v}/task_supplements.jsonl"

        # ── Finalize: consolidated eval/training dataset ───────────────────────
        self.task_final_output: str = f"{v}/task_final.jsonl"
