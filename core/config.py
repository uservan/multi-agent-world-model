import os
import dataclasses
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:

    # ── Directories ────────────────────────────────────────────────────────────
    generated_dir: str = "outputs/eval_gen"
    verified_dir: str = "outputs/eval_gen/verified"

    # ── Common LLM ─────────────────────────────────────────────────────────────
    # model: cheap model used for validation, schema, spec, etc.
    model: str = field(default_factory=lambda: os.environ.get("AWM_MODEL", "gpt-4o"))
    # gen_model: strong model used for creative generation (task.py, data.py); falls back to model
    gen_model: str = field(default_factory=lambda: os.environ.get("AWM_GEN_MODEL") or os.environ.get("AWM_MODEL", "gpt-4o"))
    api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY") or None)
    base_url: str | None = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL") or None)
    aws_region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    # LLM sampling/extra params passed verbatim to the API (temperature, top_p, and model-specific
    # extras like thinking control). Different models need different keys, so this is a free-form dict:
    #   Kimi-K2.6 instant: {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
    #   GLM-5.2 no-think:  {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    # Empty default = Kimi thinking mode (its default). Edit here / per-run, no code change.
    llm_params: dict = field(default_factory=dict)
    # Thinking-mode LLM params for the env generate/fix steps ONLY (the heaviest reasoning in the
    # pipeline). Merged OVER llm_params for those two steps; {} = thinking off, behavior unchanged.
    # Free-form like llm_params — each model family has its own switch:
    #   Bedrock Claude (Opus 4.8 / Sonnet 5): {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}
    #   GLM-style OpenAI path:                {"extra_body": {"chat_template_kwargs": {"enable_thinking": true}}}
    think_llm_params: dict = field(default_factory=dict)
    # Floor on max_completion_tokens so a small per-call cap isn't eaten by reasoning → empty content.
    min_completion_tokens: int | None = 1024 * 64
    concurrency: int = 16
    max_retries: int = 3

    # ── Production: scenario.py ────────────────────────────────────────────────
    scenario_input: str = "outputs/scenario_platform_eval.yml"

    # ── Production: task.py ────────────────────────────────────────────────────
    embed_model: str = field(default_factory=lambda: os.environ.get("AWM_EMBED_MODEL", "text-embedding-3-large"))
    embed_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY") or None)
    budget_list: list[int] = field(default_factory=lambda: [3,13,5,7,9])  # list(range(5, 20))
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
    distractor_high: int = 50          # main search tables
    distractor_medium: int = 10        # secondary tables
    distractor_low: int = 3            # supporting tables (users, settings)

    # ── Fix pipeline ───────────────────────────────────────────────────────────
    agent_run_iterations_multiplier: int = 7  # max_iterations = len(sub_ops) * this
                                              # (action-only solving needs ~3 turns/step: discover + call + confirm)
    fix_batch_size: int = 16
    fix_concurrency: int = 16           # parallel workers for env_fix server revision

    def __post_init__(self) -> None:
        self.agent_run_max_iterations: int = self.max_steps_per_subagent * self.agent_run_iterations_multiplier

        g = self.generated_dir
        v = self.verified_dir

        # ── Production paths ───────────────────────────────────────────────────
        # scenario.py
        self.scenario_output: str = f"{g}/platforms.jsonl"

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

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Build a config straight from a yml file: read it, set whatever keys it lists,
        leave the rest at defaults. No init/resume/locking — just load and go.
        Set `generated_dir` to relocate all outputs; individual output paths are derived
        from it (but may be overridden explicitly in the yml too)."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cfg = cls()
        field_names = {f.name for f in dataclasses.fields(cls)}
        for k in data:
            if not hasattr(cfg, k):
                print(f"[config] WARNING: unknown key '{k}' in {path} — ignored")
        # 1) declared fields first (incl. generated_dir / verified_dir)
        for k, v in data.items():
            if k in field_names:
                setattr(cfg, k, v)
        # 2) re-derive output paths from the (possibly overridden) dirs
        cfg.__post_init__()
        # 3) explicit derived-path overrides (e.g. tasks_output) win over derivation
        for k, v in data.items():
            if k not in field_names and hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
