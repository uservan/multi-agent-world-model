import argparse
from contextlib import nullcontext

from loguru import logger

from core.config import PipelineConfig
from core.env import (
    load_specs, load_schemas, load_existing_results,
    process_platform, _read_server_code,
)
from utils.fix_locks import PlatformLocks
from utils.llm import LLMClient
from utils.suggestions import load_env_suggestions


def run(args: PipelineConfig, task_file: str, *, platform_locks: PlatformLocks | None = None) -> None:
    specs = load_specs(args.specs_output)
    schemas = load_schemas(args.schemas_output)
    existing = load_existing_results(args.envs_output)
    platform_suggestions = load_env_suggestions(task_file)

    pending: list[str] = [
        name for name in specs
        if name in schemas
        and name in existing
        and existing[name].get("status") == "ok"
        and name in platform_suggestions
    ]

    logger.info(f"Servers needing revision: {len(pending)}")
    if not pending:
        logger.success("No servers need revision.")
        return

    client = LLMClient.from_config(args, extra_llm_params=args.think_llm_params)
    if args.think_llm_params:
        logger.info(f"Reason mode ON for env fix: {args.think_llm_params}")

    success = failed_count = 0
    for name in pending:
        ctx = platform_locks.get(name) if platform_locks else nullcontext()
        with ctx:
            result = process_platform(
                client, args.gen_model,
                name,
                specs[name],
                schemas[name],
                args.servers_dir,
                args.env_max_retries,
                args.max_completion_tokens,
                args.max_results,
                _read_server_code(existing[name].get("server_path", "")),
                platform_suggestions[name],
            )
        if result is not None and result["status"] == "ok":
            success += 1
        else:
            failed_count += 1

    logger.success(f"Done. {success} OK, {failed_count} failed out of {len(pending)} revisions.")
    logger.success(f"Servers saved to: {args.servers_dir}/")


def main() -> None:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Revise FastAPI servers from env suggestions")
    parser.add_argument("--specs_output", type=str, default=defaults.specs_output)
    parser.add_argument("--schemas_output", type=str, default=defaults.schemas_output)
    parser.add_argument("--envs_output", type=str, default=defaults.envs_output)
    parser.add_argument("--servers_dir", type=str, default=defaults.servers_dir)
    parser.add_argument("--suggestions_output", type=str, default=defaults.suggestions_output)
    parser.add_argument("--gen_model", type=str, default=defaults.gen_model)
    parser.add_argument("--api_key", type=str, default=defaults.api_key)
    parser.add_argument("--base_url", type=str, default=defaults.base_url)
    parser.add_argument("--concurrency", type=int, default=defaults.concurrency)
    parser.add_argument("--env_max_retries", type=int, default=defaults.env_max_retries)
    parsed = parser.parse_args()

    cfg = PipelineConfig()
    for k, v in vars(parsed).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    run(cfg)


if __name__ == "__main__":
    main()
