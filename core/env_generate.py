import os
import json
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from tqdm import tqdm

from core.config import PipelineConfig
from core.env import (
    load_specs, load_schemas, load_existing_results,
    process_platform, append_result,
)
from utils.llm import LLMClient


def run(args: PipelineConfig) -> None:
    specs = load_specs(args.specs_output)
    schemas = load_schemas(args.schemas_output)
    existing = load_existing_results(args.envs_output)

    stale: set[str] = set()
    failed: set[str] = set()
    pending: list[str] = []
    for name in specs:
        if name not in schemas:
            continue
        current_fp = specs[name].get("ops_fingerprint", "")
        if name not in existing:
            pending.append(name)
        elif existing[name].get("status") != "ok":
            failed.add(name)
            pending.append(name)
        elif existing[name].get("ops_fingerprint") != current_fp:
            stale.add(name)
            pending.append(name)

    to_remove = stale | failed
    if to_remove and os.path.exists(args.envs_output):
        if stale:
            logger.info(f"Re-generating {len(stale)} servers with changed spec: {stale}")
        if failed:
            logger.info(f"Re-generating {len(failed)} previously failed servers: {failed}")
        kept_lines: list[str] = []
        with open(args.envs_output, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                    if item.get("name") not in to_remove:
                        kept_lines.append(stripped)
                except json.JSONDecodeError:
                    kept_lines.append(stripped)
        with open(args.envs_output, "w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line + "\n")

    logger.info(f"Already done: {len(existing) - len(to_remove)}, Pending: {len(pending)}")

    if not pending:
        logger.success("All platforms already processed.")
        return

    # Shuffle so retries don't always front-load the same (hard) platforms.
    random.shuffle(pending)

    client = LLMClient.from_config(args, extra_llm_params=args.think_llm_params)
    if args.think_llm_params:
        logger.info(f"Reason mode ON for env generation: {args.think_llm_params}")
    else:
        logger.info("Reason mode OFF for env generation (think_llm_params empty)")

    success = failed_count = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                process_platform,
                client, args.gen_model,
                name,
                specs[name],
                schemas[name],
                args.servers_dir,
                args.env_max_retries,
                args.max_completion_tokens,
                args.max_results,
                None,
                None,
            ): name
            for name in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating servers"):
            name = futures[future]
            result = future.result()
            if result is not None:
                append_result(args.envs_output, result)
                if result["status"] == "ok":
                    success += 1
                else:
                    failed_count += 1

    logger.success(f"Done. {success} OK, {failed_count} failed out of {len(pending)} platforms.")
    logger.success(f"Servers saved to: {args.servers_dir}/")
    logger.success(f"Index saved to: {args.envs_output}")


def main() -> None:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Generate FastAPI servers for each platform")
    parser.add_argument("--specs_output", type=str, default=defaults.specs_output)
    parser.add_argument("--schemas_output", type=str, default=defaults.schemas_output)
    parser.add_argument("--envs_output", type=str, default=defaults.envs_output)
    parser.add_argument("--servers_dir", type=str, default=defaults.servers_dir)
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
