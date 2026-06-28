import os
import json
import queue
import threading
import argparse
from pathlib import Path

from loguru import logger

from core.config import PipelineConfig
from utils.fix_locks import FixLocks
import core.scenario as scenario_step
import core.task as task_step
import core.schema as schema_step
import core.spec as spec_step
import core.env_generate as env_generate_step
import core.env_fix as env_fix_step
import core.data as data_step
import core.data_fix as data_fix_step
import core.goal_fix as goal_fix_step
import core.verifier_gen as verifier_gen_step
import core.verifier_fix as verifier_fix_step
import core.verifier_step as verifier_step
import core.finalize as finalize_step


# ── Cleanup ────────────────────────────────────────────────────────────────────

def cleanup_failed_tasks(args: PipelineConfig) -> None:
    """Remove tasks whose verifiers have any failed platform from tasks.jsonl in-place."""
    if not os.path.exists(args.verifiers_output):
        logger.warning("Verifiers file not found, skipping cleanup.")
        return
    if not os.path.exists(args.tasks_output):
        logger.warning("Tasks file not found, skipping cleanup.")
        return

    failed_task_ids: set[str] = set()
    ok_task_ids: set[str] = set()
    with open(args.verifiers_output, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                task_id = item.get("task_id", "")
                if not task_id:
                    continue
                if item.get("status") != "ok":
                    failed_task_ids.add(task_id)
                else:
                    ok_task_ids.add(task_id)
            except json.JSONDecodeError:
                pass

    remove_ids = failed_task_ids
    if not remove_ids:
        logger.success("Cleanup: no tasks to remove.")
        return

    kept = []
    removed = 0
    with open(args.tasks_output, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                if task.get("task_id", "") in remove_ids:
                    removed += 1
                else:
                    kept.append(line)
            except json.JSONDecodeError:
                kept.append(line)

    with open(args.tasks_output, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")

    logger.success(f"Cleanup: removed {removed} tasks with failed verifiers, kept {len(kept)}.")


# ── Pipelines ──────────────────────────────────────────────────────────────────

def run_generate(args: PipelineConfig) -> None:
    logger.info("=== Step 1/7: Scenario ===")
    scenario_step.run(args)

    logger.info("=== Step 2/7: Task ===")
    for attempt in range(1, args.max_task_attempts + 1):
        logger.info(f"Task generation attempt {attempt}/{args.max_task_attempts}")
        task_step.run(args)

    # logger.info("=== Step 3/7: Schema ===")
    # schema_step.run(args)

    # logger.info("=== Step 4/7: Spec ===")
    # spec_step.run(args)

    # logger.info("=== Step 5/7: Env (FastAPI servers) [generate] ===")
    # env_generate_step.run(args)

    # logger.info("=== Step 6/7: Data (seed DBs) ===")
    # data_step.run(args)

    # logger.info("=== Step 6.5/7: Verifier Gen (generate + sanity check) ===")
    # verifier_gen_step.run(args)

    logger.success("Generate pipeline complete.")


def _get_in_batch_ids(suggestions_dir: str) -> set[str]:
    """Return task_ids that have a pending suggestion file in the suggestions dir."""
    return {f.stem for f in Path(suggestions_dir).glob("*.json")}


def _fix_worker(args: PipelineConfig, q: queue.Queue, locks: FixLocks) -> None:
    """Consumer thread: process one task file at a time until sentinel None."""
    while True:
        task_file = q.get()
        if task_file is None:
            q.task_done()
            break
        name = Path(task_file).name
        logger.info(f"Fix thread: processing {name}")
        try:
            goal_fix_step.run(args, task_file, lock=locks.goal)
            data_fix_step.run(args, task_file, platform_locks=locks.data)
            verifier_fix_step.run(args, task_file, lock=locks.verifier)
            env_fix_step.run(args, task_file, platform_locks=locks.env)
            try:
                os.remove(task_file)
            except OSError:
                pass
        finally:
            q.task_done()
        logger.info(f"Fix thread: done with {name}")


def run_fix_debug(args: PipelineConfig) -> None:
    """Sequential debug mode: verifier → fix steps in the same thread, one task at a time."""
    suggestions_dir = Path(args.verified_suggestions_dir)
    suggestions_dir.mkdir(parents=True, exist_ok=True)

    # Process any leftover task files first
    for task_file in sorted(suggestions_dir.glob("*.json")):
        logger.info(f"[debug] Recovered pending task file: {task_file.name}")
        goal_fix_step.run(args, str(task_file))
        data_fix_step.run(args, str(task_file))
        verifier_fix_step.run(args, str(task_file))
        env_fix_step.run(args, str(task_file))
        try:
            os.remove(task_file)
        except OSError:
            pass

    while True:
        in_batch_ids = _get_in_batch_ids(args.verified_suggestions_dir)
        pending = verifier_step.load_pending_tasks(args, in_batch_ids)

        if not pending:
            logger.info("[debug] No more pending tasks.")
            break

        batch_tasks = pending[:args.fix_batch_size]
        logger.info(f"[debug] Verifier step: {len(batch_tasks)} tasks ({len(pending)} remaining)")

        def _run_fix_on_file(task_file: str) -> None:
            logger.info(f"[debug] Running fix steps on {Path(task_file).name}")
            goal_fix_step.run(args, task_file)
            data_fix_step.run(args, task_file)
            verifier_fix_step.run(args, task_file)
            env_fix_step.run(args, task_file)
            try:
                os.remove(task_file)
            except OSError:
                pass

        verifier_step.process_batch(args, batch_tasks, on_task_file=_run_fix_on_file)

    logger.success("[debug] Fix pipeline complete.")


def run_fix(args: PipelineConfig) -> None:
    suggestions_dir = Path(args.verified_suggestions_dir)
    suggestions_dir.mkdir(parents=True, exist_ok=True)

    q: queue.Queue = queue.Queue()
    locks = FixLocks()

    # Crash recovery: enqueue any leftover task files from a previous run
    for task_file in sorted(suggestions_dir.glob("*.json")):
        logger.info(f"Recovered pending task file: {task_file.name}")
        q.put(str(task_file))

    # Start multiple fix worker threads
    workers = []
    for _ in range(args.fix_concurrency):
        t = threading.Thread(target=_fix_worker, args=(args, q, locks), daemon=True)
        t.start()
        workers.append(t)

    # Producer loop: keep running verifier batches until all tasks are done
    while True:
        in_batch_ids = _get_in_batch_ids(args.verified_suggestions_dir)
        pending = verifier_step.load_pending_tasks(args, in_batch_ids)
        logger.opt(colors=True).info(f"<green>{len(pending)} tasks still pending verification.</green>")
        if not pending:
            # Wait for all queued fix tasks to finish (files deleted), then re-check.
            # Tasks that were in-flight may need another verification round after fixing.
            q.join()
            in_batch_ids = _get_in_batch_ids(args.verified_suggestions_dir)
            pending = verifier_step.load_pending_tasks(args, in_batch_ids)
            if not pending:
                logger.info("No more pending tasks.")
                break

        batch_tasks = pending[:args.fix_batch_size]
        logger.info(f"Verifier step: processing batch of {len(batch_tasks)} tasks ({len(pending)} remaining)")
        verifier_step.process_batch(args, batch_tasks, on_task_file=q.put)

    # Signal all fix workers to stop and wait
    for _ in workers:
        q.put(None)
    for t in workers:
        t.join()
    logger.success("Fix pipeline complete.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent world model data generation pipeline")

    parser.add_argument("--mode", type=str, default="fix", choices=["generate", "fix", "debug", "finalize"],
                        help="Pipeline mode: 'generate' for data generation, 'fix' for revision, "
                             "'debug' for sequential fix, 'finalize' to consolidate verified tasks into task_final.jsonl")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a yml config. When set, all settings are read from it "
                             "(defaults fill the rest); CLI flags other than --mode are ignored.")

    parsed = parser.parse_args()

    # Everything tunable lives in the yml (or PipelineConfig defaults). Edit the yml, not the CLI.
    cfg = PipelineConfig.from_yaml(parsed.config) if parsed.config else PipelineConfig()

    if parsed.mode == "generate":
        run_generate(cfg)
    elif parsed.mode == "debug":
        run_fix_debug(cfg)
    elif parsed.mode == "finalize":
        finalize_step.run(cfg)
    else:
        run_fix(cfg)


if __name__ == "__main__":
    main()
