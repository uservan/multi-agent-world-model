"""Agent eval entry point.

New run (creates a result folder, copies the yml in, locks frozen fields):
    python eval_main.py --init eval/config/single.yml
    python eval_main.py --init eval/config/multi_same.yml
    python eval_main.py --init eval/config/multi_diff.yml

Resume an existing run (re-runs only missing tasks, then aggregates):
    python eval_main.py --config outputs/eval/<run_folder>

Completed trajectories are cached, so re-running --config is cheap. To refresh
cost numbers, edit the *_cost fields in the run's config.yml and run --config
again — finished tasks are skipped and only eval.json is recomputed.

Ready-made templates live in eval/config/. After --init, only the cost fields and
data paths in the result's config.yml may be edited; model/run fields are locked.
"""
from __future__ import annotations

import argparse
import sys

from loguru import logger

import eval.run as eval_run
from eval.config import EvalConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent world model — agent eval")
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--init", metavar="TEMPLATE_YML", # default="eval/config/multi_same.yml",
                   help="Start a NEW run from a config template (default: eval/config/single.yml).")
    g.add_argument("--config", metavar="RUN_DIR", # default cleared so --init works for new runs
                   help="Resume an existing run folder (cached tasks are skipped).")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Number of task-runs to execute concurrently (default: 8).")
    args = parser.parse_args()

    try:
        # --config (resume) wins when given; otherwise --init (which has a default).
        if args.config:
            cfg = EvalConfig.load_run(args.config)
        else:
            cfg = EvalConfig.init_run(args.init)
            logger.success(f"New eval run: {cfg.run_dir}")
        eval_run.run(cfg, parallel=args.parallel)
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
