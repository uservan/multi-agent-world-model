"""Verifier execution and result aggregation.

Scoring rule: a platform passes if its verify_fn returns "complete" on
(initial_db, final_db, task_id). Platforms with an empty verify_fn (read-only)
are excluded from the denominator. acc = passed / verified-platform-count.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger


def run_verifier(verify_fn: str, initial_db: str, final_db: str, task_id: str) -> str:
    """Execute a verify_fn string. Returns 'complete' / 'others'."""
    if not verify_fn.strip():
        return "skip"
    import sqlite3
    ns: dict = {"sqlite3": sqlite3, "json": json, "os": os, "__builtins__": __builtins__}
    try:
        exec(verify_fn, ns)
        fn = ns.get("verify_task_completion")
        if fn is None:
            return "others"
        result = fn(initial_db, final_db, task_id)
        if isinstance(result, dict):
            return "complete" if result.get("result") == "complete" else "others"
        return "complete" if result is True else "others"
    except Exception as e:
        logger.debug(f"verifier exec error: {e}")
        return "others"


def score_task(runtime, verifiers: dict[str, str], task_id: str) -> tuple[dict, float]:
    """Run each platform's verifier on initial vs final DB.

    Returns (verifier_results {platform: 'complete'/'others'/'skip'}, acc).
    Read-only platforms (empty verify_fn → 'skip') are excluded from acc.
    """
    results: dict[str, str] = {}
    passed = scored = 0
    for platform, info in runtime.platforms.items():
        verify_fn = verifiers.get(platform, "") or ""
        outcome = run_verifier(verify_fn, info["seed_db"], info["final_db"], task_id)
        results[platform] = outcome
        if outcome == "skip":
            continue
        scored += 1
        if outcome == "complete":
            passed += 1
    acc = passed / scored if scored else 0.0
    return results, acc


# ── Aggregation ────────────────────────────────────────────────────────────────

def _cost(tokens: dict, price: tuple[float, float] | None) -> float:
    if not price:
        return 0.0
    # YAML may parse "1e-06" (no decimal point) as a string — coerce to be safe.
    return tokens.get("in", 0) * float(price[0]) + tokens.get("out", 0) * float(price[1])


def aggregate(
    run_dir: str,
    orch_price: tuple[float, float] | None = None,
    sub_price: tuple[float, float] | None = None,
) -> dict:
    """Read all traj files, aggregate per-task + overall stats. Writes eval.json.

    orch_price / sub_price = (input_cost, output_cost) per token, or None to skip.
    """
    traj_dir = Path(run_dir) / "traj"
    has_cost = bool(orch_price or sub_price)

    by_task: dict[str, list[dict]] = {}
    for f in sorted(traj_dir.glob("*.json")):
        try:
            traj = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if traj.get("status") != "complete":
            continue
        by_task.setdefault(traj["task_id"], []).append(traj)

    tasks_out: dict[str, dict] = {}
    all_accs: list[float] = []
    all_totals: list[int] = []
    all_costs: list[float] = []

    for task_id, trajs in by_task.items():
        runs = []
        for t in trajs:
            orch_tok = t.get("tokens", {}).get("orch", {"in": 0, "out": 0})
            sub_tok = t.get("tokens", {}).get("sub", {"in": 0, "out": 0})
            run_entry = {
                "run_idx": t.get("run_idx"),
                "acc": t.get("acc", 0.0),
                "orch_tokens": orch_tok,
                "sub_tokens": sub_tok,
            }
            if has_cost:
                oc = _cost(orch_tok, orch_price)
                sc = _cost(sub_tok, sub_price)
                run_entry["cost"] = {"orch": oc, "sub": sc, "total": oc + sc}
            runs.append(run_entry)

        n = len(runs)
        mean_acc = sum(r["acc"] for r in runs) / n if n else 0.0
        orch_in = sum(r["orch_tokens"]["in"] for r in runs)
        orch_out = sum(r["orch_tokens"]["out"] for r in runs)
        sub_in = sum(r["sub_tokens"]["in"] for r in runs)
        sub_out = sum(r["sub_tokens"]["out"] for r in runs)
        total_tokens = orch_in + orch_out + sub_in + sub_out

        entry = {
            "runs": runs,
            "mean_acc": mean_acc,
            "orch_tokens": {"in": orch_in, "out": orch_out},
            "sub_tokens": {"in": sub_in, "out": sub_out},
            "total_tokens": total_tokens,
        }
        if has_cost:
            tc = sum(r["cost"]["total"] for r in runs)
            entry["cost"] = {
                "orch": sum(r["cost"]["orch"] for r in runs),
                "sub": sum(r["cost"]["sub"] for r in runs),
                "total": tc,
            }
            all_costs.append(tc / n if n else 0.0)
        tasks_out[task_id] = entry
        all_accs.append(mean_acc)
        all_totals.append(total_tokens / n if n else 0)

    overall = {
        "n_tasks": len(tasks_out),
        "mean_acc": sum(all_accs) / len(all_accs) if all_accs else 0.0,
        "mean_total_tokens": sum(all_totals) / len(all_totals) if all_totals else 0.0,
    }
    if has_cost:
        overall["mean_cost"] = sum(all_costs) / len(all_costs) if all_costs else 0.0

    out = {"overall": overall, "tasks": tasks_out}
    (Path(run_dir) / "eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.success(
        f"Aggregated {len(tasks_out)} tasks → mean_acc={overall['mean_acc']:.3f}"
        + (f", mean_cost={overall['mean_cost']:.4f}" if has_cost else "")
    )
    return out
