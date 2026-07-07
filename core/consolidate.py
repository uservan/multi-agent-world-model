import os
import json
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from utils.llm import LLMClient

from core.config import PipelineConfig
from core.spec import extract_action_params, _format_actions_params


# ── Action consolidation (for platforms with too many near-duplicate actions) ──
#
# Independently-generated tasks invent their own names for the same operation
# (select_account / select_destination_account / select_funding_account / ...),
# so a popular platform accumulates 100+ near-duplicate actions → one endpoint each
# → a server too large to generate within the output-token limit. When a platform
# exceeds CONSOLIDATE_THRESHOLD actions, an LLM clusters synonyms into a smaller
# canonical set; the mapping is then written back into tasks.jsonl so every later
# step (schema, spec, …) references a canonical action that has a real endpoint.
#
# This runs BEFORE schema so the whole chain reads one consistent set of action
# names: rename → schema → spec. Doing it here (rather than inside spec) keeps the
# ops_fingerprint stable across steps and avoids regenerating schema/spec just
# because actions were relabelled.

CONSOLIDATE_THRESHOLD = 80

CONSOLIDATE_SYSTEM = """You are an API consolidation expert. A platform accumulated too many near-duplicate action names because many independently-generated tasks each invented their own name for the SAME operation (e.g. select_account / view_account / get_account_overview / customer_account_overview all just "look at an account"). Your job is to merge these synonyms into one canonical name. Most of these duplicates SHOULD be merged — be decisive and collapse every group of genuine synonyms.

A genuine synonym = SAME operation kind, on the SAME object, with the SAME role — differing only in wording. Merge ALL of those, e.g.:
- login / authenticate_member → authenticate_user
- search_drug / search_medication → search_medication
- get_order_status / track_order / get_delivery_status / track_order_status → track_order
- verify_insurance_info / verify_insurance_coverage / verify_insurance_billing → verify_insurance
- list_prescriptions / lookup_prescription / search_prescriptions → search_prescriptions
Differing parameter lists are NOT a blocker — params get unioned on merge, so two reads of the same object that take different lookup params (by id vs by name) still merge.

There are EXACTLY THREE reasons to keep two actions separate. These are the ONLY brakes — if none of the three applies, MERGE:
1. Different operation KIND — read vs compute vs estimate vs check are different even on the same object: lookup_price ≠ compare_prices ≠ estimate_cost ≠ verify_coverage.
2. Different OBJECT / resource — same verb word, different thing: update_step_goal ≠ update_sleep_goal; get_vault_balance ≠ get_account_balance; verify_insurance ≠ estimate_copay; compile_sync_report ≠ compute_activity_summary.
3. Different ROLE / direction — deposit ≠ withdraw; send ≠ receive; source vs destination vs funding account; internal vs external transfer; sender vs recipient; book ≠ cancel.
Never use a parameter to paper over reason 2 or 3. Outside these three, default to merging.

Rules:
- EVERY input action MUST appear exactly once as a key in "mapping". Never drop one. An action with no synonym maps to ITSELF.
- Canonical names are clear generic verb_noun (e.g. select_account, get_account_balance).
- Aim to bring the total distinct count below {target} by merging every genuine synonym — but never merge across the three reasons above just to hit the target.

Output ONLY this JSON object (canonical is a plain string; do NOT add parameters or any other field):
{{
  "mapping": {{
    "<original_action>": "<canonical_action>",
    ...
  }}
}}"""

CONSOLIDATE_USER = """Platform: {name}
It currently has {n} distinct actions (target: below {target}). Merge every group of genuine synonyms (same operation kind + same object + same role, differing only in wording) — most duplicates here SHOULD merge, so be decisive. Keep two actions separate ONLY when the operation KIND, the OBJECT, or the ROLE/direction genuinely differs (e.g. sleep_goal vs step_goal; lookup vs compare vs estimate vs verify; source vs destination account; internal vs external transfer). Params with different keys still merge (they get unioned).

Actions (name: params / returns):
{actions}

Return the COMPLETE mapping covering EVERY one of the {n} actions (each maps to a canonical name or to itself)."""


def _robust_json_obj(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise


def _consolidate_actions(
    client: LLMClient, model: str, platform_name: str,
    action_ops: dict[str, dict], target: int, max_completion_tokens: int,
) -> dict[str, str]:
    """Return a TOTAL mapping {original_action: canonical_action}.

    Every original action is guaranteed a key; any the LLM dropped maps to itself.
    Only truly-synonymous actions are merged — params are never touched.
    """
    user = CONSOLIDATE_USER.format(
        name=platform_name, n=len(action_ops), target=target,
        actions=_format_actions_params(action_ops),
    )
    raw = client.complete(model, [
        {"role": "system", "content": CONSOLIDATE_SYSTEM.format(target=target)},
        {"role": "user", "content": user},
    ], max_completion_tokens)
    data = _robust_json_obj(raw)
    llm_map = data.get("mapping", {}) if isinstance(data, dict) else {}

    # Deterministic total-coverage guard: every original gets a canonical; missing → self.
    clean: dict[str, str] = {}
    for orig in action_ops:
        canon = llm_map.get(orig)
        clean[orig] = canon.strip() if isinstance(canon, str) and canon.strip() else orig
    return clean


def _rewrite_tasks_with_mappings(tasks_path: str, platform_mappings: dict[str, dict]) -> int:
    """Apply per-platform action mappings to tasks.jsonl (atomic, with .bak backup).

    Renames each step's action to its canonical name. Params are NEVER touched —
    only the action label changes. Returns the number of steps renamed.
    """
    if not platform_mappings or not os.path.exists(tasks_path):
        return 0
    import shutil
    shutil.copy2(tasks_path, tasks_path + ".bak")

    changed = 0
    out_lines: list[str] = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            task = json.loads(s)
            ops = task.get("metadata", {}).get("task_operations", {})
            for platform, mapping in platform_mappings.items():
                sub_agents = ops.get(platform)
                if not isinstance(sub_agents, list):
                    continue
                for steps in sub_agents:
                    if not isinstance(steps, list):
                        continue
                    for step in steps:
                        if not isinstance(step, dict):
                            continue
                        canon = mapping.get(step.get("action", ""))
                        if canon and canon != step.get("action"):
                            step["action"] = canon
                            changed += 1
            out_lines.append(json.dumps(task, ensure_ascii=False))

    tmp = tasks_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    os.replace(tmp, tasks_path)
    return changed


def run(args: PipelineConfig) -> None:
    """Consolidate over-large action sets and rewrite tasks.jsonl in place.

    Runs before schema/spec so every downstream step reads canonical action names.
    Idempotent in practice: after the first pass a platform's action count drops
    below the threshold, so re-runs are no-ops (nothing to consolidate → no .bak,
    no rewrite).
    """
    action_params = extract_action_params(args.tasks_output)

    to_consolidate = [n for n in action_params if len(action_params[n]) > CONSOLIDATE_THRESHOLD]
    if not to_consolidate:
        logger.success(f"Consolidate: no platform exceeds {CONSOLIDATE_THRESHOLD} actions; nothing to do.")
        return

    logger.info(f"Consolidating {len(to_consolidate)} platforms with >{CONSOLIDATE_THRESHOLD} actions: {to_consolidate}")
    client = LLMClient.from_config(args)

    platform_mappings: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                _consolidate_actions, client, args.model, name,
                dict(action_params[name]), CONSOLIDATE_THRESHOLD, args.max_completion_tokens,
            ): name
            for name in to_consolidate
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                mapping = future.result()
                distinct = len(set(mapping.values()))
                platform_mappings[name] = mapping
                logger.success(f"[{name}] consolidated {len(mapping)} → {distinct} actions")
            except Exception as e:
                logger.warning(f"[{name}] action consolidation failed ({e}); keeping all actions")

    if platform_mappings:
        n = _rewrite_tasks_with_mappings(args.tasks_output, platform_mappings)
        logger.success(
            f"Consolidation: rewrote {n} task steps across {len(platform_mappings)} platform(s); "
            f"backup at {args.tasks_output}.bak"
        )
    else:
        logger.warning("Consolidation: no mappings produced; tasks.jsonl left unchanged.")


def main() -> None:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Consolidate near-duplicate action names in tasks.jsonl")
    parser.add_argument("--tasks_output", type=str, default=defaults.tasks_output)
    parser.add_argument("--model", type=str, default=defaults.model)
    parser.add_argument("--api_key", type=str, default=defaults.api_key)
    parser.add_argument("--base_url", type=str, default=defaults.base_url)
    parser.add_argument("--concurrency", type=int, default=defaults.concurrency)
    parser.add_argument("--max_completion_tokens", type=int, default=defaults.max_completion_tokens)
    parsed = parser.parse_args()

    cfg = PipelineConfig()
    for k, v in vars(parsed).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    run(cfg)


if __name__ == "__main__":
    main()
