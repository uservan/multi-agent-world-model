from __future__ import annotations
import os
import json
import uuid
import random
import argparse
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from loguru import logger
from openai import OpenAI  # used only for embeddings
from utils.llm import LLMClient
from tqdm import tqdm

from core.config import PipelineConfig


# ── Structure Sampling ────────────────────────────────────────────────────────

def _random_partition(total: int, max_parts: int, max_val: int) -> list[int] | None:
    """Random partition of total into k parts, each in [1, max_val]."""
    min_k = -(-total // max_val)  # ceil(total / max_val)
    max_k = min(max_parts, total)
    if min_k > max_k:
        return None
    k = random.randint(min_k, max_k)
    parts = [1] * k
    remaining = total - k
    while remaining > 0:
        eligible = [i for i in range(k) if parts[i] < max_val]
        if not eligible:
            return None
        parts[random.choice(eligible)] += 1
        remaining -= 1
    random.shuffle(parts)
    return parts


def sample_random_structure(
    budget: int,
    max_scenes: int,
    max_platforms_per_scene: int,
    max_agents_per_platform: int,
) -> list[list[int]] | None:
    """Returns a nested structure e.g. [[2,3],[1]] — outer=scenes, inner=sub-agents per platform slot."""
    scene_budgets = _random_partition(budget, max_scenes, max_platforms_per_scene * max_agents_per_platform)
    if scene_budgets is None:
        return None
    structure = []
    for sb in scene_budgets:
        platform_agents = _random_partition(sb, max_platforms_per_scene, max_agents_per_platform)
        if platform_agents is None:
            return None
        structure.append(platform_agents)
    return structure


def sample_unique_structures(
    budget: int,
    n: int,
    existing_keys: set[str],
    max_scenes: int,
    max_platforms_per_scene: int,
    max_agents_per_platform: int,
    max_attempts: int = 2000,
) -> list[list[list[int]]]:
    seen = set(existing_keys)
    results: list[list[list[int]]] = []
    attempts = 0
    while len(results) < n and attempts < max_attempts:
        s = sample_random_structure(budget, max_scenes, max_platforms_per_scene, max_agents_per_platform)
        attempts += 1
        if s is None:
            continue
        key = json.dumps(s)
        if key not in seen:
            seen.add(key)
            results.append(s)
    if len(results) < n:
        logger.warning(f"Budget {budget}: only sampled {len(results)}/{n} unique structures after {max_attempts} attempts")
    return results


# ── Platform IO ────────────────────────────────────────────────────────────────

def load_platforms(path: str) -> dict[str, dict]:
    platforms: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
                if p.get("name"):
                    platforms[p["name"]] = p
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded {len(platforms)} platforms from {path}")
    return platforms


def load_existing_tasks(output_path: str) -> tuple[set[str], dict[tuple[int, str], int]]:
    """
    Returns:
      - set of existing task_ids
      - dict mapping (budget, structure_key) -> count of tasks already generated
    """
    existing_ids: set[str] = set()
    structure_counts: dict[tuple[int, str], int] = defaultdict(int)

    if not os.path.exists(output_path):
        return existing_ids, structure_counts

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                if t.get("task_id"):
                    existing_ids.add(t["task_id"])
                    meta = t.get("metadata", t)
                    budget = meta.get("budget", 0)
                    sk = json.dumps(meta.get("structure", []))
                    structure_counts[(budget, sk)] += 1
            except json.JSONDecodeError:
                pass

    logger.info(f"Found {len(existing_ids)} existing tasks across {len(structure_counts)} structures")
    return existing_ids, structure_counts


def append_task(output_path: str, task: dict) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(task, ensure_ascii=False) + "\n")


def load_existing_embeddings(
    path: str, embed_model: str
) -> dict[tuple[int, str], list[np.ndarray]]:
    """Load saved embeddings keyed by (budget, structure_key), filtered by embed_model."""
    result: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    if not os.path.exists(path):
        return result
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("embed_model") == embed_model:
                    key = (item["budget"], item["structure_key"])
                    result[key].append(np.array(item["embedding"], dtype=np.float32))
            except (json.JSONDecodeError, KeyError):
                pass
    logger.info(f"Loaded embeddings for {len(result)} structures from {path}")
    return result


def append_embedding(
    path: str, task_id: str, embed_model: str,
    budget: int, structure_key: str, emb: np.ndarray
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "task_id": task_id,
            "embed_model": embed_model,
            "budget": budget,
            "structure_key": structure_key,
            "embedding": emb.tolist(),
        }) + "\n")


# ── Platform Sampling ──────────────────────────────────────────────────────────

def load_platform_counts(path: str) -> dict[str, int]:
    """Count how many accepted tasks each platform appears in (for priority weighting)."""
    counts: dict[str, int] = defaultdict(int)
    if not os.path.exists(path):
        return counts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                meta = t.get("metadata", t)
                for scene_plats in meta.get("scene_platforms", []):
                    for p in scene_plats:
                        counts[p] += 1
            except json.JSONDecodeError:
                pass
    return counts


def load_pattern_counts(path: str) -> dict[str, int]:
    """Count how many times each transition pattern has been used across all accepted tasks."""
    counts: dict[str, int] = defaultdict(int)
    if not os.path.exists(path):
        return counts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                meta = t.get("metadata", t)
                for trans in meta.get("scene_transitions", []):
                    name = trans.get("pattern")
                    if name:
                        counts[name] += 1
            except json.JSONDecodeError:
                pass
    return counts


def sample_candidate_platforms(
    platforms: dict[str, dict],
    platform_counts: dict[str, int],
    n_candidates: int = 50,
) -> list[dict]:
    """Sample up to n_candidates platforms weighted by 1/(count+1), sorted by usage ascending."""
    names = list(platforms.keys())
    if not names:
        return []
    counts = np.array([platform_counts.get(n, 0) for n in names], dtype=float)
    weights = 1.0 / (counts + 1.0)
    weights /= weights.sum()
    k = min(n_candidates, len(names))
    chosen_idx = np.random.choice(len(names), size=k, replace=False, p=weights)
    result = []
    for i in chosen_idx:
        name = names[i]
        p = dict(platforms[name])
        p["usage_count"] = int(platform_counts.get(name, 0))
        result.append(p)
    result.sort(key=lambda x: x["usage_count"])
    return result


def _format_platform_info(scene_platforms: list[list[str]], platforms: dict[str, dict]) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for scene in scene_platforms:
        for name in scene:
            if name in seen:
                continue
            seen.add(name)
            p = platforms[name]
            features = "; ".join(p.get("features", [])[:6])
            lines.append(f"- {name}: {p.get('description', '')}\n  Features: {features}")
    return "\n".join(lines)


# ── Scene Interaction Patterns ─────────────────────────────────────────────────

SCENE_INTERNAL_PATTERNS = [
    {
        "name": "parallel_search",
        "desc": "Sub-agents each search for different items/categories independently and in parallel; results collected separately",
    },
    {
        "name": "act_then_verify",
        "desc": "Sub-agent 1 performs a write action (place order, make booking, create post); sub-agent 2 reads back the result using the ID or reference returned by sub-agent 1",
    },
    {
        "name": "check_then_act",
        "desc": "Sub-agent 1 checks a condition (eligibility, availability, price threshold); sub-agent 2 conditionally executes an action based on sub-agent 1's output value",
    },
    {
        "name": "cross_validate",
        "desc": "Sub-agents independently retrieve the same or related data from different endpoints or angles, then results are compared or merged",
    },
]

SCENE_INTERACTION_PATTERNS = [
    {
        "name": "pick_best",
        "desc": (
            "Scene i+1 acts on only the single best result from scene i "
            "(e.g. buy only the cheapest item found)"
        ),
    },
    {
        "name": "act_on_all",
        "desc": (
            "Scene i+1 executes actions on ALL results from scene i "
            "(e.g. place orders for every item found across platforms)"
        ),
    },
    {
        "name": "filter_subset",
        "desc": (
            "Scene i+1 acts only on results from scene i that meet a condition "
            "(e.g. only items with rating > 4.5, or price < $50)"
        ),
    },
    {
        "name": "aggregate_decide",
        "desc": (
            "Scene i+1 makes a decision based on combining multiple signals from scene i "
            "(e.g. weighted score of price + rating + availability)"
        ),
    },
    {
        "name": "conditional_branch",
        "desc": (
            "Scene i+1 behavior is determined by scene i's outcome: "
            "if scene i finds no results or fails a check, scene i+1 takes a fallback path "
            "(e.g. cancel a reservation on platform A and rebook on platform B, "
            "or send a 'not found' notification, or log the failure and exit)"
        ),
    },
    {
        "name": "cascaded_dependency",
        "desc": (
            "Scene i+1's parameters are fully determined by scene i's output "
            "(e.g. use the seller ID from scene i to fetch detailed seller info)"
        ),
    },
    {
        "name": "independent",
        "desc": (
            "Scene i+1 is a completely separate task unrelated to scene i — "
            "the user is performing two distinct workflows in the same session "
            "(e.g. scene 1 manages email, scene 2 independently books a flight)"
        ),
    },
]


# Strict, shared criteria for each inter-scene pattern. Injected into BOTH the
# scene-plan generation prompt and the _check_transition validator so the model
# builds to the exact standard it is later judged against. Criteria mirror the
# real discard reasons seen in generation logs.
PATTERN_RULES = {
    "pick_best": {
        "criteria": (
            "Scene i+1 acts on EXACTLY ONE result from scene i — the single best by a "
            "named attribute (cheapest, highest-rated, soonest). At least one step in "
            "scene i+1 must consume a return value from scene i."
        ),
        "good": "Scene 1 compares 4 laptops by price → Scene 2 buys ONLY the cheapest (uses its product_id).",
        "bad":  "Scene 1 finds 4 laptops → Scene 2 buys two of them (that is filter_subset, not pick_best).",
    },
    "act_on_all": {
        "criteria": (
            "Scene i+1 performs the action on EVERY result scene i produced. If scene i "
            "returned K items (counting across all its sub-agents/platforms), scene i+1 "
            "must contain K corresponding actions — not one, not a subset. Each action "
            "must consume a scene i return value."
        ),
        "good": "Scene 1 finds 3 products (RKT-338201, -338205, -338210) → Scene 2 places 3 orders, one per product.",
        "bad":  "Scene 1 returns 3 products → Scene 2 orders only 1 of them.",
    },
    "filter_subset": {
        "criteria": (
            "Scene i+1 applies an EXPLICIT PER-ITEM condition (a threshold on a per-result "
            "attribute, e.g. rating > 4.2, price < $50) to the FULL set of scene i results, "
            "keeps only the items that pass, and acts on that subset. It must (1) reference "
            "the full result set, (2) apply the stated condition item-by-item, (3) drop the "
            "failing items. It is NOT filter_subset if scene i+1 just reuses items scene i "
            "already pre-selected (pass-through), or applies an AGGREGATE/combined threshold "
            "(e.g. sum > $2,000) — that is aggregate_decide."
        ),
        "good": "Scene 1 returns 5 listings with ratings → Scene 2 keeps the 3 with rating > 4.2 and books them.",
        "bad":  "Scene 1 already returns 'recommended' items and Scene 2 just books them (no own filter), OR Scene 2 checks combined total > $2,000 (aggregate, not per-item).",
    },
    "aggregate_decide": {
        "criteria": (
            "Scene i+1 has an explicit step that COMBINES MULTIPLE DISTINCT signals (>=2 "
            "different factors: price + distance + availability, etc.) from scene i into a "
            "single NEW decision or score, then acts on it. Passing through one pre-computed "
            "value from scene i (e.g. the already-chosen lowest-price option) is NOT "
            "aggregate_decide."
        ),
        "good": "Scene 1 returns price, distance, coupon per pharmacy → Scene 2 computes a weighted score across all three and picks one.",
        "bad":  "Scene 1 already determined the lowest-price pharmacy → Scene 2 just uses it (single passed-through signal).",
    },
    "conditional_branch": {
        "criteria": (
            "Scene i+1's path is DETERMINED BY scene i's actual outcome shown above. If scene "
            "i SUCCEEDED (found results / passed its check), scene i+1 MUST take the success "
            "path; ONLY if scene i failed or returned empty may scene i+1 take the fallback "
            "path. The branch scene i+1 actually takes must be CONSISTENT with scene i's real "
            "outcome — a fallback that fires while scene i succeeded is wrong."
        ),
        "good": "Scene 1 finds no availability → Scene 2 rebooks on the backup platform (fallback fires correctly).",
        "bad":  "Scene 1 confirmed both reservations (success), yet Scene 2 still runs the fallback path.",
    },
    "cascaded_dependency": {
        "criteria": (
            "Scene i+1's parameters are directly determined by a SPECIFIC identifier/value "
            "returned by scene i (an ID, token, name). At least one step in scene i+1 must "
            "consume that exact scene i return value as an input param."
        ),
        "good": "Scene 1 returns seller_id S-4821 → Scene 2 fetches that seller's detail using S-4821.",
        "bad":  "Scene 2 independently searches and never uses any scene 1 return value.",
    },
}


def _pattern_rule_block(name: str) -> str:
    """Formatted strict-criteria block for a pattern; empty for unknown/independent."""
    r = PATTERN_RULES.get(name)
    if not r:
        return ""
    return (
        f"\n  REQUIRED to satisfy '{name}':\n"
        f"  {r['criteria']}\n"
        f"  GOOD example: {r['good']}\n"
        f"  BAD example:  {r['bad']}"
    )


# Producer-side guidance for the SCENE THAT FEEDS a transition (scene i). The
# common failure is scene i doing the downstream work itself (pre-filtering,
# pre-selecting, returning a single item) so scene i+1 has nothing left to do
# and is judged a pass-through. This tells scene i to leave the raw materials
# the pattern needs and NOT complete the next scene's job.
PATTERN_PRODUCER_RULES = {
    "pick_best": "Produce MULTIPLE comparable options with the ranking attribute (price/rating). Do NOT pre-select the best — the next scene picks it.",
    "act_on_all": "Produce a MULTI-ITEM result set (return several distinct ids/items). The next scene acts on every one, so there must be several to act on.",
    "filter_subset": "Produce the FULL UNFILTERED result set (several items) each carrying the attribute to be filtered (e.g. each listing's rating/price). Do NOT pre-filter or pre-select — leave the per-item filtering to the next scene.",
    "aggregate_decide": "Produce the MULTIPLE DISTINCT signals (>=2 factors: price, distance, availability...) as separate return values. Do NOT pre-combine them into a decision — the next scene aggregates.",
    "conditional_branch": "Produce a concrete OUTCOME the next scene branches on — a success/failure flag plus data as actual return values (not just echoing original goal params), so the next scene can both branch on it AND consume it.",
    "cascaded_dependency": "Produce the SPECIFIC identifier (id/token/name) the next scene will consume as an input param.",
}


def _pattern_producer_block(name: str) -> str:
    """Producer-side hint for scene i so it leaves work for scene i+1; empty if none."""
    rule = PATTERN_PRODUCER_RULES.get(name)
    if not rule:
        return ""
    return f"\n  PRODUCER duty for '{name}': {rule}"


def _assign_transition_patterns(
    num_transitions: int,
    pattern_counts: dict[str, int],
) -> list[dict]:
    """Assign one pattern per transition, sampling globally least-used patterns first."""
    if num_transitions == 0:
        return []
    names = [p["name"] for p in SCENE_INTERACTION_PATTERNS]
    counts = np.array([pattern_counts.get(n, 0) for n in names], dtype=float)
    weights = 1.0 / (counts + 1.0)
    weights /= weights.sum()
    name_to_pattern = {p["name"]: p for p in SCENE_INTERACTION_PATTERNS}
    chosen_names = np.random.choice(names, size=num_transitions, replace=True, p=weights)
    return [name_to_pattern[n] for n in chosen_names]


# ── LLM Prompts ────────────────────────────────────────────────────────────────

GOAL_GEN_SYSTEM = """You are a task designer for multi-agent AI systems.
Choose platforms for each scene and generate a realistic overall user goal for a multi-scene workflow.
CRITICAL: goals must span the full breadth of human activity — healthcare, finance, HR, legal, logistics, retail, education, real estate, food service, manufacturing, media, government services, and more. Do NOT default to travel or e-commerce. Match the goal domain to the chosen platforms; if platforms are unrelated to travel, the goal must not involve flights or hotels.
Return ONLY valid JSON, no markdown."""

SCENE_GEN_SYSTEM = """You are a task designer for multi-agent AI systems.
Generate task operations for one platform in a multi-agent workflow scene.
Return ONLY valid JSON, no markdown."""

PLAN_SYSTEM = """You are a task planner for multi-agent AI systems.
For each platform in a scene, determine: what concrete values it consumes as params, what it does, and what concrete values it produces as returns.
Invent realistic concrete values for produces (real-looking IDs, names, prices). Use exact values from goal or previous scene returns for consumes.
Return ONLY valid JSON, no markdown."""


def _format_ops_returns(ops: dict) -> str:
    """Summarise the non-trivial returns from a task_operations dict."""
    lines = []
    for plat, sub_list in ops.items():
        for k, sub in enumerate(sub_list if isinstance(sub_list, list) else []):
            for step in (sub or []):
                r = step.get("returns", {})
                if r and r != {"status": "success"}:
                    lines.append(
                        f"  {plat} sub-agent {k + 1} · {step.get('action')}: "
                        f"returns {json.dumps(r)}"
                    )
    return "\n".join(lines)


def _build_goal_prompt(
    structure: list[list[int]],
    scene_platforms: list[list[str]],
    patterns: list[dict],
    platform_info_str: str,
) -> str:
    scene_lines = []
    for i, (scene, plats) in enumerate(zip(structure, scene_platforms)):
        parts = ", ".join(f"{name} × {n}" for n, name in zip(scene, plats))
        scene_lines.append(f"  Scene {i + 1}: {parts}")

    trans_lines = [
        f"  Scene {t + 1}→{t + 2}: {p['name']} — {p['desc']}"
        for t, p in enumerate(patterns)
    ]

    has_independent = any(p["name"] == "independent" for p in patterns)
    independent_note = (
        "\nNote: some scene transitions are 'independent' — those scenes are completely unrelated tasks. "
        "For those, the goal should describe separate workflows the user wants done in the same session, "
        "NOT a single continuous narrative. Each independent segment should stand on its own."
        if has_independent else ""
    )

    trans_goal_hints = []
    for t, p in enumerate(patterns):
        name = p["name"]
        if name == "independent":
            trans_goal_hints.append(f"  Scene {t+1}→{t+2}: completely separate — describe as two distinct tasks the user wants done in the same session")
        elif name == "pick_best":
            trans_goal_hints.append(
                f"  Scene {t+1}→{t+2}: goal must explicitly say that scene {t+1} finds/compares multiple options "
                f"and scene {t+2} takes action on only the single best one (e.g. 'buy the cheapest item found', "
                f"'book the highest-rated result') — mention what specific attribute determines 'best'"
            )
        elif name == "act_on_all":
            trans_goal_hints.append(
                f"  Scene {t+1}→{t+2}: goal must explicitly say that scene {t+1} collects a set of results (IDs, items, entries) "
                f"and scene {t+2} performs an action on every single one of them — mention what kind of results and what action"
            )
        elif name == "filter_subset":
            trans_goal_hints.append(
                f"  Scene {t+1}→{t+2}: goal must explicitly state a filter condition (e.g. rating > 4.5, price < $50) "
                f"that scene {t+2} applies to scene {t+1}'s results, and what action is taken on the qualifying subset"
            )
        elif name == "aggregate_decide":
            trans_goal_hints.append(
                f"  Scene {t+1}→{t+2}: goal must explicitly say that scene {t+1} gathers specific signals "
                f"(e.g. prices, ratings, availability counts) and scene {t+2} makes a concrete decision by combining them"
            )
        elif name == "conditional_branch":
            trans_goal_hints.append(
                f"  Scene {t+1}→{t+2}: goal must explicitly describe the condition from scene {t+1}'s result "
                f"(e.g. 'if no results found', 'if price exceeds budget') and what scene {t+2} does in each case"
            )
        elif name == "cascaded_dependency":
            trans_goal_hints.append(
                f"  Scene {t+1}→{t+2}: goal must explicitly say that a specific result from scene {t+1} "
                f"(e.g. a returned ID, name, or token) is directly used as input in scene {t+2} — name the type of result"
            )

    trans_goal_str = "\n".join(trans_goal_hints)

    return f"""Design an overall user goal for this multi-scene workflow:

Scene assignments:
{chr(10).join(scene_lines)}

Inter-scene transitions:
{chr(10).join(trans_lines) if trans_lines else "  (single scene, no transitions)"}

Platform information:
{platform_info_str}
{independent_note}
Requirements:
- Domain diversity: the goal must fit the chosen platforms' domain. Actively vary across: healthcare, legal, HR/payroll, supply chain, real estate, food service, education, finance/accounting, manufacturing, media/publishing, government/civic services, retail, logistics, and more. Do NOT default to travel or generic shopping unless the platforms are travel/retail specific.
- 2-4 sentences describing what the user wants to accomplish overall
- Must naturally motivate all scenes in order
- The goal MUST contain concrete specifics that the platform agents can use directly as params:
    - Specific locations, cities, or addresses (e.g. "Chicago, IL", "94110", "Unit 4B at 220 Oak St")
    - Specific dates or time ranges (e.g. "March 15 2025", "Q2 2025", "within 48 hours")
    - Concrete names: people, companies, products, models, SKUs (e.g. "Lena Hoffmann", "Acme Supplies Ltd", "Dell P2422H monitor")
    - Specific numeric targets: budget, quantity, threshold (e.g. "$1,200 budget", "minimum 4.2 rating", "3 units")
    - Concrete identity fields where relevant: email, employee ID, invoice number, etc.
  At least 3-4 such concrete values MUST appear in the goal text.
- Sub-agent counts: for each platform slot with N sub-agents, the goal must provide EXACTLY N distinct targets
  for that platform — one target per sub-agent. Pick a scenario type that naturally produces N parallel items:
    N=2: two employees onboarding on different teams; two rental properties to evaluate; a couple with separate insurance claims
    N=3: three team members each needing a software license; a restaurant ordering from three different suppliers; three patients scheduling different specialist appointments
    N=4+: a department bulk-ordering equipment (different specs per person); a school enrolling multiple students in different courses
  The subject is whoever fits naturally — a company, a household, a team, a small business, a clinic, etc.
  NEVER default to travel/flight booking unless platforms are specifically travel-related.
  NEVER give a platform with N>1 sub-agents only a single item/person to work on — each sub-agent needs its own distinct target.
- The goal wording must naturally reflect each transition:
{trans_goal_str}
- score: 5=highly realistic; 3=plausible; 1=implausible or contrived

Return JSON: {{"goal": "...", "score": <1-5>}}"""


def _build_trans_goal_hints(patterns: list[dict]) -> str:
    hints = []
    for t, p in enumerate(patterns):
        name = p["name"]
        if name == "independent":
            hints.append(f"  Scene {t+1}→{t+2}: completely separate — two distinct tasks the user wants done in the same session")
        elif name == "pick_best":
            hints.append(
                f"  Scene {t+1}→{t+2}: goal must say scene {t+1} finds/compares multiple options "
                f"and scene {t+2} acts on only the single best one (mention the attribute that determines 'best')"
            )
        elif name == "act_on_all":
            hints.append(
                f"  Scene {t+1}→{t+2}: goal must say scene {t+1} collects a set of results "
                f"and scene {t+2} performs an action on every single one of them"
            )
        elif name == "filter_subset":
            hints.append(
                f"  Scene {t+1}→{t+2}: goal must state a filter condition (e.g. rating > 4.5, price < $50) "
                f"that scene {t+2} applies to scene {t+1}'s results"
            )
        elif name == "aggregate_decide":
            hints.append(
                f"  Scene {t+1}→{t+2}: goal must say scene {t+1} gathers specific signals "
                f"and scene {t+2} makes a concrete decision by combining them"
            )
        elif name == "conditional_branch":
            hints.append(
                f"  Scene {t+1}→{t+2}: goal must describe the condition from scene {t+1}'s result "
                f"and what scene {t+2} does in each case"
            )
        elif name == "cascaded_dependency":
            hints.append(
                f"  Scene {t+1}→{t+2}: goal must say a specific result from scene {t+1} "
                f"(e.g. a returned ID, name, or token) is directly used as input in scene {t+2}"
            )
    return "\n".join(hints)


def _build_goal_and_platform_prompt(
    structure: list[list[int]],
    patterns: list[dict],
    candidate_platforms: list[dict],
) -> str:
    scene_lines = []
    for i, scene in enumerate(structure):
        slot_desc = "; ".join(
            f"slot {j+1} → {n} sub-agent{'s' if n > 1 else ''} (goal needs {n} distinct target{'s' if n > 1 else ''})"
            for j, n in enumerate(scene)
        )
        scene_lines.append(f"  Scene {i+1}: choose {len(scene)} platform(s) in order — {slot_desc}")

    trans_lines = [
        f"  Scene {t+1}→{t+2}: {p['name']} — {p['desc']}"
        for t, p in enumerate(patterns)
    ]

    plat_lines = []
    for p in candidate_platforms:
        usage = p.get("usage_count", 0)
        desc = (p.get("description", "") or "")[:100]
        plat_lines.append(
            f"  [{p['name']}] ({p.get('category', '')}/{p.get('subcategory', '')}, used {usage}x): {desc}"
        )

    trans_goal_str = _build_trans_goal_hints(patterns)

    has_independent = any(p["name"] == "independent" for p in patterns)
    independent_note = (
        "\nNote: scenes with 'independent' transitions are completely unrelated tasks — "
        "describe them as separate workflows in the same session, not a single narrative.\n"
        if has_independent else ""
    )

    example_sp = json.dumps([[f"Platform{chr(65+j)}" for j in range(len(scene))] for scene in structure])

    return f"""Choose platforms for each scene and write an overall user goal for this multi-scene workflow.

Scene structure (you must assign exactly this many platforms per scene):
{chr(10).join(scene_lines)}

Inter-scene transitions:
{chr(10).join(trans_lines) if trans_lines else "  (single scene, no transitions)"}
{independent_note}
Available platforms (prefer platforms with lower used Nx count for dataset diversity):
{chr(10).join(plat_lines)}

Requirements:
- Choose platforms that naturally fit together — pick ones where the transition pattern makes semantic sense
- Prefer platforms with lower usage_count; platforms earlier in the list have been used less
- Each scene may mix platforms from different categories if the goal benefits from it
- Do NOT invent platform names — only use names from the list above
- Domain diversity: the goal must fit the chosen platforms' domain. Actively vary across: healthcare, legal, HR/payroll, supply chain, real estate, food service, education, finance/accounting, manufacturing, media/publishing, government/civic services, retail, logistics, and more. Do NOT default to travel or generic shopping unless the platforms are travel/retail specific.
- Goal: 2-4 sentences describing what the user wants to accomplish overall
- Must naturally motivate all scenes in order
- The goal MUST contain concrete specifics that the platform agents can use directly as params:
    - Specific locations, cities, or addresses (e.g. "Chicago, IL", "94110", "Unit 4B at 220 Oak St")
    - Specific dates or time ranges (e.g. "March 15 2025", "Q2 2025", "within 48 hours")
    - Concrete names: people, companies, products, models, SKUs (e.g. "Lena Hoffmann", "Acme Supplies Ltd", "Dell P2422H monitor")
    - Specific numeric targets: budget, quantity, threshold (e.g. "$1,200 budget", "minimum 4.2 rating", "3 units")
    - Concrete identity fields where relevant: email, employee ID, invoice number, etc.
  At least 3-4 such concrete values MUST appear in the goal text.
- Sub-agent targets: the slot order in scene_platforms MUST match the slot order listed above.
  For each slot with N sub-agents, the goal must explicitly name N distinct targets (people, items, accounts, etc.) for that platform.
  Plan your platforms first, then write the goal with the right number of targets per slot:
  Each sub-agent must have something distinct to do. There are two valid patterns — choose whichever fits:
  Pattern A — parallel targets (same task, different objects):
    each sub-agent handles a different item/person/account, e.g.:
    - "compare prices for Samsung Galaxy S24, iPhone 15 Pro, and Pixel 8" → 3 sub-agents each searching one model
    - "onboard 3 new hires: Alice (Eng), Bob (Sales), Carol (Design)" → each sub-agent handles one person
  Pattern B — role division (different operations on the same pre-existing object):
    each sub-agent performs a different operation on the same object that already exists in the DB, e.g.:
    - "update billing address and payment method for account #C-8821" → sub-agent 1 updates address, sub-agent 2 updates payment method
    - "post progress update and assign reviewer on project PRJ-4401" → sub-agent 1 posts update, sub-agent 2 assigns reviewer
    IMPORTANT: each sub-agent must be fully independent — no sub-agent may depend on another sub-agent's runtime-created output.
    A sequential pipeline (search → compare → purchase) where each step needs the previous step's result is NOT valid Pattern B.
  Mixed (both patterns at once) is also valid.
  The goal text must make it obvious what each sub-agent should do — name distinct targets OR distinct roles explicitly.
  NEVER give N>1 sub-agents identical work with no differentiation.
  NEVER default to travel/flight booking unless platforms are specifically travel-related.
- The goal wording must naturally reflect each transition:
{trans_goal_str}
- score: 5=highly realistic; 3=plausible; 1=implausible or contrived

Return JSON: {{"scene_platforms": {example_sp}, "goal": "...", "score": <1-5>}}"""




def _build_plan_prompt(
    goal: str,
    scene_idx: int,
    scene_plats: list[str],
    scene: list[int],
    transition_in: dict | None,
    transition_out: dict | None,
    prev_ops_returns: str,
    prev_scene_plan: dict,
) -> str:
    is_independent = transition_in is not None and transition_in["name"] == "independent"

    trans_lines = ""
    if transition_in and not is_independent:
        trans_lines += f"\nTransition from previous scene — '{transition_in['name']}': {transition_in['desc']}"
        trans_lines += _pattern_rule_block(transition_in["name"])
    if transition_out:
        trans_lines += f"\nTransition to next scene — '{transition_out['name']}': {transition_out['desc']}"
        trans_lines += _pattern_producer_block(transition_out["name"])

    prev_context = ""
    if not is_independent:
        if prev_scene_plan.get("plan"):
            prev_context += f"\nPrevious scene plan:\n{json.dumps(prev_scene_plan, ensure_ascii=False, indent=2)}"
        if prev_ops_returns:
            prev_context += f"\nPrevious scene actual returns:\n{prev_ops_returns}"

    slot_lines = [f"  {name} × {n} sub-agent(s)" for n, name in zip(scene, scene_plats)]

    return f"""Plan Scene {scene_idx + 1} of this multi-agent task.

Goal: {goal}{trans_lines}{prev_context}

Platforms in this scene:
{chr(10).join(slot_lines)}

For each platform produce a structured plan with:
- consumes: params this platform needs — use EXACT values from goal text or previous scene returns above
- role: 2-3 sentences describing EXACTLY what this platform does — name the specific actions (e.g. "search listings", "place order", "send invoice"), the concrete objects it operates on (e.g. "Logitech MX Master 3 mouse", "Unit 4B at 220 Oak St", "invoice #INV-2024-0831"), and the concrete outcome (e.g. "adds item to cart and checks out for $89", "schedules viewing for March 12"). Do NOT write generic descriptions like "handles the task".
- sub_agents: list of N strings, one per sub-agent, each describing exactly what that sub-agent does.
  Two valid patterns — use whichever fits the goal:
  Pattern A (parallel targets — same task, different objects):
    ["searches Samsung Galaxy S24 price and availability", "searches iPhone 15 Pro price and availability", "searches Pixel 8 price and availability"]
    ["processes payroll for Alice Chen (Eng, $9,200/mo)", "processes payroll for Bob Lee (Sales, $7,800/mo)"]
  Pattern B (role division — different operations on the same pre-existing object):
    ["updates billing address for account acct_8821", "updates payment method for account acct_8821"]
    ["posts progress update on project PRJ-4401", "assigns reviewer to project PRJ-4401"]
    Each sub-agent operates independently on the same pre-existing DB object — no sub-agent depends on another's runtime output.
  For N=1 write a single-element list. Be concrete — name specific targets, roles, or actions from the goal.
- Shared variables rule (Rule A): sub-agents MAY share params that come from pre-existing DB values,
  the goal text, or a prior scene's returns (e.g. course_id, project_id, user_id) — this is valid as
  long as each sub-agent performs a different type of operation. NEVER design a pattern where sub-agent
  k-1 creates a value at runtime and sub-agent k consumes it directly (Rule C). Each sub-agent must
  independently query any shared pre-existing value via its own lookup step.
- Unique creation rule: if multiple sub-agents each create a resource (e.g. create_lead, create_order),
  their WRITE actions MUST return different resource identifiers (IDs, tokens, URLs, reference codes) —
  never copy an identifier from one sub-agent's returns into another's. Generic status fields
  (e.g. status: "success") may naturally be identical and are not subject to this rule.
- produces: return values this platform will produce — invent realistic concrete values (e.g. IDs like "ord_8f3k29", names, prices)
  Order platforms so producers come before consumers.

Return JSON:
{{
  "platform_order": {json.dumps(scene_plats)},
  "plan": {{
    "{scene_plats[0]}": {{"consumes": {{}}, "role": "...", "sub_agents": ["..."], "produces": {{}}}}
  }}
}}"""


def _build_subagent_prompt(
    goal: str,
    scene_idx: int,
    platform_name: str,
    platform_info: str,
    platform_plan: dict,
    sub_agent_idx: int,
    n_subagents: int,
    sub_agent_desc: str,
    prev_subagents_ops: list,
    scene_ops_so_far: dict,
    min_steps: int,
    max_steps: int,
    transition_in: dict | None = None,
    prev_scene_returns: str = "",
) -> str:
    consumes_str = json.dumps(platform_plan.get("consumes", {}), ensure_ascii=False, indent=2)
    produces_str = json.dumps(platform_plan.get("produces", {}), ensure_ascii=False, indent=2)
    role = platform_plan.get("role", "")
    is_last = (sub_agent_idx == n_subagents - 1)

    transition_block = ""
    if transition_in and transition_in.get("name") != "independent" and prev_scene_returns:
        transition_block = (
            f"\nScene transition from previous scene — '{transition_in['name']}': {transition_in['desc']}\n"
            f"Previous scene actual return values (you MUST use these to implement the transition):\n"
            f"{prev_scene_returns}"
        )

    scene_context = ""
    if scene_ops_so_far:
        s = _format_ops_returns(scene_ops_so_far)
        if s:
            scene_context = f"\nSame-scene platforms already generated (actual return values):\n{s}"

    prev_block = ""
    if prev_subagents_ops:
        lines = []
        for k, steps in enumerate(prev_subagents_ops):
            # Show action + params only — strip returns so this sub-agent cannot reuse runtime-created IDs
            stripped = [{"action": s.get("action", ""), "params": s.get("params", {})} for s in steps]
            lines.append(f"  Sub-agent {k+1} actions:\n{json.dumps(stripped, ensure_ascii=False, indent=4)}")
        prev_block = "\nSame-platform sub-agents already generated (actions only — returns hidden):\n" + "\n".join(lines) + "\nUse these to understand what operations are already covered — do NOT repeat the same operation type, and do NOT use any IDs or values you might infer from their params as your own returns."

    produces_note = (
        f"  Produces — your final step's returns MUST include these keys with these exact values:\n{produces_str}"
        if is_last else
        f"  Produces (final platform output, handled by the last sub-agent): {produces_str}\n"
        f"  Your steps produce intermediate values — make them realistic and usable by later sub-agents."
    )

    return f"""Generate steps for sub-agent {sub_agent_idx + 1}/{n_subagents} of {platform_name} in Scene {scene_idx + 1}.

Goal: {goal}{transition_block}

Platform plan:
  Overall role: {role}
  Consumes — use these EXACT values as params in your steps:
{consumes_str}
{produces_note}

This sub-agent's specific task: {sub_agent_desc}{prev_block}
{scene_context}

{platform_info}

Requirements:
- {min_steps}-{max_steps} ordered steps
- Each step: {{"action": "<action>", "params": {{...}}, "returns": {{...}}}}
  — params: use values from Consumes above, from preceding steps' returns, or from previous sub-agents' returns above
  — returns: realistic concrete values needed by subsequent steps
- Filter params belong inside the search/list action — do NOT add separate filter steps
- Sub-agent shared variable rules:
  (1) Shared params allowed: you MAY use the same param values as other sub-agents if they come from
      the goal text, pre-existing DB values, or a prior scene's returns (e.g. course_id, project_id,
      user_id) — valid as long as you perform a different operation type than other sub-agents.
  (2) Unique returns required: your WRITE actions MUST produce unique resource identifiers (IDs,
      tokens, URLs, reference codes) — never return the same identifier that a previous sub-agent
      already returned. Generic status fields (e.g. status: "success") are exempt from this rule.
  (3) No cross-sub-agent WRITE params: you CANNOT use values produced by other sub-agents' WRITE
      actions (add_*, create_*, place_*, book_*, submit_*) as your params — these are runtime-created
      and inaccessible to you. Create your own separate instance if you need a similar resource.
- Use realistic concrete values — real-looking names, emails, IDs, prices, dates
  — NEVER use placeholders like "example.com", "test_user", "user_123", "***", "dummy"
  — IDs/tokens: e.g. "ord_8f3k29", "tok_a7bc14d9" — not "order_001"
  — Emails: e.g. "marco.jensen@outlook.com" — not "buyer@example.com"

Return JSON:
{{"steps": [...]}}"""


def _build_outcome_prompt(
    goal: str,
    platform_name: str,
    platform_plan: dict,
    platform_ops: list,
) -> str:
    role = platform_plan.get("role", "")
    ops_str = json.dumps(platform_ops, ensure_ascii=False, indent=2)
    return f"""Write expected_outcome for {platform_name}.

Goal: {goal}
Platform role: {role}

All sub-agents' completed steps:
{ops_str}

Write one paragraph describing what this platform accomplished, referencing specific IDs, names, values, and outcomes from the steps above.

Return JSON:
{{"expected_outcome": "..."}}"""


# ── LLM / Embedding Calls ──────────────────────────────────────────────────────

def _parse_json_obj(text: str) -> dict:
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


def call_generate_goal(
    client: LLMClient,
    model: str,
    structure: list[list[int]],
    scene_platforms: list[list[str]],
    patterns: list[dict],
    platform_info_str: str,
    max_retries: int,
) -> tuple[str, int]:
    """Returns (goal, score). Empty string on failure."""
    prompt = _build_goal_prompt(structure, scene_platforms, patterns, platform_info_str)
    messages = [
        {"role": "system", "content": GOAL_GEN_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(1, max_retries + 1):
        try:
            result = _parse_json_obj(client.complete(model, messages, 1024))
            goal = str(result.get("goal", "")).strip()
            score = int(result.get("score", 3))
            if goal:
                return goal, score
        except Exception as e:
            logger.warning(f"goal gen attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return "", 0


def call_generate_goal_and_platforms(
    client: LLMClient,
    model: str,
    structure: list[list[int]],
    patterns: list[dict],
    candidate_platforms: list[dict],
    platforms: dict[str, dict],
    max_retries: int,
    max_completion_tokens: int = 4096,
) -> tuple[list[list[str]], str, int]:
    """Let LLM choose platforms from candidates and write goal. Returns (scene_platforms, goal, score)."""
    prompt = _build_goal_and_platform_prompt(structure, patterns, candidate_platforms)
    messages = [
        {"role": "system", "content": GOAL_GEN_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(1, max_retries + 1):
        try:
            result = _parse_json_obj(client.complete(model, messages, max_completion_tokens))
            sp_raw = result.get("scene_platforms", [])
            goal = str(result.get("goal", "")).strip()
            score = int(result.get("score", 3))

            if len(sp_raw) != len(structure):
                raise ValueError(f"scene_platforms length {len(sp_raw)} != {len(structure)}")

            sp: list[list[str]] = []
            for i, (scene, chosen) in enumerate(zip(structure, sp_raw)):
                if len(chosen) != len(scene):
                    raise ValueError(f"Scene {i+1}: expected {len(scene)} platforms, got {len(chosen)}")
                for name in chosen:
                    if name not in platforms:
                        raise ValueError(f"Unknown platform: '{name}'")
                sp.append(list(chosen))

            if goal:
                return sp, goal, score
        except Exception as e:
            logger.warning(f"goal+platform gen attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return [], "", 0


def call_plan_scene(
    client: LLMClient,
    model: str,
    goal: str,
    scene_idx: int,
    scene: list[int],
    scene_plats: list[str],
    transition_in: dict | None,
    transition_out: dict | None,
    prev_ops: dict,
    prev_scene_plan: dict,
    max_retries: int,
    max_completion_tokens: int = 8192,
) -> dict:
    """Returns {"platform_order": [...], "plan": {plat: {consumes, role, produces}}}."""
    _empty_plan = lambda: {"consumes": {}, "role": "", "produces": {}}
    if len(scene_plats) == 1:
        return {"platform_order": scene_plats, "plan": {scene_plats[0]: _empty_plan()}}

    prev_ops_returns = _format_ops_returns(prev_ops) if prev_ops else ""
    prompt = _build_plan_prompt(
        goal, scene_idx, scene_plats, scene,
        transition_in, transition_out, prev_ops_returns, prev_scene_plan,
    )
    messages = [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.complete(model, messages, max_completion_tokens)
            result = _extract_json(raw)
            order = result.get("platform_order", [])
            plan = result.get("plan", {})
            if set(order) == set(scene_plats) and all(p in plan for p in scene_plats):
                return result
            logger.warning(f"plan scene {scene_idx + 1} attempt {attempt}/{max_retries}: incomplete plan (got {list(plan.keys())}, expected {scene_plats})")
        except Exception as e:
            raw_len = len(raw) if "raw" in dir() else 0
            logger.warning(f"plan scene {scene_idx + 1} attempt {attempt}/{max_retries} failed ({raw_len} chars): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return {"platform_order": scene_plats, "plan": {p: _empty_plan() for p in scene_plats}}


def call_generate_scene(
    client: LLMClient,
    model: str,
    goal: str,
    scene_idx: int,
    scene: list[int],
    scene_plats: list[str],
    platforms_dict: dict[str, dict],
    transition_in: dict | None,
    transition_out: dict | None,
    prev_ops: dict,
    prev_scene_plan: dict,
    max_retries: int,
    min_steps: int,
    max_steps: int,
    max_completion_tokens: int,
) -> tuple[dict, dict]:
    """Plan then generate each platform sequentially.
    Returns ({"task_operations": {...}, "expected_outcome": {...}}, scene_plan) or ({}, {})."""
    scene_plan = call_plan_scene(
        client, model, goal, scene_idx, scene, scene_plats,
        transition_in, transition_out, prev_ops, prev_scene_plan, max_retries,
        max_completion_tokens=max_completion_tokens,
    )
    platform_order = scene_plan.get("platform_order", scene_plats)
    n_map = {name: n for n, name in zip(scene, scene_plats)}

    scene_ops: dict = {}
    scene_outcomes: dict = {}
    prev_scene_returns = _format_ops_returns(prev_ops) if prev_ops else ""

    for platform_name in platform_order:
        n = n_map.get(platform_name, 1)
        platform_info = _format_platform_info([[platform_name]], platforms_dict)
        platform_plan = scene_plan.get("plan", {}).get(platform_name, {})
        sub_agents_plan: list = platform_plan.get("sub_agents", [])

        # ── Generate each sub-agent's steps individually ──────────────────────
        platform_ops: list = []
        sub_agent_failed = False
        for k in range(n):
            sub_agent_desc = sub_agents_plan[k] if k < len(sub_agents_plan) else f"sub-agent {k+1}"
            prompt = _build_subagent_prompt(
                goal, scene_idx, platform_name, platform_info, platform_plan,
                k, n, sub_agent_desc, platform_ops, scene_ops,
                min_steps, max_steps,
                transition_in=transition_in,
                prev_scene_returns=prev_scene_returns,
            )
            messages = [
                {"role": "system", "content": SCENE_GEN_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            cur_messages = list(messages)
            for attempt in range(1, max_retries + 1):
                try:
                    raw_text = client.complete(model, cur_messages, max_completion_tokens)
                    result = _parse_json_obj(raw_text)
                    steps = result.get("steps")
                    if isinstance(steps, list) and steps:
                        if len(steps) < min_steps:
                            logger.warning(
                                f"scene {scene_idx+1} {platform_name} sub-agent {k+1} "
                                f"attempt {attempt}/{max_retries}: only {len(steps)} steps < min {min_steps}, retrying"
                            )
                            cur_messages = cur_messages + [
                                {"role": "assistant", "content": raw_text},
                                {"role": "user", "content":
                                    f"Your response only has {len(steps)} steps but the minimum is {min_steps}. "
                                    f"Regenerate with at least {min_steps} steps. Return JSON: {{\"steps\": [...]}}"},
                            ]
                            if attempt < max_retries:
                                time.sleep(2 ** attempt)
                            continue
                        platform_ops.append(steps)
                        break
                except Exception as e:
                    logger.warning(f"scene {scene_idx+1} {platform_name} sub-agent {k+1} attempt {attempt}/{max_retries} failed: {e}")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
            else:
                logger.warning(f"scene {scene_idx+1} {platform_name} sub-agent {k+1} failed after {max_retries} attempts")
                sub_agent_failed = True
                break

        if sub_agent_failed:
            return {}, {}

        # ── Generate outcome after all sub-agents complete ────────────────────
        outcome = ""
        outcome_prompt = _build_outcome_prompt(goal, platform_name, platform_plan, platform_ops)
        outcome_messages = [
            {"role": "system", "content": SCENE_GEN_SYSTEM},
            {"role": "user", "content": outcome_prompt},
        ]
        for attempt in range(1, max_retries + 1):
            try:
                result = _parse_json_obj(client.complete(model, outcome_messages, max_completion_tokens))
                outcome = result.get("expected_outcome", "")
                if outcome:
                    break
            except Exception as e:
                logger.warning(f"scene {scene_idx+1} {platform_name} outcome attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        scene_ops[platform_name] = platform_ops
        scene_outcomes[platform_name] = outcome

    return {"task_operations": scene_ops, "expected_outcome": scene_outcomes}, scene_plan


def _extract_json(text: str) -> dict:
    """Extract the first complete JSON object from a model response."""
    text = text.strip()
    if not text:
        raise ValueError("empty response")
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    # Walk to find the matching closing brace
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object")


VALIDATE_SYSTEM = """You are a strict quality reviewer for multi-agent task datasets.
Answer the single question asked. Return ONLY valid JSON, no markdown."""


def _call_validate_one(
    client: LLMClient,
    model: str,
    check_name: str,
    user_content: str,
    max_retries: int,
) -> dict:
    """Single focused validation check. Returns {"ok": bool, "reason": str}."""
    messages = [
        {"role": "system", "content": VALIDATE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.complete(model, messages, 1024 * 16)
            result = _extract_json(raw)
            return {"ok": bool(result.get("ok", False)), "reason": result.get("reason", "")}
        except Exception as e:
            logger.warning(f"validate_one({check_name}) attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return {"ok": False, "reason": f"{check_name} check exhausted retries"}


def call_validate(
    client: LLMClient,
    model: str,
    task: dict,
    structure: list[list[int]],
    scene_platforms: list[list[str]],
    max_retries: int,
    min_steps: int = 5,
    patterns: list[dict] | None = None,
) -> dict:
    """Validate a task: arithmetic pre-check then 5 sequential LLM checks (fail-fast).
    Returns dict with keys: pass, reason, goal_ok, steps_ok, deps_ok, outcome_ok, transitions_ok."""
    task_ops = task.get("task_operations", {})

    # Arithmetic pre-check (no LLM)
    for i, (scene, scene_plats) in enumerate(zip(structure, scene_platforms)):
        for expected, name in zip(scene, scene_plats):
            actual = task_ops.get(name, [])
            if len(actual) != expected:
                reason = f"Scene {i+1} {name}: expected {expected} sub-agents, got {len(actual)}"
                logger.debug(f"arithmetic: {reason}")
                return {"pass": False, "reason": reason, "goal_ok": True, "steps_ok": False, "deps_ok": True, "outcome_ok": True, "transitions_ok": True}
            for k, sub_steps in enumerate(actual):
                if not isinstance(sub_steps, list) or len(sub_steps) < min_steps:
                    n = len(sub_steps) if isinstance(sub_steps, list) else 0
                    reason = f"Scene {i+1} {name} sub-agent {k+1}: {n} steps < {min_steps}"
                    logger.debug(f"arithmetic: {reason}")
                    return {"pass": False, "reason": reason, "goal_ok": True, "steps_ok": False, "deps_ok": True, "outcome_ok": True, "transitions_ok": True}
                _TYPE_NAMES = {"string", "integer", "int", "float", "number", "boolean", "bool", "object", "array", "list", "null", "none"}
                for step in sub_steps:
                    for rv in (step.get("returns") or {}).values():
                        if isinstance(rv, str) and rv.lower() in _TYPE_NAMES:
                            reason = f"Scene {i+1} {name} sub-agent {k+1} step '{step.get('action')}': returns contains type name '{rv}'"
                            logger.debug(f"arithmetic: {reason}")
                            return {"pass": False, "reason": reason, "goal_ok": True, "steps_ok": False, "deps_ok": True, "outcome_ok": True, "transitions_ok": True}

    # Build shared context strings
    goal = task.get("goal", "")
    ops_json = json.dumps(task_ops, ensure_ascii=False, indent=2)
    expected_outcome = task.get("expected_outcome", {})

    transitions_lines: list[str] = []
    if patterns:
        for t, p in enumerate(patterns):
            scene_i_plats = scene_platforms[t] if t < len(scene_platforms) else []
            scene_j_plats = scene_platforms[t + 1] if t + 1 < len(scene_platforms) else []
            transitions_lines.append(
                f"  Scene {t + 1}→{t + 2} ({', '.join(scene_i_plats)} → {', '.join(scene_j_plats)}): "
                f"{p['name']} — {p['desc']}"
            )
    transitions_section = (
        "\nScene transitions:\n" + "\n".join(transitions_lines)
        if transitions_lines else ""
    )

    # 4 sequential focused checks — break on first failure
    checks = [
        ("steps_ok", f"""Check ONLY: For EACH sub-agent's step list — does it have >= {min_steps} steps, each with a realistic action name and non-empty params with specific values, forming a logically ordered workflow? If a step has a returns field, its values must be concrete (NOT type names like "string" or "integer"). steps_ok is false if ANY sub-agent fails. When a platform has N > 1 sub-agents:
- ALLOWED (Rule A): sub-agents may share params that are pre-existing DB values, goal-specified values, or prior-scene returns (e.g. same course_id, project_id, user_id) — as long as each sub-agent performs a different type of operation (e.g. one writes, one reads; or they act on different aspects).
- FAILURE: sub-agents perform truly identical or redundant operations — same action type on the same target with near-identical params and no distinct purpose.
- FAILURE (Rule B): WRITE actions across sub-agents return the same resource identifier (ID, token, URL, reference code) — this means they are not truly independently creating resources. Generic status fields (e.g. status: "success") are exempt.

task_operations:
{ops_json}

Return JSON: {{"ok": bool, "reason": "one-line explanation if ok=false, else empty string"}}"""),

        ("deps_ok", f"""Check ONLY: Does every param value come from a valid source? Valid sources: (1) the goal text, (2) a preceding step's returns field in the same sub-agent, (3) another sub-agent's returns in the same scene, (4) a prior scene's sub-agent returns (referenced in the first step's params of a later scene). System context (user_id, task_id) must never appear as params. A param value is invalid only if it appears in NONE of these four sources.
Rule 5 — cross-sub-agent creation dependency (same platform only): within the same platform, if sub-agent k (k ≥ 2) uses as a param a value that FIRST appears in sub-agent k-1's returns from a WRITE action (add_*, create_*, place_*, book_*, submit_*, or any action that creates a new resource), that is INVALID — each sub-agent must independently obtain shared values via its own query step or create its own instance. This rule does NOT apply across different platforms or across scenes: values produced by one platform's WRITE actions being used by another platform (in the same or a later scene) are valid.

Goal: {goal}
task_operations (each step has a returns field):
{ops_json}

Return JSON: {{"ok": bool, "reason": "one-line explanation if ok=false, else empty string"}}"""),

        ("outcome_ok", f"""Check ONLY: Are all specific values in expected_outcome (IDs, names, ratings, prices, statuses, counts) consistent with the returns fields of the relevant steps in task_operations? No contradictions allowed.

task_operations (each step has a returns field):
{ops_json}

expected_outcome:
{json.dumps(expected_outcome, ensure_ascii=False, indent=2)}

Return JSON: {{"ok": bool, "reason": "one-line explanation if ok=false, else empty string"}}""") if expected_outcome else None,

        ("transitions_ok", f"""Check ONLY: Does each receiving scene actually implement the stated transition pattern?
filter_subset: Scene i+1 operates only on items from Scene i that meet a condition
cascaded_dependency: Scene i+1's params are directly taken from Scene i's output
pick_best: Scene i+1 acts on only the single best result from Scene i
act_on_all: Scene i+1 acts on ALL results from Scene i
aggregate_decide: Scene i+1 makes a decision based on combined signals from Scene i
conditional_branch: Scene i+1 has a conditional path determined by Scene i's outcome
independent: no cross-scene reference required
{transitions_section}

task_operations:
{ops_json}

Return JSON: {{"ok": bool, "reason": "one-line explanation if ok=false, else empty string"}}""") if transitions_lines else None,
    ]

    results: dict = {"steps_ok": True, "deps_ok": True, "outcome_ok": True, "transitions_ok": True}

    for entry in checks:
        if entry is None:
            continue
        check_name, prompt = entry
        r = _call_validate_one(client, model, check_name, prompt, max_retries)
        results[check_name] = r["ok"]
        if not r["ok"]:
            logger.debug(f"Validation failed at {check_name}: {r['reason']}")
            results["pass"] = False
            results["reason"] = r["reason"]
            return results

    results["pass"] = True
    results["reason"] = ""
    return results


REPAIR_SYSTEM = """You are an expert at fixing multi-agent task workflows.
Fix ONLY the described issue. Do not restructure the task.
Return ONLY valid JSON, no markdown."""

_REPAIR_INSTRUCTIONS: dict[str, str] = {
    "cross_subagent_deps": """\
cross-sub-agent creation dependency: sub-agent k uses a value first produced by
sub-agent k-1's write/create action at runtime.

Fix by choosing ONE approach for each such dependency:
(A) Convert to database query: change sub-agent k-1's creation action to a query/lookup
    that retrieves a pre-existing value from the database (e.g. get_cart, get_active_order);
    add an identical independent query step at the start of sub-agent k to retrieve the
    same value. Use natural action naming — the database will have this data pre-populated.
(B) Independent creation: make sub-agent k create its own separate new instance of the
    resource, with no reference to sub-agent k-1's returned value.

Do NOT keep any direct reference from sub-agent k to sub-agent k-1's created IDs.
Do NOT change sub-agent counts.""",

    "deps": """\
deps_ok failed: one or more param values have no valid source.
A param value is VALID if it comes from ANY of these four sources:
  1. Stated explicitly in the goal (user's intent: a name, city, date, product, etc.)
  2. A preceding step's returns field in the SAME sub-agent
  3. Another sub-agent's returns in the SAME scene
  4. A prior scene's sub-agent returns (referenced via cross-scene transition)

A param is INVALID only if it names a specific ID, title, username, or resource that satisfies NONE of the above.

Fix by choosing ONE approach for each invalid param:
  (A) Add the value to the goal — use this when the value represents the user's explicit intent.
  (B) Insert a search/list/get step BEFORE the step that uses the value, with a returns field that produces the needed value.
When inserting a new step, its returns must contain the value that the next step's params reference.
Do NOT change sub-agent counts. Do NOT invent new platforms. Keep expected_outcome unchanged.""",

    "steps": """\
steps_ok failed: one or more sub-agents have too few steps, non-specific params, or a broken workflow order.

Fix by:
  - Adding realistic intermediate steps (confirmation, status check, follow-up) until each sub-agent has at least {min_steps} steps.
  - Making every param value specific and concrete (not empty strings or generic placeholders).
  - Ensuring steps follow a logical order (search before act, create before update, etc.).
  - Each new step must have a returns field: include produced IDs/attributes if downstream steps need them, or {{"status": "success"}} for terminal steps.
Do NOT change the goal, sub-agent counts, or expected_outcome.""",

    "outcome": """\
outcome_ok failed: one or more values in expected_outcome contradict the returns fields of the steps in task_operations.

Fix by rewriting expected_outcome so that:
  - Every specific value (ID, name, rating, price, status, count) matches what the corresponding steps actually produce in their returns fields.
  - No value in expected_outcome is invented or inconsistent with step returns.
Keep task_operations and goal unchanged.""",

    "transitions": """\
transitions_ok failed: one or more scene transitions are not actually implemented by the operations.

Fix by updating task_operations so that the receiving scene's steps reflect the stated transition pattern:
  - filter_subset: Scene i+1 must operate only on items from Scene i that meet the stated condition.
  - cascaded_dependency: Scene i+1's params must be taken directly from Scene i's step returns.
  - pick_best: Scene i+1 must act on only the single best result from Scene i's returns.
  - act_on_all: Scene i+1 must act on ALL results produced by Scene i's returns.
  - aggregate_decide: Scene i+1 must make a decision by combining multiple signals from Scene i's returns.
  - conditional_branch: Scene i+1 must have a conditional path based on Scene i's step returns.
When updating steps, keep each step's params consistent with its preceding steps' returns.
Do NOT change sub-agent counts or goal. Update expected_outcome if needed to stay consistent with the updated task_operations.""",
}


def call_repair_task(
    client: LLMClient,
    model: str,
    task: dict,
    reason: str,
    fail_type: str,
    scene_platforms: list[list[str]],
    max_retries: int,
    min_steps: int = 5,
    max_tokens: int = 32768,
) -> dict | None:
    """Attempt to fix a validation-failed task with targeted instructions per fail_type.
    fail_type: 'deps' | 'steps' | 'goal' | 'outcome' | 'transitions'
    Returns the repaired task dict or None."""
    platform_slots = {
        p: len(task.get("task_operations", {}).get(p, []))
        for scene_plats in scene_platforms
        for p in scene_plats
    }
    instructions = _REPAIR_INSTRUCTIONS.get(fail_type, _REPAIR_INSTRUCTIONS["steps"])
    if fail_type == "steps":
        instructions = instructions.format(min_steps=min_steps)

    expected_outcome = task.get("expected_outcome", {})
    outcome_section = (
        f"\n--- Current expected_outcome ---\n{json.dumps(expected_outcome, ensure_ascii=False, indent=2)}"
        if expected_outcome else ""
    )

    user_content = f"""This multi-agent task failed validation.

Failure reason: {reason}

--- Fix instructions ---
{instructions}

--- Current goal ---
{task.get('goal', '')}

--- Platform sub-agent counts (do NOT change) ---
{json.dumps(platform_slots)}

--- Current task_operations ---
{json.dumps(task.get('task_operations', {}), ensure_ascii=False, indent=2)}{outcome_section}

Return ONLY this JSON (no markdown):
{{"goal": "...", "task_operations": {{...}}, "expected_outcome": {{...}}}}"""

    messages = [
        {"role": "system", "content": REPAIR_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            raw = client.complete(model, messages, max_tokens)
            result = _extract_json(raw)
            if "task_operations" in result or "expected_outcome" in result:
                return result
        except Exception as e:
            logger.warning(f"repair attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return None


# ── Detailed Validation & Repair (Phase 1) ────────────────────────────────────

def _repair_call(client: LLMClient, model: str, prompt: str, max_retries: int, max_tokens: int = 16384) -> dict | None:
    messages = [{"role": "system", "content": REPAIR_SYSTEM}, {"role": "user", "content": prompt}]
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.complete(model, messages, max_tokens)
            return _extract_json(raw)
        except Exception as e:
            logger.warning(f"repair_call attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return None


def _check_and_repair_steps_platform(
    client: LLMClient, model: str, platform: str, platform_ops: list,
    min_steps: int, max_retries: int, max_tokens: int = 16384, task_id: str = "?",
) -> list:
    # ── Python arithmetic pre-check (exact) ──────────────────────────────────
    step_counts = [len(sub) for sub in platform_ops if isinstance(sub, list)]
    count_ok = all(c >= min_steps for c in step_counts)

    if not count_ok:
        short = [c for c in step_counts if c < min_steps]
        logger.debug(f"[{task_id}] steps short {platform} {short} < {min_steps} — repairing")
        ops_str = json.dumps({platform: platform_ops}, ensure_ascii=False, indent=2)
        repair_prompt = f"""Fix the steps for platform: {platform}

Some sub-agents have fewer than {min_steps} steps: {short} steps found, need >= {min_steps}.
Add realistic intermediate steps (confirmation, status check, follow-up) until each sub-agent reaches {min_steps} steps.
Make every param value specific and concrete. Keep logical order. Do NOT change sub-agent counts.

Current task_operations:
{ops_str}

Return JSON: {{"{platform}": [<updated sub-agent steps>]}}"""
        result = _repair_call(client, model, repair_prompt, max_retries, max_tokens=max_tokens)
        if result and platform in result:
            platform_ops = result[platform]

    # ── LLM quality check (realistic values, logical order) ──────────────────
    ops_str = json.dumps({platform: platform_ops}, ensure_ascii=False, indent=2)
    check_prompt = f"""Check ONLY the step quality for platform: {platform}

Do NOT check step counts — only check:
1. Each action name is realistic and domain-appropriate (not generic like "do_task")
2. Every param value is specific and concrete — no empty strings, no type-name placeholders like "string" or "integer"
3. Steps follow a logical order (search before act, create before update)
4. Returns fields (if present) contain concrete values, not type names

task_operations:
{ops_str}

Return JSON: {{"ok": bool, "reason": "one-line if ok=false, else empty"}}"""

    r = _call_validate_one(client, model, "steps_quality", check_prompt, max_retries)
    if r["ok"]:
        return platform_ops

    logger.debug(f"[{task_id}] steps quality failed {platform}: {r['reason']}")
    repair_prompt = f"""Fix the step quality for platform: {platform}

Problem: {r['reason']}

Make action names realistic. Replace any generic/placeholder param values with specific concrete values.
Fix step ordering if needed. Do NOT change sub-agent counts or add/remove steps.

Current task_operations:
{ops_str}

Return JSON: {{"{platform}": [<updated sub-agent steps>]}}"""

    result = _repair_call(client, model, repair_prompt, max_retries, max_tokens=max_tokens)
    if result and platform in result:
        return result[platform]
    return platform_ops


def _check_and_repair_returns_platform(
    client: LLMClient, model: str, platform: str, platform_ops: list,
    goal: str, platform_plan: dict, scene_ops: dict,
    max_retries: int, max_tokens: int = 16384, task_id: str = "?",
) -> list:
    """Check that a platform's returns are logically consistent with the full scene context.
    Considers the platform's own params, other platforms' ops, and sub-agent interactions."""
    ops_str = json.dumps({platform: platform_ops}, ensure_ascii=False, indent=2)
    produces_str = json.dumps(platform_plan.get("produces", {}), ensure_ascii=False)

    others = {p: ops for p, ops in scene_ops.items() if p != platform}
    scene_ctx = (
        f"\nOther platforms in this scene:\n{json.dumps(others, ensure_ascii=False, indent=2)}"
        if others else ""
    )

    check_prompt = f"""Review all return values for platform {platform} and identify logical inconsistencies.

Goal: {goal}
Expected produces (planned): {produces_str}

{platform} task_operations (ALL sub-agents):
{ops_str}{scene_ctx}

Check for issues such as:
- A returned date/time is outside the range or window given in that step's params
- A returned ID, name, or entity doesn't match what the step was acting on
- A returned location or address belongs to the wrong party (e.g. patient's home address returned instead of clinic address)
- A returned value contradicts what another sub-agent or platform in the same scene established
- A returned status is impossible given the action (e.g. "confirmed" when params show no valid slot)
- A returned price, count, or quantity violates constraints visible in the scene

Only flag clear logical contradictions — do NOT flag values that are merely invented or unverifiable in isolation.

Return JSON:
{{
  "issues": [
    {{"action": "...", "sub_agent": <1-based index>, "return_key": "...", "bad_value": "...", "reason": "...", "correct_value": "..."}}
  ]
}}
Return {{"issues": []}} if all returns are consistent."""

    for attempt in range(1, max_retries + 1):
        try:
            raw = client.complete(model, [{"role": "system", "content": VALIDATE_SYSTEM}, {"role": "user", "content": check_prompt}], 1024 * 16)
            result = _extract_json(raw)
            issues = result.get("issues", [])
            if not issues:
                return platform_ops

            issues_str = "\n".join(
                f"  - sub-agent {iss.get('sub_agent','?')} {iss['action']} returns['{iss['return_key']}'] = {json.dumps(iss.get('bad_value', ''))} — {iss['reason']} → should be: {json.dumps(iss.get('correct_value', ''))}"
                for iss in issues
            )
            logger.debug(f"[{task_id}] returns inconsistency {platform} ({len(issues)} issue(s)):\n{issues_str}")

            repair_prompt = f"""Fix return value inconsistencies for platform {platform}.
Do NOT change params, action names, step order, or sub-agent counts. Only correct the specific return values listed.

Issues to fix:
{issues_str}

Goal: {goal}

{platform} task_operations:
{ops_str}{scene_ctx}

Return JSON: {{"{platform}": [<updated sub-agent steps>]}}"""

            fixed = _repair_call(client, model, repair_prompt, max_retries, max_tokens=max_tokens)
            if fixed and platform in fixed:
                return fixed[platform]
            return platform_ops
        except Exception as e:
            logger.warning(f"[{task_id}] returns check {platform} attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return platform_ops


def _check_and_repair_deps_platform(
    client: LLMClient, model: str, goal: str, platform: str, platform_ops: list,
    scene_ops: dict, prev_scene_returns: str, max_retries: int, max_tokens: int = 16384,
    task_id: str = "?", platform_plan: dict | None = None, platform_outcome: str = "",
) -> tuple[str, list]:

    # ── Step 0: cross-sub-agent creation dependency check ────────────────────
    if len(platform_ops) > 1:
        ops_str_0 = json.dumps({platform: platform_ops}, ensure_ascii=False, indent=2)
        cross_check_prompt = f"""Check for two types of cross-sub-agent creation violations in {platform}'s task_operations.

Type A — param dependency: sub-agent k (k > 0) uses as a param a value that FIRST appears in a
previous sub-agent's returns from a WRITE action (add_*, create_*, place_*, book_*, submit_*, or
any action that creates a new resource at runtime).

Type B — duplicate write return: two or more sub-agents have WRITE actions that return the same
resource identifier (same ID, URL, token, reference code, etc.), meaning they are NOT truly
independently creating resources. Generic status fields (e.g. status: "success") are exempt.
Each sub-agent's WRITE actions must produce unique IDs/values.
READ actions (get_*, fetch_*, list_*, search_*, retrieve_*) returning the same ID as another sub-agent's
action is expected and normal — do NOT flag these as Type B violations.

{platform} task_operations:
{ops_str_0}

For each violation found, include a "violation_type" field ("param" or "duplicate_return") plus:
- Type A: consuming_subagent (1-based), param_key, value, producing_action
- Type B: subagent_a (1-based), action_a (the action in sub-agent A that returns the value), subagent_b (1-based), action_b (the action in sub-agent B that returns the value), shared_value, field_key

Return JSON:
{{"dependencies": [{{"violation_type": "param", "consuming_subagent": <int>, "param_key": "...", "value": "...", "producing_action": "..."}} | {{"violation_type": "duplicate_return", "subagent_a": <int>, "action_a": "...", "subagent_b": <int>, "action_b": "...", "shared_value": "...", "field_key": "..."}}]}}
Return {{"dependencies": []}} if none found."""

        cross_result = _repair_call(client, model, cross_check_prompt, max_retries, max_tokens=min(max_tokens, 4096))
        if cross_result and cross_result.get("dependencies"):
            dep_lines = []
            for d in cross_result["dependencies"]:
                if d.get("violation_type") == "duplicate_return":
                    dep_lines.append(
                        f"  - [duplicate_return] sub-agent {d.get('subagent_a', '?')} ('{d.get('action_a', '?')}') and "
                        f"sub-agent {d.get('subagent_b', '?')} ('{d.get('action_b', '?')}') "
                        f"both return '{d.get('field_key', '?')}' = {json.dumps(d.get('shared_value', ''))}"
                    )
                else:
                    dep_lines.append(
                        f"  - [param] sub-agent {d['consuming_subagent']} param '{d.get('param_key', '?')}' = "
                        f"{json.dumps(d.get('value', ''))} "
                        f"(from sub-agent {d.get('consuming_subagent', 1) - 1} action '{d.get('producing_action', '')}')"
                    )
            deps_str_0 = "\n".join(dep_lines)
            logger.debug(f"[{task_id}] cross-subagent deps {platform} ({len(cross_result['dependencies'])} found):\n{deps_str_0}")

            cross_repair_prompt = f"""Fix cross-sub-agent creation violations in {platform}.

Violations found:
{deps_str_0}

For [param] violations (sub-agent k uses a value from sub-agent k-1's write action):
(A) Convert to database query: change sub-agent k-1's creation action to a query/lookup that
    retrieves a pre-existing value from the database (e.g. get_cart, get_active_order); add an
    identical independent query step at the start of sub-agent k to retrieve the same value.
(B) Independent creation: make sub-agent k create its own separate new instance of the resource,
    removing any reference to sub-agent k-1's created value entirely.

For [duplicate_return] violations (two sub-agents' WRITE actions return the same ID/value):
Change sub-agent B's WRITE action to return a new, unique ID/value (different from sub-agent A's),
then update all subsequent steps in sub-agent B that reference the old value to use the new one.

Do NOT change sub-agent counts. Do NOT keep any direct reference from sub-agent k to sub-agent k-1's created IDs.

Goal: {goal}

{platform} task_operations:
{ops_str_0}

Return JSON: {{"{platform}": [<updated sub-agent steps>]}}"""

            fixed_0 = _repair_call(client, model, cross_repair_prompt, max_retries, max_tokens=max_tokens)
            if fixed_0 and platform in fixed_0:
                platform_ops = fixed_0[platform]
                logger.debug(f"[{task_id}] cross-subagent deps repaired for {platform}")

    # ── Step 1: replace placeholder values first ───────────────────────────────
    ops_str = json.dumps({platform: platform_ops}, ensure_ascii=False, indent=2)
    ph_prompt = f"""Inspect ALL param and return values for platform {platform}.
Flag any value that is clearly a placeholder/fake — e.g. contains "example.com", "test_", "dummy", "***",
patterns like "user_123", "buyer_001", "club_buyer_3", generic tokens like "token_xxx", or any obviously non-realistic string.

{platform} task_operations:
{ops_str}

For each placeholder found, provide a realistic replacement that fits the context.
Return JSON: {{"placeholders": [{{"action": "...", "field": "params or returns", "key": "...", "old": "...", "new": "..."}}]}}
Return {{"placeholders": []}} if all values are realistic."""

    ph_result = _repair_call(client, model, ph_prompt, max_retries, max_tokens=min(max_tokens, 4096))
    if ph_result and ph_result.get("placeholders"):
        replace_map: dict[tuple, str] = {
            (p["action"], p["field"], p["key"]): p["new"]
            for p in ph_result["placeholders"]
            if p.get("action") and p.get("field") and p.get("key") and p.get("new")
        }
        for sub in platform_ops:
            for step in sub:
                action = step.get("action", "")
                for field in ("params", "returns"):
                    d = step.get(field, {})
                    for key in list(d.keys()):
                        replacement = replace_map.get((action, field, key))
                        if replacement:
                            d[key] = replacement
        logger.debug(f"[{task_id}] replaced {len(replace_map)} placeholder(s) in {platform}")

    # ── Step 2: classify params as mismatch vs missing ────────────────────────
    ops_str = json.dumps({platform: platform_ops}, ensure_ascii=False, indent=2)
    others = {p: ops for p, ops in scene_ops.items() if p != platform}
    scene_ctx = f"\nOther platforms in same scene:\n{json.dumps(others, ensure_ascii=False, indent=2)}" if others else ""
    prev_ctx = f"\nPrevious scene returns:\n{prev_scene_returns}" if prev_scene_returns else ""
    plan_ctx = f"\nThis platform's plan (consumes/role/produces):\n{json.dumps(platform_plan, ensure_ascii=False, indent=2)}" if platform_plan else ""
    outcome_ctx = f"\nThis platform's expected_outcome:\n{platform_outcome}" if platform_outcome else ""

    check_prompt = f"""For each concrete param value in {platform}'s task_operations, classify it as:
- "ok": correct and matches its source
- "mismatch": a source exists (goal or returns) but the param uses a DIFFERENT value — report the correct value
- "missing": no source exists at all

Valid sources: (1) goal text, (2) preceding step returns in same sub-agent, (3) same-scene platform returns, (4) previous scene returns.
The plan's "consumes" shows what values this platform should be using — treat those as the intended source.

EXACT-VALUE rule — do NOT let the agent guess arbitrary values:
- If a param is an EXACT ARBITRARY token that the agent must reproduce character-for-character — a specific enum value (e.g. 'warm_tone', 'Open - Ready for Outreach', 'full vehicle history'), an exact title/tag/subject string, a specific hex color, a specific code, or a specific stored format (e.g. '08:00 AM') — and that exact token does NOT appear VERBATIM in the goal and is NOT returned by a prior read step, classify it as "missing". The agent could not know such a value, so it must be stated in the goal.
- Only skip truly universal constants the agent is expected to know without being told (e.g. HTTP 200, a currency code like 'USD', the literal boolean true/false, the number of items the goal already names). When in doubt, treat an exact arbitrary token as "missing" rather than skipping it.

Goal: {goal}{plan_ctx}{outcome_ctx}

{platform} task_operations:
{ops_str}{scene_ctx}{prev_ctx}

Return JSON:
{{
  "mismatches": [{{"action": "...", "param_key": "...", "wrong_value": "...", "correct_value": "..."}}],
  "missing": [{{"action": "...", "param_key": "...", "param_value": "..."}}]
}}
Return {{"mismatches": [], "missing": []}} if all params are correct."""

    for attempt in range(1, max_retries + 1):
        try:
            raw = client.complete(model, [{"role": "system", "content": VALIDATE_SYSTEM}, {"role": "user", "content": check_prompt}], 1024 * 16)
            result = _extract_json(raw)
            mismatches = result.get("mismatches", [])
            missing = result.get("missing", [])

            if not mismatches and not missing:
                break

            mismatch_str = "\n".join(
                f"  - {m['action']} param '{m['param_key']}' = {json.dumps(m.get('wrong_value', ''))} (source has: {json.dumps(m.get('correct_value', ''))})"
                for m in mismatches
            )
            missing_str = "\n".join(
                f"  - {m['action']} param '{m['param_key']}' = {json.dumps(m['param_value'])}"
                for m in missing
            )

            mismatch_block = f"\nMismatched params (a source exists but the ops value doesn't match it):\n{mismatch_str}" if mismatches else ""
            missing_block = f"\nMissing params (no source found anywhere):\n{missing_str}" if missing else ""

            repair_prompt = f"""Fix param source issues for platform {platform}. Do NOT change sub-agent counts.
{mismatch_block}{missing_block}

For each issue above, choose whichever fix makes the most sense in context:
  (A) Correct the ops param value to match what the goal or a prior step's returns actually says
  (B) Insert a new step earlier in the sub-agent to produce the needed value (use this when the value is data-derived and discoverable at runtime, e.g. an ID from a search/list)
  (C) Add the value to the goal text — use this for the user's explicit intent (target city, product name, budget) AND for any EXACT ARBITRARY token the agent must reproduce but cannot discover (a specific enum value, exact title/tag/subject string, hex color, code, or stored format). State the exact token verbatim in the goal so the agent never has to guess it.
You may use different fixes for different params. You may also change both goal and ops if needed.

Goal: {goal}{plan_ctx}{outcome_ctx}

{platform} task_operations:
{ops_str}{scene_ctx}{prev_ctx}

Return JSON: {{"goal": "updated or unchanged goal", "{platform}": [<updated or unchanged ops>]}}"""

            fixed = _repair_call(client, model, repair_prompt, max_retries, max_tokens=max_tokens)
            if fixed:
                goal = fixed.get("goal", goal) or goal
                platform_ops = fixed.get(platform, platform_ops)
                logger.debug(f"[{task_id}] deps repaired {platform}: {len(mismatches)} mismatches, {len(missing)} missing")

            break
        except Exception as e:
            logger.warning(f"[{task_id}] deps check {platform} attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    return goal, platform_ops


def _check_and_repair_outcome_scene(
    client: LLMClient, model: str, scene_ops: dict, scene_outcome: dict,
    max_retries: int, max_tokens: int = 16384, task_id: str = "?",
) -> dict:
    ops_str = json.dumps(scene_ops, ensure_ascii=False, indent=2)
    outcome_str = json.dumps(scene_outcome, ensure_ascii=False, indent=2)

    check_prompt = f"""Check ONLY: Are all specific values in expected_outcome consistent with the step returns?

task_operations:
{ops_str}

expected_outcome:
{outcome_str}

Return JSON: {{"ok": bool, "reason": "one-line if ok=false, else empty"}}"""

    r = _call_validate_one(client, model, "outcome", check_prompt, max_retries)
    if r["ok"]:
        return scene_outcome

    orig_snippet = {p: str(v)[:120] for p, v in scene_outcome.items()}
    logger.debug(f"[{task_id}] outcome failed (original: {orig_snippet}): {r['reason']}")
    repair_prompt = f"""Fix expected_outcome to match the step returns.

Problem: {r['reason']}

Rewrite expected_outcome so every specific value matches step returns. Do NOT change task_operations.

task_operations:
{ops_str}

Current expected_outcome:
{outcome_str}

Return JSON: {{"expected_outcome": {{<platform>: "...", ...}}}}"""

    result = _repair_call(client, model, repair_prompt, max_retries, max_tokens=max_tokens)
    if result and "expected_outcome" in result:
        return result["expected_outcome"]
    return scene_outcome


def _check_transition(
    client: LLMClient, model: str,
    scene_i_ops: dict, scene_j_ops: dict,
    scene_i_outcomes: dict, scene_j_outcomes: dict,
    pattern: dict, t: int, max_retries: int, task_id: str = "?",
) -> bool:
    """Returns True if transition is correctly implemented, False to discard the task."""
    ops_i_str = json.dumps(scene_i_ops, ensure_ascii=False, indent=2)
    ops_j_str = json.dumps(scene_j_ops, ensure_ascii=False, indent=2)
    outcome_i_str = json.dumps(scene_i_outcomes, ensure_ascii=False, indent=2)
    outcome_j_str = json.dumps(scene_j_outcomes, ensure_ascii=False, indent=2)

    check_prompt = f"""Check ONLY: Does scene {t+2} properly implement the '{pattern['name']}' transition from scene {t+1}?

Transition — '{pattern['name']}': {pattern['desc']}{_pattern_rule_block(pattern['name'])}

Check both:
1. The transition EFFECT is correctly realized — scene {t+2}'s behavior matches the REQUIRED criteria above
2. At least one step in scene {t+2} actually uses a return value from scene {t+1}

Scene {t+1} task_operations:
{ops_i_str}

Scene {t+1} expected_outcome:
{outcome_i_str}

Scene {t+2} task_operations:
{ops_j_str}

Scene {t+2} expected_outcome:
{outcome_j_str}

Return JSON: {{"ok": bool, "reason": "one-line if ok=false, else empty"}}"""

    r = _call_validate_one(client, model, "transitions", check_prompt, max_retries)
    if not r["ok"]:
        i_plats = "/".join(scene_i_ops.keys())
        j_plats = "/".join(scene_j_ops.keys())
        logger.debug(
            f"[{task_id}] transition scene {t+1}→{t+2} ({i_plats}→{j_plats}) "
            f"pattern='{pattern['name']}' failed — discarding: {r['reason']}"
        )
    return r["ok"]


def detailed_validate_and_repair(
    client: LLMClient, model: str, task: dict,
    structure: list[list[int]], scene_platforms: list[list[str]],
    patterns: list[dict], min_steps: int, max_retries: int, max_tokens: int = 16384,
    task_id: str = "?",
    platform_plans: dict[str, dict] | None = None,
) -> dict | None:
    """Phase 1: targeted per-platform then per-scene validation + repair.
    Transition check is check-only — returns None to discard if any transition fails."""
    goal = task.get("goal", "")
    all_ops: dict = dict(task.get("task_operations", {}))
    all_outcomes: dict = dict(task.get("expected_outcome", {}))

    # ── 1: per-transition check only — discard on failure ─────────────────────
    for t, pattern in enumerate(patterns):
        if pattern["name"] == "independent":
            continue
        scene_i_plats = scene_platforms[t]
        scene_j_plats = scene_platforms[t + 1]
        scene_i_ops = {p: all_ops.get(p, []) for p in scene_i_plats}
        scene_j_ops = {p: all_ops.get(p, []) for p in scene_j_plats}
        scene_i_outcomes = {p: all_outcomes.get(p, "") for p in scene_i_plats}
        scene_j_outcomes = {p: all_outcomes.get(p, "") for p in scene_j_plats}

        if not _check_transition(
            client, model, scene_i_ops, scene_j_ops,
            scene_i_outcomes, scene_j_outcomes, pattern, t, max_retries, task_id,
        ):
            return None

    # ── 2: per-platform steps → deps (modifies ops) ───────────────────────────
    for si, (_, scene_plats) in enumerate(zip(structure, scene_platforms)):
        prev_returns = ""
        if si > 0:
            prev_plats = scene_platforms[si - 1]
            prev_returns = _format_ops_returns({p: all_ops.get(p, []) for p in prev_plats})

        scene_ops = {p: all_ops.get(p, []) for p in scene_plats}

        for platform in scene_plats:
            fixed_ops = _check_and_repair_steps_platform(
                client, model, platform, all_ops.get(platform, []), min_steps, max_retries, max_tokens, task_id,
            )
            all_ops[platform] = fixed_ops
            scene_ops[platform] = fixed_ops

            platform_plan = (platform_plans or {}).get(platform, {})
            fixed_ops = _check_and_repair_returns_platform(
                client, model, platform, fixed_ops, goal, platform_plan, scene_ops, max_retries, max_tokens, task_id,
            )
            all_ops[platform] = fixed_ops
            scene_ops[platform] = fixed_ops

            goal, fixed_ops = _check_and_repair_deps_platform(
                client, model, goal, platform, fixed_ops, scene_ops, prev_returns, max_retries, max_tokens,
                task_id=task_id,
                platform_plan=(platform_plans or {}).get(platform),
                platform_outcome=all_outcomes.get(platform, ""),
            )
            all_ops[platform] = fixed_ops
            scene_ops[platform] = fixed_ops

    # ── 3: per-scene outcome (ops now final) ──────────────────────────────────
    for si, scene_plats in enumerate(scene_platforms):
        scene_ops = {p: all_ops.get(p, []) for p in scene_plats}
        scene_outcome = {p: all_outcomes[p] for p in scene_plats if p in all_outcomes}
        if not scene_outcome:
            continue
        fixed_outcome = _check_and_repair_outcome_scene(
            client, model, scene_ops, scene_outcome, max_retries, max_tokens, task_id,
        )
        all_outcomes.update(fixed_outcome)

    return {"goal": goal, "task_operations": all_ops, "expected_outcome": all_outcomes}


def get_embedding(client: OpenAI, model: str, text: str) -> np.ndarray:
    response = client.embeddings.create(model=model, input=text)
    return np.array(response.data[0].embedding, dtype=np.float32)


def is_too_similar(emb: np.ndarray, accepted: list[np.ndarray], threshold: float) -> bool:
    if not accepted:
        return False
    matrix = np.stack(accepted)
    norm_emb = emb / (np.linalg.norm(emb) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    sims = (matrix / norms) @ norm_emb
    return float(sims.max()) > threshold


# ── Run ────────────────────────────────────────────────────────────────────────

def run(args: PipelineConfig) -> None:
    platforms = load_platforms(args.scenario_output)
    existing_ids, structure_counts = load_existing_tasks(args.tasks_output)
    existing_embeddings = load_existing_embeddings(args.embeddings_output, args.embed_model)
    platform_counts = load_platform_counts(args.tasks_output)
    pattern_counts = load_pattern_counts(args.tasks_output)

    gen_client = LLMClient.from_config(args)
    embed_client =OpenAI(
        api_key=args.embed_api_key or os.environ.get("EMBEDDING_OPENAI_API_KEY")
              or os.environ.get("OPENAI_API_KEY"),
        # pin to real OpenAI so OPENAI_BASE_URL (now the local GLM endpoint) doesn't redirect embeddings
        base_url=os.environ.get("EMBEDDING_OPENAI_BASE_URL") or "https://api.openai.com/v1",
    )

    # ── Step 1: determine which (budget, structure) pairs need more tasks ──────
    all_jobs: list[tuple[int, list[int], int]] = []

    for budget in args.budget_list:
        existing_for_budget = {sk: cnt for (b, sk), cnt in structure_counts.items() if b == budget}

        for sk, cnt in existing_for_budget.items():
            need = args.tasks_per_structure - cnt
            if need > 0:
                all_jobs.append((budget, json.loads(sk), need))

        new_needed = args.num_structures - len(existing_for_budget)
        jobs_before = len(all_jobs)
        if new_needed > 0:
            new_structures = sample_unique_structures(
                budget, new_needed, set(existing_for_budget.keys()),
                args.max_scenes, args.max_platforms_per_scene, args.max_agents_per_platform,
                max_attempts=args.max_structure_sample_attempts,
            )
            logger.info(f"Budget {budget}: adding {len(new_structures)} new structures")
            for s in new_structures:
                all_jobs.append((budget, s, args.tasks_per_structure))

        completed = sum(1 for cnt in existing_for_budget.values() if cnt >= args.tasks_per_structure)
        logger.info(
            f"Budget {budget}: {completed}/{len(existing_for_budget)} structures complete, "
            f"{len(all_jobs) - jobs_before} new jobs queued"
        )

    if not all_jobs:
        logger.success("All structures already have enough tasks.")
        return

    logger.info(f"Total structures to process: {len(all_jobs)}")

    # Shared mutable state — protected by write_lock
    write_lock = threading.Lock()

    # ── Per-structure generator (sequential within, parallel across) ──────────
    def generate_for_structure(
        budget: int,
        structure: list[list[int]],
        need: int,
        init_embeddings: list[np.ndarray],
        pos: int = 0,
    ) -> int:
        sk = json.dumps(structure)
        num_transitions = len(structure) - 1
        accepted_embeddings = list(init_embeddings)
        accepted_count = 0
        attempts = 0
        max_attempts = need * args.max_attempts_multiplier
        desc = f"b{budget} {sk[:30]}"
        pbar = tqdm(total=need, desc=desc, position=pos, leave=True, unit="task")

        while accepted_count < need and attempts < max_attempts:
            attempts += 1

            with write_lock:
                current_counts = dict(platform_counts)
                current_pattern_counts = dict(pattern_counts)

            patterns = _assign_transition_patterns(num_transitions, current_pattern_counts)
            candidates = sample_candidate_platforms(platforms, current_counts, n_candidates=50)

            # ── Step 1: LLM chooses platforms + writes goal ────────────────────
            sp, goal, score = call_generate_goal_and_platforms(
                gen_client, args.gen_model,
                structure, patterns, candidates, platforms,
                args.max_retries, args.max_completion_tokens,
            )
            if not sp or not goal or score < args.validity_threshold:
                continue

            task_id = str(uuid.uuid4())

            # ── Step 2: Generate each scene's operations sequentially ──────────
            all_ops: dict = {}
            all_outcomes: dict = {}
            all_platform_plans: dict = {}
            all_scene_plans: list = []
            prev_ops: dict = {}
            prev_scene_plan: dict = {}
            scene_failed = False

            for i, (scene, scene_plats) in enumerate(zip(structure, sp)):
                transition_in = patterns[i - 1] if i > 0 else None
                transition_out = patterns[i] if i < len(patterns) else None

                scene_result, scene_plan = call_generate_scene(
                    gen_client, args.gen_model,
                    goal, i, scene, scene_plats, platforms,
                    transition_in, transition_out, prev_ops, prev_scene_plan,
                    args.max_retries, args.min_steps_per_subagent,
                    args.max_steps_per_subagent, args.max_completion_tokens,
                )
                if not scene_result:
                    scene_failed = True
                    break

                scene_ops = scene_result.get("task_operations", {})

                # Strip extra platforms the model may have hallucinated
                assigned = set(scene_plats)
                for k in set(scene_ops.keys()) - assigned:
                    logger.debug(f"Scene {i+1}: stripping extra platform {k}")
                    scene_ops.pop(k)

                all_ops.update(scene_ops)
                all_outcomes.update(scene_result.get("expected_outcome", {}))
                for p in scene_plats:
                    all_platform_plans[p] = scene_plan.get("plan", {}).get(p, {})
                all_scene_plans.append(scene_plan)
                prev_ops = scene_ops
                prev_scene_plan = scene_plan

            if scene_failed:
                continue

            # ── Step 3: Phase 1 — detailed per-platform/scene/transition repair ──
            raw = {"goal": goal, "task_operations": all_ops, "expected_outcome": all_outcomes}
            raw = detailed_validate_and_repair(
                gen_client, args.gen_model, raw, structure, sp, patterns,
                args.min_steps_per_subagent, args.max_retries, args.max_completion_tokens,
                task_id=task_id, platform_plans=all_platform_plans,
            )
            if raw is None:
                logger.opt(colors=True).info(f"<red>[{task_id}] Phase 1: transition check failed — discarding</red>")
                continue
            goal = raw["goal"]
            all_ops = raw["task_operations"]
            all_outcomes = raw["expected_outcome"]

            # ── Step 4: Phase 2 — coarse validation (discard if fails) ──────────
            vresult = call_validate(
                gen_client, args.model, raw, structure,
                sp, args.max_retries, args.min_steps_per_subagent,
                patterns=patterns,
            )
            if not vresult.get("pass", False):
                # Escape '<' so loguru's color parser doesn't treat LLM reason text
                # like "<integer>" as a color tag.
                _vreason = str(vresult.get("reason", "")).replace("<", r"\<")
                if not vresult.get("transitions_ok", True):
                    logger.opt(colors=True).info(f"<red>[Phase 2] transitions warning (keeping task): {_vreason}</red>")
                else:
                    logger.opt(colors=True).info(f"<red>[Phase 2] discard: {_vreason}</red>")
                    continue

            # ── Step 4: Embedding dedup ───────────────────────────────────────
            try:
                emb = get_embedding(embed_client, args.embed_model, goal)
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")
                continue

            if is_too_similar(emb, accepted_embeddings, args.embed_threshold):
                logger.opt(colors=True).info(f"<red>[Dedup] discard: goal too similar to existing task</red>")
                continue

            # ── Step 5: Commit ────────────────────────────────────────────────
            task: dict = {
                "task_id": task_id,
                "goal": goal,
                "metadata": {
                    "budget": budget,
                    "structure": structure,
                    "scene_platforms": sp,
                    "task_operations": all_ops,
                    "expected_outcome": all_outcomes,
                    "scene_transitions": [
                        {"from": t, "to": t + 1, "pattern": p["name"]}
                        for t, p in enumerate(patterns)
                    ],
                    "score": score,
                },
            }

            with write_lock:
                if task["task_id"] in existing_ids:
                    continue
                append_task(args.tasks_output, task)
                with open(args.task_plans_output, "a", encoding="utf-8") as _pf:
                    _pf.write(json.dumps({"task_id": task_id, "scene_plans": all_scene_plans}, ensure_ascii=False) + "\n")
                append_embedding(args.embeddings_output, task["task_id"],
                                 args.embed_model, budget, sk, emb)
                existing_ids.add(task["task_id"])
                for scene_plats in sp:
                    for p in scene_plats:
                        platform_counts[p] += 1
                for pat in patterns:
                    pattern_counts[pat["name"]] += 1

            accepted_count += 1
            accepted_embeddings.append(emb)
            pbar.update(1)

        pbar.close()
        logger.info(f"Budget {budget}, structure {sk[:40]}: {accepted_count}/{need} accepted in {attempts} attempts")
        return accepted_count

    # ── Run structures in parallel ─────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                generate_for_structure,
                budget, structure, need,
                list(existing_embeddings.get((budget, json.dumps(structure)), [])),
                pos,
            ): (budget, json.dumps(structure))
            for pos, (budget, structure, need) in enumerate(all_jobs)
        }
        for future in as_completed(futures):
            future.result()

    with write_lock:
        total = len(existing_ids)
    logger.success(f"Done. {total} total tasks in {args.tasks_output}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Generate multi-agent tasks")
    parser.add_argument("--scenario_output", type=str, default=defaults.scenario_output)
    parser.add_argument("--tasks_output", type=str, default=defaults.tasks_output)
    parser.add_argument("--model", type=str, default=defaults.model)
    parser.add_argument("--embed_model", type=str, default=defaults.embed_model)
    parser.add_argument("--api_key", type=str, default=defaults.api_key)
    parser.add_argument("--base_url", type=str, default=defaults.base_url)
    parser.add_argument("--embed_api_key", type=str, default=defaults.embed_api_key)
    parser.add_argument("--budget_list", type=int, nargs="+", default=defaults.budget_list)
    parser.add_argument("--num_structures", type=int, default=defaults.num_structures)
    parser.add_argument("--tasks_per_structure", type=int, default=defaults.tasks_per_structure)
    parser.add_argument("--max_scenes", type=int, default=defaults.max_scenes)
    parser.add_argument("--max_platforms_per_scene", type=int, default=defaults.max_platforms_per_scene)
    parser.add_argument("--max_agents_per_platform", type=int, default=defaults.max_agents_per_platform)
    parser.add_argument("--embed_threshold", type=float, default=defaults.embed_threshold)
    parser.add_argument("--validity_threshold", type=int, default=defaults.validity_threshold)
    parser.add_argument("--min_steps_per_subagent", type=int, default=defaults.min_steps_per_subagent)
    parser.add_argument("--max_steps_per_subagent", type=int, default=defaults.max_steps_per_subagent)
    parser.add_argument("--concurrency", type=int, default=defaults.concurrency)
    parser.add_argument("--max_retries", type=int, default=defaults.max_retries)
    parser.add_argument("--max_attempts_multiplier", type=int, default=defaults.max_attempts_multiplier)
    parsed = parser.parse_args()

    cfg = PipelineConfig()
    for k, v in vars(parsed).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    run(cfg)


if __name__ == "__main__":
    main()
