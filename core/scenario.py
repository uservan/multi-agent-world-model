import os
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from loguru import logger
from utils.llm import LLMClient
from tqdm import tqdm

from core.config import PipelineConfig


FEW_SHOT_EXAMPLES = [
    {
        "name": "Amazon",
        "description": "Amazon is the world's largest e-commerce platform where users can browse millions of products, compare prices, read reviews, and place orders. Users manage their account including saved addresses, payment methods, wish lists, and purchase history. The platform supports order tracking, returns, and subscription-based Prime membership for expedited shipping.",
        "features": [
            "product catalog browsing and filtering",
            "shopping cart add/update/remove",
            "wish lists and saved items",
            "order creation and order history",
            "order status tracking",
            "returns and refunds",
            "customer reviews and ratings",
            "shipping address management",
            "payment method management",
        ],
    },
    {
        "name": "GitHub",
        "description": "GitHub is a code hosting and collaboration platform used by developers to manage source code with Git version control. Users can create repositories, open pull requests, file issues, review code, and track project progress with boards and milestones. Teams collaborate through comments, labels, and fine-grained access permissions.",
        "features": [
            "repository creation and settings management",
            "branch and commit management",
            "pull request creation, review, and merge",
            "issue tracking with labels and milestones",
            "project boards and task cards",
            "team access control and permissions",
            "release and tag management",
            "user and organization profiles",
        ],
    },
    {
        "name": "Airbnb",
        "description": "Airbnb is a vacation rental marketplace where hosts list properties and guests search, book, and review stays. Users can filter listings by location, dates, price, and amenities, then manage reservations including cancellations and refunds. Hosts manage their listings, pricing calendars, and guest communication.",
        "features": [
            "property listing search and filtering",
            "booking creation and cancellation",
            "reservation management for guests and hosts",
            "pricing and availability calendar",
            "guest and host reviews",
            "wishlist for saved properties",
            "payment and payout management",
            "messaging between guests and hosts",
        ],
    },
]

SYSTEM_PROMPT = """You are an expert at analyzing websites, apps, and digital platforms. For each platform, produce a concise description of what users can DO on the platform and a list of concrete features (operations, workflows, data entities) that could be implemented as API endpoints.

Focus on actionable, CRUD-style features: things like browsing, searching, creating orders, managing accounts, tracking status, leaving reviews, etc.

Return ONLY valid JSON, no markdown fences."""

USER_PROMPT_TEMPLATE = """Here are {num_examples} example platforms:

{examples}

---

Now analyze the following platform in the same format:

Platform name: {name}
Category: {category}
Subcategory: {subcategory}

Output format (must be valid JSON):
{{
    "name": "{name}",
    "description": "2-4 sentence description focusing on what users can DO, what entities exist, and what workflows are available",
    "features": ["list of concrete features or operations available on the platform"]
}}"""


def format_few_shot_examples(examples: list[dict]) -> str:
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"{i}. {json.dumps(ex, ensure_ascii=False)}")
    return "\n\n".join(lines)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_existing_results(output_path: str) -> dict[str, dict]:
    existing = {}
    if not os.path.exists(output_path):
        return existing
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                key = item.get("name", "")
                if key:
                    existing[key] = item
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded {len(existing)} existing results from {output_path}")
    return existing


def append_result(output_path: str, item: dict):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def robust_json_loads(text: str) -> dict:
    text = text.strip()
    # strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise


def collect_all_platforms(yml_data: dict) -> list[dict]:
    platforms = []
    for category, subcategories in yml_data.items():
        if not isinstance(subcategories, dict):
            continue
        for subcategory, names in subcategories.items():
            if not isinstance(names, list):
                continue
            for name in names:
                platforms.append({
                    "name": str(name),
                    "category": category,
                    "subcategory": subcategory,
                })
    return platforms


def call_llm(client: LLMClient, model: str, name: str, category: str, subcategory: str, max_completion_tokens: int = 4096) -> str:
    examples_text = format_few_shot_examples(FEW_SHOT_EXAMPLES)
    user_content = USER_PROMPT_TEMPLATE.format(
        num_examples=len(FEW_SHOT_EXAMPLES),
        examples=examples_text,
        name=name,
        category=category,
        subcategory=subcategory,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return client.complete(model, messages, max_completion_tokens)


def process_platform(client: LLMClient, model: str, platform: dict, max_retries: int, max_completion_tokens: int = 4096) -> dict | None:
    name = platform["name"]
    category = platform["category"]
    subcategory = platform["subcategory"]

    for attempt in range(1, max_retries + 1):
        try:
            raw = call_llm(client, model, name, category, subcategory, max_completion_tokens)
            result = robust_json_loads(raw)

            result["name"] = name
            result["category"] = category
            result["subcategory"] = subcategory

            logger.success(f"[{name}] Done (attempt {attempt})")
            return result

        except Exception as e:
            logger.warning(f"[{name}] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    logger.error(f"[{name}] All {max_retries} attempts failed, skipping.")
    return None


def run(args: PipelineConfig):
    yml_data = load_yaml(args.scenario_input)
    all_platforms = collect_all_platforms(yml_data)
    logger.info(f"Total platforms in YAML: {len(all_platforms)}")

    existing = load_existing_results(args.scenario_output)
    pending = [p for p in all_platforms if p["name"] not in existing]
    logger.info(f"Already processed: {len(existing)}, Pending: {len(pending)}")

    if not pending:
        logger.success("All platforms already processed.")
        return

    client = LLMClient.from_config(args)

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(process_platform, client, args.model, p, args.max_retries, args.max_completion_tokens): p
            for p in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing platforms"):
            result = future.result()
            if result is not None:
                append_result(args.scenario_output, result)
                results.append(result)

    logger.success(f"Done. {len(results)}/{len(pending)} platforms processed successfully.")
    logger.success(f"Output saved to: {args.scenario_output}")


def main():
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Generate platform descriptions and features")
    parser.add_argument("--scenario_input", type=str, default=defaults.scenario_input)
    parser.add_argument("--scenario_output", type=str, default=defaults.scenario_output)
    parser.add_argument("--model", type=str, default=defaults.model)
    parser.add_argument("--api_key", type=str, default=defaults.api_key)
    parser.add_argument("--base_url", type=str, default=defaults.base_url)
    parser.add_argument("--concurrency", type=int, default=defaults.concurrency)
    parser.add_argument("--max_retries", type=int, default=defaults.max_retries)
    parsed = parser.parse_args()

    cfg = PipelineConfig()
    for k, v in vars(parsed).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    run(cfg)


if __name__ == "__main__":
    main()
