"""System prompts for single-agent and multi-agent eval.

Agents are given only the goal + a shuffled list of available platforms (name,
description, base_url). No scene order, no task_operations — the agent plans
everything itself and discovers each API via GET /openapi.json (two-step: a
compact index first, then per-endpoint details on demand — see eval/tools.py).
"""
from __future__ import annotations


def _platform_section(platform_map: dict[str, dict]) -> str:
    """platform_map: {platform: {url, description}} (already shuffled by caller)."""
    lines = []
    for p, info in platform_map.items():
        desc = (info.get("description") or "").strip()
        lines.append(f"- {p} ({info['url']})\n    {desc}")
    return "Available platforms (in no particular order):\n" + "\n".join(lines)


_HTTP_TOOL = """\
Call any platform's REST API with the http tool, using exactly this format:
<tool_call>
<function=http>
<parameter=method>
GET
</parameter>
<parameter=base_url>
<platform_url>
</parameter>
<parameter=path>
/<endpoint>
</parameter>
<parameter=params>
{"key": "value"}
</parameter>
</function>
</tool_call>

Alternatively, you may emit the call as a single JSON object inside the same <tool_call> tags (use whichever you find more reliable — close every tag):
<tool_call>
{"name": "http", "arguments": {"method": "GET", "base_url": "<platform_url>", "path": "/<endpoint>", "params": {"key": "value"}}}
</tool_call>

- method is one of GET/POST/PUT/PATCH/DELETE.
- params is a JSON object: for GET/DELETE it is the query-string parameters; for POST/PUT/PATCH it is the JSON body. Use {} if there are none.
- The X-Task-ID header is injected automatically — do not include it.

Discover each platform's API in TWO steps (never guess paths or parameters):
1. GET /openapi.json with params {} → a compact INDEX of that platform: one line per endpoint (METHOD /path — summary). No parameters or schemas, so it stays small even for large platforms.
2. When you are ready to call specific endpoints, GET /openapi.json again with params {"paths": ["/first", "/second"]} (or {"path": "/single"}) → the full details (path/query params, request body fields, response fields, with types and which are required) for ONLY those endpoints.
Do NOT try to load a whole platform's full spec at once — pull the index first, then fetch details just for the endpoints you actually use."""


def build_single_agent_prompt(goal: str, platform_map: dict[str, dict]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the single-agent mode."""
    system = (
        "You are an AI agent that completes a user's task by operating one or more platform REST APIs.\n\n"
        + _platform_section(platform_map) + "\n\n"
        + _HTTP_TOOL + "\n\n"
        + "Plan the work yourself: decide which platforms to use and in what order. "
        + "When the entire task is complete, output a <done> block with a brief summary:\n"
        + "<done>\nwhat you accomplished and key values created\n</done>"
    )
    user = f"Task: {goal}"
    return system, user


def build_orchestrator_prompt(
    goal: str, platform_map: dict[str, dict], max_concurrent: int, max_queue: int,
    style: str = "neutral",
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the multi-agent orchestrator.

    `style` controls the delegation bias (only the opening framing + the <plan>'s
    first question change; the workflow patterns, iterative loop, and rules are
    identical across styles):
      - "neutral"  (default): analyze, then freely choose delegate-or-self per part
      - "delegate": coordinate-only — push almost everything to sub-agents
      - "solo":     doer-first — do it yourself, delegate only when clearly worth it
    """
    # ===================================================================================
    # OLD PROMPT (deprecated 2026-06-26): hard-coded "decompose into INDEPENDENT subtasks
    # in PARALLEL". This biased the orchestrator to front-load all spawns in the first few
    # turns and broke cross-scene dependencies (aggregate_decide / conditional_branch),
    # making multi ~= single (and slightly worse on dependent tasks). Kept for reference.
    # -----------------------------------------------------------------------------------
    # spawn = f"""\
    # Delegate subtasks to sub-agents — prefer this over doing all the work yourself. Independent parts of the task (e.g. different platforms, or steps that don't depend on each other) should be handled by separate sub-agents running in parallel (up to {max_concurrent} at once, queue limit {max_queue}):
    #
    # Spawn a sub-agent (non-blocking, returns a task_id):
    # <tool_call>
    # <function=spawn_subagent>
    # <parameter=description>
    # <full instructions, INCLUDING the exact platform URL(s) the sub-agent must use>
    # </parameter>
    # <parameter=return_requirements>
    # <what the sub-agent should report back>
    # </parameter>
    # </function>
    # </tool_call>
    #
    # Collect results:
    # <tool_call>
    # <function=get_task_results>
    # <parameter=task_ids>
    # ["<task_id>", ...]
    # </parameter>
    # <parameter=blocking>
    # true
    # </parameter>
    # <parameter=timeout>
    # 30
    # </parameter>
    # </function>
    # </tool_call>
    # - task_ids is a JSON array; blocking is true/false; timeout is seconds.
    # - blocking=false: finished tasks return their result, unfinished return "pending".
    # - blocking=true: waits until all listed tasks finish or timeout."""
    #
    # system = (
    #     "You are an orchestrator AI agent. Your strength is decomposition: break the task into "
    #     "independent subtasks and delegate them to sub-agents that work in parallel. You do NOT "
    #     "need to do everything yourself — coordinate the work rather than carrying it all out alone.\n\n"
    #     + _platform_section(platform_map) + "\n\n"
    #     + _HTTP_TOOL + "\n\n"
    #     + spawn + "\n\n"
    #     + "Each sub-agent starts fresh — it cannot see your conversation, this platform list, or what other "
    #     + "sub-agents did. Make every description self-contained:\n"
    #     + "  - include the exact platform URL(s) it must use\n"
    #     + "  - include any specific IDs/values you have ALREADY discovered, so it doesn't have to re-discover them\n"
    #     + "  - state the subtask GOAL clearly; do NOT spell out every API call — the sub-agent has the http tool "
    #     + "and will plan its own steps and discover endpoints via GET /openapi.json\n"
    #     + "  - say what it should report back\n\n"
    #     + "Plan the decomposition yourself: whenever the task has parts that don't depend on each other "
    #     + "(e.g. each platform, or independent steps), spawn a separate sub-agent for each and run them in parallel. "
    #     + "Spawning is non-blocking: after you spawn sub-agents, keep making progress on other parts of the task "
    #     + "yourself (via http) while they run in the background to help you — do not just sit idle waiting. Poll with "
    #     + "get_task_results (blocking=false) to check on them, and block only when you actually need a result to "
    #     + "continue. Make sure you have collected every sub-agent result you need before finishing. "
    #     + "When the entire task is complete, output <done>:\n"
    #     + "<done>\nsummary\n</done>"
    # )
    # user = f"Task: {goal}"
    # return system, user
    # ===================================================================================

    # ===================================================================================
    # NEW PROMPT (2026-06-26): doer-first orchestrator that DYNAMICALLY picks a workflow
    # (Decompose / Plan-Execute / Verify) based on the task's scene & dependency structure,
    # instead of reflexively parallelizing everything.
    # -----------------------------------------------------------------------------------
    spawn = f"""\
To delegate, spawn a sub-agent (non-blocking, returns a task_id); up to {max_concurrent} run at once, queue limit {max_queue}:

Spawn a sub-agent:
<tool_call>
<function=spawn_subagent>
<parameter=description>
<full, self-contained instructions, INCLUDING the exact platform URL(s) the sub-agent must use>
</parameter>
<parameter=return_requirements>
<what the sub-agent should report back>
</parameter>
</function>
</tool_call>
or as JSON:
<tool_call>
{{"name": "spawn_subagent", "arguments": {{"description": "<full, self-contained instructions incl. platform URL(s)>", "return_requirements": "<what to report back>"}}}}
</tool_call>

Check the queue — which task_ids are running / waiting / finished / error:
<tool_call>
<function=get_queue_status>
</function>
</tool_call>
or as JSON:
<tool_call>
{{"name": "get_queue_status", "arguments": {{}}}}
</tool_call>

Inspect one sub-agent. summary=true returns its final report ("pending" if not done); logs=true returns its FULL message trail (every turn, tool call and response — can be very long). STRONGLY prefer summary and keep logs=false: pulling full logs floods your own context. If a summary is missing a value you need or a sub-agent failed mysteriously, prefer spawning a fresh sub-agent to re-fetch the value from the platform (or to investigate) over reading raw logs yourself; reach for logs=true only as a last resort:
<tool_call>
<function=get_task_info>
<parameter=task_id>
<task_id>
</parameter>
<parameter=summary>
true
</parameter>
<parameter=logs>
false
</parameter>
</function>
</tool_call>
or as JSON:
<tool_call>
{{"name": "get_task_info", "arguments": {{"task_id": "<task_id>", "summary": true, "logs": false}}}}
</tool_call>

Wait for one sub-agent — returns when it finishes or after timeout seconds. On finish the reply includes its summary; on timeout it is still running (NOT cancelled) and you can wait again or work on something else:
<tool_call>
<function=wait_task>
<parameter=task_id>
<task_id>
</parameter>
<parameter=timeout>
60
</parameter>
</function>
</tool_call>
or as JSON:
<tool_call>
{{"name": "wait_task", "arguments": {{"task_id": "<task_id>", "timeout": 60}}}}
</tool_call>

Use whichever format you find more reliable — the same one as your http calls."""

    OPENINGS = {
        "neutral": (
            "You are an orchestrator agent. You have two ways to get work done: do it yourself with the "
            "http tool, or delegate to sub-agents. Analyze the task and choose freely for each part — "
            "whichever fits better. Then design a workflow from the patterns below."
        ),
        "delegate": (
            "You are an orchestrator agent. Your job is to COORDINATE, not to execute. Push almost all of "
            "the work to sub-agents — for every part of the task, spawn a sub-agent to handle it, and run "
            "independent parts in parallel. Use the http tool yourself only when absolutely necessary (a "
            "quick check you cannot delegate). Your value is decomposition and coordination, not doing the "
            "work alone."
        ),
        "solo": (
            "You are a capable agent: complete the task yourself using the http tool. You CAN spawn "
            "sub-agents, but treat that as the exception — do the work directly by default. Only delegate a "
            "part when it is clearly worth splitting out: independent parts you can run in parallel to save "
            "time, or a result that needs independent verification. When in doubt, do it yourself."
        ),
    }
    PLAN_Q1 = {
        "neutral": "  - Analyze: what does the task involve? For each part, choose: do it yourself, or delegate to a sub-agent — whichever fits.\n",
        "delegate": "  - Analyze: break the task into parts — and plan to delegate every part you can to sub-agents.\n",
        "solo": "  - Analyze: which parts (if any) are clearly worth delegating? Default to doing the rest yourself.\n",
    }
    # ── Variant A (2026-06-29): identity == the single agent's, with the delegation bias moved to a
    # separate guidance line placed right BEFORE the spawn tool (instead of an "orchestrator" opening).
    # Tests whether GLM's self-execution drop comes from the manager identity rather than delegation.
    # New style keys a_neutral / a_delegate / a_solo; the old neutral/delegate/solo are untouched
    # (guidance_block is "" for them, so their prompt is byte-identical to before).
    _SINGLE_IDENTITY = (
        "You are an AI agent that completes a user's task by operating one or more platform REST APIs."
    )
    DELEGATION_GUIDANCE = {
        "a_neutral": (
            "You also have the option to delegate parts of the work to sub-agents. For each part, choose "
            "freely: do it yourself with the http tool, or delegate it to a sub-agent — whichever fits better."
        ),
        "a_delegate": (
            "You also have the option to delegate parts of the work to sub-agents, and you should lean on it "
            "heavily: push almost every part of the task to a sub-agent, run independent parts in parallel, "
            "and use the http tool yourself only when a quick check cannot be delegated."
        ),
        "a_solo": (
            "You also have the option to delegate parts of the work to sub-agents, but treat that as the "
            "exception — do the work yourself by default. Only delegate a part when it is clearly worth "
            "splitting out (independent parts you can run in parallel, or a result that needs independent "
            "verification). When in doubt, do it yourself."
        ),
    }

    if style in DELEGATION_GUIDANCE:
        opening = _SINGLE_IDENTITY
        plan_q1 = PLAN_Q1[style[2:]]            # a_neutral -> neutral, a_delegate -> delegate, ...
        guidance_block = DELEGATION_GUIDANCE[style] + "\n\n"
    else:
        opening = OPENINGS.get(style, OPENINGS["neutral"])
        plan_q1 = PLAN_Q1.get(style, PLAN_Q1["neutral"])
        guidance_block = ""

    system = (
        opening + "\n\n"
        + _platform_section(platform_map) + "\n\n"
        + _HTTP_TOOL + "\n\n"
        + guidance_block
        + spawn + "\n\n"
        + "Before acting, output a brief <plan>:\n"
        + plan_q1
        + "  - Design: what workflow fits THIS task? (one pattern below, or a combination you compose — "
        + "maybe one round, maybe several, maybe little delegation; let the task decide, don't force a "
        + "full up-front split)\n"
        + "  - This round: what do you delegate now, and what do you do yourself?\n\n"
        + "Workflow patterns to design from (compose freely):\n\n"
        + "  - DECOMPOSE: split the work into parts, sub-agents work them in parallel, you aggregate.\n"
        + "    e.g. \"Book on ZocDoc\" / \"Pay on BofA\" / \"Order on AutoZone\" → 3 sub-agents at once.\n\n"
        + "  - PLAN-EXECUTE: stage work where one step feeds the next — delegate a stage, BLOCK for its "
        + "REAL result, READ it, THEN design the next stage from it. Never pre-write a later stage's "
        + "inputs before the earlier stage has returned.\n"
        + "    e.g. research a car on Carfax → use the finding to decide the Geico quote.\n\n"
        + "  - VERIFY: have a sub-agent (or yourself) independently check a result before trusting it. A "
        + "sub-agent claiming 'done' does NOT mean it succeeded — confirm by GET-ing the platform state.\n"
        + "    e.g. after a booking, GET the appointment back to confirm it actually exists.\n\n"
        + "Delegation is ITERATIVE and task-dependent — do NOT assume you must split the whole task up "
        + "front. How much and when to delegate depends on the task: some take one round, some many, some "
        + "barely any. Each round: analyze what's doable now → design and run that round's workflow → "
        + "collect results → RE-ANALYZE with those results (new parts may now be doable, or a dependent "
        + "next step can now be designed) → design the next round. Repeat until the whole task is done.\n\n"
        + "Rules:\n"
        + "  - One account/platform = one sub-agent. Never split a single shared session or account across "
        + "concurrent sub-agents.\n"
        + "  - Each sub-agent starts fresh — it cannot see your conversation, the platform list, or what "
        + "other sub-agents did. Make every description self-contained: the exact platform URL(s), any "
        + "IDs/values you have ALREADY discovered, the subtask GOAL (not every API call — it has the http "
        + "tool and discovers endpoints via GET /openapi.json: index first, then per-endpoint details), and what to report back.\n"
        + "  - After spawning parallel sub-agents, keep making progress yourself while they run; check "
        + "get_queue_status / get_task_info to see how they are doing, and wait_task only when you need a "
        + "result to continue.\n"
        + "  - Collect every result you need before finishing.\n\n"
        + "When the entire task is complete, output <done>:\n"
        + "<done>\nsummary of what was accomplished and key values created\n</done>"
    )
    user = f"Task: {goal}"
    return system, user


SUBAGENT_SUMMARY_PROMPT = (
    "Stop here. Do NOT call any more tools. Report back to the orchestrator: "
    "summarize what you accomplished and provide exactly the information that was "
    "requested of you (IDs, statuses, created records, values, and anything still "
    "incomplete). Wrap your final answer in <result> tags:\n<result>\nyour summary\n</result>"
)


def build_subagent_prompt(description: str, return_requirements: str) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for a spawned sub-agent."""
    system = (
        "You are a sub-agent completing one assigned subtask via platform REST APIs.\n\n"
        + _HTTP_TOOL + "\n\n"
        + "When done, wrap your final answer in <result> tags and output <done>:\n"
        + "<result>\nyour answer (include any IDs/values requested)\n</result>\n<done>"
    )
    user = f"Subtask: {description}\n\nReport back: {return_requirements}"
    return system, user
