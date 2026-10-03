"""The agent: Claude + deterministic tools, with explicit step and cost guards.

A manual tool-use loop (instead of the SDK tool runner) keeps three things in our hands:
a hard cap on steps, a hard cap on spend per request, and a trivially testable seam
(`client` is injectable, so unit tests never touch the network).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from . import analysis, finance
from .data import demo_dataset, load_costs_csv
from .llm import (
    AnthropicProvider,
    Provider,
    estimate_cost_usd,
    make_provider,
    supports_effort,
)

__all__ = ["estimate_cost_usd", "supports_effort"]  # re-exported for tests/back-compat

DEFAULT_MODEL = "claude-haiku-4-5"  # cheapest current Claude; override with FINOPS_MODEL


SYSTEM_PROMPT = """You are FinOps Agent, an assistant that helps engineers understand and \
reduce their AWS bill.

Rules:
- Never compute totals, percentages or rankings yourself from memory. Always call a tool \
and quote its numbers.
- If a question needs several facts, call several tools (in parallel when independent).
- Dollar amounts: use USD with thousands separators, round to whole dollars unless the \
amount is under $100.
- Savings are estimates based on documented assumptions. Say so, and mention the riskiest \
action (anything touching prod) before recommending it.
- If the data cannot answer the question, say what is missing instead of guessing.
- Be concise: lead with the answer, then a short ranked list. No preamble."""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "total_cost",
        "description": "Total AWS spend in USD over the last N days of the dataset.",
        "input_schema": {
            "type": "object",
            "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 120}},
            "required": ["days"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "top_services",
        "description": "Top N services by spend over the last N days, with share of total.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "minimum": 1, "maximum": 120},
                "n": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["days", "n"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "cost_trend",
        "description": (
            "Compare one service's spend over the last N days vs the previous N days. "
            "Service names look like 'Amazon EC2', 'AWS Lambda', 'NAT Gateway'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "days": {"type": "integer", "minimum": 1, "maximum": 60},
            },
            "required": ["service", "days"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "detect_anomalies",
        "description": "Find days where a service's cost spiked far above its trailing 30-day mean.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "find_waste",
        "description": (
            "List individual wasteful resources (idle EC2, unattached EBS, old snapshots, "
            "oversized RDS, dev NAT gateways) with an estimated monthly saving each. "
            "Pass env 'prod', 'staging' or 'dev' to filter; omit to see everything."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"env": {"type": "string", "enum": ["prod", "staging", "dev"]}},
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "savings_summary",
        "description": (
            "Aggregate estimated savings by category plus monthly and annual totals. "
            "Optional env filter ('prod', 'staging', 'dev')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"env": {"type": "string", "enum": ["prod", "staging", "dev"]}},
            "additionalProperties": False,
        },
        "strict": True,
    },
]

_DISPATCH = {
    "total_cost": lambda ds, a: {
        "days": a["days"],
        "total_usd": analysis.total_cost(ds, a["days"]),
    },
    "top_services": lambda ds, a: analysis.top_services(ds, a["days"], a["n"]),
    "cost_trend": lambda ds, a: analysis.cost_trend(ds, a["service"], a["days"]),
    "detect_anomalies": lambda ds, a: analysis.detect_anomalies(ds),
    "find_waste": lambda ds, a: analysis.find_waste(ds, a.get("env")),
    "savings_summary": lambda ds, a: analysis.savings_summary(ds, a.get("env")),
}


@dataclass(frozen=True)
class Domain:
    """Everything that makes the agent specific to one use case."""

    name: str
    system_prompt: str
    tools: list[dict[str, Any]]
    dispatch: dict[str, Any]
    demo: Any  # () -> dataset
    load_csv: Any  # (text) -> dataset


FINOPS = Domain("finops", SYSTEM_PROMPT, TOOLS, _DISPATCH, demo_dataset, load_costs_csv)
FINANCE = Domain(
    "finance",
    finance.SYSTEM_PROMPT,
    finance.TOOLS,
    finance.DISPATCH,
    finance.demo_ledger,
    finance.load_statement_csv,
)
DOMAINS = {d.name: d for d in (FINOPS, FINANCE)}


def run_tool(
    ds: Any, name: str, args: dict[str, Any], dispatch: dict[str, Any] | None = None
) -> str:
    """Execute one tool and return a JSON string. Errors become results, never exceptions."""
    fn = (dispatch if dispatch is not None else _DISPATCH).get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool '{name}'"})
    try:
        return json.dumps(fn(ds, args))
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@dataclass
class AgentResult:
    answer: str
    steps: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    stopped_reason: str = "end_turn"


@dataclass
class AgentConfig:
    model: str = field(default_factory=lambda: os.getenv("FINOPS_MODEL", DEFAULT_MODEL))
    effort: str = field(default_factory=lambda: os.getenv("FINOPS_EFFORT", "medium"))
    max_steps: int = 6
    max_cost_usd: float = field(
        default_factory=lambda: float(os.getenv("FINOPS_MAX_COST_PER_REQUEST", "0.25"))
    )
    # Free tiers cost $0 per call, so also cap tokens (protects rate limits and latency).
    max_total_tokens: int = field(
        default_factory=lambda: int(os.getenv("FINOPS_MAX_TOKENS_PER_REQUEST", "60000"))
    )
    max_tokens: int = 4000


def ask(
    question: str,
    config: AgentConfig | None = None,
    client: Any | None = None,  # Anthropic-style client (kept for tests / backwards compat)
    dataset: Any | None = None,
    domain: Domain = FINOPS,
    provider: Provider | None = None,
) -> AgentResult:
    cfg = config or AgentConfig()
    ds = dataset or domain.demo()
    if provider is None:
        if client is not None:
            provider = AnthropicProvider(client, cfg.model, cfg.effort, cfg.max_tokens)
        else:
            provider = make_provider(
                model=os.getenv("FINOPS_MODEL"), effort=cfg.effort, max_tokens=cfg.max_tokens
            )
    session = provider.new_session(domain.system_prompt, domain.tools, question)
    result = AgentResult(answer="", steps=0)
    nudges = 0

    for step in range(1, cfg.max_steps + 1):
        turn = session.step()
        result.steps = step
        result.input_tokens += turn.input_tokens
        result.output_tokens += turn.output_tokens
        result.cost_usd += turn.cost_usd

        if turn.stop == "refusal":
            result.stopped_reason = "refusal"
            result.answer = "The request was declined by the model's safety checks."
            return result
        if not turn.tool_calls and not turn.text.strip() and result.tool_calls and nudges < 1:
            nudges += 1
            if session.nudge("Using the tool results above, answer the original question now."):
                continue
        if not turn.tool_calls:
            result.answer = turn.text
            result.stopped_reason = turn.stop
            return result

        outputs = []
        for call in turn.tool_calls:
            outputs.append((call.id, run_tool(ds, call.name, call.input, domain.dispatch)))
            result.tool_calls.append({"name": call.name, "input": call.input})
        session.add_tool_results(outputs)

        if result.cost_usd >= cfg.max_cost_usd:
            result.stopped_reason = "cost_cap"
            result.answer = turn.text or "Stopped: per-request cost cap reached."
            return result
        if result.input_tokens + result.output_tokens >= cfg.max_total_tokens:
            result.stopped_reason = "token_cap"
            result.answer = turn.text or "Stopped: per-request token cap reached."
            return result

    result.stopped_reason = "max_steps"
    result.answer = result.answer or "Stopped: step limit reached before a final answer."
    return result
