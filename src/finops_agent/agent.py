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

import anthropic

from . import analysis, finance
from .data import demo_dataset, load_costs_csv

DEFAULT_MODEL = "claude-opus-5-5"

# (input $/MTok, output $/MTok, cache-read $/MTok). Update with the pricing page.
PRICING = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
}

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


def estimate_cost_usd(model: str, usage: Any) -> float:
    inp, out, cache_read = PRICING.get(model, PRICING[DEFAULT_MODEL])
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cached = getattr(usage, "cache_read_input_tokens", 0) or 0
    return (
        (usage.input_tokens or 0) * inp
        + (usage.output_tokens or 0) * out
        + cached * cache_read
        + cache_write * inp * 1.25
    ) / 1_000_000


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
    max_tokens: int = 4000


def ask(
    question: str,
    config: AgentConfig | None = None,
    client: anthropic.Anthropic | None = None,
    dataset: Any | None = None,
    domain: Domain = FINOPS,
) -> AgentResult:
    cfg = config or AgentConfig()
    ds = dataset or domain.demo()
    client = client or anthropic.Anthropic()
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    result = AgentResult(answer="", steps=0)

    for step in range(1, cfg.max_steps + 1):
        response = client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=domain.system_prompt,
            tools=domain.tools,
            messages=messages,
            output_config={"effort": cfg.effort},
        )
        result.steps = step
        result.input_tokens += response.usage.input_tokens or 0
        result.output_tokens += response.usage.output_tokens or 0
        result.cost_usd += estimate_cost_usd(cfg.model, response.usage)

        if response.stop_reason == "refusal":
            result.stopped_reason = "refusal"
            result.answer = "The request was declined by the model's safety checks."
            return result

        text = "".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason != "tool_use":
            result.answer = text
            result.stopped_reason = response.stop_reason or "end_turn"
            return result

        # Echo the assistant turn back unchanged (required for thinking blocks).
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_tool(ds, block.name, dict(block.input), domain.dispatch)
                result.tool_calls.append({"name": block.name, "input": dict(block.input)})
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
        messages.append({"role": "user", "content": tool_results})

        if result.cost_usd >= cfg.max_cost_usd:
            result.stopped_reason = "cost_cap"
            result.answer = text or "Stopped: per-request cost cap reached."
            return result

    result.stopped_reason = "max_steps"
    result.answer = result.answer or "Stopped: step limit reached before a final answer."
    return result
