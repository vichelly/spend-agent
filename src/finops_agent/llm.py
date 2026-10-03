"""LLM providers behind one tiny interface.

The agent loop only knows `Provider.new_session(...) -> Session`, `Session.step() -> Turn` and
`Session.add_tool_results(...)`. Each provider keeps its own message format internally, so adding
a provider never touches the agent or the analysis code.

Providers:
  * anthropic        - Claude via the Anthropic SDK
  * openai-compat    - any OpenAI-compatible Chat Completions endpoint with tool calling:
                       Google Gemini, Z.ai GLM, DeepSeek, Groq, OpenRouter, a local server, ...
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

# ----------------------------------------------------------------------------- neutral types


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    stop: str = "end_turn"  # end_turn | tool_use | refusal | max_tokens


class Session(Protocol):
    def step(self) -> Turn: ...
    def add_tool_results(self, results: list[tuple[str, str]]) -> None: ...


class Provider(Protocol):
    name: str
    model: str

    def new_session(self, system: str, tools: list[dict[str, Any]], question: str) -> Session: ...


# ----------------------------------------------------------------------------- pricing

# (input $/MTok, output $/MTok, cache-read $/MTok). Verify against each provider's pricing page.
PRICING = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
    "deepseek-chat": (0.28, 0.42, 0.028),
    "deepseek-v4-flash": (0.14, 0.28, 0.028),
}  # anything not listed (Gemini free tier, Z.ai Flash, Groq free) is priced at $0


def price_for(model: str) -> tuple[float, float, float]:
    return PRICING.get(model, (0.0, 0.0, 0.0))


def supports_effort(model: str) -> bool:
    """Haiku 4.5 rejects output_config.effort; the Sonnet/Opus 5.x models accept it."""
    return not model.startswith("claude-haiku")


def estimate_cost_usd(model: str, usage: Any) -> float:
    inp, out, cache_read = PRICING.get(model, PRICING["claude-opus-5-5"])
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cached = getattr(usage, "cache_read_input_tokens", 0) or 0
    return (
        (usage.input_tokens or 0) * inp
        + (usage.output_tokens or 0) * out
        + cached * cache_read
        + cache_write * inp * 1.25
    ) / 1_000_000


# ----------------------------------------------------------------------------- Anthropic


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self, client: Any, model: str, effort: str = "medium", max_tokens: int = 4000
    ) -> None:
        self.client, self.model, self.effort, self.max_tokens = client, model, effort, max_tokens

    def new_session(self, system: str, tools: list[dict[str, Any]], question: str) -> Session:
        return _AnthropicSession(self, system, tools, question)


class _AnthropicSession:
    def __init__(self, p: AnthropicProvider, system: str, tools: list, question: str) -> None:
        self.p, self.system, self.tools = p, system, tools
        self.messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        self._last_content: Any = None

    def step(self) -> Turn:
        extra: dict[str, Any] = {}
        if supports_effort(self.p.model):
            extra["output_config"] = {"effort": self.p.effort}
        r = self.p.client.messages.create(
            model=self.p.model,
            max_tokens=self.p.max_tokens,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
            **extra,
        )
        self._last_content = r.content
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
        stop = {"tool_use": "tool_use", "refusal": "refusal", "max_tokens": "max_tokens"}.get(
            r.stop_reason, "end_turn"
        )
        return Turn(
            text="".join(b.text for b in r.content if b.type == "text"),
            tool_calls=calls,
            input_tokens=r.usage.input_tokens or 0,
            output_tokens=r.usage.output_tokens or 0,
            cost_usd=estimate_cost_usd(self.p.model, r.usage),
            stop=stop,
        )

    def add_tool_results(self, results: list[tuple[str, str]]) -> None:
        # Echo the assistant turn back unchanged (required for thinking blocks).
        self.messages.append({"role": "assistant", "content": self._last_content})
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": i, "content": c} for i, c in results
                ],
            }
        )


# ----------------------------------------------------------------------------- OpenAI-compatible


class OpenAICompatProvider:
    def __init__(
        self, client: Any, model: str, name: str = "openai-compat", max_tokens: int = 4000
    ) -> None:
        self.client, self.model, self.name, self.max_tokens = client, model, name, max_tokens

    def new_session(self, system: str, tools: list[dict[str, Any]], question: str) -> Session:
        return _OpenAISession(self, system, tools, question)


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools  # `strict` is Anthropic-specific; other providers may reject it
    ]


class _OpenAISession:
    def __init__(self, p: OpenAICompatProvider, system: str, tools: list, question: str) -> None:
        self.p, self.tools = p, _to_openai_tools(tools)
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        self._assistant: dict[str, Any] | None = None

    def step(self) -> Turn:
        r = self.p.client.chat.completions.create(
            model=self.p.model,
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto",
            max_tokens=self.p.max_tokens,
        )
        choice = r.choices[0]
        msg = choice.message
        raw_calls = list(getattr(msg, "tool_calls", None) or [])
        calls: list[ToolCall] = []
        for c in raw_calls:
            try:
                args = json.loads(c.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise TypeError("arguments must be an object")
            except (ValueError, TypeError):
                args = {"__invalid_arguments__": str(c.function.arguments)[:200]}
            calls.append(ToolCall(c.id, c.function.name, args))
        self._assistant = {"role": "assistant", "content": msg.content or None}
        if raw_calls:
            self._assistant["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments or "{}",
                    },
                }
                for c in raw_calls
            ]
        usage = getattr(r, "usage", None)
        tin = getattr(usage, "prompt_tokens", 0) or 0
        tout = getattr(usage, "completion_tokens", 0) or 0
        pin, pout, _ = price_for(self.p.model)
        finish = choice.finish_reason
        stop = (
            "tool_use"
            if calls
            else "max_tokens"
            if finish == "length"
            else "refusal"
            if finish == "content_filter"
            else "end_turn"
        )
        return Turn(
            text=msg.content or "",
            tool_calls=calls,
            input_tokens=tin,
            output_tokens=tout,
            cost_usd=(tin * pin + tout * pout) / 1_000_000,
            stop=stop,
        )

    def add_tool_results(self, results: list[tuple[str, str]]) -> None:
        assert self._assistant is not None
        self.messages.append(self._assistant)
        for call_id, content in results:
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": content})


# ----------------------------------------------------------------------------- factory

# preset: (base_url, key env var, default model). Model ids drift; override with FINOPS_MODEL.
PRESETS: dict[str, tuple[str, str, str]] = {
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-2.5-flash",
    ),
    "zai": ("https://api.z.ai/api/paas/v4", "ZAI_API_KEY", "glm-4.7-flash"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-chat"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "deepseek/deepseek-chat"),
}
PROVIDER_NAMES = ["anthropic", *PRESETS, "openai-compat"]


def _autodetect() -> str:
    """Pick the first provider that has a key set; free-tier providers come first."""
    for name in ("gemini", "zai", "groq", "deepseek", "openrouter"):
        if os.getenv(PRESETS[name][1]):
            return name
    return "anthropic"


def make_provider(
    name: str | None = None,
    model: str | None = None,
    effort: str = "medium",
    max_tokens: int = 4000,
) -> Provider:
    """Build a provider from FINOPS_PROVIDER / FINOPS_MODEL (or the arguments)."""
    name = (name or os.getenv("FINOPS_PROVIDER") or _autodetect()).lower()
    model = model or os.getenv("FINOPS_MODEL")

    if name == "anthropic":
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError(
                "Missing ANTHROPIC_API_KEY for provider 'anthropic'. Put it in .env."
            )
        import anthropic

        return AnthropicProvider(
            anthropic.Anthropic(), model or "claude-haiku-4-5", effort, max_tokens
        )

    if name == "openai-compat":
        base_url, key_var, default_model = os.getenv("FINOPS_BASE_URL"), "OPENAI_API_KEY", None
        if not base_url:
            raise RuntimeError("openai-compat needs FINOPS_BASE_URL (and OPENAI_API_KEY).")
    elif name in PRESETS:
        base_url, key_var, default_model = PRESETS[name]
    else:
        raise RuntimeError(f"Unknown provider '{name}'. Choose one of: {', '.join(PROVIDER_NAMES)}")

    api_key = os.getenv(key_var)
    if not api_key:
        raise RuntimeError(
            f"Missing {key_var} for provider '{name}'. Put it in .env (see .env.example)."
        )
    model = model or default_model
    if not model:
        raise RuntimeError("Set FINOPS_MODEL for provider 'openai-compat'.")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'Install the OpenAI-compatible client: pip install -e ".[openai]"'
        ) from exc
    return OpenAICompatProvider(OpenAI(api_key=api_key, base_url=base_url), model, name, max_tokens)
