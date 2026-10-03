import json
from types import SimpleNamespace as NS

import pytest

from finops_agent import llm
from finops_agent.agent import FINANCE, AgentConfig, ask


class FakeOpenAI:
    """Mimics openai.OpenAI().chat.completions.create with scripted replies."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        return self.replies.pop(0)


def reply(content=None, calls=None, finish="stop", pt=800, ct=60):
    msg = NS(content=content, tool_calls=calls)
    return NS(
        choices=[NS(message=msg, finish_reason=finish)],
        usage=NS(prompt_tokens=pt, completion_tokens=ct),
    )


def call(id_, name, args):
    return NS(
        id=id_,
        function=NS(name=name, arguments=args if isinstance(args, str) else json.dumps(args)),
    )


def test_openai_tools_conversion_drops_strict():
    from finops_agent.agent import TOOLS

    converted = llm._to_openai_tools(TOOLS)
    assert converted[0]["type"] == "function"
    assert "strict" not in json.dumps(converted)
    assert converted[0]["function"]["parameters"]["additionalProperties"] is False


def test_agent_runs_with_openai_compatible_provider():
    client = FakeOpenAI(
        [
            reply(calls=[call("c1", "find_subscriptions", {})], finish="tool_calls"),
            reply(content="Cancel one of the cloud-storage plans."),
        ]
    )
    p = llm.OpenAICompatProvider(client, "gemini-2.5-flash", "gemini")
    out = ask("subscriptions?", AgentConfig(), provider=p, domain=FINANCE)
    assert out.answer.startswith("Cancel") and out.steps == 2
    assert [c["name"] for c in out.tool_calls] == ["find_subscriptions"]
    assert out.cost_usd == 0.0  # free-tier model
    # second request carries the assistant tool_calls + a role=tool message with real tool output
    msgs = client.requests[1]["messages"]
    assert msgs[-2]["role"] == "assistant" and msgs[-2]["tool_calls"][0]["id"] == "c1"
    assert msgs[-1]["role"] == "tool" and "icloud storage" in msgs[-1]["content"]
    assert client.requests[0]["tool_choice"] == "auto"


def test_openai_invalid_tool_arguments_become_a_tool_error_not_a_crash():
    client = FakeOpenAI(
        [
            reply(calls=[call("c1", "top_merchants", "{not json")], finish="tool_calls"),
            reply(content="sorry"),
        ]
    )
    p = llm.OpenAICompatProvider(client, "glm-4.7-flash", "zai")
    out = ask("q", AgentConfig(), provider=p, domain=FINANCE)
    assert out.answer == "sorry"
    assert "error" in client.requests[1]["messages"][-1]["content"]


def test_deepseek_cost_is_computed():
    client = FakeOpenAI([reply(content="hi", pt=1_000_000, ct=1_000_000)])
    p = llm.OpenAICompatProvider(client, "deepseek-chat", "deepseek")
    out = ask("q", AgentConfig(), provider=p)
    assert out.cost_usd == pytest.approx(0.28 + 0.42)


def test_token_cap_stops_free_tier_loops():
    client = FakeOpenAI(
        [
            reply(
                calls=[call(f"c{i}", "monthly_summary", {})], finish="tool_calls", pt=50_000, ct=0
            )
            for i in range(5)
        ]
    )
    p = llm.OpenAICompatProvider(client, "gemini-2.5-flash", "gemini")
    out = ask("q", AgentConfig(max_total_tokens=60_000), provider=p, domain=FINANCE)
    assert out.stopped_reason == "token_cap" and out.steps == 2


def test_length_and_filter_finish_reasons():
    p = llm.OpenAICompatProvider(FakeOpenAI([reply(content="cut", finish="length")]), "m")
    assert ask("q", AgentConfig(), provider=p).stopped_reason == "max_tokens"
    p = llm.OpenAICompatProvider(FakeOpenAI([reply(content="", finish="content_filter")]), "m")
    assert ask("q", AgentConfig(), provider=p).stopped_reason == "refusal"


def test_make_provider_errors_are_actionable(monkeypatch):
    for k in ("GEMINI_API_KEY", "ZAI_API_KEY", "FINOPS_BASE_URL", "FINOPS_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm.make_provider("gemini")
    with pytest.raises(RuntimeError, match="Unknown provider"):
        llm.make_provider("nope")
    with pytest.raises(RuntimeError, match="FINOPS_BASE_URL"):
        llm.make_provider("openai-compat")


def test_every_preset_is_wellformed():
    for name, (url, key_var, model) in llm.PRESETS.items():
        assert url.startswith("https://") and key_var.endswith("_API_KEY") and model, name


def test_autodetect_prefers_whichever_key_is_present(monkeypatch):
    for k in (
        "GEMINI_API_KEY",
        "ZAI_API_KEY",
        "GROQ_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY",
        "FINOPS_PROVIDER",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)
    assert llm._autodetect() == "anthropic"
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert llm._autodetect() == "groq"
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert llm._autodetect() == "gemini"
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm.make_provider()
