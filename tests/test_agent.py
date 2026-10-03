import json
from types import SimpleNamespace as NS

from finops_agent.agent import TOOLS, AgentConfig, ask, estimate_cost_usd, run_tool
from finops_agent.data import demo_dataset


def usage(i=1000, o=200, cr=0, cw=0):
    return NS(
        input_tokens=i, output_tokens=o, cache_read_input_tokens=cr, cache_creation_input_tokens=cw
    )


def tool_use(name, args, id_="tu_1"):
    return NS(type="tool_use", id=id_, name=name, input=args)


def text(t):
    return NS(type="text", text=t)


class FakeClient:
    """Replays scripted responses and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def resp(content, stop, u=None):
    return NS(content=content, stop_reason=stop, usage=u or usage())


def test_tool_schemas_are_strict_and_closed():
    for t in TOOLS:
        assert t["strict"] is True
        assert t["input_schema"]["additionalProperties"] is False


def test_run_tool_unknown_and_bad_args_return_errors_not_exceptions():
    ds = demo_dataset()
    assert "unknown tool" in json.loads(run_tool(ds, "nope", {}))["error"]
    assert "error" in json.loads(run_tool(ds, "top_services", {}))  # missing keys


def test_agent_calls_tool_then_answers():
    client = FakeClient(
        [
            resp([tool_use("total_cost", {"days": 30})], "tool_use"),
            resp([text("You spent $36,908 in 30 days.")], "end_turn"),
        ]
    )
    out = ask("How much did I spend?", AgentConfig(), client=client)
    assert out.answer.startswith("You spent")
    assert out.steps == 2 and [c["name"] for c in out.tool_calls] == ["total_cost"]
    # second request must carry the assistant turn + our tool_result
    msgs = client.requests[1]["messages"]
    assert msgs[-1]["content"][0]["type"] == "tool_result"
    assert (
        "36908" in msgs[-1]["content"][0]["content"].replace(".", "")[:200]
        or "total_usd" in msgs[-1]["content"][0]["content"]
    )


def test_parallel_tool_calls_return_all_results_in_one_message():
    client = FakeClient(
        [
            resp(
                [tool_use("total_cost", {"days": 7}, "a"), tool_use("detect_anomalies", {}, "b")],
                "tool_use",
            ),
            resp([text("done")], "end_turn"),
        ]
    )
    ask("q", AgentConfig(), client=client)
    results = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]


def test_haiku_default_request_has_no_effort():
    client = FakeClient([resp([text("hi")], "end_turn")])
    ask("q", AgentConfig(), client=client)
    assert AgentConfig().model == "claude-haiku-4-5"
    assert "output_config" not in client.requests[0]


def test_requests_use_effort_and_never_forced_tool_choice():
    client = FakeClient([resp([text("hi")], "end_turn")])
    ask("q", AgentConfig(model="claude-opus-5-5", effort="low"), client=client)
    req = client.requests[0]
    assert req["output_config"] == {"effort": "low"}
    assert "tool_choice" not in req and "temperature" not in req and "thinking" not in req


def test_cost_cap_stops_the_loop():
    big = usage(i=200_000, o=20_000)
    client = FakeClient([resp([tool_use("total_cost", {"days": 1})], "tool_use", big)])
    out = ask("q", AgentConfig(max_cost_usd=0.10), client=client)
    assert out.stopped_reason == "cost_cap" and out.cost_usd > 0.10


def test_max_steps_stops_a_tool_loop():
    client = FakeClient(
        [resp([tool_use("total_cost", {"days": 1})], "tool_use") for _ in range(10)]
    )
    out = ask("q", AgentConfig(max_steps=3, max_cost_usd=99), client=client)
    assert out.stopped_reason == "max_steps" and out.steps == 3 and len(client.requests) == 3


def test_refusal_is_handled():
    out = ask("q", AgentConfig(), client=FakeClient([resp([], "refusal")]))
    assert out.stopped_reason == "refusal" and "declined" in out.answer


def test_cost_estimate_uses_cache_pricing():
    base = estimate_cost_usd("claude-opus-5-5", usage(i=1_000_000, o=0))
    cached = estimate_cost_usd("claude-opus-5-5", usage(i=0, o=0, cr=1_000_000))
    assert base == 4.0 and cached == 0.2
