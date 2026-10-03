"""Run the full web app WITHOUT an API key, using a scripted stand-in for Claude.

    python scripts/offline_demo.py            # http://127.0.0.1:8000

The stand-in picks a tool from keywords in the question and summarizes the real tool output,
so you see the real UI, API guards, CSV upload and tool results. It is for UI/dev only.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn

from finops_agent import agent

ROUTES = [  # (keywords, tool, args)
    (("weird", "anomal", "spike"), ("detect_anomalies", {})),
    (("subscription", "overlap"), ("find_subscriptions", {})),
    (("duplicate", "unusual"), ("find_duplicate_charges", {})),
    (("delivery",), ("category_trend", {"category": "Food Delivery"})),
    (("each month", "go each", "where does my money"), ("spending_by_category", {"months": 1})),
    (("save", "waste"), ("savings_summary", {})),
    (("nat",), ("cost_trend", {"service": "NAT Gateway", "days": 30})),
    (("most", "top"), ("top_services", {"days": 30, "n": 3})),
]


def pick(question: str, tool_names: set[str]):
    q = question.lower()
    for words, (tool, args) in ROUTES:
        if tool in tool_names and any(w in q for w in words):
            return tool, args
    return ("total_cost", {"days": 30}) if "total_cost" in tool_names else ("monthly_summary", {})


class FakeClient:
    def __init__(self):
        self.messages = self

    def create(self, **kw):
        msgs = kw["messages"]
        if len(msgs) == 1:
            tool, args = pick(msgs[0]["content"], {t["name"] for t in kw["tools"]})
            block = NS(type="tool_use", id="demo_1", name=tool, input=args)
            return NS(
                content=[block],
                stop_reason="tool_use",
                usage=NS(input_tokens=900, output_tokens=60),
            )
        raw = msgs[-1]["content"][0]["content"]
        data = json.loads(raw)
        text = "**(offline demo, no Claude call)** Here is what the tool returned:\n\n"
        items = data if isinstance(data, list) else [data]
        for it in items[:5]:
            text += "- " + ", ".join(f"{k}: {v}" for k, v in list(it.items())[:5]) + "\n"
        return NS(
            content=[NS(type="text", text=text)],
            stop_reason="end_turn",
            usage=NS(input_tokens=1400, output_tokens=180),
        )


agent.anthropic.Anthropic = FakeClient

if __name__ == "__main__":
    uvicorn.run("finops_agent.api:app", host="127.0.0.1", port=8000)
