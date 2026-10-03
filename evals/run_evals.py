"""Live evals: runs the real agent against questions whose answers are computed from the same
dataset the tools use. Needs ANTHROPIC_API_KEY and spends real (small) money.

    python evals/run_evals.py            # all cases
    python evals/run_evals.py --max 3    # cheap smoke run

Each case checks (1) the right tools were called, (2) the key number appears in the answer.
Exit code is non-zero if the pass rate is below --threshold, so CI can gate on it.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finops_agent.env import load_dotenv

load_dotenv()

from finops_agent import analysis, finance
from finops_agent.agent import FINANCE, FINOPS, AgentConfig, ask
from finops_agent.data import demo_dataset
from finops_agent.llm import make_provider

DS = demo_dataset()
LD = finance.demo_ledger()


def usd(n: float) -> list[str]:
    """Acceptable renderings of a dollar amount (whole dollars or cents, with/without commas)."""
    whole = f"{round(n):,}"
    return [whole, whole.replace(",", ""), f"{n:,.2f}"]


@dataclass
class Case:
    name: str
    question: str
    expect_tools: set[str]
    expect_any: list[str]  # at least one of these substrings must appear in the answer
    domain: str = "finops"


def build_cases() -> list[Case]:
    top = analysis.top_services(DS, 30, 1)[0]
    trend = analysis.cost_trend(DS, "NAT Gateway", 30)
    summary = analysis.savings_summary(DS)
    dev = analysis.savings_summary(DS, "dev")
    spike = max(analysis.detect_anomalies(DS), key=lambda a: a["z_score"])
    base = [
        Case(
            "total_30d",
            "How much did we spend in the last 30 days?",
            {"total_cost"},
            usd(analysis.total_cost(DS, 30)),
        ),
        Case(
            "top_service",
            "Which AWS service costs us the most this month?",
            {"top_services"},
            [top["service"]],
        ),
        Case(
            "trend_nat",
            "Is our NAT Gateway cost going up compared to the previous 30 days?",
            {"cost_trend"},
            [f"{trend['change_pct']}", f"{round(trend['change_pct'])}%"],
        ),
        Case(
            "anomaly",
            "Did anything weird happen to our bill recently?",
            {"detect_anomalies"},
            ["NAT Gateway", spike["date"]],
        ),
        Case(
            "savings_total",
            "How much could we save per month by cleaning up waste?",
            {"savings_summary"},
            usd(summary["total_est_monthly_saving_usd"]),
        ),
        Case(
            "savings_dev",
            "What is the potential monthly saving in the dev environment only?",
            {"savings_summary", "find_waste"},
            usd(dev["total_est_monthly_saving_usd"]),
        ),
        Case(
            "no_hallucination",
            "What was our spend on Amazon Kinesis last month?",
            set(),  # any behaviour is fine as long as it admits the data is missing
            ["not", "no data", "unknown", "don't", "doesn't", "cannot", "unable", "no "],
        ),
    ]
    subs = finance.find_subscriptions(LD)
    dup = finance.find_duplicate_charges(LD)[0]
    cases_fin = [
        Case(
            "fin_subs_overlap",
            "Which subscriptions overlap or look redundant?",
            {"find_subscriptions"},
            ["icloud", "google one"],
            "finance",
        ),
        Case(
            "fin_subs_total",
            "How much do I spend on subscriptions per year?",
            {"find_subscriptions"},
            usd(subs["total_annual"]),
            "finance",
        ),
        Case(
            "fin_duplicate",
            "Was I charged twice for anything?",
            {"find_duplicate_charges"},
            [dup["merchant"].split()[0], f"{dup['amount']}"],
            "finance",
        ),
        Case(
            "fin_delivery",
            "Is my food delivery spending going up?",
            {"category_trend"},
            ["increase", "up", "grew", "grow", "rose"],
            "finance",
        ),
    ]
    return base + cases_fin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--only", help="comma-separated case names to run")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument(
        "--sleep", type=float, default=2.0, help="seconds between cases (free-tier rate limits)"
    )
    args = ap.parse_args()
    try:
        provider = make_provider()
    except RuntimeError as exc:
        print(f"{exc}\nSee .env.example (free options: gemini, zai, groq).")
        return 2
    print(f"provider={provider.name} model={provider.model}\n")

    cases = build_cases()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c.name in wanted]
    cases = cases[: args.max or None]
    passed, total_cost = 0, 0.0
    for c in cases:
        dom = FINANCE if c.domain == "finance" else FINOPS
        try:
            r = ask(c.question, AgentConfig(), domain=dom, provider=provider)
        except Exception as exc:  # noqa: BLE001 - a provider outage is a FAIL, not a crash
            print(f"FAIL  {c.name:<18} provider error: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        time.sleep(args.sleep)
        used = {t["name"] for t in r.tool_calls}
        tools_ok = bool(used & c.expect_tools) if c.expect_tools else True
        answer_ok = any(s.lower() in r.answer.lower() for s in c.expect_any)
        ok = tools_ok and answer_ok
        passed += ok
        total_cost += r.cost_usd
        print(
            f"{'PASS' if ok else 'FAIL'}  {c.name:<18} tools={sorted(used)} "
            f"steps={r.steps} ${r.cost_usd:.4f}"
        )
        if not ok:
            print(f"      expected tools∩{sorted(c.expect_tools)}, any of {c.expect_any[:3]}")
            print(f"      answer: {r.answer[:200]!r}")
    rate = passed / len(cases)
    print(f"\n{passed}/{len(cases)} passed ({rate:.0%}), total cost ${total_cost:.4f}")
    return 0 if rate >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
