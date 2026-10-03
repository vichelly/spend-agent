"""CLI:  python -m finops_agent "What are my top 3 services?" [--csv my_costs.csv]"""

import argparse
import sys
from pathlib import Path

from .agent import DOMAINS, AgentConfig, ask
from .env import load_dotenv


def main() -> int:
    ap = argparse.ArgumentParser(prog="finops_agent")
    ap.add_argument("question")
    ap.add_argument("--mode", choices=sorted(DOMAINS), default="finops")
    ap.add_argument("--csv", type=Path, help="your own CSV (default: demo data)")
    ap.add_argument("--verbose", "-v", action="store_true")
    load_dotenv()
    args = ap.parse_args()

    domain = DOMAINS[args.mode]
    dataset = domain.load_csv(args.csv.read_text()) if args.csv else domain.demo()
    result = ask(args.question, AgentConfig(), dataset=dataset, domain=domain)
    print(result.answer)
    if args.verbose:
        tools = ", ".join(c["name"] for c in result.tool_calls) or "none"
        print(
            f"\n[{result.steps} steps | tools: {tools} | {result.input_tokens} in / "
            f"{result.output_tokens} out | ${result.cost_usd:.4f} | {result.stopped_reason}]",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
