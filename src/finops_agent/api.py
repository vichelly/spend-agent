"""HTTP API (FastAPI) + AWS Lambda entrypoint (Mangum).

Public-demo safeguards: input length cap, per-IP rate limit, and a global daily budget.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from mangum import Mangum
from pydantic import BaseModel, Field

from . import __version__
from .agent import DOMAINS, AgentConfig, ask

app = FastAPI(title="FinOps Agent", version=__version__)

RATE_LIMIT = int(os.getenv("FINOPS_RATE_LIMIT_PER_MIN", "6"))
DAILY_BUDGET_USD = float(os.getenv("FINOPS_DAILY_BUDGET_USD", "3.00"))

_hits: dict[str, deque[float]] = defaultdict(deque)
_spend = {"day": time.strftime("%Y-%m-%d"), "usd": 0.0}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    # Optional: your own AWS cost export (Cost Explorer CSV). Never stored; used for this call only.
    csv_text: str | None = Field(default=None, max_length=1_000_000)
    # "finops" = AWS cost export; "finance" = personal bank statement
    mode: Literal["finops", "finance"] = "finops"


class AskResponse(BaseModel):
    answer: str
    steps: int
    tools_used: list[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    stopped_reason: str


def _check_rate_limit(client_id: str) -> None:
    now = time.time()
    window = _hits[client_id]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(429, "Rate limit exceeded. Try again in a minute.")
    window.append(now)


def _check_budget() -> None:
    today = time.strftime("%Y-%m-%d")
    if _spend["day"] != today:
        _spend.update(day=today, usd=0.0)
    if _spend["usd"] >= DAILY_BUDGET_USD:
        raise HTTPException(503, "Daily demo budget exhausted. Please try again tomorrow.")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest, request: Request) -> AskResponse:
    _check_rate_limit(request.client.host if request.client else "unknown")
    _check_budget()
    domain = DOMAINS[body.mode]
    if body.csv_text:
        try:
            dataset = domain.load_csv(body.csv_text)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    else:
        dataset = domain.demo()
    result = ask(body.question, AgentConfig(), dataset=dataset, domain=domain)
    _spend["usd"] += result.cost_usd
    return AskResponse(
        answer=result.answer,
        steps=result.steps,
        tools_used=[c["name"] for c in result.tool_calls],
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=round(result.cost_usd, 4),
        stopped_reason=result.stopped_reason,
    )


handler = Mangum(app)  # AWS Lambda entrypoint: finops_agent.api.handler
