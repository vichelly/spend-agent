import pytest
from fastapi import HTTPException

from finops_agent import api


def test_rate_limit_blocks_after_limit(monkeypatch):
    monkeypatch.setattr(api, "RATE_LIMIT", 2)
    api._hits.clear()
    api._check_rate_limit("1.2.3.4")
    api._check_rate_limit("1.2.3.4")
    with pytest.raises(HTTPException) as e:
        api._check_rate_limit("1.2.3.4")
    assert e.value.status_code == 429
    api._check_rate_limit("5.6.7.8")  # another client is unaffected


def test_daily_budget_blocks(monkeypatch):
    monkeypatch.setattr(api, "DAILY_BUDGET_USD", 1.0)
    api._spend.update(day=__import__("time").strftime("%Y-%m-%d"), usd=1.0)
    with pytest.raises(HTTPException) as e:
        api._check_budget()
    assert e.value.status_code == 503
    api._spend.update(usd=0.0)


def test_health():
    assert api.health()["status"] == "ok"


def test_lambda_handler_exists():
    assert callable(api.handler)
