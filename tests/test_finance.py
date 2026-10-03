from types import SimpleNamespace as NS

import pytest

from finops_agent import finance as f
from finops_agent.agent import FINANCE, AgentConfig, ask


@pytest.fixture(scope="module")
def ld():
    return f.demo_ledger()


def test_categorize_pt_and_en():
    assert f.categorize("IFOOD *RESTAURANTE", -40) == "Food Delivery"
    assert f.categorize("Netflix.com 09/24", -55.9) == "Subscriptions"
    assert f.categorize("SALARIO EMPRESA", 9000) == "Income"
    assert f.categorize("SALARIO ESTORNO", -10) != "Income"
    assert f.categorize("XPTO LTDA", -10) == "Other"


def test_merchant_key_strips_noise():
    assert f.merchant_key("NETFLIX 09/24 *1234") == f.merchant_key("netflix")


def test_category_totals_add_up(ld):
    r = f.spending_by_category(ld, 6)
    assert r["total_spent"] == pytest.approx(sum(c["spent"] for c in r["by_category"]), abs=0.1)
    assert r["by_category"][0]["category"] in {"Rent & Condo", "Groceries"}


def test_subscriptions_find_the_overlapping_storage_plans(ld):
    names = {s["merchant"] for s in f.find_subscriptions(ld)["subscriptions"]}
    assert {"netflix", "spotify", "icloud storage", "google one"} <= names
    assert not ({"aluguel apto", "condominio"} & names)  # rent is not a subscription
    assert f.find_subscriptions(ld)["total_annual"] == pytest.approx(
        f.find_subscriptions(ld)["total_monthly"] * 12, abs=1
    )


def test_planted_duplicate_and_outlier_are_found(ld):
    dups = f.find_duplicate_charges(ld)
    assert any(d["merchant"] == "farmacia drogaria" and d["amount"] == 64.5 for d in dups)
    assert any("NOTEBOOK" in u["description"] for u in f.unusual_transactions(ld))


def test_delivery_trend_is_up(ld):
    assert f.category_trend(ld, "Food Delivery")["change_first_to_last_pct"] > 0
    assert "known_categories" in f.category_trend(ld, "Nope")


def test_csv_brazilian_format():
    text = "Data;Descrição;Valor\n05/09/2026;SALARIO;9.800,00\n08/09/2026;ALUGUEL;-2.600,50\n"
    ld = f.load_statement_csv(text)
    assert [t.amount for t in ld.transactions] == [9800.0, -2600.5]
    assert ld.transactions[0].day.month == 9 and ld.transactions[1].category == "Rent & Condo"


def test_csv_us_format_and_parentheses():
    ld = f.load_statement_csv(
        'Date,Description,Amount\n2026-09-01,Netflix,(15.99)\n2026-09-02,Salary,"3,000.00"\n'
    )
    assert [t.amount for t in ld.transactions] == [-15.99, 3000.0]


@pytest.mark.parametrize(
    "bad", ["", "a,b\n", "x,y,z\n1,2,3\n", "Date,Description,Amount\nfoo,bar,baz\n"]
)
def test_csv_rejects_garbage(bad):
    with pytest.raises(ValueError):
        f.load_statement_csv(bad)


def test_finance_domain_runs_through_the_agent(ld):
    class Client:
        def __init__(self):
            self.n, self.messages = 0, self

        def create(self, **kw):
            self.n += 1
            assert kw["system"] == FINANCE.system_prompt and kw["tools"] is FINANCE.tools
            if self.n == 1:
                block = NS(type="tool_use", id="t1", name="find_subscriptions", input={})
                return NS(
                    content=[block],
                    stop_reason="tool_use",
                    usage=NS(input_tokens=500, output_tokens=50),
                )
            result = kw["messages"][-1]["content"][0]["content"]
            assert "icloud storage" in result
            return NS(
                content=[NS(type="text", text="ok")],
                stop_reason="end_turn",
                usage=NS(input_tokens=700, output_tokens=40),
            )

    out = ask("subscriptions?", AgentConfig(), client=Client(), dataset=ld, domain=FINANCE)
    assert out.answer == "ok" and [c["name"] for c in out.tool_calls] == ["find_subscriptions"]


def test_finance_tools_are_strict():
    for t in f.TOOLS:
        assert t["strict"] and t["input_schema"]["additionalProperties"] is False
        assert t["name"] in f.DISPATCH
