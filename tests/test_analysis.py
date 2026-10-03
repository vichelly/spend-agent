import pytest

from finops_agent import analysis
from finops_agent.data import Dataset, demo_dataset, load_costs_csv


@pytest.fixture(scope="module")
def ds():
    return demo_dataset()


def test_total_cost_is_stable_and_positive(ds):
    assert analysis.total_cost(ds, 30) == analysis.total_cost(ds, 30)
    assert analysis.total_cost(ds, 30) > analysis.total_cost(ds, 7) > 0


def test_top_services_sorted_and_shares_sane(ds):
    top = analysis.top_services(ds, 30, 3)
    assert [t["usd"] for t in top] == sorted((t["usd"] for t in top), reverse=True)
    assert top[0]["service"] == "Amazon EC2"
    assert 0 < sum(t["share_pct"] for t in top) <= 100


def test_cost_trend_unknown_service_lists_known(ds):
    out = analysis.cost_trend(ds, "Amazon Nope", 30)
    assert "error" in out and "Amazon EC2" in out["known_services"]


def test_cost_trend_nat_is_growing(ds):
    assert analysis.cost_trend(ds, "NAT Gateway", 30)["change_pct"] > 0


def test_planted_nat_spike_is_detected(ds):
    flagged = analysis.detect_anomalies(ds)
    assert any(a["service"] == "NAT Gateway" and a["z_score"] > 10 for a in flagged)


def test_waste_total_matches_summary(ds):
    waste = analysis.find_waste(ds)
    summary = analysis.savings_summary(ds)
    assert summary["resources_flagged"] == len(waste)
    assert summary["total_est_monthly_saving_usd"] == pytest.approx(
        sum(w["est_monthly_saving_usd"] for w in waste), abs=0.05
    )
    assert summary["total_est_annual_saving_usd"] == pytest.approx(
        summary["total_est_monthly_saving_usd"] * 12, abs=0.1
    )


def test_env_filter_only_returns_that_env(ds):
    assert {w["env"] for w in analysis.find_waste(ds, "dev")} == {"dev"}


def test_dataset_without_inventory_reports_it():
    d = load_costs_csv("Date,Service,Cost\n2026-09-01,Amazon S3,5\n2026-09-02,Amazon S3,6\n")
    assert "error" in analysis.find_waste(d)[0]
    assert "error" in analysis.savings_summary(d)


def test_csv_long_and_wide_agree():
    long_ = load_costs_csv(
        "Date,Service,Cost\n2026-09-01,Amazon EC2,10\n2026-09-02,Amazon EC2,12\n"
    )
    wide = load_costs_csv(
        "Service,2026-09-01,2026-09-02,Service total\nAmazon EC2,10,12,22\nTotal,10,12,22\n"
    )
    assert analysis.total_cost(long_, 2) == analysis.total_cost(wide, 2) == 22.0


def test_csv_handles_currency_symbols_and_bom():
    d = load_costs_csv('﻿Date,Service,Cost\n2026-09-01,Amazon S3,"$1,234.50"\n')
    assert isinstance(d, Dataset) and d.costs[0].usd == 1234.5


@pytest.mark.parametrize("bad", ["", "a,b\n", "foo,bar\n1,2\n", "Date,Service,Cost\nx,y,z\n"])
def test_csv_rejects_garbage(bad):
    with pytest.raises(ValueError):
        load_costs_csv(bad)
