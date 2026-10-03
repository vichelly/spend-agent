"""Deterministic cost analysis. Pure functions: easy to test, no LLM involved.

The LLM never does arithmetic on raw rows. It calls these functions and explains results.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from statistics import mean, pstdev
from typing import Any

from .data import Dataset

# Conservative, documented savings assumptions (fraction of monthly cost recoverable).
SAVINGS_RATE = {
    "idle_ec2": 1.00,  # stop/terminate
    "unattached_ebs": 1.00,  # snapshot then delete
    "old_snapshot": 1.00,  # delete after retention review
    "oversized_rds": 0.50,  # downsize one instance class
    "dev_nat": 0.70,  # replace with VPC endpoints / schedule off-hours
}
IDLE_CPU_PCT = 7.0
OLD_SNAPSHOT_DAYS = 180


def _window(ds: Dataset, days: int):
    start = ds.end_date - timedelta(days=days - 1)
    return [r for r in ds.costs if r.day >= start]


def total_cost(ds: Dataset, days: int = 30) -> float:
    return round(sum(r.usd for r in _window(ds, days)), 2)


def top_services(ds: Dataset, days: int = 30, n: int = 5) -> list[dict[str, Any]]:
    by_service: dict[str, float] = defaultdict(float)
    for r in _window(ds, days):
        by_service[r.service] += r.usd
    total = sum(by_service.values())
    ranked = sorted(by_service.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [
        {"service": s, "usd": round(v, 2), "share_pct": round(100 * v / total, 1)}
        for s, v in ranked
    ]


def cost_trend(ds: Dataset, service: str, days: int = 30) -> dict[str, Any]:
    """Compare the last `days` against the previous `days` for one service."""
    rows = [r for r in ds.costs if r.service == service]
    if not rows:
        known = sorted({r.service for r in ds.costs})
        return {"error": f"unknown service '{service}'", "known_services": known}
    cur_start = ds.end_date - timedelta(days=days - 1)
    prev_start = cur_start - timedelta(days=days)
    cur = sum(r.usd for r in rows if r.day >= cur_start)
    prev = sum(r.usd for r in rows if prev_start <= r.day < cur_start)
    change = (cur - prev) / prev * 100 if prev else 0.0
    return {
        "service": service,
        "period_days": days,
        "current_usd": round(cur, 2),
        "previous_usd": round(prev, 2),
        "change_pct": round(change, 1),
    }


def detect_anomalies(
    ds: Dataset, z_threshold: float = 3.0, lookback: int = 30
) -> list[dict[str, Any]]:
    """Flag days where a service cost deviates strongly from its trailing mean."""
    by_service: dict[str, list] = defaultdict(list)
    for r in ds.costs:
        by_service[r.service].append(r)
    found: list[dict[str, Any]] = []
    for service, rows in by_service.items():
        rows.sort(key=lambda r: r.day)
        for i in range(lookback, len(rows)):
            hist = [x.usd for x in rows[i - lookback : i]]
            mu, sd = mean(hist), pstdev(hist)
            if sd == 0:
                continue
            z = (rows[i].usd - mu) / sd
            if z >= z_threshold:
                found.append(
                    {
                        "service": service,
                        "date": rows[i].day.isoformat(),
                        "usd": rows[i].usd,
                        "baseline_usd": round(mu, 2),
                        "z_score": round(z, 1),
                    }
                )
    return sorted(found, key=lambda a: (a["date"], a["service"]))


def find_waste(ds: Dataset, env: str | None = None) -> list[dict[str, Any]]:
    """Resources that look wasteful, with an estimated monthly saving for each."""
    if not ds.has_resources:
        return [{"error": "no resource inventory in this dataset; only cost totals are available"}]
    items: list[dict[str, Any]] = []
    for r in ds.resources:
        if env and r.env != env:
            continue
        reason = None
        rate = 0.0
        if r.kind == "ec2" and r.avg_cpu_pct is not None and r.avg_cpu_pct < IDLE_CPU_PCT:
            reason, rate = f"idle EC2 (avg CPU {r.avg_cpu_pct}%)", SAVINGS_RATE["idle_ec2"]
        elif r.kind == "ebs" and r.attached is False:
            reason, rate = "unattached EBS volume", SAVINGS_RATE["unattached_ebs"]
        elif r.kind == "snapshot" and r.age_days > OLD_SNAPSHOT_DAYS:
            reason, rate = (
                f"snapshot older than {OLD_SNAPSHOT_DAYS} days",
                SAVINGS_RATE["old_snapshot"],
            )
        elif r.kind == "rds" and r.avg_cpu_pct is not None and r.avg_cpu_pct < IDLE_CPU_PCT + 3:
            reason, rate = (
                f"oversized RDS (avg CPU {r.avg_cpu_pct}%)",
                SAVINGS_RATE["oversized_rds"],
            )
        elif r.kind == "nat" and r.env == "dev":
            reason, rate = "NAT Gateway in dev", SAVINGS_RATE["dev_nat"]
        if reason:
            items.append(
                {
                    "resource_id": r.resource_id,
                    "kind": r.kind,
                    "env": r.env,
                    "owner": r.owner,
                    "monthly_usd": r.monthly_usd,
                    "reason": reason,
                    "est_monthly_saving_usd": round(r.monthly_usd * rate, 2),
                }
            )
    return sorted(items, key=lambda i: i["est_monthly_saving_usd"], reverse=True)


def savings_summary(ds: Dataset, env: str | None = None) -> dict[str, Any]:
    """Aggregate find_waste() by reason category, with a grand total."""
    waste = find_waste(ds, env)
    if waste and "error" in waste[0]:
        return waste[0]
    by_kind: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "saving": 0.0})
    for w in waste:
        key = w["reason"].split(" (")[0]
        by_kind[key]["count"] += 1
        by_kind[key]["saving"] += w["est_monthly_saving_usd"]
    total = round(sum(w["est_monthly_saving_usd"] for w in waste), 2)
    return {
        "env": env or "all",
        "total_est_monthly_saving_usd": total,
        "total_est_annual_saving_usd": round(total * 12, 2),
        "resources_flagged": len(waste),
        "by_category": {
            k: {"count": int(v["count"]), "est_monthly_saving_usd": round(v["saving"], 2)}
            for k, v in sorted(by_kind.items(), key=lambda kv: kv[1]["saving"], reverse=True)
        },
    }
