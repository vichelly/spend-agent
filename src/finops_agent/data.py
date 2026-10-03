"""Synthetic AWS cost dataset.

The public demo never touches a real AWS account. Data is generated with a fixed seed so
results are reproducible and the eval suite can compute expected answers from the same
source of truth the agent's tools use.
"""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache

SERVICES = {
    # service: (base daily USD, weekly seasonality, monthly growth)
    "Amazon EC2": (410.0, 0.06, 0.010),
    "Amazon RDS": (230.0, 0.02, 0.006),
    "Amazon S3": (95.0, 0.01, 0.015),
    "AWS Lambda": (40.0, 0.10, 0.020),
    "Amazon CloudWatch": (62.0, 0.01, 0.030),
    "NAT Gateway": (118.0, 0.03, 0.008),
    "Amazon DynamoDB": (55.0, 0.04, 0.012),
    "Amazon ECS": (160.0, 0.05, 0.004),
}

END_DATE = date(2026, 9, 30)
DAYS = 120


@dataclass(frozen=True)
class DailyCost:
    day: date
    service: str
    usd: float


@dataclass(frozen=True)
class Resource:
    resource_id: str
    kind: str  # ec2 | ebs | snapshot | nat | rds | elb
    service: str
    monthly_usd: float
    env: str  # prod | staging | dev
    owner: str
    avg_cpu_pct: float | None = None  # ec2 / rds
    attached: bool | None = None  # ebs / elb
    age_days: int = 0
    last_access_days: int | None = None


@lru_cache(maxsize=1)
def daily_costs() -> tuple[DailyCost, ...]:
    rng = random.Random(42)
    rows: list[DailyCost] = []
    for i in range(DAYS):
        day = END_DATE - timedelta(days=DAYS - 1 - i)
        for service, (base, seasonality, growth) in SERVICES.items():
            weekday_factor = 1 - seasonality if day.weekday() >= 5 else 1.0
            trend = 1 + growth * (i / 30)
            noise = rng.uniform(0.97, 1.03)
            rows.append(DailyCost(day, service, round(base * weekday_factor * trend * noise, 2)))
    # a planted anomaly: a runaway NAT Gateway data-transfer spike
    spike_days = {END_DATE - timedelta(days=d) for d in (9, 8, 7)}
    return tuple(
        DailyCost(r.day, r.service, round(r.usd * 3.4, 2))
        if r.service == "NAT Gateway" and r.day in spike_days
        else r
        for r in rows
    )


@lru_cache(maxsize=1)
def resources() -> tuple[Resource, ...]:
    rng = random.Random(7)
    owners = ["payments", "platform", "data", "growth", "search"]
    envs = ["prod", "staging", "dev"]
    out: list[Resource] = []

    for i in range(40):  # EC2 fleet, some clearly idle
        idle = i % 5 == 0
        out.append(
            Resource(
                f"i-{i:04d}a",
                "ec2",
                "Amazon EC2",
                round(rng.uniform(60, 420), 2),
                rng.choice(envs),
                rng.choice(owners),
                avg_cpu_pct=round(rng.uniform(1, 6) if idle else rng.uniform(25, 80), 1),
                age_days=rng.randint(20, 700),
            )
        )
    for i in range(25):  # EBS volumes, ~40% unattached
        attached = i % 5 not in (0, 1)
        out.append(
            Resource(
                f"vol-{i:04d}b",
                "ebs",
                "Amazon EC2",
                round(rng.uniform(8, 95), 2),
                rng.choice(envs),
                rng.choice(owners),
                attached=attached,
                age_days=rng.randint(10, 900),
                last_access_days=rng.randint(0, 5) if attached else rng.randint(40, 300),
            )
        )
    for i in range(30):  # old snapshots
        out.append(
            Resource(
                f"snap-{i:04d}c",
                "snapshot",
                "Amazon EC2",
                round(rng.uniform(2, 30), 2),
                rng.choice(envs),
                rng.choice(owners),
                age_days=rng.randint(30, 800),
            )
        )
    for i in range(6):  # RDS, a couple oversized
        small = i % 3 == 0
        out.append(
            Resource(
                f"db-{i:02d}d",
                "rds",
                "Amazon RDS",
                round(rng.uniform(180, 900), 2),
                rng.choice(envs),
                rng.choice(owners),
                avg_cpu_pct=round(rng.uniform(3, 9) if small else rng.uniform(30, 70), 1),
                age_days=rng.randint(60, 900),
            )
        )
    for i in range(4):  # NAT gateways in non-prod
        out.append(
            Resource(
                f"nat-{i:02d}e",
                "nat",
                "NAT Gateway",
                round(rng.uniform(60, 130), 2),
                "dev" if i < 2 else "prod",
                rng.choice(owners),
                age_days=rng.randint(100, 600),
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class Dataset:
    costs: tuple[DailyCost, ...]
    resources: tuple[Resource, ...] = field(default_factory=tuple)
    label: str = "demo"

    @property
    def end_date(self) -> date:
        return max(c.day for c in self.costs)

    @property
    def has_resources(self) -> bool:
        return bool(self.resources)


def demo_dataset() -> Dataset:
    return Dataset(daily_costs(), resources(), "synthetic demo data")


_DATE_KEYS = ("date", "usagedate", "usagestartdate", "chargeperiodstart", "day")
_SERVICE_KEYS = ("service", "servicename", "productname", "lineitem/productcode", "servicecode")
_COST_KEYS = ("cost", "unblendedcost", "billedcost", "amount", "blendedcost", "effectivecost")


def _pick(header: list[str], keys: tuple[str, ...]) -> int | None:
    lowered = [h.strip().lower() for h in header]
    for k in keys:
        if k in lowered:
            return lowered.index(k)
    return None


def load_costs_csv(text: str, label: str = "uploaded CSV", max_rows: int = 200_000) -> Dataset:
    """Parse an AWS cost export.

    Supported shapes:
      * long:  columns like Date, Service, Cost  (Cost Explorer grouped daily, CUR/FOCUS exports)
      * wide:  a Service column plus one column per ISO date (Cost Explorer "Download CSV")
    Raises ValueError with a human-readable message when the file is not recognized.
    """
    reader = csv.reader(io.StringIO(text.lstrip("\ufeff")))
    rows = list(reader)
    if len(rows) < 2:
        raise ValueError("CSV needs a header row and at least one data row.")
    if len(rows) > max_rows:
        raise ValueError(f"CSV too large ({len(rows)} rows, max {max_rows}).")
    header = rows[0]
    out: list[DailyCost] = []

    di, si, ci = _pick(header, _DATE_KEYS), _pick(header, _SERVICE_KEYS), _pick(header, _COST_KEYS)
    if di is not None and si is not None and ci is not None:
        for line in rows[1:]:
            if len(line) <= max(di, si, ci):
                continue
            try:
                day = date.fromisoformat(line[di].strip()[:10])
                usd = float(line[ci].replace("$", "").replace(",", ""))
            except ValueError:
                continue
            out.append(DailyCost(day, line[si].strip(), usd))
    else:
        # wide format: first column is the service, date columns are ISO dates
        date_cols: dict[int, date] = {}
        for idx, name in enumerate(header[1:], start=1):
            try:
                date_cols[idx] = date.fromisoformat(name.strip()[:10])
            except ValueError:
                continue
        if not date_cols:
            raise ValueError(
                "Unrecognized CSV. Expected columns Date/Service/Cost, or a Service column "
                "followed by ISO-date columns (Cost Explorer export)."
            )
        for line in rows[1:]:
            service = line[0].strip()
            if not service or service.lower().endswith("total"):
                continue
            for idx, day in date_cols.items():
                if idx < len(line) and line[idx].strip():
                    try:
                        out.append(
                            DailyCost(
                                day, service, float(line[idx].replace("$", "").replace(",", ""))
                            )
                        )
                    except ValueError:
                        continue
    if not out:
        raise ValueError("No cost rows could be parsed from the CSV.")
    return Dataset(tuple(out), (), label)
