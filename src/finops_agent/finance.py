"""Personal-finance domain: bank-statement CSV -> deterministic analysis -> Claude explains.

Privacy by design: the raw statement never goes to the model. Claude only sees the compact
JSON results of the tools below (aggregates and a handful of flagged rows).
"""

from __future__ import annotations

import csv
import io
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from statistics import mean, median, pstdev
from typing import Any

# --- categorization (keyword rules, PT + EN; first match wins) ---------------------------
CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Income", ("salary", "salario", "payroll", "deposito", "pix recebido", "freelance")),
    ("Rent & Condo", ("rent", "aluguel", "condominio", "mortgage")),
    ("Utilities", ("energia", "agua", "internet", "luz")),
    ("Groceries", ("supermercado", "mercado", "grocery", "carrefour", "extra", "pao de acucar")),
    ("Food Delivery", ("ifood", "rappi", "ubereats", "uber eats", "delivery", "doordash")),
    ("Restaurants", ("restaurante", "restaurant", "lanchonete", "cafe", "padaria", "starbucks")),
    ("Transport", ("uber", "99", "taxi", "gasolina", "fuel", "metro", "sptrans", "estacionamento")),
    (
        "Subscriptions",
        (
            "netflix",
            "spotify",
            "disney",
            "hbo",
            "prime video",
            "youtube",
            "icloud",
            "google one",
            "gym",
            "academia",
            "smartfit",
            "adobe",
            "chatgpt",
            "notion",
        ),
    ),
    ("Health", ("farmacia", "pharmacy", "drogaria", "medico", "clinica", "plano de saude")),
    ("Shopping", ("amazon", "mercado livre", "shopee", "magalu", "aliexpress", "zara", "loja")),
    ("Entertainment", ("cinema", "ingresso", "steam", "playstation", "show")),
    ("Travel", ("hotel", "airbnb", "latam", "azul", "gol", "booking", "passagem")),
]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", text).strip()


def categorize(description: str, amount: float) -> str:
    d = _norm(description)
    for category, words in CATEGORY_RULES:
        if any(w in d for w in words):
            if category == "Income" and amount < 0:
                continue
            return category
    return "Income" if amount > 0 else "Other"


def merchant_key(description: str) -> str:
    """Stable merchant id: strip digits, dates and noise so 'NETFLIX 09/24' == 'NETFLIX'."""
    d = _norm(description)
    d = re.sub(r"[\d/\-\.\*#]+", " ", d)
    d = re.sub(r"\b(compra|pagto|pagamento|debito|credito|cartao|pix|ref)\b", " ", d)
    return re.sub(r"\s+", " ", d).strip() or "unknown"


@dataclass(frozen=True)
class Transaction:
    day: date
    description: str
    amount: float  # negative = money out
    category: str


@dataclass(frozen=True)
class Ledger:
    transactions: tuple[Transaction, ...]
    currency: str = "R$"
    label: str = "synthetic demo statement"

    @property
    def end_date(self) -> date:
        return max(t.day for t in self.transactions)


# --- demo data ---------------------------------------------------------------------------
def demo_ledger() -> Ledger:
    rng = random.Random(11)
    rows: list[tuple[date, str, float]] = []
    start = date(2026, 4, 1)
    for m in range(6):  # Apr..Sep 2026
        y, mo = (start.year, start.month + m)
        d0 = date(y, mo, 1)
        rows.append((d0.replace(day=5), "SALARIO EMPRESA XYZ", 9800.0))
        rows.append((d0.replace(day=8), "ALUGUEL APTO", -2600.0))
        rows.append((d0.replace(day=9), "CONDOMINIO", -780.0))
        rows.append((d0.replace(day=12), "ENERGIA ELETRICA", -round(rng.uniform(130, 210), 2)))
        rows.append((d0.replace(day=12), "INTERNET FIBRA", -119.9))
        for sub, price, day in [
            ("NETFLIX", 55.9, 3),
            ("SPOTIFY", 21.9, 7),
            ("SMARTFIT ACADEMIA", 129.9, 10),
        ]:
            rows.append((d0.replace(day=day), sub, -price))
        # forgotten overlap: two cloud-storage plans
        rows.append((d0.replace(day=15), "ICLOUD STORAGE", -14.9))
        rows.append((d0.replace(day=16), "GOOGLE ONE", -9.9))
        for _ in range(4):
            rows.append(
                (
                    d0.replace(day=rng.randint(1, 28)),
                    "SUPERMERCADO PAO DE ACUCAR",
                    -round(rng.uniform(180, 420), 2),
                )
            )
        for _ in range(rng.randint(9, 14)):  # delivery habit that grows over time
            rows.append(
                (
                    d0.replace(day=rng.randint(1, 28)),
                    "IFOOD *RESTAURANTE",
                    -round(rng.uniform(32, 78) * (1 + m * 0.05), 2),
                )
            )
        for _ in range(rng.randint(5, 9)):
            rows.append(
                (d0.replace(day=rng.randint(1, 28)), "UBER *TRIP", -round(rng.uniform(14, 48), 2))
            )
    rows.append((date(2026, 9, 18), "AMAZON COMPRA NOTEBOOK", -5890.0))  # unusual
    rows.append((date(2026, 9, 21), "FARMACIA DROGARIA", -64.5))
    rows.append((date(2026, 9, 21), "FARMACIA DROGARIA", -64.5))  # duplicate charge
    txs = tuple(Transaction(d, desc, amt, categorize(desc, amt)) for d, desc, amt in sorted(rows))
    return Ledger(txs)


# --- CSV loader ---------------------------------------------------------------------------
_DATE_KEYS = ("date", "data", "posted date", "transaction date", "data lancamento")
_DESC_KEYS = (
    "description",
    "descricao",
    "memo",
    "name",
    "historico",
    "lancamento",
    "estabelecimento",
)
_AMOUNT_KEYS = ("amount", "valor", "value", "total")


def _find(header: list[str], keys: tuple[str, ...]) -> int | None:
    h = [_norm(x) for x in header]
    for k in keys:
        if k in h:
            return h.index(k)
    return None


def _parse_amount(raw: str) -> float:
    s = raw.strip().replace("R$", "").replace("$", "").replace(" ", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    if "," in s and "." in s:  # 1.234,56 (BR) vs 1,234.56 (US)
        s = (
            s.replace(".", "").replace(",", ".")
            if s.rfind(",") > s.rfind(".")
            else s.replace(",", "")
        )
    elif "," in s:
        s = s.replace(",", ".")
    value = float(s)
    return -abs(value) if neg else value


def _parse_date(raw: str) -> date:
    s = raw.strip()[:10]
    if re.match(r"\d{4}-\d{2}-\d{2}", s):
        return date.fromisoformat(s)
    m = re.match(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", s)
    if m:
        return date(int(m[3]), int(m[2]), int(m[1]))  # day-first (Brazil)
    raise ValueError(raw)


def load_statement_csv(
    text: str, label: str = "uploaded statement", max_rows: int = 50_000
) -> Ledger:
    """Parse a bank statement: columns Date/Data, Description/Descrição, Amount/Valor.
    Negative amounts are expenses. Handles dd/mm/yyyy and 1.234,56."""
    sample = text.lstrip("﻿")
    dialect = csv.Sniffer().sniff(sample[:2000], delimiters=",;\t") if sample else csv.excel
    rows = list(csv.reader(io.StringIO(sample), dialect))
    if len(rows) < 2:
        raise ValueError("CSV needs a header row and at least one transaction.")
    if len(rows) > max_rows:
        raise ValueError(f"CSV too large ({len(rows)} rows, max {max_rows}).")
    header = rows[0]
    di, ni, ai = _find(header, _DATE_KEYS), _find(header, _DESC_KEYS), _find(header, _AMOUNT_KEYS)
    if di is None or ni is None or ai is None:
        raise ValueError("Unrecognized statement. Expected columns Date, Description, Amount.")
    out: list[Transaction] = []
    for line in rows[1:]:
        if len(line) <= max(di, ni, ai):
            continue
        try:
            d, amt = _parse_date(line[di]), _parse_amount(line[ai])
        except ValueError:
            continue
        out.append(Transaction(d, line[ni].strip(), amt, categorize(line[ni], amt)))
    if not out:
        raise ValueError("No transactions could be parsed from the CSV.")
    return Ledger(tuple(sorted(out, key=lambda t: t.day)), "R$", label)


# --- analysis (pure) ----------------------------------------------------------------------
def _since(ledger: Ledger, months: int) -> list[Transaction]:
    start = ledger.end_date - timedelta(days=30 * months - 1)
    return [t for t in ledger.transactions if t.day >= start]


def spending_by_category(ledger: Ledger, months: int = 1) -> dict[str, Any]:
    spent: dict[str, float] = defaultdict(float)
    for t in _since(ledger, months):
        if t.amount < 0:
            spent[t.category] += -t.amount
    total = sum(spent.values())
    return {
        "months": months,
        "currency": ledger.currency,
        "total_spent": round(total, 2),
        "by_category": [
            {
                "category": c,
                "spent": round(v, 2),
                "share_pct": round(100 * v / total, 1) if total else 0,
            }
            for c, v in sorted(spent.items(), key=lambda kv: kv[1], reverse=True)
        ],
    }


def top_merchants(ledger: Ledger, months: int = 1, n: int = 5) -> list[dict[str, Any]]:
    by: dict[str, list[float]] = defaultdict(list)
    for t in _since(ledger, months):
        if t.amount < 0:
            by[merchant_key(t.description)].append(-t.amount)
    ranked = sorted(by.items(), key=lambda kv: sum(kv[1]), reverse=True)[:n]
    return [{"merchant": k, "spent": round(sum(v), 2), "transactions": len(v)} for k, v in ranked]


def monthly_summary(ledger: Ledger) -> list[dict[str, Any]]:
    inc: dict[str, float] = defaultdict(float)
    out: dict[str, float] = defaultdict(float)
    for t in ledger.transactions:
        key = t.day.strftime("%Y-%m")
        (inc if t.amount > 0 else out)[key] += abs(t.amount)
    return [
        {
            "month": m,
            "income": round(inc[m], 2),
            "expenses": round(out[m], 2),
            "net": round(inc[m] - out[m], 2),
        }
        for m in sorted(set(inc) | set(out))
    ]


def category_trend(ledger: Ledger, category: str) -> dict[str, Any]:
    by_month: dict[str, float] = defaultdict(float)
    for t in ledger.transactions:
        if t.category.lower() == category.lower() and t.amount < 0:
            by_month[t.day.strftime("%Y-%m")] += -t.amount
    if not by_month:
        return {
            "error": f"no spending in category '{category}'",
            "known_categories": sorted({t.category for t in ledger.transactions}),
        }
    months = sorted(by_month)
    first, last = by_month[months[0]], by_month[months[-1]]
    return {
        "category": category,
        "monthly": [{"month": m, "spent": round(by_month[m], 2)} for m in months],
        "change_first_to_last_pct": round((last - first) / first * 100, 1) if first else None,
    }


def find_subscriptions(ledger: Ledger) -> dict[str, Any]:
    """Merchants charged in 3+ distinct months with a near-constant amount."""
    by: dict[str, list[Transaction]] = defaultdict(list)
    for t in ledger.transactions:
        if t.amount < 0 and t.category != "Rent & Condo":  # fixed obligations, not subscriptions
            by[merchant_key(t.description)].append(t)
    subs = []
    for m, txs in by.items():
        months = {t.day.strftime("%Y-%m") for t in txs}
        amounts = [-t.amount for t in txs]
        if (
            len(months) >= 3
            and len(txs) <= len(months) * 1.5
            and (max(amounts) - min(amounts)) <= 0.1 * mean(amounts)
        ):
            subs.append(
                {
                    "merchant": m,
                    "monthly_cost": round(mean(amounts), 2),
                    "annual_cost": round(mean(amounts) * 12, 2),
                    "months_seen": len(months),
                }
            )
    subs.sort(key=lambda s: s["monthly_cost"], reverse=True)
    return {
        "currency": ledger.currency,
        "subscriptions": subs,
        "total_monthly": round(sum(s["monthly_cost"] for s in subs), 2),
        "total_annual": round(sum(s["annual_cost"] for s in subs), 2),
    }


def find_duplicate_charges(ledger: Ledger, window_days: int = 3) -> list[dict[str, Any]]:
    """Same merchant and exact amount within a few days (excluding monthly-recurring patterns)."""
    by: dict[tuple[str, float], list[Transaction]] = defaultdict(list)
    for t in ledger.transactions:
        if t.amount < 0:
            by[(merchant_key(t.description), t.amount)].append(t)
    found = []
    for (m, amt), txs in by.items():
        txs.sort(key=lambda t: t.day)
        for a, b in pairwise(txs):
            if (b.day - a.day).days <= window_days:
                found.append(
                    {"merchant": m, "amount": -amt, "dates": [a.day.isoformat(), b.day.isoformat()]}
                )
    return found


def unusual_transactions(ledger: Ledger, z: float = 3.0) -> list[dict[str, Any]]:
    """Expenses far above normal. Uses the category's own history when there are 5+ samples,
    otherwise falls back to all non-housing expenses (so a first-ever big purchase is caught)."""
    by_cat: dict[str, list[Transaction]] = defaultdict(list)
    for t in ledger.transactions:
        if t.amount < 0:
            by_cat[t.category].append(t)
    general = [-t.amount for c, txs in by_cat.items() if c != "Rent & Condo" for t in txs]
    flagged = []
    for cat, txs in by_cat.items():
        if cat == "Rent & Condo":
            continue
        amts = [-t.amount for t in txs]
        pool = amts if len(amts) >= 5 else general
        if len(pool) < 5:
            continue
        mu, sd, med = mean(pool), pstdev(pool), median(pool)
        for t in txs:
            if sd and (-t.amount - mu) / sd >= z and -t.amount > 2 * med:
                flagged.append(
                    {
                        "date": t.day.isoformat(),
                        "description": t.description,
                        "amount": -t.amount,
                        "category": cat,
                        "typical": round(med, 2),
                    }
                )
    return sorted(flagged, key=lambda f: f["amount"], reverse=True)


# --- tool definitions for Claude ---------------------------------------------------------
_MONTHS = {"type": "integer", "minimum": 1, "maximum": 24}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "spending_by_category",
        "description": "Total spent and breakdown by category over the last N months (30-day blocks).",
        "input_schema": {
            "type": "object",
            "properties": {"months": _MONTHS},
            "required": ["months"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "top_merchants",
        "description": "Top N merchants by spend over the last N months.",
        "input_schema": {
            "type": "object",
            "properties": {
                "months": _MONTHS,
                "n": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["months", "n"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "monthly_summary",
        "description": "Income, expenses and net per calendar month for the whole statement.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "category_trend",
        "description": "Month-by-month spend for one category (e.g. 'Food Delivery', 'Groceries', 'Transport').",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "find_subscriptions",
        "description": "Recurring monthly charges with monthly and annual cost, including overlapping services.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "find_duplicate_charges",
        "description": "Possible double charges: same merchant and amount within 3 days.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "unusual_transactions",
        "description": "Expenses far above normal for their category.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
]

DISPATCH = {
    "spending_by_category": lambda ld, a: spending_by_category(ld, a["months"]),
    "top_merchants": lambda ld, a: top_merchants(ld, a["months"], a["n"]),
    "monthly_summary": lambda ld, a: monthly_summary(ld),
    "category_trend": lambda ld, a: category_trend(ld, a["category"]),
    "find_subscriptions": lambda ld, a: find_subscriptions(ld),
    "find_duplicate_charges": lambda ld, a: find_duplicate_charges(ld),
    "unusual_transactions": lambda ld, a: unusual_transactions(ld),
}

SYSTEM_PROMPT = """You are a personal-finance assistant. You help one person understand their \
bank statement and spend less on things they do not value.

Rules:
- Never compute totals, percentages or rankings from memory. Always call a tool and quote its numbers.
- Use the currency symbol the tool returns. Round to whole units unless under 100.
- Be practical and non-judgmental: lead with the answer, then at most three concrete actions, \
each with the monthly and yearly amount it would free up.
- You only see tool results, never the raw statement. If the data cannot answer the question, say so.
- You are not a licensed financial advisor. Do not recommend investments; for those, say so briefly."""
