"""Generate a synthetic mobile-money dataset in the same format as PaySim.

PaySim (https://www.kaggle.com/datasets/ealaxi/paysim1) is itself synthetic. This script makes a
smaller file with the same columns, so the whole project runs without a Kaggle download, and the
real PaySim CSV can be dropped in instead (see docs/data.md).

What is simulated:
  * Customers ("C...") with their own typical amount and activity level, paying merchants
    ("M..."), cashing in and out, and sending money to a few regular contacts.
  * Some legitimate one-off large transfers to new people (rent deposits, family help), so a
    "big transfer to someone new" is not automatically fraud.
  * Fraud cases: an account is taken over, money is TRANSFERred to a mule account, and the mule
    quickly CASH_OUTs it. A mule is reused by several victims. Both transactions are fraud
    (isFraud = 1), as in PaySim. A quarter of cases are small ("low and slow").
  * Balance columns behave like PaySim's, including the leak: a fraud empties the victim's
    account. ml/data.py drops these columns because they give the answer away.

Usage:
    python -m sim.generate --out data/transactions.csv
    python -m sim.generate --out data/shifted.csv --shift 3 --seed 99   # for the drift check
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
TYPE_PROBS = [0.20, 0.15, 0.06, 0.47, 0.12]
TYPE_AMOUNT_FACTOR = {"CASH_IN": 1.5, "CASH_OUT": 1.2, "DEBIT": 0.4, "PAYMENT": 0.5,
                      "TRANSFER": 2.0}
FLAG_LIMIT = 200_000.0  # PaySim's own simple rule: flag TRANSFERs above this amount


@dataclass(frozen=True)
class SimConfig:
    customers: int = 25_000
    merchants: int = 2_000
    days: int = 30
    tx_per_customer_day: float = 0.7
    fraud_case_rate: float = 0.02  # share of customers whose account is taken over
    big_legit_rate: float = 0.01  # share making one legit large transfer to someone new
    shift: float = 1.0  # >1 makes amounts bigger and transfers more common (drift test)
    seed: int = 7


@dataclass
class Tx:
    t: float  # continuous time in hours; step = int(t)
    type: str
    amount: float
    orig: str
    dest: str
    fraud: int
    old_orig: float = 0.0
    new_orig: float = 0.0
    old_dest: float = 0.0
    new_dest: float = 0.0


def _ids(rng: np.random.Generator, prefix: str, n: int, taken: set[str]) -> list[str]:
    """n unique PaySim-style ids such as C1231006815."""
    out: list[str] = []
    while len(out) < n:
        name = f"{prefix}{int(rng.integers(100_000_000, 2_000_000_000))}"
        if name not in taken:
            taken.add(name)
            out.append(name)
    return out


def _hour_of_day(rng: np.random.Generator, size: int) -> np.ndarray:
    """Hours 0-23, busier in the daytime."""
    weights = np.array([1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 10, 10, 10, 10, 9, 9, 9, 9, 8, 7, 5, 3, 2, 1],
                       dtype=float)
    return rng.choice(24, size=size, p=weights / weights.sum())


def simulate(cfg: SimConfig) -> list[Tx]:
    """All transactions, sorted by time."""
    rng = np.random.default_rng(cfg.seed)
    taken: set[str] = set()
    customers = _ids(rng, "C", cfg.customers, taken)
    merchants = _ids(rng, "M", cfg.merchants, taken)
    hours = cfg.days * 24

    scale = np.exp(rng.normal(np.log(150.0), 0.8, cfg.customers)) * cfg.shift
    rate = rng.gamma(2.0, cfg.tx_per_customer_day / 2.0, cfg.customers)
    contacts = [rng.choice(cfg.customers, size=int(rng.integers(2, 6)), replace=False)
                for _ in range(cfg.customers)]
    probs = np.array(TYPE_PROBS)
    if cfg.shift != 1.0:  # drift: more transfers, fewer payments
        probs = probs * np.array([1.0, 1.0, 1.0, 1.0 / cfg.shift, cfg.shift])
        probs = probs / probs.sum()

    txs: list[Tx] = []
    for c in range(cfg.customers):
        n = rng.poisson(rate[c] * cfg.days)
        if n == 0:
            continue
        days = rng.integers(0, cfg.days, n)
        t = days * 24 + _hour_of_day(rng, n) + rng.random(n)
        types = rng.choice(len(TYPES), size=n, p=probs)
        noise = np.exp(rng.normal(0.0, 0.6, n))
        for i in range(n):
            ttype = TYPES[types[i]]
            amount = round(float(scale[c] * TYPE_AMOUNT_FACTOR[ttype] * noise[i]), 2)
            if ttype == "TRANSFER":
                # 80% to a regular contact, 20% to anyone
                other = (int(rng.choice(contacts[c])) if rng.random() < 0.8
                         else int(rng.integers(cfg.customers)))
                dest = customers[other] if other != c else customers[(c + 1) % cfg.customers]
            else:
                dest = merchants[int(rng.integers(cfg.merchants))]
            txs.append(Tx(float(t[i]), ttype, amount, customers[c], dest, 0))

    # Legitimate one-off large transfers to someone new (hard negatives).
    for c in rng.choice(cfg.customers, size=int(cfg.customers * cfg.big_legit_rate), replace=False):
        t = float(rng.uniform(24, hours - 1))
        amount = round(float(scale[c] * rng.uniform(8, 30)), 2)
        dest = customers[int(rng.integers(cfg.customers))]
        txs.append(Tx(t, "TRANSFER", amount, customers[int(c)], dest, 0))

    # Fraud: account takeover -> TRANSFER to a mule -> mule CASH_OUT soon after.
    n_cases = int(cfg.customers * cfg.fraud_case_rate)
    mules = _ids(rng, "C", max(1, n_cases // 3), taken)
    victims = rng.choice(cfg.customers, size=n_cases, replace=False)
    for v in victims:
        mule = mules[int(rng.integers(len(mules)))]
        t0 = float(rng.uniform(24, hours - 4))
        if rng.random() < 0.25:  # low and slow: looks like a normal amount
            amount = float(scale[v] * rng.uniform(1.0, 3.0))
        else:
            amount = float(scale[v] * rng.uniform(5.0, 40.0))
        bursts = 2 if rng.random() < 0.3 else 1
        for b in range(bursts):
            t = t0 + b * rng.uniform(0.01, 0.5)
            amt = round(amount / bursts, 2)
            txs.append(Tx(t, "TRANSFER", amt, customers[int(v)], mule, 1))
            t_out = t + rng.uniform(0.05, 3.0)
            if t_out < hours:
                agent = merchants[int(rng.integers(cfg.merchants))]
                txs.append(Tx(t_out, "CASH_OUT", amt, mule, agent, 1))

    txs = [tx for tx in txs if tx.t < hours]
    txs.sort(key=lambda tx: tx.t)
    _fill_balances(txs, rng, scale, customers)
    return txs


def _fill_balances(txs: list[Tx], rng: np.random.Generator, scale: np.ndarray,
                   customers: list[str]) -> None:
    """PaySim-like balance columns. Fraud drains the victim (the leak ml/data.py removes)."""
    scale_of = dict(zip(customers, scale, strict=True))
    balance: dict[str, float] = {}
    for tx in txs:
        base = float(scale_of.get(tx.orig, 100.0))
        old = balance.get(tx.orig, round(base * float(rng.uniform(3, 20)), 2))
        if tx.fraud and tx.type == "TRANSFER":
            old = max(old, tx.amount)
            new = round(old - tx.amount, 2) if rng.random() < 0.3 else 0.0
        elif tx.type == "CASH_IN":
            new = round(old + tx.amount, 2)
        else:
            old = max(old, tx.amount)
            new = round(old - tx.amount, 2)
        tx.old_orig, tx.new_orig = old, new
        balance[tx.orig] = new
        if tx.dest.startswith("C") and not tx.fraud:
            d_old = balance.get(tx.dest, round(float(rng.uniform(100, 3000)), 2))
            tx.old_dest, tx.new_dest = d_old, round(d_old + tx.amount, 2)
            balance[tx.dest] = tx.new_dest
        # merchants and mules: PaySim leaves destination balances at 0


def write_csv(txs: Sequence[Tx], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for tx in txs:
            flagged = int(tx.type == "TRANSFER" and tx.amount > FLAG_LIMIT)
            w.writerow([int(tx.t) + 1, tx.type, f"{tx.amount:.2f}", tx.orig, f"{tx.old_orig:.2f}",
                        f"{tx.new_orig:.2f}", tx.dest, f"{tx.old_dest:.2f}", f"{tx.new_dest:.2f}",
                        tx.fraud, flagged])


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate a PaySim-format synthetic dataset.")
    p.add_argument("--out", type=Path, default=Path("data/transactions.csv"))
    p.add_argument("--customers", type=int, default=SimConfig.customers)
    p.add_argument("--days", type=int, default=SimConfig.days)
    p.add_argument("--shift", type=float, default=1.0,
                   help="drift factor: >1 = bigger amounts and more transfers")
    p.add_argument("--seed", type=int, default=SimConfig.seed)
    args = p.parse_args(argv)
    if args.customers < 10 or args.days < 2 or args.shift <= 0:
        p.error("need --customers >= 10, --days >= 2 and --shift > 0")
    cfg = SimConfig(customers=args.customers, merchants=max(5, args.customers // 10),
                    days=args.days, shift=args.shift, seed=args.seed)
    txs = simulate(cfg)
    write_csv(txs, args.out)
    n_fraud = sum(tx.fraud for tx in txs)
    print(f"wrote {len(txs):,} transactions ({n_fraud:,} fraud, {n_fraud / len(txs):.3%}) "
          f"to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
