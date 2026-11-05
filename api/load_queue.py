"""Fill the review queue: score the test period through the scoring service into SQLite.

History before the test period (train + validation) is loaded into the engine first, so
every test transaction is scored with the account history it would have had live.
Also writes data/history.tx.bin, which the API loads at start-up (FD_WARM_FILE).

Usage:
    python -m api.load_queue --csv data/transactions.csv --split data/split.json \
        --db data/decisions.db
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from api.schemas import TransactionIn
from api.service import ScoringService, Settings
from ml.data import load_transactions, write_tx_file

BATCH = 5000


def load_queue(csv: Path, split: Path, db: Path, models: Path, engine: str = "auto",
               max_rows: int | None = None) -> dict:
    df = load_transactions(csv)
    val_end = json.loads(split.read_text(encoding="utf-8"))["validation_end_step"]
    history = csv.parent / "history.tx.bin"
    write_tx_file(df[df["step"] <= val_end], history)
    if db.exists():
        db.unlink()  # a fresh queue each run
    service = ScoringService(Settings(models_dir=models, db_path=db, warm_file=history,
                                      engine=engine))
    test = df[df["step"] > val_end]
    if max_rows is not None:
        test = test.head(max_rows)
    t0 = time.perf_counter()
    rows: list[dict] = []
    counts = {"approve": 0, "review": 0, "block": 0}
    for r in test.itertuples(index=False):
        tx = TransactionIn(step=int(r.step), type=r.type, amount=float(r.amount),
                           nameOrig=r.nameOrig, nameDest=r.nameDest)
        row = service.score(tx, tx_id=int(r.tx_id), is_fraud=int(r.isFraud), save=False)
        counts[row["decision"]] += 1
        rows.append(row)
        if len(rows) == BATCH:
            service.store.add_decisions(rows)
            rows = []
    service.store.add_decisions(rows)
    return {"engine": service.engine.kind, "history": service.warmed, "scored": len(test),
            "seconds": round(time.perf_counter() - t0, 1), **counts}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score the test period into the review queue.")
    p.add_argument("--csv", type=Path, default=Path("data/transactions.csv"))
    p.add_argument("--split", type=Path, default=Path("data/split.json"))
    p.add_argument("--db", type=Path, default=Path("data/decisions.db"))
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--engine", choices=["auto", "cpp", "python"], default="auto")
    p.add_argument("--max-rows", type=int, help="score only the first N test transactions")
    args = p.parse_args(argv)
    s = load_queue(args.csv, args.split, args.db, args.models, args.engine, args.max_rows)
    print(f"{s['engine']} engine, {s['history']:,} past transactions loaded; scored "
          f"{s['scored']:,} in {s['seconds']} s: {s['approve']:,} approve, {s['review']:,} "
          f"review, {s['block']:,} block -> {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
