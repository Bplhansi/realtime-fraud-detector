"""SQLite log of every scoring decision and every reviewer verdict.

Reviews are append-only: a changed mind is a new row, never an edit, so the history stays
auditable. Times are UTC.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    tx_id INTEGER,
    step INTEGER NOT NULL,
    type TEXT NOT NULL,
    amount REAL NOT NULL,
    name_orig TEXT NOT NULL,
    name_dest TEXT NOT NULL,
    score REAL NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('approve', 'review', 'block')),
    reasons TEXT NOT NULL,
    model_version TEXT NOT NULL,
    engine TEXT NOT NULL,
    latency_us REAL NOT NULL,
    is_fraud INTEGER
);
CREATE INDEX IF NOT EXISTS idx_decisions_flagged ON decisions (decision, score);
CREATE INDEX IF NOT EXISTS idx_decisions_orig ON decisions (name_orig, step);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER NOT NULL REFERENCES decisions (id),
    label TEXT NOT NULL CHECK (label IN ('fraud', 'false_alarm')),
    reviewer TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

DECISION_COLUMNS = ["created_at", "tx_id", "step", "type", "amount", "name_orig", "name_dest",
                    "score", "decision", "reasons", "model_version", "engine", "latency_us",
                    "is_fraud"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DecisionStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection that commits on success, rolls back on error, and always closes."""
        con = sqlite3.connect(self.path)
        try:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            with con:
                yield con
        finally:
            con.close()

    def add_decisions(self, rows: Iterable[dict]) -> list[int]:
        """Inserts decisions (dicts with DECISION_COLUMNS; reasons as a list). Returns ids."""
        ids = []
        sql = (f"INSERT INTO decisions ({', '.join(DECISION_COLUMNS)}) "
               f"VALUES ({', '.join('?' for _ in DECISION_COLUMNS)})")
        with self._connect() as con:
            for r in rows:
                values = {**r, "created_at": r.get("created_at") or utc_now(),
                          "reasons": json.dumps(list(r.get("reasons", []))),
                          "tx_id": r.get("tx_id"), "is_fraud": r.get("is_fraud")}
                cur = con.execute(sql, [values[c] for c in DECISION_COLUMNS])
                ids.append(int(cur.lastrowid))
        return ids

    def add_decision(self, row: dict) -> int:
        return self.add_decisions([row])[0]

    def add_review(self, decision_id: int, label: str, reviewer: str) -> int:
        if label not in ("fraud", "false_alarm"):
            raise ValueError("label must be 'fraud' or 'false_alarm'")
        with self._connect() as con:
            found = con.execute("SELECT 1 FROM decisions WHERE id = ?", (decision_id,)).fetchone()
            if found is None:
                raise KeyError(f"no decision {decision_id}")
            cur = con.execute(
                "INSERT INTO reviews (decision_id, label, reviewer, created_at) "
                "VALUES (?, ?, ?, ?)",
                (decision_id, label, reviewer, utc_now()))
            return int(cur.lastrowid)

    def get(self, decision_id: int) -> dict | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()
        return _decode(row) if row else None

    def flagged(self, pending_only: bool = False, limit: int = 500) -> list[dict]:
        """Flagged decisions (review or block), highest score first, with the latest verdict."""
        sql = """
            SELECT d.*, (SELECT label FROM reviews r WHERE r.decision_id = d.id
                         ORDER BY r.id DESC LIMIT 1) AS review_label
            FROM decisions d WHERE d.decision != 'approve'
        """
        if pending_only:
            sql += " AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.decision_id = d.id)"
        sql += " ORDER BY d.score DESC, d.id LIMIT ?"
        with self._connect() as con:
            return [_decode(r) for r in con.execute(sql, (limit,)).fetchall()]

    def account_activity(self, name: str, before_step: int | None = None,
                         limit: int = 20) -> list[dict]:
        """Recent decisions where the account sent or received money, newest first."""
        sql = "SELECT * FROM decisions WHERE (name_orig = ? OR name_dest = ?)"
        params: list = [name, name]
        if before_step is not None:
            sql += " AND step <= ?"
            params.append(before_step)
        sql += " ORDER BY step DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as con:
            return [_decode(r) for r in con.execute(sql, params).fetchall()]

    def stats(self) -> dict:
        with self._connect() as con:
            counts = dict(con.execute(
                "SELECT decision, COUNT(*) FROM decisions GROUP BY decision").fetchall())
            reviews = dict(con.execute(
                """SELECT label, COUNT(*) FROM reviews r
                   WHERE id = (SELECT MAX(id) FROM reviews WHERE decision_id = r.decision_id)
                   GROUP BY label""").fetchall())
            latencies = [r[0] for r in con.execute(
                "SELECT latency_us FROM decisions ORDER BY latency_us").fetchall()]
        return {"decisions": {k: counts.get(k, 0) for k in ("approve", "review", "block")},
                "reviews": {k: reviews.get(k, 0) for k in ("fraud", "false_alarm")},
                "latency_us_p50": _pct(latencies, 50), "latency_us_p99": _pct(latencies, 99)}

    def alerts_by_step(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                """SELECT step, SUM(decision = 'review') AS review,
                          SUM(decision = 'block') AS block, COUNT(*) AS total
                   FROM decisions GROUP BY step ORDER BY step""").fetchall()
        return [dict(r) for r in rows]


def _pct(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    rank = max(1, math.ceil(p / 100 * len(sorted_values)))
    return float(sorted_values[rank - 1])


def _decode(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["reasons"] = json.loads(d["reasons"])
    return d
