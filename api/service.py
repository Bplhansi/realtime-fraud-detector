"""The scoring service: engine score -> decision -> reasons (flagged only) -> SQLite log."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import joblib

from api.engine import load_engine
from api.schemas import TransactionIn
from api.store import DecisionStore
from ml.data import account_id
from ml.explain import Explainer


@dataclass(frozen=True)
class Settings:
    models_dir: Path = Path("models")
    db_path: Path = Path("data/decisions.db")
    warm_file: Path | None = None  # past transactions (.tx.bin) to load into history at start
    engine: str = "auto"  # auto | cpp | python

    @classmethod
    def from_env(cls, env: dict[str, str]) -> Settings:
        warm = env.get("FD_WARM_FILE")
        return cls(models_dir=Path(env.get("FD_MODELS_DIR", "models")),
                   db_path=Path(env.get("FD_DB_PATH", "data/decisions.db")),
                   warm_file=Path(warm) if warm else None,
                   engine=env.get("FD_ENGINE", "auto"))


class ScoringService:
    def __init__(self, settings: Settings) -> None:
        m = settings.models_dir
        self.engine = load_engine(m / "fraud_model.onnx", m / "thresholds.ini", settings.engine)
        self.explainer = Explainer(joblib.load(m / "fraud_model.joblib"))
        meta = json.loads((m / "model_meta.json").read_text(encoding="utf-8"))
        if meta["model_version"] != self.engine.model_version:
            raise RuntimeError("model_meta.json and thresholds.ini are from different models")
        self.store = DecisionStore(settings.db_path)
        self.warmed = self.engine.remember_file(settings.warm_file) if settings.warm_file else 0
        # The engine's account history must be updated one transaction at a time.
        self._lock = threading.Lock()

    def score(self, tx: TransactionIn, tx_id: int | None = None,
              is_fraud: int | None = None, save: bool = True) -> dict:
        orig, dest = account_id(tx.nameOrig), account_id(tx.nameDest)
        with self._lock:
            r = self.engine.score(tx.step, tx.type, tx.amount, orig, dest)
        reasons = []
        if r["decision"] != "approve":
            reasons = [x.text for x in self.explainer.reasons(r["features"])]
        row = {"tx_id": tx_id, "step": tx.step, "type": tx.type, "amount": tx.amount,
               "name_orig": tx.nameOrig, "name_dest": tx.nameDest, "score": r["score"],
               "decision": r["decision"], "reasons": reasons,
               "model_version": self.engine.model_version, "engine": self.engine.kind,
               "latency_us": r["latency_ns"] / 1000.0, "is_fraud": is_fraud}
        if save:
            row["decision_id"] = self.store.add_decision(row)
        return row
