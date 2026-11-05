"""FastAPI scoring service.

    POST /score                      one transaction -> score, decision, top reasons
    GET  /decisions?pending=true     flagged transactions for the review queue
    POST /decisions/{id}/review      a reviewer's verdict (fraud / false_alarm)
    GET  /health

Run:  python -m uvicorn api.app:app --port 8000      (docs at http://127.0.0.1:8000/docs)
Settings (environment): FD_MODELS_DIR, FD_DB_PATH, FD_WARM_FILE, FD_ENGINE (auto|cpp|python).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from api.schemas import ReviewIn, ReviewOut, ScoreOut, TransactionIn
from api.service import ScoringService, Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env(dict(os.environ))
    state: dict = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["service"] = ScoringService(settings)
        yield
        state.clear()

    app = FastAPI(title="Real-time fraud detector", version="1.0", lifespan=lifespan)

    def service() -> ScoringService:
        return state["service"]

    @app.get("/health")
    def health() -> dict:
        s = service()
        return {"status": "ok", "engine": s.engine.kind, "model_version": s.engine.model_version,
                "accounts_in_memory": s.engine.accounts, "warmed_transactions": s.warmed}

    # Account history lives in this process's memory: run uvicorn with one worker.
    @app.post("/score", response_model=ScoreOut)
    def score(tx: TransactionIn) -> dict:
        return service().score(tx)

    @app.get("/decisions")
    def decisions(pending: bool = False, limit: int = Query(100, ge=1, le=1000)) -> list[dict]:
        return service().store.flagged(pending_only=pending, limit=limit)

    @app.post("/decisions/{decision_id}/review", response_model=ReviewOut)
    def review(decision_id: int, body: ReviewIn) -> dict:
        try:
            rid = service().store.add_review(decision_id, body.label, body.reviewer)
        except KeyError:
            raise HTTPException(status_code=404, detail="no such decision") from None
        return {"review_id": rid, "decision_id": decision_id, "label": body.label}

    return app


app = create_app()
