"""Request and response shapes for the scoring API, checked by Pydantic."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TxTypeName = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
ACCOUNT_PATTERN = r"^[CM][0-9]{1,18}$"  # PaySim style: C = customer, M = merchant


class TransactionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1, le=1_000_000, description="time step; 1 step = 1 hour")
    type: TxTypeName
    amount: float = Field(ge=0, le=1e10)
    nameOrig: str = Field(pattern=ACCOUNT_PATTERN, description="sender, e.g. C1231006815")
    nameDest: str = Field(pattern=ACCOUNT_PATTERN, description="receiver, e.g. M1979787155")

    @field_validator("amount")
    @classmethod
    def finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("amount must be a finite number")
        return v


class ScoreOut(BaseModel):
    decision_id: int
    score: float
    decision: Literal["approve", "review", "block"]
    reasons: list[str]
    model_version: str
    engine: Literal["cpp", "python"]
    latency_us: float


class ReviewIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["fraud", "false_alarm"]
    reviewer: str = Field(default="reviewer", min_length=1, max_length=64)


class ReviewOut(BaseModel):
    review_id: int
    decision_id: int
    label: str
