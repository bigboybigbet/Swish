"""POST /api/predict and GET /api/odds."""
from datetime import date as Date
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from server import clock
from server.services import market, teams
from server.services import predict as predict_service

router = APIRouter(prefix="/api", tags=["predict"])

InjuryStatus = Literal["out", "doubtful", "questionable", "day-to-day"]


class Prices(BaseModel):
    home_ml: int
    away_ml: int

    @field_validator("home_ml", "away_ml")
    @classmethod
    def american(cls, v: int) -> int:
        if -100 < v < 100 or abs(v) > 100_000:
            raise ValueError("American odds are ≤ −100 or ≥ +100")
        return v


class InjuredPlayer(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    status: InjuryStatus
    detail: str = Field("", max_length=200)


class CustomInjuries(BaseModel):
    home: list[InjuredPlayer] = Field(default_factory=list, max_length=30)
    away: list[InjuredPlayer] = Field(default_factory=list, max_length=30)


class PredictRequest(BaseModel):
    home_id: int
    away_id: int
    date: Date | None = None
    odds: Prices | None = None
    injuries: Literal["live", "none"] | CustomInjuries = "live"
    include: list[Literal["hist", "shap", "poisson"]] = Field(default_factory=lambda: ["hist"])


@router.post("/predict")
def predict(req: PredictRequest):
    injuries = req.injuries if isinstance(req.injuries, str) else req.injuries.model_dump()
    return predict_service.predict_core(
        req.home_id, req.away_id, req.date,
        odds=req.odds.model_dump() if req.odds else None,
        injuries=injuries, include=req.include)


@router.get("/odds")
def odds(home_id: int, away_id: int, date: Date | None = None):
    """Market moneyline (+ spread/total) for a matchup: The Odds API if a key is set, else ESPN."""
    home, away = teams.require(home_id), teams.require(away_id)
    return market.public_odds(market.lookup(home, away, date or clock.today_et()))
