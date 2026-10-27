"""GET /api/ratings/teams, /api/ratings/players and /api/accuracy."""
from typing import Literal

from fastapi import APIRouter

from server.services import ratings

router = APIRouter(prefix="/api", tags=["ratings"])


@router.get("/ratings/teams")
def team_ratings():
    return ratings.team_table()


@router.get("/ratings/players")
def player_ratings():
    return ratings.player_table()


@router.get("/accuracy")
def accuracy(window: Literal["season", "30d", "all"] = "season"):
    return ratings.accuracy(window)
