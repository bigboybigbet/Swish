"""GET /api/ratings/teams, /api/ratings/players and /api/accuracy."""
import os
from typing import Literal

from fastapi import APIRouter

from server import clock
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


def on_startup(app):
    """`python -m server` sets SWISH_AUTO_REFRESH: keep the ratings current by
    themselves (tests and demo mode never touch the network)."""
    if os.environ.get("SWISH_AUTO_REFRESH") == "1" and not clock.is_demo():
        ratings.start_auto_refresh()
