"""GET /api/recap/{event_id} and /api/news."""
from typing import Literal

from fastapi import APIRouter

from server.services import news as news_service
from server.services import recap as recap_service

router = APIRouter(prefix="/api", tags=["recap"])


@router.get("/recap/{event_id}")
def game_recap(event_id: str):
    return recap_service.recap(event_id)


@router.get("/news")
def nba_news(team_id: int | None = None,
             only: Literal["availability", "injury", "rest", "suspension", "roster", "streak"] | None = None,
             limit: int = 40):
    return news_service.news(team_id, only, max(1, min(limit, 100)))
