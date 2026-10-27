"""GET /api/teams, /api/teams/{team_id}/form and /api/teams/{team_id}/availability."""
from datetime import date as Date

from fastapi import APIRouter

from server import clock
from server.services import predict as predict_service
from server.services import teams as team_service

router = APIRouter(prefix="/api", tags=["teams"])


@router.get("/teams")
def list_teams():
    return [team_service.public(t) for t in team_service.all_teams()]


@router.get("/teams/{team_id}/form")
def team_form(team_id: int, date: Date | None = None):
    """Rolling form before an ET date (default today): get_team_advanced_stats, snake-cased."""
    team = team_service.require(team_id)
    return predict_service.form_response(team, date or clock.today_et())


@router.get("/teams/{team_id}/availability")
def team_availability(team_id: int, date: Date | None = None):
    """Injury report and strength (healthy vs current) for a game date (default today)."""
    team = team_service.require(team_id)
    return predict_service.availability_response(team, date or clock.today_et())
