"""GET /api/teams (M0). M1 adds /api/teams/{team_id}/form and /availability here."""
from fastapi import APIRouter

from server.services import teams as team_service

router = APIRouter(prefix="/api", tags=["teams"])


@router.get("/teams")
def list_teams():
    return [team_service.public(t) for t in team_service.all_teams()]
