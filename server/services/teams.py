"""
Team identity: nba_api team IDs everywhere in the API.

web/static/data/teams.json holds colours, the ESPN display name and
abbreviation, and historical/alternate names (aliases). nba_api's static
team list (works offline) adds nickname and city. resolve() maps any
name or abbreviation seen in ESPN feeds, game logs, the Elo file or The
Odds API to a team.
"""
import json
import unicodedata
from functools import lru_cache

from nba_api.stats.static import teams as nba_static

from server import paths
from server.errors import ApiError


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", "").split())


@lru_cache(maxsize=1)
def _load() -> tuple:
    with open(paths.TEAMS_JSON, encoding="utf-8") as f:
        rows = json.load(f)
    static = {t["id"]: t for t in nba_static.get_teams()}
    teams = []
    for r in rows:
        s = static.get(r["id"], {})
        teams.append({**r, "nickname": s.get("nickname"), "city": s.get("city")})
    teams.sort(key=lambda t: t["name"])

    by_id = {t["id"]: t for t in teams}
    by_key = {}
    for t in teams:
        for key in names(t):
            by_key.setdefault(_norm(key), t)
    return tuple(teams), by_id, by_key


def names(team: dict) -> list:
    """Every name/abbreviation a team is known by (first = canonical)."""
    keys = [team["name"], team["espn_name"], *team.get("aliases", []),
            team["abbr"], team["espn_abbr"]]
    if team.get("nickname"):
        keys.append(team["nickname"])
    return list(dict.fromkeys(k for k in keys if k))


def all_teams() -> list:
    return list(_load()[0])


def get(team_id) -> dict | None:
    try:
        return _load()[1].get(int(team_id))
    except (TypeError, ValueError):
        return None


def require(team_id) -> dict:
    """The team for an id, or ApiError team_not_found (404)."""
    team = get(team_id)
    if team is None:
        raise ApiError("team_not_found", f"No NBA team with id {team_id}.",
                       hint="Use an id from /api/teams.")
    return team


def resolve(name: str) -> dict | None:
    """Exact (case/accent-insensitive) match on any known name or abbreviation."""
    if not name:
        return None
    return _load()[2].get(_norm(name))


def lookup(mapping: dict, team: dict, default=None):
    """Value for a team in a dict keyed by team name (e.g. the Elo file), trying all names."""
    for key in names(team):
        if key in mapping:
            return mapping[key]
    return default


def public(team: dict) -> dict:
    """The /api/teams row for a team."""
    return {
        "id": team["id"], "abbr": team["abbr"], "name": team["name"],
        "nickname": team.get("nickname"), "city": team.get("city"),
        "espn_name": team["espn_name"],
        "colors": {"primary": team["primary"], "secondary": team["secondary"]},
    }


def ref(team: dict) -> dict:
    """Short reference used inside other payloads: {id, abbr, name}."""
    return {"id": team["id"], "abbr": team["abbr"], "name": team["name"]}
