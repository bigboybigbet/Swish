"""
NBA news from ESPN, tagged for decisions: each article is linked to the
teams and players ESPN tags it with, and classified by keywords (injury,
rest, suspension, roster move, streak). Articles that change who plays are
flagged `availability`, the news most likely to move a prediction.
"""
import re

import requests

from data import cache as _cache
from server import clock
from server.errors import ApiError
from server.services import teams

NEWS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/news"
TTL = 10 * 60

TAGS = {
    "injury": r"injur|ruled out|questionable|doubtful|sidelined|sprain|strain|surgery|fractur|torn|"
              r"concussion|illness|day-to-day|mri|out for|will miss|return(?:s|ing)? from|hamstring|ankle|knee|"
              r"absence|without (?:star|starter)",
    "rest": r"\brest(?:ed|ing)?\b|load management|sit(?:s|ting)? out|day off",
    "suspension": r"suspend|suspension",
    "roster": r"\btrade[sd]?\b|\bsign(?:s|ed|ing)?\b|waive|release[sd]?|call(?:s|ed)? up|two-way|extension",
    "streak": r"winning streak|losing streak|\bskid\b|\bslide\b|straight (?:win|loss|victor)|"
              r"\d+(?:st|nd|rd|th) straight|\bwin(?:s)? \d+ straight",
}
AVAILABILITY = {"injury", "rest", "suspension"}
_RX = {k: re.compile(v, re.I) for k, v in TAGS.items()}


def _fetch(limit: int) -> dict:
    r = requests.get(NEWS_URL, params={"limit": limit}, timeout=15)
    r.raise_for_status()
    return r.json()


def _feed(limit: int = 100) -> dict:
    if clock.is_demo():
        raise ApiError("upstream_unavailable", "News needs ESPN's live feed, so it is not available in "
                       "demo mode.", status=503, source="espn")
    key = f"web:espn_news:{limit}"
    hit = _cache.get(key)
    if hit is not None:
        return hit
    try:
        data = _fetch(limit)
    except Exception as e:                                   # noqa: BLE001
        raise ApiError("upstream_unavailable", f"ESPN news is unavailable ({type(e).__name__}).",
                       status=503, source="espn") from None
    _cache.set(key, data, TTL)
    return data


def classify(text: str) -> list:
    return [k for k, rx in _RX.items() if rx.search(text or "")]


def article(a: dict) -> dict:
    cats = a.get("categories") or []
    tagged_teams, athletes = [], []
    for c in cats:
        if c.get("type") == "team":
            t = teams.resolve((c.get("team") or {}).get("description") or c.get("description") or "")
            if t and t["id"] not in [x["id"] for x in tagged_teams]:
                tagged_teams.append({"id": t["id"], "abbr": t["abbr"], "name": t["name"]})
        elif c.get("type") == "athlete" and c.get("description"):
            athletes.append(c["description"])
    tags = classify(f"{a.get('headline', '')} {a.get('description', '')}")
    links = a.get("links") or {}
    return {
        "id": a.get("id"), "headline": a.get("headline"), "description": a.get("description"),
        "published": a.get("published"), "type": a.get("type"),
        "url": (links.get("web") or {}).get("href"),
        "image": ((a.get("images") or [{}])[0] or {}).get("url"),
        "teams": tagged_teams, "athletes": athletes[:6],
        "tags": tags, "availability": bool(AVAILABILITY & set(tags)),
    }


def news(team_id: int | None = None, only: str | None = None, limit: int = 40) -> dict:
    """GET /api/news: newest first, optionally for one team or one tag."""
    items = [article(a) for a in _feed().get("articles", [])]
    if team_id:
        items = [a for a in items if any(t["id"] == team_id for t in a["teams"])]
    if only == "availability":
        items = [a for a in items if a["availability"]]
    elif only:
        items = [a for a in items if only in a["tags"]]
    items.sort(key=lambda a: a["published"] or "", reverse=True)
    return {"as_of": clock.iso_utc(), "count": len(items), "articles": items[:limit]}
