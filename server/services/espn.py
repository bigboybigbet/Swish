"""
ESPN feeds with caching and an offline fallback.

- scoreboard(d): schedule, finals and market odds for one ET date, from one
  cached scoreboard call (5 min for today and later, 24 h for past dates).
  When ESPN is unreachable an expired copy of any age is served with
  stale=True; with nothing cached it raises upstream_unavailable.
- injury_feed(): the league-wide injury feed, cached 10 min; stale copies are
  used for up to 24 h, after that the status is "unavailable".

The module-level _fetch_* functions and _cache are the data sources; demo
mode (server/services/demo.py) swaps them, live code paths never change.
"""
import logging
import time
from datetime import date, datetime, timezone

from data import cache, injuries, schedule
from server import clock
from server.errors import ApiError
from server.security import redact
from server.services import teams

log = logging.getLogger("server")

SCOREBOARD_TTL_LIVE = 5 * 60
SCOREBOARD_TTL_PAST = 24 * 3600
INJURY_TTL          = 10 * 60
INJURY_MAX_STALE    = 24 * 3600

# Data sources (replaced by demo.install()).
_fetch_scoreboard  = schedule.fetch_scoreboard
_fetch_injury_feed = injuries.fetch_injury_feed
_cache             = cache


def iso_ts(ts) -> str | None:
    """Unix timestamp → ISO-8601 UTC (None stays None)."""
    if ts is None:
        return None
    return clock.iso_utc(datetime.fromtimestamp(ts, timezone.utc))


# ── scoreboard ───────────────────────────────────────────────────────────────

def _tipoff(raw) -> str | None:
    """ESPN's "2024-03-15T23:00Z" → "2024-03-15T23:00:00Z"."""
    if not raw:
        return None
    try:
        return clock.iso_utc(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
    except ValueError:
        return None


def _with_ids(ev: dict) -> dict:
    """Attach nba_api team ids (and our abbreviations) and tipoff_utc to a parsed event."""
    out = dict(ev)
    out["tipoff_utc"] = _tipoff(ev.get("date"))
    for side in ("home", "away"):
        team = teams.resolve(ev[side]["name"]) or teams.resolve(ev[side].get("abbr") or "")
        out[side] = {**ev[side], "id": team["id"] if team else None,
                     "abbr": team["abbr"] if team else ev[side].get("abbr")}
    return out


def scoreboard(d: date) -> dict:
    """{date, as_of, stale, events:[…]} for an ET date (see data.schedule.parse_scoreboard)."""
    key = f"web:espn_scoreboard:{d:%Y%m%d}"
    hit = _cache.get_with_age(key)
    stale = False
    if hit is None:
        try:
            events = schedule.parse_scoreboard(_fetch_scoreboard(d))
        except Exception as e:
            hit = _cache.get_with_age(key, allow_stale=True)
            if hit is None:
                raise ApiError(
                    "upstream_unavailable",
                    f"ESPN scoreboard for {d.isoformat()} is unavailable and nothing is cached "
                    f"({redact(type(e).__name__)}).",
                    hint="Check the internet connection and retry.", source="espn") from None
            log.warning("ESPN scoreboard offline, serving cached copy for %s", d)
            stale = True
        else:
            ttl = SCOREBOARD_TTL_PAST if d < clock.today_et() else SCOREBOARD_TTL_LIVE
            _cache.set(key, events, ttl)
            hit = (events, time.time())
    events, stored_at = hit
    return {
        "date": d.isoformat(),
        "as_of": iso_ts(stored_at),
        "stale": stale,
        "events": [_with_ids(ev) for ev in events],
    }


def find_event(board: dict, home_id: int, away_id: int) -> dict | None:
    """The scoreboard event for a matchup (home and away as given), or None."""
    for ev in board["events"]:
        if ev["home"]["id"] == home_id and ev["away"]["id"] == away_id:
            return ev
    return None


# ── injuries ─────────────────────────────────────────────────────────────────

def injury_feed() -> dict:
    """{feed, status: "ok"|"stale"|"unavailable", fetched_at}."""
    key = "web:espn_injuries"
    hit = _cache.get_with_age(key)
    if hit is not None:
        return {"feed": hit[0], "status": "ok", "fetched_at": iso_ts(hit[1])}
    try:
        feed = _fetch_injury_feed()
    except Exception as e:
        log.warning("ESPN injury feed unavailable: %s", redact(type(e).__name__))
        hit = _cache.get_with_age(key, allow_stale=True)
        if hit is not None and hit[1] and time.time() - hit[1] <= INJURY_MAX_STALE:
            return {"feed": hit[0], "status": "stale", "fetched_at": iso_ts(hit[1])}
        return {"feed": None, "status": "unavailable", "fetched_at": None}
    _cache.set(key, feed, INJURY_TTL)
    return {"feed": feed, "status": "ok", "fetched_at": clock.iso_utc()}


def team_injuries(team: dict, feed: dict | None) -> list:
    """[{name, status, detail}] for a team from an injury feed.

    Feed entries are matched to the team by id (any known name), not by the
    substring match of data.injuries, so "LA Clippers" and "Los Angeles
    Clippers" both work and never match the Lakers.
    """
    out = []
    for entry in (feed or {}).get("injuries", []):
        name = entry.get("displayName", "")
        found = teams.resolve(name)
        if found and found["id"] == team["id"]:
            out += injuries.parse_injury_report({"injuries": [entry]}, name)
    return out
