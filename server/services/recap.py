"""
Game recap from ESPN's summary feed (one call per game): score flow with the
win probability after every play, quarter scores, team stats, the rotation
(who was on the floor when, rebuilt from substitutions), player impact (box
score, Game Score and win probability added) and key moments (biggest win
probability swings, scoring runs, lead changes, clutch baskets).

Definitions follow common usage (sportsdataverse game_flow): a run is
unanswered points by one team; a lead change is the lead passing from one
team to the other.
"""
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from data import cache as _cache
from server import clock
from server.errors import ApiError
from server.services import teams

log = logging.getLogger(__name__)

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
TTL_FINAL = 24 * 3600
TTL_LIVE = 30
RUN_MIN = 8               # unanswered points that count as a run
CLUTCH_SECONDS = 300      # last 5 minutes of the 4th quarter or overtime …
CLUTCH_MARGIN = 5         # … with the score within 5


def _fetch_summary(event_id: str) -> dict:
    r = requests.get(SUMMARY_URL, params={"event": event_id}, timeout=15)
    r.raise_for_status()
    return r.json()


def summary(event_id: str) -> dict:
    """Cached ESPN summary: a day for finished games, 30 s otherwise."""
    if not str(event_id).isdigit():
        raise ApiError("validation", "event_id must be an ESPN event number", status=422)
    if clock.is_demo():
        raise ApiError("upstream_unavailable", "Game recaps need ESPN's live feed, so they are not "
                       "available in demo mode.", status=503, source="espn")
    key = f"web:espn_summary:{event_id}"
    hit = _cache.get(key)
    if hit is not None:
        return hit
    try:
        data = _fetch_summary(str(event_id))
    except Exception as e:                                   # noqa: BLE001
        raise ApiError("upstream_unavailable", f"ESPN summary for event {event_id} is unavailable "
                       f"({type(e).__name__}).", status=503, source="espn") from None
    done = (data.get("header", {}).get("competitions") or [{}])[0] \
        .get("status", {}).get("type", {}).get("completed")
    _cache.set(key, data, TTL_FINAL if done else TTL_LIVE)
    return data


# ── helpers ───────────────────────────────────────────────────────────────────

def _clock_seconds(display: str) -> float:
    """"7:19" → 439, "45.3" → 45.3."""
    try:
        if ":" in display:
            m, s = display.split(":")
            return int(m) * 60 + float(s)
        return float(display)
    except (ValueError, AttributeError):
        return 0.0


def _elapsed(period: int, clock_display: str) -> float:
    """Game seconds elapsed at a play: 12-minute quarters, 5-minute overtimes."""
    left = _clock_seconds(clock_display)
    if period <= 4:
        return (period - 1) * 720 + (720 - left)
    return 2880 + (period - 5) * 300 + (300 - left)


def _period_end(period: int) -> float:
    return period * 720 if period <= 4 else 2880 + (period - 4) * 300


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── parse ─────────────────────────────────────────────────────────────────────

def _sides(data: dict) -> dict:
    comp = data["header"]["competitions"][0]
    out = {}
    for c in comp["competitors"]:
        t = c["team"]
        nba = teams.resolve(t.get("displayName", "")) or teams.resolve(t.get("abbreviation", "")) or {}
        out[c["homeAway"]] = {
            "espn_id": str(t["id"]), "id": nba.get("id"), "abbr": nba.get("abbr") or t.get("abbreviation"),
            "name": t.get("displayName"), "score": int(c.get("score") or 0),
            "color": "#" + t["color"] if t.get("color") else None,
            "linescores": [int(_num(l.get("displayValue")) or 0) for l in c.get("linescores", [])],
            "winner": bool(c.get("winner")),
        }
    return out


def _box(data: dict, sides: dict) -> tuple:
    """Per-player box rows and per-team stats, keyed by home/away."""
    side_of = {v["espn_id"]: k for k, v in sides.items()}
    players, team_stats = {"home": [], "away": []}, {}
    for t in data.get("boxscore", {}).get("players", []):
        side = side_of.get(str(t["team"]["id"]))
        if not side:
            continue
        st = t["statistics"][0]
        keys = st.get("keys", [])
        for a in st.get("athletes", []):
            raw = dict(zip(keys, a.get("stats") or []))
            row = {"id": a["athlete"]["id"], "name": a["athlete"].get("displayName"),
                   "short": a["athlete"].get("shortName"), "starter": bool(a.get("starter")),
                   "dnp": bool(a.get("didNotPlay")) or not a.get("stats"), "reason": a.get("reason")}
            if not row["dnp"]:
                fgm, fga = (raw.get("fieldGoalsMade-fieldGoalsAttempted") or "0-0").split("-")
                ftm, fta = (raw.get("freeThrowsMade-freeThrowsAttempted") or "0-0").split("-")
                g = {k: _num(raw.get(k)) or 0 for k in ("points", "rebounds", "assists", "turnovers", "steals",
                                                        "blocks", "offensiveRebounds", "defensiveRebounds",
                                                        "fouls", "plusMinus")}
                fgm, fga, ftm, fta = (int(x or 0) for x in (fgm, fga, ftm, fta))
                game_score = (g["points"] + 0.4 * fgm - 0.7 * fga - 0.4 * (fta - ftm)
                              + 0.7 * g["offensiveRebounds"] + 0.3 * g["defensiveRebounds"] + g["steals"]
                              + 0.7 * g["assists"] + 0.7 * g["blocks"] - 0.4 * g["fouls"] - g["turnovers"])
                row.update({
                    "min": _num(raw.get("minutes")) or 0, "pts": int(g["points"]), "reb": int(g["rebounds"]),
                    "ast": int(g["assists"]), "stl": int(g["steals"]), "blk": int(g["blocks"]),
                    "to": int(g["turnovers"]), "pf": int(g["fouls"]), "pm": int(g["plusMinus"]),
                    "fg": raw.get("fieldGoalsMade-fieldGoalsAttempted"),
                    "fg3": raw.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"),
                    "ft": raw.get("freeThrowsMade-freeThrowsAttempted"),
                    "game_score": round(game_score, 1),
                })
            players[side].append(row)
    for t in data.get("boxscore", {}).get("teams", []):
        side = side_of.get(str(t["team"]["id"]))
        if side:
            team_stats[side] = {s["name"]: s.get("displayValue") for s in t.get("statistics", [])}
    return players, team_stats


def _flow(data: dict, sides: dict) -> list:
    """One point per play: game time, scores, margin (home − away), home win prob."""
    wp = {w["playId"]: w.get("homeWinPercentage") for w in data.get("winprobability", [])}
    side_of = {v["espn_id"]: k for k, v in sides.items()}
    out = []
    for p in data.get("plays", []):
        period = (p.get("period") or {}).get("number") or 1
        clk = (p.get("clock") or {}).get("displayValue") or "12:00"
        h, a = p.get("homeScore") or 0, p.get("awayScore") or 0
        out.append({
            "id": p.get("id"), "t": round(_elapsed(period, clk), 1), "period": period, "clock": clk,
            "home": h, "away": a, "margin": h - a, "wp": wp.get(p.get("id")),
            "text": p.get("text") or (p.get("type") or {}).get("text"),
            "type": (p.get("type") or {}).get("text"),
            "side": side_of.get(str((p.get("team") or {}).get("id"))),
            "scoring": bool(p.get("scoringPlay")), "value": p.get("scoreValue") or 0,
            "actors": [x["athlete"]["id"] for x in p.get("participants", []) if x.get("athlete")],
        })
    # fill win probability gaps forward so the line has no holes
    last = None
    for f in out:
        if f["wp"] is None:
            f["wp"] = last
        last = f["wp"]
    return out


def _rotation(flow: list, players: dict) -> dict:
    """On-floor stints per player: [[start_s, end_s], …] from the starters
    and every substitution (ESPN logs between-period changes at 12:00)."""
    on = {s: {p["id"] for p in players[s] if p["starter"]} for s in ("home", "away")}
    side_of_player = {p["id"]: s for s in ("home", "away") for p in players[s]}
    start = {pid: 0.0 for s in on for pid in on[s]}
    stints = {pid: [] for s in players for pid in (p["id"] for p in players[s])}
    end_t = max((f["t"] for f in flow), default=2880.0)
    end_t = max(end_t, _period_end(max((f["period"] for f in flow), default=4)))
    for f in flow:
        if f["type"] != "Substitution" or len(f["actors"]) < 2:
            continue
        enters, leaves = f["actors"][0], f["actors"][1]
        side = side_of_player.get(enters) or side_of_player.get(leaves)
        if not side:
            continue
        if leaves in start:
            stints[leaves].append([start.pop(leaves), f["t"]])
        on[side].discard(leaves)
        if enters not in start:
            start[enters] = f["t"]
        on[side].add(enters)
    for pid, t0 in start.items():
        stints.setdefault(pid, []).append([t0, end_t])
    return {pid: [[round(a, 1), round(b, 1)] for a, b in s if b > a] for pid, s in stints.items()}


def _moments(flow: list, sides: dict, names: dict) -> dict:
    """Key moments plus game-flow totals (lead changes, ties, largest leads, runs)."""
    def label(f):
        return {"t": f["t"], "period": f["period"], "clock": f["clock"], "home": f["home"],
                "away": f["away"], "wp": f["wp"], "text": f["text"], "side": f["side"]}

    swings, prev_wp = [], None
    lead_changes = ties = 0
    leader = 0
    largest = {"home": 0, "away": 0}
    runs, run_side, run_pts, run_start, run_from = [], None, 0, None, None
    clutch = []
    prev = {"home": 0, "away": 0}
    for f in flow:
        if f["wp"] is not None and prev_wp is not None and f["type"] != "End Period":
            swings.append((abs(f["wp"] - prev_wp), f["wp"] - prev_wp, f))
        prev_wp = f["wp"] if f["wp"] is not None else prev_wp
        if not f["scoring"]:
            continue
        m = f["margin"]
        now = (m > 0) - (m < 0)
        if now and leader and now != leader:
            lead_changes += 1
        if m == 0 and (f["home"] or f["away"]):
            ties += 1
        if now:
            leader = now
        largest["home"] = max(largest["home"], m)
        largest["away"] = max(largest["away"], -m)
        scorer = "home" if f["home"] > prev["home"] else ("away" if f["away"] > prev["away"] else None)
        pts = (f["home"] - prev["home"]) + (f["away"] - prev["away"])
        prev = {"home": f["home"], "away": f["away"]}
        if scorer and scorer == run_side:
            run_pts += pts
        else:
            if run_side and run_pts >= RUN_MIN:
                runs.append({"side": run_side, "points": run_pts, "from": run_from, "to": run_start})
            run_side, run_pts, run_from = scorer, pts, label(f)
        run_start = label(f)
        late = (f["period"] >= 4 and _clock_seconds(f["clock"]) <= CLUTCH_SECONDS)
        if late and abs(m) <= CLUTCH_MARGIN + 3:
            clutch.append(label(f))
    if run_side and run_pts >= RUN_MIN:
        runs.append({"side": run_side, "points": run_pts, "from": run_from, "to": run_start})

    swings.sort(key=lambda s: -s[0])
    top = []
    for size, delta, f in swings[:6]:
        actor = names.get(f["actors"][0]) if f["actors"] else None
        top.append({**label(f), "delta": round(delta, 4), "actor": actor})
    top.sort(key=lambda x: x["t"])
    return {"swings": top, "runs": runs, "clutch": clutch[-8:],
            "lead_changes": lead_changes, "times_tied": ties,
            "largest_lead": largest}


def _wpa(flow: list, players: dict) -> dict:
    """Win probability added: each play's change in win chance credited to its
    first participant, from that player's team's side."""
    side_of_player = {p["id"]: s for s in ("home", "away") for p in players[s]}
    total, prev = {}, None
    for f in flow:
        if f["wp"] is None:
            continue
        if prev is not None and f["actors"] and f["type"] not in ("Substitution", "End Period"):
            pid = f["actors"][0]
            side = side_of_player.get(pid)
            if side:
                d = f["wp"] - prev
                total[pid] = total.get(pid, 0.0) + (d if side == "home" else -d)
        prev = f["wp"]
    return total


def recap(event_id: str) -> dict:
    """GET /api/recap/{event_id}."""
    data = summary(event_id)
    sides = _sides(data)
    players, team_stats = _box(data, sides)
    flow = _flow(data, sides)
    names = {p["id"]: p["short"] or p["name"] for s in players for p in players[s]}
    rotation = _rotation(flow, players)
    wpa = _wpa(flow, players)
    for s in players:
        for p in players[s]:
            p["wpa"] = round(wpa.get(p["id"], 0.0), 4)
            p["stints"] = rotation.get(p["id"], [])
    comp = data["header"]["competitions"][0]
    status = comp.get("status", {}).get("type", {})
    game_date = _et_date(comp.get("date"))
    return {
        "event_id": str(event_id),
        "date": game_date,
        "status": {"state": status.get("state"), "completed": bool(status.get("completed")),
                   "detail": status.get("detail")},
        "home": sides.get("home"), "away": sides.get("away"),
        "periods": max((f["period"] for f in flow), default=4),
        "flow": [{k: f[k] for k in ("t", "period", "clock", "home", "away", "margin", "wp", "text", "side", "scoring")}
                 for f in flow if f["type"] != "Substitution"],
        "subs": [{"t": f["t"], "period": f["period"], "clock": f["clock"], "text": f["text"], "side": f["side"]}
                 for f in flow if f["type"] == "Substitution"],
        "moments": _moments(flow, sides, names),
        "players": players,
        "team_stats": team_stats,
        "pregame": _pregame(sides, game_date),
        "injuries": _injuries(data),
        "news": [_article(a) for a in (data.get("news") or {}).get("articles", [])[:6]],
    }


def _et_date(iso: str | None) -> str | None:
    """ESPN's UTC tip-off → the US Eastern game date."""
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    except ValueError:
        return iso[:10]


def _pregame(sides: dict, game_date: str) -> dict | None:
    """What the Swish model said before tip-off (ratings as of that date)."""
    from engine import team_strength
    st = team_strength.get_state()
    h, a = sides.get("home") or {}, sides.get("away") or {}
    if not st or not h.get("id") or not a.get("id") or not game_date:
        return None
    try:
        p = team_strength.predict(st, str(h["id"]), str(a["id"]), game_date)
    except KeyError:
        return None
    picked_home = p["p_home"] >= 0.5
    done = h.get("score") is not None and (h.get("winner") or a.get("winner"))
    return {"p_home": round(p["p_home"], 4), "exp_margin": round(p["exp_margin"], 1),
            "pick": "home" if picked_home else "away",
            "correct": (picked_home == bool(h.get("winner"))) if done else None,
            "actual_margin": (h.get("score") or 0) - (a.get("score") or 0)}


def _injuries(data: dict) -> list:
    out = []
    for t in data.get("injuries", []):
        for i in t.get("injuries", []):
            out.append({"team": (t.get("team") or {}).get("abbreviation"),
                        "player": ((i.get("athlete") or {}).get("displayName")),
                        "status": i.get("status"), "detail": (i.get("details") or {}).get("detail")
                        or i.get("shortComment") or i.get("longComment")})
    return out


def _article(a: dict) -> dict:
    links = a.get("links") or {}
    return {"headline": a.get("headline"), "description": a.get("description"),
            "published": a.get("published"), "type": a.get("type"),
            "url": ((links.get("web") or {}).get("href")),
            "image": ((a.get("images") or [{}])[0] or {}).get("url")}
