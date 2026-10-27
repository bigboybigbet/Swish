"""
Slate: the day's schedule and the `scan` job that predicts every game.

The schedule comes straight from the cached ESPN scoreboard, so the page can
draw the cards at once; the scan then runs predict_core per game in a
background thread and streams one `item` event per finished card. Each scan
is saved under runs/scans/<date>/<scan_id>.json and the latest one is served
back when the page is reopened.
"""
import json
import uuid
from datetime import date
from datetime import date as Date

from pydantic import BaseModel, ConfigDict

from data.atomic import atomic_write_json
from server import clock, jobs, jsonutil, paths
from server.errors import ApiError
from server.services import espn
from server.services import predict as predict_service


def _side(ev_side: dict) -> dict:
    return {"id": ev_side.get("id"), "abbr": ev_side.get("abbr"),
            "name": ev_side.get("name"), "score": ev_side.get("score")}


def schedule(d: date) -> dict:
    """GET /api/schedule: every game of an ET date with its market odds."""
    sb = espn.scoreboard(d)
    games = []
    for ev in sb.get("events", []):
        games.append({
            "event_id": ev.get("event_id"),
            "tipoff_utc": ev.get("tipoff_utc"),
            "status": ev.get("status"),
            "completed": bool(ev.get("completed")),
            "period": ev.get("period"),
            "home": _side(ev["home"]),
            "away": _side(ev["away"]),
            "market": ev.get("odds"),
        })
    games.sort(key=lambda g: (g["tipoff_utc"] or "", g["event_id"] or ""))
    return {"date": d.isoformat(), "as_of": sb.get("as_of"), "stale": bool(sb.get("stale")),
            "games": games}


def card(game: dict, r: dict) -> dict:
    """The slice of a /api/predict response a slate card needs."""
    margin, total = r.get("margin") or {}, r.get("total") or {}
    av = r.get("availability") or {}
    out_count = {s: sum(1 for p in (av.get(s) or {}).get("players", []) if p.get("status") == "out")
                 for s in ("home", "away")}
    return {
        "event_id": game["event_id"],
        "home_id": game["home"]["id"], "away_id": game["away"]["id"],
        "mode": r.get("mode"),
        "win_prob": r.get("win_prob"),
        "pick": r.get("pick"),
        "flags": r.get("flags"),
        "home_spread": margin.get("home_spread"),
        "margin_p50": (margin.get("intervals") or {}).get("p50"),
        "total": total.get("median"),
        "total_p50": (total.get("intervals") or {}).get("p50"),
        "scores": r.get("scores"),
        "market": r.get("market"),
        "kelly": r.get("kelly"),
        "fair_line": r.get("fair_line"),
        "ratings": {s: (r.get("ratings") or {}).get(s, {}).get("kalman") for s in ("home", "away")},
        "injured_out": out_count,
        "caveats": r.get("caveats", []),
    }


def _scan_dir(d: date):
    return paths.SCANS_DIR / d.isoformat()


def latest(d: date) -> dict | None:
    folder = _scan_dir(d)
    files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime) if folder.exists() else []
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def load(scan_id: str) -> dict:
    if not scan_id.isalnum():
        raise ApiError("not_found", "No such scan.")
    for f in paths.SCANS_DIR.glob(f"*/{scan_id}.json"):
        return json.loads(f.read_text(encoding="utf-8"))
    raise ApiError("not_found", "No such scan.")


class ScanParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: Date | None = None


def run_scan(ctx: jobs.JobContext):
    d = ctx.params.get("date") or clock.today_et()
    if isinstance(d, str):
        d = date.fromisoformat(d)
    games = schedule(d)["games"]
    n = len(games)
    ctx.progress(done=0, total=n, stage=f"Predicting {n} games" if n else "No games")
    cards = []
    for i, g in enumerate(games):
        ctx.check_cancel()
        try:
            if g["home"]["id"] is None or g["away"]["id"] is None:
                raise ApiError("team_not_found", "Team not recognised.")
            c = card(g, predict_service.predict_core(g["home"]["id"], g["away"]["id"], d,
                                                     include=()))
        except Exception as e:                      # noqa: BLE001  (one bad game must not stop the slate)
            c = {"event_id": g["event_id"], "error": str(e) or type(e).__name__}
        c = jsonutil.clean(c)
        cards.append(c)
        ctx.item(i, c)
        ctx.progress(done=i + 1, total=n,
                     stage=f"{g['away'].get('abbr')} @ {g['home'].get('abbr')}")
    scan_id = uuid.uuid4().hex[:12]
    result = {"scan_id": scan_id, "date": d.isoformat(), "created_at": clock.iso_utc(),
              "games": cards}
    folder = _scan_dir(d)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write_json(folder / f"{scan_id}.json", result, ensure_ascii=False)
    ctx.result_url = f"/api/scans/{scan_id}"
    return result


jobs.register_kind(jobs.JobKind("scan", run_scan, writer=False, params=ScanParams,
                                label="Slate scan"))
