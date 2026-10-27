"""
Ratings and Accuracy pages, both read from the Kalman-Elo state
(engine.team_strength) — no network, so they work offline and in demo mode.

Accuracy grades every game the state replayed with the prediction made from
the ratings *before* that game.  The logistic map (two numbers) was fitted on
the same seasons, a negligible in-sample effect next to the ratings, which
never see the game they predict.
"""
from datetime import date, timedelta

import numpy as np

from engine import team_strength
from server.errors import ApiError
from server.services import teams

EPOCH = date(2000, 1, 1)
TIERS = ((0.65, "high"), (0.55, "medium"), (0.0, "low"))


def _state() -> dict:
    st = team_strength.get_state()
    if not st:
        raise ApiError("no_model", "No team ratings yet: run Train on the Model page.", status=412)
    return st


def _day(n: int) -> str:
    return (EPOCH + timedelta(days=int(n))).isoformat()


# ── ratings ───────────────────────────────────────────────────────────────────

def team_table(points: int = 40) -> dict:
    """GET /api/ratings/teams: every team ranked, with ±sd, 7-day change,
    season record and the rating path for a sparkline."""
    st = _state()
    last = date.fromisoformat(st["last_game_date"])
    week_ago = (last - timedelta(days=7) - EPOCH).days
    record = {}
    for d, h, a, _e, won, _m in st.get("games", []):
        if team_strength.current_season(date.fromisoformat(d)) != st["season"]:
            continue
        for t, w in ((h, won), (a, 1 - won)):
            rec = record.setdefault(t, [0, 0])
            rec[0 if w else 1] += 1
    rows = []
    for tid, t in st["teams"].items():
        hist = t.get("history", [])
        season_hist = [h for h in hist if h[3] == st["season"]]
        before = [h for h in hist if h[0] <= week_ago]
        prev = before[-1][1] if before else None
        info = teams.get(int(tid)) or {}
        w, l = record.get(int(tid), [0, 0])
        rows.append({
            "team_id": int(tid), "name": t["name"], "abbr": info.get("abbr"),
            "rating": round(t["rating"], 2), "sd": round(float(np.sqrt(t["var"])), 2),
            "change_7d": round(t["rating"] - prev, 2) if prev is not None else None,
            "wins": w, "losses": l,
            "path": [[_day(h[0]), round(h[1], 2)] for h in season_hist[-points:]],
        })
    rows.sort(key=lambda r: r["rating"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {"season": st["season"], "last_game_date": st["last_game_date"],
            "built_at": st.get("built_at"), "teams": rows}


def player_table(limit: int = 150) -> dict:
    """GET /api/ratings/players: the per-player impact the injury adjustment uses."""
    from data.injuries import load_player_impact, load_player_impact_meta
    impact = load_player_impact()
    rows = sorted(({"name": n, "impact": round(float(v), 4)} for n, v in impact.items()),
                  key=lambda r: r["impact"], reverse=True)[:limit]
    return {**load_player_impact_meta(), "players": rows}


# ── accuracy ──────────────────────────────────────────────────────────────────

def _window(games: list, st: dict, window: str) -> list:
    if window == "season":
        return [g for g in games if team_strength.current_season(date.fromisoformat(g[0])) == st["season"]]
    if window == "30d":
        start = (date.fromisoformat(st["last_game_date"]) - timedelta(days=30)).isoformat()
        return [g for g in games if g[0] > start]
    if window == "all":
        fit = set(st["coef"].get("seasons") or [])      # skip the warm-up season
        return [g for g in games if not fit or team_strength.current_season(date.fromisoformat(g[0])) in fit]
    raise ApiError("validation", "window must be season, 30d or all", status=422)


def accuracy(window: str = "season") -> dict:
    """GET /api/accuracy: how the pre-game predictions did."""
    st = _state()
    games = _window(st.get("games", []), st, window)
    if not games:
        return {"window": window, "n": 0, "season": st["season"]}
    c = st["coef"]
    m = np.array([g[3] for g in games])
    y = np.array([g[4] for g in games], dtype=float)
    p = 1 / (1 + np.exp(-(c["a"] + c["b"] * m)))
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    picked_home = p >= 0.5
    correct = picked_home == (y == 1)
    conf = np.maximum(p, 1 - p)

    calib = []
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.sum():
            calib.append({"p": round(float(p[mask].mean()), 4), "actual": round(float(y[mask].mean()), 4),
                          "n": int(mask.sum())})

    tiers = []
    for (lo, name), hi in zip(TIERS, (1.01, 0.65, 0.55)):
        mask = (conf >= lo) & (conf < hi)
        tiers.append({"tier": name, "n": int(mask.sum()),
                      "acc": round(float(correct[mask].mean()), 4) if mask.sum() else None})

    # accuracy by ISO week, and the running total
    weeks, run_n, run_c = {}, 0, 0
    for (d, *_rest), ok in zip(games, correct):
        y_, w, _ = date.fromisoformat(d).isocalendar()
        weeks.setdefault((y_, w, d), []).append(bool(ok))
    trend = []
    by_week = {}
    for (y_, w, d), oks in sorted(weeks.items()):
        by_week.setdefault((y_, w), {"start": d, "oks": []})["oks"].extend(oks)
    for (_y, _w), v in sorted(by_week.items()):
        run_n += len(v["oks"])
        run_c += sum(v["oks"])
        trend.append({"week": v["start"], "n": len(v["oks"]), "acc": round(sum(v["oks"]) / len(v["oks"]), 4),
                      "cum_acc": round(run_c / run_n, 4)})

    return {
        "window": window, "season": st["season"],
        "from": games[0][0], "to": games[-1][0], "n": len(games),
        "acc": round(float(correct.mean()), 4),
        "log_loss": round(float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()), 4),
        "brier": round(float(((p - y) ** 2).mean()), 4),
        "baseline_home": round(float(y.mean()), 4),
        "margin_mae": round(float(np.abs(m - np.array([g[5] for g in games])).mean()), 2),
        "calibration": calib, "tiers": tiers, "trend": trend,
    }
