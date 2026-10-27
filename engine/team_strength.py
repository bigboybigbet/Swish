"""
Team strength — Kalman-filter Elo on the point scale (model v6, option A).

Each team carries a rating R (points better than an average team) and a
rating variance V.  A game's margin is modelled as

    margin ~ N(R_home − R_away + HFA + fatigue, σ²)

and both ratings move toward the observed margin by their Kalman gain
V / (V_home + V_away + σ²), so an uncertain rating (new season, long layoff)
moves fast and a settled one slowly.  At each new season ratings keep CARRY
of their value and their variance widens.  The expected margin maps to a
win probability with a 2-parameter logistic fitted on the training seasons.

Walk-forward holdout 2017-18 … 2023-24 (8,874 games, research/benchmark.py):
log loss 0.6283, accuracy 64.5%, vs 0.6521 / 62.6% for the v5 XGBoost.

The state is rebuilt by replaying every game log from scratch, so no game
can ever be applied twice.
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import (TEAM_STRENGTH_FILE, current_season, recent_seasons,
                    season_string)

VALID_PREFIXES = ("002", "004", "005")   # regular season, playoffs, play-in
EPOCH = pd.Timestamp("2000-01-01")

PARAMS = dict(
    sigma=13.0,      # sd of a game margin around its expectation (points)
    q=0.04,          # rating variance added per day between a team's games
    v0=30.0,         # rating variance of a team at its first game
    v_season=5.0,    # extra variance at each new season (roster churn)
    v_max=60.0,      # cap on a team's rating variance
    carry=0.8,       # share of the rating kept at a new season
    hfa=2.5,         # home advantage inside the update (points)
    b2b=1.0,         # points lost on the 2nd night of a back-to-back
)
# Seasons replayed to build the state: one more than the logistic is fitted
# on, so the first fitted season starts from warmed-up ratings.
REPLAY_SEASONS = 4


def us_eastern_today() -> pd.Timestamp:
    """NBA game dates are US Eastern; the user's clock may be UTC+7."""
    return pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)


# ── replay ────────────────────────────────────────────────────────────────────

def _season_of(df: pd.DataFrame) -> pd.Series:
    if "SEASON_YEAR" in df.columns:
        return df["SEASON_YEAR"].astype(str)
    # nba_api leaguegamefinder: SEASON_ID like "22026" / "42026"
    return df["SEASON_ID"].astype(str).str[-4:].astype(int).map(season_string)


def game_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Team game logs (leaguegamefinder format) → one row per finished game,
    home vs away, in date order."""
    df = raw.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df = df[df["GAME_ID"].str[:3].isin(VALID_PREFIXES)]
    df = df.dropna(subset=["WL", "PLUS_MINUS"])
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    df["SEASON"] = _season_of(df)
    is_home = df["MATCHUP"].str.contains("vs.", regex=False)
    home = df[is_home][["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID",
                        "TEAM_NAME", "WL", "PLUS_MINUS"]]
    away = df[~is_home][["GAME_ID", "TEAM_ID", "TEAM_NAME"]]
    g = home.merge(away, on="GAME_ID", suffixes=("_H", "_A"))
    g["RESULT"] = (g["WL"] == "W").astype(int)
    g["MARGIN"] = g["PLUS_MINUS"].astype(float)
    g = g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return g[["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID_H", "TEAM_ID_A",
              "TEAM_NAME_H", "TEAM_NAME_A", "RESULT", "MARGIN"]]


def replay(games: pd.DataFrame, **params) -> pd.DataFrame:
    """Pre-game ratings for every row of game_table().

    Returns R_H, R_A, V_H, V_A, B2B_H, B2B_A and EXP_MARGIN aligned with
    `games`; the end state is in .attrs["state"].
    """
    p = {**PARAMS, **params}
    s2, q, vmax, hfa, b2b = p["sigma"] ** 2, p["q"], p["v_max"], p["hfa"], p["b2b"]
    R, V, last, season = {}, {}, {}, None
    hist = {}                      # team → [[day, R_after, V_after, season], ...]
    day = ((games["GAME_DATE"] - EPOCH).dt.days).tolist()

    n = len(games)
    cols = {k: np.zeros(n) for k in ("R_H", "R_A", "V_H", "V_A", "B2B_H", "B2B_A")}
    rows = zip(games["SEASON"], games["TEAM_ID_H"], games["TEAM_ID_A"],
               games["MARGIN"], day)
    for i, (s, h, a, m, d) in enumerate(rows):
        if s != season:
            if season is not None:            # offseason: regress + widen
                R = {t: p["carry"] * r for t, r in R.items()}
                V = {t: min(v + p["v_season"], vmax) for t, v in V.items()}
            season, last = s, {}
        tired = []
        for t in (h, a):
            R.setdefault(t, 0.0)
            V.setdefault(t, p["v0"])
            gap = d - last[t] if t in last else None
            tired.append(1.0 if gap == 1 else 0.0)
            if gap is not None:
                V[t] = min(V[t] + q * gap, vmax)
            last[t] = d
        rh, ra, vh, va = R[h], R[a], V[h], V[a]
        cols["R_H"][i], cols["R_A"][i] = rh, ra
        cols["V_H"][i], cols["V_A"][i] = vh, va
        cols["B2B_H"][i], cols["B2B_A"][i] = tired

        # update with the result, after the pre-game state is recorded
        S = vh + va + s2
        err = m - (rh - ra + hfa + b2b * (tired[1] - tired[0]))
        R[h] = rh + vh / S * err
        R[a] = ra - va / S * err
        V[h] = vh - vh * vh / S
        V[a] = va - va * va / S
        for t in (h, a):
            hist.setdefault(t, []).append([d, round(R[t], 4), round(V[t], 4), s])

    out = pd.DataFrame(cols, index=games.index)
    out["EXP_MARGIN"] = (out["R_H"] - out["R_A"] + hfa
                         + b2b * (out["B2B_A"] - out["B2B_H"]))
    out.attrs["state"] = {"R": R, "V": V, "last_day": last, "season": season,
                          "history": hist}
    return out


def fit_probability(exp_margin: np.ndarray, result: np.ndarray) -> tuple:
    """Logistic P(home win) = 1 / (1 + exp(−(a + b·margin))) by Newton's
    method; two parameters, so no regularisation is needed."""
    x = np.column_stack([np.ones_like(exp_margin), exp_margin])
    w = np.array([0.0, 0.1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-x @ w))
        grad = x.T @ (result - p)
        hess = (x * (p * (1 - p))[:, None]).T @ x
        step = np.linalg.solve(hess, grad)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


# ── state ─────────────────────────────────────────────────────────────────────

def build_state(raw: pd.DataFrame, fit_seasons=None) -> dict:
    """Replay every game in `raw` and fit the probability map on
    `fit_seasons` (default: all but the first season replayed)."""
    games = game_table(raw)
    if games.empty:
        raise ValueError("no finished games in the logs")
    pre = replay(games)
    seasons = sorted(games["SEASON"].unique())
    fit_seasons = fit_seasons or seasons[1:] or seasons
    mask = games["SEASON"].isin(fit_seasons).values
    a, b = fit_probability(pre["EXP_MARGIN"].values[mask],
                           games["RESULT"].values[mask].astype(float))
    st = pre.attrs["state"]
    names = {}
    for side in ("H", "A"):
        names.update(dict(zip(games[f"TEAM_ID_{side}"], games[f"TEAM_NAME_{side}"])))
    return {
        "params": PARAMS,
        "coef": {"a": a, "b": b, "n_fit": int(mask.sum()), "seasons": list(fit_seasons)},
        "season": st["season"],
        "teams": {str(t): {"name": names[t], "rating": round(st["R"][t], 3),
                           "var": round(st["V"][t], 3),
                           "last_day": st["last_day"].get(t),
                           "history": st["history"].get(t, [])}
                  for t in st["R"]},
        "last_game_date": str(games["GAME_DATE"].max().date()),
        "n_games": len(games),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_state(state: dict, path: str = TEAM_STRENGTH_FILE):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, path)


def load_state(path: str = TEAM_STRENGTH_FILE) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def refresh(status_cb=None) -> dict | None:
    """Download the last REPLAY_SEASONS seasons of logs, rebuild and save."""
    from engine.models import fetch_game_logs
    report = status_cb or print
    raw = fetch_game_logs(recent_seasons(REPLAY_SEASONS), report)
    if raw is None:
        return None
    state = build_state(raw)
    save_state(state)
    report(f"Team strength rebuilt from {state['n_games']:,} games "
           f"(last {state['last_game_date']})")
    return state


# ── prediction ────────────────────────────────────────────────────────────────

def find_team(state: dict, team_name: str) -> dict | None:
    """Look a team up by id string or (partial) name, case-insensitive."""
    if team_name in state["teams"]:
        return state["teams"][team_name]
    key = team_name.lower().strip()
    for t in state["teams"].values():
        n = t["name"].lower()
        if key == n or key in n or n in key:
            return t
    nick = key.split()[-1] if key else ""
    for t in state["teams"].values():           # "Los Angeles Clippers" → "LA Clippers"
        if nick and t["name"].lower().split()[-1] == nick:
            return t
    return None


def _as_of(team: dict, day: int, season: str, p: dict) -> tuple:
    """(rating, variance, back_to_back) of a team before a game on `day`,
    from its post-game history — so past dates never see later games."""
    prev = [h for h in team.get("history", []) if h[0] < day]
    if not prev:
        return 0.0, p["v0"], False
    d, r, v, s = prev[-1]
    if s != season:                      # first game of a new season
        return p["carry"] * r, min(v + p["v_season"], p["v_max"]), False
    return r, min(v + p["q"] * (day - d), p["v_max"]), day - d == 1


def predict(state: dict, home: str, away: str, game_date=None,
            home_pts_lost: float = 0.0, away_pts_lost: float = 0.0) -> dict:
    """Pre-game expected home margin and P(home win) for a game on
    `game_date` (default: today, US Eastern), using only games before it.

    `*_pts_lost` are points of strength removed for injured players.
    """
    p = state["params"]
    h, a = find_team(state, str(home)), find_team(state, str(away))
    if h is None or a is None:
        raise KeyError(f"team not found: {home if h is None else away}")
    date = pd.Timestamp(game_date).normalize() if game_date is not None else us_eastern_today()
    day = (date - EPOCH).days
    season = current_season(date.to_pydatetime())

    rh, vh, b2b_h = _as_of(h, day, season, p)
    ra, va, b2b_a = _as_of(a, day, season, p)
    rating_term = rh - ra
    b2b_term = p["b2b"] * (float(b2b_a) - float(b2b_h))
    injury_term = away_pts_lost - home_pts_lost
    margin = rating_term + p["hfa"] + b2b_term + injury_term
    c = state["coef"]
    prob = 1 / (1 + np.exp(-(c["a"] + c["b"] * margin)))
    return {
        "p_home": float(prob), "exp_margin": float(margin),
        "rating_home": rh, "rating_away": ra,
        "sd_home": float(np.sqrt(vh)), "sd_away": float(np.sqrt(va)),
        "b2b_home": bool(b2b_h), "b2b_away": bool(b2b_a),
        "pts_lost_home": home_pts_lost, "pts_lost_away": away_pts_lost,
        # logit decomposition: a + b·(rating + home + b2b + injuries)
        "terms": {"rating": rating_term, "home": p["hfa"], "b2b": b2b_term,
                  "injuries": injury_term},
        "coef": dict(c),
    }


_cache = {"mtime": None, "state": None}


def get_state(path: str = TEAM_STRENGTH_FILE) -> dict | None:
    """The saved state, re-read only when the file changes."""
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    if _cache["mtime"] != mtime:
        _cache["state"], _cache["mtime"] = load_state(path), mtime
    return _cache["state"]


def ratings_table(state: dict) -> pd.DataFrame:
    """Teams ranked by rating, with a ±1 sd band."""
    rows = [{"TEAM_ID": int(k), "TEAM": t["name"], "RATING": t["rating"],
             "SD": float(np.sqrt(t["var"]))}
            for k, t in state["teams"].items()]
    df = pd.DataFrame(rows).sort_values("RATING", ascending=False).reset_index(drop=True)
    df.index += 1
    return df
