"""
Rating engine for the elo_plus variant: a Kalman-filter Elo on the point
scale.

Each team carries a rating (points better than an average team) and a
rating variance.  A game's margin is modelled as

    margin ~ N(R_home − R_away + HFA + fatigue, σ²)

and both ratings move toward the observed margin by their Kalman gain, so a
team whose rating is uncertain (start of season, long layoff) moves faster
than one with a settled rating.  At each new season ratings are pulled
toward the league average and their variance is widened.

One chronological pass over every game since 2010-11.  Each row records the
PRE-game state; the game's own result is only used after it is recorded.
"""
import numpy as np
import pandas as pd

VALID_PREFIXES = ("002", "004", "005")   # regular season, playoffs, play-in

PARAMS = dict(
    sigma=13.0,      # sd of a game margin around its expectation (points)
    q=0.04,          # rating variance added per day between a team's games
    v0=30.0,         # rating variance of every team at its first game
    v_season=5.0,    # extra variance added at each new season (roster churn)
    v_max=60.0,      # cap on a team's rating variance
    carry=0.8,       # share of the rating kept at a new season
    hfa=2.5,         # home advantage used inside the update (points)
    b2b=1.0,         # points lost on the 2nd night of a back-to-back
)


def game_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Team game logs (leaguegamefinder format) → one row per game, home vs
    away, in date order."""
    df = raw.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df = df[df["GAME_ID"].str[:3].isin(VALID_PREFIXES)]
    df = df.dropna(subset=["WL", "PLUS_MINUS"])
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    is_home = df["MATCHUP"].str.contains("vs.", regex=False)
    home = df[is_home][["GAME_ID", "GAME_DATE", "SEASON_YEAR", "TEAM_ID",
                        "WL", "PLUS_MINUS"]]
    away = df[~is_home][["GAME_ID", "TEAM_ID"]]
    g = home.merge(away, on="GAME_ID", suffixes=("_H", "_A"))
    g = g.rename(columns={"SEASON_YEAR": "SEASON"})
    g["RESULT"] = (g["WL"] == "W").astype(int)
    g["MARGIN"] = g["PLUS_MINUS"].astype(float)
    g["PLAYOFF"] = (g["GAME_ID"].str[:3] == "004").astype(int)
    g = g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return g[["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID_H", "TEAM_ID_A",
              "RESULT", "MARGIN", "PLAYOFF"]]


def replay(games: pd.DataFrame, **params) -> pd.DataFrame:
    """Pre-game ratings for every row of game_table().

    Returns a frame aligned with `games` holding R_H, R_A (ratings), V_H,
    V_A (variances), B2B_H, B2B_A and EXP_MARGIN, the pre-game expected home
    margin.  The end state (ratings, variances, last game day per team) is
    attached as .attrs["state"] for live use.
    """
    p = {**PARAMS, **params}
    s2, q, vmax, carry, hfa, b2b = (p["sigma"] ** 2, p["q"], p["v_max"],
                                    p["carry"], p["hfa"], p["b2b"])
    teams = sorted(set(games["TEAM_ID_H"]) | set(games["TEAM_ID_A"]))
    R = {t: 0.0 for t in teams}
    V = {t: p["v0"] for t in teams}
    last = {}
    day = (games["GAME_DATE"] - pd.Timestamp("2000-01-01")).dt.days.tolist()

    n = len(games)
    cols = {k: np.zeros(n) for k in ("R_H", "R_A", "V_H", "V_A", "B2B_H", "B2B_A")}
    season = None
    rows = zip(games["SEASON"], games["TEAM_ID_H"], games["TEAM_ID_A"],
               games["MARGIN"], day)
    for i, (s, h, a, m, d) in enumerate(rows):
        if s != season:
            if season is not None:            # offseason: regress + widen
                R = {t: carry * r for t, r in R.items()}
                V = {t: min(v + p["v_season"], vmax) for t, v in V.items()}
            season, last = s, {}
        tired = []
        for t in (h, a):
            gap = d - last[t] if t in last else None
            tired.append(1.0 if gap == 1 else 0.0)
            if gap is not None:
                V[t] = min(V[t] + q * gap, vmax)
            last[t] = d
        rh, ra, vh, va = R[h], R[a], V[h], V[a]
        cols["R_H"][i], cols["R_A"][i] = rh, ra
        cols["V_H"][i], cols["V_A"][i] = vh, va
        cols["B2B_H"][i], cols["B2B_A"][i] = tired

        # ── update with the result (after the pre-game state is recorded) ──
        S = vh + va + s2
        err = m - (rh - ra + hfa + b2b * (tired[1] - tired[0]))
        R[h] = rh + vh / S * err
        R[a] = ra - va / S * err
        V[h] = vh - vh * vh / S
        V[a] = va - va * va / S

    out = pd.DataFrame(cols, index=games.index)
    out["EXP_MARGIN"] = (out["R_H"] - out["R_A"] + hfa
                         + b2b * (out["B2B_A"] - out["B2B_H"]))
    out.attrs["state"] = {"R": R, "V": V, "last_day": last, "season": season}
    return out
