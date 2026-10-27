"""
Opponent-adjusted team ratings (decayed ridge / Massey style) -> P(home win).

Walking through history one game-day at a time, every team gets ratings
from ridge regressions on all EARLIER games only (same-day games never see
each other), with exponentially time-decayed game weights:

  margin model   MARGIN = home + r_home - r_away              (margin capped)
  per-100 model  100*PTS/POSS = mu +/- home/2 + O_team - D_opp (two rows/game)
  pace model     POSS = mu + p_home + p_away

Ratings are shrunk toward a prior: at each new season the prior becomes the
previous season's final rating times SHRINK, and old data is kept at weight
CARRY.  The normal equations are updated incrementally, so a refit is one
small linear solve per game-day.

The home edge comes from a second, long-memory pass of the margin model
(HOME_HALFLIFE days, no season reset), so it adapts to league-wide drift.

fit_predict: logistic regression of RESULT on ADJ_PRED = r_home - r_away +
home edge, plus back-to-back flags, fitted on the train frame.
The per-100 / pace ratings are exported for spreads and totals but are not
used by the classifier (no gain on the dev window).
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.benchmark import CACHE_DIR

# Tuned on the dev window (2014-15..2016-17); the dev optimum is a flat plateau
# (halflife 60-90, lam 8-12, shrink 0.6-0.8, carry 0.2-0.35, cap >= 30).
PARAMS = {
    "halflife": 75.0,   # days; in-season decay of game weights
    "lam":      10.0,   # ridge strength toward the prior (in game-equivalents)
    "carry":    0.3,    # weight kept on last season's games at a new season
    "shrink":   0.7,    # prior = SHRINK * last season's final rating
    "cap":      40.0,   # margins clipped to +/- CAP points
}
HOME_HALFLIFE = 400.0   # days; memory of the home-edge estimate
MODEL_FEATS = ["ADJ_PRED", "H_B2B", "A_B2B"]
VERSION = 2             # bump to invalidate research/cache files


# ── games ─────────────────────────────────────────────────────────────────────

def game_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per game (date-sorted): teams, points, possessions, margin,
    back-to-back flags (team played the previous day)."""
    df = raw.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df = df[df["GAME_ID"].str[:3].isin(("002", "004", "005"))]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False)
    df["POSS"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    df["B2B"] = (df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days == 1).astype(int)

    cols = ["GAME_ID", "GAME_DATE", "SEASON_YEAR", "TEAM_ID", "PTS", "POSS", "B2B"]
    h = df[df["IS_HOME"]][cols]
    a = df[~df["IS_HOME"]][cols[:1] + cols[3:]]
    g = h.merge(a, on="GAME_ID", suffixes=("_H", "_A"))
    g = g.rename(columns={"SEASON_YEAR": "SEASON", "TEAM_ID_H": "H", "TEAM_ID_A": "A",
                          "B2B_H": "H_B2B", "B2B_A": "A_B2B"})
    g["POSS"] = (g["POSS_H"] + g["POSS_A"]) / 2
    g["MARGIN"] = g["PTS_H"] - g["PTS_A"]
    g["RESULT"] = (g["MARGIN"] > 0).astype(int)
    return g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


# ── decayed ridge ─────────────────────────────────────────────────────────────

class DecayedRidge:
    """Weighted ridge regression kept as decayed normal equations.
    Coefficients flagged in `pen` are shrunk toward `prior` with strength lam."""

    def __init__(self, n: int, pen: np.ndarray, lam: float):
        self.A = np.zeros((n, n))
        self.b = np.zeros(n)
        self.pen = lam * pen.astype(float)
        self.prior = np.zeros(n)
        self.beta = np.zeros(n)

    def decay(self, f: float):
        self.A *= f
        self.b *= f

    def add(self, idx, coef, y: float):
        x = np.zeros(len(self.b))
        x[idx] = coef
        self.A += np.outer(x, x)
        self.b += y * x

    def solve(self) -> np.ndarray:
        self.beta = np.linalg.solve(self.A + np.diag(self.pen + 1e-6),
                                    self.b + self.pen * self.prior)
        return self.beta

    def new_season(self, teams: slice, carry: float, shrink: float):
        """Prior <- shrink * current team ratings; old data kept at `carry`."""
        self.prior = self.beta.copy()
        self.prior[teams] *= shrink
        self.decay(carry)


def rate_history(games: pd.DataFrame, halflife: float, lam: float,
                 carry: float, shrink: float, cap: float) -> pd.DataFrame:
    """Pre-game adjusted ratings for every game in `games` (one refit per day)."""
    teams = sorted(set(games["H"]) | set(games["A"]))
    tix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    H = games["H"].map(tix).values
    A = games["A"].map(tix).values

    mm = DecayedRidge(n + 1, np.r_[np.ones(n), 0], lam)           # [r.., home]
    od = DecayedRidge(2 * n + 2, np.r_[np.ones(2 * n), 0, 0], lam)  # [O.., D.., mu, home]
    pc = DecayedRidge(n + 1, np.r_[np.ones(n), 0], lam)           # [p.., mu]
    od.prior[2 * n], pc.prior[n] = 105.0, 95.0

    margin = games["MARGIN"].clip(-cap, cap).values
    off_h = 100 * games["PTS_H"].values / games["POSS"].values
    off_a = 100 * games["PTS_A"].values / games["POSS"].values
    poss = games["POSS"].values
    dates = games["GAME_DATE"].values
    seasons = games["SEASON"].values

    out = np.zeros((len(games), 7))
    played = np.zeros(n)
    starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(games)]
    for s, e in zip(starts[:-1], starts[1:]):
        if s > 0 and seasons[s] != seasons[s - 1]:
            mm.new_season(slice(0, n), carry, shrink)
            od.new_season(slice(0, 2 * n), carry, shrink)
            pc.new_season(slice(0, n), carry, shrink)
            played[:] = 0
        elif s > 0:
            f = 0.5 ** ((dates[s] - dates[s - 1]) / np.timedelta64(1, "D") / halflife)
            for m in (mm, od, pc):
                m.decay(f)
        bm, bo, bp = mm.solve(), od.solve(), pc.solve()

        for k in range(s, e):                       # predict with earlier days only
            h, a = H[k], A[k]
            net100 = (bo[h] + bo[n + h]) - (bo[a] + bo[n + a]) + bo[2 * n + 1]
            out[k] = [bm[h] - bm[a], bm[n], net100, bp[n] + bp[h] + bp[a],
                      bo[h] - bo[n + a], bo[a] - bo[n + h], min(played[h], played[a])]
        for k in range(s, e):                       # then learn from this day
            h, a = H[k], A[k]
            mm.add([h, a, n], [1, -1, 1], margin[k])
            od.add([h, n + a, 2 * n, 2 * n + 1], [1, -1, 1, 0.5], off_h[k])
            od.add([a, n + h, 2 * n, 2 * n + 1], [1, -1, 1, -0.5], off_a[k])
            pc.add([h, a, n], [1, 1, 1], poss[k])
            played[h] += 1
            played[a] += 1

    cols = ["ADJ_DIFF", "ADJ_HOME", "NET100_PRED", "PACE_PRED",
            "OFF100_H", "OFF100_A", "MIN_GAMES"]
    res = pd.DataFrame(out, columns=cols)
    res.insert(0, "GAME_ID", games["GAME_ID"].values)
    return res


# ── benchmark interface ───────────────────────────────────────────────────────

def build_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    key = json.dumps([VERSION, PARAMS, HOME_HALFLIFE], sort_keys=True)
    path = os.path.join(CACHE_DIR, f"adj_ratings_{hashlib.md5(key.encode()).hexdigest()[:10]}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"GAME_ID": str})
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df

    games = game_table(raw)
    feats = rate_history(games, **PARAMS)
    home = rate_history(games, **{**PARAMS, "halflife": HOME_HALFLIFE,
                                  "carry": 1.0, "shrink": 1.0})
    feats["HOME_EDGE"] = home["ADJ_HOME"].values
    feats["ADJ_PRED"] = feats["ADJ_DIFF"] + feats["HOME_EDGE"]
    df = games[["GAME_ID", "GAME_DATE", "SEASON", "RESULT", "MARGIN",
                "H_B2B", "A_B2B"]].merge(feats, on="GAME_ID")
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    lr = LogisticRegression(max_iter=1000)
    lr.fit(train[MODEL_FEATS].values, train["RESULT"].values)
    return lr.predict_proba(test[MODEL_FEATS].values)[:, 1]
