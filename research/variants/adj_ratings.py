"""
Opponent-adjusted team ratings (decayed ridge / Massey style) -> P(home win).

Walking through history one game-day at a time, every team gets ratings
from a ridge regression on all EARLIER games only (same-day games never see
each other), with exponentially time-decayed weights:

  margin model   MARGIN = home + r_home - r_away              (capped margin)
  per-100 model  100*PTS/POSS = mu +/- home/2 + O_team - D_opp (two rows/game)
  pace model     POSS = mu + p_home + p_away

Ratings are shrunk toward a prior: at each new season the prior becomes the
previous season's final rating times SHRINK (and old data is kept at weight
CARRY).  The normal equations are updated incrementally, so a refit is one
small linear solve per game-day.

fit_predict fits a logistic regression of RESULT on the adjusted rating gap
(+ Elo and rest features) on the train frame and applies it to the test frame.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.benchmark import CACHE_DIR, baseline_dataset

# Rating-system hyperparameters (tuned on the dev window 2014-15..2016-17)
PARAMS = {
    "halflife": 60.0,   # days; in-season time decay of game weights
    "lam":      4.0,    # ridge strength toward the prior (in "games")
    "carry":    0.0,    # weight kept on last season's data at a new season
    "shrink":   0.6,    # prior = SHRINK * last season's final rating
    "cap":      25.0,   # margins clipped to +/- CAP points
}

# Features the logistic regression uses
MODEL_FEATS = ["ADJ_DIFF", "ELO_DIFF", "H_B2B", "A_B2B"]
C_REG = 1.0

_V5_KEEP = ["GAME_ID", "ELO_DIFF", "REST_DIFF", "H_B2B", "A_B2B"]


# ── games ─────────────────────────────────────────────────────────────────────

def game_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per game: teams, points, possessions, margin (date-sorted)."""
    df = raw.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df = df[df["GAME_ID"].str[:3].isin(("002", "004", "005"))]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False)
    df["POSS"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    rest = df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days
    df["B2B"] = (rest == 1).astype(int)
    cols = ["GAME_ID", "GAME_DATE", "SEASON_YEAR", "TEAM_ID", "PTS", "POSS", "B2B",
            "FG3M", "FG3A", "FTM", "FTA"]
    h = df[df["IS_HOME"]][cols]
    a = df[~df["IS_HOME"]][cols[:1] + cols[3:]]
    g = h.merge(a, on="GAME_ID", suffixes=("_H", "_A"))
    g = g.rename(columns={"SEASON_YEAR": "SEASON", "TEAM_ID_H": "H", "TEAM_ID_A": "A"})
    g["POSS"] = (g["POSS_H"] + g["POSS_A"]) / 2
    g["MARGIN"] = g["PTS_H"] - g["PTS_A"]
    g["RESULT"] = (g["MARGIN"] > 0).astype(int)
    return g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def luck_adjusted_margin(g: pd.DataFrame, luck3: float, luckft: float,
                         halflife: float = 60.0) -> np.ndarray:
    """Home margin with a share of 3P / FT shooting luck removed: makes above
    the league rate (decayed league average up to that date) are discounted.
    Only used as a regression TARGET, so it only affects later games."""
    def league_rate(made, att):
        day = g.groupby("GAME_DATE")[[made + "_H", made + "_A", att + "_H", att + "_A"]].sum()
        m = day[made + "_H"] + day[made + "_A"]
        a = day[att + "_H"] + day[att + "_A"]
        t = (day.index - day.index[0]).days.values
        w = 0.5 ** (-t / halflife)          # growing weights == decay of the past
        rate = np.cumsum(m.values * w) / np.cumsum(a.values * w)
        return g["GAME_DATE"].map(pd.Series(rate, index=day.index)).values

    p3 = league_rate("FG3M", "FG3A")
    pft = league_rate("FTM", "FTA")
    l3 = (g["FG3M_H"] - g["FG3A_H"] * p3) - (g["FG3M_A"] - g["FG3A_A"] * p3)
    lft = (g["FTM_H"] - g["FTA_H"] * pft) - (g["FTM_A"] - g["FTA_A"] * pft)
    return (g["MARGIN"] - luck3 * 3 * l3 - luckft * lft).values


# ── decayed ridge tracker ─────────────────────────────────────────────────────

class DecayedRidge:
    """Weighted ridge regression kept as decayed normal equations.
    Coefficients in `pen` are shrunk toward `prior` with strength lam."""

    def __init__(self, n: int, pen: np.ndarray, lam: float):
        self.A = np.zeros((n, n))
        self.b = np.zeros(n)
        self.pen = pen.astype(float)
        self.lam = lam
        self.prior = np.zeros(n)
        self.beta = np.zeros(n)

    def decay(self, f: float):
        self.A *= f
        self.b *= f

    def add(self, idx, coef, y: float, w: float = 1.0):
        x = np.zeros(len(self.b))
        x[idx] = coef
        self.A += w * np.outer(x, x)
        self.b += w * y * x

    def solve(self) -> np.ndarray:
        P = self.lam * self.pen
        self.beta = np.linalg.solve(self.A + np.diag(P + 1e-6),
                                    self.b + P * self.prior)
        return self.beta


def rate_history(games: pd.DataFrame, halflife: float, lam: float,
                 carry: float, shrink: float, cap: float,
                 team_hca: float = 0.0, b2b: bool = False,
                 luck3: float = 0.0, luckft: float = 0.0) -> pd.DataFrame:
    """Pre-game adjusted ratings for every game (refit once per game-day).
    team_hca > 0 adds per-team home edges shrunk to 0 with that strength;
    b2b adds a back-to-back term to the margin regression."""
    teams = sorted(set(games["H"]) | set(games["A"]))
    tix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    H = games["H"].map(tix).values
    A = games["A"].map(tix).values
    B2B = (games["B2B_H"] - games["B2B_A"]).values

    # margin: [r_0..r_n-1, home, b2b, hca_0..hca_n-1]
    mpen = np.r_[np.ones(n), 0.0, 0.0, np.full(n, team_hca / max(lam, 1e-9))]
    mm = DecayedRidge(2 * n + 2, mpen, lam)
    # per-100 offense/defense: [O_0.., D_0.., mu, home]
    opd = np.r_[np.ones(2 * n), 0.0, 0.0]
    od = DecayedRidge(2 * n + 2, opd, lam)
    # pace: [p_0.., mu]
    pc = DecayedRidge(n + 1, np.r_[np.ones(n), 0.0], lam)
    od.prior[2 * n] = 105.0
    pc.prior[n] = 95.0

    margin = np.clip(luck_adjusted_margin(games, luck3, luckft), -cap, cap)
    off_h = 100 * games["PTS_H"].values / games["POSS"].values
    off_a = 100 * games["PTS_A"].values / games["POSS"].values
    poss = games["POSS"].values
    dates = games["GAME_DATE"].values
    seasons = games["SEASON"].values

    out = np.zeros((len(games), 8))
    n_played = np.zeros(n)
    last_date, last_season = None, None
    starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(games)]
    for s, e in zip(starts[:-1], starts[1:]):
        d, season = dates[s], seasons[s]
        if last_season is not None and season != last_season:
            _new_season(mm, n, carry, shrink, team_slice=slice(0, n))
            _new_season(od, n, carry, shrink, team_slice=slice(0, 2 * n))
            _new_season(pc, n, carry, shrink, team_slice=slice(0, n))
            n_played[:] = 0
        elif last_date is not None:
            f = 0.5 ** ((d - last_date) / np.timedelta64(1, "D") / halflife)
            for m in (mm, od, pc):
                m.decay(f)
        bm, bo, bp = mm.solve(), od.solve(), pc.solve()
        for k in range(s, e):
            h, a = H[k], A[k]
            net_h = (bo[h] + bo[n + h]) - (bo[a] + bo[n + a]) + bo[2 * n + 1]
            pace = bp[n] + bp[h] + bp[a]
            out[k] = [bm[h] - bm[a], bm[n] + bm[n + 2 + h], bm[n + 1] * B2B[k],
                      net_h, pace, net_h * pace / 100, n_played[h], n_played[a]]
        for k in range(s, e):
            h, a = H[k], A[k]
            idx, coef = [h, a, n], [1, -1, 1]
            if b2b:
                idx, coef = idx + [n + 1], coef + [B2B[k]]
            if team_hca > 0:
                idx, coef = idx + [n + 2 + h], coef + [1]
            mm.add(idx, coef, margin[k])
            od.add([h, n + a, 2 * n, 2 * n + 1], [1, -1, 1, 0.5], off_h[k])
            od.add([a, n + h, 2 * n, 2 * n + 1], [1, -1, 1, -0.5], off_a[k])
            pc.add([h, a, n], [1, 1, 1], poss[k])
            n_played[h] += 1
            n_played[a] += 1
        last_date, last_season = d, season

    cols = ["ADJ_DIFF", "ADJ_HOME", "ADJ_B2B", "NET100_PRED", "PACE_PRED",
            "NET_PTS_PRED", "NG_H", "NG_A"]
    res = pd.DataFrame(out, columns=cols)
    res.insert(0, "GAME_ID", games["GAME_ID"].values)
    return res


def _new_season(m: DecayedRidge, n: int, carry: float, shrink: float, team_slice):
    """Season boundary: prior <- shrink * final ratings, old data kept at carry."""
    m.prior = m.beta.copy()
    m.prior[team_slice] *= shrink
    m.decay(carry)


# ── benchmark interface ───────────────────────────────────────────────────────

def build_dataset(raw: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    params = {**PARAMS, **(params or {})}
    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    path = os.path.join(CACHE_DIR, f"adj_ratings_{key}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"GAME_ID": str})
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df
    games = game_table(raw)
    feats = rate_history(games, **params)
    v5 = baseline_dataset(raw)[_V5_KEEP]
    df = (games[["GAME_ID", "GAME_DATE", "SEASON", "RESULT", "MARGIN"]]
          .merge(feats, on="GAME_ID").merge(v5, on="GAME_ID"))
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _X(df: pd.DataFrame, feats=None) -> np.ndarray:
    X = df[feats or MODEL_FEATS].astype(float).copy()
    if "ELO_DIFF" in X:
        X["ELO_DIFF"] = X["ELO_DIFF"] / 25.0     # ~ points scale
    return X.values


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, feats=None,
                C: float = C_REG) -> np.ndarray:
    lr = LogisticRegression(C=C, max_iter=1000)
    lr.fit(_X(train, feats), train["RESULT"].values)
    return lr.predict_proba(_X(test, feats))[:, 1]
