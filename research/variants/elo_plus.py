"""
elo_plus — Elo upgraded to a Kalman-filter rating on the point scale.

Changes vs the production Elo (K=20, HFA=70, no offseason regression):
  • ratings are in points and track the margin directly, with a per-team
    rating variance, so K is adaptive (big after the offseason or a long
    layoff, small once a team's rating has settled);
  • 20% regression toward the league mean at each new season, plus extra
    rating variance for roster churn;
  • a 1-point fatigue penalty for the 2nd night of a back-to-back inside the
    expected margin, so tired losses do not drag a team's rating down;
  • expected margin → P(home win) by a logistic fit on the train seasons
    (learns slope and the current home edge) instead of the fixed 400-pt curve.

All hyperparameters were chosen on the dev window (2014-15 … 2016-17).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.variants.elo_plus_engine import PARAMS, game_table, replay

FEATURES = ["EXP_MARGIN"]


def build_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per game with the pre-game Kalman-Elo state."""
    games = game_table(raw)
    return pd.concat([games, replay(games)], axis=1)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(train[FEATURES].values, train["RESULT"].values)
    return model.predict_proba(test[FEATURES].values)[:, 1]


# ── live use ──────────────────────────────────────────────────────────────────

def expected_margin(state: dict, home, away, game_date, season: str,
                    **params) -> float:
    """Pre-game expected home margin for an upcoming game, from the end state
    of replay() over the leaguegamefinder logs available before tip-off."""
    p = {**PARAMS, **params}
    new_season = season != state["season"]
    day = (pd.Timestamp(game_date).normalize() - pd.Timestamp("2000-01-01")).days
    r = {t: state["R"].get(t, 0.0) * (p["carry"] if new_season else 1.0)
         for t in (home, away)}
    tired = {t: (not new_season and day - state["last_day"].get(t, -9) == 1)
             for t in (home, away)}
    return (r[home] - r[away] + p["hfa"]
            + p["b2b"] * (float(tired[away]) - float(tired[home])))
