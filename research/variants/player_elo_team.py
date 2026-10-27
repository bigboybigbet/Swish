"""
player_elo_team — the like-for-like baseline for player_elo: the same
Kalman engine with every team as a single "player" (no box score, no
plus-minus, no lineups), tuned with the same coordinate-descent budget on
the dev window 2014-15 … 2016-17.  The gap to player_elo is what player-level
ratings (lineups, injuries, trades) add over a team rating.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.variants.player_elo import game_table
from research.variants.player_elo_engine import replay, team_prepare

# tuned on dev (teams exist from the first season, so p0 / r0 do not apply)
TEAM_PARAMS = dict(q=0.003, v_season=0.2, p_max=2.4, carry=0.8, cap=40.0,
                   sigma_pm=0.0, gamma=0.0)
FEATURES = ["PE_DIFF"]


def build_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    prep = team_prepare(game_table(raw))
    out, _ = replay(prep, **TEAM_PARAMS)
    return pd.concat([prep["games"], out], axis=1)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(train[FEATURES].values, train["RESULT"].values)
    return model.predict_proba(test[FEATURES].values)[:, 1]
