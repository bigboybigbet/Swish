"""
features_v6 — feature engineering on a fixed, simple model.

Model: StandardScaler + L2 logistic regression (C=0.03, tuned on the dev
window), fitted on the 3 train seasons.  The intercept is the home edge.

Features (home minus away), all from information available before tip-off:
  ELO_DIFF    production MOV Elo (data/elo.py), replayed by TEAM_ID
  NRTG_DIFF   season-to-date net rating per 100 possessions, shrunk toward
              0.85 x last season's final net rating with 10 pseudo-games
  AVAIL_DIFF  roster availability tonight vs the roster mix behind the
              season-to-date rating (features_v6_players.py)
  G6_DIFF     games played in the previous 6 days (schedule density)

Dev window (2014-15..2016-17, pooled log loss; raw Elo 0.6036, LR on
ELO_DIFF 0.6028):
  + shrunk season-to-date NRTG          0.6000   (EWMA 5..82 games, raw
    margin, opponent-adjusted, margin-capped versions: no better)
  + G6 (games in last 7 days)           0.5990
  + AVAIL                               0.5922
  Tried and dropped (each made dev log loss worse or < 2e-4 better):
  EWMA form (spans 5/10/20/40), O/D ratings, four factors (EWMA and shrunk),
  win%, pace, 3PA rate, rest-day buckets, B2B, 3-in-4, 4-in-6, road-trip /
  homestand length, travel km (game and 7-day), time-zone shift, body-clock
  offset, altitude, playoff flag and interactions, early-season
  interactions, league home-edge drift, recency-weighted training.

Live (nba_app): ELO_DIFF from elo_ratings.json; NRTG from the current and
previous seasons' leaguegamefinder logs (possessions from FGA, FTA, OREB,
TOV of both teams); G6 from the team's game dates; AVAIL from
leaguedashplayerstats (per-game Game Score, GP with the team) and the ESPN
injury report (expected availability by status) — see features_v6_players.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import ELO_DEFAULT
from data.elo import update_elo_game
from research.benchmark import CACHE_DIR, DEFAULT_DATA_DIR
from research.variants import features_v6_build as fb
from research.variants import features_v6_players as fpl

FEATURES = ["ELO_DIFF", "NRTG_DIFF", "AVAIL_DIFF", "G6_DIFF"]
C = 0.03
RATING = {"regress": 0.85, "k": 10.0}                 # shrunk season-to-date NRTG
AVAIL = {"decay": 0.97, "k0": 5.0, "prior": 0.0, "k": 10.0}
REPLACEMENT = 0.15                                    # Game Score per minute
VERSION = 1


def load_box(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, "features_v6_box.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    box = fpl.load_box(data_dir)
    os.makedirs(CACHE_DIR, exist_ok=True)
    box.to_parquet(path, index=False)
    return box


def replay_elo(games: pd.DataFrame) -> np.ndarray:
    """Pre-game production Elo difference (home - away), keyed by TEAM_ID."""
    elo, out = {}, np.zeros(len(games))
    for i, (h, a, m) in enumerate(zip(games["H_TEAM_ID"], games["A_TEAM_ID"],
                                      games["MARGIN"])):
        out[i] = elo.get(h, ELO_DEFAULT) - elo.get(a, ELO_DEFAULT)
        update_elo_game(h, a, m > 0, float(m), elo)
    return out


def build_dataset(raw: pd.DataFrame, data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    key = json.dumps([VERSION, RATING, AVAIL, REPLACEMENT, len(raw)], sort_keys=True)
    path = os.path.join(CACHE_DIR, "features_v6_%s.parquet"
                        % hashlib.md5(key.encode()).hexdigest()[:10])
    if os.path.exists(path):
        return pd.read_parquet(path)

    tg = fb.team_games(raw)
    prior = fb.season_prior(tg, "NRTG", RATING["regress"])
    tg["NRTG_S"] = fb.shrunk_std(tg, "NRTG", prior, RATING["k"])
    tg["G6"] = fb.schedule_features(tg)["G6"]

    box = load_box(data_dir).copy()
    box["GMSC"] = box["GMSC"] - REPLACEMENT * box["MIN"]
    av = fpl.availability(box, tg, **AVAIL)
    tg = tg.merge(av, on=["GAME_ID", "TEAM_ID"], how="left")
    tg["AVAIL"] = tg["AVAIL"].fillna(0.0)

    games = fb.to_games(tg, ["NRTG_S", "AVAIL", "G6"])
    games["ELO_DIFF"] = replay_elo(games)
    games["NRTG_DIFF"] = games["H_NRTG_S"] - games["A_NRTG_S"]
    games["AVAIL_DIFF"] = games["H_AVAIL"] - games["A_AVAIL"]
    games["G6_DIFF"] = games["H_G6"] - games["A_G6"]
    df = games[["GAME_ID", "GAME_DATE", "SEASON", "RESULT", "MARGIN",
                "H_TEAM_ID", "A_TEAM_ID", "NEUTRAL"] + FEATURES]
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
    model.fit(train[FEATURES].values, train["RESULT"].values)
    return model.predict_proba(test[FEATURES].values)[:, 1]
