"""
margin_model — predict the point margin, then convert it to P(home win).

Target: MARGIN (home pts − away pts) instead of win/loss.  A ridge regression
on standardized features gives the expected margin m; then
    P(home win) = Φ(m / s)
with s fitted by maximum likelihood on the binary results of the train frame.
s comes out ≈10.5 pts, well below the ≈12 pt residual SD: final margins are
fat-tailed (blowouts, garbage time), which inflates the SD without moving the
win probability, so using the residual SD gives under-confident probabilities.

Features (margin_model_features.py, all known before tip-off):
  SRS_DIFF   opponent-adjusted margin rating diff, re-solved daily from earlier
             games (120-day half-life, previous season weighted 0.75)
  B2B_H/A, REST_D, G4_D, G7_D   back-to-backs, rest days, schedule density
  STR_DIFF   summed value of each team's dressed players (EWMA game score of
             earlier games, 40-game half-life) — catches rest / injuries / trades
  MISS_DIFF  summed value of players on the inactive list (DND / NWT)

All choices were made on the dev window (2014-15 … 2016-17); see the result
notes for the comparison against a direct classifier on the same features.
"""
import hashlib
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.benchmark import CACHE_DIR, DEFAULT_DATA_DIR
from research.variants import margin_model_features as mf

FEATURES = ["SRS_DIFF", "B2B_H", "B2B_A", "REST_D", "G4_D", "G7_D",
            "STR_DIFF", "MISS_DIFF"]
RIDGE_ALPHA = 1.0


def build_dataset(raw: pd.DataFrame, data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """One row per game with the pre-game features (cached under research/cache)."""
    key = hashlib.md5(f"{len(raw)}|{raw['GAME_ID'].max()}|{mf.SRS_HALFLIFE}|{mf.SRS_PREV_W}|"
                      f"{mf.SRS_RIDGE}|{mf.VALUE_HALFLIFE}".encode()).hexdigest()[:10]
    path = os.path.join(CACHE_DIR, f"margin_model_{key}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"GAME_ID": str})
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df

    g = mf.game_table(raw)
    srs = mf.srs_ratings(g)
    df = pd.concat([
        g,
        pd.DataFrame({"SRS_DIFF": srs["SRS_H"] - srs["SRS_A"], "SRS_HCA": srs["SRS_HCA"]}),
        mf.rest_features(g),
        mf.strength_features(g, mf.load_box_scores(data_dir)),
    ], axis=1)
    df[["STR_DIFF", "MISS_DIFF"]] = df[["STR_DIFF", "MISS_DIFF"]].fillna(0.0)
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def fit_scale(m: np.ndarray, y: np.ndarray) -> float:
    """MLE of s in P(win) = Φ(m / s) on the binary outcomes."""
    def nll(s):
        p = np.clip(norm.cdf(m / s), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(5.0, 30.0), method="bounded").x)


def fit(train: pd.DataFrame) -> tuple:
    """(margin regressor, scale s) fitted on the train frame."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(train[FEATURES].values, train["MARGIN"].values)
    s = fit_scale(model.predict(train[FEATURES].values), train["RESULT"].values)
    return model, s


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model, s = fit(train)
    return norm.cdf(model.predict(test[FEATURES].values) / s)
