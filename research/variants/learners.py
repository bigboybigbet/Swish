"""
learners — model class, regularization and calibration on the v5 features.

The 17 production features add little beyond ELO_DIFF, and the v5
RandomizedSearch XGBoost overfits them.  This variant keeps the features and
replaces the learner with an equal logit-average of two heavily regularized
models, both anchored on Elo:

  • L2 logistic regression on the standardized features (C = 0.03);
  • a shallow monotone LightGBM boosted on top of a fitted Elo logistic
    (init score = a + b*ELO_DIFF): 4 leaves, lr 0.02, min 200 games per
    leaf, L2 10, monotone in every *_DIFF / B2B / venue feature, trees
    chosen by early stopping on the last 20% of the train frame (time
    order), then refit on the whole frame.

No extra calibration layer: Platt on a held-out slice was neutral and
isotonic hurt (dev).  All choices were made on the dev window
(2014-15 … 2016-17); see the result notes for the numbers.

build_dataset also adds the 10-game score SDs used by the live Monte Carlo
and its analytic probability MC_P, so the production blend can be measured.
The ensemble itself uses only config.FEATURE_COLS.
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLS, HOME_ADV, MC_CORR
from research.benchmark import CACHE_DIR, baseline_dataset

VERSION = 1   # bump to invalidate the research/cache file
_VALID = ("002", "004", "005")
_NEEDED = ["PLUS_MINUS", "PTS", "FGM", "FGA", "FG3M",
           "FTA", "AST", "TOV", "REB", "OREB", "DREB", "WL"]

LR_C = 0.03
GBM_PARAMS = {
    "objective": "binary", "learning_rate": 0.02, "num_leaves": 4,
    "min_data_in_leaf": 200, "lambda_l2": 10.0, "feature_fraction": 0.7,
    "bagging_fraction": 0.8, "bagging_freq": 1, "num_threads": 1,
    "verbose": -1, "seed": 0,
}
GBM_VAL_FRAC = 0.2
GBM_PATIENCE = 100
GBM_MAX_TREES = 3000

# +1: larger value favours the home team, -1: favours the away team, 0: free
MONOTONE = {
    "ELO_DIFF": 1, "NET5_DIFF": 1, "NET15_DIFF": 1, "PTS10_DIFF": 1,
    "PA10_DIFF": -1, "EFG10_DIFF": 1, "TOV10_DIFF": -1, "ORB10_DIFF": 1,
    "FTR10_DIFF": 1, "PACE10_MEAN": 0, "WPCT10_DIFF": 1, "STREAK_DIFF": 1,
    "REST_DIFF": 1, "H_B2B": -1, "A_B2B": 1, "H_HOME_WPCT": 1,
    "A_AWAY_WPCT": -1,
}


# ── dataset ───────────────────────────────────────────────────────────────────

def team_score_sds(raw: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: std of points for / against over the previous 10 games
    (same filtering, ordering and window as engine.models / engine.features)."""
    df = raw.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df = df[df["GAME_ID"].str[:3].isin(_VALID)].dropna(subset=_NEEDED)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False)
    df["PA"] = df["PTS"] - df["PLUS_MINUS"]
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    g = df.groupby("TEAM_ID")
    for col in ("PTS", "PA"):
        df[col + "_SD"] = g[col].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).std()).clip(lower=3.0)
    return df[["GAME_ID", "IS_HOME", "PTS_SD", "PA_SD"]]


def mc_prob(df: pd.DataFrame) -> np.ndarray:
    """Analytic limit of engine.simulate.run_monte_carlo: P(margin > 0) for a
    bivariate normal with the live app's means, stds, HOME_ADV and MC_CORR."""
    mu = (df["PTS10_DIFF"] - df["PA10_DIFF"]) / 2 + HOME_ADV
    h_sd = np.maximum(np.hypot(df["H_PTS_SD"], df["A_PA_SD"]) / np.sqrt(2), 4.0)
    a_sd = np.maximum(np.hypot(df["A_PTS_SD"], df["H_PA_SD"]) / np.sqrt(2), 4.0)
    sd = np.sqrt(h_sd ** 2 + a_sd ** 2 - 2 * MC_CORR * h_sd * a_sd)
    return norm.cdf(mu / sd)


def build_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """v5 rows (same games) + team score SDs + MC_P, cached on disk."""
    path = os.path.join(CACHE_DIR, f"learners_v{VERSION}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"GAME_ID": str})
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df
    df = baseline_dataset(raw)
    sds = team_score_sds(raw)
    for side, home in (("H", True), ("A", False)):
        s = sds[sds["IS_HOME"] == home].drop(columns="IS_HOME")
        s = s.rename(columns={"PTS_SD": f"{side}_PTS_SD", "PA_SD": f"{side}_PA_SD"})
        df = df.merge(s, on="GAME_ID", how="left")
    sd_cols = ["H_PTS_SD", "H_PA_SD", "A_PTS_SD", "A_PA_SD"]
    df[sd_cols] = df[sd_cols].fillna(6.0)   # live fallback when < 2 games
    df["MC_P"] = mc_prob(df)
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ── model ─────────────────────────────────────────────────────────────────────

class EloAnchoredEnsemble:
    """Logit-average of an L2 logistic and an Elo-initialized monotone GBM.
    Fit on time-ordered matchup rows; predicts P(home win)."""

    def __init__(self, features=FEATURE_COLS):
        self.features = list(features)

    @staticmethod
    def _elo_lr(frame: pd.DataFrame) -> LogisticRegression:
        """P(home win) from ELO_DIFF alone: learns the home edge and slope."""
        return LogisticRegression(C=1e6).fit(frame[["ELO_DIFF"]], frame["RESULT"])

    def _boost(self, frame: pd.DataFrame, offset: np.ndarray, rounds: int,
               valid: tuple = None) -> lgb.Booster:
        """Monotone GBM on top of the Elo logit (native API: init_score works
        the same in every LightGBM version the app may have)."""
        params = {**GBM_PARAMS,
                  "monotone_constraints": [MONOTONE.get(c, 0) for c in self.features]}
        data = lgb.Dataset(frame[self.features], frame["RESULT"], init_score=offset)
        kw = {}
        if valid is not None:
            vf, voff = valid
            kw["valid_sets"] = [lgb.Dataset(vf[self.features], vf["RESULT"],
                                            init_score=voff, reference=data)]
            kw["callbacks"] = [lgb.early_stopping(GBM_PATIENCE, verbose=False)]
        return lgb.train(params, data, num_boost_round=rounds, **kw)

    def fit(self, df: pd.DataFrame) -> "EloAnchoredEnsemble":
        df = df.sort_values("GAME_DATE")
        X, y = df[self.features], df["RESULT"]
        self.lr = make_pipeline(StandardScaler(),
                                LogisticRegression(C=LR_C, max_iter=2000)).fit(X, y)

        # number of trees: early stopping on the most recent slice
        n = int(len(df) * (1 - GBM_VAL_FRAC))
        a, b = df.iloc[:n], df.iloc[n:]
        elo = self._elo_lr(a)
        es = self._boost(a, elo.decision_function(a[["ELO_DIFF"]]), GBM_MAX_TREES,
                         valid=(b, elo.decision_function(b[["ELO_DIFF"]])))
        self.n_trees = max(int(es.best_iteration or 1), 1)

        self.elo = self._elo_lr(df)
        self.gbm = self._boost(df, self._offset(df), self.n_trees)
        return self

    def _offset(self, df: pd.DataFrame) -> np.ndarray:
        return self.elo.decision_function(df[["ELO_DIFF"]])

    def decision_function(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features]
        z_lr = self.lr.decision_function(X)
        z_gb = self.gbm.predict(X, raw_score=True) + self._offset(df)
        return (z_lr + z_gb) / 2

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = 1 / (1 + np.exp(-self.decision_function(df)))
        return np.c_[1 - p, p]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return EloAnchoredEnsemble().fit(train).predict_proba(test)[:, 1]
