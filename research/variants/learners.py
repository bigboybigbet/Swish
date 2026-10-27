"""
learners — model class, regularization and calibration on the v5 features.

build_dataset: production v5 features plus the pieces the live Monte Carlo
uses (10-game std of points scored / allowed per team, shift(1) like every
other rolling stat) so the analytic MC probability can be tested as a blend
component.  fit_predict is filled in below.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import FEATURE_COLS, HOME_ADV, MC_CORR
from research.benchmark import CACHE_DIR, baseline_dataset

VERSION = 1   # bump to invalidate the research/cache file
_VALID = ("002", "004", "005")
_NEEDED = ["PLUS_MINUS", "PTS", "FGM", "FGA", "FG3M",
           "FTA", "AST", "TOV", "REB", "OREB", "DREB", "WL"]


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


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    raise NotImplementedError
