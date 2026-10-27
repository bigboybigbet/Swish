"""
player_elo — Elo for players (see player_elo_engine for the maths).

A team's pre-game strength is the expected-minutes-weighted sum of the
ratings of the players who play, so injuries, rest and trades move the
prediction the moment the lineup changes.  After each game the margin error
is split over the players who played in proportion to minutes and rating
uncertainty, on-court plus-minus redistributes credit inside the team, and a
box-score rating gives a fast prior for new or changing players.

build_dataset() replays every game since 2010-11 once (box scores parsed and
cached under research/cache/); fit_predict() maps the pre-game expected
margin to P(home win) with a logistic fit on the train seasons.
Hyperparameters in player_elo_engine.PARAMS were tuned on the dev window
2014-15 … 2016-17 only.
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.benchmark import CACHE_DIR, DEFAULT_DATA_DIR, load_history
from research.variants.player_elo_engine import BOX_STATS, prepare, replay

NAME = "player_elo"
FEATURES = ["PE_DIFF"]
BOX_CACHE = os.path.join(CACHE_DIR, "player_elo_box.parquet")
SIGNAL_FILE = os.path.join(CACHE_DIR, "signals", f"{NAME}.csv")

_BOX_COLS = {
    "points": "PTS", "fieldGoalsMade": "FGM", "fieldGoalsAttempted": "FGA",
    "threePointersMade": "FG3M", "threePointersAttempted": "FG3A",
    "freeThrowsMade": "FTM", "freeThrowsAttempted": "FTA",
    "reboundsOffensive": "OREB", "reboundsDefensive": "DREB",
    "assists": "AST", "steals": "STL", "blocks": "BLK", "turnovers": "TOV",
    "foulsPersonal": "PF", "plusMinusPoints": "PM",
}


# ── data ──────────────────────────────────────────────────────────────────────

def _minutes(s: pd.Series) -> np.ndarray:
    """'MM:SS' (or empty for DNP) → float minutes."""
    s = s.fillna("").astype(str)
    mm = s.str.extract(r"^(\d+):(\d+)")
    out = mm[0].astype(float) + mm[1].astype(float) / 60
    plain = pd.to_numeric(s.where(mm[0].isna()), errors="coerce")
    return out.fillna(plain).fillna(0.0).values


def load_box(data_dir: str = DEFAULT_DATA_DIR, use_cache: bool = True) -> pd.DataFrame:
    """Player box scores (regular season + playoffs), one row per player-game,
    cached as parquet."""
    if use_cache and os.path.exists(BOX_CACHE):
        return pd.read_parquet(BOX_CACHE)
    files = sorted(glob.glob(os.path.join(
        data_dir, "regular_season_box_scores_2010_2024_part_*.csv")))
    files.append(os.path.join(data_dir, "play_off_box_scores_2010_2024.csv"))
    df = pd.concat([pd.read_csv(f) for f in files if os.path.exists(f)],
                   ignore_index=True)
    df["GAME_ID"] = df["gameId"].astype(str).str.zfill(10)
    df = df.drop_duplicates(["GAME_ID", "personId"])
    out = pd.DataFrame({
        "GAME_ID": df["GAME_ID"].values,
        "TEAM_ID": df["teamId"].values,
        "PLAYER_ID": df["personId"].values,
        "PLAYER_NAME": df["personName"].astype(str).values,
        "COMMENT": df["comment"].fillna("").astype(str).values,
        "MIN": _minutes(df["minutes"]),
    })
    for c, k in _BOX_COLS.items():
        out[k] = df[c].fillna(0).astype(float).values
    os.makedirs(CACHE_DIR, exist_ok=True)
    out.to_parquet(BOX_CACHE, index=False)
    return out


def game_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Team game logs → one row per game (home vs away), date-sorted."""
    df = raw.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df = df[df["GAME_ID"].str[:3].isin(("002", "004", "005"))]
    df = df.dropna(subset=["WL", "PLUS_MINUS"])
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    is_home = df["MATCHUP"].str.contains("vs.", regex=False)
    h = df[is_home][["GAME_ID", "GAME_DATE", "SEASON_YEAR", "TEAM_ID", "WL", "PLUS_MINUS"]]
    a = df[~is_home][["GAME_ID", "TEAM_ID"]]
    g = h.merge(a, on="GAME_ID", suffixes=("_H", "_A"))
    g = g.rename(columns={"SEASON_YEAR": "SEASON"})
    g["RESULT"] = (g["WL"] == "W").astype(int)
    g["MARGIN"] = g["PLUS_MINUS"].astype(float)
    g = g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return g[["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID_H", "TEAM_ID_A",
              "RESULT", "MARGIN"]]


_PREP = {}


def prepared(raw: pd.DataFrame = None, data_dir: str = DEFAULT_DATA_DIR) -> dict:
    """Packed game + box arrays (memoised per process)."""
    if "prep" not in _PREP:
        raw = load_history(data_dir) if raw is None else raw
        _PREP["prep"] = prepare(game_table(raw), load_box(data_dir))
    return _PREP["prep"]


# ── benchmark interface ───────────────────────────────────────────────────────

def build_dataset(raw: pd.DataFrame, **params) -> pd.DataFrame:
    """One row per game with the pre-game player-Elo state."""
    prep = prepared(raw)
    out, _ = replay(prep, **params)
    return pd.concat([prep["games"], out], axis=1)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(train[FEATURES].values, train["RESULT"].values)
    return model.predict_proba(test[FEATURES].values)[:, 1]
