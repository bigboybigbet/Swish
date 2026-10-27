"""
player_rapm — regularized adjusted plus-minus player ratings with a jointly
fitted box-score prior (see player_rapm_engine for the maths).

Every finished game adds one margin equation (home margin = home advantage
+ minutes-weighted sum of home player ratings - same for the away side) and
one on/off equation per player (his plus-minus minus his share of the team
margin).  A player's rating is a box-score part (a BPM-style linear formula
in his past per-48 box rates, whose weights are fitted in the same ridge
regression) plus a plus-minus part shrunk toward 0 (or toward his previous
season).  Old games decay exponentially; the ridge is refit weekly from
earlier games only.  A team's pre-game strength is the expected-minutes-
weighted sum of the ratings of the players who play.

build_dataset() replays every game since 2010-11 (box scores parsed and
cached under research/cache/); fit_predict() maps the pre-game expected
margin to P(home win) with a logistic fit on the train seasons.
Hyperparameters in player_rapm_engine.PARAMS were tuned on the dev window
2014-15 … 2016-17 only.
"""
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.benchmark import CACHE_DIR, DEFAULT_DATA_DIR, load_history
from research.variants.player_rapm_engine import PARAMS, STATS, prepare, replay

NAME = "player_rapm"
FEATURES = ["RP_DIFF"]
BOX_CACHE = os.path.join(CACHE_DIR, "player_rapm_box.parquet")
SIGNAL_FILE = os.path.join(CACHE_DIR, "signals", f"{NAME}.csv")
VERSION = 3                    # bump when the engine output changes (cache key)

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
    """'MM:SS' / 'MM.000000:SS' (or empty for DNP) → float minutes."""
    s = s.fillna("").astype(str)
    mm = s.str.extract(r"^(\d+)(?:\.\d+)?:(\d+)")
    out = mm[0].astype(float) + mm[1].astype(float) / 60
    plain = pd.to_numeric(s.where(mm[0].isna()), errors="coerce")
    return out.fillna(plain).fillna(0.0).values


def load_box(data_dir: str = DEFAULT_DATA_DIR, use_cache: bool = True) -> pd.DataFrame:
    """Player box scores (regular season + playoffs), one row per
    player-game, cached as parquet."""
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
        out[k] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float).values
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

def build_dataset(raw: pd.DataFrame, use_cache: bool = True, **params) -> pd.DataFrame:
    """One row per game with the pre-game RAPM state (cached per parameter
    set under research/cache/)."""
    p = {**PARAMS, **params}
    key = hashlib.md5(json.dumps([p, len(raw), VERSION], sort_keys=True)
                      .encode()).hexdigest()[:10]
    path = os.path.join(CACHE_DIR, f"player_rapm_{key}.parquet")
    if use_cache and os.path.exists(path):
        return pd.read_parquet(path)
    prep = prepared(raw)
    out, _ = replay(prep, **p)
    df = pd.concat([prep["games"], out], axis=1)
    df.to_parquet(path, index=False)
    return df


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(train[FEATURES].values, train["RESULT"].values)
    return model.predict_proba(test[FEATURES].values)[:, 1]


# ── signal file for the ensemble stage ───────────────────────────────────────

SIGNAL_COLS = {
    "RP_DIFF": "diff",            # expected home margin, home advantage included
    "RP_HFA": "home",             # home advantage inside diff
    "RP_SD": "sd",                # predictive sd of the margin
    "RP_H": "home_strength",      # home side, points vs an average team
    "RP_A": "away_strength",
    "RP_LIVE_DIFF": "live_diff",  # same from the active roster (live inputs)
    "RP_FULL_DIFF": "full_diff",  # same with every recent regular available
    "RP_ABSENCE": "absence",      # diff - full_diff: net effect of absences
    "RP_BOX": "box_diff",         # box-score part of diff
    "RP_IMP": "pm_diff",          # plus-minus part of diff
}


def write_signals(path: str = SIGNAL_FILE, first_season: str = "2011-12") -> pd.DataFrame:
    """Regenerate research/cache/signals/player_rapm.csv: one row per game
    from 2011-12 on with pre-game values only (no result columns)."""
    df = build_dataset(load_history())
    df["RP_ABSENCE"] = df["RP_DIFF"] - df["RP_FULL_DIFF"]
    df = df[(df["SEASON"] >= first_season) & df["RP_DIFF"].notna()]
    out = df[["GAME_ID", "SEASON"]].copy()
    for c, k in SIGNAL_COLS.items():
        out[f"{NAME}_{k}"] = df[c].round(4).values
    out["GAME_ID"] = out["GAME_ID"].astype(str).str.zfill(10)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    return out


# ── inspection and live use ──────────────────────────────────────────────────

def build_engine(raw: pd.DataFrame = None, until=None, **params):
    """Engine state after every game before `until` (a date; None = all),
    refit on everything played."""
    prep = prepared(raw)
    if until is not None:
        n = int((prep["games"]["GAME_DATE"] < pd.Timestamp(until)).sum())
        cut, ucut = prep["ptr"][2 * n], prep["uptr"][2 * n]
        prep = {"games": prep["games"].iloc[:n],
                "ptr": prep["ptr"][:2 * n + 1], "uptr": prep["uptr"][:2 * n + 1],
                **{k: prep[k][:n] for k in ("day", "season", "home", "away", "margin")},
                **{k: prep[k][:cut] for k in ("pid", "name", "min", "pm", "stats")},
                **{k: prep[k][:ucut] for k in ("upid", "uname")}}
    _, eng = replay(prep, **{**PARAMS, **params})
    eng.refit(int(prep["day"][-1]) + 1)
    return eng


def leaderboard(date=None, n: int = 10, min_games: int = 40) -> pd.DataFrame:
    """Top players by rating as of `date` (None = end of the data), among
    players who played in the 90 days before it."""
    eng = build_engine(until=date)
    last = int(eng.last_day[:len(eng.ids)].max())
    return pd.DataFrame(eng.leaderboard(n, min_games, active_since=last - 90))


# Probability that a player listed on the ESPN injury report plays
STATUS_PLAY_PROB = {"out": 0.0, "doubtful": 0.15, "questionable": 0.5,
                    "day-to-day": 0.75, "probable": 0.9}


def live_prediction(eng, home, away, home_roster, away_roster, game_date,
                    injuries: dict = None, season: str = None) -> dict:
    """Pre-game prediction for an upcoming game from a live engine.

    home_roster / away_roster: player ids on each active roster (e.g.
    eng.roster(team, since_day) or nba_api commonteamroster); injuries:
    {player_id: ESPN status}.  Players ruled out drop out, the rest are
    weighted by their chance to play and minutes are re-normalised to 240.
    Returns margin, sd, hfa, both sides and `absence` (margin lost to the
    injury report)."""
    injuries = injuries or {}
    day = (pd.Timestamp(game_date).normalize() - pd.Timestamp("2000-01-01")).days
    if season is not None and season != eng.season:
        eng.new_season(season, day)            # offseason bookkeeping first
        eng.refit(day)
    else:
        eng.maybe_refit(day)

    def avail(roster):
        prob = np.array([STATUS_PLAY_PROB.get(str(injuries.get(p, "")).lower(), 1.0)
                         for p in roster])
        keep = prob > 0
        return [p for p, k in zip(roster, keep) if k], prob[keep]

    hp, hprob = avail(home_roster)
    ap, aprob = avail(away_roster)
    pred = eng.predict(home, away, hp, ap, hprob, aprob, dressed=True)
    full = eng.predict(home, away, home_roster, away_roster, dressed=True)
    pred["absence"] = pred["margin"] - full["margin"]
    return pred
