"""
kalman — team strength as a state-space model, filtered with the exact
joint Kalman filter (see kalman_engine.py for the equations).

  • every team has an offensive and a defensive strength (points per game),
    drifting as a Gaussian random walk (variance ∝ days elapsed), partially
    mean-reverted and widened every summer; one covariance matrix couples
    all teams, so beating a team that just beat a good team moves both;
  • league home edge (drifting), per-team home edges, a back-to-back term
    and the lineup coefficients are states too, learned online;
  • lineup term: who is available tonight (box-score DNP/inactive lists,
    live: the injury report) via pre-game player values, so a missing star
    lowers the expectation and does not drag the team rating down;
  • observation noise σ is re-estimated online (scoring era drift).

Pre-game output: expected home margin μ and predictive sd s (state + noise);
fit_predict maps z = μ/s to P(home win) with a probit fitted on the train
seasons (Φ(z) itself is already close; the fit corrects the slightly
peaked margin distribution).

Hyperparameters: filter noise/dynamics by maximising the dev-window
(2014-15 … 2016-17) Gaussian log likelihood of the margins.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr

from research.benchmark import CACHE_DIR, DEFAULT_DATA_DIR
from research.variants.kalman_engine import (
    EPOCH, LINEUP_COLS, PARAMS, PLAYER_PARAMS, KalmanRatings, PlayerBook,
    box_table, game_table, lineup_strength, player_values, replay)

BOX_FILES = ["regular_season_box_scores_2010_2024_part_1.csv",
             "regular_season_box_scores_2010_2024_part_2.csv",
             "regular_season_box_scores_2010_2024_part_3.csv",
             "play_off_box_scores_2010_2024.csv"]
SIGNAL_PATH = os.path.join(CACHE_DIR, "signals", "kalman.csv")
SIGNAL_FROM = "2011-12"


# ── data ──────────────────────────────────────────────────────────────────────

def load_box(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Parsed player box scores (box_table), cached as parquet."""
    path = os.path.join(CACHE_DIR, "kalman_box.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    parts = [pd.read_csv(os.path.join(data_dir, f)) for f in BOX_FILES
             if os.path.exists(os.path.join(data_dir, f))]
    box = box_table(pd.concat(parts, ignore_index=True))
    os.makedirs(CACHE_DIR, exist_ok=True)
    box.to_parquet(path, index=False)
    return box


def build_dataset(raw: pd.DataFrame, params: dict = None, player_params: dict = None,
                  use_cache: bool = True) -> pd.DataFrame:
    """One row per game: GAME_ID, GAME_DATE, SEASON, RESULT, MARGIN and the
    pre-game filter output (MU, SD, Z, ratings, home, lineup, total)."""
    p = {**PARAMS, **(params or {})}
    pp = {**PLAYER_PARAMS, **(player_params or {})}
    key = hashlib.md5(json.dumps([p, pp, len(raw)], sort_keys=True, default=str)
                      .encode()).hexdigest()[:10]
    path = os.path.join(CACHE_DIR, f"kalman_{key}.parquet")
    if use_cache and os.path.exists(path):
        return pd.read_parquet(path)
    games = game_table(raw)
    vals = player_values(load_box(), games, **pp)
    res = replay(games, lineup=lineup_strength(vals, games, **pp), **p)
    df = pd.concat([games[["GAME_ID", "GAME_DATE", "SEASON", "RESULT", "MARGIN",
                           "H", "A", "PLAYOFF", "NEUTRAL"]], res], axis=1)
    df["Z"] = df["MU"] / df["SD"]
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


# ── benchmark interface ───────────────────────────────────────────────────────

def fit_probit(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """MLE of (a, b) in P(y = 1) = Φ(a + b·z)."""
    def nll(ab):
        u = ab[0] + ab[1] * z
        return -np.mean(y * log_ndtr(u) + (1 - y) * log_ndtr(-u))
    return minimize(nll, np.array([0.0, 1.0]), method="BFGS").x


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    a, b = fit_probit(train["Z"].values, train["RESULT"].values)
    return ndtr(a + b * test["Z"].values)


# ── signal file for the ensemble ──────────────────────────────────────────────

def write_signals(path: str = SIGNAL_PATH, data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """research/cache/signals/kalman.csv: one row per game 2011-12 … with the
    pre-game signals only (no result)."""
    from research.benchmark import load_history
    df = build_dataset(load_history(data_dir))
    df = df[df["SEASON"] >= SIGNAL_FROM].dropna(subset=["MU", "SD"])
    out = pd.DataFrame({
        "GAME_ID": df["GAME_ID"].astype(str).str.zfill(10),
        "SEASON": df["SEASON"],
        "kalman_diff": df["MU"],             # expected home margin, home edge incl.
        "kalman_sd": df["SD"],               # predictive sd of the margin
        "kalman_sd_mean": df["SD_MEAN"],     # sd of kalman_diff itself (state)
        "kalman_p": ndtr(df["Z"]),           # Φ(diff / sd), uncalibrated
        "kalman_home": df["HOME"],           # home edge inside kalman_diff
        "kalman_lineup": df["LINEUP"],       # availability part of kalman_diff
        "kalman_rating_h": df["R_H"],        # team ratings beyond lineup
        "kalman_rating_a": df["R_A"],
        "kalman_rating_sd_h": df["SD_H"],
        "kalman_rating_sd_a": df["SD_A"],
        "kalman_total": df["TOTAL"],         # expected total points
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False, float_format="%.4f")
    return out


# ── live use ──────────────────────────────────────────────────────────────────

def live_state(raw: pd.DataFrame, box: pd.DataFrame, params: dict = None,
               player_params: dict = None):
    """Replay every finished game in the leaguegamefinder logs (raw) and the
    nba.com box scores (box, box_table format) → (KalmanRatings, PlayerBook,
    latest lineup per team).  Afterwards keep both current with
    update_after_game()."""
    p = {**PARAMS, **(params or {})}
    pp = {**PLAYER_PARAMS, **(player_params or {})}
    games = game_table(raw)
    LH, LA = lineup_strength(player_values(box, games, **pp), games, **pp)
    _, kf = replay(games, lineup=(LH, LA), return_filter=True, **p)
    book = PlayerBook(**pp)
    side = pd.concat([games[["GAME_ID", "DAY", "H", "MARGIN"]].rename(columns={"H": "TEAM_ID"}),
                      games[["GAME_ID", "DAY", "A", "MARGIN"]].rename(columns={"A": "TEAM_ID"})
                      .assign(MARGIN=lambda d: -d["MARGIN"])])
    b = box.merge(side, on=["GAME_ID", "TEAM_ID"]).sort_values(["DAY", "GAME_ID"])
    b = b[b["PLAYED"]]
    for day, grp in b.groupby("DAY", sort=True):
        book.add_game(int(day), zip(grp["PID"], grp["MIN"], grp["GMSC"], grp["PM"], grp["MARGIN"]))
    last = {}
    for k in range(len(games)):
        last[games["H"].iat[k]], last[games["A"].iat[k]] = LH[k], LA[k]
    return kf, book, last


def update_after_game(kf: KalmanRatings, book: PlayerBook, game: dict, box_rows=()):
    """After a final: game = dict(home, away, date, season, home_pts, away_pts,
    neutral, playoff, b2b_home, b2b_away, lin_home, lin_away) with the
    pre-game lineups; box_rows = (pid, minutes, game_score, plus_minus,
    team_margin) per player."""
    day = (pd.Timestamp(game["date"]).normalize() - EPOCH).days
    kf.roll_season(game["season"])
    kf.advance(day)
    kf.update(game["home"], game["away"], game["home_pts"] - game["away_pts"],
              game["home_pts"] + game["away_pts"], playoff=game.get("playoff", 0),
              neutral=game.get("neutral", 0), b2b_h=game.get("b2b_home", 0),
              b2b_a=game.get("b2b_away", 0), lin_h=game["lin_home"], lin_a=game["lin_away"])
    book.add_game(day, box_rows)


def availability(roster, injuries: list) -> list:
    """P(plays) per roster player from the ESPN report entries
    ({name, status}); STATUS_WEIGHT is the app's chance of missing."""
    from config import STATUS_WEIGHT
    miss = {i["name"]: STATUS_WEIGHT.get(i["status"], 0.0) for i in injuries}
    return [1.0 - miss.get(name, 0.0) for name in roster]


def predict_game(kf: KalmanRatings, book: PlayerBook, home, away, game_date, season,
                 home_players, away_players, home_p=None, away_p=None,
                 neutral=0, playoff=0, b2b_home=0, b2b_away=0) -> dict:
    """Pre-game prediction for an upcoming game.  *_players: roster player
    ids; *_p: their chance of playing (availability())."""
    day = (pd.Timestamp(game_date).normalize() - EPOCH).days
    kf.roll_season(season)
    kf.advance(day)
    lh = book.lineup(home_players, day, home_p)
    la = book.lineup(away_players, day, away_p)
    mu, sd, sd_mean, total = kf.predict(home, away, playoff=playoff, neutral=neutral,
                                        b2b_h=b2b_home, b2b_a=b2b_away, lin_h=lh, lin_a=la)
    return {"kalman_diff": mu, "kalman_sd": sd, "kalman_sd_mean": sd_mean,
            "kalman_total": total, "p_home": float(ndtr(mu / sd)),
            "lin_home": lh, "lin_away": la}


def rankings(raw: pd.DataFrame = None, box: pd.DataFrame = None) -> pd.DataFrame:
    """Current rankings table (rating = points vs an average team on a
    neutral court with each team's latest lineup, ± 1 sd)."""
    from research.benchmark import load_history
    raw = load_history() if raw is None else raw
    box = load_box() if box is None else box
    kf, _, last = live_state(raw, box)
    names = (raw.sort_values("GAME_DATE").drop_duplicates("TEAM_ID", keep="last")
             .set_index("TEAM_ID")["TEAM_ABBREVIATION"])
    tab = kf.ratings(last)
    tab.insert(1, "TEAM", tab["TEAM_ID"].map(names))
    return tab


if __name__ == "__main__":
    sig = write_signals()
    print(f"wrote {len(sig)} rows → {SIGNAL_PATH}")
    pd.set_option("display.width", 160)
    print(rankings().round(2).head(10))
