"""
Walk-forward benchmark on real historical NBA games.

Data: team game logs 2010-11 … 2023-24 mirrored at
https://github.com/NocturneBear/NBA-Data-2010-2024 (same columns as the
nba_api leaguegamefinder endpoint the app uses).  Clone it, then:

    python -m research.benchmark --data-dir ../nba-data-2010-2024
    python -m research.benchmark --variants v5,elo,my_variant

For every test season S the model is trained on the N_TRAIN seasons before
S and frozen; every game of S is then predicted from features that only see
games played before tip-off (rolling form and Elo keep updating, exactly as
in live use).  Scores are pooled over all test seasons.

A variant is a module research/variants/<name>.py exposing
    fit_predict(train: DataFrame, test: DataFrame) -> np.ndarray   (P(home win))
and optionally
    build_dataset(raw: DataFrame) -> DataFrame
which must return one row per game with GAME_ID, GAME_DATE, SEASON,
RESULT (1 = home win) and MARGIN (home points minus away points).
Without build_dataset the variant gets the production v5 features.
"""
import argparse
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ELO_HOME_ADV, FEATURE_COLS  # noqa: E402
from data.elo import elo_expected  # noqa: E402
from engine.models import build_matchup_dataset, fit_matchup_model  # noqa: E402

DEFAULT_DATA_DIR = os.environ.get(
    "NBA_HISTORY_DIR", "/home/user/nocturnebear/nba-data-2010-2024")
CACHE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
TEST_SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21",
                "2021-22", "2022-23", "2023-24"]
N_TRAIN      = 3
EPS          = 1e-6


# ── data ──────────────────────────────────────────────────────────────────────

def load_history(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Regular season + playoff team game logs, leaguegamefinder columns."""
    parts = []
    for fn in ("regular_season_totals_2010_2024.csv", "play_off_totals_2010_2024.csv"):
        p = os.path.join(data_dir, fn)
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
    if not parts:
        raise FileNotFoundError(f"No game-log CSVs in {data_dir}")
    raw = pd.concat(parts, ignore_index=True)
    raw["GAME_ID"]   = raw["GAME_ID"].astype(str).str.zfill(10)
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"]).dt.normalize()
    raw = raw.drop_duplicates(["GAME_ID", "TEAM_ID"])
    return raw


def attach_season(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    seasons = raw.drop_duplicates("GAME_ID").set_index("GAME_ID")["SEASON_YEAR"]
    df = df.copy()
    df["SEASON"] = df["GAME_ID"].astype(str).str.zfill(10).map(seasons)
    return df


def baseline_dataset(raw: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """Production v5 features over the full history (cached on disk)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "v5_dataset.csv")
    if use_cache and os.path.exists(path):
        df = pd.read_csv(path, dtype={"GAME_ID": str})
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df
    df, _ = build_matchup_dataset(raw, report=lambda m: None)
    df = attach_season(df, raw)
    df.to_csv(path, index=False)
    return df


# ── scoring ───────────────────────────────────────────────────────────────────

def per_game_logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def score(p: np.ndarray, y: np.ndarray) -> dict:
    return {
        "n":       int(len(y)),
        "logloss": float(per_game_logloss(p, y).mean()),
        "brier":   float(np.mean((p - y) ** 2)),
        "acc":     float(np.mean((p > 0.5) == (y == 1))),
    }


def paired_diff(ll_a: np.ndarray, ll_b: np.ndarray) -> tuple:
    """Mean per-game log-loss difference (a − b) and its standard error.
    Negative = a is better."""
    d = ll_a - ll_b
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


# ── walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, fit_predict, test_seasons=TEST_SEASONS,
                 n_train: int = N_TRAIN) -> pd.DataFrame:
    """Returns one row per test game: GAME_ID, SEASON, RESULT, P."""
    order = sorted(df["SEASON"].dropna().unique())
    rows = []
    for s in test_seasons:
        i = order.index(s)
        train_seasons = order[max(0, i - n_train):i]
        train = df[df["SEASON"].isin(train_seasons)].sort_values("GAME_DATE")
        test  = df[df["SEASON"] == s].sort_values("GAME_DATE")
        p = np.asarray(fit_predict(train.reset_index(drop=True),
                                   test.reset_index(drop=True)), dtype=float)
        assert len(p) == len(test), "fit_predict must return one prob per test row"
        rows.append(pd.DataFrame({
            "GAME_ID": test["GAME_ID"].values, "SEASON": s,
            "RESULT": test["RESULT"].values, "P": p,
        }))
    return pd.concat(rows, ignore_index=True)


# ── built-in baselines ────────────────────────────────────────────────────────

def v5_fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Production pipeline: RandomizedSearch XGBoost + Platt calibration."""
    model, _ = fit_matchup_model(train, report=lambda m: None)
    return model.predict_proba(test[FEATURE_COLS])[:, 1]


def elo_fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Pure Elo win expectancy with home advantage — no learning."""
    return elo_expected(test["ELO_DIFF"].values + ELO_HOME_ADV, 0.0)


BUILTINS = {"v5": v5_fit_predict, "elo": elo_fit_predict}


def load_variant(name: str):
    """(fit_predict, build_dataset or None) for a builtin or variants module."""
    if name in BUILTINS:
        return BUILTINS[name], None
    mod = importlib.import_module(f"research.variants.{name}")
    return mod.fit_predict, getattr(mod, "build_dataset", None)


def run(names, data_dir=DEFAULT_DATA_DIR, test_seasons=TEST_SEASONS,
        n_train=N_TRAIN, reference="v5") -> dict:
    raw  = load_history(data_dir)
    base = baseline_dataset(raw)
    preds = {}
    for name in dict.fromkeys([reference] + list(names)):
        fp, build = load_variant(name)
        df = base if build is None else attach_season_if_missing(build(raw), raw)
        t0 = time.time()
        preds[name] = walk_forward(df, fp, test_seasons, n_train)
        preds[name]["_secs"] = time.time() - t0
    return preds


def attach_season_if_missing(df, raw):
    return df if "SEASON" in df.columns else attach_season(df, raw)


def report_table(preds: dict, reference: str = "v5") -> pd.DataFrame:
    """Pooled scores on the games every variant predicted, plus paired
    log-loss difference vs the reference (negative = better)."""
    common = set.intersection(*(set(p["GAME_ID"]) for p in preds.values()))
    ref = preds[reference].set_index("GAME_ID").loc[sorted(common)]
    ref_ll = per_game_logloss(ref["P"].values, ref["RESULT"].values)
    out = []
    for name, p in preds.items():
        q = p.set_index("GAME_ID").loc[sorted(common)]
        ll = per_game_logloss(q["P"].values, q["RESULT"].values)
        d, se = paired_diff(ll, ref_ll)
        row = {"variant": name, **score(q["P"].values, q["RESULT"].values),
               "d_logloss": d, "se": se, "secs": float(p["_secs"].iloc[0])}
        out.append(row)
    return pd.DataFrame(out).set_index("variant")


def season_table(preds: dict) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        for s, g in p.groupby("SEASON"):
            rows.append({"variant": name, "season": s,
                         **score(g["P"].values, g["RESULT"].values)})
    return pd.DataFrame(rows).pivot(index="season", columns="variant", values="logloss")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--variants", default="v5,elo")
    ap.add_argument("--reference", default="v5")
    ap.add_argument("--seasons", default=",".join(TEST_SEASONS))
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    a = ap.parse_args()
    preds = run(a.variants.split(","), a.data_dir, a.seasons.split(","),
                a.n_train, a.reference)
    pd.set_option("display.width", 160)
    print(report_table(preds, a.reference).round(4))
    print("\nLog loss by season:")
    print(season_table(preds).round(4))
