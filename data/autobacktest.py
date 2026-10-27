import json
import os
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams as nba_teams_static

from config import recent_seasons
from data.elo import load_elo, save_elo, update_elo_game
from engine.features import get_team_advanced_stats
from engine.models import BlendCalibrator, predict_hybrid

AUTOBACKTEST_LOG = "autobacktest_log.json"
AUTOBACKTEST_CSV = "autobacktest_results.csv"
SEASONS = recent_seasons(2)   # previous + current season

_TEAM_MAP = {t["id"]: t["full_name"] for t in nba_teams_static.get_teams()}


def _team_name(team_id: int) -> str | None:
    return _TEAM_MAP.get(int(team_id))


def load_log() -> set:
    if os.path.exists(AUTOBACKTEST_LOG):
        with open(AUTOBACKTEST_LOG) as f:
            return set(json.load(f).get("processed_game_ids", []))
    return set()


def _save_log(new_ids: set):
    existing = load_log()
    all_ids  = existing | new_ids
    with open(AUTOBACKTEST_LOG, "w") as f:
        json.dump({"processed_game_ids": list(all_ids), "total": len(all_ids)}, f)


def log_stats() -> dict:
    if os.path.exists(AUTOBACKTEST_LOG):
        with open(AUTOBACKTEST_LOG) as f:
            d = json.load(f)
        return {"total": d.get("total", 0)}
    return {"total": 0}


def _append_to_csv(rows: list):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if os.path.exists(AUTOBACKTEST_CSV):
        existing = pd.read_csv(AUTOBACKTEST_CSV)
        pd.concat([existing, new_df], ignore_index=True).to_csv(AUTOBACKTEST_CSV, index=False)
    else:
        new_df.to_csv(AUTOBACKTEST_CSV, index=False)


def _fetch_new_games(status_cb=None) -> pd.DataFrame:
    def report(msg):
        if status_cb:
            status_cb(msg)

    processed = load_log()
    all_games  = []

    for s in SEASONS:
        try:
            report(f"Fetching {s} schedule from NBA API…")
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=s, league_id_nullable="00"
            )
            all_games.append(finder.get_data_frames()[0])
            time.sleep(1.0)
        except Exception as e:
            report(f"Could not fetch {s}: {e}")

    if not all_games:
        return pd.DataFrame()

    df = pd.concat(all_games, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df[df["WL"].isin(["W", "L"])]

    home_df = df[df["MATCHUP"].str.contains(r"vs\.", na=False)].copy()
    away_df = df[df["MATCHUP"].str.contains("@", na=False)].copy()

    paired = home_df.merge(
        away_df[["GAME_ID", "TEAM_ID", "WL"]],
        on="GAME_ID", suffixes=("_h", "_a"),
    ).rename(columns={
        "TEAM_ID_h": "HOME_TEAM_ID", "TEAM_ID_a": "AWAY_TEAM_ID",
        "WL_h": "WL_HOME",
    })

    paired["HOME_WON"] = (paired["WL_HOME"] == "W").astype(int)
    paired["MARGIN"]   = paired["PLUS_MINUS"]  # home margin, drives MOV-Elo
    paired = paired[~paired["GAME_ID"].isin(processed)]
    paired = paired.sort_values("GAME_DATE").reset_index(drop=True)

    report(f"Found {len(paired)} new completed games not yet backtested.")
    return paired[["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID",
                   "HOME_WON", "MARGIN"]]


def run_auto_backtest(model, status_cb=None) -> dict:
    """
    Blindfold-predict all new completed games.

    For every game result the BlendCalibrator receives one online
    gradient-descent update — this is how the model adapts after backtest
    without corrupting the training dataset.
    """
    def report(msg):
        if status_cb:
            status_cb(msg)

    new_games = _fetch_new_games(status_cb=status_cb)

    if new_games.empty:
        return {"new": 0, "correct": 0, "lessons": 0, "acc": 0.0,
                "metrics": None, "calibrator": None}

    total         = len(new_games)
    correct       = 0
    csv_rows      = []
    processed_ids = set()

    # Collect (xgb_p, mc_p, elo_diff, actual) for batch calibrator update
    calib_updates = []

    # Elo evolves chronologically as games are replayed — each prediction
    # uses the pre-game rating, then the result updates it.
    elo_dict = load_elo()

    for i, row in new_games.iterrows():
        game_date = row["GAME_DATE"].date()
        home_name = _team_name(row["HOME_TEAM_ID"])
        away_name = _team_name(row["AWAY_TEAM_ID"])

        if not home_name or not away_name:
            continue

        hd = get_team_advanced_stats(home_name, target_date=game_date)
        ad = get_team_advanced_stats(away_name, target_date=game_date)

        if not hd or not ad:
            continue

        try:
            fp, spread, total_pts, xgb_p, mc_p, conflict, mc_full, hf, af = \
                predict_hybrid(hd, ad, model)

            pred   = 1 if fp > 0.5 else 0
            actual = int(row["HOME_WON"])
            ok     = pred == actual
            conf   = max(fp, 1 - fp)

            if ok:
                correct += 1

            elo_diff = hd.get("ELO", 1500) - ad.get("ELO", 1500)
            calib_updates.append((xgb_p, mc_p, elo_diff, actual))

            csv_rows.append({
                "GAME_ID":       row["GAME_ID"],
                "GAME_DATE":     str(game_date),
                "HOME_TEAM":     home_name,
                "AWAY_TEAM":     away_name,
                "PREDICTED":     "HOME" if pred == 1 else "AWAY",
                "ACTUAL":        "HOME" if actual == 1 else "AWAY",
                "CORRECT":       int(ok),
                "CONFIDENCE":    round(conf, 4),
                "HOME_WIN_PROB": round(fp, 4),
                "XGB_P":         round(xgb_p, 4),
                "MC_P":          round(mc_p, 4),
                "ELO_DIFF":      round(elo_diff, 1),
            })

            processed_ids.add(row["GAME_ID"])

            # MOV-Elo update + save so the next iteration's
            # get_team_advanced_stats reads the fresh rating
            update_elo_game(home_name, away_name, actual == 1,
                            float(row["MARGIN"]), elo_dict)
            save_elo(elo_dict)
        except Exception:
            continue

        if (i + 1) % 25 == 0:
            report(f"  {i + 1}/{total} games processed…")

    actual_processed = len(processed_ids)

    # Batch update BlendCalibrator — one gradient step per observed game
    calibrator = BlendCalibrator.load()
    for xgb_p, mc_p, elo_diff, actual in calib_updates:
        calibrator.update(xgb_p, mc_p, elo_diff, actual)
    calibrator.save()

    _append_to_csv(csv_rows)
    _save_log(processed_ids)

    acc = correct / actual_processed if actual_processed > 0 else 0.0
    report(
        f"Done — {actual_processed} new games · "
        f"{correct}/{actual_processed} correct ({acc * 100:.1f}%) · "
        f"blend calibrator updated ({calibrator.n_updates} total observations)"
    )

    return {
        "new":        actual_processed,
        "correct":    correct,
        "lessons":    actual_processed,  # every game is now a lesson
        "acc":        acc,
        "metrics":    None,              # XGBoost retrain is manual via sidebar
        "calibrator": calibrator.summary(),
    }
