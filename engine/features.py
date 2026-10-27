import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

import data.cache as cache
from data.elo import get_team_elo

# Regular season (002), playoffs (004), play-in (005) — mirrors models.py
_VALID_GAME_PREFIXES = ("002", "004", "005")


def _streak(wl_list) -> int:
    s = 0
    for wl in wl_list:
        if wl == "W":
            s = s + 1 if s >= 0 else 1
        else:
            s = s - 1 if s <= 0 else -1
    return s


def _venue_wpct(df: pd.DataFrame, home: bool, fallback: float) -> float:
    """Win% over the team's last 10 games at the same venue type."""
    mask  = df["MATCHUP"].str.contains("vs.", regex=False) if home \
            else df["MATCHUP"].str.contains("@", regex=False)
    venue = df[mask].tail(10)
    if len(venue) < 3:
        return fallback
    return float((venue["WL"] == "W").mean())


def get_team_advanced_stats(team_name: str, target_date=None, window: int = 10) -> dict | None:
    """
    Live team form dict — computed the same way as the training dataset
    (multi-window rolling form, four factors, pace, venue splits) so the
    model sees identical feature semantics at train and predict time.
    """
    try:
        nba_teams = teams.get_teams()
        team_info = [
            t for t in nba_teams
            if team_name.lower() in t["full_name"].lower()
            or t["full_name"].lower() in team_name.lower()
        ]
        if not team_info:
            return None
        team_id   = team_info[0]["id"]
        full_name = team_info[0]["full_name"]

        # ── SQLite cache — skip API if fresh data exists ──────────────
        cache_key = f"lgf_{team_id}"
        df = cache.get(cache_key)
        if df is None:
            finder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
            df = finder.get_data_frames()[0]
            cache.set(cache_key, df)

        df = df[df["GAME_ID"].astype(str).str[:3].isin(_VALID_GAME_PREFIXES)]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        if target_date:
            df = df[df["GAME_DATE"] < pd.to_datetime(target_date)]

        needed = ["PLUS_MINUS", "PTS", "FGM", "FGA", "FG3M",
                  "FTA", "AST", "TOV", "REB", "OREB", "DREB", "WL"]
        df = df.sort_values("GAME_DATE").dropna(subset=needed)
        if df.empty or len(df) < 3:
            return None

        r5  = df.tail(5)
        r15 = df.tail(15)
        r10 = df.tail(window).copy()
        r10["PTS_ALLOWED"] = r10["PTS"] - r10["PLUS_MINUS"]
        r10["POSS"]        = r10["FGA"] + 0.44 * r10["FTA"] - r10["OREB"] + r10["TOV"]

        tot_fga  = r10["FGA"].sum()
        efg_pct  = (r10["FGM"].sum() + 0.5 * r10["FG3M"].sum()) / tot_fga if tot_fga > 0 else 0
        ast_tov  = r10["AST"].sum() / max(r10["TOV"].sum(), 1)
        tot_pts  = r10["PTS"].sum()
        ts_den   = 2 * (tot_fga + 0.44 * r10["FTA"].sum())
        ts_pct   = tot_pts / ts_den if ts_den > 0 else 0
        tot_oreb = r10["OREB"].sum()
        tot_dreb = r10["DREB"].sum()
        oreb_pct = tot_oreb / (tot_oreb + tot_dreb) if (tot_oreb + tot_dreb) > 0 else 0
        tot_poss = r10["POSS"].sum()
        tov_rate = r10["TOV"].sum() / tot_poss if tot_poss > 0 else 0
        ft_rate  = r10["FTA"].sum() / tot_fga if tot_fga > 0 else 0
        wpct10   = float((r10["WL"] == "W").mean())

        last_game = df["GAME_DATE"].iloc[-1]
        # NBA dates are US Eastern; the user's own clock may be a day ahead.
        ref_date  = (pd.to_datetime(target_date) if target_date
                     else pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None))
        days_rest = int(np.clip((ref_date - last_game).days, 1, 10))

        avg_pts     = r10["PTS"].mean()
        std_pts     = max(r10["PTS"].std() if len(r10) > 1 else 6.0, 3.0)
        avg_pts_all = r10["PTS_ALLOWED"].mean()
        std_pts_all = max(r10["PTS_ALLOWED"].std() if len(r10) > 1 else 6.0, 3.0)

        return {
            "name":            full_name,
            "team_id":         team_id,
            "GAME_DATE":       str(ref_date.date()),
            # ── v5 model features ─────────────────────────────────────
            "ELO":             get_team_elo(full_name),
            "NET5":            float(r5["PLUS_MINUS"].mean()),
            "NET15":           float(r15["PLUS_MINUS"].mean()),
            "PTS10":           float(avg_pts),
            "PA10":            float(avg_pts_all),
            "EFG10":           float(efg_pct),
            "TOV10":           float(tov_rate),
            "ORB10":           float(oreb_pct),
            "FTR10":           float(ft_rate),
            "PACE10":          float(r10["POSS"].mean()),
            "WPCT10":          wpct10,
            "WIN_STREAK":      _streak(r10["WL"].tolist()),
            "DAYS_REST":       days_rest,
            "IS_B2B":          1 if days_rest <= 1 else 0,
            "HOME_WPCT":       _venue_wpct(df, home=True,  fallback=wpct10),
            "AWAY_WPCT":       _venue_wpct(df, home=False, fallback=wpct10),
            # ── simulation / display fields ───────────────────────────
            "PLUS_MINUS":      float(r10["PLUS_MINUS"].mean()),
            "REB":             float(r10["REB"].mean()),
            "eFG_PCT":         float(efg_pct),
            "AST_TOV":         float(ast_tov),
            "TS_PCT":          float(ts_pct),
            "OREB_PCT":        float(oreb_pct),
            "PTS":             float(avg_pts),
            "PTS_STD":         float(std_pts),
            "PTS_ALLOWED":     float(avg_pts_all),
            "PTS_ALLOWED_STD": float(std_pts_all),
        }
    except Exception:
        return None
