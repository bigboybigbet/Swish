"""
Roster-availability feature for features_v6, from player box scores.

For every team game:  AVAIL = sum_i v_i * (a_i - s_i)  over the team's players
this season, where
  a_i  1 if player i is available tonight: dressed and either played or was a
       coach's-decision DNP (game-state independent, i.e. the pre-game active
       list; injured / rest / not-with-team / not listed count as out),
  s_i  share of the team's earlier games this season he was available for,
  v_i  his pre-game value: per-game Game Score (Hollinger) over his earlier
       games, exponentially weighted and shrunk toward a bench-level prior.
So AVAIL is how much stronger tonight's available roster is than the average
roster behind the team's season-to-date rating (negative = key players out,
positive = stars back / newly acquired).  Only earlier games feed v_i and s_i.

Live: v_i from nba_api leaguedashplayerstats per-game averages, s_i = GP with
the team / team GP from the same endpoint, a_i = 1 - P(out) from the ESPN
injury report status (out 1, doubtful .8, questionable .3, day-to-day .2).
"""
import os

import numpy as np
import pandas as pd
from scipy.signal import lfilter

BOX_FILES = ["regular_season_box_scores_2010_2024_part_1.csv",
             "regular_season_box_scores_2010_2024_part_2.csv",
             "regular_season_box_scores_2010_2024_part_3.csv",
             "play_off_box_scores_2010_2024.csv"]


def load_box(data_dir: str) -> pd.DataFrame:
    """Player box scores -> one row per listed player per game."""
    cols = ["season_year", "game_date", "gameId", "teamId", "personId", "comment",
            "minutes", "fieldGoalsMade", "fieldGoalsAttempted", "freeThrowsMade",
            "freeThrowsAttempted", "reboundsOffensive", "reboundsDefensive",
            "assists", "steals", "blocks", "turnovers", "foulsPersonal", "points"]
    parts = [pd.read_csv(os.path.join(data_dir, f), usecols=cols)
             for f in BOX_FILES if os.path.exists(os.path.join(data_dir, f))]
    b = pd.concat(parts, ignore_index=True)
    b["GAME_ID"] = b["gameId"].astype(str).str.zfill(10)
    b["GAME_DATE"] = pd.to_datetime(b["game_date"]).dt.normalize()
    mins = b["minutes"].fillna("0:00").astype(str).str.split(":", expand=True)
    b["MIN"] = pd.to_numeric(mins[0], errors="coerce").fillna(0) + \
        pd.to_numeric(mins[1], errors="coerce").fillna(0) / 60
    b["PLAYED"] = b["MIN"] > 0
    coach = b["comment"].fillna("").str.contains("Coach", case=False)
    b["AVAIL"] = (b["PLAYED"] | coach).astype(float)
    b["GMSC"] = (b["points"] + 0.4 * b["fieldGoalsMade"] - 0.7 * b["fieldGoalsAttempted"]
                 - 0.4 * (b["freeThrowsAttempted"] - b["freeThrowsMade"])
                 + 0.7 * b["reboundsOffensive"] + 0.3 * b["reboundsDefensive"]
                 + b["steals"] + 0.7 * b["assists"] + 0.7 * b["blocks"]
                 - 0.4 * b["foulsPersonal"] - b["turnovers"])
    b = b.rename(columns={"teamId": "TEAM_ID", "personId": "PLAYER_ID",
                          "season_year": "SEASON"})
    b = b.drop_duplicates(["GAME_ID", "PLAYER_ID"])
    return b[["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "PLAYER_ID",
              "MIN", "PLAYED", "AVAIL", "GMSC"]]


def player_values(box: pd.DataFrame, decay: float, k0: float,
                  prior: float) -> pd.DataFrame:
    """Post-game value after each played game: decayed mean Game Score per
    game, shrunk toward `prior` with k0 pseudo-games.  Rows: PLAYER_ID,
    GAME_DATE, VALUE (usable for games strictly after GAME_DATE)."""
    p = box[box["PLAYED"]].sort_values(["PLAYER_ID", "GAME_DATE"])
    x = p["GMSC"].values
    out = np.empty(len(p))
    ids = p["PLAYER_ID"].values
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(p)]
    for s, e in zip(starts[:-1], starts[1:]):
        num = lfilter([1.0], [1.0, -decay], x[s:e])
        den = lfilter([1.0], [1.0, -decay], np.ones(e - s))
        out[s:e] = (num + k0 * prior) / (den + k0)
    return pd.DataFrame({"PLAYER_ID": ids, "GAME_DATE": p["GAME_DATE"].values,
                         "VALUE": out})


def availability(box: pd.DataFrame, games: pd.DataFrame, decay: float = 0.97,
                 k0: float = 5.0, prior: float = 0.0, k: float = 10.0) -> pd.DataFrame:
    """AVAIL per (GAME_ID, TEAM_ID) for the team games in `games`
    (columns GAME_ID, GAME_DATE, SEASON, TEAM_ID).  The share s_i is shrunk
    toward last season's share with k pseudo-games, mirroring the shrunk
    team rating, so offseason arrivals / departures count from game one."""
    tgames = games[["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID"]].drop_duplicates()
    seasons = sorted(tgames["SEASON"].unique())
    nxt = dict(zip(seasons[:-1], seasons[1:]))

    # last season's share of team games each player was available for
    tb = box.merge(tgames[["GAME_ID", "TEAM_ID"]], on=["GAME_ID", "TEAM_ID"])
    n_team = tgames.groupby(["SEASON", "TEAM_ID"]).size().rename("N")
    s0 = tb.groupby(["SEASON", "TEAM_ID", "PLAYER_ID"])["AVAIL"].sum().reset_index()
    s0 = s0.merge(n_team.reset_index(), on=["SEASON", "TEAM_ID"])
    s0["S0"] = s0["AVAIL"] / s0["N"]
    s0["SEASON"] = s0["SEASON"].map(nxt)
    s0 = s0.dropna(subset=["SEASON"])[["SEASON", "TEAM_ID", "PLAYER_ID", "S0"]]

    roster = pd.concat([tb[["SEASON", "TEAM_ID", "PLAYER_ID"]],
                        s0[["SEASON", "TEAM_ID", "PLAYER_ID"]]]).drop_duplicates()
    pairs = tgames.merge(roster, on=["SEASON", "TEAM_ID"])
    pairs = pairs.merge(tb[["GAME_ID", "TEAM_ID", "PLAYER_ID", "AVAIL"]],
                        on=["GAME_ID", "TEAM_ID", "PLAYER_ID"], how="left")
    pairs = pairs.merge(s0, on=["SEASON", "TEAM_ID", "PLAYER_ID"], how="left")
    pairs[["AVAIL", "S0"]] = pairs[["AVAIL", "S0"]].fillna(0.0)

    # share of the team's earlier games this season, shrunk toward S0
    pairs = pairs.sort_values(["TEAM_ID", "PLAYER_ID", "GAME_DATE"])
    grp = pairs.groupby(["TEAM_ID", "SEASON", "PLAYER_ID"])["AVAIL"]
    n_before = grp.cumcount()
    pairs["SHARE"] = (grp.cumsum() - pairs["AVAIL"] + k * pairs["S0"]) / (n_before + k)

    # pre-game player value: last post-game value strictly before tonight
    vals = player_values(box, decay, k0, prior).sort_values("GAME_DATE")
    pairs = pairs.sort_values("GAME_DATE")
    pairs = pd.merge_asof(pairs, vals, on="GAME_DATE", by="PLAYER_ID",
                          allow_exact_matches=False)
    pairs["VALUE"] = pairs["VALUE"].fillna(prior)
    pairs["CONTRIB"] = pairs["VALUE"] * (pairs["AVAIL"] - pairs["SHARE"])
    out = pairs.groupby(["GAME_ID", "TEAM_ID"])["CONTRIB"].sum().rename("AVAIL")
    return out.reset_index()
