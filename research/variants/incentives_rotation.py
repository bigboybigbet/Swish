"""
Who-plays control for the incentives variant.

For every team-game: the team's top-5 players by minutes over its previous
10 games of the same season (earlier games only), and the share of those 5
who do not play tonight (MISS5).  MISSW is the minutes-weighted version over
the top 8.  Knowing who plays is legitimate pre-game information (inactive
lists are public before tip-off; live: the injury report / starting lineups).
"""
import os

import numpy as np
import pandas as pd

from research.benchmark import CACHE_DIR, DEFAULT_DATA_DIR

BOX_FILES = ["regular_season_box_scores_2010_2024_part_1.csv",
             "regular_season_box_scores_2010_2024_part_2.csv",
             "regular_season_box_scores_2010_2024_part_3.csv",
             "play_off_box_scores_2010_2024.csv"]
WINDOW, TOP, TOP_W, MIN_HISTORY = 10, 5, 8, 3
VERSION = 1


def load_minutes(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """GAME_ID, TEAM_ID, PID, MIN (0 for DNP / inactive)."""
    parts = []
    for fn in BOX_FILES:
        p = os.path.join(data_dir, fn)
        if os.path.exists(p):
            parts.append(pd.read_csv(p, usecols=["gameId", "teamId", "personId", "minutes"]))
    b = pd.concat(parts, ignore_index=True)
    b["GAME_ID"] = b["gameId"].astype(str).str.zfill(10)
    b = b.drop_duplicates(["GAME_ID", "personId"])
    mm = b["minutes"].astype("string").str.split(":", expand=True)
    mins = (mm[0].astype(float) + mm[1].astype(float) / 60).fillna(0.0)
    return pd.DataFrame({"GAME_ID": b["GAME_ID"].values, "TEAM_ID": b["teamId"].astype(int).values,
                         "PID": b["personId"].astype(int).values, "MIN": mins.values})


def rotation_control(raw: pd.DataFrame, data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """GAME_ID, TEAM_ID, MISS5, MISSW, HAS_BOX for every team-game (cached)."""
    path = os.path.join(CACHE_DIR, f"incentives_rotation_v{VERSION}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    logs = raw[raw["GAME_ID"].str[:3].isin(("002", "004", "005"))]
    logs = logs[["GAME_ID", "TEAM_ID", "SEASON_YEAR", "GAME_DATE"]].copy()
    box = dict(tuple(load_minutes(data_dir).groupby("TEAM_ID")))
    out = []
    for (team, season), tg in logs.groupby(["TEAM_ID", "SEASON_YEAR"]):
        tg = tg.sort_values(["GAME_DATE", "GAME_ID"])
        gb = box[team][box[team]["GAME_ID"].isin(tg["GAME_ID"])]
        M = (gb.pivot_table(index="GAME_ID", columns="PID", values="MIN", aggfunc="sum")
               .reindex(tg["GAME_ID"]).fillna(0.0).values)
        has_box = tg["GAME_ID"].isin(gb["GAME_ID"]).values
        csum = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(M, axis=0)])
        n = len(tg)
        miss5 = np.zeros(n)
        missw = np.zeros(n)
        for k in range(n):
            lo = max(0, k - WINDOW)
            if k - lo < MIN_HISTORY or not has_box[k]:
                continue
            recent = csum[k] - csum[lo]
            order = np.argsort(-recent)
            top5, top8 = order[:TOP], order[:TOP_W]
            played = M[k] > 0
            miss5[k] = np.mean(~played[top5])
            w = recent[top8]
            missw[k] = np.sum(w * ~played[top8]) / max(w.sum(), 1e-9)
        out.append(pd.DataFrame({"GAME_ID": tg["GAME_ID"].values, "TEAM_ID": team,
                                 "MISS5": miss5, "MISSW": missw, "HAS_BOX": has_box}))
    res = pd.concat(out, ignore_index=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    res.to_parquet(path, index=False)
    return res
