"""
Standings helpers for the incentives variant.

  • CONFERENCE: static TEAM_ID -> 'E' / 'W' table (no team switched
    conference 2010-11 … 2023-24; NOH->NOP, NJN->BKN, CHA kept their ids);
  • era rules: playoff cut (top 8, or top 6 + play-in 7-10 from 2020-21),
    draft-lottery odds (flattened from the 2018-19 season / 2019 draft);
  • regular-season game table with each team's known schedule length;
  • pre-game standings for every regular-season game, from strictly earlier
    dates only (same-day games never see each other).
"""
import numpy as np
import pandas as pd

EAST = {
    1610612737: "ATL", 1610612738: "BOS", 1610612751: "BKN", 1610612766: "CHA",
    1610612741: "CHI", 1610612739: "CLE", 1610612765: "DET", 1610612754: "IND",
    1610612748: "MIA", 1610612749: "MIL", 1610612752: "NYK", 1610612753: "ORL",
    1610612755: "PHI", 1610612761: "TOR", 1610612764: "WAS",
}
WEST = {
    1610612742: "DAL", 1610612743: "DEN", 1610612744: "GSW", 1610612745: "HOU",
    1610612746: "LAC", 1610612747: "LAL", 1610612763: "MEM", 1610612750: "MIN",
    1610612740: "NOP", 1610612760: "OKC", 1610612756: "PHX", 1610612757: "POR",
    1610612758: "SAC", 1610612759: "SAS", 1610612762: "UTA",
}
CONFERENCE = {**{t: "E" for t in EAST}, **{t: "W" for t in WEST}}
TEAMS = sorted(CONFERENCE)                       # fixed index 0..29
TEAM_IX = {t: i for i, t in enumerate(TEAMS)}
IS_EAST = np.array([CONFERENCE[t] == "E" for t in TEAMS])

# Chance of the #1 pick by position in the reverse standings of the 14
# lottery teams (1 = worst record).
LOTTERY_OLD = np.array([25.0, 19.9, 15.6, 11.9, 8.8, 6.3, 4.3, 2.8, 1.7, 1.1,
                        0.8, 0.7, 0.6, 0.5]) / 100      # 2010-11 … 2017-18
LOTTERY_NEW = np.array([14.0, 14.0, 14.0, 12.5, 10.5, 9.0, 7.5, 6.0, 4.5, 3.0,
                        2.0, 1.5, 1.0, 0.5]) / 100      # 2018-19 season onward


def season_start_year(season: str) -> int:
    return int(season[:4])


def has_play_in(season: str) -> bool:
    """Seeds 7-10 go to a play-in from 2020-21 (2019-20 bubble rule ignored)."""
    return season_start_year(season) >= 2020


def lottery_odds(season: str) -> np.ndarray:
    return LOTTERY_NEW if season_start_year(season) >= 2018 else LOTTERY_OLD


def postseason_weight(seed: np.ndarray, season: str) -> np.ndarray:
    """Chance of reaching the playoff bracket from a final conference seed.
    Play-in era: 7-8 need one win in two tries (~0.75), 9-10 two in a row
    (~0.25)."""
    if not has_play_in(season):
        return (seed <= 8).astype(float)
    return np.select([seed <= 6, seed <= 8, seed <= 10], [1.0, 0.75, 0.25], 0.0)


# ── games ─────────────────────────────────────────────────────────────────────

def regular_season_games(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per regular-season game (date-sorted): GAME_ID, GAME_DATE,
    SEASON, H, A (TEAM_IDs), HI, AI (team index), RESULT (1 = home win)."""
    df = raw.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df = df[df["GAME_ID"].str[:3] == "002"]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    home = df["MATCHUP"].str.contains("vs.", regex=False)
    h = df[home][["GAME_ID", "GAME_DATE", "SEASON_YEAR", "TEAM_ID", "WL"]]
    a = df[~home][["GAME_ID", "TEAM_ID"]]
    g = h.merge(a, on="GAME_ID", suffixes=("_H", "_A"))
    g = g.rename(columns={"SEASON_YEAR": "SEASON", "TEAM_ID_H": "H", "TEAM_ID_A": "A"})
    g["RESULT"] = (g["WL"] == "W").astype(int)
    g["HI"] = g["H"].map(TEAM_IX)
    g["AI"] = g["A"].map(TEAM_IX)
    assert g[["HI", "AI"]].notna().all().all(), "team missing from CONFERENCE"
    g = g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return g[["GAME_ID", "GAME_DATE", "SEASON", "H", "A", "HI", "AI", "RESULT"]]


def conference_ranks(wpct: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Seed (1 = best) of every team inside its conference, for arrays of
    shape (..., 30).  Ties broken by `tiebreak` (higher wins)."""
    key = wpct + 1e-7 * tiebreak
    seed = np.zeros(key.shape, dtype=np.int16)
    for conf in (IS_EAST, ~IS_EAST):
        k = key[..., conf]
        seed[..., conf] = np.argsort(np.argsort(-k, axis=-1), axis=-1) + 1
    return seed


# ── pre-game standings ────────────────────────────────────────────────────────

def pregame_standings(games: pd.DataFrame) -> pd.DataFrame:
    """Per regular-season game and side: W, L, games remaining, conference
    rank and games back of the 1st / 6th / 8th / 10th place team, all from
    games on strictly earlier dates.  Columns are suffixed _H / _A."""
    rows = []
    for season, g in games.groupby("SEASON", sort=False):
        total = np.bincount(g["HI"], minlength=30) + np.bincount(g["AI"], minlength=30)
        W = np.zeros(30)
        L = np.zeros(30)
        dates = g["GAME_DATE"].values
        HI, AI, res = g["HI"].values, g["AI"].values, g["RESULT"].values
        starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(g)]
        out = np.zeros((len(g), 2, 8))
        for s, e in zip(starts[:-1], starts[1:]):
            gp = W + L
            wpct = np.where(gp > 0, W / np.maximum(gp, 1), 0.5)
            seed = conference_ranks(wpct, -np.arange(30.0))
            feats = np.zeros((30, 8))
            for conf in (IS_EAST, ~IS_EAST):
                idx = np.flatnonzero(conf)
                by_seed = idx[np.argsort(seed[idx])]
                for j, place in enumerate((1, 6, 8, 10)):
                    k = by_seed[place - 1]
                    feats[idx, 4 + j] = ((W[k] - W[idx]) + (L[idx] - L[k])) / 2
            feats[:, 0], feats[:, 1] = W, L
            feats[:, 2] = total - gp
            feats[:, 3] = seed
            out[s:e, 0] = feats[HI[s:e]]
            out[s:e, 1] = feats[AI[s:e]]
            np.add.at(W, HI[s:e], res[s:e])
            np.add.at(L, HI[s:e], 1 - res[s:e])
            np.add.at(W, AI[s:e], 1 - res[s:e])
            np.add.at(L, AI[s:e], res[s:e])
        names = ["W", "L", "GREM", "CRANK", "GB1", "GB6", "GB8", "GB10"]
        df = pd.DataFrame({f"{n}_{side}": out[:, k, j]
                           for k, side in enumerate("HA") for j, n in enumerate(names)})
        df.insert(0, "GAME_ID", g["GAME_ID"].values)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)
