"""
Feature builder for features_v6: team game logs (leaguegamefinder format)
-> one row per team per game with pre-game features, then one row per game.

Everything is computed from strictly earlier games of the same team (shifted
before any rolling/EWMA/expanding step), so a game never sees its own result
or box score, and same-day games never see each other.  The only pre-game
facts taken from the game itself are its date, venue and who is home.

Building blocks
  * per-game possession-based ratings: ORTG / DRTG / NRTG per 100 possessions,
    four factors for and against;
  * across-season EWMAs (several spans) of those per-game values;
  * season-to-date means shrunk toward last season's final mean (regressed
    toward the league), weight k pseudo-games -> sensible in week one;
  * opponent-adjusted net rating (margin per 100 corrected by the opponent's
    pre-game rating and the venue);
  * schedule: rest buckets, games in last 4 / 7 days, 3-in-4, 4-in-6,
    road-trip / homestand length;
  * travel: km since the previous game, time-zone shift, venue altitude, from
    the static arena table below (2020 bubble and Toronto's 2020-21 Tampa
    season handled as venue overrides).
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

VALID_PREFIXES = ("002", "004", "005")

# abbreviation -> (lat, lon, UTC offset (standard time), altitude m)
ARENAS = {
    "ATL": (33.757, -84.396, -5, 320),   "BOS": (42.366, -71.062, -5, 5),
    "BKN": (40.683, -73.976, -5, 10),    "NJN": (40.733, -74.171, -5, 10),
    "CHA": (35.225, -80.839, -5, 230),   "CHI": (41.881, -87.674, -6, 180),
    "CLE": (41.496, -81.688, -5, 200),   "DAL": (32.790, -96.810, -6, 140),
    "DEN": (39.749, -105.008, -7, 1609), "DET": (42.341, -83.055, -5, 190),
    "GSW": (37.768, -122.388, -8, 5),    "HOU": (29.751, -95.362, -6, 15),
    "IND": (39.764, -86.156, -5, 220),   "LAC": (34.043, -118.267, -8, 90),
    "LAL": (34.043, -118.267, -8, 90),   "MEM": (35.138, -90.051, -6, 80),
    "MIA": (25.781, -80.188, -5, 2),     "MIL": (43.045, -87.917, -6, 190),
    "MIN": (44.979, -93.276, -6, 260),   "NOP": (29.949, -90.082, -6, 2),
    "NOH": (29.949, -90.082, -6, 2),     "NYK": (40.751, -73.993, -5, 10),
    "OKC": (35.463, -97.515, -6, 365),   "ORL": (28.539, -81.384, -5, 30),
    "PHI": (39.901, -75.172, -5, 5),     "PHX": (33.446, -112.071, -7, 330),
    "POR": (45.532, -122.667, -8, 15),   "SAC": (38.580, -121.500, -8, 10),
    "SAS": (29.427, -98.438, -6, 200),   "TOR": (43.643, -79.379, -5, 80),
    "UTA": (40.768, -111.901, -7, 1288), "WAS": (38.898, -77.021, -5, 10),
}
BUBBLE = (pd.Timestamp("2020-07-30"), pd.Timestamp("2020-10-31"))
BUBBLE_SITE = (28.370, -81.550, -5, 30)        # Lake Buena Vista, FL
TAMPA = (27.943, -82.452, -5, 5)               # Raptors' 2020-21 home games
HIGH_ALT = 1000                                # m; Denver, Salt Lake City


# ── per-game table ────────────────────────────────────────────────────────────

def team_games(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with opponent box stats, venue and per-game
    rates.  Sorted by team and date."""
    df = raw.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df = df[df["GAME_ID"].str[:3].isin(VALID_PREFIXES)]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    df["SEASON"] = df["SEASON_YEAR"]
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    df["POSS_OWN"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]
    box = ["PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB",
           "TOV", "POSS_OWN"]
    opp = df[["GAME_ID", "TEAM_ID"] + box].rename(
        columns={c: "O_" + c for c in ["TEAM_ID"] + box})
    df = df.merge(opp, on="GAME_ID")
    df = df[df["TEAM_ID"] != df["O_TEAM_ID"]].copy()

    df["WIN"] = (df["PTS"] > df["O_PTS"]).astype(float)
    df["MARGIN"] = (df["PTS"] - df["O_PTS"]).astype(float)
    df["POSS"] = (df["POSS_OWN"] + df["O_POSS_OWN"]) / 2
    df["PACE"] = df["POSS"] * 48 / df["MIN"].clip(lower=48)
    df["ORTG"] = 100 * df["PTS"] / df["POSS"]
    df["DRTG"] = 100 * df["O_PTS"] / df["POSS"]
    df["NRTG"] = df["ORTG"] - df["DRTG"]
    df["EFG"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"]
    df["O_EFG"] = (df["O_FGM"] + 0.5 * df["O_FG3M"]) / df["O_FGA"]
    df["TOVR"] = df["TOV"] / df["POSS_OWN"]
    df["O_TOVR"] = df["O_TOV"] / df["O_POSS_OWN"]
    df["ORBR"] = df["OREB"] / (df["OREB"] + df["O_DREB"])
    df["DRBR"] = df["DREB"] / (df["DREB"] + df["O_OREB"])
    df["FTR"] = df["FTM"] / df["FGA"]
    df["O_FTR"] = df["O_FTM"] / df["O_FGA"]
    df["F3R"] = df["FG3A"] / df["FGA"]

    # venue: home team's arena, with the 2020 bubble / Tampa overrides
    home_abbr = np.where(df["IS_HOME"] == 1, df["MATCHUP"].str[:3],
                         df["MATCHUP"].str[-3:])
    site = np.array([ARENAS[a] for a in home_abbr], dtype=float)
    bubble = ((df["SEASON"] == "2019-20") & (df["GAME_DATE"] >= BUBBLE[0])).values
    tampa = ((df["SEASON"] == "2020-21") & (home_abbr == "TOR"))
    site[bubble] = BUBBLE_SITE
    site[np.asarray(tampa)] = TAMPA
    df["LAT"], df["LON"], df["TZ"], df["ALT"] = site.T
    df["NEUTRAL"] = bubble.astype(int)
    own = np.array([ARENAS[a] for a in df["TEAM_ABBREVIATION"]], dtype=float)
    df["HOME_LAT"], df["HOME_LON"], df["HOME_TZ"], df["HOME_ALT"] = own.T

    return df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ── rolling primitives (all strictly pre-game) ────────────────────────────────

def ewm_pre(tg: pd.DataFrame, col: str, span: float) -> pd.Series:
    """EWMA over the team's earlier games, continuing across seasons."""
    return tg.groupby("TEAM_ID")[col].transform(
        lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())


def season_prior(tg: pd.DataFrame, col: str, regress: float) -> pd.Series:
    """Per row: last season's full mean of `col` for the team, regressed
    toward last season's league mean (first season: league mean)."""
    team = tg.groupby(["TEAM_ID", "SEASON"])[col].mean()
    league = tg.groupby("SEASON")[col].mean()
    seasons = sorted(league.index)
    prev = dict(zip(seasons[1:], seasons[:-1]))
    keys = list(zip(tg["TEAM_ID"], tg["SEASON"].map(prev)))
    last_team = pd.Series(team.reindex(keys).values, index=tg.index)
    last_league = tg["SEASON"].map(prev).map(league)
    last_league = last_league.fillna(tg["SEASON"].map(league))   # first season only
    return regress * last_team.fillna(last_league) + (1 - regress) * last_league


def shrunk_std(tg: pd.DataFrame, col: str, prior: pd.Series, k: float,
               decay: float = 1.0) -> pd.Series:
    """Season-to-date mean of `col` over the team's earlier games this season
    (game weights decay**age), shrunk toward `prior` with k pseudo-games."""
    x = tg[col].values.astype(float)
    pr = prior.values.astype(float)
    out = np.empty(len(tg))
    keys = (tg["TEAM_ID"].astype(str) + tg["SEASON"]).values
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1, len(tg)]
    for s, e in zip(starts[:-1], starts[1:]):
        num = lfilter([1.0], [1.0, -decay], x[s:e])      # inclusive weighted sums
        den = lfilter([1.0], [1.0, -decay], np.ones(e - s))
        num = np.r_[0.0, num[:-1]]                        # shift: earlier games only
        den = np.r_[0.0, den[:-1]]
        out[s:e] = (num + k * pr[s:e]) / (den + k)
    return pd.Series(out, index=tg.index)


def opp_lookup(tg: pd.DataFrame, col: str) -> pd.Series:
    """The opponent's value of a pre-game column in the same game."""
    m = tg.set_index(["GAME_ID", "TEAM_ID"])[col]
    return pd.Series(m.reindex(list(zip(tg["GAME_ID"], tg["O_TEAM_ID"]))).values,
                     index=tg.index)


# ── schedule & travel ─────────────────────────────────────────────────────────

def schedule_features(tg: pd.DataFrame) -> pd.DataFrame:
    """Rest, density, trip length, travel, time-zone shift, altitude."""
    g = tg.groupby(["TEAM_ID", "SEASON"])
    first = g.cumcount() == 0
    days = (tg["GAME_DATE"] - g["GAME_DATE"].shift(1)).dt.days
    out = pd.DataFrame(index=tg.index)
    out["DAYS"] = days.fillna(10).clip(1, 10)
    out["B2B"] = (out["DAYS"] == 1).astype(int)
    out["REST1"] = (out["DAYS"] == 2).astype(int)
    out["REST2"] = (out["DAYS"] == 3).astype(int)
    out["REST3P"] = (out["DAYS"] >= 4).astype(int)

    # games played in the previous n days (this game excluded)
    day = (tg["GAME_DATE"] - pd.Timestamp("2000-01-01")).dt.days.values
    team = tg["TEAM_ID"].values
    for n in (3, 5, 6):
        cnt = np.zeros(len(tg), dtype=int)
        for lag in range(1, 5):
            prev_day = np.r_[np.full(lag, -10**6), day[:-lag]]
            same = np.r_[np.zeros(lag, bool), team[lag:] == team[:-lag]]
            cnt += (same & (day - prev_day <= n)).astype(int)
        out[f"G{n}"] = cnt
    out["3IN4"] = (out["G3"] >= 2).astype(int)
    out["4IN6"] = (out["G5"] >= 3).astype(int)

    # consecutive games at the same venue type (this game included)
    run = (tg["IS_HOME"] != g["IS_HOME"].shift(1)) | first
    out["STREAK_VENUE"] = tg.groupby(run.cumsum()).cumcount() + 1
    out["ROAD_TRIP"] = np.where(tg["IS_HOME"] == 0, out["STREAK_VENUE"], 0)
    out["HOMESTAND"] = np.where(tg["IS_HOME"] == 1, out["STREAK_VENUE"], 0)

    # travel from the previous game's venue (season opener: from home)
    plat = g["LAT"].shift(1).where(~first, tg["HOME_LAT"])
    plon = g["LON"].shift(1).where(~first, tg["HOME_LON"])
    ptz = g["TZ"].shift(1).where(~first, tg["HOME_TZ"])
    out["TRAVEL"] = haversine_km(plat, plon, tg["LAT"], tg["LON"]) / 1000
    out["TZ_SHIFT"] = tg["TZ"] - ptz                   # + = moved east
    out["TZ_ABS"] = out["TZ_SHIFT"].abs()
    out["TRAVEL7"] = _window_sum(tg, out["TRAVEL"], 6)
    out["TZ_HOME"] = tg["TZ"] - tg["HOME_TZ"]          # body clock vs home
    out["ALT_VISIT"] = ((tg["ALT"] >= HIGH_ALT) & (tg["HOME_ALT"] < HIGH_ALT)).astype(int)
    return out


def _window_sum(tg: pd.DataFrame, v: pd.Series, n: int) -> pd.Series:
    """Sum of v over this game and the team's games in the previous n days."""
    s = pd.Series(v.values, index=tg["GAME_DATE"])
    res = s.groupby(tg["TEAM_ID"].values).rolling(f"{n + 1}D").sum()
    return pd.Series(res.values, index=tg.index)   # groupby keeps team/date order


# ── matchup rows ──────────────────────────────────────────────────────────────

def to_games(tg: pd.DataFrame, cols: list) -> pd.DataFrame:
    """One row per game: H_<col>, A_<col> for each team column."""
    keep = ["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "MARGIN", "WIN",
            "NEUTRAL"] + cols
    h = tg[tg["IS_HOME"] == 1][keep]
    a = tg[tg["IS_HOME"] == 0][["GAME_ID", "TEAM_ID"] + cols]
    h = h.rename(columns={c: "H_" + c for c in ["TEAM_ID"] + cols})
    a = a.rename(columns={c: "A_" + c for c in ["TEAM_ID"] + cols})
    games = h.merge(a, on="GAME_ID")
    games["RESULT"] = games.pop("WIN").astype(int)
    return games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
