"""
Pre-game features for the margin_model variant.

Everything here only looks at games played strictly before the game being
predicted, except the dressed/inactive list of the game itself, which is
public before tip-off.
"""
import os

import numpy as np
import pandas as pd

# Regular season (002), playoffs (004), play-in (005) — same as production.
VALID_PREFIXES = ("002", "004", "005")

SRS_HALFLIFE = 120     # days
SRS_PREV_W   = 0.75    # weight per season back
SRS_RIDGE    = 4.0     # shrinks team ratings toward 0
HCA_PRIOR    = 2.5     # prior home edge in points (weight of one game)
VALUE_HALFLIFE = 40    # player value: EWMA half-life in dressed games

BOX_FILES = [f"regular_season_box_scores_2010_2024_part_{i}.csv" for i in (1, 2, 3)] \
    + ["play_off_box_scores_2010_2024.csv"]
INACTIVE = r"^(DND|NWT|Did Not Dress|Not With Team|Not with team|Inactive)"


# ── games ─────────────────────────────────────────────────────────────────────

def game_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Team game logs → one row per game: ids, date, season, home/away team, margin."""
    df = raw.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df = df[df["GAME_ID"].str[:3].isin(VALID_PREFIXES)]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    home_mask = df["MATCHUP"].str.contains("vs.", regex=False)
    keep = ["GAME_ID", "GAME_DATE", "SEASON_YEAR", "TEAM_ID", "PTS"]
    g = df[home_mask][keep].merge(df[~home_mask][["GAME_ID", "TEAM_ID", "PTS"]],
                                  on="GAME_ID", suffixes=("_H", "_A"))
    g = g.dropna(subset=["PTS_H", "PTS_A"])
    g = g.rename(columns={"SEASON_YEAR": "SEASON", "TEAM_ID_H": "HOME", "TEAM_ID_A": "AWAY"})
    g["MARGIN"] = (g["PTS_H"] - g["PTS_A"]).astype(float)
    g["RESULT"] = (g["MARGIN"] > 0).astype(int)
    g = g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return g[["GAME_ID", "GAME_DATE", "SEASON", "HOME", "AWAY", "MARGIN", "RESULT"]]


# ── team ratings ──────────────────────────────────────────────────────────────

def srs_ratings(g: pd.DataFrame, halflife=SRS_HALFLIFE, prev_w=SRS_PREV_W,
                lam=SRS_RIDGE) -> pd.DataFrame:
    """
    Opponent-adjusted margin ratings, re-solved every game day from earlier days:
        margin ≈ R_home − R_away + HCA
    by weighted ridge least squares; a past game weighs
    0.5 ** (age_days / halflife) * prev_w ** (seasons back).
    """
    teams = np.unique(np.r_[g["HOME"].values, g["AWAY"].values])
    n = len(teams)
    tix = {t: i for i, t in enumerate(teams)}
    hi, ai = g["HOME"].map(tix).values, g["AWAY"].map(tix).values
    X = np.zeros((len(g), n + 1))
    X[np.arange(len(g)), hi] = 1.0
    X[np.arange(len(g)), ai] = -1.0
    X[:, n] = 1.0
    y = g["MARGIN"].values
    day = (g["GAME_DATE"] - g["GAME_DATE"].min()).dt.days.values
    season = pd.factorize(g["SEASON"], sort=True)[0]

    prior = np.full(n + 1, lam)
    prior[n] = 1.0
    out = np.zeros((len(g), 3))
    for d in np.unique(day):
        today = day == d
        past = np.where((day < d) & (day > d - 6 * halflife))[0]
        w = 0.5 ** ((d - day[past]) / halflife) * prev_w ** (season[today][0] - season[past])
        Xw = X[past] * w[:, None]
        A = X[past].T @ Xw + np.diag(prior)
        b = Xw.T @ y[past]
        b[n] += HCA_PRIOR
        beta = np.linalg.solve(A, b)
        out[today] = np.c_[beta[hi[today]], beta[ai[today]], np.full(today.sum(), beta[n])]
    return pd.DataFrame(out, columns=["SRS_H", "SRS_A", "SRS_HCA"], index=g.index)


# ── rest / schedule ───────────────────────────────────────────────────────────

def rest_features(g: pd.DataFrame) -> pd.DataFrame:
    """Back-to-backs, rest-day difference, games in the previous 3 and 6 days."""
    long = pd.concat([
        pd.DataFrame({"GAME_ID": g["GAME_ID"], "TEAM": g["HOME"], "D": g["GAME_DATE"], "SIDE": "H"}),
        pd.DataFrame({"GAME_ID": g["GAME_ID"], "TEAM": g["AWAY"], "D": g["GAME_DATE"], "SIDE": "A"}),
    ]).sort_values(["TEAM", "D"]).reset_index(drop=True)
    long["DAY"] = (long["D"] - pd.Timestamp("2000-01-01")).dt.days
    long["REST"] = long.groupby("TEAM")["DAY"].diff().fillna(7).clip(1, 7)
    for w in (4, 7):
        long[f"G{w}"] = 0
        for _, idx in long.groupby("TEAM").groups.items():
            d = long.loc[idx, "DAY"].values
            long.loc[idx, f"G{w}"] = np.arange(len(d)) - np.searchsorted(d, d - (w - 1))
    wide = long.pivot(index="GAME_ID", columns="SIDE", values=["REST", "G4", "G7"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    out = pd.DataFrame({
        "B2B_H":  (wide["REST_H"] == 1).astype(int),
        "B2B_A":  (wide["REST_A"] == 1).astype(int),
        "REST_D": wide["REST_H"].clip(1, 4) - wide["REST_A"].clip(1, 4),
        "G4_D":   wide["G4_H"] - wide["G4_A"],
        "G7_D":   wide["G7_H"] - wide["G7_A"],
    })
    return out.reindex(g["GAME_ID"].values).reset_index(drop=True)


# ── roster strength ───────────────────────────────────────────────────────────

def game_score(df: pd.DataFrame) -> pd.Series:
    """Hollinger game score from box-score columns (leaguedashplayerstats names)."""
    return (df["PTS"] + 0.4 * df["FGM"] - 0.7 * df["FGA"] - 0.4 * (df["FTA"] - df["FTM"])
            + 0.7 * df["OREB"] + 0.3 * df["DREB"] + df["STL"] + 0.7 * df["AST"]
            + 0.7 * df["BLK"] - 0.4 * df["PF"] - df["TOV"])


def load_box_scores(data_dir: str) -> pd.DataFrame:
    cols = {"gameId": "GAME_ID", "game_date": "GAME_DATE", "teamId": "TEAM_ID",
            "personId": "PLAYER_ID", "comment": "COMMENT", "points": "PTS",
            "fieldGoalsMade": "FGM", "fieldGoalsAttempted": "FGA",
            "freeThrowsMade": "FTM", "freeThrowsAttempted": "FTA",
            "reboundsOffensive": "OREB", "reboundsDefensive": "DREB", "assists": "AST",
            "steals": "STL", "blocks": "BLK", "turnovers": "TOV", "foulsPersonal": "PF"}
    b = pd.concat([pd.read_csv(os.path.join(data_dir, f), usecols=list(cols))
                   for f in BOX_FILES], ignore_index=True).rename(columns=cols)
    b["GAME_ID"] = b["GAME_ID"].astype(str).str.zfill(10)
    b["GAME_DATE"] = pd.to_datetime(b["GAME_DATE"])
    return b.drop_duplicates(["GAME_ID", "PLAYER_ID"])


def team_strength(box: pd.DataFrame, halflife=VALUE_HALFLIFE) -> pd.DataFrame:
    """
    Per (game, team): STR = summed value of dressed players, MISS = summed value
    of players listed inactive (DND / NWT).  A player's value is the EWMA of his
    game score over his earlier dressed games (0 when he dressed but sat), taken
    as of his previous game — the game itself never enters its own feature.
    Dressed/inactive comes from the inactive codes only, not from minutes.
    """
    b = box.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    b["DRESSED"] = ~b["COMMENT"].fillna("").str.match(INACTIVE)
    b["GMSC"] = game_score(b)
    d = b[b["DRESSED"]]
    b["POST"] = np.nan
    b.loc[d.index, "POST"] = d.groupby("PLAYER_ID")["GMSC"].transform(
        lambda x: x.ewm(halflife=halflife).mean())
    b["V"] = b.groupby("PLAYER_ID")["POST"].transform(lambda x: x.shift(1).ffill()).fillna(0.0)
    b["V_DRESSED"] = b["V"].where(b["DRESSED"], 0.0)
    b["V_INACTIVE"] = b["V"].where(~b["DRESSED"], 0.0)
    return (b.groupby(["GAME_ID", "TEAM_ID"])[["V_DRESSED", "V_INACTIVE"]].sum()
            .rename(columns={"V_DRESSED": "STR", "V_INACTIVE": "MISS"}).reset_index())


def strength_features(g: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    s = team_strength(box)
    h = g[["GAME_ID", "HOME"]].merge(s, left_on=["GAME_ID", "HOME"],
                                     right_on=["GAME_ID", "TEAM_ID"], how="left")
    a = g[["GAME_ID", "AWAY"]].merge(s, left_on=["GAME_ID", "AWAY"],
                                     right_on=["GAME_ID", "TEAM_ID"], how="left")
    return pd.DataFrame({"STR_DIFF":  (h["STR"] - a["STR"]).values,
                         "MISS_DIFF": (h["MISS"] - a["MISS"]).values})


# ── live ──────────────────────────────────────────────────────────────────────

def live_team_strength(player_stats: pd.DataFrame, injured: list) -> tuple:
    """
    (STR, MISS) for one team before an upcoming game.

    player_stats: leaguedashplayerstats rows of the team's current players
    (PerMode=PerGame, last_n_games≈40 to mimic the EWMA); game score per game
    is scaled by GP / team games so games sat out count as 0, like the
    historical value.  injured: data.injuries.get_injury_report(team) — each
    player's value moves from STR to MISS by config.STATUS_WEIGHT[status].
    """
    from config import STATUS_WEIGHT
    ps = player_stats.copy()
    ps["V"] = game_score(ps) * ps["GP"] / max(float(ps["GP"].max()), 1.0)
    status = {p["name"]: p["status"] for p in injured}
    out_w = ps["PLAYER_NAME"].map(lambda n: STATUS_WEIGHT.get(status.get(n, ""), 0.0))
    return float((ps["V"] * (1 - out_w)).sum()), float((ps["V"] * out_w).sum())
