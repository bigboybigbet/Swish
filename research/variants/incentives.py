"""
incentives — does what a team has to play for move the result beyond its
rating?

Base signal: the adj_ratings variant (decayed ridge ratings, ADJ_PRED + back-
to-back flags, holdout logloss 0.6277), taken unchanged from its cache.

Added per regular-season game (incentives_standings / incentives_sim, all
pre-game: results before the game's date only, future fixtures allowed):
  • standings: W-L, conference rank, games back of 1st / 6th / 8th / 10th,
    games remaining;
  • a 2,000-run rest-of-season Monte Carlo from pre-game ratings: P(post-
    season, era-aware: top 8, or top 6 + play-in 7-10 from 2020-21),
    P(top 1 / top 4 / bottom 3), expected seed, expected #1-pick odds
    (old odds table up to 2017-18, flattened table from 2018-19);
  • leverage = P(goal | win tonight) - P(goal | lose tonight), a tank
    incentive (#1-pick odds gained by losing), clinched / eliminated /
    seed-locked flags.
Who-plays control (incentives_rotation): share of the team's top-5 recent-
minutes players who sit tonight.

fit_predict: logistic regression (standardised, L2 C=1) on the base
features plus home-minus-away incentive features (all zero in playoffs).

Chosen on the dev window (2014-15..2016-17): a single feature,
D_STAKE_LATE = (stake_home - stake_away) when either team has <= 20 games
left, else 0, where stake = P(postseason | W) - P(postseason | L) +
same for top-4 seed + same for the 1 seed.  Full-season stake, the tank
incentive and the rest / dead flags added nothing on dev (FULL_INCENTIVE is
reported as a pre-declared secondary set, not selected).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research.variants import adj_ratings
from research.variants.incentives_rotation import rotation_control
from research.variants.incentives_sim import season_features
from research.variants.incentives_standings import pregame_standings, regular_season_games

BASE = ["ADJ_PRED", "H_B2B", "A_B2B"]
WHO = ["D_MISS5"]
INCENTIVE = ["D_STAKE_LATE"]
FULL_INCENTIVE = ["D_STAKE_LATE", "D_TANK_LATE", "D_REST"]
FEATURES = BASE + INCENTIVE
C = 1.0
LATE_GAMES = 20          # "late season": team has at most this many games left
NO_STAKE = 0.01          # stake below this = nothing left to play for


def team_features(df: pd.DataFrame, s: str) -> pd.DataFrame:
    """Derived per-team incentive measures for side s ('H' or 'A')."""
    stake = df[f"LEV_POST_{s}"] + df[f"LEV_TOP4_{s}"] + df[f"LEV_TOP1_{s}"]
    tank = -100 * df[f"LEV_ODDS_{s}"]                 # pp of #1-pick odds gained by losing
    late = df[f"GREM_{s}"] <= LATE_GAMES
    safe = df[f"P_POST_{s}"] >= 0.99                  # playoff spot (virtually) sure
    rest = late & safe & (stake < NO_STAKE)           # seed settled: rest starters
    return pd.DataFrame({
        f"STAKE_{s}": stake, f"TANK_{s}": tank, f"LATE_{s}": late.astype(float),
        f"REST_{s}": rest.astype(float),
        f"DEAD_{s}": (late & ~safe & (stake < NO_STAKE)).astype(float),
    })


def build_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    base = adj_ratings.build_dataset(raw)
    rs = regular_season_games(raw)
    st = pregame_standings(rs)
    mc = season_features(raw)
    df = base.merge(st, on="GAME_ID", how="left").merge(mc, on="GAME_ID", how="left")
    df = df.merge(rs[["GAME_ID", "H", "A"]], on="GAME_ID", how="left")

    rot = rotation_control(raw)[["GAME_ID", "TEAM_ID", "MISS5", "MISSW"]]
    teams = adj_ratings.game_table(raw)[["GAME_ID", "H", "A"]].rename(columns={"H": "TH", "A": "TA"})
    df = df.merge(teams, on="GAME_ID", how="left")
    for s, col in (("H", "TH"), ("A", "TA")):
        r = rot.rename(columns={"TEAM_ID": col, "MISS5": f"MISS5_{s}", "MISSW": f"MISSW_{s}"})
        df = df.merge(r, on=["GAME_ID", col], how="left")

    df["IS_RS"] = (df["GAME_ID"].str[:3] == "002").astype(int)
    df = pd.concat([df, team_features(df, "H"), team_features(df, "A")], axis=1)
    for f in ("STAKE", "TANK", "REST", "DEAD", "MISS5", "MISSW"):
        df[f"D_{f}"] = (df[f"{f}_H"] - df[f"{f}_A"]).fillna(0.0)
    late = ((df["LATE_H"] + df["LATE_A"]) > 0).astype(float)
    df["D_STAKE_LATE"] = df["D_STAKE"] * late
    df["D_TANK_LATE"] = df["D_TANK"] * late
    df["D_POST_LATE"] = (df["LEV_POST_H"] - df["LEV_POST_A"]).fillna(0.0) * late
    return df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def make_fit_predict(features, c: float = C):
    """fit_predict for a given feature list (standardised L2 logistic)."""
    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        sc = StandardScaler().fit(train[features].values)
        lr = LogisticRegression(C=c, max_iter=2000)
        lr.fit(sc.transform(train[features].values), train["RESULT"].values)
        return lr.predict_proba(sc.transform(test[features].values))[:, 1]
    return fit_predict


fit_predict = make_fit_predict(FEATURES)
