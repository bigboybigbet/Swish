"""
Subset report for the incentives variant.

    python -m research.variants.incentives_eval --seasons 2014-15,2015-16,2016-17
    python -m research.variants.incentives_eval            # holdout (run once)

Walk-forward (benchmark.walk_forward, 3 train seasons) for four models on
the same games: base, base + incentives, base + who-plays, base + who-plays
+ incentives.  Prints overall and subset log loss with paired differences,
and the seeding-manipulation / tanking outcome checks.
"""
import argparse

import numpy as np
import pandas as pd

from research.benchmark import (TEST_SEASONS, attach_season_if_missing, load_history,
                                per_game_logloss, walk_forward)
from research.variants import incentives as inc

LEV_ZERO = 0.005        # |P(postseason | W) - P(postseason | L)| below this
MANIP_DAYS = 5          # seeding manipulation: last N days of the season
MANIP_GAIN = 0.02       # bracket value (exp. series wins) gained by losing


def model_sets(incentive=None):
    i = incentive or inc.INCENTIVE
    return {"base": inc.BASE, "base+inc": inc.BASE + i,
            "base+who": inc.BASE + inc.WHO, "base+who+inc": inc.BASE + inc.WHO + i}


def predict_all(df, seasons, sets, c=inc.C):
    out = None
    for name, feats in sets.items():
        p = walk_forward(df, inc.make_fit_predict(feats, c), seasons)
        p = p.rename(columns={"P": name})
        out = p if out is None else out.merge(p[["GAME_ID", name]], on="GAME_ID")
    return out.merge(df.drop(columns=["SEASON", "RESULT"]), on="GAME_ID")


def subset_masks(t: pd.DataFrame) -> dict:
    rs = t["IS_RS"] == 1
    last_date = t["SEASON"].map(t[rs].groupby("SEASON")["GAME_DATE"].max())
    rank = t[rs].groupby("SEASON")["GAME_DATE"].rank(method="first", ascending=False)
    grem = np.minimum(t["GREM_H"], t["GREM_A"])
    zero = lambda s: ((t[f"LEV_POST_{s}"].abs() < LEV_ZERO)
                      & ((t[f"P_POST_{s}"] >= 0.995) | (t[f"P_POST_{s}"] <= 0.005)))
    return {
        "all": pd.Series(True, index=t.index),
        "regular season (playoffs excluded)": rs,
        "last 20 RS games/season (league, by date)": rs & (rank.reindex(t.index) <= 20),
        "either team in its last 20 RS games": rs & (grem <= 20),
        "last 10 days of RS": rs & (t["GAME_DATE"] > last_date - pd.Timedelta(days=10)),
        "either team playoff leverage ~0": rs & (zero("H") | zero("A")),
        "both teams playoff leverage ~0": rs & zero("H") & zero("A"),
    }


def compare(t, mask, a, b):
    y = t.loc[mask, "RESULT"].values
    la, lb = per_game_logloss(t.loc[mask, a].values, y), per_game_logloss(t.loc[mask, b].values, y)
    d = lb - la
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return {"n": int(mask.sum()), f"ll_{a}": la.mean(), f"ll_{b}": lb.mean(),
            "d": d.mean(), "se": se, "win": bool(d.mean() < -2 * se)}


def subset_table(t):
    rows = []
    for name, m in subset_masks(t).items():
        for a, b in (("base", "base+inc"), ("base+who", "base+who+inc"), ("base", "base+who")):
            r = compare(t, m, a, b)
            rows.append({"subset": name, "pair": f"{b} vs {a}", "n": r["n"],
                         "ll_a": r[f"ll_{a}"], "ll_b": r[f"ll_{b}"],
                         "d": r["d"], "se": r["se"], "win": r["win"]})
    return pd.DataFrame(rows)


def side_rows(t):
    """One row per team-game in the regular season (team's view)."""
    parts = []
    for s, o in (("H", "A"), ("A", "H")):
        sign = 1 if s == "H" else -1
        d = pd.DataFrame({
            "GAME_ID": t["GAME_ID"], "SEASON": t["SEASON"], "GAME_DATE": t["GAME_DATE"],
            "WIN": t["RESULT"] if s == "H" else 1 - t["RESULT"],
            "P_BASE": t["base"] if s == "H" else 1 - t["base"],
            "P_WHO": t["base+who"] if s == "H" else 1 - t["base+who"],
            "P_INC": t["base+inc"] if s == "H" else 1 - t["base+inc"],
            "BVAL_W": t.get(f"BVAL_W_{s}"), "BVAL_L": t.get(f"BVAL_L_{s}"),
            "TANK": t[f"TANK_{s}"], "STAKE": t[f"STAKE_{s}"], "P_POST": t[f"P_POST_{s}"],
            "REST": t[f"REST_{s}"], "MISS5": t[f"MISS5_{s}"], "MISS5_OPP": t[f"MISS5_{o}"],
            "OPP_STAKE": t[f"STAKE_{o}"], "GREM": t[f"GREM_{s}"], "IS_RS": t["IS_RS"],
        })
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def outcome_check(rows, mask, label):
    r = rows[mask]
    n = len(r)
    if n == 0:
        return {"group": label, "n": 0}
    win = r["WIN"].mean()
    out = {"group": label, "n": n, "win_rate": win}
    for k in ("P_BASE", "P_WHO", "P_INC"):
        e = r[k].mean()
        out[f"exp_{k[2:].lower()}"] = e
        out[f"diff_{k[2:].lower()}"] = win - e
    out["se"] = np.sqrt((r["P_BASE"] * (1 - r["P_BASE"])).sum()) / n
    out["miss5"] = r["MISS5"].mean()
    out["miss5_opp"] = r["MISS5_OPP"].mean()
    return out


def manipulation_table(t):
    rows = side_rows(t[t["IS_RS"] == 1])
    last_date = rows["SEASON"].map(rows.groupby("SEASON")["GAME_DATE"].max())
    lastn = rows["GAME_DATE"] > last_date - pd.Timedelta(days=MANIP_DAYS)
    gain = rows["BVAL_L"] - rows["BVAL_W"]
    checks = [
        (lastn, f"all team-games, last {MANIP_DAYS} days"),
        (lastn & (gain > MANIP_GAIN), f"losing improves bracket value by > {MANIP_GAIN}"),
        (lastn & (gain > MANIP_GAIN) & (rows["P_POST"] >= 0.99), "  ... and playoff spot sure"),
        (lastn & (gain > 0.05), "losing improves bracket value by > 0.05"),
        (lastn & (gain < -MANIP_GAIN), f"winning improves bracket value by > {MANIP_GAIN}"),
        ((rows["IS_RS"] == 1) & (rows["TANK"] > 1.0) & (rows["GREM"] <= 20),
         "tank incentive > 1pp #1-pick odds, last 20 games"),
        ((rows["REST"] == 1), "REST flag (late, playoff sure, nothing at stake)"),
        ((rows["GREM"] <= 20) & (rows["STAKE"] > 0.15), "stake > 0.15, last 20 games"),
    ]
    return pd.DataFrame([outcome_check(rows, m, lab) for m, lab in checks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=",".join(TEST_SEASONS))
    ap.add_argument("--incentive", default=",".join(inc.INCENTIVE))
    ap.add_argument("--out", default=None, help="save per-game predictions (csv)")
    a = ap.parse_args()
    raw = load_history()
    df = attach_season_if_missing(inc.build_dataset(raw), raw)
    t = predict_all(df, a.seasons.split(","), model_sets(a.incentive.split(",")))
    if a.out:
        t.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(subset_table(t).round(4).to_string(index=False))
    print()
    print(manipulation_table(t).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
